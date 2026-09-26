from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_PROPOSER_SYSTEM = """You are an expert evaluator designing a task-specific evaluation checklist.
Given a conversation history, the current user query, and (optionally) the user's overall intent, your job is to produce 5-10 yes/no evaluation criteria that a judge can use to assess whether an AI response adequately addresses the request.

Guidelines for good checklist items:
- Each item must be a clear, self-contained yes/no question.
- Items should cover different quality dimensions: correctness, completeness, relevance, adherence to constraints, format, and safety where applicable.
- Items must be verifiable against the response text alone.
- Avoid duplicate or highly overlapping items.
- Do NOT assess the response itself — only write the evaluation criteria."""

_PROPOSER_USER = """# Conversation History
<|begin_of_history|>
{history}
<|end_of_history|>

# Current User Query
<|begin_of_query|>
{user_query}
<|end_of_query|>

# User's High-Level Intent
<|begin_of_intent|>
{intent_block}
<|end_of_intent|>

# Task
Generate a checklist of 5-10 yes/no evaluation criteria for judging an AI response to the above query. Output ONLY valid JSON in this exact format:

```json
{{
  "checklist": [
    "<checklist item 1>",
    "<checklist item 2>",
    ...
  ]
}}
```"""

_MERGER_SYSTEM = """You are an expert evaluator merging two independently generated evaluation checklists into one unified checklist.

Guidelines:
- Keep 5-10 items in the final checklist.
- Remove exact duplicates and near-duplicates; keep the clearer phrasing.
- Preserve items that cover distinct quality dimensions not addressed by the other list.
- All items must remain yes/no questions that are verifiable from a response.
- Output ONLY valid JSON — no commentary outside the JSON block."""

_MERGER_USER = """# Conversation History
<|begin_of_history|>
{history}
<|end_of_history|>

# Current User Query
<|begin_of_query|>
{user_query}
<|end_of_query|>

# User's High-Level Intent
<|begin_of_intent|>
{intent_block}
<|end_of_intent|>

# Checklist A
{checklist_a_str}

# Checklist B
{checklist_b_str}

# Task
Merge the two checklists above into a single unified checklist of 5-10 yes/no
evaluation criteria. Output ONLY valid JSON in this exact format:

```json
{{
  "checklist": [
    "<checklist item 1>",
    "<checklist item 2>",
    ...
  ]
}}
```"""


def _format_history(turns: List[Dict]) -> str:
    if not turns:
        return "(No prior conversation history.)"
    lines = []
    for t in turns:
        role = "Human" if t["role"] == "user" else "AI"
        lines.append(f"{role}: {t['content']}")
    return "\n\n".join(lines)


def _parse_checklist(text: str) -> Optional[List[str]]:
    """
    Extract the 'checklist' list from a model response that contains a JSON block.
    Returns None if extraction fails.
    """
    # Try fenced code block first
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = code_block.group(1) if code_block else text
    try:
        data = json.loads(raw)
        items = data.get("checklist", [])
        if isinstance(items, list) and all(isinstance(i, str) for i in items) and items:
            return [i.strip() for i in items if i.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass

    m = re.search(r'"checklist"\s*:\s*(\[[^\]]+\])', text, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(1))
            if isinstance(items, list) and items:
                return [str(i).strip() for i in items if str(i).strip()]
        except json.JSONDecodeError:
            pass

    return None


def _extra_kwargs(model: str, top_p: Optional[float]) -> Dict:
    """
    Build extra kwargs for a model call: top_p + reasoning-disable for Qwen3.
    If using other reasoning models, modify as needed.
    """
    kwargs: Dict = {}
    if top_p is not None:
        kwargs["top_p"] = top_p
    if "qwen3" in model.lower():
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return kwargs


def _call_model(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    label: str,
    top_p: Optional[float] = None,
) -> Optional[List[str]]:
    """
    Call a model and parse the response as a checklist. Returns None on failure.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    extra = _extra_kwargs(model, top_p)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            content = resp.choices[0].message.content or ""
            result = _parse_checklist(content)
            if result:
                return result
            logger.warning(
                "[%s] attempt %d/%d: could not parse checklist. Output (first 2048 chars):\n%s",
                label, attempt + 1, max_retries, content[:2048],
            )
        except Exception as e:
            logger.warning("[%s] attempt %d/%d error: %s", label, attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                time.sleep(1.5 ** attempt)
    return None


def _get_local_client(port: int) -> OpenAI:
    """
    Create an OpenAI-compatible client pointing at a local vLLM server.
    """
    return OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="EMPTY",
    )


def _verify_client_model(client: OpenAI, model: str, label: str) -> None:
    """
    Assert that `model` is actually served at `client`'s endpoint.
    """
    try:
        ids = [m.id for m in client.models.list().data]
        assert model in ids, (
            f"[{label}] Model '{model}' not found at endpoint. Available: {ids}"
        )
    except Exception as e:
        raise RuntimeError(f"[{label}] Could not verify model: {e}") from e


def generate_rubric(
    conversation: List[Dict],
    client_a: OpenAI,
    model_a: str,
    client_b: OpenAI,
    model_b: str,
    client_merger: Optional[OpenAI] = None,
    model_merger: Optional[str] = None,
    intent: Optional[str] = None,
    proposer_temperature: float = 0.7,
    merger_temperature: float = 0.0,
    top_p: Optional[float] = None,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> Optional[Dict]:
    """
    Generate a WildBench-style evaluation checklist for a conversation.

    Args:
        conversation:       Full conversation as OpenAI-format message dicts.
                            The last turn must be {"role": "user", "content": ...}.
                            (The assistant response being evaluated is NOT included —
                            rubric generation is response-agnostic.)
        client_a:           OpenAI-compatible client for proposer A (local vLLM).
        model_a:            Model name served at client_a's endpoint.
        client_b:           OpenAI-compatible client for proposer B (local vLLM).
        model_b:            Model name served at client_b's endpoint.
        client_merger:      Client for the merger model. Defaults to client_a.
        model_merger:       Model for the merger step. Defaults to model_a.
        intent:             Optional natural-language description of the user's
                            overall conversational goal.
        proposer_temperature: Sampling temperature for both proposers (default 0.7).
        merger_temperature: Sampling temperature for the merger (default 0.0).
        max_tokens:         Max tokens for each model call.
        max_retries:        Number of retry attempts per call.

    Returns:
        {
            "checklist":   list[str],   # final merged checklist (5-10 items)
            "checklist_a": list[str],   # proposer A output
            "checklist_b": list[str],   # proposer B output
        }
        or None if both proposers fail.
    """
    if client_merger is None:
        client_merger = client_a
    if model_merger is None:
        model_merger = model_a

    # Extract history and last user query from conversation
    user_indices = [i for i, t in enumerate(conversation) if t.get("role") == "user"]
    if not user_indices:
        logger.error("Conversation contains no user turns.")
        return None

    last_u = user_indices[-1]
    history_turns = conversation[:last_u]
    user_query = conversation[last_u].get("content", "")
    history_text = _format_history(history_turns)

    proposer_user = _PROPOSER_USER.format(
        history=history_text,
        user_query=user_query,
        intent_block=intent.strip(),
    )

    # ── Step 1: Run both proposers in parallel ────────────────────────────────
    checklist_a: Optional[List[str]] = None
    checklist_b: Optional[List[str]] = None

    def _run_a() -> None:
        nonlocal checklist_a
        checklist_a = _call_model(
            client_a, model_a,
            _PROPOSER_SYSTEM, proposer_user,
            proposer_temperature, max_tokens, max_retries,
            label="proposer_a", top_p=top_p,
        )

    def _run_b() -> None:
        nonlocal checklist_b
        checklist_b = _call_model(
            client_b, model_b,
            _PROPOSER_SYSTEM, proposer_user,
            proposer_temperature, max_tokens, max_retries,
            label="proposer_b", top_p=top_p,
        )

    t_a = threading.Thread(target=_run_a)
    t_b = threading.Thread(target=_run_b)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    if not checklist_a and not checklist_b:
        logger.error("Both proposers failed to produce a checklist.")
        return None

    # ── Step 2: Merge, or fall back to whichever proposer succeeded ───────────
    if not checklist_a or not checklist_b:
        checklist = checklist_a or checklist_b
        logger.warning(
            "One proposer failed (%s). Using the other as the final checklist.",
            "proposer_a" if not checklist_a else "proposer_b",
        )
    else:
        checklist_a_str = "\n".join(f"- {q}" for q in checklist_a)
        checklist_b_str = "\n".join(f"- {q}" for q in checklist_b)

        merger_user = _MERGER_USER.format(
            history=history_text,
            user_query=user_query,
            intent_block=intent.strip(),
            checklist_a_str=checklist_a_str,
            checklist_b_str=checklist_b_str,
        )
        merged = _call_model(
            client_merger, model_merger,
            _MERGER_SYSTEM, merger_user,
            merger_temperature, max_tokens, max_retries,
            label="merger", top_p=top_p,
        )
        if merged:
            checklist = merged
        else:
            logger.warning("Merger failed; falling back to proposer_a checklist.")
            checklist = checklist_a

    return {
        "checklist":   checklist,
        "checklist_a": checklist_a,
        "checklist_b": checklist_b,
    }


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_wildbench() -> List[Dict]:
    from datasets import load_dataset  # local import to keep module lightweight
    logger.info("Loading WildBench v2 …")
    ds = load_dataset("allenai/WildBench", "v2", split="test")
    tasks = []
    for item in ds:
        turns = item["conversation_input"]
        last_u = max(i for i, t in enumerate(turns) if t["role"] == "user")
        history_turns = turns[:last_u]
        tasks.append({
            "session_id":   item["session_id"],
            "primary_tag":  item.get("primary_tag"),
            "conversation": turns[: last_u + 1],
            "history":      _format_history(history_turns),
            "user_query":   turns[last_u]["content"],
            "intent":       item.get("intent"),
        })
    logger.info("Loaded %d WildBench tasks.", len(tasks))
    return tasks


def _load_jsonl(path: str) -> List[Dict]:
    """
    Load tasks from the WildChat-style JSONL format used by userlm-replication.
    Expected fields per record: conversation_hash, conversation (list of turns
    with 'role' and 'content'), intent.
    """
    logger.info("Loading tasks from %s …", path)
    tasks = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            turns = item["conversation"]
            # Strip per-turn metadata; keep only role + content
            clean_turns = [{"role": t["role"], "content": t["content"]} for t in turns]
            user_indices = [i for i, t in enumerate(clean_turns) if t["role"] == "user"]
            if not user_indices:
                continue
            last_u = user_indices[-1]
            history_turns = clean_turns[:last_u]
            tasks.append({
                "session_id":        item.get("conversation_hash", str(len(tasks))),
                "conversation_hash": item.get("conversation_hash"),
                "primary_tag":       None,
                "conversation": clean_turns[: last_u + 1],
                "history":      _format_history(history_turns),
                "user_query":   clean_turns[last_u]["content"],
                "intent":       item.get("intent"),
            })
    logger.info("Loaded %d tasks from %s.", len(tasks), path)
    return tasks


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def generate_rubrics_batch(
    model_a: str,
    model_b: str,
    port_a: int = 8000,
    port_b: int = 8001,
    model_merger: Optional[str] = None,
    port_merger: Optional[int] = None,
    input_file: Optional[str] = None,
    max_workers: int = 16,
    max_tokens: int = 1024,
    proposer_temperature: float = 0.7,
    merger_temperature: float = 0.0,
    top_p: Optional[float] = None,
    max_retries: int = 5,
    output_dir: str = "results/rubrics",
    n_tasks: Optional[int] = None,
) -> Dict:
    """
    Generate rubrics for every task in WildBench v2 and save to disk.

    Output files:
      <output_dir>/rubrics_details.jsonl   — one JSON record per task
      <output_dir>/rubrics_summary.json    — run metadata + failure count

    Each JSONL record contains:
      session_id, primary_tag, user_query, history,
      checklist, checklist_a, checklist_b, failed
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build clients
    client_a = _get_local_client(port_a)
    client_b = _get_local_client(port_b)
    _verify_client_model(client_a, model_a, "proposer_a")
    _verify_client_model(client_b, model_b, "proposer_b")

    _port_merger = port_merger if port_merger is not None else port_a
    _model_merger = model_merger if model_merger is not None else model_a
    client_merger = _get_local_client(_port_merger)
    if _port_merger != port_a:
        _verify_client_model(client_merger, _model_merger, "merger")
    else:
        client_merger = client_a  # reuse same object

    # Load tasks
    tasks = _load_jsonl(input_file) if input_file else _load_wildbench()
    if n_tasks:
        tasks = tasks[:n_tasks]
    logger.info("Loaded %d tasks.", len(tasks))

    detail_path  = Path(output_dir) / "rubrics_details.jsonl"
    summary_path = Path(output_dir) / "rubrics_summary.json"

    # ── Resume: skip already-completed session_ids ────────────────────────────
    done_ids: set = set()
    if detail_path.exists():
        with open(detail_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sid = rec.get("session_id")
                    if sid:
                        done_ids.add(sid)
                except json.JSONDecodeError:
                    pass
        logger.info("Resuming: %d tasks already done, skipping them.", len(done_ids))

    pending = [t for t in tasks if t["session_id"] not in done_ids]
    logger.info("%d tasks remaining.", len(pending))

    results: List[Dict] = []
    lock = threading.Lock()
    failures = 0
    completed_since_flush = 0
    FLUSH_EVERY = 1000

    def _flush(records: List[Dict]) -> None:
        """Append records to the JSONL file."""
        with open(detail_path, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _process(idx: int) -> Dict:
        task = pending[idx]
        result = generate_rubric(
            conversation=task["conversation"],
            client_a=client_a,
            model_a=model_a,
            client_b=client_b,
            model_b=model_b,
            client_merger=client_merger,
            model_merger=_model_merger,
            intent=task.get("intent"),
            proposer_temperature=proposer_temperature,
            merger_temperature=merger_temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        failed = result is None
        return {
            "session_id":        task["session_id"],
            "conversation_hash": task.get("conversation_hash"),
            "primary_tag":       task["primary_tag"],
            "user_query":        task["user_query"],
            "history":           task["history"],
            "checklist":         result["checklist"]   if result else None,
            "checklist_a":       result["checklist_a"] if result else None,
            "checklist_b":       result["checklist_b"] if result else None,
            "failed":            failed,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, i): i for i in range(len(pending))}
        for fut in tqdm(as_completed(futures), total=len(pending), desc="Generating rubrics"):
            try:
                rec = fut.result()
            except Exception as e:
                logger.warning("Error processing task %d: %s", futures[fut], e)
                with lock:
                    failures += 1
                continue

            with lock:
                results.append(rec)
                if rec["failed"]:
                    failures += 1
                completed_since_flush += 1
                if completed_since_flush >= FLUSH_EVERY:
                    to_flush = results[-completed_since_flush:]
                    completed_since_flush = 0
                    _flush(to_flush)
                    logger.info("Flushed %d records to %s", len(to_flush), detail_path)

    # Final flush for whatever remains unflushed
    with lock:
        if completed_since_flush > 0:
            _flush(results[-completed_since_flush:])

    total_done = len(done_ids) + len(results)
    summary = {
        "proposer_a":            model_a,
        "proposer_b":            model_b,
        "merger":                _model_merger,
        "n_tasks_this_run":      len(results),
        "n_tasks_total":         total_done,
        "n_failures_this_run":   failures,
        "proposer_temperature":  proposer_temperature,
        "merger_temperature":    merger_temperature,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Done. %d/%d tasks succeeded this run (%d total in file). Saved to %s",
        len(results) - failures, len(results), total_done, output_dir,
    )
    return {"summary": summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate WildBench-style rubrics using two local vLLM proposers."
    )
    parser.add_argument("--proposer_a",
                        required=True,
                        help="Model name for proposer A (must be served at --port_a).")
    parser.add_argument("--port_a",
                        type=int, default=8000,
                        help="vLLM port for proposer A (default 8000).")
    parser.add_argument("--proposer_b",
                        required=True,
                        help="Model name for proposer B (must be served at --port_b).")
    parser.add_argument("--port_b",
                        type=int, default=8001,
                        help="vLLM port for proposer B (default 8001).")
    parser.add_argument("--merger_model",
                        default=None,
                        help="Model name for the merger step (default: same as proposer_a).")
    parser.add_argument("--port_merger",
                        type=int, default=None,
                        help="vLLM port for the merger model (default: same as port_a).")
    parser.add_argument("--max_workers",
                        type=int, default=16,
                        help="Thread pool concurrency (default 16).")
    parser.add_argument("--max_tokens",
                        type=int, default=2048,
                        help="Max tokens per model call (default 2048).")
    parser.add_argument("--proposer_temperature",
                        type=float, default=0.3,
                        help="Sampling temperature for proposers (default 0.3).")
    parser.add_argument("--merger_temperature",
                        type=float, default=0.3,
                        help="Sampling temperature for the merger (default 0.3).")
    parser.add_argument("--top_p",
                        type=float, default=0.9,
                        help="Top-p nucleus sampling for all model calls (default: 0.9).")
    parser.add_argument("--max_retries",
                        type=int, default=4,
                        help="Max retry attempts per API call (default 4).")
    parser.add_argument("--output_dir",
                        default="results/rubrics",
                        help="Output directory (default results/rubrics).")
    parser.add_argument("--n_tasks",
                        type=int, default=None,
                        help="Limit to first N tasks — useful for testing.")
    parser.add_argument("--input_file",
                        default=None,
                        help="Path to a WildChat-style JSONL file. If omitted, loads WildBench v2 from HuggingFace.")
    args = parser.parse_args()

    generate_rubrics_batch(
        model_a=args.proposer_a,
        model_b=args.proposer_b,
        port_a=args.port_a,
        port_b=args.port_b,
        model_merger=args.merger_model,
        port_merger=args.port_merger,
        input_file=args.input_file,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        proposer_temperature=args.proposer_temperature,
        merger_temperature=args.merger_temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
        n_tasks=args.n_tasks,
    )


if __name__ == "__main__":
    main()

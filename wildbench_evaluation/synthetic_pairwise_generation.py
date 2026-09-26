"""
Synthetic Pairwise Generation
==============================
Generates synthetic multi-turn conversations for pairwise evaluation
in the WildBench style. Three models are used:

  - UserLM:   Simulates the user; generates follow-up queries from turn 2 onward.
  - Model1:   First agent model to evaluate.
  - Model2:   Second agent model to evaluate.

Turn 1 uses the *actual* first user message from WildChat data
(JSONL files in the data_with_intents format).  Subsequent turns are
produced by the UserLM, conditioned on one agent's conversation history.

Two conditioning runs are produced per conversation to reduce position bias
(matching the wildbench swap-positions protocol):
  - run_a:  UserLM conditioned on model1's responses
  - run_b:  UserLM conditioned on model2's responses

UserLM role-reversal convention
---------------------------------
The UserLM is prompted with *swapped* roles so that the model naturally
generates the next user message in its own "assistant" completion slot.
The leading real user turn (q1) is omitted — the intent covers that context.

  At turn 2 (generating q2), agent_history = [q1, r1]:
    system   : "You are simulating a user … Your goal: {intent}"
    user     : <r1 — agent response to q1>
    → model generates q2

  At turn 3 (generating q3), agent_history = [q1, r1, q2, r2]:
    system   : …
    user     : <r1>
    assistant: <q2>   (UserLM's previous output)
    user     : <r2>
    → model generates q3

Output format (JSONL)
----------------------
Each line is one record:
  {
    "conversation_hash": str,
    "intent": str,
    "conditioning_model": "model1" | "model2",
    "model1_name": str,
    "model2_name": str,
    "turns": [
      {
        "turn_idx": int,
        "user_query": str,            # real at turn 0, UserLM-generated otherwise
        "model1_response": str,
        "model2_response": str,
        # history BEFORE this turn, from the conditioning model's perspective:
        "history_list": [...],        # list of {role, content} dicts
        "history_str": str,           # human-readable string (for the judge)
        "userlm_input_messages": ...  # present for turn_idx > 0
      },
      ...
    ]
  }

The output can be fed directly into wildbench_pairwise.py by constructing
task dicts from each turn record; see --help for details.

Usage example
--------------
  python synthetic_pairwise_generation.py \\
    --userlm  UserLM-7B        --port_userlm  8000 \\
    --model1  Llama-3-8B-Instruct --port_model1 8001 \\
    --model2  Mistral-7B-Instruct  --port_model2 8002 \\
    --data_path /path/to/test_with_intents_Qwen--Qwen3-32B.jsonl \\
    --n_turns 3 \\
    --output_dir synthetic_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


DEFAULT_USERLM_SYSTEM_PROMPT = """\
You are simulating a real user interacting with an AI assistant.
Your goal is described below. Based on the conversation so far, generate your \
next message to the assistant.
Stay in character as a user — be natural, curious, and focused on your goal.
Output ONLY the next user message, with no preamble or explanation."""


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

def _get_client(model: str, port: int) -> OpenAI:
    """Return an OpenAI-compatible client for the given model."""
    if model.lower().startswith("gpt"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if model.lower().startswith("claude"):
        return OpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.anthropic.com/v1/",
        )
    if model.lower().startswith("gemini"):
        return OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    # Local vLLM / OpenAI-compatible server — the SDK still requires a non-empty
    # api_key even when the server ignores it, so fall back to a sentinel.
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    model_ids = [m.id for m in client.models.list().data]
    assert model in model_ids, (
        f"Model {model!r} not found on server at port {port}. "
        f"Available: {model_ids}"
    )
    return client


def _chat(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    max_retries: int = 8,
) -> str:
    """Call the chat completions API and return the response text."""
    extra_kwargs: Dict = {"top_p": top_p} if top_p is not None else {}
    last_exc: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_exc = e
    logger.warning("Chat error for model %s: %s", model, last_exc)
    return ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: str, n_tasks: Optional[int] = None) -> List[Dict]:
    """
    Load WildChat conversations from a data_with_intents JSONL file.

    Each record returned has:
      conversation_hash, intent, first_user_query
    """
    tasks: List[Dict] = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            conv = item.get("conversation", [])
            first_user_query: Optional[str] = None
            for turn in conv:
                if turn.get("role") == "user":
                    first_user_query = turn["content"]
                    break
            if first_user_query is None:
                logger.warning(
                    "Skipping %s: no user turn found.",
                    item.get("conversation_hash", "?"),
                )
                continue
            tasks.append(
                {
                    "conversation_hash": item["conversation_hash"],
                    "intent": item.get("intent", ""),
                    "first_user_query": first_user_query,
                }
            )
            if n_tasks and len(tasks) >= n_tasks:
                break
    logger.info("Loaded %d tasks from %s.", len(tasks), data_path)
    return tasks


# ---------------------------------------------------------------------------
# Message building
# ---------------------------------------------------------------------------

def _format_history_str(history_list: List[Dict]) -> str:
    """Format a history list into a human-readable string (for the judge)."""
    if not history_list:
        return "(No prior conversation history.)"
    lines = []
    for t in history_list:
        role_label = "Human" if t["role"] == "user" else "AI"
        lines.append(f"{role_label}: {t['content']}")
    return "\n\n".join(lines)


def _build_agent_messages(history: List[Dict], user_query: str) -> List[Dict]:
    """Build the message list for an agent model (standard chat format)."""
    return history + [{"role": "user", "content": user_query}]


def _build_userlm_messages(
    intent: str,
    agent_history: List[Dict],
    system_prompt: str,
) -> List[Dict]:
    """
    Build the message list for the UserLM using role reversal.

    agent_history alternates: user (query), assistant (agent response), ...
    We swap the roles so that:
      - agent responses  → "user"    (what the UserLM "receives")
      - user queries     → "assistant" (what the UserLM previously "said")
    Then the model naturally generates the *next* user message.

    The leading user turn (q1, the real first user query) is skipped because
    the OpenAI API requires the first non-system message to have role "user",
    and after role-reversal q1 would become an "assistant" message.  The
    UserLM already knows the goal from the system prompt + intent string.

    agent_history must be non-empty and end with an assistant message.
    """
    messages: List[Dict] = [
        {
            "role": "system",
            "content": f"{system_prompt}\n\nYour goal: {intent}",
        }
    ]
    # Skip all leading user turns before the first assistant message so that
    # the first reversed message (assistant→user) is valid as the opening
    # "user" turn required by the chat API.
    found_assistant = False
    for turn in agent_history:
        if not found_assistant:
            if turn["role"] == "assistant":
                found_assistant = True
            else:
                continue  # skip leading user turn(s)
        if turn["role"] == "user":
            # User query (UserLM's previous output) → "assistant"
            messages.append({"role": "assistant", "content": turn["content"]})
        else:
            # Agent response (what the UserLM "receives") → "user"
            messages.append({"role": "user", "content": turn["content"]})
    return messages


# ---------------------------------------------------------------------------
# Parallel response generation for a single turn
# ---------------------------------------------------------------------------

def _generate_pair(
    client1: OpenAI,
    model1: str,
    hist1: List[Dict],
    client2: OpenAI,
    model2: str,
    hist2: List[Dict],
    user_query: str,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
) -> Tuple[str, str]:
    """Generate responses from model1 and model2 in parallel for a single turn."""
    msgs1 = _build_agent_messages(hist1, user_query)
    msgs2 = _build_agent_messages(hist2, user_query)
    results: List[Optional[str]] = [None, None]

    def _gen(idx: int, client: OpenAI, model: str, msgs: List[Dict]) -> None:
        results[idx] = _chat(client, model, msgs, max_tokens, temperature, top_p)

    t1 = threading.Thread(target=_gen, args=(0, client1, model1, msgs1))
    t2 = threading.Thread(target=_gen, args=(1, client2, model2, msgs2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results[0] or "", results[1] or ""


# ---------------------------------------------------------------------------
# Single-conversation generation
# ---------------------------------------------------------------------------

def _generate_conversation(
    task: Dict,
    userlm_client: OpenAI,
    userlm_model: str,
    agent1_client: OpenAI,
    agent1_model: str,
    agent2_client: OpenAI,
    agent2_model: str,
    n_turns: int,
    conditioning_model: str,
    max_tokens: int,
    userlm_max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    userlm_system_prompt: str,
) -> Dict:
    """
    Generate a full multi-turn conversation for one WildChat task.

    conditioning_model: "model1" or "model2" — whose history the UserLM
      conditions on when generating follow-up user queries.

    Each model maintains its own conversation history (with its own responses),
    so the agent sees a coherent dialogue at every turn.

    Returns a dict matching the output format described in the module docstring.
    """
    intent = task["intent"]
    first_query = task["first_user_query"]

    # Per-agent conversation histories (list of {role, content})
    hist1: List[Dict] = []
    hist2: List[Dict] = []

    turns: List[Dict] = []

    for turn_idx in range(n_turns):
        userlm_input_messages: Optional[List[Dict]] = None

        if turn_idx == 0:
            user_query = first_query
        else:
            # UserLM generates the next user query from the conditioning model's history
            cond_hist = hist1 if conditioning_model == "model1" else hist2
            userlm_input_messages = _build_userlm_messages(
                intent, cond_hist, userlm_system_prompt
            )
            user_query = _chat(
                userlm_client,
                userlm_model,
                userlm_input_messages,
                userlm_max_tokens,
                temperature,
                top_p,
            )
            if not user_query.strip():
                logger.warning(
                    "UserLM returned empty response for hash=%s turn=%d; stopping.",
                    task["conversation_hash"],
                    turn_idx,
                )
                break

        # Snapshot history *before* this turn (from conditioning model's perspective)
        cond_hist_snapshot = (hist1 if conditioning_model == "model1" else hist2)[:]

        # Generate both agent responses in parallel
        r1, r2 = _generate_pair(
            agent1_client, agent1_model, hist1,
            agent2_client, agent2_model, hist2,
            user_query,
            max_tokens, temperature, top_p,
        )

        turn_record: Dict = {
            "turn_idx": turn_idx,
            "user_query": user_query,
            "model1_response": r1,
            "model2_response": r2,
            # History before this turn — used as "history" field for the judge
            "history_list": cond_hist_snapshot,
            "history_str": _format_history_str(cond_hist_snapshot),
        }
        if userlm_input_messages is not None:
            turn_record["userlm_input_messages"] = userlm_input_messages

        turns.append(turn_record)

        # Advance each model's own conversation history with its own response
        hist1.append({"role": "user", "content": user_query})
        hist1.append({"role": "assistant", "content": r1})
        hist2.append({"role": "user", "content": user_query})
        hist2.append({"role": "assistant", "content": r2})

    return {
        "conversation_hash": task["conversation_hash"],
        "intent": intent,
        "conditioning_model": conditioning_model,
        "model1_name": agent1_model,
        "model2_name": agent2_model,
        "userlm_name": userlm_model,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_generation(
    userlm: str,
    port_userlm: int,
    model1: str,
    port_model1: int,
    model2: str,
    port_model2: int,
    data_path: str,
    n_turns: int = 3,
    n_tasks: Optional[int] = None,
    max_workers: int = 16,
    max_tokens: int = 2048,
    userlm_max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    output_dir: str = "synthetic_results",
    userlm_system_prompt: str = DEFAULT_USERLM_SYSTEM_PROMPT,
) -> None:
    """Run synthetic pairwise generation and write results to JSONL."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to models…")
    userlm_client = _get_client(userlm, port_userlm)
    client1 = _get_client(model1, port_model1)
    client2 = _get_client(model2, port_model2)

    tasks = load_data(data_path, n_tasks)

    # Two conditioning runs per task (model1-conditioned and model2-conditioned)
    all_jobs: List[Tuple[Dict, str]] = (
        [(task, "model1") for task in tasks]
        + [(task, "model2") for task in tasks]
    )

    out_stem = (
        f"{model1.replace('/', '_')}_vs_{model2.replace('/', '_')}"
        f"_{n_turns}turns"
    )
    out_path = Path(output_dir) / f"{out_stem}.jsonl"

    results: List[Dict] = []
    lock = threading.Lock()

    def _process(args: Tuple[Dict, str]) -> Dict:
        task, conditioning = args
        return _generate_conversation(
            task=task,
            userlm_client=userlm_client,
            userlm_model=userlm,
            agent1_client=client1,
            agent1_model=model1,
            agent2_client=client2,
            agent2_model=model2,
            n_turns=n_turns,
            conditioning_model=conditioning,
            max_tokens=max_tokens,
            userlm_max_tokens=userlm_max_tokens,
            temperature=temperature,
            top_p=top_p,
            userlm_system_prompt=userlm_system_prompt,
        )

    logger.info(
        "Running %d jobs (%d tasks × 2 conditioning runs) with %d workers…",
        len(all_jobs), len(tasks), max_workers,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, job): job for job in all_jobs}
        for fut in tqdm(as_completed(futures), total=len(all_jobs), desc="Generating"):
            task, cond = futures[fut]
            try:
                rec = fut.result()
                with lock:
                    results.append(rec)
            except Exception as e:
                logger.warning(
                    "Error for hash=%s cond=%s: %s",
                    task["conversation_hash"], cond, e,
                )

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info("Saved %d records to %s.", len(results), out_path)
    print(f"\nDone. {len(results)} records written to {out_path}")
    print(
        f"Pass this file to wildbench_pairwise.py via --input_jsonl "
        f"(or load and construct task dicts from each turn's history_list / user_query)."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic pairwise conversation generation for WildBench-style evaluation. "
            "Generates N-turn conversations using a UserLM to simulate follow-up queries, "
            "and two agent models to produce responses at each turn."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # UserLM
    parser.add_argument("--userlm", required=True, help="UserLM model name.")
    parser.add_argument("--port_userlm", type=int, default=8000, help="Port for UserLM vLLM server.")

    # Agent models
    parser.add_argument("--model1", required=True, help="Agent model 1 name.")
    parser.add_argument("--port_model1", type=int, default=8001, help="Port for model1 vLLM server.")
    parser.add_argument("--model2", required=True, help="Agent model 2 name.")
    parser.add_argument("--port_model2", type=int, default=8002, help="Port for model2 vLLM server.")

    # Data
    parser.add_argument(
        "--data_path",
        required=True,
        help=(
            "Path to a WildChat JSONL file in the data_with_intents format "
            "(e.g. test_with_intents_Qwen--Qwen3-32B.jsonl)."
        ),
    )

    # Generation settings
    parser.add_argument("--n_turns", type=int, default=3, help="Number of conversation turns to generate.")
    parser.add_argument("--n_tasks", type=int, default=None, help="Limit to the first N tasks (for testing).")
    parser.add_argument("--max_workers", type=int, default=16, help="Number of parallel worker threads.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens for agent model responses.")
    parser.add_argument("--userlm_max_tokens", type=int, default=512, help="Max new tokens for UserLM outputs.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature applied to all models (UserLM and agents).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-p nucleus sampling probability (applied to all models).",
    )
    parser.add_argument(
        "--userlm_system_prompt",
        type=str,
        default=DEFAULT_USERLM_SYSTEM_PROMPT,
        help="System prompt for the UserLM (overrides the default).",
    )

    # Output
    parser.add_argument("--output_dir", default="synthetic_results", help="Directory to write output files.")

    args = parser.parse_args()

    run_generation(
        userlm=args.userlm,
        port_userlm=args.port_userlm,
        model1=args.model1,
        port_model1=args.port_model1,
        model2=args.model2,
        port_model2=args.port_model2,
        data_path=args.data_path,
        n_turns=args.n_turns,
        n_tasks=args.n_tasks,
        max_workers=args.max_workers,
        max_tokens=args.max_tokens,
        userlm_max_tokens=args.userlm_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_dir=args.output_dir,
        userlm_system_prompt=args.userlm_system_prompt,
    )


if __name__ == "__main__":
    main()

"""
Stage 1: Structural Abstraction (model-agnostic)

Pipeline:
  1. Read a WildBench checklist jsonl (any model — checklist content is identical
     across model runs). Extract ALL checklist items, not just failed ones.
  2. For each item, call the LLM with (conversation_prefix + checklist_item) only
     — no model response, no judge reasoning. Produce a task-agnostic
     description of what capability/property the rubric is testing.
  3. Write enriched records to JSONL (one per checklist item) for downstream
     embedding/clustering.

Output schema (one JSON object per line):
{
  "session_id":          "<string>",
  "primary_tag":         "<string>",
  "item_index":          <int, 1-indexed>,
  "raw_checklist_item":  "<verbatim rubric text>",
  "conversation":        "<truncated, formatted conversation prefix>",
  "abstract_description":"<task-agnostic restatement of the rubric>",
  "capability_type":     "<one of CAPABILITY_TYPES>",
  "capability_label":    "<3-6 word capability label>",
  "abstraction_confidence": <float>,
  "raw_abstraction_output": "<verbatim LLM output, for audit>"
}
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

MODEL  = "gpt-4.1-mini"
MAX_TOKENS = 2048
SLEEP_BETWEEN_CALLS = 0.3

MAX_CHARS_PER_TURN = 1024

CAPABILITY_TYPES = [
    "uniqueness",
    "count / quantity",
    "format / schema",
    "balance / calibration",
    "consistency",
    "completeness",
    "factuality / accuracy",
    "specificity",
    "relevance",
    "instruction-following",
    "other",
]

ABSTRACTION_SYSTEM_PROMPT = """You are an expert at characterizing what a rubric item is testing.
You will be given:
  1. The conversation history between a user and an AI (the prompt the AI was asked to respond to)
  2. A specific checklist item used to evaluate the AI's response

Your job is to ABSTRACT the checklist item into a task-agnostic description that
captures the underlying capability or property being tested — removing all
domain-specific nouns while preserving the constraint type and granularity.

Examples of what to strip out:
  - Specific topics ("urban design", "Twitter sentiment")
  - Specific entities ("Mary", "AnyCity")
  - Domain vocabulary that locks the description to one task

Keep the *kind of property* being tested (e.g. "the response correctly classifies
items into a specified taxonomy", "the response includes the requested numeric
quantity").

Return ONLY a valid JSON object with exactly these fields:
{
  "abstract_description":   "<rewritten item, task-agnostic, 1 sentence>",
  "capability_type":        "<one of the allowed capability types>",
  "capability_label":       "<3-6 word label for this capability>",
  "abstraction_confidence": <float 0.0-1.0, how confident the abstraction is valid>
}

Allowed capability_type values:
""" + "\n".join(f"  - {c}" for c in CAPABILITY_TYPES)


# ── Context formatting helpers ────────────────────────────────────────────────

def format_conversation(session: dict) -> str:
    """Format the conversation prefix (model_input) into a readable, truncated string."""
    turns = session.get("model_input", [])
    lines = []
    for turn in turns:
        role    = turn["role"].upper()
        content = turn["content"].replace("\n", " ").strip()
        if len(content) > MAX_CHARS_PER_TURN:
            content = content[:MAX_CHARS_PER_TURN] + "... [truncated]"
        lines.append(f"[{role}]: {content}")
    return "\n".join(lines)


# ── Core helpers ──────────────────────────────────────────────────────────────

def extract_all_items(session: dict) -> list[dict]:
    """
    Returns ALL checklist items in this session, each paired with the
    (truncated) conversation prefix. Model output / judge reasoning are
    intentionally excluded — abstraction is model-agnostic.
    """
    checklist    = session.get("checklist", [])
    session_id   = session.get("session_id", "unknown")
    primary_tag  = session.get("primary_tag", "unknown task")
    conversation = format_conversation(session)

    items = []
    for idx, raw in enumerate(checklist, start=1):
        items.append({
            "session_id":         session_id,
            "primary_tag":        primary_tag,
            "item_index":         idx,
            "raw_checklist_item": raw,
            "conversation":       conversation,
        })
    return items


def abstract_one_item(client: OpenAI, item: dict, max_retries: int = 4) -> dict:
    """Call the LLM to abstract one checklist item. Retries with backoff on failure."""
    user_prompt = f"""## Conversation History
{item['conversation']}

## Checklist Item to Abstract
\"{item['raw_checklist_item']}\"

Now abstract this checklist item into a task-agnostic description of what
capability or property is being tested."""

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": ABSTRACTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )

            raw_text = response.choices[0].message.content.strip()
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            abstraction = json.loads(clean)

            return {
                **item,
                "abstract_description":   abstraction.get("abstract_description", ""),
                "capability_type":        abstraction.get("capability_type", "other"),
                "capability_label":       abstraction.get("capability_label", ""),
                "abstraction_confidence": abstraction.get("abstraction_confidence", 0.0),
                "raw_abstraction_output": raw_text,
            }

        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    raise last_exc


# ── Resumability ──────────────────────────────────────────────────────────────

def load_already_abstracted(path: str) -> set[tuple[str, int]]:
    """Return the set of (session_id, item_index) tuples already present in the output."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["session_id"], rec["item_index"]))
            except Exception:
                continue
    return done


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_abstraction_pipeline(
    sessions: list[dict],
    output_path: str,
    num_workers: int = 1,
    debug: bool = False,
) -> list[dict]:
    client = OpenAI()

    all_items = []
    for session in sessions:
        all_items.extend(extract_all_items(session))
    print(f"Found {len(all_items)} checklist items across {len(sessions)} sessions.")

    already = load_already_abstracted(output_path)
    if already:
        print(f"Resuming: {len(already)} items already abstracted in '{output_path}'.")
    todo = [it for it in all_items if (it["session_id"], it["item_index"]) not in already]
    print(f"To process: {len(todo)} items.")

    abstracted = []
    errors     = []

    # append mode for resumability
    with open(output_path, "a") as out_file:
        if debug:
            for i, item in enumerate(todo):
                print(f"  [{i+1}/{len(todo)}] Abstracting: \"{item['raw_checklist_item'][:60]}...\"")
                try:
                    result = abstract_one_item(client, item)
                    abstracted.append(result)
                    out_file.write(json.dumps(result) + "\n")
                    out_file.flush()
                except json.JSONDecodeError as e:
                    print(f"    ⚠ JSON parse error on item {i+1}: {e}")
                    errors.append({"item": item, "error": str(e)})
                except Exception as e:
                    print(f"    ⚠ API error on item {i+1}: {e}")
                    errors.append({"item": item, "error": str(e)})
                time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            write_lock = threading.Lock()

            def process_item(args):
                i, item = args
                time.sleep(SLEEP_BETWEEN_CALLS)
                return i, item, abstract_one_item(client, item)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process_item, (i, item)): i
                           for i, item in enumerate(todo)}
                for future in as_completed(futures):
                    i = futures[future]
                    item = todo[i]
                    try:
                        _, _, result = future.result()
                        print(f"  [{i+1}/{len(todo)}] Abstracted: \"{item['raw_checklist_item'][:60]}...\"")
                        with write_lock:
                            abstracted.append(result)
                            out_file.write(json.dumps(result) + "\n")
                            out_file.flush()
                    except json.JSONDecodeError as e:
                        print(f"    ⚠ JSON parse error on item {i+1}: {e}")
                        with write_lock:
                            errors.append({"item": item, "error": str(e)})
                    except Exception as e:
                        print(f"    ⚠ API error on item {i+1}: {e}")
                        with write_lock:
                            errors.append({"item": item, "error": str(e)})

    print(f"\n✓ Done. {len(abstracted)} newly abstracted, {len(errors)} errors.")
    print(f"  Output appended to: {output_path}")

    if errors:
        error_path = output_path.replace(".jsonl", "_errors.jsonl")
        with open(error_path, "a") as ef:
            for e in errors:
                ef.write(json.dumps(e) + "\n")
        print(f"  Errors written to: {error_path}")

    return abstracted


def summarize_results(records: list[dict]) -> None:
    from collections import Counter
    if not records:
        return
    counts = Counter(r["capability_type"] for r in records)
    avg_conf = sum(r["abstraction_confidence"] for r in records) / len(records)

    print("\n── Capability Type Distribution ──────────────────")
    for ctype, count in counts.most_common():
        bar = "█" * int(count / max(counts.values()) * 30)
        print(f"  {ctype:<25} {count:>4}  {bar}")
    print(f"\n  Avg abstraction confidence: {avg_conf:.2f}")
    print(f"  Total abstracted items:     {len(records)}")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Entry point ───────────────────────────────────────────────────────────────

DEFAULT_SOURCE = "/path/to/usersim/evaluation/results/absolute_binary_checklist/Qwen_Qwen2.5-3B-Instruct_binary_checklist_details.jsonl"
DEFAULT_OUTPUT_DIR = "output"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Any model's checklist jsonl — only checklist + conversation are read, "
                             "so model identity does not matter.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "abstracted_items.jsonl")

    sessions = load_jsonl(args.source)
    abstracted = run_abstraction_pipeline(
        sessions,
        output_path=output_path,
        num_workers=args.num_workers,
        debug=args.debug,
    )

    # Reload everything (incl. previously-resumed records) for an accurate summary
    all_records = load_jsonl(output_path)
    summarize_results(all_records)

    print("\n── Sample Abstracted Records ─────────────────────")
    for r in all_records[:3]:
        print(f"\n  Raw:        {r['raw_checklist_item']}")
        print(f"  Abstract:   {r['abstract_description']}")
        print(f"  Type:       {r['capability_type']}")
        print(f"  Label:      {r['capability_label']}")
        print(f"  Conf:       {r['abstraction_confidence']:.2f}")

"""
Stage 6: Post-Taxonomy Binary Inclusion
  Inputs:
    - abstracted_items.jsonl   (only used as a convenient index of unique
                                (session_id, item_index, raw_checklist_item,
                                 conversation) tuples — abstract_description
                                is NOT used here).

  Outputs:
    - representative_clusters.json   (the ~10 hand-curated representative
                                      clusters + yes/no question for each)
    - cluster_inclusion.jsonl        (one record per checklist item, with a
                                      multi-label inclusion map)

For each checklist item, the LLM sees the (truncated) conversation prefix +
the raw checklist item — exactly the same context shape as stage 1
(1_structural_abstraction.py) — and answers YES/NO for each of ~10
representative-cluster questions in a single batched call. Multi-label: a
single item can be YES for several clusters.

Resumable + parallel like stage 1.

Output schema (one JSON object per line in cluster_inclusion.jsonl):
{
  "session_id":         "<string>",
  "primary_tag":        "<string>",
  "item_index":         <int, 1-indexed>,
  "raw_checklist_item": "<verbatim rubric text>",
  "inclusion": {
    "<cluster_key>": true/false,
    ...
  },
  "raw_inclusion_output": "<verbatim LLM output, for audit>"
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

MODEL = "gpt-4.1-mini"
MAX_TOKENS = 1024
SLEEP_BETWEEN_CALLS = 0.3


# ── Representative clusters ──────────────────────────────────────────────────
#
# Hand-curated from output/clusters.json (taxonomy.json) — one slot per
# minimally-overlapping LLM-behavior dimension that's commonly used to break
# down model failures. Keys are short slugs used in the inclusion map.

REPRESENTATIVE_CLUSTERS = [
    {
        "key": "numerical_computation",
        "source_cluster_ids": [25, 26],
        "name": "Numerical & Computational Accuracy",
        "description": (
            "Whether the response correctly carries out and verifies numerical "
            "computations, applies the right formulas, and produces precise "
            "quantitative outputs (with correct units, formatting, and "
            "mathematical reasoning)."
        ),
        "question": (
            "Does this checklist item primarily test the response's ability to "
            "produce a correct numerical, mathematical, or computational result "
            "(e.g. arithmetic, probability, formula application, unit handling, "
            "verification of a numeric answer)?"
        ),
    },
    {
        "key": "code_correctness",
        "source_cluster_ids": [15],
        "name": "Code Correctness & Programming Standards",
        "description": (
            "Whether generated code is syntactically valid, free of bugs, and "
            "follows best practices/conventions for the specified language and "
            "execution environment."
        ),
        "question": (
            "Does this checklist item primarily test whether generated code is "
            "syntactically valid, runnable, free of bugs, or compliant with "
            "language-specific conventions and best practices?"
        ),
    },
    {
        "key": "length_format_adherence",
        "source_cluster_ids": [2],
        "name": "Length & Format Adherence",
        "description": (
            "Whether the output meets explicit length constraints (word count, "
            "sentence count, page length) and structural formatting "
            "specifications stated in the prompt."
        ),
        "question": (
            "Does this checklist item primarily test whether the response "
            "satisfies an explicit length or formatting constraint (e.g. word "
            "count, sentence count, page length, layout/structural format "
            "specification)?"
        ),
    },
    {
        "key": "citation_attribution",
        "source_cluster_ids": [13],
        "name": "Source Attribution & Citation Accuracy",
        "description": (
            "Whether the response references credible sources and attributes "
            "claims with correct, precise citations to verifiable original "
            "material."
        ),
        "question": (
            "Does this checklist item primarily test whether the response "
            "correctly attributes information to sources or cites references "
            "accurately (e.g. citation format, linking claims to credible "
            "sources, verifiability)?"
        ),
    },
    {
        "key": "neutrality_sensitivity",
        "source_cluster_ids": [0],
        "name": "Neutrality, Tone & Cultural/Ethical Sensitivity",
        "description": (
            "Whether the response stays impartial, avoids personal bias and "
            "subjective judgement, and remains respectful and sensitive across "
            "cultural, ethical, and personal contexts."
        ),
        "question": (
            "Does this checklist item primarily test whether the response is "
            "neutral, unbiased, and respectful — avoiding subjective opinions, "
            "personal bias, or insensitive treatment of cultural/ethical/"
            "personal topics?"
        ),
    },
    {
        "key": "originality_creativity",
        "source_cluster_ids": [1],
        "name": "Originality & Creative Novelty",
        "description": (
            "Whether the response avoids generic phrasing or verbatim reuse and "
            "instead produces original, distinctive, or imaginative content."
        ),
        "question": (
            "Does this checklist item primarily test whether the response is "
            "original, novel, creative, or non-generic (avoiding repetition, "
            "verbatim reuse, or conventional phrasing)?"
        ),
    },
    {
        "key": "stepwise_reasoning",
        "source_cluster_ids": [22, 24],
        "name": "Step-by-step Reasoning & Instruction Clarity",
        "description": (
            "Whether the response explains its reasoning or instructions in a "
            "clear, transparent, sequential way — making the methodology and "
            "logical chain easy to follow."
        ),
        "question": (
            "Does this checklist item primarily test whether the response lays "
            "out clear, step-by-step reasoning or sequential "
            "instructions/procedures (transparent intermediate steps, "
            "stepwise breakdown of problem-solving, ordered procedure)?"
        ),
    },
    {
        "key": "robustness_error_handling",
        "source_cluster_ids": [12],
        "name": "Robustness & Error Handling",
        "description": (
            "Whether the response correctly handles a variety of valid, "
            "invalid, boundary, and exceptional inputs — with appropriate "
            "validation, error detection, and graceful recovery."
        ),
        "question": (
            "Does this checklist item primarily test whether the response (or "
            "the system/code it produces) handles invalid inputs, edge cases, "
            "boundary conditions, or errors gracefully and robustly?"
        ),
    },
    {
        "key": "visual_interactive_output",
        "source_cluster_ids": [16],
        "name": "Visual & Interactive Output Following",
        "description": (
            "Whether the response correctly produces visual or interactive "
            "elements (UI, layout, charts, dynamic behaviour) that obey the "
            "user's detailed visual/interactive instructions."
        ),
        "question": (
            "Does this checklist item primarily test whether the response "
            "correctly produces a requested visual or interactive artefact "
            "(e.g. UI, layout, chart, diagram, animation, interactive "
            "behaviour) that follows the user's visual/interactive "
            "specification?"
        ),
    },
    {
        "key": "domain_factual_accuracy",
        "source_cluster_ids": [4],
        "name": "Domain-Specific Factual Accuracy",
        "description": (
            "Whether the response uses specialised terminology, concepts, "
            "rules, or symbolic representations from a specific domain "
            "correctly and precisely (alignment with factual / contextual "
            "domain standards)."
        ),
        "question": (
            "Does this checklist item primarily test whether the response is "
            "factually correct in its use of specialised, domain-specific "
            "knowledge — including correct terminology, technical concepts, "
            "rules, or symbolic representations?"
        ),
    },
]


# ── Prompt ────────────────────────────────────────────────────────────────────

def _format_clusters_block() -> str:
    lines = []
    for c in REPRESENTATIVE_CLUSTERS:
        lines.append(
            f'- "{c["key"]}" — {c["name"]}\n'
            f'    Definition: {c["description"]}\n'
            f'    Question:   {c["question"]}'
        )
    return "\n".join(lines)


INCLUSION_SYSTEM_PROMPT = """You are an expert at characterizing what a rubric item is testing.
You will be given:
  1. The conversation history between a user and an AI (the prompt the AI was asked to respond to)
  2. A specific checklist item used to evaluate the AI's response

For EACH of the capability categories listed below, answer YES (true) or
NO (false) on whether the checklist item PRIMARILY tests that specific
capability. A single checklist item may be YES for multiple categories if it
genuinely tests several. Be conservative — only answer YES if the category is
a clear, primary thing the rubric is checking, not a tangential side-effect.

Capability categories:
""" + _format_clusters_block() + """

Return ONLY a valid JSON object with one boolean per category, using the
quoted keys exactly:
{
""" + ",\n".join(f'  "{c["key"]}": <true|false>' for c in REPRESENTATIVE_CLUSTERS) + """
}"""


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_already_done(path: str) -> set[tuple[str, int]]:
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


# ── Per-item LLM call ────────────────────────────────────────────────────────

CLUSTER_KEYS = [c["key"] for c in REPRESENTATIVE_CLUSTERS]


def classify_one_item(client: OpenAI, item: dict, max_retries: int = 4) -> dict:
    user_prompt = f"""## Conversation History
{item['conversation']}

## Checklist Item
\"{item['raw_checklist_item']}\"

For each capability category listed in the system prompt, decide YES or NO
on whether this checklist item primarily tests that capability. Output the
JSON object as instructed."""

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": INCLUSION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw_text = response.choices[0].message.content.strip()
            clean = raw_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)

            inclusion = {}
            for k in CLUSTER_KEYS:
                v = parsed.get(k)
                if isinstance(v, bool):
                    inclusion[k] = v
                elif isinstance(v, str) and v.lower() in ("true", "false"):
                    inclusion[k] = v.lower() == "true"
                else:
                    inclusion[k] = False

            return {
                "session_id":          item["session_id"],
                "primary_tag":         item.get("primary_tag", "unknown"),
                "item_index":          item["item_index"],
                "raw_checklist_item":  item["raw_checklist_item"],
                "inclusion":           inclusion,
                "raw_inclusion_output": raw_text,
            }
        except Exception as e:
            last_exc = e
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"    ⚠ Attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    raise last_exc


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_inclusion_pipeline(
    items: list[dict],
    output_path: str,
    num_workers: int = 4,
    debug: bool = False,
) -> None:
    client = OpenAI()

    already = load_already_done(output_path)
    if already:
        print(f"Resuming: {len(already)} items already classified in '{output_path}'.")
    todo = [it for it in items if (it["session_id"], it["item_index"]) not in already]
    print(f"To process: {len(todo)} items.")

    errors = []

    with open(output_path, "a") as out_file:
        if debug:
            for i, item in enumerate(todo):
                print(f"  [{i+1}/{len(todo)}] Classifying: \"{item['raw_checklist_item'][:60]}...\"")
                try:
                    result = classify_one_item(client, item)
                    out_file.write(json.dumps(result) + "\n")
                    out_file.flush()
                except Exception as e:
                    print(f"    ⚠ ERROR on item {i+1}: {e}")
                    errors.append({"item": item, "error": str(e)})
                time.sleep(SLEEP_BETWEEN_CALLS)
        else:
            write_lock = threading.Lock()

            def process(args):
                i, item = args
                time.sleep(SLEEP_BETWEEN_CALLS)
                return i, classify_one_item(client, item)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(process, (i, item)): i
                           for i, item in enumerate(todo)}
                for future in as_completed(futures):
                    i = futures[future]
                    item = todo[i]
                    try:
                        _, result = future.result()
                        print(f"  [{i+1}/{len(todo)}] Classified: \"{item['raw_checklist_item'][:60]}...\"")
                        with write_lock:
                            out_file.write(json.dumps(result) + "\n")
                            out_file.flush()
                    except Exception as e:
                        print(f"    ⚠ ERROR on item {i+1}: {e}")
                        with write_lock:
                            errors.append({"item": item, "error": str(e)})

    print(f"\n✓ Done. {len(errors)} errors.")
    print(f"  Output appended to: {output_path}")

    if errors:
        error_path = output_path.replace(".jsonl", "_errors.jsonl")
        with open(error_path, "a") as ef:
            for e in errors:
                ef.write(json.dumps(e) + "\n")
        print(f"  Errors written to: {error_path}")


def summarize_inclusion(records: list[dict]) -> None:
    if not records:
        return
    print("\n── Per-cluster inclusion counts ──────────────────")
    for c in REPRESENTATIVE_CLUSTERS:
        n = sum(1 for r in records if r["inclusion"].get(c["key"]))
        bar = "█" * int(n / max(1, len(records)) * 30)
        print(f"  {c['key']:<28} {n:>5}/{len(records):<5}  {bar}")
    multi = sum(1 for r in records if sum(r["inclusion"].values()) >= 2)
    none  = sum(1 for r in records if sum(r["inclusion"].values()) == 0)
    print(f"\n  Items matching ≥2 clusters: {multi} / {len(records)}")
    print(f"  Items matching 0 clusters:  {none} / {len(records)}")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="output",
        help="Reads abstracted_items.jsonl (only raw_checklist_item + "
             "conversation fields are used) and writes outputs here.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    items_path     = os.path.join(args.output_dir, "abstracted_items.jsonl")
    clusters_path  = os.path.join(args.output_dir, "representative_clusters.json")
    inclusion_path = os.path.join(args.output_dir, "cluster_inclusion.jsonl")

    # Persist the cluster definitions for downstream stage 7
    with open(clusters_path, "w") as f:
        json.dump(REPRESENTATIVE_CLUSTERS, f, indent=2)
    print(f"Saved {len(REPRESENTATIVE_CLUSTERS)} representative clusters → '{clusters_path}'.")

    items = load_jsonl(items_path)
    print(f"Loaded {len(items)} checklist items from '{items_path}'.")

    run_inclusion_pipeline(
        items,
        output_path=inclusion_path,
        num_workers=args.num_workers,
        debug=args.debug,
    )

    all_records = load_jsonl(inclusion_path)
    summarize_inclusion(all_records)

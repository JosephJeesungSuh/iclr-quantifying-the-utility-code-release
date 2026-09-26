"""
Stage 4: Label
  Input:  cluster_assignments.jsonl
  Output: clusters.json   (per-cluster label, description, stats, samples)

Run: python 4_label.py [--output_dir output]

Sampling strategy: stratified by capability_type + primary_tag, so the LLM
sees a representative cross-section of each cluster, not just the first N items.
"""

import argparse
import json
import os
import time
import random
from collections import Counter, defaultdict
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
CHAT_MODEL  = "gpt-4.1-mini"
SAMPLE_SIZE = 50
SLEEP       = 0.3
RANDOM_SEED = 42


# ── Stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(items: list[dict], n: int) -> list[dict]:
    """Sample n items, stratified by capability_type then by primary_tag."""
    if len(items) <= n:
        return items

    random.seed(RANDOM_SEED)

    by_type = defaultdict(list)
    for item in items:
        by_type[item.get("capability_type", "other")].append(item)

    total       = len(items)
    sample      = []
    allocated   = 0
    type_counts = {k: len(v) for k, v in by_type.items()}

    for ctype, bucket in sorted(by_type.items(), key=lambda x: -len(x[1])):
        slots = max(1, round(n * type_counts[ctype] / total))
        slots = min(slots, len(bucket), n - allocated)
        if slots <= 0:
            break

        by_tag = defaultdict(list)
        for item in bucket:
            by_tag[item.get("primary_tag", "unknown")].append(item)

        tag_iters = {tag: iter(random.sample(v, len(v))) for tag, v in by_tag.items()}
        tag_cycle = list(tag_iters.keys())
        picked    = []
        i         = 0
        while len(picked) < slots:
            tag = tag_cycle[i % len(tag_cycle)]
            try:
                picked.append(next(tag_iters[tag]))
            except StopIteration:
                tag_cycle.pop(i % len(tag_cycle))
                if not tag_cycle:
                    break
                continue
            i += 1

        sample    += picked
        allocated += len(picked)
        if allocated >= n:
            break

    if len(sample) < n:
        remaining = [x for x in items if x not in sample]
        sample   += random.sample(remaining, min(n - len(sample), len(remaining)))

    return sample[:n]


# ── Prompts ───────────────────────────────────────────────────────────────────

CLUSTER_SYSTEM_PROMPT = """You are an expert at characterizing AI evaluation rubrics.
You will be given a stratified sample of task-agnostic rubric descriptions from a
single cluster. Each item describes a capability or property the rubric tests.

Synthesize them into a coherent cluster label that captures the underlying
capability or property the cluster represents.

Return ONLY valid JSON with exactly these fields:
{
  "label": "<5 words max — the capability category name>",
  "description": "<2 sentences — what the rubrics in this cluster test>",
  "capability_dimension": "<one of: format/schema | completeness | specificity | factuality | consistency | instruction-following | balance/calibration | relevance | other>",
  "representative_example": "<the single most representative abstract_description from the list, verbatim>"
}"""

NOISE_SYSTEM_PROMPT = """You are analyzing AI evaluation rubrics that could NOT be grouped with others —
they are too task-specific or unique to cluster.

Briefly characterize what makes these rubrics resist generalization.

Return ONLY valid JSON:
{
  "label": "Idiosyncratic / Task-Specific",
  "description": "<2 sentences on why these rubrics resist generalization>",
  "capability_dimension": "other",
  "representative_example": "<one example verbatim>"
}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} records from '{path}'.")
    return records


def group_by_cluster(records: list[dict]) -> dict[int, list[dict]]:
    groups = defaultdict(list)
    for r in records:
        groups[r["cluster_id"]].append(r)
    return groups


def call_llm(client: OpenAI, system: str, user: str) -> dict:
    response = client.chat.completions.create(
        model      = CHAT_MODEL,
        max_tokens = 400,
        messages   = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    raw   = response.choices[0].message.content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def build_user_prompt(items: list[dict], sample: list[dict]) -> str:
    abstracts = [r["abstract_description"] for r in sample]
    type_dist = Counter(r.get("capability_type", "other") for r in items)
    type_str  = ", ".join(f"{k}: {v}" for k, v in type_dist.most_common())

    prompt  = f"CLUSTER SIZE: {len(items)} rubric items total\n"
    prompt += f"CAPABILITY TYPE DISTRIBUTION (full cluster): {type_str}\n"
    prompt += f"\nSTRATIFIED SAMPLE ({len(sample)} items, proportional across capability types):\n"
    prompt += "\n".join(f"- {t}" for t in abstracts)
    return prompt


def label_cluster(client: OpenAI, cluster_id: int, items: list[dict]) -> dict:
    sample = stratified_sample(items, SAMPLE_SIZE)
    result = call_llm(client, CLUSTER_SYSTEM_PROMPT, build_user_prompt(items, sample))
    return {
        **result,
        "cluster_id":            cluster_id,
        "size":                  len(items),
        "sample_size_used":      len(sample),
        "capability_type_dist":  dict(Counter(r.get("capability_type", "other") for r in items)),
        "primary_tags":          dict(Counter(r.get("primary_tag", "unknown") for r in items).most_common(5)),
        # Keep a small set of verbatim samples on disk for human inspection
        "sample_abstracts":      [r["abstract_description"] for r in sample[:10]],
        "sample_raw_items":      [r["raw_checklist_item"] for r in sample[:10]],
    }


def label_noise(client: OpenAI, items: list[dict]) -> dict:
    sample = stratified_sample(items, SAMPLE_SIZE)
    result = call_llm(client, NOISE_SYSTEM_PROMPT, build_user_prompt(items, sample))
    return {
        **result,
        "cluster_id":           -1,
        "size":                 len(items),
        "sample_size_used":     len(sample),
        "primary_tags":         dict(Counter(r.get("primary_tag", "unknown") for r in items).most_common(5)),
        "sample_abstracts":     [r["abstract_description"] for r in sample[:10]],
        "sample_raw_items":     [r["raw_checklist_item"] for r in sample[:10]],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    input_path  = os.path.join(args.output_dir, "cluster_assignments.jsonl")
    output_path = os.path.join(args.output_dir, "clusters.json")

    client  = OpenAI()
    records = load_records(input_path)
    groups  = group_by_cluster(records)

    cluster_ids = sorted(k for k in groups if k != -1)
    print(f"Labeling {len(cluster_ids)} clusters + noise bucket...\n")

    all_clusters = {}
    errors       = []

    for cid in cluster_ids:
        items = groups[cid]
        n_sampled = min(SAMPLE_SIZE, len(items))
        print(f"  Cluster {cid:>2}  ({len(items):>4} items, sampling {n_sampled})  ...", end=" ", flush=True)
        try:
            result = label_cluster(client, cid, items)
            all_clusters[cid] = result
            print(f"→ \"{result['label']}\"")
        except Exception as e:
            print(f"⚠ ERROR: {e}")
            errors.append({"cluster_id": cid, "error": str(e)})
        time.sleep(SLEEP)

    if -1 in groups:
        noise_items = groups[-1]
        n_sampled   = min(SAMPLE_SIZE, len(noise_items))
        print(f"  Noise bucket  ({len(noise_items):>4} items, sampling {n_sampled})  ...", end=" ", flush=True)
        try:
            result = label_noise(client, noise_items)
            all_clusters[-1] = result
            print(f"→ \"{result['label']}\"")
        except Exception as e:
            print(f"⚠ ERROR: {e}")
            errors.append({"cluster_id": -1, "error": str(e)})

    with open(output_path, "w") as f:
        json.dump({str(k): v for k, v in all_clusters.items()}, f, indent=2)

    print(f"\n✓ Saved {len(all_clusters)} cluster labels to '{output_path}'.")
    if errors:
        print(f"  ⚠ {len(errors)} errors: {[e['cluster_id'] for e in errors]}")

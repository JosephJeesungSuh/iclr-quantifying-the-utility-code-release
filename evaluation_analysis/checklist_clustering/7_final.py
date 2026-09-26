"""
Stage 7: Final — per-representative-cluster, per-model failure breakdown.

  Inputs:
    - representative_clusters.json   (output of stage 6)
    - cluster_inclusion.jsonl        (output of stage 6 — one row per checklist
                                      item with a multi-label inclusion map)
    - One or more model checklist jsonl files (the wildbench_absolute_binary_
      checking.py "<...>_details.jsonl" outputs, with `checklist_results`)

  Outputs:
    - final_breakdown.json   (per-cluster, per-model satisfied/failed counts
                              and percentages; multi-label means a single item
                              can contribute to multiple clusters)
    - Printed comparative report

Usage:
    python 7_final.py
    python 7_final.py --model_log my-model=/path/to/details.jsonl  (repeatable)
"""

import argparse
import json
import os
from collections import defaultdict


# ── Defaults ─────────────────────────────────────────────────────────────────

_judge_model="claude-haiku-4-5-20251001"
_judge_model="gpt-5"
_judge_model="gemini-2.5-flash-lite"

DEFAULT_MODEL_LOGS = {
    "initial": f"/path/to/usersim/evaluation/results/absolute_binary_checklist/post_generation_judge/Qwen_Qwen2.5-3B-Instruct_{_judge_model}_details.jsonl",
    "trained-with-baseline": f"/path/to/usersim/evaluation/results/absolute_binary_checklist/post_generation_judge/baseline-basic-trained-qwen2.5-3b_{_judge_model}_details.jsonl",
    "trained-with-userlm":   f"/path/to/usersim/evaluation/results/absolute_binary_checklist/post_generation_judge/userlm-trained-qwen2.5-3b_{_judge_model}_details.jsonl",
}


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_satisfaction_lookup(model_log_path: str) -> dict[tuple[str, int], bool]:
    """{(session_id, item_index): satisfied_bool} — same schema as stage 5."""
    lookup = {}
    sessions = load_jsonl(model_log_path)
    for s in sessions:
        sid = s["session_id"]
        for r in s.get("checklist_results") or []:
            lookup[(sid, r["item"])] = bool(r.get("satisfied"))
    return lookup


# ── Aggregation ──────────────────────────────────────────────────────────────

def compute_breakdown(
    inclusion_records: list[dict],
    representative_clusters: list[dict],
    model_lookups: dict[str, dict[tuple[str, int], bool]],
) -> dict:
    """For each representative cluster + model, count satisfied/failed/missing
    over the items the inclusion classifier marked YES for that cluster."""
    cluster_keys = [c["key"] for c in representative_clusters]
    cluster_meta = {c["key"]: c for c in representative_clusters}

    # cluster_key → list of (session_id, item_index)
    members: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for rec in inclusion_records:
        key_pair = (rec["session_id"], rec["item_index"])
        for k in cluster_keys:
            if rec["inclusion"].get(k):
                members[k].append(key_pair)

    breakdown = {}
    for k in cluster_keys:
        keys = members.get(k, [])
        per_model = {}
        for model_name, lookup in model_lookups.items():
            n_satisfied = n_failed = n_missing = 0
            for kp in keys:
                if kp not in lookup:
                    n_missing += 1
                    continue
                if lookup[kp]:
                    n_satisfied += 1
                else:
                    n_failed += 1
            n_total = n_satisfied + n_failed
            per_model[model_name] = {
                "n_total":       n_total,
                "n_satisfied":   n_satisfied,
                "n_failed":      n_failed,
                "n_missing":     n_missing,
                "pct_satisfied": round(n_satisfied / n_total * 100, 2) if n_total else None,
                "pct_failed":    round(n_failed    / n_total * 100, 2) if n_total else None,
            }
        breakdown[k] = {
            "name":         cluster_meta[k]["name"],
            "description":  cluster_meta[k]["description"],
            "question":     cluster_meta[k]["question"],
            "source_cluster_ids": cluster_meta[k].get("source_cluster_ids", []),
            "n_items_in_cluster": len(keys),
            "per_model":    per_model,
        }
    return breakdown


# ── Reporting ────────────────────────────────────────────────────────────────

def print_report(breakdown: dict, model_names: list[str], n_items_total: int) -> None:
    print("\n" + "═" * 84)
    print("  PER-REPRESENTATIVE-CLUSTER × PER-MODEL BREAKDOWN")
    print("═" * 84)
    print(f"  Total checklist items classified: {n_items_total}")
    print(f"  Models compared: {', '.join(model_names)}")

    for k, info in breakdown.items():
        share = info["n_items_in_cluster"] / n_items_total * 100 if n_items_total else 0.0
        print(f"\n▶ {info['name'].upper()}  "
              f"({info['n_items_in_cluster']} items, {share:.1f}% of all items)")
        print(f"  {info['description']}")
        for m in model_names:
            pm = info["per_model"][m]
            ps = f"{pm['pct_satisfied']:.1f}%" if pm["pct_satisfied"] is not None else "—"
            pf = f"{pm['pct_failed']:.1f}%"    if pm["pct_failed"]    is not None else "—"
            print(f"    [{m:<32}]  satisfied {ps:>6}  /  failed {pf:>6}   "
                  f"({pm['n_satisfied']}/{pm['n_total']} sat, missing={pm['n_missing']})")

    print("\n" + "═" * 84)


# ── Argparse ─────────────────────────────────────────────────────────────────

def parse_model_log(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("--model_log expects NAME=PATH")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    parser.add_argument(
        "--model_log", action="append", type=parse_model_log, default=None,
        help="NAME=PATH for a model's wildbench_absolute_binary_checking "
             "details jsonl. Repeatable. If omitted, uses built-in defaults.",
    )
    args = parser.parse_args()

    clusters_path  = os.path.join(args.output_dir, "representative_clusters.json")
    inclusion_path = os.path.join(args.output_dir, "cluster_inclusion.jsonl")
    out_path       = os.path.join(args.output_dir, "final_breakdown.json")

    with open(clusters_path) as f:
        representative_clusters = json.load(f)
    inclusion_records = load_jsonl(inclusion_path)
    print(f"Loaded {len(representative_clusters)} representative clusters.")
    print(f"Loaded {len(inclusion_records)} inclusion records.")

    model_logs = dict(args.model_log) if args.model_log else DEFAULT_MODEL_LOGS
    print(f"\nModels being compared:")
    for name, path in model_logs.items():
        print(f"  - {name:<32} {path}")

    model_lookups = {}
    for name, path in model_logs.items():
        model_lookups[name] = build_satisfaction_lookup(path)
        print(f"    {name}: {len(model_lookups[name])} (session, item) verdicts loaded.")

    print("\nComputing breakdown ...")
    breakdown = compute_breakdown(inclusion_records, representative_clusters, model_lookups)

    payload = {
        "models":                  list(model_logs.keys()),
        "model_log_paths":         model_logs,
        "n_inclusion_records":     len(inclusion_records),
        "representative_clusters": representative_clusters,
        "breakdown":               breakdown,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Saved breakdown to '{out_path}'.")

    print_report(breakdown, list(model_logs.keys()), len(inclusion_records))

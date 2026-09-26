"""
Stage 5: Taxonomy + Per-Model Failure Breakdown
  Inputs:
    - clusters.json                    (output of stage 4)
    - cluster_assignments.jsonl        (output of stage 3 — has session_id, item_index, cluster_id)
    - One or more model checklist jsonl files (the model "performance logs")

  Outputs:
    - taxonomy.json                    (top-level dimensions + per-model failure rates per cluster)
    - per_cluster_model_breakdown.json (full per-cluster, per-model satisfied/failed counts)
    - per_dimension_model_breakdown.json (rolled up to taxonomy dimensions)
    - Printed comparative report

For each model jsonl, we look up satisfied/failed for every (session_id, item_index)
pair and aggregate into the cluster + dimension structure produced by stages 1-4.
"""

import argparse
import json
import os
from collections import defaultdict
from openai import OpenAI


# ── Config ────────────────────────────────────────────────────────────────────
CHAT_MODEL = "gpt-4.1-mini"

# Default model logs to compare. Override via --model_log NAME=PATH (repeatable).
DEFAULT_MODEL_LOGS = {
    "trained-with-baseline": "/path/to/usersim/evaluation/results/absolute_binary_checklist/trained-with-baseline--step-140_binary_checklist_details.jsonl",
    "trained-with-userlm":   "/path/to/usersim/evaluation/results/absolute_binary_checklist/trained-with-userlm--dedup-exact--step-120_binary_checklist_details.jsonl",
}


# ── Prompt ────────────────────────────────────────────────────────────────────

TAXONOMY_SYSTEM_PROMPT = """You are building a structured taxonomy of AI capability evaluation rubrics.
You will be given ~15-30 cluster labels and descriptions, where each cluster
groups rubric items that test a similar capability.
Group them into ~10 top-level capability dimensions — broad categories that
capture a common underlying capability across clusters.

Return ONLY valid JSON:
{
  "dimensions": [
    {
      "name": "<dimension name, 3-5 words>",
      "description": "<1 sentence capturing the shared capability>",
      "cluster_ids": [<list of integer cluster IDs belonging to this dimension>]
    }
  ]
}

Do NOT include cluster_id -1 (the noise bucket) — it is handled separately."""


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_clusters(path: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_satisfaction_lookup(model_log_path: str) -> dict[tuple[str, int], bool]:
    """
    Returns {(session_id, item_index): satisfied_bool} for every checklist
    result in this model's log. item_index is 1-indexed to match the checklist.
    """
    lookup = {}
    sessions = load_jsonl(model_log_path)
    for s in sessions:
        sid = s["session_id"]
        for r in s.get("checklist_results", []):
            lookup[(sid, r["item"])] = bool(r.get("satisfied"))
    return lookup


# ── Per-cluster, per-model aggregation ────────────────────────────────────────

def compute_per_cluster_breakdown(
    cluster_assignments: list[dict],
    model_lookups: dict[str, dict[tuple[str, int], bool]],
) -> dict[int, dict]:
    """
    For each cluster_id, compute per-model:
      - n_total      (items in the cluster that this model has a verdict for)
      - n_satisfied  (passed)
      - n_failed     (failed)
      - n_missing    (cluster items with no verdict in this model's log — should be 0 if logs cover the same sessions)
      - pct_failed   (n_failed / n_total * 100)
    """
    by_cluster: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for rec in cluster_assignments:
        by_cluster[rec["cluster_id"]].append((rec["session_id"], rec["item_index"]))

    breakdown = {}
    for cid, keys in by_cluster.items():
        per_model = {}
        for model_name, lookup in model_lookups.items():
            n_satisfied = 0
            n_failed    = 0
            n_missing   = 0
            for k in keys:
                if k not in lookup:
                    n_missing += 1
                    continue
                if lookup[k]:
                    n_satisfied += 1
                else:
                    n_failed += 1
            n_total = n_satisfied + n_failed
            per_model[model_name] = {
                "n_total":     n_total,
                "n_satisfied": n_satisfied,
                "n_failed":    n_failed,
                "n_missing":   n_missing,
                "pct_failed":  round(n_failed / n_total * 100, 2) if n_total else None,
                "pct_satisfied": round(n_satisfied / n_total * 100, 2) if n_total else None,
            }
        breakdown[cid] = {
            "cluster_size": len(keys),
            "per_model":    per_model,
        }
    return breakdown


# ── Taxonomy construction ─────────────────────────────────────────────────────

def build_taxonomy(client: OpenAI, clusters: dict) -> dict:
    summaries = []
    for cid, c in sorted(clusters.items()):
        if cid == -1:
            continue
        summaries.append(
            f"[ID {cid}] \"{c.get('label', '?')}\"  ({c['size']} items)\n"
            f"  {c.get('description', '')}"
        )

    user_prompt = "Here are the cluster labels and descriptions:\n\n"
    user_prompt += "\n\n".join(summaries)
    user_prompt += "\n\nGroup these into 5-7 top-level capability dimensions."

    response = client.chat.completions.create(
        model      = CHAT_MODEL,
        max_tokens = 900,
        messages   = [
            {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    )

    raw   = response.choices[0].message.content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(clean)


def enrich_taxonomy(
    taxonomy: dict,
    clusters: dict,
    cluster_breakdown: dict[int, dict],
    model_names: list[str],
) -> dict:
    """
    Decorate each dimension with cluster details + per-model rollups, and
    attach the noise bucket separately with its own per-model breakdown.
    """
    for dim in taxonomy["dimensions"]:
        dim_clusters = []
        per_model_totals = {m: {"n_total": 0, "n_satisfied": 0, "n_failed": 0, "n_missing": 0}
                            for m in model_names}

        for cid in dim["cluster_ids"]:
            if cid not in clusters:
                continue
            c = clusters[cid]
            cb = cluster_breakdown.get(cid, {"per_model": {}})
            dim_clusters.append({
                "id":          cid,
                "label":       c.get("label", "?"),
                "description": c.get("description", ""),
                "size":        c["size"],
                "per_model":   cb["per_model"],
            })
            for m in model_names:
                pm = cb["per_model"].get(m, {})
                for field in ("n_total", "n_satisfied", "n_failed", "n_missing"):
                    per_model_totals[m][field] += pm.get(field, 0)

        # Compute pct fields after rollup
        for m in model_names:
            t = per_model_totals[m]
            t["pct_failed"]    = round(t["n_failed"] / t["n_total"] * 100, 2)    if t["n_total"] else None
            t["pct_satisfied"] = round(t["n_satisfied"] / t["n_total"] * 100, 2) if t["n_total"] else None

        dim["clusters"]       = dim_clusters
        dim["total_items"]    = sum(c["size"] for c in dim_clusters)
        dim["per_model"]      = per_model_totals

    # Noise bucket
    if -1 in clusters:
        noise = clusters[-1]
        cb    = cluster_breakdown.get(-1, {"per_model": {}})
        total_all = sum(c["size"] for c in clusters.values())
        taxonomy["idiosyncratic_bucket"] = {
            "name":           noise.get("label", "Idiosyncratic / Task-Specific"),
            "description":    noise.get("description", ""),
            "total_items":    noise["size"],
            "pct_of_total":   round(noise["size"] / total_all * 100, 1),
            "per_model":      cb["per_model"],
        }

    return taxonomy


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(taxonomy: dict, model_names: list[str], total: int) -> None:
    print("\n" + "═" * 78)
    print("  CAPABILITY TAXONOMY REPORT — per-model failure rates")
    print("═" * 78)

    for dim in taxonomy.get("dimensions", []):
        share = dim["total_items"] / total * 100
        print(f"\n▶ {dim['name'].upper()}  ({dim['total_items']} items, {share:.1f}% of all rubrics)")
        print(f"  {dim['description']}")
        for m in model_names:
            pm = dim["per_model"][m]
            pf = f"{pm['pct_failed']:.1f}%" if pm["pct_failed"] is not None else "—"
            print(f"    [{m:<30}]  fail rate: {pf:>6}   ({pm['n_failed']:>4}/{pm['n_total']:<4} items)")

        print("  CLUSTERS:")
        for c in dim.get("clusters", []):
            print(f"    • [{c['size']:>4}]  {c['label']}")
            print(f"             {c['description'][:90]}")
            for m in model_names:
                pm = c["per_model"].get(m, {})
                pf = f"{pm.get('pct_failed'):.1f}%" if pm.get("pct_failed") is not None else "—"
                print(f"             {m:<30}  fail rate: {pf:>6}   ({pm.get('n_failed', 0)}/{pm.get('n_total', 0)})")

    bucket = taxonomy.get("idiosyncratic_bucket", {})
    if bucket:
        print(f"\n▶ {bucket['name'].upper()}  ({bucket['total_items']} items, {bucket['pct_of_total']}%)")
        print(f"  {bucket['description']}")
        for m in model_names:
            pm = bucket["per_model"].get(m, {})
            pf = f"{pm.get('pct_failed'):.1f}%" if pm.get("pct_failed") is not None else "—"
            print(f"    [{m:<30}]  fail rate: {pf:>6}   ({pm.get('n_failed', 0)}/{pm.get('n_total', 0)})")

    print("\n" + "═" * 78)


# ── Argparse helper ───────────────────────────────────────────────────────────

def parse_model_log(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("--model_log expects NAME=PATH")
    name, path = s.split("=", 1)
    return name.strip(), path.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model_log", action="append", type=parse_model_log,
                        default=None,
                        help="NAME=PATH for a model's checklist jsonl. Repeatable. "
                             "If omitted, uses built-in defaults.")
    args = parser.parse_args()

    clusters_path    = os.path.join(args.output_dir, "clusters.json")
    assignments_path = os.path.join(args.output_dir, "cluster_assignments.jsonl")
    taxonomy_path    = os.path.join(args.output_dir, "taxonomy.json")
    cluster_bd_path  = os.path.join(args.output_dir, "per_cluster_model_breakdown.json")
    dim_bd_path      = os.path.join(args.output_dir, "per_dimension_model_breakdown.json")

    model_logs = dict(args.model_log) if args.model_log else DEFAULT_MODEL_LOGS
    print(f"Models being compared:")
    for name, path in model_logs.items():
        print(f"  - {name:<30} {path}")

    clusters    = load_clusters(clusters_path)
    assignments = load_jsonl(assignments_path)
    total_items = sum(c["size"] for c in clusters.values())
    print(f"\nLoaded {len(clusters)} clusters ({total_items} total items).")
    print(f"Loaded {len(assignments)} cluster assignments.")

    # Build per-model satisfaction lookups
    model_lookups = {}
    for name, path in model_logs.items():
        model_lookups[name] = build_satisfaction_lookup(path)
        print(f"  {name}: {len(model_lookups[name])} (session, item) verdicts loaded.")

    # Per-cluster, per-model breakdown
    print("\nComputing per-cluster, per-model breakdown...")
    cluster_breakdown = compute_per_cluster_breakdown(assignments, model_lookups)
    with open(cluster_bd_path, "w") as f:
        json.dump({str(k): v for k, v in cluster_breakdown.items()}, f, indent=2)
    print(f"✓ Saved per-cluster breakdown to '{cluster_bd_path}'.")

    # Taxonomy
    print("\nBuilding taxonomy from cluster labels...")
    client   = OpenAI()
    taxonomy = build_taxonomy(client, clusters)

    model_names = list(model_logs.keys())
    taxonomy = enrich_taxonomy(taxonomy, clusters, cluster_breakdown, model_names)

    with open(taxonomy_path, "w") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"✓ Taxonomy saved to '{taxonomy_path}'.")

    # Per-dimension rollup as a standalone artifact
    dim_breakdown = {
        dim["name"]: {
            "description": dim["description"],
            "total_items": dim["total_items"],
            "per_model":   dim["per_model"],
            "cluster_ids": dim.get("cluster_ids", []),
        }
        for dim in taxonomy["dimensions"]
    }
    if "idiosyncratic_bucket" in taxonomy:
        dim_breakdown["__noise__"] = {
            "description": taxonomy["idiosyncratic_bucket"]["description"],
            "total_items": taxonomy["idiosyncratic_bucket"]["total_items"],
            "per_model":   taxonomy["idiosyncratic_bucket"]["per_model"],
            "cluster_ids": [-1],
        }
    with open(dim_bd_path, "w") as f:
        json.dump(dim_breakdown, f, indent=2)
    print(f"✓ Saved per-dimension breakdown to '{dim_bd_path}'.")

    print_report(taxonomy, model_names, total_items)

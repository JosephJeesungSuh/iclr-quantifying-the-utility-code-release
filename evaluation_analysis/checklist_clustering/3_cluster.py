"""
Stage 3: Cluster
  Input:  embeddings.npy + abstracted_items.jsonl
  Output: cluster_assignments.jsonl   (records + cluster_id field)
          embeddings_reduced.npy      (cached UMAP output)

Run: python 3_clsuter.py [--output_dir output]

UMAP reduces high-dim embeddings → low-dim before HDBSCAN
(HDBSCAN fails on raw high-dim due to curse of dimensionality).

Tune:
  MIN_CLUSTER_SIZE  →  larger = fewer clusters, smaller = more clusters
  Tune to land in ~15-30 cluster range for ~10k items.
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors

try:
    import hdbscan
    import umap
except ImportError:
    raise ImportError("Run: pip install hdbscan umap-learn scikit-learn")

# ── Config ────────────────────────────────────────────────────────────────────
UMAP_N_COMPONENTS = 20
UMAP_N_NEIGHBORS  = 30
UMAP_MIN_DIST     = 0.05
UMAP_RANDOM_STATE = 42

MIN_CLUSTER_SIZE  = 100   # primary tuning knob — ~10k items → ~20-30 clusters
MIN_SAMPLES       = 3
CLUSTER_SELECTION = "eom"


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def check_embedding_quality(reduced: np.ndarray, k: int = 5) -> None:
    nbrs = NearestNeighbors(n_neighbors=k).fit(reduced)
    distances, _ = nbrs.kneighbors(reduced)
    kth_distances = np.sort(distances[:, -1])

    print(f"\n── k-NN Distance Stats (k={k}) ────────────────────")
    print(f"  min:    {kth_distances.min():.4f}")
    print(f"  median: {np.median(kth_distances):.4f}")
    print(f"  max:    {kth_distances.max():.4f}")
    print(f"  std:    {kth_distances.std():.4f}")
    print(f"  ratio max/median: {kth_distances.max()/np.median(kth_distances):.1f}x")
    print(f"  → {'✓ Good separation' if kth_distances.std() > 0.1 else '⚠ Flat — try lower n_neighbors or n_components'}")


def reduce_dimensions(embeddings: np.ndarray, reduced_path: str) -> np.ndarray:
    if os.path.exists(reduced_path):
        print(f"Loading cached UMAP output from '{reduced_path}'...")
        return np.load(reduced_path)

    print(f"Normalizing {embeddings.shape[1]}-dim embeddings...")
    normed = normalize(embeddings)

    print(f"Running UMAP: {embeddings.shape[1]} → {UMAP_N_COMPONENTS} dims  "
          f"(n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST})...")
    print("  This may take a few minutes for ~10k points...")

    reducer = umap.UMAP(
        n_components  = UMAP_N_COMPONENTS,
        n_neighbors   = UMAP_N_NEIGHBORS,
        min_dist      = UMAP_MIN_DIST,
        metric        = "cosine",
        random_state  = UMAP_RANDOM_STATE,
        low_memory    = False,
    )
    reduced = reducer.fit_transform(normed)
    check_embedding_quality(reduced)

    np.save(reduced_path, reduced)
    print(f"UMAP output saved to '{reduced_path}'.")
    return reduced


def cluster(reduced: np.ndarray) -> np.ndarray:
    print(f"\nRunning HDBSCAN on {reduced.shape[1]}-dim embeddings  "
          f"(min_cluster_size={MIN_CLUSTER_SIZE}, min_samples={MIN_SAMPLES})...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size         = MIN_CLUSTER_SIZE,
        min_samples              = MIN_SAMPLES,
        metric                   = "euclidean",
        cluster_selection_method = CLUSTER_SELECTION,
    )
    labels = clusterer.fit_predict(reduced)
    return labels


def print_summary(labels: np.ndarray) -> None:
    counts     = Counter(labels)
    n_clusters = len([k for k in counts if k != -1])
    n_noise    = counts.get(-1, 0)
    total      = len(labels)

    print(f"\n── Clustering Summary ─────────────────────────────")
    print(f"  Clusters found : {n_clusters}")
    print(f"  Noise points   : {n_noise} ({n_noise / total * 100:.1f}%)  → idiosyncratic bucket")

    print(f"\n  Cluster sizes:")
    for cid, count in sorted(counts.items(), key=lambda x: -x[1]):
        tag = "NOISE" if cid == -1 else f"Cluster {cid:>2}"
        bar = "█" * int(count / max(counts.values()) * 30)
        print(f"    {tag:<12}  {count:>4}  {bar}")

    if n_clusters == 0:
        print(f"\n  ⚠  No clusters — try lowering MIN_CLUSTER_SIZE")
    elif n_clusters < 10:
        print(f"\n  💡 Too few clusters — lower MIN_CLUSTER_SIZE to ~{MIN_CLUSTER_SIZE // 2}")
    elif n_clusters > 40:
        print(f"\n  💡 Too many clusters — raise MIN_CLUSTER_SIZE to ~{MIN_CLUSTER_SIZE * 2}")
    else:
        print(f"\n  ✓ Cluster count looks good. Proceed to 4_label.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    embeddings_path = os.path.join(args.output_dir, "embeddings.npy")
    records_path    = os.path.join(args.output_dir, "abstracted_items.jsonl")
    reduced_path    = os.path.join(args.output_dir, "embeddings_reduced.npy")
    output_path     = os.path.join(args.output_dir, "cluster_assignments.jsonl")

    print(f"Loading embeddings from '{embeddings_path}'...")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")

    records = load_records(records_path)
    assert len(records) == len(embeddings), \
        f"Mismatch: {len(records)} records vs {len(embeddings)} embeddings"

    reduced = reduce_dimensions(embeddings, reduced_path)
    labels  = cluster(reduced)
    print_summary(labels)

    with open(output_path, "w") as f:
        for record, label in zip(records, labels):
            f.write(json.dumps({**record, "cluster_id": int(label)}) + "\n")

    print(f"\n✓ Saved cluster assignments to '{output_path}'.")

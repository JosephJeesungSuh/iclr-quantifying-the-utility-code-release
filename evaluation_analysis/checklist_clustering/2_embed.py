"""
Stage 2: Embed
  Input:  abstracted_items.jsonl   (output of stage 1)
  Output: embeddings.npy           (one row per record, in file order)

Run: python 2_embed.py [--output_dir output]

The embedding is over the `abstract_description` field — the task-agnostic
restatement of the rubric, so semantically similar capabilities cluster together
regardless of source domain.
"""

import argparse
import json
import os
import time
import numpy as np
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE      = 512
SLEEP           = 0.1


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} records from '{path}'.")
    return records


def embed(records: list[dict], output_path: str) -> np.ndarray:
    if os.path.exists(output_path):
        print(f"'{output_path}' already exists. Delete it to re-embed. Loading cached.")
        cached = np.load(output_path)
        if cached.shape[0] != len(records):
            raise ValueError(
                f"Cached embeddings ({cached.shape[0]}) do not match record count "
                f"({len(records)}). Delete '{output_path}' and re-run."
            )
        return cached

    client = OpenAI()
    texts  = [r["abstract_description"] for r in records]
    all_embeddings = []

    n_batches = (len(texts) - 1) // BATCH_SIZE + 1
    print(f"Embedding {len(texts)} texts in {n_batches} batches...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{n_batches}  ({len(batch)} items)")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([d.embedding for d in response.data])
        time.sleep(SLEEP)

    embeddings = np.array(all_embeddings)
    np.save(output_path, embeddings)
    print(f"\n✓ Saved {embeddings.shape} embeddings to '{output_path}'.")
    return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    input_path  = os.path.join(args.output_dir, "abstracted_items.jsonl")
    output_path = os.path.join(args.output_dir, "embeddings.npy")

    records = load_records(input_path)
    embed(records, output_path)

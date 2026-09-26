# Checklist clustering pipeline (Appendix G.3)

Seven-stage pipeline that turns the 11,667 conversation-specific checklist
items in WildBench-v2 into nine representative response dimensions used in
the paper’s per-dimension analysis.

| Stage | Script                                       | Purpose                                                                                                    |
| ----- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1     | `1_structural_abstraction.py`                | Rewrite each (conversation, checklist item) into a task-agnostic 1-sentence description via `gpt-5`. |
| 2     | `2_embed.py`                                 | Embed the abstracted descriptions with `text-embedding-3-large` (3 072 dims).                              |
| 3     | `3_cluster.py`                               | UMAP → 20-D, HDBSCAN with `min_cluster_size=100`, EOM cluster selection.                                   |
| 4     | `4_label.py`                                 | Stratified-sample 100 items per cluster and ask `gpt-5` for a 5-word cluster label and description.   |
| 5     | `5_taxonomy.py`                              | Group the cluster labels into ~10 top-level capability dimensions.                                         |
| 6     | `6_post_taxonomy_binary_inclusion.py`        | For each item, ask `gpt-5` whether it primarily tests each of the nine curated dimensions.                 |
| 7     | `7_final.py`                                 | Aggregate per-(dimension, assistant) satisfaction rates from the binary inclusion + checklist results.     |

The pipeline is **agnostic to the assistant under evaluation** — it only reads
(conversation prefix, checklist item) pairs, never any assistant's response.
This guarantees that the resulting per-dimension breakdown is comparable across
the initial, RPUSER1-, RPUSER2-, and SFTUSER-trained assistants.

## Inputs / outputs

* Stage 1 reads WildBench-v2 directly via `datasets.load_dataset`.
* Stage 6 also reads the per-assistant binary checklist outputs produced by
  `../../wildbench_evaluation/wildbench_absolute_binary_checking.py`. Update the path
  constants at the top of `5_taxonomy.py` and `6_post_taxonomy_binary_inclusion.py`
  to point at your binary-checking output JSONLs.
* Each stage writes its output to a JSON / JSONL / Parquet file in the working
  directory; downstream stages pick those up automatically.

## Curation note

The nine dimensions reported in the paper (Neutrality & Cultural Sensitivity,
Specificity & Detail, Response Clarity, Length & Format Adherence,
Anticipating Limitations & Offering Alternatives, UI / Layout / Chart
Production, Code Correctness, Numerical Accuracy, Sourcing & Citation
Accuracy) are a **manual curation** of the LLM-generated taxonomy from stage 5
(see App. G.3). Stage 6 directly classifies each item along these nine
dimensions; stages 4–5 are kept primarily for reproducibility of how we
arrived at the curated set.

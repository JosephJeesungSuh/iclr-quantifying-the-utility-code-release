# Additional analysis

- [Checklist clustering](checklist_clustering/README.md) reconstructs the
  response dimensions used for WildBench analysis (Appendix G.3).
- [Fidelity metrics](fidelity_metrics/README.md) implements Appendix I.1 comparisons
  between simulated and human utterances. These scripts are not required for
  the main RL or WildBench pipeline.

These analysis scripts expect generated evaluation outputs and optional NLP,
embedding, clustering, and plotting dependencies. Inspect each script's imports
and configure its input paths before running it. No generated outputs are included.

# User-simulator fidelity metrics (Appendix I.1)

Replays each WildBench-v2 conversation turn-by-turn through a candidate user
simulator and compares the simulated `ut` to the gold human `ut` on nine
fidelity metrics :

* **Average word length** — ℓ1 distance between mean characters per word.
* **Utterance length** — JSD over word-count distributions.
* **Typo rate** — ℓ1 distance over the fraction of tokens flagged by
  `pyspellchecker`.
* **POS distribution** — JSD over `spaCy` POS tags.
* **Punctuation distribution** — JSD over punctuation characters.
* **Capitalization** — JSD over capitalization patterns.
* **Sentiment** — JSD over labels from
  `distilbert-base-multilingual-cased-sentiments-student`.
* **SBERT semantic distance** — `1 − cos` similarity from
  `all-MiniLM-L6-v2`.
* **Terminal F1** — binary F1 over continue / terminate decisions; ground
  truth is `terminate` only past the last human turn.

Lower is better for the eight distance metrics; higher is better for terminal
F1. The script also reports correlation of these metrics with downstream
WildBench WB-Score / WB-Reward gaps when a `--wb_details` jsonl is supplied.

## Files

* `generate_sim_turns.py` — for each WildBench v2 conversation, condition the
  candidate simulator on the prefix `{u1, a1, …, ut-1, at-1}` and elicit a
  candidate `ut`. Both the SFT-style raw-completion path (UserLM) and the
  prompted RP path are supported.
* `analyze_usersim_metrics.py` — compute the nine metrics over the saved
  simulator turns and produce `summary_table.csv`, `per_session.csv`,
  `correlation.csv`, and a Spearman bar plot.

## Inputs

`generate_sim_turns.py` expects two vLLM endpoints (one for the UserLM and one
for the RP simulator) and writes a single JSONL with the simulated and gold
turns. `analyze_usersim_metrics.py` reads that JSONL and (optionally) a
WildBench pairwise details JSONL produced by
`../../wildbench_evaluation/wildbench_pairwise.py`.

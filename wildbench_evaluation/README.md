# WildBench checklist evaluation

Appendix G evaluates WildBench v2 with three judges: `gpt-5-mini`,
`gemini-2.5-flash`, and `claude-haiku-4-5`. Each judge evaluates the same cached
assistant response against the task's binary checklist. The release implements
majority voting per checklist item, as specified in Appendix G.2.

## Run

Install `requirements.txt` in a separate evaluation environment. Supply the
provider credentials through `OPENAI_API_KEY`, `GEMINI_API_KEY`, and
`ANTHROPIC_API_KEY`. From this directory:

```bash
for judge in gpt-5-mini gemini-2.5-flash claude-haiku-4-5; do
  python wildbench_absolute_binary_checking.py \
    --model model-A --model_path /path/to/model-A --judge_model "$judge" \
    --output_dir /path/to/evaluation --temperature 1.0 --top_p 1.0 --max_tokens 2048
  python wildbench_absolute_binary_checking.py \
    --model model-B --model_path /path/to/model-B --judge_model "$judge" \
    --output_dir /path/to/evaluation --temperature 1.0 --top_p 1.0 --max_tokens 2048
done
python aggregate_judges.py --models model-A model-B \
  --input-dir /path/to/evaluation/post_generation_judge \
  --output-dir /path/to/evaluation/majority
```

Local models generate through vLLM; API models use their provider clients.
`--model` names the output and `--model_path` identifies the actual weights/API
model. Generation caches are under `generation/`; individual judge results are
under `post_generation_judge/`. Use a new output directory whenever weights,
tasks, or sampling settings change: existing caches are reused.

The aggregator requires identical session coverage, checklists, and response
text across judges, and rejects incomplete judgments. Its outputs retain the
input and response text for the checklist-clustering pipeline. Satisfaction is
averaged over conversations, with a standard error of the mean. Pairwise wins
compare the number of satisfied items; ties receive half credit. `metrics.py`
implements the tie-adjusted Wald standard error and 95% confidence interval
from Appendix G.1. Rates inside `tie_accounted_statistics` are fractions;
headline pairwise summary rates are percentages.

For a single-judge diagnostic, compare two complete result files with
`wildbench_absolute_binary_checking_then_pairwise.py --model1 model-A
--model2 model-B --judge_model <judge> --input_dir <details-directory>`.
The other inherited rubric/generation scripts are optional utilities, not the
three-judge paper protocol. No evaluation outputs are bundled.

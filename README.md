# Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants

Anonymous source release for the **ICLR 2027** submission.

This snapshot contains source code only. Checkpoints, WildChat conversations,
generated trajectories, participant records, and the manuscript are not bundled.
Supply your own data and model paths; trained checkpoints and de-identified
study data are planned for release upon acceptance.

## Layout

| Directory | Purpose | Paper reference |
| --- | --- | --- |
| [sftuser_training](sftuser_training/README.md) | WildChat preparation, intent generation, dialogue flipping, and learned user simulator SFT; includes the customized llama-cookbook implementation | §3.2, Appendix C |
| [rl_training](rl_training/README.md) | Assistant GRPO training, role-playing simulators, multi-judge reward, and cross-simulator evaluation; minimal integration with pinned public SkyRL | §3.1, §4.1, §4.4, Appendices D–F |
| [wildbench_evaluation](wildbench_evaluation/README.md) | Checklist scoring and pairwise evaluation | §4.2, Appendix G |
| [Writing task bank](pairwise_evaluation_web_serving/writing_task_bank.py) | Human evaluation writing tasks and pre-writing questions | §5, Appendix H |
| [evaluation_analysis](evaluation_analysis/README.md) | Checklist clustering and simulator-fidelity analysis | Appendices G.3 and I.1 |

## Reproduction

1. Follow the SFT README to prepare the WildChat splits, generate intents, and
   train a user simulator. The larger-corpus preprocessing script is also included.
2. Follow the RL README to convert the same intent-annotated splits to Parquet,
   serve the frozen user simulator and judges, and launch assistant training.
   `rl_training/run_paper.py` explicitly sets the main configuration described
   in the paper and supports a CPU-only `--dry-run`.
3. Use the WildBench scripts for automatic evaluation. The human evaluation
   writing task bank contains the document types, intents, and pre-writing questions.

Use separate Python environments for SFT, RL, and evaluation because their GPU
packages have different version constraints. Component READMEs contain setup
commands and required inputs. Full training requires CUDA GPUs, model weights,
and prepared datasets.

## Dependencies

API credentials are supplied through environment variables. Third-party source
attribution and dependency license notices are retained. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

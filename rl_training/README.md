# Assistant reinforcement learning

Minimal Appendix D–F implementation: three user environments, paper prompts,
data preparation, and a launcher. The training framework is an external, pinned
public SkyRL dependency. No private framework fork, unrelated environments,
experimental launch scripts, checkpoints, or trajectories are included.

## Install

Use Python 3.12 and CUDA compatible with PyTorch 2.8 and vLLM 0.11.
Run from the release root, replacing `/path/to/SkyRL` with a new checkout:

```bash
git clone --branch skyrl_train-v0.3.0 --depth 1 https://github.com/NovaSky-AI/SkyRL.git /path/to/SkyRL
# install.py verifies commit b2a08a0fc3f01f562d47db31a544a6fa1ce46bbe
python rl_training/install.py /path/to/SkyRL
cd /path/to/SkyRL/skyrl-train
uv sync --extra vllm
```

The installer copies only `env.py`, `prompts.py`, and `main.py` into
`examples/userlm_paper`. Upstream's custom environment interface handles the
rewritten first turn, multi-turn loss masks, and terminal rewards without a
framework patch. Installation may require a CUDA build toolchain.

## Prepare data

Use the same intent-annotated, user-disjoint splits as SFT. With pandas and
pyarrow installed, run from the release root:

```bash
for env in userlm userlm_baseline userlm_baseline_mixture; do
  python rl_training/prepare_data.py \
    --input_dir /path/to/data_with_intents --output_dir /path/to/rl_data \
    --intent_model_name Qwen--Qwen3-32B --env_class "$env" \
    --seed 42 --validation_size 10000
done
```

Inputs are `train_with_intents_Qwen--Qwen3-32B.jsonl`, `val_with_intents_...jsonl`,
and `test_with_intents_...jsonl`. Outputs contain full splits and a deterministic
validation subset of up to 10,000 examples. RPUSER3 receives one persona seed per
example, keeping the persona fixed across its GRPO group.

## Serve frozen models

From the external checkout's `skyrl-train` directory, run each server separately.
The GPU assignments below are examples; allocate hardware for your model sizes.

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run --extra vllm vllm serve /path/to/user-simulator \
  --tensor-parallel-size 2 --port 8002 --host 127.0.0.1
CUDA_VISIBLE_DEVICES=4,5 uv run --extra vllm vllm serve mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
  --tensor-parallel-size 2 --port 8003 --host 127.0.0.1
CUDA_VISIBLE_DEVICES=6,7 uv run --extra vllm vllm serve Qwen/Qwen3-32B \
  --tensor-parallel-size 2 --port 8004 --host 127.0.0.1
```

For RPUSER1–3, serve `Qwen/Qwen2.5-14B-Instruct` on port 8002 instead. An SFTUSER
checkpoint must include the prepared tokenizer and be converted to Hugging Face
format. Launcher model identifiers must match those exposed by the servers.

## Train and evaluate

From the release root:

```bash
CUDA_VISIBLE_DEVICES=0,1 python rl_training/run_paper.py \
  --skyrl-dir /path/to/SkyRL --simulator sftuser \
  --user-model /path/to/user-simulator \
  --data-dir /path/to/rl_data --output-dir /path/to/run

# Print the command without installing packages or loading data/models.
python rl_training/run_paper.py --simulator rpuser3 \
  --data-dir /path/to/rl_data --output-dir /path/to/run --dry-run

# Cross-simulator evaluation: choose the simulator and an exported assistant.
CUDA_VISIBLE_DEVICES=0,1 python rl_training/run_paper.py \
  --skyrl-dir /path/to/SkyRL --simulator sftuser --evaluate \
  --agent-model /path/to/assistant --user-model /path/to/evaluation-user-simulator \
  --data-dir /path/to/rl_data --output-dir /path/to/evaluation
```

| Variant | Environment | Dataset suffix | First user turn |
| --- | --- | --- | --- |
| `sftuser` | `userlm` | none | Real utterance |
| `rpuser1` | `userlm_baseline` | `_baseline` | Generated |
| `rpuser2` | `userlm_baseline` | `_baseline` | Real utterance |
| `rpuser3` | `userlm_baseline_mixture` | `_baseline_mixture` | Real utterance; sampled persona |

Defaults implement Qwen2.5-3B-Instruct, synchronous GRPO, batch 64, group 5,
minibatch 16 (four updates), fixed learning rate 8e-7, KL3 coefficient 0.001,
clip 0.2, and gradient norm 1. Validation and checkpoints occur every 20 steps.
Select the checkpoint at the onset of the validation plateau from the saved
metrics; the launcher does not invent an automatic plateau threshold.

Conversations last at most five assistant turns. Assistant sampling uses
2,048 tokens and temperature 1; users use 1,024 tokens, temperature 0.7 and top-p
0.9. Repeated SFTUSER utterances trigger retries with temperature increments of
0.05, capped at 1. Two temperature-zero judges score the Appendix E rubric;
rewards average their scores divided by ten. Evaluation uses the held-out test
split and the same generation/judging settings. Scores print to the console.

Use `--agent-model`, `--user-model`, `--user-base-model`, `--user-tokenizer`, and
`--num-gpus` for other model sizes. `SKYRL_DIR` can replace `--skyrl-dir`.
Additional Hydra overrides apply last and can depart from the paper defaults.
Console logging is enabled by default.

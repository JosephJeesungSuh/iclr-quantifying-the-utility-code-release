# User Simulator SFT (SFTuser)

Supervised fine-tuning pipeline for training a **user language model** (a model that plays the *user* side of a conversation) from WildChat. It reproduces the "flipping the dialogue" recipe: preprocess WildChat, summarize each conversation's intent, mask so that only user turns contribute to the loss, and fine-tune a base LLM.

Training is driven by a lightly customized fork of [`llama-cookbook`](https://github.com/meta-llama/llama-cookbook) (in [llama-cookbook/](llama-cookbook/)) that adds a `userlm_dataset` and a `<|endconversation|>` special token (token embedding initialized from the EOS token of base LLM vocabulary), plus FSDP + torch.compile implementation. For more details, please refer to the Appendix of our paper.

## Checkpoints

Trained checkpoints are omitted from this anonymous review release. Train locally with the pipeline below and supply your checkpoint directory to downstream scripts.

## Pipeline

| Step | Script | Input → Output |
|------|--------|----------------|
| 1. Preprocess | [1_load_and_preprocess_wildchat.py](1_load_and_preprocess_wildchat.py) | WildChat → `processed_data/{split:train,val,test}.jsonl` |
| 2. Generate intents | [2_generate_intents.py](2_generate_intents.py) | `processed_data/` → `data_with_intents/{split}_with_intents_{intent-generation-model}.jsonl` |
| 3. Flip & tokenize | [3_flip_dialogue_prepare_training.py](3_flip_dialogue_prepare_training.py) | `data_with_intents/` → `training_data/{split}_{base-model-tokenizer}_samples.jsonl` |
| 4. Prepare tokenizer | [4_prepare_tokenizer.py](4_prepare_tokenizer.py) | base tokenizer → `tokenizers_and_configs--{base-model}/` |
| 5. Train | [launch_multi_gpu.sh](launch_multi_gpu.sh) | `training_data/`, `tokenizers_and_configs--{base-model}/` → checkpoints |

## Installation

Use Python 3.12. The supplied `requirements.txt` pins PyTorch 2.10.0 with CUDA 12.8 and Accelerate 1.12.0. Review these constraints for your hardware; use a separate environment from RL training.

```bash
conda create -n usersim python=3.12 -y
conda activate usersim
pip install -e llama-cookbook
pip install -r requirements.txt
```

You will also need a Hugging Face account with access to the base model you fine-tune (e.g. `meta-llama/Meta-Llama-3-8B`): `huggingface-cli login`.

## Usage

```bash
# 1. Preprocess WildChat (streams allenai/WildChat-1M from the Hub)
python 1_load_and_preprocess_wildchat.py --output_dir ./processed_data

# 2. Generate intents using the paper's Qwen3-32B (thinking enabled).
# Run the server separately: vllm serve Qwen/Qwen3-32B --port 8000
python 2_generate_intents.py \
    --data_dir ./processed_data --output_dir ./data_with_intents \
    --model Qwen/Qwen3-32B --ports 8000

# 3. Flip dialogues into masked training samples
python 3_flip_dialogue_prepare_training.py \
    --data_dir ./data_with_intents \
    --output_dir ./training_data \
    --tokenizer Qwen/Qwen2.5-14B-Instruct \
    --intent_gen_model Qwen/Qwen3-32B

# 4. Add the <|endconversation|> token to the tokenizer
python 4_prepare_tokenizer.py \
    --base_tokenizer Qwen/Qwen2.5-14B-Instruct \
    --output_dir ./tokenizers_and_configs

# 5. Train (multi-GPU FSDP)
NUM_GPUS=8 MODEL_NAME=Qwen/Qwen2.5-14B-Instruct \
    DATA_PATH=./training_data OUTPUT_DIR=./userlm_checkpoints \
    ./launch_multi_gpu.sh
```

Step 5 is configured through environment variables (`NUM_GPUS`, `MODEL_NAME`, `TOTAL_BS`, `GRAD_ACC`, `DATA_PATH`, `OUTPUT_DIR`, ...); see the top of [launch_multi_gpu.sh](launch_multi_gpu.sh). Weights & Biases logging is off by default; opt in with `USE_WANDB=True`.

We also provide two conversion scripts (`convert_full_dict_to_safetensor.py` used when trained with world_size=1; `convert_sharded_dict_to_safetensor.py` used with FSDP training) for converting trained model weights into vLLM-compatible safetensor format.

## Supported base models

Chat templates for the masking step live in [tokenizers_and_configs/tokenizer_configs.py](tokenizers_and_configs/tokenizer_configs.py) and currently cover the Llama-3 and Qwen2.5 families.

For the larger-corpus variant, run `1_load_and_preprocess_wildchat_4p8m.py --help` and use its output directory as the input to steps 2–5.

## Appendix C settings

Preprocessing deduplicates conversation hashes, retains English conversations,
and removes conversations whose first user utterance contains a 7-gram occurring
in more than 100 retained conversations. Users are partitioned 89%/5%/6% using
`hashed_ip` alone, independent of location. The 4.8M script also excludes every
conversation hash in the 1M corpus. Benchmark-user holdout is optional and changes
the split; it is disabled by default. Split manifests are not bundled.

Intent generation sees the full conversation. Training scores user tokens and
turn endings only, including the appended `<|endconversation|>` user turn.
The new token's actual input/output embedding rows are initialized from EOS.
Complete conversations are packed to 16,384 tokens with masked PAD separators;
overlong samples are dropped, the final partial pack is retained, and batches
are left-padded to the context limit.

The launcher uses full-parameter Adam, two epochs, batch 64, microbatch 1 on eight
GPUs with accumulation 8, peak learning rate 2e-5, 10% linear warmup and cosine
decay to 2e-6. Validation runs every 100 optimizer updates and at epoch boundaries.
Checkpoint selection uses corpus token-weighted user-side cross-entropy; dummy
validation examples equalize distributed forward counts without scoring duplicate
tokens. The best saved checkpoint is recorded in `best_checkpoint.json`.

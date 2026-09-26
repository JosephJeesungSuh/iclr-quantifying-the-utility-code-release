#!/usr/bin/env python3
"""Launch the Appendix F configuration; --dry-run prints it without starting jobs."""
import argparse
import json
import os
from pathlib import Path
import shlex

SIMULATORS = {
    "sftuser": ("userlm", "", False),
    "rpuser1": ("userlm_baseline", "_baseline", True),
    "rpuser2": ("userlm_baseline", "_baseline", False),
    "rpuser3": ("userlm_baseline_mixture", "_baseline_mixture", False),
}


def build_overrides(args):
    env, suffix, first_turn = SIMULATORS[args.simulator]
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    user_model = args.user_model or "Qwen/Qwen2.5-14B-Instruct"
    if args.simulator == "sftuser" and not args.user_model:
        raise ValueError("--user-model must point to your trained SFTUSER checkpoint")
    judges = [
        {"model_name": model, "is_local": True, "local_port": port,
         "base_url": "http://localhost:{port}/v1", "temperature": 0.0,
         "enable_structured_output": True, "max_retries": 8}
        for model, port in [(args.judge_model, args.judge_port),
                            (args.second_judge_model, args.second_judge_port)]
    ]
    values = {
        "data.train_data": [str(data_dir / f"train{suffix}.parquet")],
        "data.val_data": [str(data_dir / f"sampled_val{suffix}.parquet")],
        "trainer.policy.model.path": args.agent_model,
        "trainer.strategy": "fsdp2",
        "trainer.placement.colocate_all": True,
        "trainer.placement.policy_num_gpus_per_node": args.num_gpus,
        "trainer.placement.critic_num_gpus_per_node": args.num_gpus,
        "trainer.placement.ref_num_gpus_per_node": args.num_gpus,
        "trainer.epochs": 1,
        "trainer.train_batch_size": 64,
        "trainer.policy_mini_batch_size": 16,
        "trainer.micro_forward_batch_size_per_gpu": 1,
        "trainer.micro_train_batch_size_per_gpu": 1,
        "trainer.update_epochs_per_batch": 1,
        "trainer.eval_batch_size": 1024,
        "trainer.eval_interval": 20,
        "trainer.eval_before_train": False,
        "trainer.ckpt_interval": 20,
        "trainer.max_prompt_length": 2048,
        "trainer.policy.optimizer_config.lr": 8e-7,
        "trainer.policy.optimizer_config.max_grad_norm": 1.0,
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.algorithm.use_kl_loss": True,
        "trainer.algorithm.kl_estimator_type": "k3",
        "trainer.algorithm.kl_loss_coef": 0.001,
        "trainer.algorithm.eps_clip_low": 0.2,
        "trainer.algorithm.eps_clip_high": 0.2,
        "trainer.logger": "console",
        "trainer.project_name": "user-simulator-rl",
        "trainer.run_name": args.simulator,
        "trainer.ckpt_path": str(output_dir / "checkpoints"),
        "generator.backend": "vllm",
        "generator.num_inference_engines": args.num_gpus,
        "generator.inference_engine_tensor_parallel_size": 1,
        "generator.n_samples_per_prompt": 5,
        "generator.eval_n_samples_per_prompt": 1,
        "generator.sampling_params.temperature": 1.0,
        "generator.sampling_params.max_generate_length": 2048,
        "generator.eval_sampling_params.temperature": 1.0,
        "generator.eval_sampling_params.max_generate_length": 2048,
        "generator.gpu_memory_utilization": 0.8,
        "generator.run_engines_locally": True,
        "generator.weight_sync_backend": "nccl",
        "generator.async_engine": True,
        "generator.batched": False,
        "generator.use_conversation_multi_turn": True,
        "generator.max_turns": 5,
        "generator.max_input_length": 16384,
        "environment.env_class": env,
        "environment.skyrl_gym.max_env_workers": 48,
        f"environment.skyrl_gym.{env}.max_turns": 5,
        f"environment.skyrl_gym.{env}.llm_judges": judges,
        f"environment.skyrl_gym.{env}.userlm.model_path": user_model,
        f"environment.skyrl_gym.{env}.userlm.port": args.user_port,
        f"environment.skyrl_gym.{env}.userlm.temperature": 0.7,
        f"environment.skyrl_gym.{env}.userlm.top_p": 0.9,
        f"environment.skyrl_gym.{env}.userlm.max_new_tokens": 1024,
        f"environment.skyrl_gym.{env}.userlm.generate_turn_one": first_turn,
    }
    if args.simulator == "sftuser":
        values[f"environment.skyrl_gym.{env}.userlm.tokenizer_path"] = args.user_tokenizer or user_model
        values[f"environment.skyrl_gym.{env}.userlm.base_model"] = args.user_base_model
        values[f"environment.skyrl_gym.{env}.userlm.max_dedup_retries"] = 5
    else:
        values[f"environment.skyrl_gym.{env}.userlm.enable_structured_output"] = True
    prefix = f"environment.skyrl_gym.{env}."
    environment = {"max_turns": 5, "llm_judge": {"enabled": True}, "userlm": {"enabled": True,
                   "base_url": "http://localhost:{port}/v1", "terminal_signal": "<|endconversation|>"}}
    for key in list(values):
        if key.startswith(prefix):
            parts = key[len(prefix):].split(".")
            if len(parts) == 1:
                environment[parts[0]] = values.pop(key)
            else:
                environment.setdefault(parts[0], {})[parts[1]] = values.pop(key)
    values[f"+environment.skyrl_gym.{env}"] = environment
    if getattr(args, "evaluate", False):
        values["+userlm_evaluate"] = True
        values["data.val_data"] = [str(data_dir / f"test{suffix}.parquet")]
        values["trainer.placement.colocate_all"] = False
    return values


def hydra_value(value):
    """Encode nested Hydra values, retaining quoted string values and unquoted keys."""
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{hydra_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ",".join(map(hydra_value, value)) + "]"
    return json.dumps(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skyrl-dir", default=os.getenv("SKYRL_DIR", "/path/to/SkyRL"))
    parser.add_argument("--simulator", choices=SIMULATORS, default="sftuser")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--user-model")
    parser.add_argument("--user-tokenizer")
    parser.add_argument("--user-base-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--user-port", type=int, default=8002)
    parser.add_argument("--judge-model", default="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    parser.add_argument("--judge-port", type=int, default=8003)
    parser.add_argument("--second-judge-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--second-judge-port", type=int, default=8004)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--evaluate", action="store_true", help="Cross-simulator evaluation on the held-out test split")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Additional Hydra key=value overrides, applied last")
    args = parser.parse_args()
    if args.num_gpus < 1:
        parser.error("--num-gpus must be positive")
    try:
        values = build_overrides(args)
    except ValueError as error:
        parser.error(str(error))
    command = ["uv", "run", "--extra", "vllm", "-m", "examples.userlm_paper.main"]
    command += [f"{key}={hydra_value(value)}" for key, value in values.items()]
    command += args.overrides
    if args.dry_run:
        print(shlex.join(command))
        return
    for key in (("data.val_data",) if args.evaluate else ("data.train_data", "data.val_data")):
        for filename in values[key]:
            if not Path(filename).is_file():
                parser.error(f"Missing dataset: {filename}; see rl_training/README.md")
    Path(args.output_dir).expanduser().mkdir(parents=True, exist_ok=True)
    checkout = Path(args.skyrl_dir).expanduser().resolve()
    if not (checkout / "skyrl-train/examples/userlm_paper/main.py").is_file():
        parser.error("Install the paper example first with rl_training/install.py")
    os.chdir(checkout / "skyrl-train")
    # Local OpenAI-compatible servers require a nonempty SDK token, not a private key.
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()

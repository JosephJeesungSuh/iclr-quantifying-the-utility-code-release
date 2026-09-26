import argparse
import sys
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "microsoft/UserLM-8b"


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    """
    Load the model and tokenizer from the local/remote Hugging Face cache.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).to(device)

    return model, tokenizer, device


def build_messages(task_intent: str) -> List[Dict[str, str]]:
    """
    Build a minimal conversation with a single system message describing
    the task intent for the user simulator.
    """
    return [
        {
            "role": "system",
            "content": task_intent,
        }
    ]


def generate_user_utternace(
    model,
    tokenizer,
    device: str,
    task_intent: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.8,
) -> str:
    """
    Run a single generation step from UserLM given a task intent.
    Mirrors the usage shown in the model card:
    https://huggingface.co/microsoft/UserLM-8b
    """
    messages = build_messages(task_intent)
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # End-of-turn and end-of-conversation tokens, following the model card example.
    end_token = "<|eot_id|>"
    end_token_id = tokenizer.encode(end_token, add_special_tokens=False)

    end_conv_token = "<|endconversation|>"
    end_conv_token_id = tokenizer.encode(end_conv_token, add_special_tokens=False)

    outputs = model.generate(
        input_ids=inputs,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        eos_token_id=end_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bad_words_ids=[[token_id] for token_id in end_conv_token_id],
    )

    response = tokenizer.decode(
        outputs[0][inputs.shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick inference script for microsoft/UserLM-8b."
    )
    parser.add_argument(
        "--intent",
        type=str,
        required=False,
        help=(
            "Task intent describing what the simulated user wants. "
            "If omitted, a default coding-related intent is used."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling p.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    task_intent = args.intent or (
        "You are a user who wants to implement a special type of sequence. "
        "The sequence sums up the two previous numbers in the sequence and adds 1 "
        "to the result. The first two numbers in the sequence are 1 and 1."
    )

    print(f"Loading {MODEL_NAME}...", file=sys.stderr)
    model, tokenizer, device = load_model_and_tokenizer()
    print(f"Model loaded on device: {device}", file=sys.stderr)

    print("Generating user utterance...", file=sys.stderr)
    response = generate_user_utternace(
        model=model,
        tokenizer=tokenizer,
        device=device,
        task_intent=task_intent,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("\n=== Generated user message ===")
    print(response)


if __name__ == "__main__":
    main(sys.argv[1:])


"""
Generate Simulated User Turns for Traditional Metrics Evaluation
================================================================
For every WildBench v2 conversation (including single-turn ones), simulate
user turns from position 2 through N+1 (one past the last GT user turn) using
both userLM (raw completion with the Qwen ``<|im_*|>`` format) and RP2
(prompted qwen2.5-14b-instruct via chat completions). Within a session we
generate sequentially and stop as soon as the model emits the terminal
``<|endconversation|>`` signal or we step past the GT length, whichever comes
first.

Usage:
    python generate_sim_turns.py \
        --userlm_model   userlm \
        --userlm_url     http://localhost:8000/v1 \
        --sysprompt_model sysprompt \
        --sysprompt_url  http://localhost:8001/v1 \
        --num_workers    8 \
        --limit          50   # omit for full dataset
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

OUTPUT_PATH = Path(
    "/path/to/wildbench_simulation.jsonl"
)

_TERMINAL_SIGNAL = "<|endconversation|>"

# ---------------------------------------------------------------------------
# RP2 (prompted qwen2.5-14b-instruct) prompt
# ---------------------------------------------------------------------------

RP2_SYSPROMPT_TEMPLATE = """You are role-playing as a human USER interacting with an AI collaborator to complete a specific task. Your goal is to generate realistic, natural responses that a user might give in this scenario.

## Input Information:
You will be provided with:
- Task Description: The type of task you are trying to accomplish.
- Complete Prompt or Reference Goal: This field may include the complete user request/query or a reference answer to user's request. Use this field to understand the user's intent, requirements, or what would count as a satisfactory outcome.
- Chat History: The ongoing conversation between you (as the user) and the AI

Inputs:
<|The Start of Task Description (Not visible to the AI collaborator)|>
{{task_desc}}
<|The End of Task Description|>

<|The Start of Complete Prompt or Reference Goal (Not visible to the AI collaborator)|>
{{single_turn_prompt}}
<|The End of Complete Prompt or Reference Goal|>

<|The Start of Chat History|>
{{chat_history}}
<|The End of Chat History|>

{guidelines}

## Output Format:
You should output a JSON object with three entries:
- "current_answer" (str): Briefly summerize the AI's current solution to the task.
- "thought" (str): Output your thought process as a user deciding what to say next. Consider:
    1. Have you obtained a satisfactory solution from the AI? If yes, you can terminate this chat.
    2. If not, what specific part of the problem or solution are you struggling with?
    3. Has the AI asked you to perform a task or answer a question? If so, how should you approach it?
    4. Are you noticing any patterns or potential misunderstandings that need clarification?
    5. If you're stuck, how can you phrase your question to get the most helpful response while demonstrating your current understanding?
- "response" (str): Based on your thought process, respond to the AI as the user you are role-playing. Stop immediately when the user's response is completed.

## Important Notes:
- Respond Based on Previous Messages: Your responses should be based on the context of the current chat history. Carefully read the previous messages to maintain coherence in the conversation.
- Conversation Flow: If "Current Chat History" is empty, start the conversation from scratch with an initial request. Otherwise, continue based on the existing conversation.
- Don't Copy Input Directly: Use the provided information for understanding context only. Avoid copying target queries or any provided information directly in your responses.
- Completion Signal: Use "{{terminal_signal}}" as your response when you believe your goal has been solved or if you determine the AI cannot help further.
- Double check if the JSON object is formatted correctly. Ensure that all fields are present and properly structured.

Remember to stay in character as a user throughout your response, and follow the instructions and guidelines carefully."""

GUIDELINES_DEFAULT = """## Guidelines:
- Stay in Character: Role-play as a human USER. You are NOT an AI. Maintain a consistent personality throughout the chat.
- Minimize Effort: IMPORTANT! As a user, avoid being too detailed in your responses. Provide vague or incomplete demands in the early stages of the conversation to minimize your effort. Let the AI ask for clarification rather than providing everything upfront.
- Knowledge Background: Reflect the user's knowledge level in the role-playing. If the user is less knowledgeable about a task, they might not notice incorrect statements. Ask questions that demonstrate your current understanding and areas of confusion.
- Occasionally Make Mistakes: Real-world users might misspell words, provide incorrect dates, give wrong information, or ask unclear questions. Simulate this behavior to reflect natural interactions.
- Mention Personal Preferences: Include preferences or constraints that might influence your requests or responses. For example, "I prefer short answers," "I need this done quickly," or "I like detailed comments in code."
- Goal-Oriented: Keep the chat focused on your intent. Avoid small talk or digressions. Redirect the chat back to the main objective if it starts to stray."""


# ---------------------------------------------------------------------------
# JSON extraction (claude-generated extract_outer_dict)
# ---------------------------------------------------------------------------

def _outer_braces_span(s: str) -> Optional[Tuple[int, int]]:
    """Return (start, end_exclusive) for the outermost {...}, handling quoted strings."""
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return (start, i + 1)
    return None


def extract_outer_dict(s: str) -> Dict:
    """Extract the outermost {...} from s and parse it into a dict."""
    s = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL).strip()
    span = _outer_braces_span(s)
    if span is None:
        raise ValueError(f"No balanced {{...}} found in: {s[:6400]!r}")
    obj_str = s[span[0]:span[1]]
    try:
        parsed = ast.literal_eval(obj_str)
    except Exception as e1:
        try:
            parsed = json.loads(obj_str, strict=False)
        except json.JSONDecodeError as e2:
            raise ValueError(
                f"Failed to parse as Python literal: {e1}\n"
                f"Failed to parse as JSON: {e2}\n"
                f"Object: {obj_str[:6400]!r}"
            )
    if not isinstance(parsed, dict):
        raise ValueError(f"Extracted object is not a dict but {type(parsed)}")
    return parsed


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_INTENT_PREFIX_RE = re.compile(r"^\s*the user wants to\s+", re.IGNORECASE)


def _clean_intent(intent: str) -> str:
    """Strip 'The user wants to ' prefix and trailing period so the system line reads cleanly."""
    out = _INTENT_PREFIX_RE.sub("", intent or "").strip()
    while out.endswith("."):
        out = out[:-1].rstrip()
    return out


def _build_userlm_prompt(intent: str, context_turns: List[Dict]) -> str:
    """Raw-text prompt in the exact format the userLM was trained on.

    Always ends with ``<|im_start|>user\n`` so the model continues with the
    next user turn.
    """
    lines = [
        f"<|im_start|>system\nYou are a user chatting with an assistant language model to {_clean_intent(intent)}.<|im_end|>"
    ]
    for turn in context_turns:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            lines.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            lines.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    lines.append("<|im_start|>user\n")
    return "\n".join(lines)


def _build_rp2_prompt(intent: str, context_turns: List[Dict]) -> str:
    history_lines: List[str] = []
    for turn in context_turns:
        role = turn.get("role", "")
        if role == "user":
            history_lines.append(f"USER: {turn['content']}")
        elif role == "assistant":
            history_lines.append(f"ASSISTANT: {turn['content']}")
    return (
        RP2_SYSPROMPT_TEMPLATE
        .replace("{{task_desc}}", "chatting with an AI assistant")
        .replace("{{single_turn_prompt}}", intent)
        .replace("{{chat_history}}", "\n".join(history_lines))
        .replace("{guidelines}", GUIDELINES_DEFAULT)
        .replace("{{terminal_signal}}", _TERMINAL_SIGNAL)
    )


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _call_userlm(
    client: httpx.Client, base_url: str, model: str, prompt: str
) -> Tuple[Optional[str], bool]:
    """Call userLM via /completions. Returns (response_or_None, is_terminal)."""
    for attempt in range(7):
        try:
            resp = client.post(
                f"{base_url}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["<|im_end|>", "<|endoftext|>"],
                    "truncate_prompt_tokens": 32256,
                    "skip_special_tokens": False,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"]
            if _TERMINAL_SIGNAL in text:
                return None, True
            # Strip residual special-token markers that may slip through.
            for tok in ("<|im_end|>", "<|endoftext|>", "<|im_start|>"):
                text = text.replace(tok, "")
            text = text.strip()
            if text:
                return text, False
            return None, False
        except Exception as e:
            if attempt < 6:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                logger.warning("userLM failed after 7 attempts: %s", e)
    return None, False


def _call_rp2(
    client: httpx.Client, base_url: str, model: str, prompt: str
) -> Tuple[Optional[str], bool]:
    """Call RP2 via /chat/completions; resample until JSON parses."""
    for attempt in range(7):
        try:
            resp = client.post(
                f"{base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "truncate_prompt_tokens": 32256,
                },
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            obj = extract_outer_dict(text)
            response = str(obj.get("response", "")).strip()
            if not response:
                raise ValueError("empty 'response' field in RP2 output")
            if _TERMINAL_SIGNAL in response:
                return None, True
            return response, False
        except Exception as e:
            if attempt < 6:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                logger.warning("RP2 failed after 7 attempts: %s", e)
    return None, False


# ---------------------------------------------------------------------------
# Per-session sequential simulation
# ---------------------------------------------------------------------------

def _simulate_session(
    task: Dict,
    client: httpx.Client,
    base_url: str,
    model: str,
    call_fn,
    build_prompt_fn,
) -> Dict[int, Dict]:
    """Generate sim user turns at positions 2..N+1; stop on terminal."""
    conv = task["conv"]
    user_indices = task["user_indices"]
    intent = task["intent"]
    n = len(user_indices)

    last_assistant = task["last_assistant"]
    out: Dict[int, Dict] = {}
    for k in range(2, n + 2):
        if k <= n:
            ctx_end = user_indices[k - 1]  # exclusive
            context = conv[:ctx_end]
            gold = conv[ctx_end]["content"]
        else:
            # End-check position: append the (otherwise missing) final assistant turn.
            context = conv + [{"role": "assistant", "content": last_assistant}]
            gold = None

        prompt = build_prompt_fn(intent, context)
        response, terminal = call_fn(client, base_url, model, prompt)
        out[k] = {"response": response, "terminal": terminal, "gold": gold}
        if terminal:
            break
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_wildbench(max_samples: Optional[int] = None, cache_dir: Optional[str] = None) -> List[Dict]:
    """Load WildBench v2; keep every conversation that has at least one user turn."""
    logger.info("Loading WildBench v2 ...")
    kwargs: Dict = {"name": "v2", "split": "test"}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset("allenai/WildBench", **kwargs)

    tasks = []
    for item in ds:
        conv = item["conversation_input"]
        user_indices = [i for i, t in enumerate(conv) if t["role"] == "user"]
        if not user_indices:
            continue
        # WildBench v2: conversation_input ends with a user turn for every row;
        # the corresponding assistant reply is in references["gpt-4"].
        refs = item.get("references") or {}
        last_assistant = refs.get("gpt-4") or refs.get("gpt-4-turbo") or next(iter(refs.values()), None)
        if last_assistant is None:
            continue
        tasks.append({
            "session_id": item["session_id"],
            "primary_tag": item.get("primary_tag"),
            "intent": item["intent"],
            "conv": conv,
            "user_indices": user_indices,
            "last_assistant": last_assistant,
        })

    if max_samples:
        tasks = tasks[:max_samples]
    total_positions = sum(len(t["user_indices"]) for t in tasks)  # positions 2..N+1 == N each
    logger.info("Loaded %d tasks (%d total sim positions).", len(tasks), total_positions)
    return tasks


def generate_userlm(
    tasks: List[Dict],
    model: str,
    base_url: str,
    num_workers: int = 8,
) -> Dict[str, Dict[int, Dict]]:
    client = httpx.Client(base_url=base_url, timeout=120.0)

    def _run(task: Dict):
        return task["session_id"], _simulate_session(
            task, client, base_url, model, _call_userlm, _build_userlm_prompt
        )

    results: Dict[str, Dict[int, Dict]] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_run, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="userLM"):
            sid, turns = fut.result()
            results[sid] = turns
    return results


def generate_sysprompt(
    tasks: List[Dict],
    model: str,
    base_url: str,
    num_workers: int = 8,
) -> Dict[str, Dict[int, Dict]]:
    client = httpx.Client(base_url=base_url, timeout=120.0)

    def _run(task: Dict):
        return task["session_id"], _simulate_session(
            task, client, base_url, model, _call_rp2, _build_rp2_prompt
        )

    results: Dict[str, Dict[int, Dict]] = {}
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_run, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="sysPrompt"):
            sid, turns = fut.result()
            results[sid] = turns
    return results


def save_results(
    tasks: List[Dict],
    userlm_results: Dict,
    sysprompt_results: Dict,
    output_path: Path = OUTPUT_PATH,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for task in tasks:
            sid = task["session_id"]
            conv = task["conv"]
            ui = task["user_indices"]
            n = len(ui)
            last_asst = task["last_assistant"]
            ul_turns = userlm_results.get(sid, {})
            sp_turns = sysprompt_results.get(sid, {})

            conversation: List[Dict] = []
            for i in range(1, n + 1):
                # User utterance i. For i==1 it's just GT; for i>=2 also attach sims.
                user_entry: Dict = {"role": "user", "gt": conv[ui[i - 1]]["content"]}
                if i >= 2:
                    ul = ul_turns.get(i, {})
                    sp = sp_turns.get(i, {})
                    user_entry.update({
                        "userlm": ul.get("response"),
                        "userlm_terminal": ul.get("terminal", False),
                        "sysprompt": sp.get("response"),
                        "sysprompt_terminal": sp.get("terminal", False),
                    })
                conversation.append(user_entry)

                # Assistant utterance i. For i<N pull from conversation_input;
                # for the final pair use references["gpt-4"] (see load_wildbench).
                if i < n:
                    asst_idx = ui[i - 1] + 1
                    asst_content = (
                        conv[asst_idx]["content"]
                        if asst_idx < len(conv) and conv[asst_idx]["role"] == "assistant"
                        else None
                    )
                else:
                    asst_content = last_asst
                conversation.append({"role": "assistant", "gt": asst_content})

            # End-check sim turn at position k = N+1 (no GT counterpart).
            end_k = n + 1
            ul_end = ul_turns.get(end_k, {})
            sp_end = sp_turns.get(end_k, {})
            if ul_end or sp_end:
                conversation.append({
                    "role": "user",
                    "gt": None,
                    "userlm": ul_end.get("response"),
                    "userlm_terminal": ul_end.get("terminal", False),
                    "sysprompt": sp_end.get("response"),
                    "sysprompt_terminal": sp_end.get("terminal", False),
                })

            f.write(json.dumps({
                "session_id": sid,
                "primary_tag": task["primary_tag"],
                "intent": task["intent"],
                "num_gold_user_turns": n,
                "conversation": conversation,
            }) + "\n")
    logger.info("Saved %d sessions to %s", len(tasks), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--userlm_model", required=True)
    parser.add_argument("--userlm_url", default="http://localhost:8001/v1")
    parser.add_argument("--sysprompt_model", required=True)
    parser.add_argument("--sysprompt_url", default="http://localhost:8002/v1")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N conversations (for testing)")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--output", default=None, help="Output .jsonl path (overrides default)")
    args = parser.parse_args()

    tasks = load_wildbench(max_samples=args.limit, cache_dir=args.cache_dir)
    userlm_results = generate_userlm(tasks, args.userlm_model, args.userlm_url, args.num_workers)
    sysprompt_results = generate_sysprompt(tasks, args.sysprompt_model, args.sysprompt_url, args.num_workers)
    save_results(tasks, userlm_results, sysprompt_results,
                 output_path=Path(args.output) if args.output else OUTPUT_PATH)


if __name__ == "__main__":
    main()

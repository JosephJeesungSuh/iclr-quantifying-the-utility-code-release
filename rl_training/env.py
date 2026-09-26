import time
import os
import httpx
import json
import random
import threading
import ast, re
from typing import Any, ClassVar, Dict, List, Optional, Union, Tuple

from omegaconf import DictConfig
from openai import OpenAI
from loguru import logger
from transformers import AutoTokenizer

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from .prompts import JUDGE_PROMPT, DEFAULT_JSON_SCHEMA_JUDGE
from .prompts import (
    USER_SIM_SYSPROMPT,
    ALL_USER_SIM_SYSPROMPTS,
    DEFAULT_JSON_SCHEMA_USERSIM,
)

logger.disable("httpx")
logger.disable("httpcore")

# A new env is constructed per trajectory, so HTTP clients must never be built per
# instance: each one carries its own connection pool that is only reclaimed by GC.
# Clients are cached per base_url (see `UserLMMultiTurnEnv._get_client`) and given an
# explicitly bounded pool -- the OpenAI SDK default is max_connections=1000, which lets
# live socket count grow far past what an `ssh -L` tunnel (one fd per forwarded
# connection, in a single sshd child) can carry.
#
# max_keepalive_connections == max_connections makes this a true fixed pool: sockets are
# reused instead of churned, so connection count stays flat under bursts. Size it at or
# above `skyrl_gym.max_env_workers`; the generator runs in a single process, so this is
# the global connection count per endpoint.
_HTTP_LIMITS = httpx.Limits(max_connections=48, max_keepalive_connections=48, keepalive_expiry=30.0)
# `pool` is how long a request waits for a free connection. It has to be generous relative
# to `read`, since a bounded pool means requests queue once all connections are in use.
_HTTP_TIMEOUT = httpx.Timeout(timeout=1800.0, connect=10.0, read=1800.0, write=1800.0, pool=600.0)
_CLIENT_LOCK = threading.Lock()


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
    """Extract the outermost {...} from s and parse it into a dict.

    Strips <think>...</think> blocks emitted by reasoning models before
    searching for the dict, so the thinking text doesn't confuse the parser.
    """
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


def _extract_judge_dict(s: str) -> Dict:
    """Parse an LLM-judge response, with regex fallback for the score field.

    Handles models (e.g. Qwen) that emit single-quoted string values with
    unescaped apostrophes inside (e.g. "thought": 'it's good'), which are
    neither valid JSON nor valid ast.literal_eval.
    """
    try:
        return extract_outer_dict(s)
    except ValueError:
        score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]*)?)', s)
        if score_match is None:
            raise
        logger.debug(f"Used regex fallback to extract score from: {s[:6400]!r}")
        return {"score": float(score_match.group(1)), "thought": None}
def _format_hist_conversation(conversation, last_role="assistant"):
    if not conversation or conversation[-1]["role"] != last_role:
        raise ValueError("Unexpected last conversation role")
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in conversation)


def to_userlm_input(messages: ConversationType, tokenizer: AutoTokenizer, base_model: Optional[str] = None) -> tuple[str, Dict]:
    """
    Convert messages to UserLM input format.
    1. Chat template applied to messages
    2. BOS token prepended
    3. Generation prompt appended based on model type
    Args:
        messages: List of conversation messages
        tokenizer: The tokenizer for the UserLM model
    Returns:
        Tuple of (formatted_input_string, tokenized_tensors)
    """

    def _genprompt() -> str:
        model_name = (base_model or tokenizer.name_or_path).lower()
        if "qwen2.5" in model_name:
            return "<|im_start|>user\n"
        else:
            raise NotImplementedError(f"Generation prompter not defined for {model_name}")

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    input_text = (
        (tokenizer.bos_token if tokenizer.bos_token is not None else "")
        + input_text + _genprompt()
    )
    tokenized = tokenizer(input_text, return_tensors="pt")
    return input_text, tokenized


def _get_stop_token(model_name: str) -> str:
    model_name = model_name.lower()
    if "qwen2.5" in model_name:
        return "<|im_end|>"
    else:
        raise NotImplementedError(f"Stop token not defined for {model_name}")


class UserLMMultiTurnEnv(BaseTextEnv):
    """
    Multi-turn environment using UserLM for user simulation.
    UserLM is a fine-tuned causal language model that generates user responses given conversation history.
    It can output <|endconversation|> to signal conversation termination.
    The environment receives rewards from an LLM-as-a-judge based on the entire conversation.
    Supports multiple LLM judges — when multiple judges are configured, the final reward is
    the unweighted average of all judge scores.
    """
    _tokenizer_cache: ClassVar[Dict[str, Any]] = {}
    _client_cache: ClassVar[Dict[str, OpenAI]] = {}

    @classmethod
    def _get_client(cls, base_url: Optional[str] = None) -> OpenAI:
        """Return the process-wide shared client for ``base_url``, creating it if needed.

        All clients live in ``UserLMMultiTurnEnv._client_cache`` (shared across subclasses)
        and use a bounded connection pool, so live TCP connections per endpoint stay at
        ``_HTTP_LIMITS.max_connections`` no matter how many envs are constructed.

        ``base_url=None`` means the OpenAI-default endpoint (remote judge).
        """
        cache_key = base_url if base_url is not None else "openai_default"
        with _CLIENT_LOCK:
            if cache_key not in UserLMMultiTurnEnv._client_cache:
                client_kwargs: Dict[str, Any] = {
                    "http_client": httpx.Client(limits=_HTTP_LIMITS, timeout=_HTTP_TIMEOUT),
                    "timeout": _HTTP_TIMEOUT,
                    # SDK-level retries absorb single connection blips before the caller's
                    # own retry loop (which is much slower) has to get involved.
                    "max_retries": 5,
                    "api_key": os.environ.get("OPENAI_API_KEY") or "EMPTY",
                }
                if base_url is not None:
                    client_kwargs["base_url"] = base_url
                UserLMMultiTurnEnv._client_cache[cache_key] = OpenAI(**client_kwargs)
            return UserLMMultiTurnEnv._client_cache[cache_key]

    @staticmethod
    def _init_judges(env_config: DictConfig) -> List[Dict]:
        """Initialize judge clients from config.
        Supports both a single ``llm_judge`` and a ``llm_judges`` list.
        When ``llm_judges`` is a non-empty list, each entry uses the same schema
        as ``llm_judge``.  Otherwise falls back to the single ``llm_judge`` entry.
        Returns a list of dicts, each with keys:
            client, model, temp, max_retries, enable_structured
        """
        assert env_config.llm_judge.enabled, "llm_judge must be enabled in env_config"

        judges: List[Dict] = []
        llm_judges_cfg = env_config.get("llm_judges", [])
        judge_cfgs = list(llm_judges_cfg) if llm_judges_cfg else [env_config.llm_judge]
        for cfg in judge_cfgs:
            judge: Dict[str, Any] = {}
            if cfg.get("is_local", False):
                base_url = cfg.get("base_url", "http://localhost:{port}/v1")
                judge_url = base_url.format(port=cfg.local_port)
            else:
                assert 'gpt' in cfg.model_name.lower(), (
                    "Currently only OpenAI API is considered for remote LLM judge."
                )
                judge_url = None
            judge["client"] = UserLMMultiTurnEnv._get_client(judge_url)
            judge["model"] = cfg.model_name
            judge["temp"] = cfg.temperature
            judge["max_retries"] = cfg.get("max_retries", 8)
            judge["enable_structured"] = cfg.get("enable_structured_output", False)
            judges.append(judge)

        return judges

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """
        env_config (DictConfig): comes from skyrl-train/skyrl_train/config/skyrl_gym_config/default.yaml
        Expected structure:
            userlm:
                enabled: true
                model_path: "/path/to/userlm/checkpoint"
                tokenizer_path: "/path/to/tokenizer"  # optional, defaults to model_path
                base_model: "/base/model/hf/name"  # base model for loading architecture
                base_url: "http://localhost:{port}/v1"
                port: 8002
                temperature: 0.7
                top_p: 0.9
                max_new_tokens: 1024
                terminal_signal: "<|endconversation|>"
                generate_turn_one: false
            llm_judge: # single judge (backward compatible)
                enabled: true
                model_name: "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
                base_url: "http://localhost:{port}/v1"
                is_local: true
                local_port: 8003
                temperature: 0.0
                enable_structured_output: false
                max_retries: 8
            llm_judges: [] # list of judges (same schema as llm_judge); overrides llm_judge
            max_turns: 5
        """
        super().__init__()

        self._extra_info = extras.get("extra_info", {})

        # Initialize LLM judge client(s)
        self._judges = self._init_judges(env_config)
        # Backward-compat aliases (point to the first judge)
        self._judge_client = self._judges[0]["client"]
        self._judge_model = self._judges[0]["model"]
        self._judge_temp = self._judges[0]["temp"]
        self._judge_max_retries = self._judges[0]["max_retries"]
        self._judge_enable_structured = self._judges[0]["enable_structured"]

        # Initialize UserLM
        userlm_cfg = env_config.get("userlm")
        self._userlm_enabled = userlm_cfg is not None and userlm_cfg.get("enabled", True)
        if self._userlm_enabled:
            self._userlm_max_retries = userlm_cfg.get("max_retries", self._judge_max_retries)
            self._userlm_temperature = userlm_cfg.get("temperature", 0.7)
            self._userlm_top_p = userlm_cfg.get("top_p", 0.9)
            self._userlm_max_new_tokens = userlm_cfg.get("max_new_tokens", 1024)
            self._terminal_signal = userlm_cfg.get("terminal_signal", "<|endconversation|>")
            self._stop_token = _get_stop_token(userlm_cfg.base_model)
            userlm_url = userlm_cfg.base_url.format(port=userlm_cfg.port)
            self._userlm_client = self._get_client(userlm_url)
            self._userlm_model_name = userlm_cfg.model_path
            self._base_model = userlm_cfg.base_model
            tokenizer_path = userlm_cfg.get("tokenizer_path", userlm_cfg.model_path)
            if tokenizer_path not in self.__class__._tokenizer_cache:
                self.__class__._tokenizer_cache[tokenizer_path] = AutoTokenizer.from_pretrained(tokenizer_path)
            self._tokenizer = self.__class__._tokenizer_cache[tokenizer_path]
            self._generate_turn_one = userlm_cfg.get("generate_turn_one", False)
            self._max_dedup_retries = userlm_cfg.get("max_dedup_retries", 7)
        else:
            self._userlm_client = None
            self._terminal_signal = userlm_cfg.get("terminal_signal", "<|endconversation|>") if userlm_cfg else "<|endconversation|>"
            self._generate_turn_one = False

        self._conversation: Optional[ConversationType] = [
            {"role": "system", "content": self._extra_info.get("intent", "")}
        ]
        self.max_turns = env_config.get("max_turns", self._extra_info.get("max_turns", -1))
        self.turns = 0

    def init(self, prompt: ConversationType):
        """
        Initialize the conversation with the initial prompt.
        Supports both single-turn input context (len==1)
        and multi-turn input context (prior user-assistant context + final user query).
        Run the first user turn to set the conversation history if _generate_turn_one is True.
        """
        assert len(prompt) >= 1 and prompt[-1]["role"] == "user", (
            "Prompt must be non-empty and end with a user message for UserLM environment."
        )
        self.turns = 0
        _meta = {}

        if self._generate_turn_one:
            assert len(prompt) == 1, (
                "When generate_turn_one is True, prompt should contain only the initial user query."
            )
            _intent = self._extra_info.get("intent", "")
            assert _intent, (
                "Intent must be provided in extra_info when generate_turn_one is True"
            )
            _turn_one_uttr, _meta = self._generate_user_response()
            self._conversation.append({"role": "user", "content": _turn_one_uttr})
            agent_prompt = [{"role": "user", "content": _turn_one_uttr}]
        elif len(prompt) > 1:
            # Multi-turn prompt: prior context turns + final user query
            for msg in prompt:
                self._conversation.append({"role": msg["role"], "content": msg["content"]})
            agent_prompt = list(prompt)
        else:
            self._conversation.append({"role": "user", "content": prompt[-1]["content"]})
            agent_prompt = [{"role": "user", "content": prompt[-1]["content"]}]

        return (
            agent_prompt,
            {"user_simulator": _meta} if _meta else {}
        ) # do not return self._conversation,
          # because in that case the agent model would be conditioned on the system prompt (intent)

    def _get_last_user_utterance(self) -> Optional[str]:
        """Return the content of the last user message in the conversation, or None."""
        for msg in reversed(self._conversation):
            if msg["role"] == "user":
                return msg["content"]
        return None

    def _generate_user_response(
        self,
        debug: bool = False,
        return_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Generate user response using UserLM.
        If the generated utterance is exactly identical to the previous user turn,
        retry up to ``max_dedup_retries`` times with a boosted temperature.
        """
        prev_user_uttr = self._get_last_user_utterance()
        dedup_attempt = 0
        for _attempt in range(1, self._userlm_max_retries + self._max_dedup_retries + 1):
            try:
                input_text, _ = to_userlm_input(self._conversation, self._tokenizer, self._base_model)
                if debug:
                    logger.info(f"UserLM input:\n{input_text}")

                # Boost temperature on dedup retries to encourage diversity
                temperature = self._userlm_temperature
                if dedup_attempt > 0:
                    temperature = min(self._userlm_temperature + 0.05 * dedup_attempt, 1.0)

                response = self._userlm_client.completions.create(
                    model=self._userlm_model_name,
                    prompt=input_text,
                    stop=[self._stop_token, self._terminal_signal],
                    top_p=self._userlm_top_p,
                    temperature=temperature,
                    max_tokens=self._userlm_max_new_tokens,
                    extra_body={ # extra body required to match custom tokenizer behavior of UserLM
                        "skip_special_tokens": False,
                        "include_stop_str_in_output": True,
                        "spaces_between_special_tokens": False,
                        "add_special_tokens": False,
                    },
                )
                user_reply = response.choices[0].text.strip().removesuffix(self._stop_token)
                if debug:
                    logger.info(f"UserLM output:\n{user_reply}")

                # Check for exact repetition of the previous user utterance
                if (prev_user_uttr is not None
                        and user_reply.strip() == prev_user_uttr.strip()
                        and dedup_attempt < self._max_dedup_retries):
                    dedup_attempt += 1
                    logger.warning(
                        f"UserLM generated duplicate utterance (dedup retry "
                        f"{dedup_attempt}/{self._max_dedup_retries}), re-generating with "
                        f"temperature {min(self._userlm_temperature + 0.05 * dedup_attempt, 1.0):.2f}."
                    )
                    continue

                if return_metadata:
                    return user_reply, {
                        "input": input_text,
                        "raw_response": response.choices[0].text,
                        "model_name": self._userlm_model_name,
                        "temperature": temperature,
                        "top_p": self._userlm_top_p,
                        "max_new_tokens": self._userlm_max_new_tokens,
                        "dedup_retries": dedup_attempt,
                    }
                return user_reply, {}

            except Exception as e:
                logger.exception(f"Attempt {_attempt}/{self._userlm_max_retries}: "
                                 f"UserLM failed with exception {e}.")
                if _attempt < self._userlm_max_retries:
                    time.sleep(1.3 ** (_attempt - 1))
                else:
                    logger.error(f"UserLM failed after {_attempt} attempts. Returning terminal signal.")
                    return self._terminal_signal, {"error": str(e)}

    def _get_reward_from_judge(
        self,
        judge: Dict,
        message: str,
        debug: bool = False,
        return_metadata: bool = False,
    ) -> Union[float, Dict]:
        """
        Get reward from a single LLM judge.
        """
        judge_client = judge["client"]
        judge_model = judge["model"]
        judge_temp = judge["temp"]
        judge_max_retries = judge["max_retries"]
        judge_enable_structured = judge["enable_structured"]
        if judge_model.lower().startswith("qwen/qwen3"):
            _special_configs = {
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            }
        else:
            _special_configs = {}

        for attempt in range(1, judge_max_retries + 1):
            response: Optional[str] = None
            try:
                if judge_enable_structured:
                    completion = judge_client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": message}],
                        temperature=judge_temp,
                        response_format=DEFAULT_JSON_SCHEMA_JUDGE,
                        **_special_configs,
                    )
                    response = completion.choices[0].message.content.strip()
                    parsed_dict = json.loads(response)
                    score = float(parsed_dict.get("score")) / 10.0
                    if return_metadata:
                        return {
                            "raw_response": response, "reward": score,
                            "model": judge_model, "temperature": judge_temp,
                        }
                    return score
                else:
                    completion = judge_client.chat.completions.create(
                        model=judge_model,
                        messages=[{"role": "user", "content": message}],
                        temperature=judge_temp,
                        **_special_configs,
                    )
                    response = completion.choices[0].message.content.strip()
                    parsed_dict = _extract_judge_dict(response)
                    score = float(parsed_dict.get("score")) / 10.0
                    if return_metadata:
                        return {
                            "raw_response": response, "reward": score,
                            "model": judge_model, "temperature": judge_temp,
                        }
                    return score
            except Exception:
                if judge_enable_structured:
                    # Fallback when structured output not supported
                    logger.warning("json-schema response format not supported, fallback.")
                    try:
                        completion = judge_client.chat.completions.create(
                            model=judge_model,
                            messages=[{"role": "user", "content": message}],
                            temperature=judge_temp,
                            **_special_configs,
                        )
                        response = completion.choices[0].message.content.strip()
                        parsed_dict = _extract_judge_dict(response)
                        score = float(parsed_dict.get("score")) / 10.0
                        if return_metadata:
                            return {
                                "raw_response": response, "reward": score,
                                "model": judge_model, "temperature": judge_temp,
                            }
                        return score
                    except Exception:
                        pass
                if response is None:
                    logger.exception(f"Attempt {attempt}/{judge_max_retries}: "
                                     f"LLM judge ({judge_model}) failed.")
                else:
                    logger.exception(f"Attempt {attempt}/{judge_max_retries}: "
                                     f"extracting from LLM judge ({judge_model}) response failed.")
                if attempt < judge_max_retries:
                    time.sleep(1.5 ** (attempt - 1))
                else:
                    logger.error(f"LLM judge ({judge_model}) failed after {judge_max_retries} attempts. "
                                  "Returning reward 0.0.")
                    return 0.0

        logger.error(f"LLM judge ({judge_model}) did not return a valid score. Returning reward 0.0.")
        return 0.0

    def _get_reward(
        self,
        conversation: Optional[ConversationType] = None,
        debug: bool = False,
        return_metadata: bool = False,
    ) -> Union[float, Dict]:
        """
        Get reward from LLM judge(s) based on the conversation and agent's action.
        When multiple judges are configured, the reward is the unweighted average of all judge scores.
        """
        conv = conversation if conversation is not None else self._conversation
        conv = conv[1:] if conv and conv[0]["role"] == "system" else conv
        assert conv.__len__() % 2 == 0, "Without system prompt, length should be even"
        message = JUDGE_PROMPT.format(
            question=self._extra_info.get("intent", ""),
            chat_history=_format_hist_conversation(conv, last_role="assistant")
        )
        if debug:
            logger.info(f"Messages to LLM judge:\n{message}")

        scores = []
        all_metadata = []
        for judge in self._judges:
            result = self._get_reward_from_judge(judge, message, debug, return_metadata)
            if isinstance(result, dict):
                scores.append(result["reward"])
                all_metadata.append(result)
            else:
                scores.append(result)

        avg_score = sum(scores) / len(scores)
        if return_metadata:
            return {
                "reward": avg_score,
                "judges": all_metadata,
                "prompt": message,
            }
        return avg_score

    def step(
        self,
        action: str,
        debug: bool = False,
        return_judge_metadata: bool = False,
        return_user_simulator_metadata: bool = False
    ) -> BaseTextEnvStepOutput:
        """
        Execute one step in the environment.
        Args:
            action: The agent's response
            debug: Whether to log debug information
            return_judge_metadata: Whether to return judge metadata
            return_user_simulator_metadata: Whether to return UserLM metadata
        Returns:
            BaseTextEnvStepOutput containing observations, reward, done flag, and metadata
        """
        assert self._conversation is not None, "Environment not initialized with a prompt."

        if debug:
            logger.info(f"Conversation so far at turn {self.turns + 1}:\n{self._conversation}")
            logger.info(f"Agent action at turn {self.turns + 1}:\n{action}")

        self._conversation.append({"role": "assistant", "content": action})
        self.turns += 1
        done = self.turns >= self.max_turns
        reward = 0.0
        observations: ConversationType = []
        userlm_metadata, judge_metadata = {}, {}

        if not done:
            _user_uttr, userlm_metadata = self._generate_user_response(
                debug=debug,
                return_metadata=return_user_simulator_metadata,
            )
            done = self._terminal_signal in _user_uttr.strip()
            if not done:
                user_message = {"role": "user", "content": _user_uttr}
                self._conversation.append(user_message)
                observations = [user_message]

        if done:
            reward_result = self._get_reward(
                conversation=self._conversation,
                debug=debug,
                return_metadata=return_judge_metadata
            )
            if isinstance(reward_result, dict):
                reward = reward_result["reward"]
                judge_metadata = reward_result
            else:
                reward = reward_result

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={"judge": judge_metadata, "user_simulator": userlm_metadata},
        )

    def close(self) -> None:
        # Clients are shared process-wide via `_get_client` (this holds for subclasses too);
        # closing one here would tear down the pool for every other live env.
        pass


class UserLMBaselineEnv(UserLMMultiTurnEnv):
    """
    Baseline environment that uses a simple prompt-based user simulator instead of UserLM.
    Uses a generic instruction-tuned model with USER_SIM_SYSPROMPT instead of a fine-tuned UserLM.
    """
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """
        """
        # Initialize via parent __init__ but we'll override UserLM client behavior
        # We only need the judge client from parent, so we call BaseTextEnv.__init__ instead
        BaseTextEnv.__init__(self)

        self._extra_info = extras.get("extra_info", {})

        # Initialize LLM judge client(s) — supports multiple judges
        self._judges = self._init_judges(env_config)
        self._judge_max_retries = self._judges[0]["max_retries"]

        # Initialize baseline user simulator (using generic instruction-tuned model)
        userlm_cfg = env_config.get("userlm")
        self._userlm_enabled = userlm_cfg is not None and userlm_cfg.get("enabled", True)
        if self._userlm_enabled:
            self._userlm_max_retries = userlm_cfg.get("max_retries", self._judge_max_retries)
            self._userlm_temperature = userlm_cfg.get("temperature", 0.7)
            self._userlm_top_p = userlm_cfg.get("top_p", 0.9)
            self._userlm_max_new_tokens = userlm_cfg.get("max_new_tokens", 1024)
            self._terminal_signal = userlm_cfg.get("terminal_signal", "<|endconversation|>")
            self._userlm_enable_structured = userlm_cfg.get("enable_structured_output", False)
            self._userlm_client = self._get_client(userlm_cfg.base_url.format(port=userlm_cfg.port))
            self._userlm_model_name = userlm_cfg.model_path
            self._generate_turn_one = userlm_cfg.get("generate_turn_one", False)
        else:
            self._userlm_client = None
            self._terminal_signal = userlm_cfg.get("terminal_signal", "<|endconversation|>") if userlm_cfg else "<|endconversation|>"
            self._generate_turn_one = False

        # Set task description for baseline user simulator
        self._task_desc = "chatting with an AI assistant"

        # System prompt used for the user simulator (may be overridden per episode by subclasses)
        self._user_sim_sysprompt = USER_SIM_SYSPROMPT

        self._conversation: Optional[ConversationType] = []
        self.max_turns = env_config.get("max_turns", self._extra_info.get("max_turns", -1))
        self.turns = 0

    def _generate_user_response(
        self,
        debug: bool = False,
        return_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Generate user response using a baseline prompt-based user simulator.
        Uses USER_SIM_SYSPROMPT with a generic instruction-tuned model.
        """
        extra_kwargs = {}
        if "qwen3" in self._userlm_model_name.lower():
            extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        for _attempt in range(1, self._userlm_max_retries + 1):
            try:
                # Get conversation history without system message
                conv = self._conversation[1:] if self._conversation and self._conversation[0]["role"] == "system" else self._conversation
                chat_history = _format_hist_conversation(conv, last_role="assistant") if conv else ""

                # Format the user simulator prompt
                prompt_message = self._user_sim_sysprompt.format(
                    task_desc=self._task_desc,
                    single_turn_prompt=self._extra_info.get("intent", ""),
                    chat_history=chat_history,
                    terminal_signal=self._terminal_signal,
                )

                if debug:
                    logger.info(f"Baseline user simulator input:\n{prompt_message}")

                # Call chat completion API
                if self._userlm_enable_structured:
                    response = self._userlm_client.chat.completions.create(
                        model=self._userlm_model_name,
                        messages=[{"role": "user", "content": prompt_message}],
                        temperature=self._userlm_temperature,
                        top_p=self._userlm_top_p,
                        max_tokens=self._userlm_max_new_tokens,
                        response_format=DEFAULT_JSON_SCHEMA_USERSIM,
                        **extra_kwargs,
                    )
                    response_text = response.choices[0].message.content.strip()
                    parsed_dict = json.loads(response_text)
                    user_reply = parsed_dict.get("response", "")
                else:
                    response = self._userlm_client.chat.completions.create(
                        model=self._userlm_model_name,
                        messages=[{"role": "user", "content": prompt_message}],
                        temperature=self._userlm_temperature,
                        top_p=self._userlm_top_p,
                        max_tokens=self._userlm_max_new_tokens,
                        **extra_kwargs,
                    )
                    response_text = response.choices[0].message.content.strip()
                    parsed_dict = extract_outer_dict(response_text)
                    user_reply = parsed_dict.get("response", "")

                if debug:
                    logger.info(f"Baseline user simulator output:\n{user_reply}")

                if return_metadata:
                    return user_reply, {
                        "input": prompt_message,
                        "raw_response": response_text,
                        "parsed_response": parsed_dict,
                        "model_name": self._userlm_model_name,
                        "temperature": self._userlm_temperature,
                        "top_p": self._userlm_top_p,
                        "max_new_tokens": self._userlm_max_new_tokens,
                    }
                return user_reply, {}

            except Exception as e:
                logger.exception(f"Attempt {_attempt}/{self._userlm_max_retries}: "
                                 f"Baseline user simulator failed with exception {e}.")
                if _attempt < self._userlm_max_retries:
                    time.sleep(1.5 ** (_attempt - 1))
                else:
                    logger.error(f"Baseline user simulator failed after {_attempt} attempts. "
                               "Returning terminal signal.")
                    return self._terminal_signal, {"error": str(e)}

    # _get_reward and step methods are inherited from UserLMMultiTurnEnv
    # init method is also inherited from UserLMMultiTurnEnv


class UserLMBaselineMixtureEnv(UserLMBaselineEnv):
    """
    Mixture baseline environment that randomly selects a user simulator system prompt
    from ALL_USER_SIM_SYSPROMPTS at the start of each episode, keeping it fixed for
    the entire episode to maintain a consistent simulated user persona.

    This is identical to UserLMBaselineEnv except that instead of always using USER_SIM_SYSPROMPT,
    it samples uniformly from all available system prompts (default, impatient, incomplete utterances, and auxiliary requests) on each call to init().
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)
        # _user_sim_sysprompt and _user_sim_sysprompt_name will be set per episode in init()
        self._user_sim_sysprompt_name: str = "default"

    def init(self, prompt: ConversationType):
        """
        Initialize the episode and sample a user simulator system prompt for this episode.
        The sampled prompt is held fixed for all turns of the episode.
        """
        _random_seed = self._extra_info.get("persona_seed")
        assert _random_seed is not None, "persona_seed should have been provided in extra_info"
        self._user_sim_sysprompt_name, self._user_sim_sysprompt = random.Random(_random_seed).choice(
            list(ALL_USER_SIM_SYSPROMPTS.items())
        )
        return super().init(prompt)

    def _generate_user_response(
        self,
        debug: bool = False,
        return_metadata: bool = True,
    ) -> Tuple[str, Dict]:
        """
        Generate user response using the episode-sampled system prompt.
        Adds the persona name to the returned metadata.
        """
        user_reply, metadata = super()._generate_user_response(
            debug=debug,
            return_metadata=return_metadata,
        )
        if return_metadata and metadata:
            metadata["user_sim_sysprompt_name"] = self._user_sim_sysprompt_name
        return user_reply, metadata
"""Build decision-point trajectories for WebShop or Paper Search."""

import argparse
import html
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from opportunity_forecasting import REPO_ROOT

os.environ.setdefault(
    "WEBSHOP_DATA_DIR",
    str(REPO_ROOT / "third_party" / "WebShop" / "data"),
)

from opportunity_forecasting.data.environments import ENV_DOMAIN_CHOICES, build_env_from_args
from opportunity_forecasting.models.search_runtime import (
    SELECTION_PROMPT_QWEN,
    _categorise_action,
    get_valid_actions_from_env,
    observation_signature,
    parse_qwen_answer_only,
    select_action_qwen,
    update_seen_products,
)
from opportunity_forecasting.data.webshop_action_prompt import WEBSHOP_ACTION_PROMPT
from opportunity_forecasting.data.webshop_rewards import is_product_page_observation


_CURRENT_PAPER_ID_RE = re.compile(r"CURRENT_PAPER_ID:\s*([^\s\]\|]+)")
_PAPER_RESULT_ID_RE = re.compile(r"PAPER_ID:\s*([^\s\]\|]+)")


def _apply_goal_overrides(env: Any, path: Optional[str]) -> None:
    if not path:
        return
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    overrides = payload.get("goals", payload)
    goals = getattr(env, "goals", None)
    if goals is None:
        server = getattr(env, "server", None)
        if server is None:
            browser = getattr(env, "browser", None)
            server = getattr(browser, "server", None)
        goals = getattr(server, "goals", None)
    if goals is None:
        raise AttributeError("WebShop environment does not expose goals")
    for raw_goal_id, values in overrides.items():
        goal_id = int(raw_goal_id)
        if goal_id < 0 or goal_id >= len(goals):
            raise IndexError(f"Goal override is outside the environment: {goal_id}")
        if not isinstance(values, dict):
            raise TypeError(f"Goal override must be an object: {goal_id}")
        goal = goals[goal_id]
        goal["instruction_text"] = str(values["instruction_text"])
        goal["price_upper"] = float(values["price_upper"])


def _paper_id_from_observation(obs: Optional[str]) -> Optional[str]:
    matches = _CURRENT_PAPER_ID_RE.findall(str(obs or ""))
    return str(matches[-1]).strip() if matches else None


def _paper_result_ids_from_observation(obs: Optional[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for pid in _PAPER_RESULT_ID_RE.findall(str(obs or "")):
        paper_id = str(pid).strip()
        if paper_id and paper_id not in seen:
            seen.add(paper_id)
            out.append(paper_id)
    return out


def _checkpoint_metadata_for_env(
    *,
    env: Any,
    obs: Optional[str],
    seen_products: Dict[str, dict],
    opened_item_ids: Set[str],
    visible_item_ids: Optional[List[str]] = None,
    current_item_id: Optional[str] = None,
) -> Dict[str, Any]:
    if getattr(env, "ccs_domain", "") != "paper_search":
        return {}
    visible_ids = visible_item_ids if visible_item_ids is not None else _paper_result_ids_from_observation(obs)
    return {
        "current_paper_id": str(current_item_id or ""),
        "visible_result_ids": list(visible_ids),
        "num_visible_results": int(len(visible_ids)),
        "opened_paper_ids": sorted(str(pid) for pid in opened_item_ids),
        "num_opened_papers": int(len(opened_item_ids)),
        "has_opened_paper": bool(opened_item_ids),
        "num_seen_papers": int(len(seen_products or {})),
    }


def _visited_page_metadata(visited: bool) -> Dict[str, Any]:
    return {"visited_product_page": bool(visited)}


def _read_goal_ids(path: Path) -> List[int]:
    """
    Load goal ids from:
    - JSON list, or JSON object containing 'goal_ids'
    - plain text file with one int per line
    """
    raw = path.read_text().strip()
    if not raw:
        return []
    goals: List[int] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            goals = [int(x) for x in obj]
        elif isinstance(obj, dict) and isinstance(obj.get("goal_ids"), list):
            goals = [int(x) for x in obj["goal_ids"]]
    except Exception:
        goals = []
    if not goals:
        tmp: List[int] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            tmp.append(int(line))
        goals = tmp

    seen: Set[int] = set()
    deduped: List[int] = []
    for g in goals:
        if g in seen:
            continue
        seen.add(g)
        deduped.append(g)
    return deduped


def _existing_goal_ids_in_output(path: Path) -> Set[int]:
    done: Set[int] = set()
    if not path.exists():
        return done
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if (
                isinstance(obj, dict)
                and obj.get("trigger") == "final"
                and "goal_idx" in obj
            ):
                try:
                    done.add(int(obj["goal_idx"]))
                except Exception:
                    pass
    return done


def load_transformers_model(path: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="auto",
    )
    return model, tokenizer


def load_vllm_model(
    path: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    quantization: Optional[str],
    dtype: str,
    enforce_eager: bool,
) -> Any:
    try:
        from vllm import LLM
    except Exception as e:
        raise RuntimeError(
            "vLLM is required for --engine vllm. Install with `pip install vllm`."
        ) from e

    kwargs: Dict[str, Any] = dict(
        model=path,
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        trust_remote_code=True,
        dtype=str(dtype),
        enforce_eager=bool(enforce_eager),
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)
    if quantization:
        kwargs["quantization"] = str(quantization)
    attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND_ARG")
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    return LLM(**kwargs)


def _tokenizer_encode(tokenizer: Any, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return list(input_ids)


def _tokenizer_decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


def _shorten_action_for_prompt(action: str, max_chars: Optional[int]) -> str:
    if max_chars is None or len(action) <= max_chars:
        return action
    if max_chars <= 12:
        return action[:max_chars]
    if "[" in action and action.endswith("]"):
        prefix, inner = action.split("[", 1)
        inner = inner[:-1]
        inner_budget = max_chars - len(prefix) - len("[]") - 3
        if inner_budget > 0:
            return f"{prefix}[{inner[:inner_budget]}...]"
    return action[: max_chars - 3] + "..."


def _build_action_prompt(
    goal: str,
    obs: str,
    valid_actions: Sequence[str],
    action_history: Optional[Sequence[str]],
    hist_k: int = 60,
    obs_char_limit: int = 1200,
    action_display_limit: Optional[int] = None,
    allow_numeric_answer: bool = False,
    domain: str = "paper_search",
) -> str:
    obs_truncated = (obs or "")[:obs_char_limit]

    search_actions: List[str] = []
    nav_actions: List[str] = []
    terminal_actions: List[str] = []
    for act in valid_actions:
        cat = _categorise_action(act)
        if cat == "search":
            search_actions.append(act)
        elif cat == "terminal":
            terminal_actions.append(act)
        else:
            nav_actions.append(act)

    lines: List[str] = []
    idx = 1
    if search_actions:
        lines.append("Search actions:")
        for act in search_actions:
            lines.append(f"  {idx}. {_shorten_action_for_prompt(act, action_display_limit)}")
            idx += 1
    if nav_actions:
        lines.append("Navigation and inspection actions:")
        for act in nav_actions:
            lines.append(f"  {idx}. {_shorten_action_for_prompt(act, action_display_limit)}")
            idx += 1
    if terminal_actions:
        if domain == "webshop":
            lines.append("Terminal actions (commit to a purchase or stop):")
        else:
            lines.append("Terminal actions (commit to the current best item or stop):")
        for act in terminal_actions:
            lines.append(f"  {idx}. {_shorten_action_for_prompt(act, action_display_limit)}")
            idx += 1
    actions_formatted = "\n".join(lines) if lines else "None"

    if action_history:
        recent = list(action_history)[-hist_k:]
        action_history_str = "\n".join([f"- {a}" for a in recent]) if recent else "None"
    else:
        action_history_str = "None"

    if hist_k == 60 and obs_char_limit == 1200 and action_display_limit is None and not allow_numeric_answer:
        template = (
            WEBSHOP_ACTION_PROMPT
            if domain == "webshop"
            else SELECTION_PROMPT_QWEN
        )
        return template.format(
            goal=goal,
            action_history=action_history_str,
            observation=obs_truncated,
            actions=actions_formatted,
            hist_k=hist_k,
        )

    answer_spec = "ACTION_NUMBER_OR_ONE_ACTION_FROM_LIST" if allow_numeric_answer else "ONE_ACTION_FROM_LIST"
    action_instruction = (
        "Available Actions (reply with the action number or copy one action exactly as written in the list below)."
        if allow_numeric_answer
        else "Available Actions (copy one EXACTLY as written)."
    )
    role = "shopping assistant" if domain == "webshop" else "search assistant"
    goal_label = "Shopping Goal" if domain == "webshop" else "Search Goal"
    search_help = (
        "find or refine the right category of products"
        if domain == "webshop"
        else "explore or refine the query"
    )
    navigation_help = (
        "scroll, move between pages, or inspect products"
        if domain == "webshop"
        else "move between pages or inspect candidate items"
    )
    return (
        f"You are a {role}. Select the single best action from the list to progress toward the goal.\n\n"
        f"{goal_label}: {goal}\n\n"
        f"Recent Action History (last {hist_k}):\n"
        f"{action_history_str}\n\n"
        f"Current Page (first {obs_char_limit} chars):\n"
        f"{obs_truncated}\n\n"
        f"{action_instruction}\n\n"
        f"- Search actions help you {search_help}.\n"
        f"- Navigation actions help you {navigation_help}.\n\n"
        "Available Actions:\n"
        f"{actions_formatted}\n\n"
        "STRICT OUTPUT FORMAT (no extra text):\n"
        "<think>brief reasoning</think>\n"
        f"<answer>{answer_spec}</answer>\n"
    )


def _emergency_trim_prompt(
    prompt: str,
    tokenizer: Any,
    *,
    max_prompt_tokens: int,
) -> Tuple[str, int]:
    token_ids = _tokenizer_encode(tokenizer, prompt)
    if len(token_ids) <= max_prompt_tokens:
        return prompt, len(token_ids)

    marker_ids = _tokenizer_encode(
        tokenizer,
        "\n\n[Earlier context omitted to fit the model context window.]\n\n",
    )
    head_tokens = min(192, max_prompt_tokens // 4)
    tail_tokens = max_prompt_tokens - head_tokens - len(marker_ids)
    if tail_tokens < 0:
        head_tokens = max(0, max_prompt_tokens - len(marker_ids))
        tail_tokens = 0

    trimmed_ids = token_ids[:head_tokens] + marker_ids
    if tail_tokens > 0:
        trimmed_ids.extend(token_ids[-tail_tokens:])

    trimmed_prompt = _tokenizer_decode(tokenizer, trimmed_ids)
    trimmed_len = len(_tokenizer_encode(tokenizer, trimmed_prompt))
    if trimmed_len > max_prompt_tokens:
        trimmed_prompt = _tokenizer_decode(tokenizer, token_ids[-max_prompt_tokens:])
        trimmed_len = len(_tokenizer_encode(tokenizer, trimmed_prompt))
    return trimmed_prompt, trimmed_len


def _build_vllm_prompt(
    goal: str,
    obs: str,
    valid_actions: Sequence[str],
    action_history: Optional[Sequence[str]],
    *,
    tokenizer: Optional[Any],
    max_model_len: Optional[int],
    max_new_tokens: int,
    domain: str,
) -> str:
    prompt = _build_action_prompt(
        goal,
        obs,
        valid_actions,
        action_history=action_history,
        domain=domain,
    )
    if tokenizer is None or max_model_len is None:
        return prompt

    max_prompt_tokens = max(256, int(max_model_len) - int(max_new_tokens) - 32)
    original_prompt_len = len(_tokenizer_encode(tokenizer, prompt))
    if original_prompt_len <= max_prompt_tokens:
        return prompt

    compact_settings = [
        dict(hist_k=30, obs_char_limit=900, action_display_limit=None, allow_numeric_answer=False),
        dict(hist_k=15, obs_char_limit=700, action_display_limit=None, allow_numeric_answer=False),
        dict(hist_k=10, obs_char_limit=500, action_display_limit=160, allow_numeric_answer=True),
        dict(hist_k=5, obs_char_limit=350, action_display_limit=120, allow_numeric_answer=True),
        dict(hist_k=0, obs_char_limit=250, action_display_limit=96, allow_numeric_answer=True),
    ]

    for settings in compact_settings:
        candidate = _build_action_prompt(
            goal,
            obs,
            valid_actions,
            action_history=action_history,
            domain=domain,
            **settings,
        )
        candidate_len = len(_tokenizer_encode(tokenizer, candidate))
        if candidate_len <= max_prompt_tokens:
            print(
                f"Prompt trimmed from {original_prompt_len} to {candidate_len} tokens "
                f"for context-window safety.",
                flush=True,
            )
            return candidate
        prompt = candidate

    trimmed_prompt, trimmed_len = _emergency_trim_prompt(
        prompt,
        tokenizer,
        max_prompt_tokens=max_prompt_tokens,
    )
    print(
        f"Prompt emergency-trimmed from {original_prompt_len} to {trimmed_len} tokens "
        f"for context-window safety.",
        flush=True,
    )
    return trimmed_prompt


def _resolve_action_to_valid(action_text: str, valid_actions: Sequence[str]) -> str:
    if not valid_actions:
        return "stop"

    action_normalized = html.unescape((action_text or "").strip())
    action_lower = action_normalized.lower()

    for valid_act in valid_actions:
        if valid_act.lower() == action_lower:
            return valid_act

    action_no_prefix = re.sub(r"^\d+\.\s*", "", action_normalized).strip()
    for valid_act in valid_actions:
        if valid_act.lower() == action_no_prefix.lower():
            return valid_act

    if (
        action_normalized
        and not action_normalized.lower().startswith("click[")
        and not action_normalized.lower().startswith("search[")
    ):
        wrapped = f"click[{action_normalized}]"
        for valid_act in valid_actions:
            if valid_act.lower() == wrapped.lower():
                return valid_act

    asin_match = re.search(r"b[0-9a-z]{9}", action_lower)
    if asin_match:
        model_asin = asin_match.group(0)
        for valid_act in valid_actions:
            if model_asin in valid_act.lower():
                return valid_act

    if len(action_normalized) > 3:
        for valid_act in valid_actions:
            valid_lower = valid_act.lower()
            if action_lower in valid_lower or valid_lower in action_lower:
                return valid_act

    try:
        idx = int(action_normalized) - 1
        if 0 <= idx < len(valid_actions):
            return list(valid_actions)[idx]
    except Exception:
        pass

    return list(valid_actions)[0]


def select_action_qwen_vllm(
    goal: str,
    obs: str,
    valid_actions: Sequence[str],
    llm: Any,
    *,
    action_history: Optional[Sequence[str]],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    tokenizer: Optional[Any],
    max_model_len: Optional[int],
    domain: str,
) -> Tuple[str, str, str, str]:
    try:
        from vllm import SamplingParams
    except Exception as e:
        raise RuntimeError(
            "vLLM is required for --engine vllm. Install with `pip install vllm`."
        ) from e

    prompt = _build_vllm_prompt(
        goal,
        obs,
        valid_actions,
        action_history=action_history,
        tokenizer=tokenizer,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        domain=domain,
    )
    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        seed=int(seed),
    )
    try:
        out = llm.generate([prompt], sampling_params)[0]
    except ValueError as e:
        if tokenizer is None or max_model_len is None or "maximum model length" not in str(e).lower():
            raise
        max_prompt_tokens = max(256, int(max_model_len) - int(max_new_tokens) - 32)
        retry_prompt, retry_len = _emergency_trim_prompt(
            prompt,
            tokenizer,
            max_prompt_tokens=max_prompt_tokens,
        )
        print(
            f"vLLM rejected an oversized prompt; retrying with {retry_len} prompt tokens.",
            flush=True,
        )
        out = llm.generate([retry_prompt], sampling_params)[0]
        prompt = retry_prompt
    raw_response = out.outputs[0].text if out.outputs else ""
    action_text, rationale = parse_qwen_answer_only(raw_response or "")
    action = _resolve_action_to_valid(action_text, valid_actions)
    return action, rationale, prompt, raw_response


def run_episode_with_checkpoints(
    env: Any,
    goal_idx: int,
    max_steps: int,
    seed: int,
    action_selector: Callable[
        [str, str, List[str], List[str], int],
        Tuple[str, str, str, str],
    ],
) -> List[Dict[str, Any]]:
    """
    Run one deterministic episode for a specific goal index and emit checkpoints.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    obs, info = env.reset(session=int(goal_idx))
    goal = info.get("goal", env.instruction_text) if info else env.instruction_text

    checkpoints: List[Dict[str, Any]] = []
    prefix_actions: List[str] = []
    seen_products: Dict[str, dict] = {}
    last_seen_count = 0
    last_product_page_signature: Optional[str] = None
    checkpointed_item_page_ids: Set[str] = set()
    opened_item_ids: Set[str] = set()
    visited_product_page = False
    action_history: List[str] = []
    done = False
    is_paper_search = getattr(env, "ccs_domain", "") == "paper_search"

    for step in range(1, max_steps + 1):
        update_seen_products(obs, seen_products, env)
        grew = len(seen_products) > last_seen_count
        last_seen_count = len(seen_products)

        on_product_page = is_product_page_observation(obs, env=env)
        current_page_signature = observation_signature(obs) if on_product_page else None
        if on_product_page:
            visited_product_page = True
            current_item_id = _paper_id_from_observation(obs) if is_paper_search else None
            if current_item_id:
                opened_item_ids.add(current_item_id)
        else:
            current_item_id = None
        visible_item_ids = _paper_result_ids_from_observation(obs) if is_paper_search else []

        checkpoint_due = False
        trigger = None
        if grew:
            checkpoint_due = True
            trigger = "new_product"
        elif on_product_page and current_page_signature:
            if is_paper_search and current_item_id:
                page_is_new_decision = current_item_id not in checkpointed_item_page_ids
            else:
                page_is_new_decision = current_page_signature != last_product_page_signature
            if page_is_new_decision:
                checkpoint_due = True
                trigger = (
                    "paper_page"
                    if is_paper_search
                    else "product_page"
                )
                last_product_page_signature = current_page_signature
                if is_paper_search and current_item_id:
                    checkpointed_item_page_ids.add(current_item_id)

        if checkpoint_due and len(prefix_actions) > 0:
            checkpoints.append(
                {
                    "goal_idx": goal_idx,
                    "goal_text": goal,
                    "checkpoint_step": step,
                    "prefix_actions": list(prefix_actions),
                    "observation": obs[:2000] if obs else "",
                    "trigger": trigger,
                    **_visited_page_metadata(visited_product_page),
                    **_checkpoint_metadata_for_env(
                        env=env,
                        obs=obs,
                        seen_products=seen_products,
                        opened_item_ids=opened_item_ids,
                        visible_item_ids=visible_item_ids,
                        current_item_id=current_item_id,
                    ),
                }
            )

        valid_actions = get_valid_actions_from_env(env)
        action, _rationale, _prompt_used, _raw_response = action_selector(
            goal,
            obs,
            valid_actions,
            action_history,
            int(seed + step * 10007),
        )

        next_obs, _reward, done, _info = env.step(action)

        prefix_actions.append(action)
        action_history.append(action)
        obs = next_obs

        if done or action.startswith("stop") or "buy now" in action.lower():
            break

    need_final_checkpoint = bool(prefix_actions)
    if checkpoints:
        last_ckpt = checkpoints[-1]
        last_prefix = list(last_ckpt.get("prefix_actions", []))
        last_trigger = str(last_ckpt.get("trigger", ""))
        if last_trigger == "final" and last_prefix == list(prefix_actions):
            need_final_checkpoint = False

    if need_final_checkpoint:
        update_seen_products(obs, seen_products, env)
        final_on_product_page = is_product_page_observation(obs, env=env)
        if final_on_product_page:
            visited_product_page = True
            final_seen_item_id = _paper_id_from_observation(obs) if is_paper_search else None
            if final_seen_item_id:
                opened_item_ids.add(final_seen_item_id)
        final_current_item_id = _paper_id_from_observation(obs) if is_paper_search else None
        final_visible_item_ids = _paper_result_ids_from_observation(obs) if is_paper_search else []
        checkpoints.append(
            {
                "goal_idx": goal_idx,
                "goal_text": goal,
                "checkpoint_step": len(prefix_actions),
                "prefix_actions": list(prefix_actions),
                "observation": obs[:2000] if obs else "",
                "trigger": "final",
                **_visited_page_metadata(visited_product_page),
                **_checkpoint_metadata_for_env(
                    env=env,
                    obs=obs,
                    seen_products=seen_products,
                    opened_item_ids=opened_item_ids,
                    visible_item_ids=final_visible_item_ids,
                    current_item_id=final_current_item_id,
                ),
            }
        )

    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description="Build checkpoints for forecast training")
    parser.add_argument(
        "--engine",
        type=str,
        default="vllm",
        choices=["transformers", "vllm"],
        help="Search-model inference backend.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or model ID for the search model.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL file for checkpoints.",
    )
    parser.add_argument("--goal_start", type=int, default=0, help="Starting goal index (inclusive).")
    parser.add_argument("--goal_end", type=int, default=100, help="Ending goal index (exclusive).")
    parser.add_argument(
        "--goal_ids_path",
        type=str,
        default=None,
        help="Optional file with explicit goal ids (JSON list/object or one-per-line text).",
    )
    parser.add_argument("--goal_overrides_path", type=str, default=None)
    parser.add_argument(
        "--domain",
        type=str,
        default="webshop",
        choices=ENV_DOMAIN_CHOICES,
        help="Environment/domain to use.",
    )
    parser.add_argument("--max_steps", type=int, default=60, help="Maximum steps per episode.")
    parser.add_argument("--seed", type=int, default=123, help="Base random seed.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set and output exists, skip goal_idx values already present in output.",
    )

    parser.add_argument(
        "--webshop_file_path",
        type=str,
        default=None,
        help=(
            "Optional WebShop product catalog path. "
            "If unset, WebShop default is used."
        ),
    )
    parser.add_argument(
        "--human_goals",
        action="store_true",
        help="If set, use human-authored goals from items_human_ins.",
    )
    parser.add_argument(
        "--limit_goals",
        type=int,
        default=None,
        help="Optional cap on available goals inside WebShop env.",
    )
    parser.add_argument(
        "--num_products",
        type=int,
        default=None,
        help="Optional cap on searchable products (debug/small-scale only).",
    )
    parser.add_argument("--paper_query_path", type=str, default="")
    parser.add_argument("--paper_corpus_path", type=str, default="")
    parser.add_argument("--paper_qrels_path", type=str, default="")
    parser.add_argument("--paper_dataset_name", type=str, default="princeton-nlp/LitSearch")
    parser.add_argument("--paper_query_config", type=str, default="query")
    parser.add_argument("--paper_corpus_config", type=str, default="corpus_clean")
    parser.add_argument("--paper_split", type=str, default="full")
    parser.add_argument("--paper_page_size", type=int, default=10)
    parser.add_argument("--paper_max_results", type=int, default=50)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="paper_page_litsearch_webshop_relevance_v4",
        help="Reward mode used by paper-search environments while building checkpoints.",
    )
    parser.add_argument("--action_temperature", type=float, default=0.0, help="Search-model decoding temperature.")
    parser.add_argument("--action_top_p", type=float, default=1.0, help="Search-model decoding top-p.")
    parser.add_argument("--action_max_new_tokens", type=int, default=200, help="Max new tokens for action decoding.")

    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel degree.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU memory utilization target.")
    parser.add_argument("--max_model_len", type=int, default=None, help="vLLM max model length override.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="Model dtype (vLLM backend).",
    )
    parser.add_argument("--quantization", type=str, default=None, help="vLLM quantization mode (optional).")
    parser.add_argument("--enforce_eager", action="store_true", help="Force eager mode for vLLM.")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.goal_ids_path:
        goal_ids = _read_goal_ids(Path(args.goal_ids_path))
        if not goal_ids:
            raise ValueError(f"No goal ids loaded from {args.goal_ids_path}")
    else:
        if args.goal_end <= args.goal_start:
            raise ValueError("--goal_end must be > --goal_start when --goal_ids_path is not used.")
        goal_ids = list(range(int(args.goal_start), int(args.goal_end)))

    existing_goal_ids: Set[int] = set()
    if bool(args.resume) and output_path.exists():
        existing_goal_ids = _existing_goal_ids_in_output(output_path)
        if existing_goal_ids:
            print(
                f"Resume enabled: found {len(existing_goal_ids)} goal_idx values already present in output.",
                flush=True,
            )
    todo_goal_ids = [g for g in goal_ids if g not in existing_goal_ids]
    if not todo_goal_ids:
        print("Nothing to do: all requested goals already present in output.", flush=True)
        return

    print(f"Loading search model from {args.model_path} (engine={args.engine})...", flush=True)
    if args.engine == "transformers":
        model, tokenizer = load_transformers_model(args.model_path)

        def action_selector(
            goal: str,
            obs: str,
            valid_actions: List[str],
            action_history: List[str],
            step_seed: int,
        ) -> Tuple[str, str, str, str]:
            del step_seed
            return select_action_qwen(
                goal=goal,
                obs=obs,
                valid_actions=valid_actions,
                model=model,
                tokenizer=tokenizer,
                action_history=action_history,
            )

    else:
        llm = load_vllm_model(
            args.model_path,
            tensor_parallel_size=int(args.tensor_parallel_size),
            gpu_memory_utilization=float(args.gpu_memory_utilization),
            max_model_len=args.max_model_len,
            quantization=args.quantization,
            dtype=args.dtype,
            enforce_eager=bool(args.enforce_eager),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=True,
            trust_remote_code=True,
        )

        def action_selector(
            goal: str,
            obs: str,
            valid_actions: List[str],
            action_history: List[str],
            step_seed: int,
        ) -> Tuple[str, str, str, str]:
            return select_action_qwen_vllm(
                goal=goal,
                obs=obs,
                valid_actions=valid_actions,
                llm=llm,
                action_history=action_history,
                temperature=float(args.action_temperature),
                top_p=float(args.action_top_p),
                max_new_tokens=int(args.action_max_new_tokens),
                seed=int(step_seed),
                tokenizer=tokenizer,
                max_model_len=args.max_model_len,
                domain=str(args.domain),
            )
    print("Search model loaded.", flush=True)

    print(f"Initializing {args.domain} environment...", flush=True)
    env = build_env_from_args(args)
    if args.domain == "webshop":
        _apply_goal_overrides(env, args.goal_overrides_path)
    print("Environment initialized.", flush=True)

    total_checkpoints = 0
    processed_goals = 0
    t0 = time.time()
    mode = "a" if (bool(args.resume) and output_path.exists()) else "w"
    with output_path.open(mode, buffering=1) as f:
        for i, goal_idx in enumerate(todo_goal_ids, start=1):
            ep_seed = int(args.seed) + int(goal_idx)
            checkpoints = run_episode_with_checkpoints(
                env=env,
                goal_idx=int(goal_idx),
                max_steps=int(args.max_steps),
                seed=ep_seed,
                action_selector=action_selector,
            )
            for ckpt in checkpoints:
                f.write(json.dumps(ckpt) + "\n")
                total_checkpoints += 1
            processed_goals += 1
            f.flush()
            os.fsync(f.fileno())

            if i % 10 == 0 or i == len(todo_goal_ids):
                elapsed = time.time() - t0
                print(
                    f"  Goal {i}/{len(todo_goal_ids)} (goal_idx={goal_idx}): "
                    f"{total_checkpoints} checkpoints this run, {elapsed/60:.1f} min elapsed",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"\nDone! Collected {total_checkpoints} checkpoints from {processed_goals} goals "
        f"(requested={len(goal_ids)}, skipped_existing={len(existing_goal_ids & set(goal_ids))}) "
        f"in {elapsed/60:.1f} min",
        flush=True,
    )
    print(f"Output saved to: {output_path}", flush=True)
    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()

"""Label WebShop decision states with Monte Carlo continuations."""

import os
import argparse
import json
import random
import time
import multiprocessing as mp
import math
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Sequence

from opportunity_forecasting import REPO_ROOT

os.environ.setdefault(
    "WEBSHOP_DATA_DIR",
    str(REPO_ROOT / "third_party" / "WebShop" / "data"),
)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from opportunity_forecasting.data.environments import build_env_from_args
from opportunity_forecasting.models.search_runtime import (
    get_valid_actions_from_env,
    update_seen_products,
    extract_asins_from_observation,
    parse_qwen_answer_only,
    _categorise_action,
)
from opportunity_forecasting.data.webshop_action_prompt import WEBSHOP_ACTION_PROMPT
from opportunity_forecasting.data.webshop_rewards import (
    COUNTERFACTUAL_REWARD_KEY,
    COUNTERFACTUAL_REWARD_MODE,
)

try:
    from vllm import LLM, SamplingParams
except Exception as e:
    LLM = None
    SamplingParams = None
    VLLM_IMPORT_ERROR = e
else:
    VLLM_IMPORT_ERROR = None

from transformers import AutoTokenizer


def _require_vllm() -> None:
    if LLM is None or SamplingParams is None:
        raise RuntimeError(
            "vLLM is required for Monte Carlo continuation generation. "
            "Create the WebShop environment from environments/webshop.yml. "
            f"Import error: {VLLM_IMPORT_ERROR}"
        ) from VLLM_IMPORT_ERROR


def _best_reward_from_seen_products(
    seen_products: Dict[str, dict],
    reward_mode: str = COUNTERFACTUAL_REWARD_MODE,
) -> float:
    best = 0.0
    for _, info in (seen_products or {}).items():
        if not isinstance(info, dict):
            continue
        for k in (COUNTERFACTUAL_REWARD_KEY, "reward"):
            v = info.get(k, None)
            if v is None:
                continue
            try:
                best = max(best, float(v))
            except Exception:
                pass
    return best


def replay_to_checkpoint(
    env,
    goal_idx: int,
    prefix_actions: List[str],
    reward_mode: str = COUNTERFACTUAL_REWARD_MODE,
) -> Tuple[str, float, Dict[str, dict]]:
    """
    Reset environment to goal_idx and replay prefix_actions.
    Returns: (observation, best_reward_seen, seen_products)
    """
    obs, info = env.reset(session=int(goal_idx))

    seen_products: Dict[str, dict] = {}
    best_reward_seen: float = 0.0

    for action in prefix_actions:
        update_seen_products(obs, seen_products, env)
        best_reward_seen = max(
            best_reward_seen,
            _best_reward_from_seen_products(seen_products, reward_mode=reward_mode),
        )
        obs, rew, done, info = env.step(action)

        try:
            current_asins = extract_asins_from_observation(obs)
            for asin in current_asins:
                if asin in seen_products:
                    prev = seen_products[asin].get("reward", None)
                    if prev is None or float(rew) > float(prev):
                        seen_products[asin]["reward"] = float(rew)
        except Exception:
            pass
        try:
            best_reward_seen = max(best_reward_seen, float(rew))
        except Exception:
            pass
        best_reward_seen = max(
            best_reward_seen,
            _best_reward_from_seen_products(seen_products, reward_mode=reward_mode),
        )

        if done:
            break

    update_seen_products(obs, seen_products, env)
    best_reward_seen = max(
        best_reward_seen,
        _best_reward_from_seen_products(seen_products, reward_mode=reward_mode),
    )
    return obs, best_reward_seen, seen_products


def _sanitize_seen_products_for_reward_mode(
    seen_products: Dict[str, dict],
    reward_mode: str,
) -> Dict[str, dict]:
    if not isinstance(seen_products, dict):
        return {}
    cleaned: Dict[str, dict] = {}
    for asin, info in seen_products.items():
        if not isinstance(info, dict):
            continue
        pruned = dict(info)
        pruned.pop("ws_reward", None)
        cleaned[str(asin)] = pruned
    return cleaned


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
) -> str:
    obs_truncated = (obs or "")[:obs_char_limit]

    search_actions = []
    nav_actions = []
    terminal_actions = []
    for act in valid_actions:
        cat = _categorise_action(act)
        if cat == "search":
            search_actions.append(act)
        elif cat == "terminal":
            terminal_actions.append(act)
        else:
            nav_actions.append(act)

    lines = []
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
        lines.append("Terminal actions (commit to a purchase or stop):")
        for act in terminal_actions:
            lines.append(f"  {idx}. {_shorten_action_for_prompt(act, action_display_limit)}")
            idx += 1
    actions_str = "\n".join(lines) if lines else "None"

    if action_history:
        recent = list(action_history)[-hist_k:]
        hist_str = "\n".join(f"- {a}" for a in recent) if recent else "None"
    else:
        hist_str = "None"

    if hist_k == 60 and obs_char_limit == 1200 and action_display_limit is None and not allow_numeric_answer:
        return WEBSHOP_ACTION_PROMPT.format(
            goal=goal,
            action_history=hist_str,
            observation=obs_truncated,
            actions=actions_str,
            hist_k=hist_k,
        )

    answer_spec = "ACTION_NUMBER_OR_ONE_ACTION_FROM_LIST" if allow_numeric_answer else "ONE_ACTION_FROM_LIST"
    action_instruction = (
        "Available Actions (reply with the action number or copy one action exactly as written in the list below)."
        if allow_numeric_answer
        else "Available Actions (copy one EXACTLY as written)."
    )
    return (
        "You are a shopping assistant. Select the single best action from the list to progress toward the goal.\n\n"
        f"Shopping Goal: {goal}\n\n"
        f"Recent Action History (last {hist_k}):\n"
        f"{hist_str}\n\n"
        f"Current Page (first {obs_char_limit} chars):\n"
        f"{obs_truncated}\n\n"
        f"{action_instruction}\n\n"
        "- Search actions help you find or refine the right category of products.\n"
        "- Navigation actions help you scroll, move between pages, or inspect products.\n\n"
        "Available Actions:\n"
        f"{actions_str}\n\n"
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


def _build_vllm_action_prompt(
    goal: str,
    obs: str,
    valid_actions: Sequence[str],
    action_history: Optional[Sequence[str]],
    *,
    tokenizer: Optional[Any],
    max_model_len: Optional[int],
    max_new_tokens: int,
) -> str:
    prompt = _build_action_prompt(goal, obs, valid_actions, action_history=action_history)
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
            **settings,
        )
        candidate_len = len(_tokenizer_encode(tokenizer, candidate))
        if candidate_len <= max_prompt_tokens:
            print(
                f"Prompt trimmed from {original_prompt_len} to {candidate_len} tokens for context-window safety.",
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
        f"Prompt emergency-trimmed from {original_prompt_len} to {trimmed_len} tokens for context-window safety.",
        flush=True,
    )
    return trimmed_prompt


def _vllm_generate_texts(
    llm: "LLM",
    prompts: List[str],
    sampling_params_list: List["SamplingParams"],
) -> List[str]:
    """
    Generate one completion per prompt.

    We prefer a single batched call (for throughput / continuous batching).
    If the installed vLLM doesn't accept per-prompt SamplingParams in a list,
    we fall back to sequential calls to preserve per-continuation seeding.
    """
    if len(prompts) != len(sampling_params_list):
        raise ValueError("prompts and sampling_params_list must have same length")

    try:
        req_outs = llm.generate(prompts, sampling_params_list)
        texts: List[str] = []
        for o in req_outs:
            if not o.outputs:
                texts.append("")
            else:
                texts.append(o.outputs[0].text)
        return texts
    except TypeError:
        texts = []
        for p, sp in zip(prompts, sampling_params_list):
            o = llm.generate([p], sp)[0]
            texts.append(o.outputs[0].text if o.outputs else "")
        return texts


def _choose_actions_batched(
    llm: "LLM",
    goals: List[str],
    obses: List[str],
    valid_actions_list: List[List[str]],
    histories: List[List[str]],
    *,
    temperature: float,
    top_p: float,
    action_max_new_tokens: int,
    seeds: List[int],
    tokenizer: Optional[Any],
    max_model_len: Optional[int],
) -> List[str]:
    prompts = [
        _build_vllm_action_prompt(
            goal=g,
            obs=o,
            valid_actions=va,
            action_history=h,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
            max_new_tokens=action_max_new_tokens,
        )
        for g, o, va, h in zip(goals, obses, valid_actions_list, histories)
    ]

    sampling_params_list = [
        SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(action_max_new_tokens),
            seed=int(s),
        )
        for s in seeds
    ]

    try:
        raw_texts = _vllm_generate_texts(llm, prompts, sampling_params_list)
    except ValueError as e:
        if tokenizer is None or max_model_len is None or "maximum model length" not in str(e).lower():
            raise
        max_prompt_tokens = max(256, int(max_model_len) - int(action_max_new_tokens) - 32)
        retry_prompts = []
        for prompt in prompts:
            retry_prompt, _retry_len = _emergency_trim_prompt(
                prompt,
                tokenizer,
                max_prompt_tokens=max_prompt_tokens,
            )
            retry_prompts.append(retry_prompt)
        print("vLLM rejected an oversized prompt batch; retrying with emergency-trimmed prompts.", flush=True)
        raw_texts = _vllm_generate_texts(llm, retry_prompts, sampling_params_list)

    chosen_actions: List[str] = []
    for raw, valid_actions in zip(raw_texts, valid_actions_list):
        action, _rationale = parse_qwen_answer_only(raw or "")
        action_normalized = (action or "").strip()
        action_lower = action_normalized.lower()

        picked = None
        for valid_act in valid_actions:
            if valid_act.lower() == action_lower:
                picked = valid_act
                break

        if picked is None and len(action_normalized) > 3:
            for valid_act in valid_actions:
                valid_lower = valid_act.lower()
                if action_lower in valid_lower or valid_lower in action_lower:
                    picked = valid_act
                    break

        if picked is None:
            picked = random.choice(valid_actions) if valid_actions else "stop"

        chosen_actions.append(picked)

    return chosen_actions


def _final_reward_from_env(env) -> float:
    session_dict = env.browser.server.user_sessions.get(env.session, {})
    session_reward = session_dict.get("reward", None)
    return float(session_reward) if session_reward is not None else -1.0


def run_continuations_lockstep(
    envs: List["WebAgentTextEnv"],
    llm: "LLM",
    *,
    goal_idx: int,
    goal_text: str,
    prefix_actions: List[str],
    max_steps: int,
    temperature: float,
    top_p: float,
    action_max_new_tokens: int,
    per_cont_seeds: List[int],
    reward_mode: str = COUNTERFACTUAL_REWARD_MODE,
    tokenizer: Optional[Any] = None,
    max_model_len: Optional[int] = None,
) -> List[float]:
    """
    Run K continuations in lockstep across K envs, batching action selection with vLLM.
    Returns per-continuation best immediate buy-now reward seen during the continuation.
    """
    K = len(envs)
    if K == 0:
        return []
    if len(per_cont_seeds) != K:
        raise ValueError("per_cont_seeds length must match number of envs")

    obses: List[str] = []
    histories: List[List[str]] = []
    seen_products_list: List[Dict[str, dict]] = []
    best_seen: List[float] = []
    done_flags: List[bool] = []
    final_best: List[float] = [0.0 for _ in range(K)]

    for env in envs:
        obs, _best, _seen = replay_to_checkpoint(env=env, goal_idx=goal_idx, prefix_actions=prefix_actions)
        obses.append(obs)
        histories.append(list(prefix_actions))
        seen_products_list.append({})
        best_seen.append(0.0)
        done_flags.append(False)

    for step in range(int(max_steps)):
        active_idxs = [i for i, d in enumerate(done_flags) if not d]
        if not active_idxs:
            break

        goals = [goal_text for _ in active_idxs]
        active_obses = [obses[i] for i in active_idxs]
        active_hists = [histories[i] for i in active_idxs]
        active_valid_actions = [get_valid_actions_from_env(envs[i]) for i in active_idxs]

        for j, va in enumerate(active_valid_actions):
            if not va:
                idx = active_idxs[j]
                done_flags[idx] = True
                final_best[idx] = max(final_best[idx], best_seen[idx])

        active_idxs = [i for i in active_idxs if not done_flags[i]]
        if not active_idxs:
            break

        goals = [goal_text for _ in active_idxs]
        active_obses = [obses[i] for i in active_idxs]
        active_hists = [histories[i] for i in active_idxs]
        active_valid_actions = [get_valid_actions_from_env(envs[i]) for i in active_idxs]

        seeds = [int(per_cont_seeds[i] + 10_007 * step) for i in active_idxs]

        actions = _choose_actions_batched(
            llm,
            goals,
            active_obses,
            active_valid_actions,
            active_hists,
            temperature=temperature,
            top_p=top_p,
            action_max_new_tokens=action_max_new_tokens,
            seeds=seeds,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
        )

        for idx, act in zip(active_idxs, actions):
            obs, rew, done, info = envs[idx].step(act)
            obses[idx] = obs
            histories[idx].append(act)
            update_seen_products(obs, seen_products_list[idx], envs[idx])
            best_seen[idx] = max(
                best_seen[idx],
                _best_reward_from_seen_products(seen_products_list[idx], reward_mode=reward_mode),
            )
            try:
                best_seen[idx] = max(best_seen[idx], float(rew))
            except Exception:
                pass

            if done or "buy now" in act.lower() or act.startswith("stop"):
                done_flags[idx] = True
                try:
                    term_r = _final_reward_from_env(envs[idx])
                    best_seen[idx] = max(best_seen[idx], float(term_r))
                except Exception:
                    pass
                final_best[idx] = max(final_best[idx], best_seen[idx])

    for i in range(K):
        if final_best[i] < 0:
            final_best[i] = max(best_seen[i], -1.0)

    return final_best


def label_checkpoint_vllm(
    env_for_baseline: "WebAgentTextEnv",
    envs_for_continuations: Optional[List["WebAgentTextEnv"]],
    llm: "LLM",
    tokenizer: Optional[Any],
    checkpoint: Dict[str, Any],
    *,
    num_continuations: int,
    continuation_max_steps: int,
    temperature: float,
    top_p: float,
    action_max_new_tokens: int,
    batch_continuations: bool,
    reward_mode: str = COUNTERFACTUAL_REWARD_MODE,
    total_horizon_steps: Optional[int] = None,
    max_model_len: Optional[int] = None,
    seed: int = 123,
) -> Dict[str, Any]:
    goal_idx = checkpoint["goal_idx"]
    goal_text = checkpoint["goal_text"]
    prefix_actions = checkpoint["prefix_actions"]

    replayed_obs, replayed_best_reward_seen, replayed_seen = replay_to_checkpoint(
        env=env_for_baseline,
        goal_idx=goal_idx,
        prefix_actions=prefix_actions,
        reward_mode=reward_mode,
    )
    baseline_seen = replayed_seen
    best_reward_seen = float(replayed_best_reward_seen)
    baseline_seen = _sanitize_seen_products_for_reward_mode(baseline_seen, reward_mode=reward_mode)
    baseline_obs = checkpoint.get("observation", replayed_obs)
    if not isinstance(baseline_obs, str) or not baseline_obs:
        baseline_obs = replayed_obs

    per_cont_seeds: List[int] = []
    ckpt_step = int(checkpoint.get("checkpoint_step", 0) or 0)
    if total_horizon_steps is not None:
        remaining = max(0, int(total_horizon_steps) - ckpt_step)
        effective_max_steps = min(int(continuation_max_steps), remaining)
    else:
        effective_max_steps = int(continuation_max_steps)
    for k in range(int(num_continuations)):
        per_seed = (
            int(1_000_003 * int(goal_idx))
            + int(97_133 * int(checkpoint.get("checkpoint_step", 0)))
            + int(1_009 * int(k))
        ) ^ int(seed)
        per_cont_seeds.append(int(per_seed))

    if batch_continuations:
        if envs_for_continuations is None or len(envs_for_continuations) != int(num_continuations):
            raise ValueError("batch_continuations requires exactly K continuation envs")
        max_future_bests = run_continuations_lockstep(
            envs_for_continuations,
            llm,
            goal_idx=int(goal_idx),
            goal_text=str(goal_text),
            prefix_actions=list(prefix_actions),
            max_steps=int(effective_max_steps),
            temperature=float(temperature),
            top_p=float(top_p),
            action_max_new_tokens=int(action_max_new_tokens),
            per_cont_seeds=per_cont_seeds,
            reward_mode=reward_mode,
            tokenizer=tokenizer,
            max_model_len=max_model_len,
        )
    else:
        max_future_bests: List[float] = []
        for k in range(int(num_continuations)):
            try:
                random.seed(per_cont_seeds[k])
            except Exception:
                pass
            tmp_env = envs_for_continuations[0] if envs_for_continuations else env_for_baseline
            replay_to_checkpoint(
                env=tmp_env,
                goal_idx=goal_idx,
                prefix_actions=prefix_actions,
                reward_mode=reward_mode,
            )
            rewards = run_continuations_lockstep(
                [tmp_env],
                llm,
                goal_idx=int(goal_idx),
                goal_text=str(goal_text),
                prefix_actions=list(prefix_actions),
                max_steps=int(effective_max_steps),
                temperature=float(temperature),
                top_p=float(top_p),
                action_max_new_tokens=int(action_max_new_tokens),
                per_cont_seeds=[per_cont_seeds[k]],
                reward_mode=reward_mode,
                tokenizer=tokenizer,
                max_model_len=max_model_len,
            )
            max_future_bests.append(float(rewards[0]))

    continuation_deltas = [max(0.0, float(b) - float(best_reward_seen)) for b in max_future_bests]
    return {
        "goal_idx": goal_idx,
        "goal_text": goal_text,
        "checkpoint_step": checkpoint["checkpoint_step"],
        "input": {
            "goal": goal_text,
            "seen_products": baseline_seen,
            "best_reward_seen": best_reward_seen,
            "observation": (baseline_obs or checkpoint.get("observation", ""))[:1000],
        },
        "continuation_deltas": continuation_deltas,
        "metadata": {
            "trigger": checkpoint.get("trigger", "unknown"),
            "num_continuations": int(num_continuations),
            "temperature": float(temperature),
            "label_seed": int(seed),
            "engine": "vllm",
            "reward_mode": str(reward_mode),
            "effective_continuation_max_steps": int(effective_max_steps),
            "total_horizon_steps": int(total_horizon_steps) if total_horizon_steps is not None else None,
        },
    }


def load_llm(
    model_path: str,
    *,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    quantization: Optional[str],
    dtype: str,
    enforce_eager: bool,
) -> "LLM":
    _require_vllm()
    kwargs: Dict[str, Any] = dict(
        model=model_path,
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
    return LLM(**kwargs)


def _count_valid_jsonl_lines_and_last_good_offset(path: Path) -> Tuple[int, int]:
    """
    Return (num_valid_json_lines, last_good_byte_offset).

    If the file contains a partially-written / corrupted last line, we stop at the
    last valid JSON line and return the byte offset right after it.
    """
    if not path.exists():
        return 0, 0
    n = 0
    last_good = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.strip():
                break
            try:
                json.loads(line.decode("utf-8"))
            except Exception:
                break
            n += 1
            last_good = f.tell()
    return n, last_good


def _truncate_to_offset(path: Path, offset: int) -> None:
    if not path.exists():
        return
    with path.open("r+b") as f:
        f.truncate(int(offset))


def main():
    ap = argparse.ArgumentParser(description="Label WebShop checkpoints with Monte Carlo continuations (vLLM backend)")
    ap.add_argument("--checkpoint_path", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)
    ap.add_argument("--model_path", type=str, required=True)

    ap.add_argument("--num_continuations", type=int, default=6)
    ap.add_argument("--continuation_max_steps", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--action_max_new_tokens", type=int, default=200)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_checkpoints", type=int, default=None)
    ap.add_argument("--shard_idx", type=int, default=None)
    ap.add_argument("--num_shards", type=int, default=None)
    ap.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If set and output_path already exists, resume by skipping the number of "
            "already-written valid JSONL lines (and truncating any partial last line)."
        ),
    )

    ap.add_argument("--webshop_file_path", type=str, default=None)
    ap.add_argument("--human_goals", action="store_true")
    ap.add_argument("--limit_goals", type=int, default=None)
    ap.add_argument("--num_products", type=int, default=None)
    ap.add_argument(
        "--reward_mode",
        type=str,
        default=COUNTERFACTUAL_REWARD_MODE,
        choices=[COUNTERFACTUAL_REWARD_MODE],
        help="Reward extraction mode for best_reward_seen and continuation targets.",
    )
    ap.add_argument(
        "--total_horizon_steps",
        type=int,
        default=60,
        help=(
            "If set, cap continuation steps so checkpoint_step + continuation_steps <= total_horizon_steps. "
            "Effective steps per checkpoint become min(continuation_max_steps, total_horizon_steps - checkpoint_step)."
        ),
    )

    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs to use for tensor parallelism.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.75, help="vLLM GPU memory utilization target (0-1).")
    ap.add_argument("--max_model_len", type=int, default=8192, help="vLLM max model length override.")
    ap.add_argument(
        "--dtype",
        type=str,
        default="half",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="vLLM model dtype. For RTX 6000 Turing, prefer 'half' (fp16).",
    )
    ap.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="vLLM quantization mode (e.g., fp8). See vLLM docs for support by GPU.",
    )
    ap.add_argument(
        "--enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set, force eager execution (can improve compatibility on some older GPUs/drivers).",
    )
    ap.add_argument(
        "--batch_continuations",
        action="store_true",
        help="If set, run K continuations in lockstep and batch action selection with vLLM (usually much faster).",
    )

    args = ap.parse_args()

    random.seed(args.seed)

    checkpoint_path = Path(args.checkpoint_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoints from {checkpoint_path}...", flush=True)
    checkpoints: List[Dict[str, Any]] = []
    with checkpoint_path.open() as f:
        for line in f:
            if line.strip():
                checkpoints.append(json.loads(line))
    print(f"Loaded {len(checkpoints)} checkpoints.", flush=True)

    if args.max_checkpoints:
        checkpoints = checkpoints[: args.max_checkpoints]
        print(f"Processing first {len(checkpoints)} checkpoints.", flush=True)

    if args.shard_idx is not None or args.num_shards is not None:
        if args.shard_idx is None or args.num_shards is None:
            raise ValueError("Must set both --shard_idx and --num_shards (or neither).")
        if args.num_shards <= 0:
            raise ValueError("--num_shards must be > 0")
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError("--shard_idx must be in [0, num_shards)")
        orig_n = len(checkpoints)
        checkpoints = [ckpt for i, ckpt in enumerate(checkpoints) if (i % args.num_shards) == args.shard_idx]
        print(
            f"Sharding enabled: shard_idx={args.shard_idx} num_shards={args.num_shards} -> {len(checkpoints)}/{orig_n} checkpoints",
            flush=True,
        )

    resume_n = 0
    if bool(args.resume) and output_path.exists():
        resume_n, last_good = _count_valid_jsonl_lines_and_last_good_offset(output_path)
        if last_good > 0:
            _truncate_to_offset(output_path, last_good)
        if resume_n > 0:
            print(
                f"Resume enabled: found {resume_n} valid JSONL lines in {output_path}; skipping those checkpoints.",
                flush=True,
            )
        else:
            print(
                f"Resume enabled: output exists but has 0 valid lines; starting from scratch: {output_path}",
                flush=True,
            )
        if resume_n >= len(checkpoints):
            print(
                f"Resume: output already contains >= total shard checkpoints ({resume_n} >= {len(checkpoints)}). Nothing to do.",
                flush=True,
            )
            return
        checkpoints = checkpoints[resume_n:]

    print(
        f"Loading vLLM model '{args.model_path}' (tp={args.tensor_parallel_size}, mem_util={args.gpu_memory_utilization}, "
        f"max_model_len={args.max_model_len}, quantization={args.quantization})...",
        flush=True,
    )
    prompt_tokenizer = None
    if args.max_model_len is not None:
        print(f"Loading tokenizer from '{args.model_path}' for prompt-length safety...", flush=True)
        prompt_tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            use_fast=True,
        )
        print("Tokenizer loaded.", flush=True)
    llm = load_llm(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        quantization=args.quantization,
        dtype=args.dtype,
        enforce_eager=bool(args.enforce_eager),
    )
    print("vLLM model loaded.", flush=True)

    print("Initializing WebShop environment(s)...", flush=True)
    env_baseline = build_env_from_args(args)
    cont_envs: Optional[List["WebAgentTextEnv"]] = None
    if args.batch_continuations:
        cont_envs = [build_env_from_args(args) for _ in range(int(args.num_continuations))]
    print("Environment(s) initialized.", flush=True)

    t0 = time.time()
    mode = "a" if (bool(args.resume) and output_path.exists()) else "w"
    with output_path.open(mode) as f:
        for i, ckpt in enumerate(checkpoints):
            labeled = label_checkpoint_vllm(
                env_for_baseline=env_baseline,
                envs_for_continuations=cont_envs,
                llm=llm,
                tokenizer=prompt_tokenizer,
                checkpoint=ckpt,
                num_continuations=args.num_continuations,
                continuation_max_steps=args.continuation_max_steps,
                temperature=args.temperature,
                top_p=args.top_p,
                action_max_new_tokens=args.action_max_new_tokens,
                batch_continuations=bool(args.batch_continuations),
                reward_mode=str(args.reward_mode),
                total_horizon_steps=args.total_horizon_steps,
                max_model_len=args.max_model_len,
                seed=int(args.seed),
            )
            f.write(json.dumps(labeled) + "\n")
            f.flush()
            os.fsync(f.fileno())

            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(checkpoints) - i - 1)
                print(
                    f"  Checkpoint {resume_n + i + 1}/{resume_n + len(checkpoints)}: "
                    f"mean_delta={sum(labeled['continuation_deltas']) / max(1, len(labeled['continuation_deltas'])):.3f}, "
                    f"ETA: {remaining/60:.1f} min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"Done. Labeled {len(checkpoints)} checkpoints in {elapsed/60:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()

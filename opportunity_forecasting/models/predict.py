"""Run the prompted Qwen forecaster on labeled decision states."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from opportunity_forecasting.models.distributions import (
    ForecastParseResult,
    forecast_implied_moments,
    forecast_numeric_domain_ok,
    parse_forecast_response,
)
from opportunity_forecasting.models.prompted_forecaster import (
    build_stopping_prompt,
    format_best_reward_seen,
    format_seen_products_for_prompt,
    format_state_context_for_prompt,
)


def read_labeled_states(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _goal_id(row: dict[str, Any]) -> int:
    inputs = row.get("input", {}) or {}
    return int(row.get("goal_idx", inputs.get("goal_idx", 0)) or 0)


def _step(row: dict[str, Any]) -> int:
    inputs = row.get("input", {}) or {}
    return int(row.get("checkpoint_step", inputs.get("checkpoint_step", 0)) or 0)


def _candidate_page(row: dict[str, Any]) -> bool:
    inputs = row.get("input", {}) or {}
    metadata = row.get("metadata", {}) or {}
    if "visited_product_page" in inputs:
        return bool(inputs["visited_product_page"])
    if "visited_product_page" in metadata:
        return bool(metadata["visited_product_page"])
    if inputs.get("has_opened_paper") or metadata.get("has_opened_paper"):
        return True
    trigger = str(metadata.get("trigger", inputs.get("trigger", ""))).lower()
    if trigger in {"product_page", "paper_page", "item_page", "new_paper_page"}:
        return True
    observation = str(inputs.get("observation", "")).lower()
    return "buy now" in observation or "current_paper_id:" in observation


def _prompt(row: dict[str, Any], tokenizer: Any, top_k_seen: int, max_length: int) -> str:
    inputs = row.get("input", {}) or {}
    metadata = row.get("metadata", {}) or {}
    goal = str(row.get("goal_text", inputs.get("goal", "")) or "")
    seen = inputs.get("seen_products", {}) or {}
    candidates = list(dict.fromkeys([top_k_seen, 12, 10, 8, 6, 5, 4, 3, 2, 1]))
    for top_k in candidates:
        seen_text = (
            format_seen_products_for_prompt(seen, None, top_n=top_k)
            if seen
            else "No items seen yet."
        )
        prompt = build_stopping_prompt(
            goal=goal,
            seen_products_text=seen_text,
            best_reward_seen=inputs.get("best_reward_seen", 0.0),
            top_k=top_k,
            checkpoint_step=_step(row),
            total_horizon_steps=metadata.get("total_horizon_steps", 60),
            trigger=metadata.get("trigger"),
            observation=inputs.get("observation"),
        )
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= max_length:
            return prompt
    raise ValueError(
        f"Prompt exceeds max model length even with one candidate: "
        f"goal={_goal_id(row)} step={_step(row)}"
    )


def _repair_prompt(row: dict[str, Any], top_k_seen: int) -> str:
    inputs = row.get("input", {}) or {}
    metadata = row.get("metadata", {}) or {}
    seen = inputs.get("seen_products", {}) or {}
    seen_text = (
        format_seen_products_for_prompt(seen, None, top_n=max(1, top_k_seen))
        if seen
        else "No items seen yet."
    )
    state = format_state_context_for_prompt(
        checkpoint_step=_step(row),
        total_horizon_steps=metadata.get("total_horizon_steps", 60),
        trigger=metadata.get("trigger"),
        observation=inputs.get("observation"),
    )
    goal = str(row.get("goal_text", inputs.get("goal", "")) or "")
    reward = format_best_reward_seen(inputs.get("best_reward_seen", 0.0))
    return f"""You are forecasting normalized reward upside from continuing a fixed search trajectory.

Search goal:
{goal}

Candidates opened or observed so far:
{seen_text}

Current decision point:
{state}

Current best reward from opened/scored candidates:
{reward}

Return only these four XML numeric fields, with no reasoning and no extra text.
Use finite decimal numbers. Enforce p0 + p1 <= 1, 0 < m_plus < 1, and 0 < k_plus <= 32.

<delta_zero_prob>p0</delta_zero_prob>
<delta_one_prob>p1</delta_one_prob>
<delta_pos_mean>m_plus</delta_pos_mean>
<delta_pos_concentration>k_plus</delta_pos_concentration>
"""


def _load_vllm(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": str(args.model_path),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "trust_remote_code": True,
        "dtype": str(args.dtype),
        "enforce_eager": bool(args.enforce_eager),
        "max_model_len": int(args.max_model_len),
    }
    model = LLM(**kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True
    )
    return model, tokenizer


def _load_transformers(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, trust_remote_code=True
    )
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def _generate_vllm(
    model: Any,
    prompts: list[str],
    *,
    max_tokens: int,
    stop: str,
    batch_size: int,
) -> list[str]:
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        stop=[stop],
        include_stop_str_in_output=True,
    )
    responses: list[str] = []
    for start in range(0, len(prompts), max(1, batch_size)):
        outputs = model.generate(
            prompts[start : start + max(1, batch_size)],
            params,
            use_tqdm=False,
        )
        responses.extend(output.outputs[0].text if output.outputs else "" for output in outputs)
    return responses


def _generate_transformers(
    model: Any,
    tokenizer: Any,
    prompts: Iterable[str],
    *,
    max_tokens: int,
) -> list[str]:
    responses: list[str] = []
    device = next(model.parameters()).device
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        generated = output[0][inputs["input_ids"].shape[1] :]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def _moments(forecast: ForecastParseResult) -> tuple[Optional[float], Optional[float]]:
    if not forecast_numeric_domain_ok(forecast):
        return None, None
    mean, std = forecast_implied_moments(forecast)
    if mean is None or not math.isfinite(mean):
        return None, None
    if std is None or not math.isfinite(std):
        return None, None
    return max(0.0, min(1.0, mean)), max(0.0, std)


def _prediction_row(
    source: dict[str, Any],
    forecast: ForecastParseResult,
    *,
    retry_attempted: bool,
    retry_succeeded: bool,
) -> dict[str, Any]:
    inputs = source.get("input", {}) or {}
    mean, std = _moments(forecast)
    return {
        "checkpoint_step": _step(source),
        "candidate_available": _candidate_page(source),
        "best_reward_seen": float(inputs.get("best_reward_seen", 0.0) or 0.0),
        "forecast_family": forecast.family,
        "delta_zero_prob": forecast.delta_zero_prob,
        "delta_one_prob": forecast.delta_one_prob,
        "delta_pos_mean": forecast.delta_pos_mean,
        "delta_pos_concentration": forecast.delta_pos_concentration,
        "expected_delta": mean,
        "expected_std_delta": std,
        "forecast_numeric_domain_ok": forecast_numeric_domain_ok(forecast),
        "retry_attempted": retry_attempted,
        "retry_succeeded": retry_succeeded,
    }


def write_predictions(
    rows: list[dict[str, Any]],
    forecasts: list[ForecastParseResult],
    *,
    output_path: Path,
    data_path: Path,
    model_path: Path,
    model_label: str | None,
    split: str,
    retry_flags: list[tuple[bool, bool]],
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[dict[str, Any], ForecastParseResult, tuple[bool, bool]]]] = defaultdict(list)
    for row, forecast, flags in zip(rows, forecasts, retry_flags):
        grouped[_goal_id(row)].append((row, forecast, flags))

    valid = sum(forecast_numeric_domain_ok(forecast) for forecast in forecasts)
    parsed = sum(forecast.family is not None for forecast in forecasts)
    payload: dict[str, Any] = {
        "checkpoint_path": str(data_path),
        "label_source": str(data_path),
        "split": split,
        "goal_ids": sorted(grouped),
        "model_path": str(model_label or model_path),
        "model_type": "prompted_forecaster",
        "engine": "prompted_forecaster",
        "top_k_seen": 15,
        "goal_predictions": {},
        "prediction_health": {
            "num_checkpoints_total": len(rows),
            "num_numeric_parsed": parsed,
            "num_valid_forecasts": valid,
            "numeric_parse_rate": parsed / max(1, len(rows)),
            "valid_forecast_rate": valid / max(1, len(rows)),
            "num_retry_attempted": sum(attempted for attempted, _ in retry_flags),
            "num_retry_succeeded": sum(succeeded for _, succeeded in retry_flags),
        },
    }
    for goal_id, values in grouped.items():
        goal_rows = []
        for checkpoint_idx, (source, forecast, flags) in enumerate(values):
            goal_rows.append(
                {
                    "checkpoint_idx": checkpoint_idx,
                    **_prediction_row(
                        source,
                        forecast,
                        retry_attempted=flags[0],
                        retry_succeeded=flags[1],
                    ),
                }
            )
        inputs = values[0][0].get("input", {}) or {}
        payload["goal_predictions"][str(goal_id)] = {
            "goal_idx": goal_id,
            "goal_text": str(values[0][0].get("goal_text", inputs.get("goal", ""))),
            "checkpoints": goal_rows,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_labeled_states(args.data)
    if args.max_examples:
        rows = rows[: args.max_examples]
    if not rows:
        raise ValueError(f"No labeled states found in {args.data}")

    model, tokenizer = (
        _load_vllm(args) if args.engine == "vllm" else _load_transformers(args)
    )
    prompts = [
        _prompt(row, tokenizer, args.top_k_seen, args.max_model_len)
        for row in rows
    ]
    responses = (
        _generate_vllm(
            model,
            prompts,
            max_tokens=args.max_new_tokens,
            stop="</analysis>",
            batch_size=args.batch_size,
        )
        if args.engine == "vllm"
        else _generate_transformers(
            model, tokenizer, prompts, max_tokens=args.max_new_tokens
        )
    )
    forecasts = [parse_forecast_response(response) for response in responses]
    retry_flags = [(False, False)] * len(rows)

    invalid = [
        index
        for index, forecast in enumerate(forecasts)
        if not forecast_numeric_domain_ok(forecast)
    ]
    if args.retry_invalid and invalid:
        repair_prompts = [
            _repair_prompt(rows[index], args.repair_top_k_seen) for index in invalid
        ]
        repair_responses = (
            _generate_vllm(
                model,
                repair_prompts,
                max_tokens=256,
                stop="</delta_pos_concentration>",
                batch_size=args.batch_size,
            )
            if args.engine == "vllm"
            else _generate_transformers(model, tokenizer, repair_prompts, max_tokens=256)
        )
        retry_flags = list(retry_flags)
        for index, response in zip(invalid, repair_responses):
            repaired = parse_forecast_response(response)
            succeeded = forecast_numeric_domain_ok(repaired)
            retry_flags[index] = (True, succeeded)
            if succeeded:
                forecasts[index] = repaired

    payload = write_predictions(
        rows,
        forecasts,
        output_path=args.output,
        data_path=args.data,
        model_path=args.model_path,
        model_label=args.model_label,
        split=args.split,
        retry_flags=retry_flags,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--engine", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--top-k-seen", type=int, default=15)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--repair-top-k-seen", type=int, default=5)
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        parser.error("--max-examples must be positive")
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prediction_health": payload["prediction_health"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

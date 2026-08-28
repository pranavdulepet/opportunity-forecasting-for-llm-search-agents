"""Evaluate local stopping rules on fixed held-out replay streams."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from opportunity_forecasting.models.distributions import (
    forecast_from_fields,
    forecast_implied_moments,
    forecast_numeric_domain_ok,
)


@dataclass
class GoalResult:
    goal_idx: int
    stopped: bool
    stop_reason: str
    stop_checkpoint_step: int
    final_reward: float
    final_steps: int


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def canonicalize_stop_formulation(value: Optional[str]) -> str:
    formulation = str(value or "policy1").strip().lower()
    if formulation not in {"policy1", "policy2"}:
        raise ValueError(f"Unknown stopping rule: {value!r}")
    return formulation


def _remaining_cost_units(checkpoint_step: int, total_horizon_steps: int) -> float:
    remaining_steps = max(1, int(total_horizon_steps) - int(checkpoint_step) + 1)
    return (float(remaining_steps) + 1.0) / 2.0


def _forecast_moments(checkpoint: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    family = str(checkpoint.get("forecast_family") or "").strip().lower()
    if family.startswith("explicit_"):
        mean = _safe_float(checkpoint.get("expected_delta"))
        std = _safe_float(checkpoint.get("expected_std_delta"))
        if mean is None:
            return None, None, "missing_expected_delta"
        return mean, 0.0 if std is None else std, ""

    forecast = forecast_from_fields(dict(checkpoint))
    if not forecast_numeric_domain_ok(forecast):
        return None, None, "invalid_zoib_parameters"
    mean, std = forecast_implied_moments(forecast)
    if mean is None or std is None:
        return None, None, "invalid_zoib_moments"
    return float(mean), float(std), ""


def score_forecast(
    checkpoint: Mapping[str, Any],
    *,
    stop_formulation: str,
    lambda_risk: float,
    c_step: float,
    checkpoint_step: int,
    total_horizon_steps: int,
) -> Dict[str, Any]:
    formulation = canonicalize_stop_formulation(stop_formulation)
    mean, std, invalid_reason = _forecast_moments(checkpoint)
    cost_units = _remaining_cost_units(checkpoint_step, total_horizon_steps)
    if mean is None or std is None:
        return {
            "score_valid": False,
            "invalid_reason": invalid_reason,
            "cost_units": cost_units,
        }
    score = float(mean) - float(c_step) * cost_units
    if formulation == "policy2":
        score -= float(lambda_risk) * float(std)
    return {
        "score_valid": True,
        "invalid_reason": "",
        "mean_delta": float(mean),
        "std_delta": float(std),
        "cost_units": float(cost_units),
        "score_val": float(score),
    }


def _parse_float_list(specification: str) -> List[float]:
    return [float(token.strip()) for token in specification.split(",") if token.strip()]


def _float_slug(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def _load_cache(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_step(checkpoint: Mapping[str, Any]) -> int:
    return int(checkpoint.get("checkpoint_step", 0) or 0)


def _stream_signature(cache: Mapping[str, Any], goal_id: int) -> Tuple[Tuple[int, float], ...]:
    goal = (cache.get("goal_predictions", {}) or {}).get(str(int(goal_id)), {}) or {}
    return tuple(
        (
            _checkpoint_step(checkpoint),
            round(float(checkpoint.get("best_reward_seen", 0.0) or 0.0), 12),
        )
        for checkpoint in (goal.get("checkpoints", []) or [])
    )


def _assert_identical_cache_streams(caches: Sequence[Mapping[str, Any]]) -> List[int]:
    if not caches:
        return []
    reference_ids = [int(value) for value in caches[0].get("goal_ids", [])]
    reference_set = set(reference_ids)
    reference_predictions = {int(value) for value in (caches[0].get("goal_predictions", {}) or {})}
    if reference_set != reference_predictions:
        raise ValueError("Reference cache goal_ids do not match its prediction keys")
    reference_streams = {
        goal_id: _stream_signature(caches[0], goal_id) for goal_id in reference_ids
    }
    for cache_index, cache in enumerate(caches[1:], start=1):
        cache_ids = {int(value) for value in cache.get("goal_ids", [])}
        prediction_ids = {int(value) for value in (cache.get("goal_predictions", {}) or {})}
        if cache_ids != reference_set or prediction_ids != reference_set:
            raise ValueError(f"Prediction cache {cache_index} does not use the reference goal set")
        for goal_id in reference_ids:
            if _stream_signature(cache, goal_id) != reference_streams[goal_id]:
                raise ValueError(
                    f"Prediction cache {cache_index} has a different reward stream for goal {goal_id}"
                )
    return reference_ids


def _final_outcome(
    checkpoints: Sequence[Mapping[str, Any]],
    *,
    stopped: bool,
    stop_step: int,
    best_so_far: float,
    offline_back_steps: int,
    offline_to_product_steps: int,
) -> Tuple[float, int]:
    if stopped:
        final_steps = (
            max(0, int(stop_step) - 1)
            + max(0, int(offline_back_steps))
            + max(0, int(offline_to_product_steps))
        )
        return float(best_so_far), final_steps
    last = checkpoints[-1]
    final_steps = max(0, int(last.get("checkpoint_step", stop_step) or stop_step))
    return float(last.get("final_reward", best_so_far) or best_so_far), final_steps


def _eval_cached_model(
    *,
    cache_payload: Mapping[str, Any],
    stop_formulation: str,
    lambda_risk: float,
    c_step: float,
    total_horizon_steps: int,
    offline_back_steps: int,
    offline_to_product_steps: int,
) -> Tuple[List[GoalResult], Dict[str, Any], Dict[str, Any]]:
    model_path = str(cache_payload["model_path"])
    model_type = str(cache_payload.get("model_type", "forecast"))
    formulation = canonicalize_stop_formulation(stop_formulation)
    predictions = cache_payload.get("goal_predictions", {}) or {}
    results: List[GoalResult] = []
    total_checkpoints = 0
    valid_scores = 0
    stops_triggered = 0
    invalid_reasons: Counter[str] = Counter()

    for goal_id in [int(value) for value in cache_payload.get("goal_ids", [])]:
        checkpoints = (predictions.get(str(goal_id), {}) or {}).get("checkpoints", []) or []
        if not checkpoints:
            continue
        best_so_far = -1.0
        candidate_available = False
        stopped = False
        stop_reason = "no_stop"
        stop_step = _checkpoint_step(checkpoints[-1])
        for checkpoint in checkpoints:
            step = _checkpoint_step(checkpoint)
            best_so_far = max(best_so_far, float(checkpoint.get("best_reward_seen", 0.0) or 0.0))
            candidate_available = candidate_available or bool(
                checkpoint.get("candidate_available", checkpoint.get("visited_product_page", False))
            )
            score = score_forecast(
                checkpoint,
                stop_formulation=formulation,
                lambda_risk=lambda_risk,
                c_step=c_step,
                checkpoint_step=step,
                total_horizon_steps=total_horizon_steps,
            )
            total_checkpoints += 1
            if score["score_valid"]:
                valid_scores += 1
            else:
                invalid_reasons[str(score.get("invalid_reason") or "unknown")] += 1
            if candidate_available and score["score_valid"] and float(score["score_val"]) <= 0.0:
                stopped = True
                stop_step = step
                stop_reason = f"non_positive_{formulation}_net_utility"
                stops_triggered += 1
                break

        final_reward, final_steps = _final_outcome(
            checkpoints,
            stopped=stopped,
            stop_step=stop_step,
            best_so_far=best_so_far,
            offline_back_steps=offline_back_steps,
            offline_to_product_steps=offline_to_product_steps,
        )
        results.append(
            GoalResult(
                goal_idx=goal_id,
                stopped=stopped,
                stop_reason=stop_reason,
                stop_checkpoint_step=stop_step,
                final_reward=final_reward,
                final_steps=final_steps,
            )
        )

    count = max(1, len(results))
    summary = {
        "model_path": model_path,
        "model_type": model_type,
        "num_goals": len(results),
        "mean_final_reward": sum(result.final_reward for result in results) / count,
        "mean_final_steps": sum(result.final_steps for result in results) / count,
        "stop_rate": sum(result.stopped for result in results) / count,
        "stop_formulation": formulation,
        "lambda_risk": float(lambda_risk),
        "c_step": float(c_step),
        "total_horizon_steps": int(total_horizon_steps),
        "top_k_seen": _safe_int(cache_payload.get("top_k_seen"), 15),
        "offline_back_steps": int(offline_back_steps),
        "offline_to_product_steps": int(offline_to_product_steps),
        "reward_mode": str(cache_payload.get("reward_mode", "")),
        "cost_unit": "replay_step",
    }
    diagnostics = {
        "model_path": model_path,
        "num_goals": len(results),
        "num_checkpoints_total": total_checkpoints,
        "valid_forecast_rate": valid_scores / max(1, total_checkpoints),
        "numeric_parse_rate": valid_scores / max(1, total_checkpoints),
        "valid_score_rate": valid_scores / max(1, total_checkpoints),
        "stops_triggered": stops_triggered,
        "invalid_reason_counts": dict(invalid_reasons),
        "stop_formulation": formulation,
        "lambda_risk": float(lambda_risk),
        "c_step": float(c_step),
    }
    return results, summary, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_cache_paths", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--policy1_c_steps", required=True)
    parser.add_argument("--policy2_c_steps", required=True)
    parser.add_argument("--policy2_lambdas", required=True)
    parser.add_argument("--total_horizon_steps", type=int, default=60)
    parser.add_argument("--offline_back_steps", type=int, default=1)
    parser.add_argument("--offline_to_product_steps", type=int, default=2)
    parser.add_argument("--skip_policy2_lambda_zero", action="store_true")
    args = parser.parse_args()

    cache_paths = [Path(value.strip()) for value in args.prediction_cache_paths.split(",") if value.strip()]
    if not cache_paths:
        raise ValueError("No prediction caches were provided")
    caches = [_load_cache(path) for path in cache_paths]
    goal_ids = _assert_identical_cache_streams(caches)
    if not goal_ids:
        raise ValueError("Prediction caches share no evaluable goals")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy2_lambdas = _parse_float_list(args.policy2_lambdas)
    if args.skip_policy2_lambda_zero:
        policy2_lambdas = [value for value in policy2_lambdas if abs(value) > 1e-12]
    settings: List[Tuple[str, float, float, Path]] = []
    for c_step in _parse_float_list(args.policy1_c_steps):
        settings.append(("policy1", c_step, 1.0, output_dir / f"eval_policy1_c{_float_slug(c_step)}.json"))
    for c_step in _parse_float_list(args.policy2_c_steps):
        for lambda_risk in policy2_lambdas:
            settings.append(
                (
                    "policy2",
                    c_step,
                    lambda_risk,
                    output_dir / f"eval_policy2_c{_float_slug(c_step)}_l{_float_slug(lambda_risk)}.json",
                )
            )

    manifest = output_dir / "sweep.tsv"
    with manifest.open("w", encoding="utf-8") as stream:
        stream.write("policy\tc_step\tlambda_risk\toutput_path\n")
        for policy, c_step, lambda_risk, output_path in settings:
            stream.write(f"{policy}\t{c_step:.12g}\t{lambda_risk:.12g}\t{output_path}\n")

    for index, (policy, c_step, lambda_risk, output_path) in enumerate(settings, start=1):
        payload: Dict[str, Any] = {
            "checkpoint_path": caches[0].get("checkpoint_path"),
            "goal_ids": goal_ids,
            "summaries": [],
            "diagnostics": {},
            "source_prediction_cache_paths": [str(path) for path in cache_paths],
        }
        for cache in caches:
            _, summary, diagnostics = _eval_cached_model(
                cache_payload=cache,
                stop_formulation=policy,
                lambda_risk=lambda_risk,
                c_step=c_step,
                total_horizon_steps=args.total_horizon_steps,
                offline_back_steps=args.offline_back_steps,
                offline_to_product_steps=args.offline_to_product_steps,
            )
            payload["summaries"].append(summary)
            payload["diagnostics"][str(cache["model_path"])] = diagnostics
        _write_json_atomic(output_path, payload)
        print(f"[{index}/{len(settings)}] {output_path.name}", flush=True)
    print(f"Wrote sweep manifest to {manifest}", flush=True)


if __name__ == "__main__":
    main()

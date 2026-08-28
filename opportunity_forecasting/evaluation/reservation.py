"""Build a train-only empirical reservation-value stopping control.

The control bins states by current best reward and trajectory step, estimates the
continuation-gain distribution from training continuations, and exports its mean
and standard deviation on held-out states. For nonnegative gains, the empirical
Pandora reservation equation E[(D-z)+] = cost continues at z=0 exactly when
E[D] exceeds cost, so the ordinary cost sweep evaluates this control directly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _rows(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as rf:
        for line_number, line in enumerate(rf, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            out.append(row)
    return out


def _input(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("input", {})
    return value if isinstance(value, dict) else {}


def _metadata(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("metadata", {})
    return value if isinstance(value, dict) else {}


def _goal_id(row: Mapping[str, Any]) -> int:
    return int(row.get("goal_idx", _input(row).get("goal_idx", 0)) or 0)


def _step(row: Mapping[str, Any]) -> int:
    return int(row.get("checkpoint_step", _input(row).get("checkpoint_step", 0)) or 0)


def _best(row: Mapping[str, Any]) -> float:
    return max(0.0, min(1.0, _safe_float(_input(row).get("best_reward_seen", row.get("best_reward_seen", 0.0)))))


def _visited(row: Mapping[str, Any]) -> bool:
    inp = _input(row)
    meta = _metadata(row)
    if "visited_product_page" in inp:
        return bool(inp.get("visited_product_page"))
    if "visited_product_page" in meta:
        return bool(meta.get("visited_product_page"))
    if bool(inp.get("has_opened_paper", meta.get("has_opened_paper", False))):
        return True
    trigger = str(meta.get("trigger", inp.get("trigger", "")) or "").lower()
    observation = str(inp.get("observation", row.get("observation", "")) or "").lower()
    return trigger in {"product_page", "paper_page", "item_page", "new_paper_page"} or "buy now" in observation or "current_paper_id:" in observation


def _deltas(row: Mapping[str, Any]) -> List[float]:
    return [max(0.0, min(1.0, _safe_float(x))) for x in (row.get("continuation_deltas", []) or [])]


def _bin(row: Mapping[str, Any], *, step_width: int, best_width: float) -> Tuple[int, int]:
    return int(_step(row) // step_width), int(min(0.999999999, _best(row)) // best_width)


def _moments(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    variance = float(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, math.sqrt(max(0.0, variance))


def fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    step_width: int,
    best_width: float,
) -> Dict[str, Any]:
    exact: DefaultDict[Tuple[int, int], List[float]] = defaultdict(list)
    by_step: DefaultDict[int, List[float]] = defaultdict(list)
    global_values: List[float] = []
    for row in rows:
        values = _deltas(row)
        if not values:
            continue
        key = _bin(row, step_width=step_width, best_width=best_width)
        exact[key].extend(values)
        by_step[key[0]].extend(values)
        global_values.extend(values)
    if not global_values:
        raise ValueError("Training data contain no continuation deltas")
    return {"exact": dict(exact), "by_step": dict(by_step), "global": global_values}


def lookup(
    fitted: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    step_width: int,
    best_width: float,
    min_bin_samples: int,
) -> Tuple[List[float], str]:
    step_bin, best_bin = _bin(row, step_width=step_width, best_width=best_width)
    exact = list((fitted.get("exact", {}) or {}).get((step_bin, best_bin), []))
    if len(exact) >= min_bin_samples:
        return exact, "step_and_current_best"
    by_step = list((fitted.get("by_step", {}) or {}).get(step_bin, []))
    if len(by_step) >= min_bin_samples:
        return by_step, "step"
    return list(fitted["global"]), "global"


def build_cache(
    eval_rows: Sequence[Mapping[str, Any]],
    *,
    fitted: Mapping[str, Any],
    output_path: Path,
    split: str,
    model_name: str,
    step_width: int,
    best_width: float,
    min_bin_samples: int,
) -> Dict[str, Any]:
    grouped: DefaultDict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        grouped[_goal_id(row)].append(row)
    predictions: Dict[str, Any] = {}
    fallback_counts: DefaultDict[str, int] = defaultdict(int)
    total = 0
    for gid in sorted(grouped):
        best_so_far = 0.0
        visited_so_far = False
        checkpoint_rows: List[Dict[str, Any]] = []
        ordered = sorted(enumerate(grouped[gid]), key=lambda pair: (_step(pair[1]), pair[0]))
        for checkpoint_idx, (_original_idx, row) in enumerate(ordered):
            samples, source = lookup(
                fitted,
                row,
                step_width=step_width,
                best_width=best_width,
                min_bin_samples=min_bin_samples,
            )
            mean, std = _moments(samples)
            best_so_far = max(best_so_far, _best(row))
            visited_so_far = bool(visited_so_far or _visited(row))
            fallback_counts[source] += 1
            total += 1
            checkpoint_rows.append(
                {
                    "checkpoint_idx": int(checkpoint_idx),
                    "checkpoint_step": int(_step(row)),
                    "candidate_available": bool(visited_so_far),
                    "best_reward_seen": float(best_so_far),
                    "forecast_family": "explicit_empirical_reservation",
                    "expected_delta": float(mean),
                    "expected_std_delta": float(std),
                    "event_probability": float(sum(x > 1e-8 for x in samples) / len(samples)),
                    "empirical_sample_count": int(len(samples)),
                    "empirical_bin_source": source,
                    "forecast_numeric_domain_ok": True,
                }
            )
        first = grouped[gid][0]
        predictions[str(gid)] = {
            "goal_idx": int(gid),
            "goal_text": str(first.get("goal_text", _input(first).get("goal", "")) or ""),
            "checkpoints": checkpoint_rows,
        }
    payload = {
        "checkpoint_path": None,
        "split": str(split),
        "goal_ids_path": None,
        "goal_ids": sorted(grouped),
        "model_path": str(model_name),
        "model_type": "empirical_reservation",
        "engine": "train_only_empirical_control",
        "goal_predictions": predictions,
        "reservation_definition": "E[(D-z)_+]=cost; the z=0 continue boundary is E[D]>cost for nonnegative gains",
        "binning": {"step_width": int(step_width), "best_width": float(best_width), "min_bin_samples": int(min_bin_samples)},
        "fallback_counts": dict(sorted(fallback_counts.items())),
        "prediction_health": {
            "num_goals": len(grouped),
            "num_checkpoints_total": int(total),
            "numeric_parse_rate": 1.0,
            "valid_forecast_rate": 1.0,
            "num_parse_failures": 0,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as wf:
        json.dump(payload, wf, indent=2, sort_keys=True)
        wf.write("\n")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model-name", default="pandora_inspired_empirical_reservation")
    ap.add_argument("--step-width", type=int, default=10)
    ap.add_argument("--best-width", type=float, default=0.2)
    ap.add_argument("--min-bin-samples", type=int, default=60)
    args = ap.parse_args()
    if args.step_width <= 0 or not (0.0 < args.best_width <= 1.0) or args.min_bin_samples <= 0:
        raise ValueError("Invalid binning arguments")
    fitted = fit(_rows(Path(args.train_data)), step_width=int(args.step_width), best_width=float(args.best_width))
    payload = build_cache(
        _rows(Path(args.eval_data)),
        fitted=fitted,
        output_path=Path(args.output_path),
        split=str(args.split),
        model_name=str(args.model_name),
        step_width=int(args.step_width),
        best_width=float(args.best_width),
        min_bin_samples=int(args.min_bin_samples),
    )
    print(json.dumps({"output": str(args.output_path), "fallback_counts": payload["fallback_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Merge goal-disjoint prediction-cache shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping


HEALTH_KEYS = (
    "num_checkpoints_total",
    "num_numeric_parsed",
    "num_valid_forecasts",
    "num_parse_failures",
)


def _read(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _health(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload.get("prediction_health", payload.get("health", {})) or {}


def _merge_health(payloads: List[Dict[str, Any]], num_goals: int) -> Dict[str, Any]:
    merged = {key: 0 for key in HEALTH_KEYS}
    for payload in payloads:
        health = _health(payload)
        for key in HEALTH_KEYS:
            merged[key] += int(health.get(key, 0) or 0)
    merged["num_goals"] = int(num_goals)
    total = max(1, merged["num_checkpoints_total"])
    merged["numeric_parse_rate"] = merged["num_numeric_parsed"] / total
    merged["valid_forecast_rate"] = merged["num_valid_forecasts"] / total
    return merged


def merge(paths: List[Path], output_path: Path) -> Dict[str, Any]:
    if not paths:
        raise ValueError("No shard paths supplied")
    payloads = [_read(path) for path in paths]
    merged = {
        key: value
        for key, value in payloads[0].items()
        if key not in {"goal_ids", "goal_predictions", "health", "prediction_health", "progress"}
    }
    predictions: Dict[str, Any] = {}
    for path, payload in zip(paths, payloads):
        for raw_goal_id in payload.get("goal_ids", []) or []:
            goal_id = str(int(raw_goal_id))
            if goal_id in predictions:
                raise ValueError(f"Goal {goal_id} occurs in more than one shard")
            goal = (payload.get("goal_predictions", {}) or {}).get(goal_id)
            if not isinstance(goal, dict):
                raise ValueError(f"Shard {path} lists goal {goal_id} without predictions")
            predictions[goal_id] = goal

    goal_ids = sorted(int(goal_id) for goal_id in predictions)
    merged["goal_ids"] = goal_ids
    merged["goal_predictions"] = {str(goal_id): predictions[str(goal_id)] for goal_id in goal_ids}
    health_key = "prediction_health" if any("prediction_health" in payload for payload in payloads) else "health"
    merged[health_key] = _merge_health(payloads, len(goal_ids))
    merged["source_shards"] = [str(path) for path in paths]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    merged = merge(list(args.shards), args.output)
    print(f"Merged {len(args.shards)} shards and {len(merged['goal_ids'])} goals into {args.output}")


if __name__ == "__main__":
    main()

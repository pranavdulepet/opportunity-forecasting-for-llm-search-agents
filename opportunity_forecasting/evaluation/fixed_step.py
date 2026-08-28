"""Evaluate a hard stop-after-step control on a fixed replay stream."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from opportunity_forecasting import REPO_ROOT

from opportunity_forecasting.evaluation.allocation import _stop_step_cost


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as rf:
        payload = json.load(rf)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _thresholds(payload: Mapping[str, Any]) -> List[int]:
    values = {
        int(row.get("checkpoint_step", 0) or 0)
        for blob in (payload.get("goal_predictions", {}) or {}).values()
        for row in (blob.get("checkpoints", []) or [])
    }
    if not values:
        raise ValueError("Cache has no checkpoints")
    ordered = sorted(values)
    return [ordered[0] - 1, *ordered, ordered[-1] + 1]


def evaluate(
    payload: Mapping[str, Any],
    *,
    threshold: int,
    offline_back_steps: int,
    offline_to_product_steps: int,
) -> Dict[str, Any]:
    rewards: List[float] = []
    costs: List[float] = []
    stopped = 0
    for gid in [int(x) for x in payload.get("goal_ids", [])]:
        blob = (payload.get("goal_predictions", {}) or {}).get(str(gid), {}) or {}
        rows = list(blob.get("checkpoints", []) or [])
        if not rows:
            continue
        selected = rows[-1]
        did_stop = False
        visited = False
        for row in rows:
            visited = bool(
                visited
                or row.get("candidate_available", row.get("visited_product_page", False))
            )
            if visited and int(row.get("checkpoint_step", 0) or 0) >= int(threshold):
                selected = row
                did_stop = row is not rows[-1]
                break
        rewards.append(float(selected.get("best_reward_seen", 0.0) or 0.0))
        costs.append(
            float(
                _stop_step_cost(
                    selected,
                    offline_back_steps=int(offline_back_steps),
                    offline_to_product_steps=int(offline_to_product_steps),
                )
            )
        )
        stopped += int(did_stop)
    n = len(rewards)
    if not n:
        raise ValueError("No evaluable goals")
    return {
        "series": "Fixed stop-after-step",
        "threshold_step": int(threshold),
        "num_goals": int(n),
        "mean_final_reward": float(sum(rewards) / n),
        "mean_final_steps": float(sum(costs) / n),
        "stop_rate": float(stopped / n),
    }


def _pareto(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    best = -math.inf
    for row in sorted(rows, key=lambda x: (float(x["mean_final_steps"]), -float(x["mean_final_reward"]))):
        reward = float(row["mean_final_reward"])
        if reward > best + 1e-12:
            out.append(dict(row))
            best = reward
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--offline-back-steps", type=int, default=1)
    ap.add_argument("--offline-to-product-steps", type=int, default=2)
    args = ap.parse_args()

    payload = _load(Path(args.cache))
    thresholds = (
        [int(x) for x in str(args.thresholds).split(",") if x.strip()]
        if str(args.thresholds).strip()
        else _thresholds(payload)
    )
    rows = [
        {
            "domain": str(args.domain),
            "split": str(args.split),
            **evaluate(
                payload,
                threshold=threshold,
                offline_back_steps=int(args.offline_back_steps),
                offline_to_product_steps=int(args.offline_to_product_steps),
            ),
        }
        for threshold in thresholds
    ]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload_out = {
        "domain": str(args.domain),
        "split": str(args.split),
        "source_cache": str(args.cache),
        "all_rows": rows,
        "pareto_frontier": _pareto(rows),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as wf:
        json.dump(payload_out, wf, indent=2, sort_keys=True)
        wf.write("\n")
    print(json.dumps({"rows": len(rows), "pareto_rows": len(payload_out["pareto_frontier"])}, sort_keys=True))


if __name__ == "__main__":
    main()

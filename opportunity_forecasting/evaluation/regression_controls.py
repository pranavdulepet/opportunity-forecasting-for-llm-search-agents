"""Train the paper's feature-only and position-only ridge controls."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from opportunity_forecasting import REPO_ROOT


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    if not math.isfinite(val):
        return float(default)
    return float(val)


def _goal_id(row: Mapping[str, Any]) -> int:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    return int(row.get("goal_idx", input_data.get("goal_idx", 0)) or 0)


def _goal_text(row: Mapping[str, Any]) -> str:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    return str(row.get("goal_text", input_data.get("goal", "")) or "")


def _checkpoint_step(row: Mapping[str, Any]) -> int:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    return int(row.get("checkpoint_step", input_data.get("checkpoint_step", 0)) or 0)


def _best_reward(row: Mapping[str, Any]) -> float:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    return _safe_float(input_data.get("best_reward_seen", row.get("best_reward_seen", 0.0)), 0.0)


def _visited_candidate(row: Mapping[str, Any]) -> bool:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
    if "visited_product_page" in input_data:
        return bool(input_data.get("visited_product_page"))
    if "visited_product_page" in metadata:
        return bool(metadata.get("visited_product_page"))
    if bool(input_data.get("has_opened_paper", metadata.get("has_opened_paper", False))):
        return True
    trigger = str(metadata.get("trigger", input_data.get("trigger", "")) or "").lower()
    if trigger in {"product_page", "paper_page", "item_page", "new_paper_page"}:
        return True
    obs = str(input_data.get("observation", row.get("observation", "")) or "").lower()
    return ("buy now" in obs) or ("paper page" in obs) or ("current_paper_id:" in obs)


def _candidate_count(row: Mapping[str, Any]) -> float:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    for key in ("num_seen_papers", "num_seen_products", "num_seen_candidates"):
        if key in input_data:
            return float(input_data.get(key) or 0)
    seen = input_data.get("seen_products", {})
    if isinstance(seen, dict):
        return float(len(seen))
    if isinstance(seen, list):
        return float(len(seen))
    return 0.0


def _opened_count(row: Mapping[str, Any]) -> float:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    for key in ("num_opened_papers", "num_opened_products", "num_opened_candidates"):
        if key in input_data:
            return float(input_data.get(key) or 0)
    opened = input_data.get("opened_paper_ids", [])
    return float(len(opened)) if isinstance(opened, list) else 0.0


def _decision_index(row: Mapping[str, Any]) -> float:
    return float(row.get("_decision_index", 0) or 0)


FEATURE_SETS = ("all", "step_only")


def feature_vector(row: Mapping[str, Any], *, feature_set: str = "all") -> List[float]:
    step = float(_checkpoint_step(row))
    best = float(_best_reward(row))
    seen = float(_candidate_count(row))
    opened = float(_opened_count(row))
    pos = float(_decision_index(row))
    if feature_set == "step_only":
        return [1.0, step / 60.0, pos / 20.0]
    if feature_set != "all":
        raise ValueError(f"Unknown feature_set={feature_set!r}")
    return [
        1.0,
        step / 60.0,
        best,
        seen / 20.0,
        opened / 20.0,
        pos / 20.0,
        best * step / 60.0,
        1.0 if _visited_candidate(row) else 0.0,
    ]


def target_value(row: Mapping[str, Any]) -> float:
    deltas = [max(0.0, min(1.0, _safe_float(x, 0.0))) for x in (row.get("continuation_deltas", []) or [])]
    return float(sum(deltas) / len(deltas)) if deltas else 0.0


def fit_ridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    ridge: float,
    feature_set: str = "all",
) -> np.ndarray:
    x = np.asarray([feature_vector(row, feature_set=feature_set) for row in rows], dtype=np.float64)
    y = np.asarray([target_value(row) for row in rows], dtype=np.float64)
    reg = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def predict_row(row: Mapping[str, Any], weights: np.ndarray, *, feature_set: str = "all") -> float:
    val = float(np.asarray(feature_vector(row, feature_set=feature_set), dtype=np.float64) @ weights)
    return max(0.0, min(1.0, val))


def build_cache(
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: np.ndarray,
    output_path: Path,
    split: str,
    model_name: str,
    feature_set: str = "all",
) -> Dict[str, Any]:
    grouped: Dict[int, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_goal_id(row), []).append(row)
    goal_predictions: Dict[str, Any] = {}
    health = {
        "num_goals": len(grouped),
        "num_checkpoints_total": 0,
        "num_numeric_parsed": 0,
        "num_valid_forecasts": 0,
        "num_parse_failures": 0,
        "num_sigma_recovered_or_clamped": 0,
        "num_failure_examples": 0,
    }
    for gid in sorted(grouped):
        best = 0.0
        seen_candidate = False
        ckpts = []
        for idx, row in enumerate(
            sorted(
                grouped[gid],
                key=lambda item: (
                    _checkpoint_step(item),
                    int(item.get("_source_row", 0) or 0),
                ),
            )
        ):
            best = max(best, _best_reward(row))
            seen_candidate = bool(seen_candidate or _visited_candidate(row))
            pred = predict_row(row, weights, feature_set=feature_set)
            ckpts.append(
                {
                    "checkpoint_idx": int(idx),
                    "checkpoint_step": _checkpoint_step(row),
                    "candidate_available": bool(seen_candidate),
                    "best_reward_seen": float(best),
                    "forecast_family": "explicit_tabular_ridge",
                    "expected_delta": float(pred),
                    "expected_std_delta": 0.0,
                    "forecast_numeric_domain_ok": True,
                }
            )
            health["num_checkpoints_total"] += 1
            health["num_numeric_parsed"] += 1
            health["num_valid_forecasts"] += 1
        goal_predictions[str(gid)] = {"goal_idx": int(gid), "goal_text": _goal_text(grouped[gid][0]), "checkpoints": ckpts}
    payload = {
        "checkpoint_path": None,
        "split": str(split),
        "goal_ids": sorted(grouped),
        "model_path": str(model_name),
        "model_type": "tabular_ridge",
        "engine": "tabular_ridge",
        "feature_set": str(feature_set),
        "health": health,
        "goal_predictions": goal_predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as wf:
        json.dump(payload, wf)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--model-name", default="tabular_ridge")
    ap.add_argument("--weights-path", default="")
    ap.add_argument("--feature-set", default="all", choices=FEATURE_SETS)
    return ap


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                row["_source_row"] = len(rows)
                rows.append(row)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_num}: {exc}") from exc
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_goal_id(row), []).append(row)
    for goal_rows in grouped.values():
        goal_rows.sort(key=lambda row: (_checkpoint_step(row), row["_source_row"]))
        for decision_index, row in enumerate(goal_rows):
            row["_decision_index"] = decision_index
    return rows


def main() -> None:
    args = build_arg_parser().parse_args()

    train_rows = load_jsonl_rows(Path(args.train_data))
    eval_rows = load_jsonl_rows(Path(args.eval_data))
    weights = fit_ridge(
        train_rows,
        ridge=float(args.ridge),
        feature_set=str(args.feature_set),
    )
    if args.weights_path:
        Path(args.weights_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.weights_path).open("w", encoding="utf-8") as wf:
            json.dump(
                {
                    "weights": weights.tolist(),
                    "target": "mean_full_horizon_remaining_upside",
                    "feature_set": str(args.feature_set),
                },
                wf,
                indent=2,
            )
    build_cache(
        eval_rows,
        weights=weights,
        output_path=Path(args.output_path),
        split=str(args.split),
        model_name=str(args.model_name),
        feature_set=str(args.feature_set),
    )
    print(f"Wrote tabular baseline cache to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

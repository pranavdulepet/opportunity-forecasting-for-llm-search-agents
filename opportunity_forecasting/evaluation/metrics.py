"""Summarize forecast quality, feasible-support mass, and position confounding."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple

from scipy.special import betainc

from opportunity_forecasting import REPO_ROOT

from opportunity_forecasting.models.distributions import (
    HURDLE_BETA_FAMILY,
    forecast_from_fields,
    forecast_implied_moments,
    forecast_numeric_domain_ok,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return float(default)
    return result if math.isfinite(result) else float(default)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as rf:
        for line_number, line in enumerate(rf, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield row


def _input(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("input", {})
    return value if isinstance(value, dict) else {}


def _goal_id(row: Mapping[str, Any]) -> int:
    return int(row.get("goal_idx", _input(row).get("goal_idx", 0)) or 0)


def _step(row: Mapping[str, Any]) -> int:
    return int(row.get("checkpoint_step", _input(row).get("checkpoint_step", 0)) or 0)


def _best(row: Mapping[str, Any]) -> float:
    return max(0.0, min(1.0, _safe_float(_input(row).get("best_reward_seen", row.get("best_reward_seen", 0.0)))))


def _deltas(row: Mapping[str, Any]) -> List[float]:
    return [max(0.0, min(1.0, _safe_float(x))) for x in (row.get("continuation_deltas", []) or [])]


def _keyed_label_rows(path: Path) -> Dict[Tuple[int, int, int], Dict[str, Any]]:
    counts: DefaultDict[Tuple[int, int], int] = defaultdict(int)
    out: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        gid = _goal_id(row)
        step = _step(row)
        occurrence = counts[(gid, step)]
        counts[(gid, step)] += 1
        values = _deltas(row)
        if not values:
            continue
        out[(gid, step, occurrence)] = {
            "goal_id": gid,
            "checkpoint_step": step,
            "current_best": _best(row),
            "deltas": values,
            "target_ev": float(sum(values) / len(values)),
        }
    return out


def _aligned(cache: Mapping[str, Any], labels: Mapping[Tuple[int, int, int], Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    cache_keys: set[Tuple[int, int, int]] = set()
    for raw_gid, blob in (cache.get("goal_predictions", {}) or {}).items():
        gid = int(blob.get("goal_idx", raw_gid))
        counts: DefaultDict[int, int] = defaultdict(int)
        for checkpoint in blob.get("checkpoints", []) or []:
            step = int(checkpoint.get("checkpoint_step", 0) or 0)
            occurrence = counts[step]
            counts[step] += 1
            key = (gid, step, occurrence)
            cache_keys.add(key)
            if key not in labels:
                raise ValueError(f"Cache row has no exact label match: {key}")
            out.append((dict(checkpoint), labels[key]))
    if cache_keys != set(labels):
        raise ValueError(
            "Cache and labels do not cover the same rows: "
            f"cache={len(cache_keys)} labels={len(labels)} missing={len(set(labels)-cache_keys)}"
        )
    return out


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _moments(checkpoint: Mapping[str, Any]) -> Tuple[float, float, bool]:
    family = str(checkpoint.get("forecast_family") or "")
    if family.startswith("explicit_"):
        mean = _safe_float(checkpoint.get("expected_delta"), float("nan"))
        std = max(0.0, _safe_float(checkpoint.get("expected_std_delta"), 0.0))
        return mean, std, math.isfinite(mean) and math.isfinite(std)
    forecast = forecast_from_fields(dict(checkpoint))
    if not forecast_numeric_domain_ok(forecast):
        return 0.0, 0.0, False
    mean, std = forecast_implied_moments(forecast)
    return _safe_float(mean), max(0.0, _safe_float(std)), mean is not None and std is not None


def _mass_above_feasible(checkpoint: Mapping[str, Any], feasible: float, mean: float, std: float) -> float:
    family = str(checkpoint.get("forecast_family") or "").lower()
    feasible = max(0.0, min(1.0, float(feasible)))
    if family == "explicit_residual_zoib_remaining_support":
        return 0.0
    if family == "explicit_residual_gaussian":
        sigma = max(std, 1e-8)
        return max(0.0, min(1.0, 1.0 - _normal_cdf((feasible - mean) / sigma)))
    if family == HURDLE_BETA_FAMILY:
        p0 = max(0.0, min(1.0, _safe_float(checkpoint.get("delta_zero_prob"), 1.0)))
        p1 = max(0.0, min(1.0 - p0, _safe_float(checkpoint.get("delta_one_prob"), 0.0)))
        interior = max(0.0, 1.0 - p0 - p1)
        if feasible >= 1.0:
            return 0.0
        if feasible <= 0.0:
            return max(0.0, min(1.0, 1.0 - p0))
        m = max(1e-6, min(1.0 - 1e-6, _safe_float(checkpoint.get("delta_pos_mean"), 0.5)))
        concentration = max(2.0, _safe_float(checkpoint.get("delta_pos_concentration"), 2.0))
        alpha = m * concentration
        beta = (1.0 - m) * concentration
        interior_tail = 1.0 - float(betainc(alpha, beta, feasible))
        return max(0.0, min(1.0, p1 + interior * interior_tail))
    return 1.0 if mean > feasible + 1e-12 else 0.0


def _rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        rank = 1.0 + 0.5 * (position + end)
        for idx in order[position : end + 1]:
            ranks[idx] = rank
        position = end + 1
    return ranks


def _corr(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_var = sum((x - left_mean) ** 2 for x in left)
    right_var = sum((x - right_mean) ** 2 for x in right)
    if left_var <= 0.0 or right_var <= 0.0:
        return 0.0
    return float(sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right)) / math.sqrt(left_var * right_var))


def _step_bin(step: int) -> str:
    start = 10 * (int(step) // 10)
    return f"{start:02d}-{start + 9:02d}" if start < 60 else "60+"


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if bool(row["valid"])]
    if not valid:
        return {
            "num_rows": len(rows),
            "num_valid": 0,
            "metric_support": "valid_numeric_forecasts",
        }
    pred = [float(row["pred_ev"]) for row in valid]
    target = [float(row["target_ev"]) for row in valid]
    steps = [float(row["checkpoint_step"]) for row in valid]
    support = [float(row["mass_above_feasible"]) for row in valid]
    errors = [x - y for x, y in zip(pred, target)]
    by_bin: DefaultDict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        by_bin[str(row["step_bin"])].append(row)
    within: List[Tuple[int, float]] = []
    for bin_rows in by_bin.values():
        if len(bin_rows) < 2:
            continue
        bin_pred = [float(row["pred_ev"]) for row in bin_rows]
        bin_target = [float(row["target_ev"]) for row in bin_rows]
        within.append((len(bin_rows), _corr(_rank(bin_pred), _rank(bin_target))))
    return {
        "num_rows": len(rows),
        "num_valid": len(valid),
        "metric_support": "valid_numeric_forecasts",
        "valid_rate": len(valid) / len(rows),
        "ev_mae": sum(abs(x) for x in errors) / len(errors),
        "ev_rmse": math.sqrt(sum(x * x for x in errors) / len(errors)),
        "mean_pred_ev": sum(pred) / len(pred),
        "mean_target_ev": sum(target) / len(target),
        "spearman_ev": _corr(_rank(pred), _rank(target)),
        "pred_step_pearson": _corr(pred, steps),
        "target_step_pearson": _corr(target, steps),
        "within_step_bin_spearman": (sum(count * value for count, value in within) / sum(count for count, _value in within)) if within else 0.0,
        "mean_mass_above_feasible": sum(support) / len(support),
        "fraction_rows_mass_above_feasible_gt_1e-3": sum(value > 1e-3 for value in support) / len(support),
        "target_support_violation_rate": sum(bool(row["target_support_violation"]) for row in valid) / len(valid),
    }


def analyze(label: str, cache_path: Path, label_rows: Mapping[Tuple[int, int, int], Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with cache_path.open("r", encoding="utf-8") as rf:
        cache = json.load(rf)
    details: List[Dict[str, Any]] = []
    for checkpoint, target in _aligned(cache, label_rows):
        mean, std, valid = _moments(checkpoint)
        feasible = max(0.0, 1.0 - float(target["current_best"]))
        deltas = list(target["deltas"])
        details.append(
            {
                "model_label": label,
                "goal_id": int(target["goal_id"]),
                "checkpoint_step": int(target["checkpoint_step"]),
                "step_bin": _step_bin(int(target["checkpoint_step"])),
                "current_best": float(target["current_best"]),
                "feasible_gain": float(feasible),
                "pred_ev": float(mean),
                "pred_std": float(std),
                "target_ev": float(target["target_ev"]),
                "valid": bool(valid),
                "mass_above_feasible": _mass_above_feasible(checkpoint, feasible, mean, std) if valid else float("nan"),
                "target_support_violation": any(delta > feasible + 1e-8 for delta in deltas),
                "forecast_family": str(checkpoint.get("forecast_family") or ""),
            }
        )
    overall = {"model_label": label, "group": "overall", **_summary(details)}
    return overall, details


def _parse_cache(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected LABEL=PATH, got {raw!r}")
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cache", action="append", required=True, help="Repeat LABEL=PATH")
    ap.add_argument("--domain", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-summary-csv", required=True)
    ap.add_argument("--output-detail-csv", default="")
    args = ap.parse_args()

    labels = _keyed_label_rows(Path(args.labels))
    summaries: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for raw in args.cache:
        label, path = _parse_cache(raw)
        summary, model_details = analyze(label, path, labels)
        summary.update({"domain": str(args.domain), "split": str(args.split), "cache": str(path)})
        summaries.append(summary)
        details.extend(model_details)
        by_bin: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in model_details:
            by_bin[str(row["step_bin"])].append(row)
        for step_bin, bin_rows in sorted(by_bin.items()):
            summaries.append(
                {
                    "model_label": label,
                    "group": f"step_{step_bin}",
                    "domain": str(args.domain),
                    "split": str(args.split),
                    "cache": str(path),
                    **_summary(bin_rows),
                }
            )

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as wf:
        json.dump({"domain": args.domain, "split": args.split, "labels": args.labels, "summaries": summaries}, wf, indent=2, sort_keys=True)
        wf.write("\n")
    output_csv = Path(args.output_summary_csv)
    fields = list(dict.fromkeys(key for row in summaries for key in row))
    with output_csv.open("w", encoding="utf-8", newline="") as wf:
        writer = csv.DictWriter(wf, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    if args.output_detail_csv:
        detail_path = Path(args.output_detail_csv)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        with detail_path.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)
    print(json.dumps({"models": len(args.cache), "label_rows": len(labels), "detail_rows": len(details)}, sort_keys=True))


if __name__ == "__main__":
    main()

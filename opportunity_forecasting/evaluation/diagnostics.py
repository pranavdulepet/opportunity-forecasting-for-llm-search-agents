"""Evaluate cached forecasts against labeled continuation deltas."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from opportunity_forecasting import REPO_ROOT

from opportunity_forecasting.models.distributions import (
    HURDLE_BETA_BETA_EPS,
    HURDLE_BETA_FAMILY,
    HURDLE_BETA_ZERO_TOL,
    MAX_REWARD_DELTA,
    forecast_brier_event_calibration,
    forecast_continuous_ranked_probability_score,
    forecast_from_fields,
    forecast_implied_moments,
    forecast_mean_log_likelihood,
    forecast_numeric_domain_ok,
)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        val = float(x)
    except Exception:
        return float(default)
    return float(val) if math.isfinite(val) else float(default)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as rf:
        return json.load(rf)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as rf:
        for line in rf:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _goal_id(row: Mapping[str, Any]) -> int:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    return int(row.get("goal_idx", input_data.get("goal_idx", 0)) or 0)


def _checkpoint_step(row: Mapping[str, Any]) -> int:
    input_data = row.get("input", {}) if isinstance(row.get("input"), dict) else {}
    return int(row.get("checkpoint_step", input_data.get("checkpoint_step", 0)) or 0)


def _clean_deltas(raw: Sequence[Any]) -> List[float]:
    return [max(0.0, min(float(MAX_REWARD_DELTA), _safe_float(x, 0.0))) for x in (raw or [])]


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * float(z) * float(z)) / math.sqrt(2.0 * math.pi)


def _normal_nll(x: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), float(HURDLE_BETA_BETA_EPS))
    z = (float(x) - float(mu)) / sigma
    return float(0.5 * math.log(2.0 * math.pi) + math.log(sigma) + 0.5 * z * z)


def _normal_crps(x: float, mu: float, sigma: float) -> float:
    sigma = max(float(sigma), float(HURDLE_BETA_BETA_EPS))
    z = (float(x) - float(mu)) / sigma
    return float(sigma * (z * (2.0 * _normal_cdf(z) - 1.0) + 2.0 * _normal_pdf(z) - 1.0 / math.sqrt(math.pi)))


def _normal_threshold_brier(deltas: Sequence[float], mu: float, sigma: float, thresholds: Sequence[float]) -> float:
    sigma = max(float(sigma), float(HURDLE_BETA_BETA_EPS))
    vals: List[float] = []
    for threshold in thresholds:
        prob = 1.0 - _normal_cdf((float(threshold) - float(mu)) / sigma)
        prob = max(0.0, min(1.0, prob))
        for delta in deltas:
            target = 1.0 if float(delta) > float(threshold) else 0.0
            vals.append((prob - target) ** 2)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _label_rows_by_goal(labels: Path) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row_idx, row in enumerate(_iter_jsonl(labels)):
        deltas = _clean_deltas(row.get("continuation_deltas", []) or [])
        if not deltas:
            continue
        grouped[_goal_id(row)].append(
            {
                "row_idx": int(row_idx),
                "checkpoint_step": _checkpoint_step(row),
                "deltas": deltas,
                "target_ev": float(sum(deltas) / len(deltas)),
                "target_event": float(sum(1 for d in deltas if d > HURDLE_BETA_ZERO_TOL) / len(deltas)),
            }
        )
    for rows in grouped.values():
        rows.sort(key=lambda r: (int(r["checkpoint_step"]), int(r["row_idx"])))
    return dict(grouped)


def _cache_rows_by_goal(cache: Mapping[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for raw_gid, blob in (cache.get("goal_predictions", {}) or {}).items():
        gid = int(blob.get("goal_idx", raw_gid))
        rows = list(blob.get("checkpoints", []) or [])
        rows.sort(key=lambda r: (int(r.get("checkpoint_idx", 0) or 0), int(r.get("checkpoint_step", 0) or 0)))
        grouped[gid] = rows
    return grouped


def _align(cache: Mapping[str, Any], labels: Path) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    labels_by_goal = _label_rows_by_goal(labels)
    cache_by_goal = _cache_rows_by_goal(cache)
    aligned: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for gid, ckpts in cache_by_goal.items():
        label_rows = labels_by_goal.get(gid, [])
        if len(label_rows) == len(ckpts):
            aligned.extend(zip(ckpts, label_rows))
            continue
        by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
        counts: Dict[int, int] = defaultdict(int)
        for row in label_rows:
            step = int(row["checkpoint_step"])
            occ = counts[step]
            counts[step] += 1
            by_key[(step, occ)] = row
        counts.clear()
        for ckpt in ckpts:
            step = int(ckpt.get("checkpoint_step", 0) or 0)
            occ = counts[step]
            counts[step] += 1
            row = by_key.get((step, occ))
            if row is not None:
                aligned.append((ckpt, row))
    if not aligned:
        raise ValueError(f"No cache rows aligned with {labels}")
    return aligned


def _forecast_metrics(ckpt: Mapping[str, Any], label: Mapping[str, Any], thresholds: Sequence[float]) -> Dict[str, float]:
    family = str(ckpt.get("forecast_family") or "").strip().lower()
    explicit = family.startswith("explicit_")
    is_gaussian = family == "explicit_residual_gaussian"
    deltas = list(label["deltas"])
    target_ev = float(label["target_ev"])
    target_event = float(label["target_event"])
    if explicit:
        pred_ev = _safe_float(ckpt.get("expected_delta"), 0.0)
        pred_event = _safe_float(ckpt.get("event_probability"), float("nan"))
        if is_gaussian and not math.isfinite(pred_event):
            sigma = max(_safe_float(ckpt.get("expected_std_delta"), 0.0), float(HURDLE_BETA_BETA_EPS))
            pred_event = 1.0 - _normal_cdf((float(HURDLE_BETA_ZERO_TOL) - pred_ev) / sigma)
        nll = float("nan")
        crps = float("nan")
        threshold_brier = float("nan")
        if is_gaussian:
            sigma = max(_safe_float(ckpt.get("expected_std_delta"), 0.0), float(HURDLE_BETA_BETA_EPS))
            nll = float(sum(_normal_nll(delta, pred_ev, sigma) for delta in deltas) / len(deltas))
            crps = float(sum(_normal_crps(delta, pred_ev, sigma) for delta in deltas) / len(deltas))
            threshold_brier = _normal_threshold_brier(deltas, pred_ev, sigma, thresholds)
        return {
            "valid": 1.0,
            "pred_ev": pred_ev,
            "target_ev": target_ev,
            "pred_event": max(0.0, min(1.0, pred_event)) if math.isfinite(pred_event) else float("nan"),
            "target_event": target_event,
            "ev_abs_error": abs(pred_ev - target_ev),
            "ev_sq_error": (pred_ev - target_ev) ** 2,
            "event_brier": (pred_event - target_event) ** 2 if math.isfinite(pred_event) else float("nan"),
            "nll": nll,
            "crps": crps,
            "threshold_brier": threshold_brier,
        }
    forecast = forecast_from_fields(dict(ckpt))
    if not forecast_numeric_domain_ok(forecast):
        return {
            "valid": 0.0,
            "pred_ev": 0.0,
            "target_ev": target_ev,
            "pred_event": 0.0,
            "target_event": target_event,
            "ev_abs_error": target_ev,
            "ev_sq_error": target_ev ** 2,
            "event_brier": target_event ** 2,
            "nll": float("nan"),
            "crps": float("nan"),
            "threshold_brier": float("nan"),
        }
    pred_ev, _pred_std = forecast_implied_moments(forecast)
    pred_ev = 0.0 if pred_ev is None else float(pred_ev)
    pred_event = 1.0 - _safe_float(ckpt.get("delta_zero_prob"), 1.0)
    return {
        "valid": 1.0,
        "pred_ev": pred_ev,
        "target_ev": target_ev,
        "pred_event": max(0.0, min(1.0, pred_event)),
        "target_event": target_event,
        "ev_abs_error": abs(pred_ev - target_ev),
        "ev_sq_error": (pred_ev - target_ev) ** 2,
        "event_brier": (pred_event - target_event) ** 2,
        "nll": -forecast_mean_log_likelihood(deltas, forecast),
        "crps": forecast_continuous_ranked_probability_score(deltas, forecast, num_grid_points=41),
        "threshold_brier": forecast_brier_event_calibration(deltas, forecast, thresholds),
    }


def _rank(vals: Sequence[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def _corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy))


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, Any]:
    valid_rows = [row for row in rows if float(row.get("valid", 0.0)) > 0.5]
    finite = lambda key: [
        float(row[key])
        for row in valid_rows
        if key in row and math.isfinite(float(row[key]))
    ]
    pred_ev = finite("pred_ev")
    target_ev = finite("target_ev")
    pred_event = finite("pred_event")
    target_event = finite("target_event")
    out: Dict[str, Any] = {
        "num_rows": len(rows),
        "num_valid": len(valid_rows),
        "metric_support": "valid_numeric_forecasts",
        "valid_rate": sum(float(r["valid"]) for r in rows) / max(1, len(rows)),
        "ev_mae": sum(finite("ev_abs_error")) / max(1, len(finite("ev_abs_error"))),
        "ev_rmse": math.sqrt(sum(finite("ev_sq_error")) / max(1, len(finite("ev_sq_error")))),
        "event_brier": sum(finite("event_brier")) / max(1, len(finite("event_brier"))),
        "mean_pred_ev": sum(pred_ev) / max(1, len(pred_ev)),
        "mean_target_ev": sum(target_ev) / max(1, len(target_ev)),
        "mean_pred_event": sum(pred_event) / max(1, len(pred_event)),
        "mean_target_event": sum(target_event) / max(1, len(target_event)),
        "spearman_ev": _corr(_rank(pred_ev), _rank(target_ev)) if len(pred_ev) == len(target_ev) else 0.0,
    }
    for key in ("nll", "crps", "threshold_brier"):
        vals = finite(key)
        out[key] = sum(vals) / len(vals) if vals else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--model-label", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", default="")
    ap.add_argument("--thresholds", default="0.01,0.05,0.1")
    args = ap.parse_args()

    thresholds = [float(x) for x in str(args.thresholds).split(",") if str(x).strip()]
    cache = _read_json(Path(args.cache))
    aligned = _align(cache, Path(args.labels))
    rows = [_forecast_metrics(ckpt, label, thresholds) for ckpt, label in aligned]
    summary = summarize(rows)
    summary.update({"model_label": str(args.model_label), "domain": str(args.domain), "split": str(args.split), "cache": str(args.cache), "labels": str(args.labels)})

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as wf:
        json.dump(summary, wf, indent=2, sort_keys=True)
    if args.output_csv:
        out_csv = Path(args.output_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=sorted(summary))
            writer.writeheader()
            writer.writerow(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

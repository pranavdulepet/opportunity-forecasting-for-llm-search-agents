"""Select stop-policy operating points on dev and report matched test rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


KEY_FIELDS = ("model_label", "policy", "c_step", "lambda_risk")


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_label",
        "selection_rule",
        "dev_target_steps",
        "selected_policy",
        "selected_c_step",
        "selected_lambda_risk",
        "dev_selection_metric",
        "dev_mean_final_reward",
        "dev_mean_final_steps",
        "dev_stop_rate",
        "test_mean_final_reward",
        "test_mean_final_steps",
        "test_stop_rate",
        "test_utility_proxy",
        "matched_test_row",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_md(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| model | rule | dev step budget | selected policy | c_step | lambda | dev reward | test reward | test steps | test stop rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model_label} | {selection_rule} | {dev_target_steps} | {selected_policy} | {selected_c_step} | "
            "{selected_lambda_risk} | {dev_selection_metric} | {test_mean_final_reward} | "
            "{test_mean_final_steps} | {test_stop_rate} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _setting_key(row: dict) -> Tuple[str, str, str, str]:
    return tuple(str(row.get(k, "")) for k in KEY_FIELDS)


def _common_support_target(grouped: Dict[str, List[dict]], fraction: float) -> float:
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("target_step_fraction must be in [0, 1]")
    spans = [
        (
            min(_float(row, "mean_final_steps") for row in rows),
            max(_float(row, "mean_final_steps") for row in rows),
        )
        for rows in grouped.values()
        if rows
    ]
    if not spans:
        raise ValueError("No development rows")
    common_min = max(span[0] for span in spans)
    common_max = min(span[1] for span in spans)
    if common_min > common_max:
        raise ValueError(f"Methods have no common dev step support: {common_min} > {common_max}")
    return common_min + float(fraction) * (common_max - common_min)


def select_rows(
    dev_rows: Sequence[dict],
    test_rows: Sequence[dict],
    *,
    metric: str,
    target_step_fraction: float = 0.5,
) -> List[dict]:
    test_by_key: Dict[Tuple[str, str, str, str], dict] = {_setting_key(row): row for row in test_rows}
    grouped: Dict[str, List[dict]] = {}
    for row in dev_rows:
        grouped.setdefault(str(row.get("model_label", "")), []).append(row)

    target_steps = None
    if metric == "budget_feasible_reward":
        target_steps = _common_support_target(grouped, target_step_fraction)

    out: List[dict] = []
    for model_label, rows in sorted(grouped.items()):
        if target_steps is not None:
            feasible = [row for row in rows if _float(row, "mean_final_steps") <= target_steps + 1e-9]
            if not feasible:
                feasible = [min(rows, key=lambda row: _float(row, "mean_final_steps"))]
            ranked = sorted(
                feasible,
                key=lambda r: (
                    _float(r, "mean_final_reward"),
                    _float(r, "valid_score_rate"),
                    -_float(r, "mean_final_steps"),
                ),
                reverse=True,
            )
            selection_metric = "mean_final_reward"
            selection_rule = "dev_common_support_midpoint_budget"
        else:
            ranked = sorted(
                rows,
                key=lambda r: (
                    _float(r, metric),
                    _float(r, "mean_final_reward"),
                    -_float(r, "mean_final_steps"),
                    _float(r, "valid_score_rate"),
                ),
                reverse=True,
            )
            selection_metric = metric
            selection_rule = f"dev_max_{metric}"
        if not ranked:
            continue
        dev = ranked[0]
        test = test_by_key.get(_setting_key(dev))
        out.append(
            {
                "model_label": model_label,
                "selection_rule": selection_rule,
                "dev_target_steps": f"{target_steps:.3f}" if target_steps is not None else "",
                "selected_policy": str(dev.get("policy", "")),
                "selected_c_step": str(dev.get("c_step", "")),
                "selected_lambda_risk": str(dev.get("lambda_risk", "")),
                "dev_selection_metric": f"{_float(dev, selection_metric):.6f}",
                "dev_mean_final_reward": f"{_float(dev, 'mean_final_reward'):.6f}",
                "dev_mean_final_steps": f"{_float(dev, 'mean_final_steps'):.3f}",
                "dev_stop_rate": f"{_float(dev, 'stop_rate'):.3f}",
                "test_mean_final_reward": f"{_float(test or {}, 'mean_final_reward'):.6f}" if test else "",
                "test_mean_final_steps": f"{_float(test or {}, 'mean_final_steps'):.3f}" if test else "",
                "test_stop_rate": f"{_float(test or {}, 'stop_rate'):.3f}" if test else "",
                "test_utility_proxy": f"{_float(test or {}, 'utility_proxy'):.6f}" if test else "",
                "matched_test_row": "1" if test else "0",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--output-md", default="")
    ap.add_argument("--metric", default="budget_feasible_reward", choices=("budget_feasible_reward", "utility_proxy", "mean_final_reward"))
    ap.add_argument("--target-step-fraction", type=float, default=0.5)
    args = ap.parse_args()

    rows = select_rows(
        _read_csv(Path(args.dev_csv)),
        _read_csv(Path(args.test_csv)),
        metric=str(args.metric),
        target_step_fraction=float(args.target_step_fraction),
    )
    _write_csv(Path(args.output_csv), rows)
    if args.output_md:
        _write_md(Path(args.output_md), rows)
    print(f"Wrote {len(rows)} dev-selected operating points to {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()

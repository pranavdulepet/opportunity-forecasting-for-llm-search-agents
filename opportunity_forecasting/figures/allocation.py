"""Render the paper's budget-allocation and absolute-reward frontiers."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from opportunity_forecasting import REPO_ROOT

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opportunity-forecasting-matplotlib-budget")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/opportunity-forecasting-budget-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from opportunity_forecasting.figures.style import BASE_PROMPT, GRID, REGRESSION_HEAD, apply_paper_style


DEFAULT_SOURCE_ROOT = REPO_ROOT / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_outputs" / "figures"
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def portable_source(path: Path, source_root: Path) -> str:
    return path.resolve().relative_to(source_root.resolve()).as_posix()


DISPLAY = {
    "Fixed-replay hindsight upper bound": "Hindsight bound",
    "Original prompted baseline": "Base Prompt",
    "Raw residual ZOIB": "ZOIB Regression",
    "Support-corrected residual ZOIB": "Support ZOIB",
    "Scalar residual": "Scalar Head",
    "Gaussian residual": "Gaussian Head",
    "Feature-only ridge": "Feature ridge",
    "Step-only ridge": "Step ridge",
    "Heuristic: earliest step": "Early-step",
    "Pandora-inspired empirical reservation": "Reservation",
}

STYLES: Mapping[str, Mapping[str, object]] = {
    "Hindsight bound": {"color": "#111827", "marker": "", "linestyle": "--"},
    "Base Prompt": {"color": BASE_PROMPT, "marker": "o", "linestyle": "-"},
    "ZOIB Regression": {"color": REGRESSION_HEAD, "marker": "s", "linestyle": "-"},
    "Support ZOIB": {"color": "#0891B2", "marker": "X", "linestyle": "-"},
    "Scalar Head": {"color": "#0F766E", "marker": "^", "linestyle": "--"},
    "Gaussian Head": {"color": "#7C3AED", "marker": "D", "linestyle": "-."},
    "Feature ridge": {"color": "#6B7280", "marker": "v", "linestyle": ":"},
    "Step ridge": {"color": "#8A7D00", "marker": "P", "linestyle": ":"},
    "Early-step": {"color": "#8A7D00", "marker": "P", "linestyle": ":"},
    "Reservation": {"color": "#B4537A", "marker": "h", "linestyle": "--"},
}

VARIANTS = {
    "head_ablation": ("Base Prompt", "ZOIB Regression", "Scalar Head", "Gaussian Head"),
    "paper_controls": (
        "Base Prompt",
        "ZOIB Regression",
        "Support ZOIB",
        "Scalar Head",
        "Gaussian Head",
        "Feature ridge",
        "Early-step",
        "Reservation",
    ),
}

ABSOLUTE_PAPER_CONTROLS = ("Hindsight bound",) + VARIANTS["paper_controls"]

CURVE_LINEWIDTH = 1.2
CURVE_MARKERSIZE = 2.0
LEGEND_LINEWIDTH = 1.3
LEGEND_MARKERSIZE = 2.8

DOMAINS = {"webshop": "WebShop", "paper_search": "Paper Search"}


def _read_curves(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_label = str(row.get("series") or row.get("model_label") or "")
            label = DISPLAY.get(raw_label, raw_label if raw_label in STYLES else None)
            if not label:
                continue
            reward = row.get("reward_gap_to_oracle", row.get("mean_final_reward"))
            grouped[label].append(
                {
                    "mean_final_steps": float(row["mean_final_steps"]),
                    "reward_gap_to_oracle": float(reward),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["mean_final_steps"]))
    return dict(grouped)


def _read_absolute_curves(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_label = str(row.get("series") or row.get("model_label") or "")
            label = DISPLAY.get(raw_label, raw_label if raw_label in STYLES else None)
            if not label:
                continue
            grouped[label].append(
                {
                    "mean_final_steps": float(row["mean_final_steps"]),
                    "mean_final_reward": float(row["mean_final_reward"]),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["mean_final_steps"]))
    return dict(grouped)


def _interpolate(rows: Sequence[dict], x: float, key: str) -> float:
    xs = [float(row["mean_final_steps"]) for row in rows]
    if x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        raise ValueError(f"x={x} is outside [{xs[0]}, {xs[-1]}]")
    pos = bisect.bisect_left(xs, x)
    if pos < len(xs) and abs(xs[pos] - x) <= 1e-9:
        return float(rows[pos][key])
    if pos == 0 or pos == len(xs):
        return float(rows[max(0, min(pos, len(rows) - 1))][key])
    left, right = rows[pos - 1], rows[pos]
    x0, x1 = float(left["mean_final_steps"]), float(right["mean_final_steps"])
    weight = (x - x0) / (x1 - x0)
    return float(left[key]) + weight * (float(right[key]) - float(left[key]))


def _common_grid(
    grouped: Mapping[str, Sequence[dict]], series: Sequence[str]
) -> tuple[float, float, List[float]]:
    missing = [label for label in series if not grouped.get(label)]
    if missing:
        raise ValueError(f"Missing budget series: {missing}")
    lo = max(float(grouped[label][0]["mean_final_steps"]) for label in series)
    hi = min(float(grouped[label][-1]["mean_final_steps"]) for label in series)
    if hi <= lo:
        raise ValueError(f"Budget series have no common x support: [{lo}, {hi}]")
    grid = {lo, hi}
    for label in series:
        grid.update(
            float(row["mean_final_steps"])
            for row in grouped[label]
            if lo <= float(row["mean_final_steps"]) <= hi
        )
    return lo, hi, sorted(grid)


def _normalized_auc(xs: Sequence[float], ys: Sequence[float]) -> float:
    area = sum(
        (x1 - x0) * (y0 + y1) / 2.0
        for x0, x1, y0, y1 in zip(xs, xs[1:], ys, ys[1:])
    )
    return area / (xs[-1] - xs[0])


def _write_rows(
    path: Path,
    grouped: Mapping[str, Sequence[dict]],
    series: Sequence[str],
    grid: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_label", "mean_final_steps", "reward_gap_to_oracle"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in series:
            for x in grid:
                writer.writerow(
                    {
                        "model_label": label,
                        "mean_final_steps": x,
                        "reward_gap_to_oracle": _interpolate(
                            grouped[label], x, "reward_gap_to_oracle"
                        ),
                    }
                )


def _render(
    *,
    domain: str,
    grouped: Mapping[str, Sequence[dict]],
    series: Sequence[str],
    out_base: Path,
) -> dict:
    lo, hi, grid = _common_grid(grouped, series)
    plotted = {
        label: [
            _interpolate(grouped[label], x, "reward_gap_to_oracle") for x in grid
        ]
        for label in series
    }
    y_min = min(value for values in plotted.values() for value in values)
    y_pad = max(0.003, 0.06 * abs(y_min))

    apply_paper_style(plt, base_font_size=9.0)
    fig, ax = plt.subplots(figsize=(4.35, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axhline(0.0, color="#343A40", linestyle="--", linewidth=1.2)

    aucs: Dict[str, float] = {}
    for label in series:
        style = STYLES[label]
        ys = plotted[label]
        ax.plot(
            grid,
            ys,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=CURVE_LINEWIDTH,
            marker=str(style["marker"]),
            markersize=CURVE_MARKERSIZE,
            markevery=max(1, len(grid) // 8),
        )
        aucs[label] = _normalized_auc(grid, ys)

    ax.set_title(DOMAINS[domain])
    ax.set_xlabel("Mean final replay steps")
    ax.set_ylabel("Reward gap to hindsight bound")
    ax.set_xlim(lo, hi)
    ax.set_ylim(y_min - y_pad, y_pad)
    ax.grid(True, color=GRID, alpha=0.32, linewidth=0.45)
    handles = [
        Line2D(
            [0],
            [0],
            color=str(STYLES[label]["color"]),
            linestyle=str(STYLES[label]["linestyle"]),
            marker=str(STYLES[label]["marker"]),
            linewidth=LEGEND_LINEWIDTH,
            markersize=LEGEND_MARKERSIZE,
            label=label,
        )
        for label in series
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.3,
    )
    fig.subplots_adjust(bottom=0.31)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_base.with_suffix(".pdf"),
        facecolor="white",
        transparent=False,
        metadata=PDF_METADATA,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=300, facecolor="white", transparent=False)
    plt.close(fig)
    _write_rows(
        out_base.with_name(out_base.name + "_rows.csv"), grouped, series, grid
    )
    return {
        "x_min": lo,
        "x_max": hi,
        "normalized_gap_auc": aucs,
    }


def _render_absolute(
    *,
    domain: str,
    grouped: Mapping[str, Sequence[dict]],
    series: Sequence[str],
    out_base: Path,
) -> dict:
    lo, hi, grid = _common_grid(grouped, series)
    plotted = {
        label: [_interpolate(grouped[label], x, "mean_final_reward") for x in grid]
        for label in series
    }
    y_min = min(value for values in plotted.values() for value in values)
    y_max = max(value for values in plotted.values() for value in values)
    y_pad = max(0.003, 0.06 * (y_max - y_min))

    apply_paper_style(plt, base_font_size=9.0)
    fig, ax = plt.subplots(figsize=(4.35, 3.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for label in series:
        style = STYLES[label]
        marker = str(style["marker"])
        ax.plot(
            grid,
            plotted[label],
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=CURVE_LINEWIDTH,
            marker=marker or None,
            markersize=CURVE_MARKERSIZE,
            markevery=max(1, len(grid) // 8),
        )

    ax.set_title(f"{DOMAINS[domain]}: Absolute reward")
    ax.set_xlabel("Mean final replay steps")
    ax.set_ylabel("Mean final reward")
    ax.set_xlim(lo, hi)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(True, color=GRID, alpha=0.32, linewidth=0.45)
    handles = [
        Line2D(
            [0],
            [0],
            color=str(STYLES[label]["color"]),
            linestyle=str(STYLES[label]["linestyle"]),
            marker=str(STYLES[label]["marker"]) or None,
            linewidth=LEGEND_LINEWIDTH,
            markersize=LEGEND_MARKERSIZE,
            label=label,
        )
        for label in series
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        columnspacing=0.8,
        handlelength=2.0,
    )
    fig.subplots_adjust(bottom=0.32)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_base.with_suffix(".pdf"),
        facecolor="white",
        transparent=False,
        metadata=PDF_METADATA,
    )
    fig.savefig(out_base.with_suffix(".png"), dpi=300, facecolor="white", transparent=False)
    with out_base.with_name(out_base.name + "_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("model_label", "mean_final_steps", "mean_final_reward"),
            lineterminator="\n",
        )
        writer.writeheader()
        for label in series:
            for x in grid:
                writer.writerow(
                    {
                        "model_label": label,
                        "mean_final_steps": x,
                        "mean_final_reward": _interpolate(
                            grouped[label], x, "mean_final_reward"
                        ),
                    }
                )
    return {"x_min": lo, "x_max": hi, "series": list(series)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant", choices=tuple(VARIANTS), default="paper_controls"
    )
    parser.add_argument(
        "--kind",
        choices=("budgeted-expansion", "absolute-reward", "all"),
        default="all",
    )
    args = parser.parse_args()

    summary = {"source_split": "test", "domains": {}}
    for domain in DOMAINS:
        summary["domains"][domain] = {}
        if args.kind in {"budgeted-expansion", "all"}:
            source = args.source_root / "budgeted_expansion" / f"{domain}.csv"
            summary["domains"][domain]["budgeted_expansion"] = {
                "source": portable_source(source, args.source_root),
                **_render(
                    domain=domain,
                    grouped=_read_curves(source),
                    series=VARIANTS[args.variant],
                    out_base=args.output_dir
                    / "budgeted_expansion"
                    / f"{domain}_budgeted_expansion_frontier",
                ),
            }
        if args.kind in {"absolute-reward", "all"}:
            source = args.source_root / "absolute_reward" / f"{domain}.csv"
            absolute_record = {
                "source": portable_source(source, args.source_root),
                **_render_absolute(
                    domain=domain,
                    grouped=_read_absolute_curves(source),
                    series=ABSOLUTE_PAPER_CONTROLS,
                    out_base=args.output_dir
                    / "absolute_reward"
                    / f"{domain}_absolute_reward_frontier",
                ),
            }
            if args.kind == "absolute-reward":
                summary["domains"][domain] = absolute_record
            else:
                summary["domains"][domain]["absolute_reward"] = absolute_record
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

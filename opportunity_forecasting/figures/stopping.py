"""Render the paper's stopping-policy frontiers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from opportunity_forecasting import REPO_ROOT

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opportunity-forecasting-matplotlib-frontiers")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/opportunity-forecasting-frontier-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from opportunity_forecasting.figures.style import BASE_PROMPT, GRID, REGRESSION_HEAD, apply_paper_style


DEFAULT_SOURCE_ROOT = REPO_ROOT / "results" / "stopping"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "paper_outputs" / "figures" / "stopping"
PDF_METADATA = {"CreationDate": None, "ModDate": None}


def portable_source(path: Path, source_root: Path) -> str:
    return path.resolve().relative_to(source_root.resolve()).as_posix()


DISPLAY = {
    "Original prompted baseline": "Base Prompt",
    "Raw residual ZOIB": "ZOIB Regression",
    "Support-corrected residual ZOIB": "Support ZOIB",
    "Scalar residual": "Scalar Head",
    "Gaussian residual": "Gaussian Head",
    "Feature-only ridge": "Feature ridge",
    "Step-only ridge": "Step ridge",
    "Pandora-inspired empirical reservation": "Reservation",
}

STYLES: Mapping[str, Mapping[str, object]] = {
    "Base Prompt": {"color": BASE_PROMPT, "marker": "o", "linestyle": "-"},
    "ZOIB Regression": {"color": REGRESSION_HEAD, "marker": "s", "linestyle": "-"},
    "Support ZOIB": {"color": "#0891B2", "marker": "X", "linestyle": "-"},
    "Scalar Head": {"color": "#0F766E", "marker": "^", "linestyle": "--"},
    "Gaussian Head": {"color": "#7C3AED", "marker": "D", "linestyle": "-."},
    "Feature ridge": {"color": "#6B7280", "marker": "v", "linestyle": ":"},
    "Step ridge": {"color": "#8A7D00", "marker": "P", "linestyle": ":"},
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
        "Step ridge",
        "Reservation",
    ),
}

DOMAINS = {
    "webshop": "WebShop",
    "paper_search": "Paper Search",
}

CURVE_LINEWIDTH = 1.2
CURVE_MARKERSIZE = 2.0
LEGEND_LINEWIDTH = 1.3
LEGEND_MARKERSIZE = 2.8


def read_frontiers(path: Path) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            raw_label = str(row.get("model_label", ""))
            label = DISPLAY.get(raw_label, raw_label if raw_label in STYLES else None)
            if not label:
                continue
            grouped[label].append(
                {
                    "mean_final_steps": float(row["mean_final_steps"]),
                    "mean_final_reward": float(row["mean_final_reward"]),
                    "policy": str(row.get("policy", "")),
                    "c_step": str(row.get("c_step", "")),
                    "lambda_risk": str(row.get("lambda_risk", "")),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["mean_final_steps"]))
    return dict(grouped)


def shared_endpoints(grouped: Mapping[str, Sequence[dict]], series: Sequence[str]) -> tuple[float, float]:
    missing = [label for label in series if not grouped.get(label)]
    if missing:
        raise ValueError(f"Missing frontier series: {missing}")
    minima = {round(float(grouped[label][0]["mean_final_steps"]), 9) for label in series}
    maxima = {round(float(grouped[label][-1]["mean_final_steps"]), 9) for label in series}
    if len(minima) != 1 or len(maxima) != 1:
        raise ValueError(f"Frontiers do not share x endpoints: min={sorted(minima)} max={sorted(maxima)}")
    return next(iter(minima)), next(iter(maxima))


def normalized_auc(rows: Sequence[dict]) -> float:
    lo = float(rows[0]["mean_final_steps"])
    hi = float(rows[-1]["mean_final_steps"])
    if hi <= lo:
        raise ValueError("Frontier has empty x support")
    area = 0.0
    for left, right in zip(rows, rows[1:]):
        x0, y0 = float(left["mean_final_steps"]), float(left["mean_final_reward"])
        x1, y1 = float(right["mean_final_steps"]), float(right["mean_final_reward"])
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area / (hi - lo)


def write_rows(path: Path, grouped: Mapping[str, Sequence[dict]], series: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("model_label", "mean_final_steps", "mean_final_reward", "policy", "c_step", "lambda_risk")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for label in series:
            for row in grouped[label]:
                writer.writerow({"model_label": label, **row})


def render(domain: str, grouped: Mapping[str, Sequence[dict]], series: Sequence[str], out_base: Path) -> dict:
    lo, hi = shared_endpoints(grouped, series)
    rewards = [float(row["mean_final_reward"]) for label in series for row in grouped[label]]
    y_lo, y_hi = min(rewards), max(rewards)
    y_pad = max(0.003, 0.06 * (y_hi - y_lo))

    apply_paper_style(plt, base_font_size=9.2)
    fig, ax = plt.subplots(figsize=(4.25, 3.45))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    aucs: Dict[str, float] = {}
    for label in series:
        rows = grouped[label]
        style = STYLES[label]
        xs = [float(row["mean_final_steps"]) for row in rows]
        ys = [float(row["mean_final_reward"]) for row in rows]
        ax.plot(
            xs,
            ys,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            linewidth=CURVE_LINEWIDTH,
            marker=str(style["marker"]),
            markersize=CURVE_MARKERSIZE,
            markevery=max(1, len(xs) // 8),
        )
        aucs[label] = normalized_auc(rows)

    ax.set_title(DOMAINS[domain])
    ax.set_xlabel("Mean final replay steps")
    ax.set_ylabel("Mean final reward")
    ax.set_xlim(lo, hi)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
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
        bbox_to_anchor=(0.5, -0.19),
        ncol=2,
        columnspacing=1.0,
        handlelength=2.3,
    )
    fig.subplots_adjust(bottom=0.30)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_base.with_suffix(".pdf"),
        facecolor="white",
        edgecolor="none",
        transparent=False,
        metadata=PDF_METADATA,
    )
    fig.savefig(
        out_base.with_suffix(".png"),
        dpi=300,
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)
    write_rows(out_base.with_name(out_base.name + "_rows.csv"), grouped, series)
    return {"x_min": lo, "x_max": hi, "normalized_auc": aucs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant", choices=tuple(VARIANTS), default="paper_controls"
    )
    args = parser.parse_args()

    summary = {
        "source_split": "test",
        "domains": {},
    }
    for domain in DOMAINS:
        source = args.source_root / f"{domain}.csv"
        out_base = args.output_dir / f"{domain}_stopping_frontier"
        summary["domains"][domain] = {
            "source": portable_source(source, args.source_root),
            **render(domain, read_frontiers(source), VARIANTS[args.variant], out_base),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

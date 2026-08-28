"""Build and render finite-lookahead gains on held-out replay streams."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


from opportunity_forecasting import REPO_ROOT

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opportunity-forecasting-search-value")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/opportunity-forecasting-search-value")

from opportunity_forecasting.evaluation.allocation import (
    CacheSpec,
    _load_cache,
    extract_threads,
    shared_goal_ids,
    validate_shared_reward_streams,
)
from opportunity_forecasting.figures.style import HEATMAP_COLORS, apply_paper_style


DEFAULT_SUMMARY = REPO_ROOT / "results" / "search_value" / "summary.csv"
DEFAULT_METADATA = REPO_ROOT / "results" / "search_value" / "meta.json"
PDF_METADATA = {"CreationDate": None, "ModDate": None}


@dataclass(frozen=True)
class GainCell:
    domain: str
    step_bin: str
    step_min_exclusive: int
    step_max_inclusive: int
    additional_steps: str
    n: int
    mean_gain: float
    positive_rate: float


def parse_cache(raw: str) -> tuple[str, CacheSpec]:
    if "=" not in raw or ":" not in raw.split("=", 1)[0]:
        raise ValueError("--cache expects DOMAIN:LABEL=PATH")
    left, path = raw.split("=", 1)
    domain, label = left.split(":", 1)
    return domain.strip(), CacheSpec(label.strip(), Path(path.strip()))


def reward_streams(
    cache_specs: Sequence[CacheSpec],
    *,
    offline_back_steps: int = 1,
    offline_to_product_steps: int = 1,
) -> list[list[object]]:
    if not cache_specs:
        raise ValueError("At least one cache is required")
    caches = [_load_cache(spec.path) for spec in cache_specs]
    goal_ids = shared_goal_ids(caches)
    streams = {
        spec.label: extract_threads(
            cache,
            goal_ids,
            offline_back_steps=offline_back_steps,
            offline_to_product_steps=offline_to_product_steps,
        )
        for spec, cache in zip(cache_specs, caches)
    }
    validated = validate_shared_reward_streams(streams)
    return validated[cache_specs[0].label]


def gain_with_budget(stream: Sequence[object], index: int, budget: int | None) -> float:
    current_reward = float(stream[index].best_reward_seen)
    current_step = int(stream[index].checkpoint_step)
    future_rewards = [
        float(checkpoint.best_reward_seen)
        for checkpoint in stream[index:]
        if budget is None
        or int(checkpoint.checkpoint_step) <= current_step + int(budget)
    ]
    return max(0.0, max(future_rewards, default=current_reward) - current_reward)


def summarize_domain(
    domain: str,
    streams: Sequence[Sequence[object]],
    *,
    step_bins: Sequence[int],
    budgets: Sequence[int | None],
    positive_eps: float = 1e-4,
) -> list[GainCell]:
    decision_points = [
        (int(checkpoint.checkpoint_step), stream, index)
        for stream in streams
        for index, checkpoint in enumerate(stream)
    ]
    rows: list[GainCell] = []
    for low, high in zip(step_bins[:-1], step_bins[1:]):
        bucket = [
            (stream, index)
            for step, stream, index in decision_points
            if int(low) < step <= int(high)
        ]
        if not bucket:
            continue
        for budget in budgets:
            gains = [gain_with_budget(stream, index, budget) for stream, index in bucket]
            rows.append(
                GainCell(
                    domain=domain,
                    step_bin=f"{int(low) + 1}-{int(high)}",
                    step_min_exclusive=int(low),
                    step_max_inclusive=int(high),
                    additional_steps="to H" if budget is None else f"+{int(budget)}",
                    n=len(gains),
                    mean_gain=float(np.mean(gains)),
                    positive_rate=float(np.mean([gain > positive_eps for gain in gains])),
                )
            )
    return rows


def build_summary(
    cache_specs: Mapping[str, Sequence[CacheSpec]],
    *,
    step_bins: Sequence[int] = (0, 10, 20, 30, 40, 50, 60),
    budgets: Sequence[int | None] = (5, 10, 20, None),
    positive_eps: float = 1e-4,
) -> tuple[list[GainCell], dict[str, object]]:
    rows: list[GainCell] = []
    domains: dict[str, object] = {}
    for domain, specs in cache_specs.items():
        streams = reward_streams(specs)
        domains[domain] = {
            "caches": {spec.label: str(spec.path) for spec in specs},
            "num_streams": len(streams),
            "num_decision_points": sum(len(stream) for stream in streams),
        }
        rows.extend(
            summarize_domain(
                domain,
                streams,
                step_bins=step_bins,
                budgets=budgets,
                positive_eps=positive_eps,
            )
        )
    metadata = {
        "domains": domains,
        "step_bins": list(step_bins),
        "budgets": ["remaining" if budget is None else budget for budget in budgets],
        "positive_eps": positive_eps,
    }
    return rows, metadata


def write_summary(path: Path, rows: Sequence[GainCell]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def load_summary(path: Path) -> list[GainCell]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            GainCell(
                domain=row["domain"],
                step_bin=row["step_bin"],
                step_min_exclusive=int(row["step_min_exclusive"]),
                step_max_inclusive=int(row["step_max_inclusive"]),
                additional_steps=row["additional_steps"],
                n=int(row["n"]),
                mean_gain=float(row["mean_gain"]),
                positive_rate=float(row["positive_rate"]),
            )
            for row in csv.DictReader(handle)
        ]


def plot(output_dir: Path, rows: Sequence[GainCell]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    apply_paper_style(plt, base_font_size=9.2)
    domains = list(dict.fromkeys(row.domain for row in rows))
    budget_labels = list(dict.fromkeys(row.additional_steps for row in rows))
    step_bins = list(dict.fromkeys(row.step_bin for row in rows))
    maximum = max((row.mean_gain for row in rows), default=1.0)
    color_map = LinearSegmentedColormap.from_list("paper_search_value", HEATMAP_COLORS)
    figure, axes = plt.subplots(
        1,
        len(domains),
        figsize=(7.2, 2.75),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for axis, domain in zip(axes[0], domains):
        matrix = np.full((len(step_bins), len(budget_labels)), np.nan, dtype=float)
        for row in rows:
            if row.domain == domain:
                matrix[
                    step_bins.index(row.step_bin),
                    budget_labels.index(row.additional_steps),
                ] = row.mean_gain
        image = axis.imshow(matrix, aspect="auto", cmap=color_map, vmin=0.0, vmax=maximum)
        axis.set_title(domain)
        axis.set_xticks(np.arange(len(budget_labels)), budget_labels)
        axis.set_yticks(np.arange(len(step_bins)), step_bins)
        axis.set_xlabel("Additional search steps")
        axis.set_ylabel("Decision-point step")
        axis.grid(False)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7.4,
                        color="white" if value > 0.55 * maximum else "#111827",
                    )
    if image is not None:
        colorbar = figure.colorbar(image, ax=list(axes[0]), shrink=0.88, pad=0.02)
        colorbar.set_label("Mean reward gain")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "search_value_profile.png", dpi=300)
    figure.savefig(output_dir / "search_value_profile.pdf", metadata=PDF_METADATA)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", default=[])
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "paper_outputs" / "figures" / "search_value",
    )
    parser.add_argument("--step-bins", default="0,10,20,30,40,50,60")
    parser.add_argument("--budgets", default="5,10,20,remaining")
    parser.add_argument("--positive-eps", type=float, default=1e-4)
    args = parser.parse_args()

    if args.cache:
        cache_specs: dict[str, list[CacheSpec]] = {}
        for raw in args.cache:
            domain, spec = parse_cache(raw)
            cache_specs.setdefault(domain, []).append(spec)
        step_bins = [int(value) for value in args.step_bins.split(",")]
        budgets = [
            None if value.strip().lower() in {"remaining", "full", "h"} else int(value)
            for value in args.budgets.split(",")
        ]
        rows, metadata = build_summary(
            cache_specs,
            step_bins=step_bins,
            budgets=budgets,
            positive_eps=args.positive_eps,
        )
        write_summary(args.output_dir / "search_value_profile_summary.csv", rows)
        (args.output_dir / "search_value_profile_meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        rows = load_summary(args.summary)
        if not args.metadata.is_file():
            raise FileNotFoundError(args.metadata)
    plot(args.output_dir, rows)
    print(f"Wrote search-value profile to {args.output_dir}")


if __name__ == "__main__":
    main()

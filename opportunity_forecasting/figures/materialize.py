"""Summarize a completed experiment into the paper's result tables."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

from opportunity_forecasting import REPO_ROOT

from opportunity_forecasting.figures.allocation import (
    ABSOLUTE_PAPER_CONTROLS,
    VARIANTS as BUDGET_VARIANTS,
    _common_grid,
    _interpolate,
    _read_absolute_curves,
    _read_curves,
    _write_rows,
)
from opportunity_forecasting.figures.stopping import (
    VARIANTS as STOPPING_VARIANTS,
    read_frontiers,
    shared_endpoints,
    write_rows,
)
from opportunity_forecasting.figures.search_value import (
    CacheSpec,
    build_summary,
    write_summary,
)


DOMAINS = ("webshop", "paper_search")
DOMAIN_NAMES = {"webshop": "WebShop", "paper_search": "Paper Search"}


def write_absolute_rows(
    path: Path,
    grouped: Mapping[str, Sequence[dict]],
    series: Sequence[str],
    grid: Sequence[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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


def materialize_search_value(
    domains: Sequence[str],
    *,
    run_root: Path,
    output_root: Path,
) -> dict:
    cache_specs = {}
    for domain in domains:
        cache_root = run_root / "predictions" / domain
        cache_specs[DOMAIN_NAMES[domain]] = (
            CacheSpec("Base Prompt", cache_root / "prompt_original_test.json"),
            CacheSpec("Raw ZOIB", cache_root / "zoib_raw_test.json"),
        )
    rows, raw_metadata = build_summary(cache_specs)
    output_dir = output_root / "search_value"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "meta.json"
    write_summary(summary_path, rows)
    metadata = {
        "budgets": raw_metadata["budgets"],
        "domains": {
            domain: {
                "num_decision_points": values["num_decision_points"],
                "num_streams": values["num_streams"],
                "source": "canonical label-consistent held-out replay caches",
            }
            for domain, values in raw_metadata["domains"].items()
        },
        "positive_eps": raw_metadata["positive_eps"],
        "step_bins": raw_metadata["step_bins"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"summary": str(summary_path), "metadata": str(metadata_path)}


def materialize_domain(
    domain: str,
    *,
    run_root: Path,
    output_root: Path,
) -> dict:
    test_root = run_root / "evaluations" / domain / "test"
    stopping_source = (
        test_root / "stopping" / "summary" / "pareto_frontier_by_model.csv"
    )
    budget_root = test_root / "budgeted_expansion_raw_priority"
    budget_source = budget_root / "budgeted_expansion_oracle_gap_curves.csv"
    absolute_source = budget_root / "budgeted_expansion_curves.csv"
    cost_source = (
        test_root
        / "budgeted_expansion_cost_normalized"
        / "budgeted_expansion_curves.csv"
    )

    stopping_grouped = read_frontiers(stopping_source)
    stopping_series = STOPPING_VARIANTS["paper_controls"]
    shared_endpoints(stopping_grouped, stopping_series)
    stopping_output = output_root / "stopping" / f"{domain}.csv"
    write_rows(stopping_output, stopping_grouped, stopping_series)

    budget_grouped = _read_curves(budget_source)
    budget_series = BUDGET_VARIANTS["paper_controls"]
    _, _, budget_grid = _common_grid(budget_grouped, budget_series)
    budget_output = output_root / "budgeted_expansion" / f"{domain}.csv"
    _write_rows(budget_output, budget_grouped, budget_series, budget_grid)

    absolute_grouped = _read_absolute_curves(absolute_source)
    _, _, absolute_grid = _common_grid(
        absolute_grouped, ABSOLUTE_PAPER_CONTROLS
    )
    absolute_output = output_root / "absolute_reward" / f"{domain}.csv"
    write_absolute_rows(
        absolute_output,
        absolute_grouped,
        ABSOLUTE_PAPER_CONTROLS,
        absolute_grid,
    )

    cost_output = output_root / "cost_normalized_allocation" / f"{domain}.csv"
    cost_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cost_source, cost_output)

    outputs = {
        "stopping": stopping_output,
        "budgeted_expansion": budget_output,
        "absolute_reward": absolute_output,
        "cost_normalized_allocation": cost_output,
    }
    result = {
        "sources": {
            "stopping": str(stopping_source),
            "budgeted_expansion": str(budget_source),
            "absolute_reward": str(absolute_source),
            "cost_normalized_allocation": str(cost_source),
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domain", action="append", choices=DOMAINS)
    args = parser.parse_args()
    domains = tuple(args.domain or DOMAINS)
    report = {
        "run_root": str(args.run_root),
        "output_root": str(args.output_root),
        "domains": {
            domain: materialize_domain(
                domain,
                run_root=args.run_root,
                output_root=args.output_root,
            )
            for domain in domains
        },
    }
    report["search_value"] = materialize_search_value(
        domains,
        run_root=args.run_root,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

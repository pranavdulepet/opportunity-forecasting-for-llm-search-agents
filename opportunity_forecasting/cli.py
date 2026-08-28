"""Command-line interface for the paper's data, experiments, and outputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from opportunity_forecasting import REPO_ROOT


PYTHON = sys.executable
DEFAULT_RESULTS = REPO_ROOT / "results"
DEFAULT_OUTPUTS = REPO_ROOT / "paper_outputs"

FIGURE_MODULES = {
    "overview": "opportunity_forecasting.figures.overview",
    "budgeted-expansion": "opportunity_forecasting.figures.allocation",
    "absolute-reward": "opportunity_forecasting.figures.allocation",
    "stopping": "opportunity_forecasting.figures.stopping",
    "search-value": "opportunity_forecasting.figures.search_value",
}


def run(module: str, *args: str) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/opportunity-forecasting-matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/opportunity-forecasting-cache")
    subprocess.run(
        [PYTHON, "-m", module, *args],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )


def render_figure(name: str, *, source_root: Path, output_root: Path) -> None:
    if name == "overview":
        arguments = ("--output-dir", str(output_root / "overview"))
    elif name == "budgeted-expansion":
        arguments = (
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_root),
            "--kind",
            "budgeted-expansion",
        )
    elif name == "absolute-reward":
        arguments = (
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_root),
            "--kind",
            "absolute-reward",
        )
    elif name == "stopping":
        arguments = (
            "--source-root",
            str(source_root / "stopping"),
            "--output-dir",
            str(output_root / "stopping"),
        )
    elif name == "search-value":
        arguments = (
            "--summary",
            str(source_root / "search_value" / "summary.csv"),
            "--metadata",
            str(source_root / "search_value" / "meta.json"),
            "--output-dir",
            str(output_root / "search_value"),
        )
    else:
        raise ValueError(name)
    run(FIGURE_MODULES[name], *arguments)


def render_all_figures(*, source_root: Path, output_root: Path) -> None:
    for name in FIGURE_MODULES:
        render_figure(name, source_root=source_root, output_root=output_root)


def write_tables(*, source_root: Path, output_root: Path, run_root: Path | None) -> None:
    if run_root is not None:
        run(
            "opportunity_forecasting.figures.tables",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output_root),
        )
        return

    source = source_root / "tables"
    output_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for table in sorted(source.glob("*.csv")):
        destination = output_root / table.name
        shutil.copyfile(table, destination)
        copied[table.name] = str(destination)
    if not copied:
        raise FileNotFoundError(f"No table sources found under {source}")
    print(json.dumps(copied, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper", help="Generate every paper figure and table.")
    paper.add_argument("--source-root", type=Path, default=DEFAULT_RESULTS)
    paper.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUTS)

    figure = subparsers.add_parser("figure", help="Generate one paper figure.")
    figure.add_argument("name", choices=tuple(FIGURE_MODULES))
    figure.add_argument("--source-root", type=Path, default=DEFAULT_RESULTS)
    figure.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUTS / "figures",
    )

    figures = subparsers.add_parser("figures", help="Generate every paper figure.")
    figures.add_argument("--source-root", type=Path, default=DEFAULT_RESULTS)
    figures.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUTS / "figures",
    )

    tables = subparsers.add_parser("tables", help="Generate every paper table source.")
    tables.add_argument("--source-root", type=Path, default=DEFAULT_RESULTS)
    tables.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUTS / "tables")
    tables.add_argument(
        "--run-root",
        type=Path,
        help="Recompute tables from an experiment run instead of copying paper values.",
    )

    summarize = subparsers.add_parser(
        "summarize",
        help="Convert prediction and policy outputs into figure-ready result tables.",
    )
    summarize.add_argument("--run-root", type=Path, required=True)
    summarize.add_argument("--output-root", type=Path)

    model = subparsers.add_parser(
        "prepare-model",
        help="Download and verify the pinned Qwen backbone.",
    )
    model.add_argument("--output-dir", type=Path)
    model.add_argument("--snapshot-dir", type=Path)
    model.add_argument("--link-snapshot", action="store_true")
    model.add_argument("--local-files-only", action="store_true")
    model.add_argument("--verify-only", action="store_true")

    webshop = subparsers.add_parser(
        "prepare-webshop",
        help="Install WebShop and optionally download and index its product data.",
    )
    webshop.add_argument("--download-data", action="store_true")
    webshop.add_argument("--build-index", action="store_true")
    webshop.add_argument("--index-threads", type=int, default=8)

    validate_data = subparsers.add_parser(
        "validate-data",
        help="Validate canonical labels, source trajectories, splits, and checksums.",
    )
    validate_data.add_argument("--include-environment", action="store_true")
    validate_data.add_argument("--webshop-asset", type=Path)

    download_data = subparsers.add_parser(
        "download-data",
        help="Download and verify the canonical training and evaluation data.",
    )
    download_data.add_argument("--archive", type=Path)

    experiment = subparsers.add_parser(
        "experiment",
        help="Run or inspect the paper experiment pipeline.",
    )
    experiment.add_argument(
        "--start-from",
        choices=("labels", "checkpoints", "predictions"),
        default="labels",
    )
    experiment.add_argument("--run-root", type=Path, default=REPO_ROOT / "runs" / "paper")
    experiment.add_argument("--conda-env")
    experiment.add_argument("--gpu-partition")
    experiment.add_argument("--cpu-partition")
    experiment.add_argument("--train-a100-partition")
    experiment.add_argument("--train-rtx-partition")
    experiment.add_argument("--scheduler", choices=("slurm", "local"), default="slurm")
    experiment.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "paper":
        render_all_figures(
            source_root=args.source_root,
            output_root=args.output_root / "figures",
        )
        write_tables(
            source_root=args.source_root,
            output_root=args.output_root / "tables",
            run_root=None,
        )
    elif args.command == "figure":
        render_figure(args.name, source_root=args.source_root, output_root=args.output_root)
    elif args.command == "figures":
        render_all_figures(source_root=args.source_root, output_root=args.output_root)
    elif args.command == "tables":
        write_tables(
            source_root=args.source_root,
            output_root=args.output_root,
            run_root=args.run_root,
        )
    elif args.command == "summarize":
        output_root = args.output_root or args.run_root / "results"
        run(
            "opportunity_forecasting.figures.materialize",
            "--run-root",
            str(args.run_root),
            "--output-root",
            str(output_root),
        )
    elif args.command == "prepare-model":
        command = []
        if args.output_dir:
            command.extend(["--output-dir", str(args.output_dir)])
        if args.snapshot_dir:
            command.extend(["--snapshot-dir", str(args.snapshot_dir)])
        if args.link_snapshot:
            command.append("--link-snapshot")
        if args.local_files_only:
            command.append("--local-files-only")
        if args.verify_only:
            command.append("--verify-only")
        run("opportunity_forecasting.models.prepare", *command)
    elif args.command == "prepare-webshop":
        command = []
        if args.download_data:
            command.append("--download-webshop-data")
        if args.build_index:
            command.extend(["--build-webshop-index", "--index-threads", str(args.index_threads)])
        run("opportunity_forecasting.data.setup", *command)
    elif args.command == "validate-data":
        command = []
        if args.include_environment:
            command.append("--include-environment")
        if args.webshop_asset:
            command.extend(["--webshop-asset", str(args.webshop_asset)])
        run("opportunity_forecasting.data.validate", *command)
    elif args.command == "download-data":
        command = []
        if args.archive:
            command.extend(["--archive", str(args.archive)])
        run("opportunity_forecasting.data.download", *command)
    elif args.command == "experiment":
        if args.scheduler == "local":
            if args.start_from != "predictions":
                parser.error(
                    "--scheduler local evaluates predictions already present in "
                    "--run-root; use --start-from predictions"
                )
            command = ["--run-root", str(args.run_root)]
            if args.conda_env:
                command.extend(["--conda-env", args.conda_env])
            if args.dry_run:
                command.append("--dry-run")
            run("opportunity_forecasting.evaluation.local_pipeline", *command)
            return

        command = [
            "--start-from",
            args.start_from,
            "--run-root",
            str(args.run_root),
            "--conda-env",
            args.conda_env or "opportunity-forecasting",
        ]
        if args.dry_run:
            command.append("--dry-run")
        for flag, value in (
            ("--gpu-partition", args.gpu_partition),
            ("--cpu-partition", args.cpu_partition),
            ("--train-a100-partition", args.train_a100_partition),
            ("--train-rtx-partition", args.train_rtx_partition),
        ):
            if value:
                command.extend([flag, value])
        run("opportunity_forecasting.experiments.submit", *command)


if __name__ == "__main__":
    main()

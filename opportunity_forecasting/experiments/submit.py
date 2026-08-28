"""Submit the paper's training, prediction, and evaluation pipeline to Slurm."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


from opportunity_forecasting import REPO_ROOT as REPO

from opportunity_forecasting.data.validate import validate_inputs
from opportunity_forecasting.manifest import (
    PAPER_CONFIG,
    load_manifest,
    resolve_artifact_path,
    resolve_backbone_path,
)

DEFAULT_MANIFEST = PAPER_CONFIG
DEFAULT_RUN_ROOT = REPO / "runs" / "paper"
DEFAULT_CONDA_ENV = "opportunity-forecasting"
DEFAULT_TRAIN_A100_PARTITION = os.environ.get("OPPORTUNITY_TRAIN_A100_PARTITION", "")
DEFAULT_TRAIN_RTX_PARTITION = os.environ.get("OPPORTUNITY_TRAIN_RTX_PARTITION", "")
DEFAULT_GPU_PARTITION = os.environ.get("OPPORTUNITY_GPU_PARTITION", "")
DEFAULT_CPU_PARTITION = os.environ.get("OPPORTUNITY_CPU_PARTITION", "")
PREDICTION_SHARDS = 2

POLICY_C_STEPS = (
    "-0.1,-0.05,-0.02,-0.01,-0.005,-0.001,-0.0005,-0.0001,-0.00005,0,"
    "0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.0075,0.01,0.015,"
    "0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.3,0.5,0.75,1.0"
)
POLICY_LAMBDAS = (
    "0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,"
    "0.18,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.75,0.9,1.0,1.25,"
    "1.5,1.75,2.0,2.5,3.0,4.0,5.0"
)

MODEL_DISPLAY = {
    "prompt_original": "Base Prompt",
    "zoib_raw": "ZOIB Regression",
    "zoib_remaining": "Support ZOIB",
    "residual_scalar": "Scalar Head",
    "residual_gaussian": "Gaussian Head",
    "ridge_all": "Feature-only ridge",
    "ridge_step": "Step-only ridge",
    "empirical_reservation": "Pandora-inspired empirical reservation",
}
PROMPT_RETRY_DISPLAY = "Base Prompt + retry"
LEARNED_MODELS = ("zoib_raw", "zoib_remaining", "residual_scalar", "residual_gaussian")
ALL_METHODS = tuple(MODEL_DISPLAY)


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _python(conda_env: str) -> str:
    expanded = Path(conda_env).expanduser()
    option = "-p" if expanded.is_absolute() or "/" in conda_env else "-n"
    value = expanded if option == "-p" else conda_env
    return f"conda run --no-capture-output {option} {shell_quote(value)} python"


def _training_partition(domain_key: str, a100_partition: str, rtx_partition: str) -> str:
    return a100_partition if domain_key == "webshop" else rtx_partition


def resolve_partitions(args: argparse.Namespace) -> Dict[str, str]:
    gpu = str(args.gpu_partition or "").strip()
    cpu = str(args.cpu_partition or "").strip()
    if not args.dry_run and (not gpu or not cpu):
        raise ValueError(
            "Real submission requires --gpu-partition and --cpu-partition "
            "or OPPORTUNITY_GPU_PARTITION and OPPORTUNITY_CPU_PARTITION"
        )
    gpu = gpu or "GPU_PARTITION"
    cpu = cpu or "CPU_PARTITION"
    return {
        "gpu": gpu,
        "cpu": cpu,
        "train_a100": str(args.train_a100_partition or "").strip() or gpu,
        "train_rtx": str(args.train_rtx_partition or "").strip() or gpu,
    }


@dataclass(frozen=True)
class Domain:
    key: str
    title: str
    reward_mode: str
    labels: Mapping[str, Path]
    checkpoints: Mapping[str, Path]
    heuristics: Tuple[str, ...]
    paper_assets: Mapping[str, Path]
    webshop_asset: Path | None


def load_domains(
    manifest: Mapping[str, Any], *, webshop_asset_override: Path | None = None
) -> List[Domain]:
    out: List[Domain] = []
    for key, title, heuristics in (
        ("webshop", "WebShop", ("step_early", "low_current_best", "few_seen", "low_stagnation")),
        ("paper_search", "Paper Search", ("step_early", "low_current_best", "few_seen", "low_stagnation", "retrieved_rank")),
    ):
        spec = manifest["domains"][key]
        labels = {
            split: resolve_artifact_path(manifest, file_spec)
            for split, file_spec in spec["labels"].items()
        }
        checkpoints = {
            split: resolve_artifact_path(manifest, file_spec)
            for split, file_spec in spec["checkpoints"].items()
        }
        paper_assets = {
            name: resolve_artifact_path(manifest, file_spec)
            for name, file_spec in spec.get("environment_assets", {}).items()
        }
        webshop_asset = (
            resolve_artifact_path(
                manifest,
                spec["environment_asset"],
                override=webshop_asset_override,
            )
            if "environment_asset" in spec
            else None
        )
        out.append(
            Domain(
                key=key,
                title=title,
                reward_mode=str(spec["reward_mode"]),
                labels=labels,
                checkpoints=checkpoints,
                heuristics=heuristics,
                paper_assets=paper_assets,
                webshop_asset=webshop_asset,
            )
        )
    return out


def stage_paper_search_inputs(run_root: Path, domains: Sequence[Domain]) -> Dict[str, Path]:
    staged: Dict[str, Path] = {}
    suffixes = {"queries": "queries.jsonl", "corpus": "corpus.jsonl", "qrels": "qrels.tsv"}
    for domain in domains:
        if domain.key != "paper_search":
            continue
        input_dir = run_root / "inputs" / domain.key
        input_dir.mkdir(parents=True, exist_ok=True)
        for name, filename in suffixes.items():
            source = domain.paper_assets[name]
            destination = input_dir / filename
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise ValueError(f"Staged input points to the wrong object: {destination}")
            elif destination.exists():
                raise FileExistsError(f"Refusing to replace staged input: {destination}")
            else:
                destination.symlink_to(source)
            staged[f"{domain.key}.{name}"] = destination
    return staged


def resume_artifact_paths(
    run_root: Path,
    domains: Sequence[Domain],
    start_from: str,
) -> list[Path]:
    paths: list[Path] = []
    if start_from == "checkpoints":
        for domain in domains:
            for method in LEARNED_MODELS:
                root = run_root / "checkpoints" / domain.key / method / "final"
                paths.extend(
                    (
                        root / "head.pt",
                        root / "lora" / "adapter_config.json",
                        root / "lora" / "adapter_model.safetensors",
                        root
                        / (
                            "hurdle_head_config.json"
                            if method.startswith("zoib_")
                            else "regression_head_config.json"
                        ),
                    )
                )
    elif start_from == "predictions":
        for domain in domains:
            root = run_root / "predictions" / domain.key
            for method in ALL_METHODS:
                for split in ("dev", "test"):
                    paths.append(root / f"{method}_{split}.json")
            paths.append(root / "prompt_retry_test.json")
    return paths


def validate_resume_artifacts(
    run_root: Path, domains: Sequence[Domain], start_from: str
) -> list[Path]:
    paths = resume_artifact_paths(run_root, domains, start_from)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:12])
        suffix = f"\n  ... and {len(missing) - 12} more" if len(missing) > 12 else ""
        raise FileNotFoundError(
            f"--start-from {start_from} resumes an existing --run-root; "
            f"missing {len(missing)} required files:\n{preview}{suffix}"
        )
    return paths


def write_sbatch(
    path: Path,
    *,
    name: str,
    partition: str,
    time_limit: str,
    cpus: int,
    memory: str,
    gpu: bool,
    log_dir: Path,
    webshop_root: Path,
    body: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --cpus-per-task={int(cpus)}",
        f"#SBATCH --mem={memory}",
    ]
    if gpu:
        lines.append("#SBATCH --gres=gpu:1")
    lines.extend(
        [
            f"#SBATCH --output={log_dir}/{name}_%j.out",
            f"#SBATCH --error={log_dir}/{name}_%j.err",
            "set -euo pipefail",
            f"cd {shell_quote(REPO)}",
            "export TOKENIZERS_PARALLELISM=false",
            "export WANDB_MODE=disabled",
            "export TRANSFORMERS_OFFLINE=1",
            f"export WEBSHOP_DATA_DIR={shell_quote(webshop_root / 'data')}",
            f"export PYTHONPATH={shell_quote(REPO)}:{shell_quote(webshop_root)}:${{PYTHONPATH:-}}",
            "echo job_started=$(date --iso-8601=seconds)",
            "echo node=$(hostname)",
            "echo slurm_job_id=${SLURM_JOB_ID:-none}",
            "echo cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-none}",
            *body,
            "echo job_finished=$(date --iso-8601=seconds)",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def submit(path: Path, *, dependencies: Sequence[str], dry_run: bool) -> str:
    command = ["sbatch", "--parsable"]
    deps = [str(value) for value in dependencies if str(value).strip()]
    if deps:
        command.append("--dependency=afterok:" + ":".join(deps))
    command.append(str(path))
    if dry_run:
        print("DRY-RUN", " ".join(shell_quote(x) for x in command), flush=True)
        return f"DRYRUN_{path.stem}"
    return subprocess.check_output(command, text=True).strip().split(";", 1)[0]


def train_body(domain: Domain, model: str, *, run_root: Path, base_model: Path, conda_env: str) -> List[str]:
    py = _python(conda_env)
    output = run_root / "checkpoints" / domain.key / model
    learning_rate = "1e-5" if (domain.key, model) == ("paper_search", "residual_gaussian") else "1e-4"
    common = (
        f"--train_data {shell_quote(domain.labels['train'])} --val_data {shell_quote(domain.labels['dev'])} "
        f"--base_model {shell_quote(base_model)} --output_dir {shell_quote(output)} "
        "--epochs 3 --batch_size 1 --eval_batch_size 2 --gradient_accumulation_steps 16 "
        f"--learning_rate {learning_rate} --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 "
        "--gradient_checkpointing --seed 42 --num_workers 2"
    )
    if model in {"zoib_raw", "zoib_remaining"}:
        support = "remaining" if model == "zoib_remaining" else "raw"
        command = (
            f"{py} -m opportunity_forecasting.models.train_zoib train {common} "
            f"--support_mode {support} --early_stop_metric loss "
            "--ev_huber_weight 0.2 --event_brier_weight 0.1"
        )
    elif model == "residual_scalar":
        command = (
            f"{py} -m opportunity_forecasting.models.train_regression train --target_family residual_scalar {common} "
            "--early_stop_metric ev_mae --head_init_std 0.001 --head_bias_init -2.5"
        )
    elif model == "residual_gaussian":
        command = (
            f"{py} -m opportunity_forecasting.models.train_regression train --target_family residual_gaussian {common} "
            "--early_stop_metric loss "
            "--gaussian_std_floor 0.01 --head_init_std 0.001 --head_bias_init -2.5"
        )
    else:
        raise ValueError(model)
    return [command]


def prediction_shard_body(
    domain: Domain,
    model: str,
    split: str,
    shard_idx: int,
    *,
    run_root: Path,
    base_model: Path,
    conda_env: str,
) -> List[str]:
    py = _python(conda_env)
    model_dir = run_root / "checkpoints" / domain.key / model / "final"
    output = run_root / "predictions" / domain.key / "shards" / f"{model}_{split}_{shard_idx:02d}.json"
    module = (
        "opportunity_forecasting.models.train_zoib"
        if model.startswith("zoib_")
        else "opportunity_forecasting.models.train_regression"
    )
    shard_mode = "" if model.startswith("zoib_") else " --shard_by_goal"
    return [
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"export OPPORTUNITY_BASE_MODEL={shell_quote(base_model)}",
        f"{py} -m {module} predict-labels --model_dir {shell_quote(model_dir)} "
        f"--data {shell_quote(domain.labels[split])} --output_path {shell_quote(output)} --split {split} "
        f"--model_label {shell_quote(MODEL_DISPLAY[model])} --batch_size 2 "
        f"--shard_idx {shard_idx} --num_shards {PREDICTION_SHARDS}{shard_mode}"
    ]


def merge_prediction_body(domain: Domain, model: str, split: str, *, run_root: Path, conda_env: str) -> List[str]:
    py = _python(conda_env)
    output = run_root / "predictions" / domain.key / f"{model}_{split}.json"
    shards = [run_root / "predictions" / domain.key / "shards" / f"{model}_{split}_{idx:02d}.json" for idx in range(PREDICTION_SHARDS)]
    return [f"{py} -m opportunity_forecasting.evaluation.merge_predictions --output {shell_quote(output)} " + " ".join(shell_quote(path) for path in shards)]


def prompt_body(
    domain: Domain,
    split: str,
    variant: str,
    *,
    run_root: Path,
    base_model: Path,
    conda_env: str,
) -> List[str]:
    py = _python(conda_env)
    retry_count = 1 if variant == "prompt_retry" else 0
    output = run_root / "predictions" / domain.key / f"{variant}_{split}.json"
    parts = [
        py,
        "-m",
        "opportunity_forecasting.models.predict",
        f"--data {shell_quote(domain.labels[split])}",
        f"--model-path {shell_quote(base_model)}",
        f"--output {shell_quote(output)}",
        f"--split {split}",
        f"--model-label {shell_quote(MODEL_DISPLAY.get(variant, PROMPT_RETRY_DISPLAY))}",
        "--engine vllm --top-k-seen 15 --max-model-len 4096",
        "--max-new-tokens 1024 --gpu-memory-utilization 0.82",
        "--dtype auto --enforce-eager --batch-size 8",
        "--repair-top-k-seen 5",
    ]
    if retry_count:
        parts.append("--retry-invalid")
    return [" ".join(parts)]


def controls_body(domain: Domain, split: str, *, run_root: Path, conda_env: str) -> List[str]:
    py = _python(conda_env)
    cache_dir = run_root / "predictions" / domain.key
    control_dir = run_root / "control_models" / domain.key
    commands = [f"mkdir -p {shell_quote(cache_dir)} {shell_quote(control_dir)}"]
    for method, feature_set in (("ridge_all", "all"), ("ridge_step", "step_only")):
        commands.append(
            f"{py} -m opportunity_forecasting.evaluation.regression_controls "
            f"--train-data {shell_quote(domain.labels['train'])} --eval-data {shell_quote(domain.labels[split])} "
            f"--split {split} --feature-set {feature_set} "
            f"--model-name {shell_quote(method)} --weights-path {shell_quote(control_dir / (method + '_' + split + '_weights.json'))} "
            f"--output-path {shell_quote(cache_dir / (method + '_' + split + '.json'))}"
        )
    commands.append(
        f"{py} -m opportunity_forecasting.evaluation.reservation "
        f"--train-data {shell_quote(domain.labels['train'])} --eval-data {shell_quote(domain.labels[split])} "
        f"--split {split} --model-name empirical_reservation "
        f"--output-path {shell_quote(cache_dir / ('empirical_reservation_' + split + '.json'))}"
    )
    return commands


def source_cache(run_root: Path, domain: Domain, method: str, split: str) -> Path:
    return run_root / "predictions" / domain.key / f"{method}_{split}.json"


def postprocess_body(
    domain: Domain,
    split: str,
    *,
    run_root: Path,
    conda_env: str,
) -> List[str]:
    py = _python(conda_env)
    result_root = run_root / "evaluations" / domain.key / split
    canonical = source_cache(run_root, domain, "zoib_raw", split)
    eval_dir = result_root / "stopping" / "evals"
    summary_dir = result_root / "stopping" / "summary"
    cache_csv = ",".join(
        str(source_cache(run_root, domain, method, split)) for method in ALL_METHODS
    )
    commands = [
        f"mkdir -p {shell_quote(result_root)}",
    ]
    commands.extend(
        [
            f"{py} -m opportunity_forecasting.evaluation.stopping "
            f"--prediction_cache_paths {shell_quote(cache_csv)} --output_dir {shell_quote(eval_dir)} "
            f"--policy1_c_steps={shell_quote(POLICY_C_STEPS)} --policy2_c_steps={shell_quote(POLICY_C_STEPS)} "
            f"--policy2_lambdas={shell_quote(POLICY_LAMBDAS)} --total_horizon_steps 60",
            f"{py} -m opportunity_forecasting.evaluation.summarize_stopping --sweep_root {shell_quote(eval_dir)} "
            f"--manifest_path {shell_quote(eval_dir / 'sweep.tsv')} --output_dir {shell_quote(summary_dir)}",
        ]
    )
    if split == "test":
        commands.append(
            f"{py} -m opportunity_forecasting.evaluation.fixed_step --cache {shell_quote(canonical)} "
            f"--domain {shell_quote(domain.title)} --split test "
            f"--output-csv {shell_quote(result_root / 'position_controls' / 'fixed_step_rows.csv')} "
            f"--output-json {shell_quote(result_root / 'position_controls' / 'fixed_step_summary.json')}"
        )
        for priority_mode, dirname in (
            ("predicted_mean_delta", "budgeted_expansion_raw_priority"),
            ("predicted_mean_delta_per_step", "budgeted_expansion_cost_normalized"),
        ):
            parts = [
                py,
                "-m",
                "opportunity_forecasting.evaluation.allocation",
                f"--domain {shell_quote(domain.title)} --split test",
            ]
            for method in ALL_METHODS:
                parts.append(
                    f"--cache {shell_quote(MODEL_DISPLAY[method] + '=' + str(source_cache(run_root, domain, method, split)))}"
                )
            for heuristic in domain.heuristics:
                parts.append(f"--heuristic_baseline {heuristic}")
            parts.extend(
                [
                    "--include_fixed_baseline",
                    f"--priority_mode {priority_mode}",
                    "--bootstrap_samples 500 --bootstrap_seed 123",
                    "--plot_metric oracle_gap",
                    f"--out_dir {shell_quote(result_root / dirname)}",
                ]
            )
            commands.append(" ".join(parts))

        analysis_root = result_root / "forecast_analysis"
        analysis_parts = [
            py,
            "-m",
            "opportunity_forecasting.evaluation.metrics",
            f"--labels {shell_quote(domain.labels[split])}",
            f"--domain {shell_quote(domain.title)} --split {split}",
            f"--output-json {shell_quote(analysis_root / 'forecast_support_position_summary.json')}",
            f"--output-summary-csv {shell_quote(analysis_root / 'forecast_support_position_summary.csv')}",
            f"--output-detail-csv {shell_quote(analysis_root / 'forecast_support_position_rows.csv')}",
        ]
        for method in ALL_METHODS:
            analysis_parts.append(
                f"--cache {shell_quote(MODEL_DISPLAY[method] + '=' + str(source_cache(run_root, domain, method, split)))}"
            )
        commands.append(" ".join(analysis_parts))
        for method in ALL_METHODS:
            commands.append(
                f"{py} -m opportunity_forecasting.evaluation.diagnostics "
                f"--cache {shell_quote(source_cache(run_root, domain, method, split))} "
                f"--labels {shell_quote(domain.labels[split])} --domain {shell_quote(domain.title)} --split {split} "
                f"--model-label {shell_quote(MODEL_DISPLAY[method])} "
                f"--output-json {shell_quote(analysis_root / (method + '_diagnostics.json'))} "
                f"--output-csv {shell_quote(analysis_root / (method + '_diagnostics.csv'))}"
            )
    return commands


def selection_body(domain: Domain, *, run_root: Path, conda_env: str) -> List[str]:
    py = _python(conda_env)
    root = run_root / "evaluations" / domain.key
    output = root / "dev_selected_test"
    return [
        f"mkdir -p {shell_quote(output)}",
        f"{py} -m opportunity_forecasting.evaluation.select_stopping "
        f"--dev-csv {shell_quote(root / 'dev' / 'stopping' / 'summary' / 'results_long.csv')} "
        f"--test-csv {shell_quote(root / 'test' / 'stopping' / 'summary' / 'results_long.csv')} "
        f"--output-csv {shell_quote(output / 'operating_points.csv')} "
        f"--output-md {shell_quote(output / 'operating_points.md')} --metric budget_feasible_reward --target-step-fraction 0.5",
    ]


def finalization_body(*, run_root: Path, conda_env: str) -> List[str]:
    py = _python(conda_env)
    results = run_root / "results"
    paper_outputs = run_root / "paper_outputs"
    return [
        f"{py} -m opportunity_forecasting.figures.materialize --run-root {shell_quote(run_root)} "
        f"--output-root {shell_quote(results)}",
        f"{py} -m opportunity_forecasting.figures.tables --run-root {shell_quote(run_root)} "
        f"--output-root {shell_quote(results / 'tables')}",
        f"{py} -m opportunity_forecasting paper --source-root {shell_quote(results)} "
        f"--output-root {shell_quote(paper_outputs)}",
    ]


def validate(
    manifest: Mapping[str, Any],
    domains: Sequence[Domain],
    *,
    base_model: Path,
    start_from: str,
    dry_run: bool,
) -> None:
    if not dry_run and start_from != "predictions" and not base_model.is_dir():
        raise FileNotFoundError(base_model)
    for domain in domains:
        if not dry_run:
            for path in [*domain.labels.values(), *domain.paper_assets.values()]:
                if not path.is_file():
                    raise FileNotFoundError(path)
    if int(manifest.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported configs/paper.json schema")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    ap.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    ap.add_argument("--train-a100-partition", default=DEFAULT_TRAIN_A100_PARTITION)
    ap.add_argument("--train-rtx-partition", default=DEFAULT_TRAIN_RTX_PARTITION)
    ap.add_argument("--gpu-partition", default=DEFAULT_GPU_PARTITION)
    ap.add_argument("--cpu-partition", default=DEFAULT_CPU_PARTITION)
    ap.add_argument(
        "--webshop-root", type=Path, default=REPO / "third_party" / "WebShop"
    )
    ap.add_argument("--webshop-asset", type=Path, default=None)
    ap.add_argument(
        "--start-from",
        choices=("labels", "checkpoints", "predictions"),
        default="labels",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    partitions = resolve_partitions(args)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_root = Path(args.run_root)
    base_model = resolve_backbone_path(manifest)
    webshop_root = Path(args.webshop_root)
    webshop_asset = args.webshop_asset or (webshop_root / "data" / "items_shuffle.json")
    domains = load_domains(manifest, webshop_asset_override=webshop_asset)
    validate(
        manifest,
        domains,
        base_model=base_model,
        start_from=str(args.start_from),
        dry_run=bool(args.dry_run),
    )

    dirty = bool(_git("status", "--porcelain"))
    try:
        code_commit = _git("rev-parse", "HEAD")
    except subprocess.CalledProcessError:
        if not args.dry_run:
            raise
        code_commit = "UNCOMMITTED_DRY_RUN"
    for directory in (
        "job_scripts",
        "logs",
        "checkpoints",
        "predictions",
        "control_models",
        "evaluations",
        "results",
        "metadata",
        "inputs",
    ):
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    input_summary = (
        {
            "status": "dry_run",
            "required_inputs": sorted(
                str(path)
                for domain in domains
                for path in (
                    *domain.labels.values(),
                    *domain.checkpoints.values(),
                    *domain.paper_assets.values(),
                )
            ),
        }
        if args.dry_run
        else validate_inputs(
            manifest_path,
            include_checkpoints=True,
        )
    )
    (run_root / "metadata" / "inputs.json").write_text(
        json.dumps(input_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staged_inputs = stage_paper_search_inputs(run_root, domains)
    resumed_inputs = (
        []
        if args.dry_run
        else validate_resume_artifacts(run_root, domains, str(args.start_from))
    )

    jobs: List[Dict[str, Any]] = []

    def launch(
        role: str,
        script_name: str,
        *,
        name: str,
        partition: str,
        time_limit: str,
        cpus: int,
        memory: str,
        gpu: bool,
        body: Sequence[str],
        dependencies: Sequence[str] = (),
    ) -> str:
        path = run_root / "job_scripts" / script_name
        write_sbatch(
            path,
            name=name,
            partition=partition,
            time_limit=time_limit,
            cpus=cpus,
            memory=memory,
            gpu=gpu,
            log_dir=run_root / "logs",
            webshop_root=webshop_root,
            body=body,
        )
        job_id = submit(path, dependencies=dependencies, dry_run=bool(args.dry_run))
        jobs.append(
            {
                "role": role,
                "job_id": job_id,
                "script": str(path),
                "dependencies": [str(x) for x in dependencies if str(x)],
                "partition": partition,
            }
        )
        return job_id

    train_jobs: Dict[Tuple[str, str], str] = {}
    merge_jobs: Dict[Tuple[str, str, str], str] = {}
    prompt_jobs: Dict[Tuple[str, str, str], str] = {}
    control_jobs: Dict[Tuple[str, str], str] = {}
    post_jobs: Dict[Tuple[str, str], str] = {}
    select_jobs: List[str] = []

    if args.start_from == "labels":
        for domain in domains:
            train_partition = _training_partition(
                domain.key,
                partitions["train_a100"],
                partitions["train_rtx"],
            )
            for model in LEARNED_MODELS:
                train_jobs[(domain.key, model)] = launch(
                    f"train_{domain.key}_{model}",
                    f"10_train_{domain.key}_{model}.sbatch",
                    name=f"opp-tr-{domain.key[:3]}-{model[:10]}",
                    partition=train_partition,
                    time_limit="23:59:00",
                    cpus=4,
                    memory="120G",
                    gpu=True,
                    body=train_body(
                        domain,
                        model,
                        run_root=run_root,
                        base_model=base_model,
                        conda_env=str(args.conda_env),
                    ),
                )

    if args.start_from in {"labels", "checkpoints"}:
        for domain in domains:
            for split in ("dev", "test"):
                variant = "prompt_original"
                prompt_jobs[(domain.key, variant, split)] = launch(
                    f"predict_{domain.key}_{variant}_{split}",
                    f"20_prompt_{domain.key}_{variant}_{split}.sbatch",
                    name=f"opp-pr-{domain.key[:3]}-{variant[-5:]}-{split}",
                    partition=partitions["gpu"],
                    time_limit="12:00:00",
                    cpus=4,
                    memory="80G",
                    gpu=True,
                    body=prompt_body(
                        domain,
                        split,
                        variant,
                        run_root=run_root,
                        base_model=base_model,
                        conda_env=str(args.conda_env),
                    ),
                )
                if split == "test":
                    retry_variant = "prompt_retry"
                    prompt_jobs[(domain.key, retry_variant, split)] = launch(
                        f"predict_{domain.key}_{retry_variant}_{split}",
                        f"20_prompt_{domain.key}_{retry_variant}_{split}.sbatch",
                        name=f"opp-pr-{domain.key[:3]}-retry-{split}",
                        partition=partitions["gpu"],
                        time_limit="12:00:00",
                        cpus=4,
                        memory="80G",
                        gpu=True,
                        body=prompt_body(
                            domain,
                            split,
                            retry_variant,
                            run_root=run_root,
                            base_model=base_model,
                            conda_env=str(args.conda_env),
                        ),
                    )
                control_jobs[(domain.key, split)] = launch(
                    f"controls_{domain.key}_{split}",
                    f"21_controls_{domain.key}_{split}.sbatch",
                    name=f"opp-ctl-{domain.key[:3]}-{split}",
                    partition=partitions["cpu"],
                    time_limit="08:00:00",
                    cpus=3,
                    memory="64G",
                    gpu=False,
                    body=controls_body(
                        domain,
                        split,
                        run_root=run_root,
                        conda_env=str(args.conda_env),
                    ),
                )

    if args.start_from in {"labels", "checkpoints"}:
        for domain in domains:
            for model in LEARNED_MODELS:
                for split in ("dev", "test"):
                    shard_jobs: List[str] = []
                    for shard_idx in range(PREDICTION_SHARDS):
                        dependencies = (
                            [train_jobs[(domain.key, model)]]
                            if args.start_from == "labels"
                            else []
                        )
                        shard_jobs.append(
                            launch(
                                f"predict_{domain.key}_{model}_{split}_shard{shard_idx}",
                                f"30_predict_{domain.key}_{model}_{split}_{shard_idx:02d}.sbatch",
                                name=f"opp-pd-{domain.key[:3]}-{model[:7]}-{split[0]}{shard_idx}",
                                partition=partitions["gpu"],
                                time_limit="10:00:00",
                                cpus=3,
                                memory="72G",
                                gpu=True,
                                body=prediction_shard_body(
                                    domain,
                                    model,
                                    split,
                                    shard_idx,
                                    run_root=run_root,
                                    base_model=base_model,
                                    conda_env=str(args.conda_env),
                                ),
                                dependencies=dependencies,
                            )
                        )
                    merge_jobs[(domain.key, model, split)] = launch(
                        f"merge_{domain.key}_{model}_{split}",
                        f"31_merge_{domain.key}_{model}_{split}.sbatch",
                        name=f"opp-mg-{domain.key[:3]}-{model[:7]}-{split}",
                        partition=partitions["cpu"],
                        time_limit="01:00:00",
                        cpus=2,
                        memory="24G",
                        gpu=False,
                        body=merge_prediction_body(
                            domain,
                            model,
                            split,
                            run_root=run_root,
                            conda_env=str(args.conda_env),
                        ),
                        dependencies=shard_jobs,
                    )

    for domain in domains:
        for split in ("dev", "test"):
            cache_deps = [
                job_id
                for job_id in (
                    prompt_jobs.get((domain.key, "prompt_original", split)),
                    control_jobs.get((domain.key, split)),
                    *[
                        merge_jobs.get((domain.key, model, split))
                        for model in LEARNED_MODELS
                    ],
                )
                if job_id
            ]
            post_jobs[(domain.key, split)] = launch(
                f"evaluate_{domain.key}_{split}",
                f"40_evaluate_{domain.key}_{split}.sbatch",
                name=f"opp-eval-{domain.key[:3]}-{split}",
                partition=partitions["cpu"],
                time_limit="18:00:00",
                cpus=8,
                memory="96G",
                gpu=False,
                body=postprocess_body(
                    domain,
                    split,
                    run_root=run_root,
                    conda_env=str(args.conda_env),
                ),
                dependencies=cache_deps,
            )
        select_jobs.append(
            launch(
                f"select_{domain.key}_operating_points",
                f"50_select_{domain.key}_operating_points.sbatch",
                name=f"opp-select-{domain.key[:3]}",
                partition=partitions["cpu"],
                time_limit="01:00:00",
                cpus=2,
                memory="16G",
                gpu=False,
                body=selection_body(
                    domain,
                    run_root=run_root,
                    conda_env=str(args.conda_env),
                ),
                dependencies=[
                    post_jobs[(domain.key, "dev")],
                    post_jobs[(domain.key, "test")],
                ],
            )
        )
    launch(
        "summarize_and_render",
        "60_summarize_and_render.sbatch",
        name="opp-final-outputs",
        partition=partitions["cpu"],
        time_limit="02:00:00",
        cpus=4,
        memory="32G",
        gpu=False,
        body=finalization_body(
            run_root=run_root,
            conda_env=str(args.conda_env),
        ),
        dependencies=[
            *select_jobs,
            *[
                prompt_jobs[(domain.key, "prompt_retry", "test")]
                for domain in domains
                if (domain.key, "prompt_retry", "test") in prompt_jobs
            ],
        ],
    )

    execution = {
        "manifest": str(manifest_path),
        "run_root": str(run_root),
        "code_commit": code_commit,
        "working_tree_dirty": dirty,
        "dry_run": bool(args.dry_run),
        "start_from": str(args.start_from),
        "prediction_shards": PREDICTION_SHARDS,
        "partitions": partitions,
        "launcher_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "numpy",
                    "torch",
                    "transformers",
                    "vllm",
                )
            },
        },
        "shared_policy_c_steps": POLICY_C_STEPS,
        "shared_policy_lambdas": POLICY_LAMBDAS,
        "staged_inputs": {key: str(path) for key, path in sorted(staged_inputs.items())},
        "resumed_inputs": [str(path) for path in resumed_inputs],
        "jobs": jobs,
    }
    execution_path = run_root / "metadata" / "execution.json"
    execution_path.write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tsv = run_root / "metadata" / "jobs.tsv"
    with tsv.open("w", encoding="utf-8") as wf:
        wf.write("role\tjob_id\tpartition\tdependencies\tscript\n")
        for job in jobs:
            wf.write(
                f"{job['role']}\t{job['job_id']}\t{job['partition']}\t"
                f"{','.join(job['dependencies'])}\t{job['script']}\n"
            )
    print(json.dumps({"execution_manifest": str(execution_path), "jobs": len(jobs), "dry_run": bool(args.dry_run)}, sort_keys=True))


if __name__ == "__main__":
    main()

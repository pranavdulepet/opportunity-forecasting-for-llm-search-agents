"""Evaluate existing predictions and generate paper outputs without Slurm."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


from opportunity_forecasting import REPO_ROOT as REPO

import opportunity_forecasting.experiments.submit as dag
from opportunity_forecasting.data.validate import validate_inputs
from opportunity_forecasting.manifest import PAPER_CONFIG, resolve_backbone_path


CURRENT_PYTHON_SENTINEL = "__opportunity_forecasting_current_python__"


@dataclass(frozen=True)
class Stage:
    name: str
    dependencies: tuple[str, ...]
    commands: tuple[str, ...]


def _commands(
    builder: Callable[..., list[str]],
    *,
    conda_env: str | None,
    **kwargs: object,
) -> tuple[str, ...]:
    body_env = conda_env or CURRENT_PYTHON_SENTINEL
    commands = builder(conda_env=body_env, **kwargs)
    if conda_env:
        return tuple(commands)
    placeholder = dag._python(CURRENT_PYTHON_SENTINEL)
    interpreter = shlex.quote(sys.executable)
    return tuple(command.replace(placeholder, interpreter) for command in commands)


def build_stages(
    domains: Sequence[dag.Domain],
    *,
    run_root: Path,
    conda_env: str | None,
) -> tuple[Stage, ...]:
    stages: list[Stage] = []
    selection_names: list[str] = []
    for domain in domains:
        evaluation_names: dict[str, str] = {}
        for split in ("dev", "test"):
            name = f"evaluate_{domain.key}_{split}"
            evaluation_names[split] = name
            stages.append(
                Stage(
                    name=name,
                    dependencies=(),
                    commands=_commands(
                        dag.postprocess_body,
                        domain=domain,
                        split=split,
                        run_root=run_root,
                        conda_env=conda_env,
                    ),
                )
            )
        name = f"select_{domain.key}_operating_points"
        selection_names.append(name)
        stages.append(
            Stage(
                name=name,
                dependencies=(evaluation_names["dev"], evaluation_names["test"]),
                commands=_commands(
                    dag.selection_body,
                    domain=domain,
                    run_root=run_root,
                    conda_env=conda_env,
                ),
            )
        )
    stages.append(
        Stage(
            name="summarize_and_render",
            dependencies=tuple(selection_names),
            commands=_commands(
                dag.finalization_body,
                run_root=run_root,
                conda_env=conda_env,
            ),
        )
    )
    return tuple(stages)


def run_shell(command: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": os.pathsep.join(
                value
                for value in (str(REPO), env.get("PYTHONPATH", ""))
                if value
            ),
        }
    )
    subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{command}"],
        cwd=REPO,
        env=env,
        check=True,
    )


def execute_stages(
    stages: Sequence[Stage],
    *,
    command_runner: Callable[[str], None] = run_shell,
    on_update: Callable[[Sequence[Mapping[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    completed: set[str] = set()
    records: list[dict[str, object]] = []
    for stage in stages:
        missing = set(stage.dependencies) - completed
        if missing:
            raise RuntimeError(
                f"{stage.name} has incomplete dependencies: {sorted(missing)}"
            )
        record: dict[str, object] = {
            "name": stage.name,
            "dependencies": list(stage.dependencies),
            "commands": list(stage.commands),
            "status": "running",
            "completed_commands": 0,
        }
        records.append(record)
        if on_update:
            on_update(records)
        try:
            for command in stage.commands:
                command_runner(command)
                record["completed_commands"] = int(record["completed_commands"]) + 1
                if on_update:
                    on_update(records)
        except BaseException:
            record["status"] = "failed"
            if on_update:
                on_update(records)
            raise
        record["status"] = "completed"
        completed.add(stage.name)
        if on_update:
            on_update(records)
    return records


def run_pipeline(
    *,
    manifest_path: Path,
    run_root: Path,
    conda_env: str | None,
    dry_run: bool = False,
    command_runner: Callable[[str], None] = run_shell,
) -> Path:
    manifest_path = manifest_path.expanduser().resolve()
    run_root = run_root.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    domains = dag.load_domains(manifest)
    dag.validate(
        manifest,
        domains,
        base_model=resolve_backbone_path(manifest),
        start_from="predictions",
        dry_run=dry_run,
    )
    directories = (
        ("metadata",)
        if dry_run
        else ("predictions", "inputs", "metadata", "evaluations", "results")
    )
    for directory in directories:
        (run_root / directory).mkdir(parents=True, exist_ok=True)
    staged_inputs: dict[str, Path] = {}
    resumed_inputs: list[Path] = []
    if not dry_run:
        input_summary = validate_inputs(manifest_path)
        (run_root / "metadata" / "inputs.json").write_text(
            json.dumps(input_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_inputs = dag.stage_paper_search_inputs(run_root, domains)
        resumed_inputs = dag.validate_resume_artifacts(
            run_root, domains, "predictions"
        )
    stages = build_stages(domains, run_root=run_root, conda_env=conda_env)
    planned_stages = [
        {
            "name": stage.name,
            "dependencies": list(stage.dependencies),
            "commands": list(stage.commands),
            "status": "planned",
            "completed_commands": 0,
        }
        for stage in stages
    ]
    execution_path = run_root / "metadata" / "local_execution.json"
    execution: dict[str, object] = {
        "manifest": str(manifest_path),
        "run_root": str(run_root),
        "start_from": "predictions",
        "scheduler": "local",
        "dry_run": dry_run,
        "runtime": {
            "conda_env": conda_env,
            "python": sys.executable,
            "python_version": platform.python_version(),
        },
        "staged_inputs": {
            key: str(path) for key, path in sorted(staged_inputs.items())
        },
        "resumed_inputs": [str(path) for path in resumed_inputs],
        "stages": planned_stages if dry_run else [],
    }

    def write_execution(records: Sequence[Mapping[str, object]]) -> None:
        execution["stages"] = list(records)
        execution_path.write_text(
            json.dumps(execution, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_execution(planned_stages if dry_run else [])
    if dry_run:
        return execution_path
    execute_stages(
        stages,
        command_runner=command_runner,
        on_update=write_execution,
    )
    return execution_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--conda-env")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    execution_path = run_pipeline(
        manifest_path=args.manifest,
        run_root=args.run_root,
        conda_env=args.conda_env,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "execution_manifest": str(execution_path),
                "status": "planned" if args.dry_run else "completed",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

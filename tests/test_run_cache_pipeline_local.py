from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import opportunity_forecasting.evaluation.local_pipeline as local
from opportunity_forecasting.experiments.submit import Domain


def domain(key: str) -> Domain:
    return Domain(
        key=key,
        title="WebShop" if key == "webshop" else "Paper Search",
        reward_mode="test",
        labels={"dev": Path("dev.jsonl"), "test": Path("test.jsonl")},
        checkpoints={"dev": Path("dev.jsonl"), "test": Path("test.jsonl")},
        heuristics=("step_early",),
        paper_assets={},
        webshop_asset=None,
    )


def test_build_stages_orders_evaluation_selection_and_rendering(tmp_path: Path) -> None:
    stages = local.build_stages(
        (domain("webshop"), domain("paper_search")),
        run_root=tmp_path,
        conda_env=None,
    )
    assert [stage.name for stage in stages] == [
        "evaluate_webshop_dev",
        "evaluate_webshop_test",
        "select_webshop_operating_points",
        "evaluate_paper_search_dev",
        "evaluate_paper_search_test",
        "select_paper_search_operating_points",
        "summarize_and_render",
    ]
    assert stages[2].dependencies == (
        "evaluate_webshop_dev",
        "evaluate_webshop_test",
    )
    assert stages[-1].dependencies == (
        "select_webshop_operating_points",
        "select_paper_search_operating_points",
    )
    assert all(
        "conda run" not in command
        for stage in stages
        for command in stage.commands
    )
    assert any(sys.executable in command for command in stages[0].commands)
    assert "figures.materialize" in stages[-1].commands[0]
    assert "opportunity_forecasting paper" in stages[-1].commands[-1]


def test_build_stages_uses_requested_conda_environment(tmp_path: Path) -> None:
    stages = local.build_stages(
        (domain("webshop"),), run_root=tmp_path, conda_env="paper-env"
    )
    commands = [command for stage in stages for command in stage.commands]
    assert any(
        "conda run --no-capture-output -n paper-env python" in command
        for command in commands
    )


def test_execute_stages_checks_dependencies_and_stops_on_failure() -> None:
    calls: list[str] = []
    stages = (
        local.Stage("first", (), ("one", "two")),
        local.Stage("second", ("first",), ("three",)),
    )

    def fail(command: str) -> None:
        calls.append(command)
        if command == "two":
            raise subprocess.CalledProcessError(1, command)

    snapshots: list[list[dict[str, object]]] = []
    with pytest.raises(subprocess.CalledProcessError):
        local.execute_stages(
            stages,
            command_runner=fail,
            on_update=lambda records: snapshots.append(
                [dict(record) for record in records]
            ),
        )
    assert calls == ["one", "two"]
    assert snapshots[-1][0]["status"] == "failed"
    assert snapshots[-1][0]["completed_commands"] == 1

    with pytest.raises(RuntimeError, match="incomplete dependencies"):
        local.execute_stages(
            (local.Stage("second", ("missing",), ("never",)),),
            command_runner=calls.append,
        )


def write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "protocol": {"backbone": {"path": "base"}},
            }
        ),
        encoding="utf-8",
    )


def test_run_pipeline_resumes_predictions_and_records_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "configs" / "paper.json"
    write_manifest(manifest_path)
    domains = (domain("webshop"), domain("paper_search"))
    calls: list[object] = []
    monkeypatch.setattr(local.dag, "load_domains", lambda manifest: domains)
    monkeypatch.setattr(
        local.dag,
        "validate",
        lambda *args, **kwargs: calls.append(("validate", kwargs["start_from"])),
    )
    monkeypatch.setattr(
        local,
        "validate_inputs",
        lambda *args, **kwargs: {"valid": True},
    )
    monkeypatch.setattr(
        local.dag,
        "stage_paper_search_inputs",
        lambda *args: {"paper.query": tmp_path / "query"},
    )

    monkeypatch.setattr(
        local.dag,
        "validate_resume_artifacts",
        lambda *args: [tmp_path / "prediction"],
    )
    monkeypatch.setattr(
        local,
        "build_stages",
        lambda *args, **kwargs: (
            local.Stage("evaluate", (), ("evaluate-command",)),
            local.Stage("render", ("evaluate",), ("render-command",)),
        ),
    )
    commands: list[str] = []
    execution_path = local.run_pipeline(
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        conda_env=None,
        command_runner=commands.append,
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert calls[0] == ("validate", "predictions")
    assert commands == ["evaluate-command", "render-command"]
    assert execution["scheduler"] == "local"
    assert execution["start_from"] == "predictions"
    assert execution["resumed_inputs"] == [str(tmp_path / "prediction")]
    assert [stage["status"] for stage in execution["stages"]] == [
        "completed",
        "completed",
    ]


def test_run_pipeline_dry_run_only_writes_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "configs" / "paper.json"
    write_manifest(manifest_path)
    calls: list[object] = []
    monkeypatch.setattr(local.dag, "load_domains", lambda manifest: (domain("webshop"),))
    monkeypatch.setattr(
        local.dag,
        "validate",
        lambda *args, **kwargs: calls.append(("validate", kwargs["dry_run"])),
    )
    monkeypatch.setattr(
        local,
        "validate_inputs",
        lambda *args, **kwargs: pytest.fail("dry run validated files"),
    )
    monkeypatch.setattr(
        local.dag,
        "stage_paper_search_inputs",
        lambda *args: pytest.fail("dry run staged inputs"),
    )
    monkeypatch.setattr(
        local.dag,
        "validate_resume_artifacts",
        lambda *args: pytest.fail("dry run validated predictions"),
    )
    monkeypatch.setattr(
        local,
        "build_stages",
        lambda *args, **kwargs: (
            local.Stage("planned", (), ("never-executed",)),
        ),
    )
    execution_path = local.run_pipeline(
        manifest_path=manifest_path,
        run_root=tmp_path / "run",
        conda_env=None,
        dry_run=True,
        command_runner=lambda command: pytest.fail("dry run executed a command"),
    )
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    assert calls == [("validate", True)]
    assert execution["dry_run"] is True
    assert execution["staged_inputs"] == {}
    assert execution["resumed_inputs"] == []
    assert execution["stages"][0]["status"] == "planned"
    assert not (tmp_path / "run" / "predictions").exists()
    assert not (tmp_path / "run" / "evaluations").exists()


def test_parse_args_supports_current_python_or_conda(tmp_path: Path) -> None:
    current = local.parse_args(["--run-root", str(tmp_path)])
    conda = local.parse_args(
        [
            "--run-root",
            str(tmp_path),
            "--conda-env",
            str(tmp_path / "env"),
            "--dry-run",
        ]
    )
    assert current.conda_env is None
    assert conda.conda_env == str(tmp_path / "env")
    assert current.dry_run is False
    assert conda.dry_run is True

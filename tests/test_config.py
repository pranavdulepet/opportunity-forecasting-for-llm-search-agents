from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from opportunity_forecasting.data.validate import validate_file
from opportunity_forecasting.data.webshop_index import contents
from opportunity_forecasting.manifest import (
    load_manifest,
    resolve_artifact_path,
    resolve_backbone_path,
)


def test_paper_config_describes_full_horizon_forecasting() -> None:
    config = load_manifest()
    assert config["schema_version"] == 1
    assert config["paper"]["title"] == "Opportunity Forecasting for LLM Search Agents"
    assert config["protocol"]["continuation_horizon_steps"] == 60
    assert config["protocol"]["monte_carlo_continuations_per_state"] == 6
    assert set(config["models"]["methods"]) == {
        "zoib_raw",
        "zoib_remaining",
        "residual_scalar",
        "residual_gaussian",
    }


def test_backbone_pins_official_weights_and_tokenizer() -> None:
    config = load_manifest()
    backbone = config["protocol"]["backbone"]
    official = set(backbone["official_files"])
    assert official == {
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }


def test_shipped_result_sources_match_config() -> None:
    config = load_manifest()
    for group_name in (
        "paper_results",
        "diagnostic_results",
        "paper_tables",
        "figure_sources",
    ):
        for spec in config[group_name].values():
            validate_file(resolve_artifact_path(config, spec), spec)
    for domain in config["domains"].values():
        for spec in domain["splits"]["goal_ids"].values():
            validate_file(resolve_artifact_path(config, spec), spec)
        validate_file(
            resolve_artifact_path(config, domain["splits"]["metadata"]),
            domain["splits"]["metadata"],
        )
    paper_search = config["domains"]["paper_search"]
    for spec in (
        paper_search["export_generation"]["source_revisions"],
        paper_search["environment_assets"]["metadata"],
    ):
        validate_file(resolve_artifact_path(config, spec), spec)


def test_paths_are_repo_relative_and_relocatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPPORTUNITY_BASE_MODEL", raising=False)
    config = {
        "protocol": {
            "backbone": {
                "path": "runs/models/Qwen2.5-7B-Instruct",
            }
        }
    }
    assert resolve_artifact_path(
        config, {"path": "data/file"}, repo_root=tmp_path
    ) == tmp_path / "data/file"
    assert resolve_backbone_path(config, repo_root=tmp_path) == (
        tmp_path / "runs/models/Qwen2.5-7B-Instruct"
    )

    model = tmp_path / "shared/model"
    monkeypatch.setenv("OPPORTUNITY_BASE_MODEL", str(model))
    assert resolve_backbone_path(config, repo_root=tmp_path) == model


def test_data_file_validation_checks_size_hash_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"{}\n{}\n")
    spec = {
        "bytes": 6,
        "rows": 2,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert validate_file(path, spec)["rows"] == 2
    with pytest.raises(ValueError):
        validate_file(path, {**spec, "bytes": 7})


def test_webshop_index_text_matches_upstream_fields() -> None:
    product = {
        "Title": "Title",
        "Description": "Description",
        "BulletPoints": ["Bullet"],
        "options": {"Color": ["Blue", "Red"]},
    }
    assert contents(product) == "title description bullet color: blue, red"

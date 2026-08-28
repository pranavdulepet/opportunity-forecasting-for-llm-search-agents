from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from opportunity_forecasting.data.trajectories import _visited_page_metadata
from opportunity_forecasting.evaluation.fixed_step import evaluate as evaluate_fixed_step
from opportunity_forecasting.evaluation.regression_controls import (
    feature_vector,
    load_jsonl_rows,
)
from opportunity_forecasting.evaluation.reservation import (
    build_cache as build_reservation_cache,
)
from opportunity_forecasting.evaluation.reservation import fit as fit_reservation
from opportunity_forecasting.experiments.submit import (
    Domain,
    _python,
    _training_partition,
    finalization_body,
    prediction_shard_body,
    resolve_partitions,
    stage_paper_search_inputs,
    submit,
    train_body,
)
from opportunity_forecasting.models.train_zoib import (
    _support_transform,
    build_cache_from_labeled_examples,
)


def labeled_row(goal_id: int, step: int, best: float, deltas: list[float]) -> dict:
    return {
        "goal_idx": goal_id,
        "goal_text": f"goal {goal_id}",
        "checkpoint_step": step,
        "input": {
            "goal": f"goal {goal_id}",
            "best_reward_seen": best,
            "seen_products": {},
            "observation": "Buy Now" if step > 1 else "search",
        },
        "metadata": {"trigger": "product_page" if step > 1 else "search"},
        "continuation_deltas": deltas,
    }


def prediction_payload(rewards: list[float], *, label: str) -> dict:
    return {
        "goal_ids": [1],
        "model_path": label,
        "goal_predictions": {
            "1": {
                "goal_idx": 1,
                "goal_text": "goal 1",
                "checkpoints": [
                    {
                        "checkpoint_idx": index,
                        "checkpoint_step": index + 2,
                        "visited_product_page": True,
                        "best_reward_seen": reward,
                        "forecast_family": "explicit_residual_scalar",
                        "expected_delta": 0.1 * (index + 1),
                        "expected_std_delta": 0.0,
                    }
                    for index, reward in enumerate(rewards)
                ],
            }
        },
    }


def domain(tmp_path: Path, key: str = "paper_search") -> Domain:
    return Domain(
        key=key,
        title="Paper Search" if key == "paper_search" else "WebShop",
        reward_mode="test",
        labels={
            "train": tmp_path / "train.jsonl",
            "dev": tmp_path / "dev.jsonl",
            "test": tmp_path / "test.jsonl",
        },
        checkpoints={"dev": tmp_path / "dev.jsonl", "test": tmp_path / "test.jsonl"},
        heuristics=(),
        paper_assets={},
        webshop_asset=None,
    )


def test_support_aware_zoib_scales_by_remaining_reward() -> None:
    row = labeled_row(1, 2, 0.75, [0.0, 0.125, 0.25])
    transformed, scale = _support_transform(
        row, row["continuation_deltas"], support_mode="remaining"
    )
    assert scale == 0.25
    assert transformed == [0.0, 0.5, 1.0]


def test_support_aware_predictions_keep_absolute_expected_gain(tmp_path: Path) -> None:
    payload = build_cache_from_labeled_examples(
        examples=[],
        predictions=[
            (
                {
                    "goal_idx": 1,
                    "goal_text": "goal 1",
                    "checkpoint_step": 2,
                    "row_idx": 0,
                    "visited_product_page": True,
                    "best_reward_seen": 0.75,
                },
                {
                    "forecast_family": "explicit_residual_zoib_remaining_support",
                    "delta_zero_prob": 0.5,
                    "delta_one_prob": 0.1,
                    "delta_pos_mean": 0.4,
                    "delta_pos_concentration": 3.0,
                    "remaining_gain_scale": 0.25,
                    "expected_delta": 0.065,
                    "expected_std_delta": 0.05,
                },
            )
        ],
        model_dir=tmp_path / "model",
        output_path=tmp_path / "predictions.json",
        split_name="test",
        label_source=tmp_path / "labels.jsonl",
    )
    checkpoint = payload["goal_predictions"]["1"]["checkpoints"][0]
    assert checkpoint["expected_delta"] == 0.065
    assert checkpoint["forecast_numeric_domain_ok"] is True


def test_position_features_are_derived_for_both_domains(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    rows = [labeled_row(1, 12, 0.4, [0.1]), labeled_row(1, 4, 0.2, [0.2])]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = load_jsonl_rows(path)
    by_step = {row["checkpoint_step"]: row for row in loaded}
    assert by_step[4]["_decision_index"] == 0
    assert by_step[12]["_decision_index"] == 1
    assert len(feature_vector(by_step[12], feature_set="all")) == 8
    assert len(feature_vector(by_step[12], feature_set="step_only")) == 3


def test_reservation_control_uses_only_training_deltas(tmp_path: Path) -> None:
    train = [
        labeled_row(1, 2, 0.0, [0.0, 0.2]),
        labeled_row(2, 2, 0.0, [0.1, 0.3]),
    ]
    fitted = fit_reservation(train, step_width=10, best_width=0.2)
    payload = build_reservation_cache(
        [labeled_row(3, 2, 0.0, [0.9])],
        fitted=fitted,
        output_path=tmp_path / "reservation.json",
        split="test",
        model_name="reservation",
        step_width=10,
        best_width=0.2,
        min_bin_samples=1,
    )
    checkpoint = payload["goal_predictions"]["3"]["checkpoints"][0]
    assert np.isclose(checkpoint["expected_delta"], 0.15)
    assert checkpoint["empirical_sample_count"] == 4


def test_paper_search_assets_receive_parser_filenames(tmp_path: Path) -> None:
    assets = {}
    for name in ("queries", "corpus", "qrels"):
        path = tmp_path / f"object_{name}"
        path.write_text(name, encoding="utf-8")
        assets[name] = path
    paper_domain = domain(tmp_path)
    paper_domain = Domain(**{**paper_domain.__dict__, "paper_assets": assets})
    staged = stage_paper_search_inputs(tmp_path / "run", [paper_domain])
    assert staged["paper_search.queries"].name == "queries.jsonl"
    assert staged["paper_search.corpus"].name == "corpus.jsonl"
    assert staged["paper_search.qrels"].name == "qrels.tsv"


def test_prediction_shards_use_goal_partitioning_for_regression(tmp_path: Path) -> None:
    webshop = domain(tmp_path, "webshop")
    zoib = prediction_shard_body(
        webshop,
        "zoib_raw",
        "test",
        0,
        run_root=tmp_path / "run",
        base_model=tmp_path / "model",
        conda_env="paper-env",
    )
    scalar = prediction_shard_body(
        webshop,
        "residual_scalar",
        "test",
        0,
        run_root=tmp_path / "run",
        base_model=tmp_path / "model",
        conda_env="paper-env",
    )
    assert zoib[1].startswith("export OPPORTUNITY_BASE_MODEL=")
    assert "--batch_size 2" in zoib[2]
    assert "--shard_by_goal" not in zoib[2]
    assert "--shard_by_goal" in scalar[2]


def test_finalization_only_summarizes_and_renders(tmp_path: Path) -> None:
    commands = finalization_body(
        run_root=tmp_path / "run",
        conda_env="paper-env",
    )
    assert len(commands) == 3
    assert "figures.materialize" in commands[0]
    assert "figures.tables" in commands[1]
    assert "opportunity_forecasting paper" in commands[2]
    assert all("compare" not in command and "verify" not in command for command in commands)


def test_launcher_configuration_matches_paper(tmp_path: Path) -> None:
    assert " -n opportunity-forecasting " in _python("opportunity-forecasting")
    assert f" -p {tmp_path / 'env'} " in _python(str(tmp_path / "env"))
    assert _training_partition("webshop", "a100", "rtx") == "a100"
    assert _training_partition("paper_search", "a100", "rtx") == "rtx"

    dry = argparse.Namespace(
        gpu_partition="",
        cpu_partition="",
        train_a100_partition="",
        train_rtx_partition="",
        dry_run=True,
    )
    assert resolve_partitions(dry)["gpu"] == "GPU_PARTITION"
    dry.dry_run = False
    with pytest.raises(ValueError, match="--gpu-partition"):
        resolve_partitions(dry)

    paper_domain = domain(tmp_path)
    gaussian = train_body(
        paper_domain,
        "residual_gaussian",
        run_root=tmp_path / "run",
        base_model=tmp_path / "model",
        conda_env="paper-env",
    )[0]
    scalar = train_body(
        paper_domain,
        "residual_scalar",
        run_root=tmp_path / "run",
        base_model=tmp_path / "model",
        conda_env="paper-env",
    )[0]
    assert "--learning_rate 1e-5" in gaussian
    assert "--learning_rate 1e-4" in scalar


def test_dry_run_preserves_dependency_identifiers(tmp_path: Path) -> None:
    job = submit(tmp_path / "10_train.sbatch", dependencies=(), dry_run=True)
    dependent = submit(
        tmp_path / "20_predict.sbatch", dependencies=(job,), dry_run=True
    )
    assert job == "DRYRUN_10_train"
    assert dependent == "DRYRUN_20_predict"


def test_fixed_step_control_uses_replay_step_cost() -> None:
    row = evaluate_fixed_step(
        prediction_payload([0.2, 0.5], label="fixed-step"),
        threshold=3,
        offline_back_steps=1,
        offline_to_product_steps=2,
    )
    assert row["mean_final_reward"] == 0.5
    assert row["mean_final_steps"] == 5.0


def test_trajectory_checkpoint_tracks_candidate_availability_in_both_domains() -> None:
    assert _visited_page_metadata(True) == {"visited_product_page": True}

from __future__ import annotations

import csv
import json
from pathlib import Path

from opportunity_forecasting.figures.materialize import (
    materialize_domain,
    materialize_search_value,
)


STOPPING = (
    "Original prompted baseline",
    "Raw residual ZOIB",
    "Support-corrected residual ZOIB",
    "Scalar residual",
    "Gaussian residual",
    "Feature-only ridge",
    "Step-only ridge",
    "Pandora-inspired empirical reservation",
)

BUDGET = (
    "Original prompted baseline",
    "Raw residual ZOIB",
    "Support-corrected residual ZOIB",
    "Scalar residual",
    "Gaussian residual",
    "Feature-only ridge",
    "Heuristic: earliest step",
    "Pandora-inspired empirical reservation",
)


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_run(root: Path, domain: str) -> None:
    test_root = root / "evaluations" / domain / "test"
    stopping = []
    for index, label in enumerate(STOPPING):
        for step, reward in ((1.0, 0.2 + index * 0.001), (3.0, 0.4 + index * 0.001)):
            stopping.append(
                {
                    "model_label": label,
                    "policy": "policy1",
                    "c_step": "0.1",
                    "lambda_risk": "1.0",
                    "mean_final_reward": reward,
                    "mean_final_steps": step,
                }
            )
    write_rows(
        test_root / "stopping" / "summary" / "pareto_frontier_by_model.csv",
        (
            "model_label",
            "policy",
            "c_step",
            "lambda_risk",
            "mean_final_reward",
            "mean_final_steps",
        ),
        stopping,
    )

    curves = []
    gap_curves = []
    for index, label in enumerate(
        ("Fixed-replay hindsight upper bound",) + BUDGET
    ):
        for step, reward in ((1.0, 0.3 + index * 0.001), (3.0, 0.5 + index * 0.001)):
            row = {
                "domain": domain,
                "split": "test",
                "series": label,
                "kind": "cache",
                "mean_final_steps": step,
                "mean_final_reward": reward,
                "expansion_count": 1,
            }
            curves.append(row)
            if label != "Fixed-replay hindsight upper bound":
                gap_curves.append({**row, "mean_final_reward": reward - 0.6})
    fields = (
        "domain",
        "split",
        "series",
        "kind",
        "mean_final_steps",
        "mean_final_reward",
        "expansion_count",
    )
    budget_root = test_root / "budgeted_expansion_raw_priority"
    write_rows(budget_root / "budgeted_expansion_curves.csv", fields, curves)
    write_rows(
        budget_root / "budgeted_expansion_oracle_gap_curves.csv",
        fields,
        gap_curves,
    )
    write_rows(
        test_root
        / "budgeted_expansion_cost_normalized"
        / "budgeted_expansion_curves.csv",
        fields,
        curves,
    )


def test_materialize_domain_writes_all_three_result_sources(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    output_root = tmp_path / "results"
    build_run(run_root, "webshop")
    report = materialize_domain(
        "webshop",
        run_root=run_root,
        output_root=output_root,
    )
    assert set(report["outputs"]) == {
        "stopping",
        "budgeted_expansion",
        "absolute_reward",
        "cost_normalized_allocation",
    }
    assert all(Path(path).is_file() for path in report["outputs"].values())


def test_materialize_search_value_reads_canonical_prediction_caches(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    for domain in ("webshop", "paper_search"):
        cache_root = run_root / "predictions" / domain
        cache_root.mkdir(parents=True)
        payload = {
            "goal_ids": [1],
            "goal_predictions": {
                "1": {
                    "goal_idx": 1,
                    "checkpoints": [
                        {
                            "checkpoint_step": 5,
                            "candidate_available": True,
                            "best_reward_seen": 0.2,
                            "expected_delta": 0.2,
                            "expected_std_delta": 0.0,
                        },
                        {
                            "checkpoint_step": 10,
                            "candidate_available": True,
                            "best_reward_seen": 0.4,
                            "expected_delta": 0.0,
                            "expected_std_delta": 0.0,
                        },
                    ],
                }
            }
        }
        for method in ("prompt_original", "zoib_raw"):
            (cache_root / f"{method}_test.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

    report = materialize_search_value(
        ("webshop", "paper_search"),
        run_root=run_root,
        output_root=tmp_path / "results",
    )
    assert Path(report["summary"]).is_file()
    assert Path(report["metadata"]).is_file()

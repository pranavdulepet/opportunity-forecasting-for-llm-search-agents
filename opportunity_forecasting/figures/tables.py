"""Recompute every table source referenced by the revised paper."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from opportunity_forecasting import REPO_ROOT

from opportunity_forecasting.evaluation.metrics import _keyed_label_rows, analyze
from opportunity_forecasting.figures.allocation import _normalized_auc
from opportunity_forecasting.figures.stopping import normalized_auc
from opportunity_forecasting.manifest import PAPER_CONFIG


DOMAIN_PATHS = {
    "webshop": "webshop",
    "paper_search": "paper_search",
}
DOMAIN_NAMES = {
    "webshop": "WebShop",
    "paper_search": "Paper Search",
}
METHODS = {
    "prompt_original": "Base Prompt",
    "zoib_raw": "Raw ZOIB",
    "zoib_remaining": "Support-aware ZOIB",
    "residual_scalar": "Scalar",
    "residual_gaussian": "Gaussian",
    "ridge_all": "Feature-only",
    "ridge_step": "Step-only",
    "empirical_reservation": "Reservation",
}
PLOT_LABELS = {
    "prompt_original": "Base Prompt",
    "zoib_raw": "ZOIB Regression",
    "zoib_remaining": "Support ZOIB",
    "residual_scalar": "Scalar Head",
    "residual_gaussian": "Gaussian Head",
    "ridge_all": "Feature ridge",
    "ridge_step": "Step ridge",
    "empirical_reservation": "Reservation",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def grouped_auc(
    path: Path,
    *,
    label_column: str,
    x_column: str,
    y_column: str,
) -> dict[str, float]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in read_csv(path):
        grouped.setdefault(row[label_column], []).append(row)
    out: dict[str, float] = {}
    for label, rows in grouped.items():
        if y_column == "mean_final_reward" and label_column == "model_label":
            out[label] = normalized_auc(rows)
        else:
            out[label] = _normalized_auc(
                [float(row[x_column]) for row in rows],
                [float(row[y_column]) for row in rows],
            )
    return out


def forecast_metrics(
    root: Path,
    cache_root: Path,
    domain: str,
) -> dict[str, dict[str, Any]]:
    labels = _keyed_label_rows(
        root
        / "data"
        / DOMAIN_PATHS[domain]
        / "labels"
        / "test.jsonl"
    )
    metrics: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        cache = cache_root / domain / f"{method}_test.json"
        row, _ = analyze(METHODS[method], cache, labels)
        metrics[method] = row
    retry, _ = analyze(
        "Prompt retry",
        cache_root / domain / "prompt_retry_test.json",
        labels,
    )
    metrics["prompt_retry"] = retry
    return metrics


def qrel_rows(root: Path, cache_root: Path) -> list[dict[str, Any]]:
    queries = [
        json.loads(line)
        for line in (
            root / "data" / "paper_search" / "export" / "queries.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qrels: dict[str, dict[str, float]] = {}
    with (
        root / "data" / "paper_search" / "export" / "qrels.tsv"
    ).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            qrels.setdefault(row["query_id"], {})[row["paper_id"]] = float(
                row["relevance"]
            )
    cache = json.loads(
        (
            cache_root / "paper_search" / "zoib_raw_test.json"
        ).read_text(encoding="utf-8")
    )
    labels: dict[int, list[dict[str, Any]]] = {}
    with (
        root / "data" / "paper_search" / "labels" / "test.jsonl"
    ).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            goal_id = int(row["goal_idx"])
            if str(goal_id) in cache["goal_predictions"]:
                labels.setdefault(goal_id, []).append(row)
    schedulable: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for goal_id, prediction in cache["goal_predictions"].items():
        numeric_goal_id = int(goal_id)
        checkpoints = prediction.get("checkpoints", [])
        label_stream = labels.get(numeric_goal_id, [])
        if len(checkpoints) != len(label_stream):
            raise ValueError(
                f"Paper Search stream mismatch for goal {goal_id}: "
                f"{len(checkpoints)} cache rows != {len(label_stream)} labels"
            )
        first_index = next(
            (
                index
                for index, checkpoint in enumerate(checkpoints)
                if bool(
                    checkpoint.get(
                        "candidate_available",
                        checkpoint.get("visited_product_page", False),
                    )
                )
            ),
            None,
        )
        if first_index is not None:
            schedulable[numeric_goal_id] = (
                label_stream[first_index],
                label_stream[-1],
            )
    rows: list[dict[str, Any]] = []
    for label, index in (
        ("First schedulable decision point", 0),
        ("Full fixed trajectory", 1),
    ):
        rewards: list[float] = []
        positive = 0
        top_grade = 0
        for goal_id, stream in schedulable.items():
            row = stream[index]
            rewards.append(float(row["input"]["best_reward_seen"]))
            opened = set(row["input"].get("opened_paper_ids", []))
            judgments = qrels.get(queries[goal_id]["queryid"], {})
            positive_ids = {
                paper_id
                for paper_id, grade in judgments.items()
                if grade > 0
            }
            highest = max(judgments.values()) if judgments else None
            top_ids = {
                paper_id
                for paper_id, grade in judgments.items()
                if highest is not None and grade == highest
            }
            positive += int(bool(opened & positive_ids))
            top_grade += int(bool(opened & top_ids))
        total = len(rewards)
        rows.append(
            {
                "held_out_state": label,
                "mean_reward": f"{sum(rewards) / total:.3f}",
                "qrel_positive_hits": positive,
                "qrel_positive_total": total,
                "qrel_positive_rate": f"{positive / total:.3f}",
                "top_grade_hits": top_grade,
                "top_grade_total": total,
                "top_grade_rate": f"{top_grade / total:.3f}",
            }
        )
    return rows


def experiment_settings_rows(manifest: dict[str, Any]) -> list[dict[str, str]]:
    protocol = manifest["protocol"]
    return [
        {
            "component": "Search model",
            "setting": (
                f"{protocol['backbone']['model_id']}; source trajectories use temperature "
                "0 and top-p=1; sampled continuations use temperature 0.7 and top-p=0.9; "
                "episode horizon 60"
            ),
        },
        {
            "component": "Data",
            "setting": (
                "Goal-disjoint split seed 123; six sampled continuations per labeled "
                "decision point; reward support [0,1]"
            ),
        },
        {
            "component": "Training",
            "setting": (
                "Three epochs; seed 42; LoRA rank 16, alpha 32, dropout 0.05; learning "
                "rate 1e-4 except Paper Search Gaussian at 1e-5; scalar selected by dev "
                "MAE and ZOIB/Gaussian by dev objective"
            ),
        },
        {
            "component": "Evaluation",
            "setting": (
                "Identical aligned held-out reward streams for every method; one shared "
                "stopping cost/risk sweep; normalized AUC over each domain's common "
                "observed support"
            ),
        },
        {
            "component": "Prompt parsing",
            "setting": (
                "Forecast metrics use valid parses with coverage reported separately; "
                "invalid allocation forecasts get zero priority; invalid stopping "
                "forecasts trigger fail-safe continue"
            ),
        },
        {
            "component": "Software",
            "setting": (
                "Python 3.11.14; PyTorch 2.5.1; Transformers 4.48.3; vLLM 0.7.3; "
                "NumPy 1.26.4; SciPy 1.16.3; scikit-learn 1.7.2; Matplotlib 3.10.7; "
                "WebShop generation used vLLM 0.8.5.post1"
            ),
        },
    ]


def build_tables(
    root: Path,
    manifest: dict[str, Any],
    *,
    cache_root: Path,
    result_root: Path,
    cost_paths: dict[str, Path],
) -> dict[str, list[dict[str, Any]]]:
    metrics = {
        domain: forecast_metrics(root, cache_root, domain)
        for domain in DOMAIN_PATHS
    }
    stopping = {
        domain: grouped_auc(
            result_root / "stopping" / f"{domain}.csv",
            label_column="model_label",
            x_column="mean_final_steps",
            y_column="mean_final_reward",
        )
        for domain in DOMAIN_PATHS
    }
    raw_allocation = {
        domain: grouped_auc(
            result_root / "absolute_reward" / f"{domain}.csv",
            label_column="model_label",
            x_column="mean_final_steps",
            y_column="mean_final_reward",
        )
        for domain in DOMAIN_PATHS
    }
    cost_allocation = {
        domain: grouped_auc(
            cost_paths[domain],
            label_column="series",
            x_column="mean_final_steps",
            y_column="mean_final_reward",
        )
        for domain in DOMAIN_PATHS
    }

    domain_summary = []
    evaluation_accounting = []
    for domain in DOMAIN_PATHS:
        spec = manifest["domains"][domain]
        evaluation = spec["evaluation"]
        domain_summary.append(
            {
                "domain": DOMAIN_NAMES[domain],
                "train_tasks": spec["splits"]["goal_ids"]["train"]["goals"],
                "dev_tasks": spec["splits"]["goal_ids"]["dev"]["goals"],
                "test_tasks": spec["splits"]["goal_ids"]["test"]["goals"],
                "train_labels": spec["labels"]["train"]["rows"],
                "dev_labels": spec["labels"]["dev"]["rows"],
                "test_labels": spec["labels"]["test"]["rows"],
            }
        )
        evaluation_accounting.append(
            {
                "domain": DOMAIN_NAMES[domain],
                "split_tasks": evaluation["source_tasks"],
                "test_labels": evaluation["test_labels"],
                "aligned_test_tasks": evaluation["aligned_test_goals"],
                "schedulable_streams": evaluation[
                    "schedulable_allocation_streams"
                ],
                "prompt_rows": evaluation["prompt_test_predictions"],
            }
        )

    head_methods = (
        "prompt_original",
        "zoib_raw",
        "residual_scalar",
        "residual_gaussian",
    )
    head_forecast = []
    for domain in DOMAIN_PATHS:
        for method in head_methods:
            row = metrics[domain][method]
            head_forecast.append(
                {
                    "domain": DOMAIN_NAMES[domain],
                    "method": METHODS[method],
                    "mae": f"{float(row['ev_mae']):.4f}",
                    "global_spearman": f"{float(row['spearman_ev']):.3f}",
                    "within_step_spearman": (
                        f"{float(row['within_step_bin_spearman']):.3f}"
                    ),
                }
            )

    head_auc = []
    for protocol_name, source in (
        ("Stopping", stopping),
        ("Raw allocation", raw_allocation),
    ):
        for domain in DOMAIN_PATHS:
            head_auc.append(
                {
                    "protocol": protocol_name,
                    "domain": DOMAIN_NAMES[domain],
                    **{
                        METHODS[method]: (
                            f"{source[domain][PLOT_LABELS[method]]:.4f}"
                        )
                        for method in head_methods
                    },
                }
            )

    position_controls = []
    for method in (
        "prompt_original",
        "zoib_raw",
        "ridge_step",
        "ridge_all",
        "empirical_reservation",
    ):
        row: dict[str, Any] = {"method": METHODS[method]}
        for domain in DOMAIN_PATHS:
            forecast = metrics[domain][method]
            prefix = domain
            row[f"{prefix}_global_spearman"] = (
                f"{float(forecast['spearman_ev']):.3f}"
            )
            row[f"{prefix}_within_step_spearman"] = (
                f"{float(forecast['within_step_bin_spearman']):.3f}"
            )
            row[f"{prefix}_stopping_auc"] = (
                f"{stopping[domain][PLOT_LABELS[method]]:.4f}"
            )
        position_controls.append(row)

    simple_controls = []
    for method in (
        "empirical_reservation",
        "zoib_raw",
        "zoib_remaining",
        "residual_gaussian",
    ):
        simple_controls.append(
            {
                "method": METHODS[method],
                "webshop_stopping_auc": (
                    f"{stopping['webshop'][PLOT_LABELS[method]]:.4f}"
                ),
                "paper_search_stopping_auc": (
                    f"{stopping['paper_search'][PLOT_LABELS[method]]:.4f}"
                ),
                "webshop_cost_normalized_allocation_auc": (
                    ""
                    if method == "empirical_reservation"
                    else f"{cost_allocation['webshop'][PLOT_LABELS[method]]:.4f}"
                ),
                "paper_search_cost_normalized_allocation_auc": (
                    ""
                    if method == "empirical_reservation"
                    else (
                        f"{cost_allocation['paper_search'][PLOT_LABELS[method]]:.4f}"
                    )
                ),
            }
        )

    parse_validity = []
    for domain in DOMAIN_PATHS:
        original = metrics[domain]["prompt_original"]
        retry = metrics[domain]["prompt_retry"]
        parse_validity.append(
            {
                "domain": DOMAIN_NAMES[domain],
                "original_valid_rate": f"{float(original['valid_rate']):.3f}",
                "retry_valid_rate": f"{float(retry['valid_rate']):.3f}",
                "original_mae": f"{float(original['ev_mae']):.4f}",
                "retry_mae": f"{float(retry['ev_mae']):.4f}",
            }
        )

    return {
        "domain_summary.csv": domain_summary,
        "evaluation_accounting.csv": evaluation_accounting,
        "head_auc.csv": head_auc,
        "head_forecast.csv": head_forecast,
        "paper_search_qrel_side_metrics.csv": qrel_rows(root, cache_root),
        "parse_validity_summary.csv": parse_validity,
        "position_controls.csv": position_controls,
        "experiment_settings.csv": experiment_settings_rows(manifest),
        "simple_controls.csv": simple_controls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(PAPER_CONFIG.read_text(encoding="utf-8"))
    cache_root = args.run_root / "predictions"
    result_root = args.run_root / "results"
    cost_paths = {
        domain: (
            args.run_root
            / "evaluations"
            / domain
            / "test"
            / "budgeted_expansion_cost_normalized"
            / "budgeted_expansion_curves.csv"
        )
        for domain in DOMAIN_PATHS
    }
    tables = build_tables(
        REPO_ROOT,
        manifest,
        cache_root=cache_root,
        result_root=result_root,
        cost_paths=cost_paths,
    )
    for name, rows in tables.items():
        write_csv(args.output_root / name, rows)
    print(
        json.dumps(
            {name: str(args.output_root / name) for name in tables},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

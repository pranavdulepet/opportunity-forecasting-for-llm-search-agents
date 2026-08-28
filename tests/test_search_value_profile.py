from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from opportunity_forecasting.figures.search_value import gain_with_budget, load_summary


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_search_value_uses_canonical_stream_support() -> None:
    root = ROOT / "results" / "search_value"
    rows = load_summary(root / "summary.csv")
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert len(rows) == 48
    assert metadata["domains"]["WebShop"]["num_streams"] == 365
    assert metadata["domains"]["WebShop"]["num_decision_points"] == 1242
    assert metadata["domains"]["Paper Search"]["num_streams"] == 360
    assert metadata["domains"]["Paper Search"]["num_decision_points"] == 1942
    webshop_first_bin = [
        row.mean_gain
        for row in rows
        if row.domain == "WebShop" and row.step_bin == "1-10"
    ]
    assert webshop_first_bin == pytest.approx(
        [0.0665633964, 0.0996156053, 0.1171547414, 0.1426731415]
    )


def test_finite_budget_gain_uses_environment_steps() -> None:
    stream = [
        SimpleNamespace(checkpoint_step=4, best_reward_seen=0.2),
        SimpleNamespace(checkpoint_step=9, best_reward_seen=0.3),
        SimpleNamespace(checkpoint_step=15, best_reward_seen=0.7),
    ]
    assert gain_with_budget(stream, 0, 5) == pytest.approx(0.1)
    assert gain_with_budget(stream, 0, 10) == pytest.approx(0.1)
    assert gain_with_budget(stream, 0, None) == pytest.approx(0.5)


def test_corrected_paper_search_accounting_is_packaged() -> None:
    table_root = ROOT / "results" / "tables"
    with (table_root / "evaluation_accounting.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        accounting = {row["domain"]: row for row in csv.DictReader(handle)}
    assert accounting["WebShop"]["aligned_test_tasks"] == "371"
    assert accounting["WebShop"]["schedulable_streams"] == "365"
    assert accounting["Paper Search"]["aligned_test_tasks"] == "364"
    assert accounting["Paper Search"]["schedulable_streams"] == "360"
    with (table_root / "paper_search_qrel_side_metrics.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        qrel_rows = list(csv.DictReader(handle))
    assert qrel_rows[0]["mean_reward"] == "0.287"
    assert qrel_rows[0]["qrel_positive_hits"] == "83"
    assert qrel_rows[0]["qrel_positive_total"] == "360"

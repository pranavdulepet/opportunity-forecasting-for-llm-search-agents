from __future__ import annotations

import json
from pathlib import Path

from opportunity_forecasting.data.splits import (
    paper_search_split,
    webshop_checkpoint_counts,
    webshop_split,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_paper_search_split_is_deterministic_and_goal_disjoint(
    tmp_path: Path,
) -> None:
    queries = tmp_path / "queries.jsonl"
    write_jsonl(queries, [{"query_id": index} for index in range(3_647)])
    first = paper_search_split(queries, seed=123)
    second = paper_search_split(queries, seed=123)
    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "train": 2_918,
        "dev": 365,
        "test": 364,
    }
    assigned = [goal_id for goal_ids in first.values() for goal_id in goal_ids]
    assert sorted(assigned) == list(range(3_647))
    assert len(assigned) == len(set(assigned))


def test_webshop_checkpoint_budget_split_is_goal_disjoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints.jsonl"
    rows = [
        {"goal_idx": goal_id, "checkpoint_step": step}
        for goal_id in range(8)
        for step in range(goal_id + 1)
    ]
    write_jsonl(checkpoint_path, rows)
    splits = webshop_split(
        checkpoint_path,
        seed=123,
        train_checkpoints=5,
        dev_checkpoints=5,
        test_checkpoints=5,
    )
    assigned = [goal_id for goal_ids in splits.values() for goal_id in goal_ids]
    assert sorted(assigned) == list(range(8))
    assert len(assigned) == len(set(assigned))


def test_webshop_checkpoint_counts_record_source(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoints.jsonl"
    write_jsonl(
        checkpoint_path,
        [
            {"goal_idx": 4},
            {"goal_idx": 4},
            {"goal_idx": 9},
            {"goal_idx": 9},
            {"goal_idx": 9},
        ],
    )
    counts, source = webshop_checkpoint_counts(checkpoint_path)
    assert counts == {4: 2, 9: 3}
    assert source["rows"] == 5

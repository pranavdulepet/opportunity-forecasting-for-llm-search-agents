"""Rebuild the goal-disjoint splits used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def paper_search_split(
    query_path: Path,
    *,
    seed: int = 123,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
) -> dict[str, list[int]]:
    query_count = len(read_jsonl(query_path))
    goal_ids = list(range(query_count))
    random.Random(seed).shuffle(goal_ids)
    train_count = int(round(train_fraction * query_count))
    dev_count = int(round(dev_fraction * query_count))
    return {
        "train": sorted(goal_ids[:train_count]),
        "dev": sorted(goal_ids[train_count : train_count + dev_count]),
        "test": sorted(goal_ids[train_count + dev_count :]),
    }


def webshop_checkpoint_counts(
    input_path: Path,
) -> tuple[dict[int, int], dict[str, Any]]:
    counts: dict[int, int] = defaultdict(int)
    digest = hashlib.sha256()
    rows = 0
    size = 0
    with input_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            digest.update(line)
            size += len(line)
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: expected a JSON object"
                )
            counts[int(row["goal_idx"])] += 1
            rows += 1
    counts = dict(counts)
    source = {
        "filename": input_path.name,
        "sha256": digest.hexdigest(),
        "bytes": size,
        "rows": rows,
    }
    if not counts or any(goal_id < 0 or count <= 0 for goal_id, count in counts.items()):
        raise ValueError("WebShop checkpoint counts must be positive")
    if int(source["rows"]) != sum(counts.values()):
        raise ValueError("WebShop checkpoint-count total does not match source rows")
    return counts, source


def webshop_split_from_counts(
    counts: dict[int, int],
    *,
    seed: int = 123,
    train_checkpoints: int = 20_000,
    dev_checkpoints: int = 2_000,
    test_checkpoints: int = 2_000,
) -> dict[str, list[int]]:
    goal_ids = sorted(counts)
    random.Random(seed).shuffle(goal_ids)
    targets = (
        ("train", train_checkpoints),
        ("dev", dev_checkpoints),
        ("test", test_checkpoints),
    )
    result = {name: [] for name, _ in targets}
    result["holdout"] = []
    phase = 0
    accumulated = 0
    for goal_id in goal_ids:
        if phase >= len(targets):
            result["holdout"].append(goal_id)
            continue
        name, target = targets[phase]
        result[name].append(goal_id)
        accumulated += counts[goal_id]
        if accumulated >= target:
            phase += 1
            accumulated = 0
    if phase != len(targets):
        raise ValueError("Checkpoint stream is too small for the requested targets")
    return {name: sorted(ids) for name, ids in result.items()}


def webshop_split(
    input_path: Path,
    *,
    seed: int = 123,
    train_checkpoints: int = 20_000,
    dev_checkpoints: int = 2_000,
    test_checkpoints: int = 2_000,
) -> dict[str, list[int]]:
    counts, _ = webshop_checkpoint_counts(input_path)
    return webshop_split_from_counts(
        counts,
        seed=seed,
        train_checkpoints=train_checkpoints,
        dev_checkpoints=dev_checkpoints,
        test_checkpoints=test_checkpoints,
    )


def compare_splits(
    actual: dict[str, list[int]],
    reference: dict[str, list[int]],
) -> None:
    if actual != reference:
        for split in sorted(set(actual) | set(reference)):
            if actual.get(split) != reference.get(split):
                raise ValueError(f"Split mismatch: {split}")
        raise ValueError("Split mismatch")


def write_goal_ids(output_dir: Path, splits: dict[str, list[int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, goal_ids in splits.items():
        (output_dir / f"{split}_goal_ids.json").write_text(
            json.dumps(goal_ids, indent=2) + "\n",
            encoding="utf-8",
        )


def write_paper_search(
    output_dir: Path,
    splits: dict[str, list[int]],
    *,
    query_path: Path,
    seed: int,
) -> None:
    write_goal_ids(output_dir, splits)
    metadata = {
        "seed": seed,
        "num_goals": sum(len(ids) for ids in splits.values()),
        "num_train": len(splits["train"]),
        "num_dev": len(splits["dev"]),
        "num_test": len(splits["test"]),
        "algorithm": (
            "shuffle query row indices with random.Random(seed), then round "
            "80/10/10 counts and sort each split"
        ),
        "paper_query_path": str(query_path),
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def write_webshop(
    output_dir: Path,
    splits: dict[str, list[int]],
    *,
    seed: int,
    source: dict[str, Any],
) -> None:
    write_goal_ids(output_dir, splits)
    metadata = {
        "algorithm": (
            "shuffle sorted goal IDs, then greedily assign whole goals until "
            "each checkpoint target is reached"
        ),
        "checkpoint_targets": {"train": 20_000, "dev": 2_000, "test": 2_000},
        "seed": seed,
        "num_goals_total": sum(len(ids) for ids in splits.values()),
        "split_goals": {name: len(ids) for name, ids in splits.items()},
        "source_checkpoint_stream": {
            key: source[key] for key in ("sha256", "bytes", "rows")
        },
    }
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("webshop", "paper_search"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()

    if args.domain == "paper_search":
        splits = paper_search_split(args.input, seed=args.seed)
        if args.reference:
            reference = {
                split: json.loads(
                    (args.reference / f"{split}_goal_ids.json").read_text(
                        encoding="utf-8"
                    )
                )
                for split in ("train", "dev", "test")
            }
            compare_splits(splits, reference)
        write_paper_search(args.output, splits, query_path=args.input, seed=args.seed)
    else:
        counts, source = webshop_checkpoint_counts(args.input)
        splits = webshop_split_from_counts(counts, seed=args.seed)
        if args.reference:
            reference = {
                split: json.loads(
                    (args.reference / f"{split}_goal_ids.json").read_text(
                        encoding="utf-8"
                    )
                )
                for split in ("train", "dev", "test", "holdout")
            }
            compare_splits(splits, reference)
        write_webshop(
            args.output,
            splits,
            seed=args.seed,
            source=source,
        )
    print(
        json.dumps(
            {
                "domain": args.domain,
                "output": str(args.output),
                "split_counts": {name: len(ids) for name, ids in splits.items()},
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

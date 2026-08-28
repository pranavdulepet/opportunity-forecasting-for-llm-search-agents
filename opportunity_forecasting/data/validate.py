"""Validate the datasets used to train and evaluate the paper's models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from opportunity_forecasting.manifest import (
    PAPER_CONFIG,
    load_manifest,
    resolve_artifact_path,
)


StateKey = tuple[int, int, int]
LABEL_FIELDS = {
    "goal_idx",
    "goal_text",
    "checkpoint_step",
    "input",
    "continuation_deltas",
    "metadata",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def row_input(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("input", {})
    return value if isinstance(value, dict) else {}


def goal_id(row: Mapping[str, Any]) -> int:
    return int(row.get("goal_idx", row_input(row).get("goal_idx", 0)) or 0)


def checkpoint_step(row: Mapping[str, Any]) -> int:
    return int(
        row.get("checkpoint_step", row_input(row).get("checkpoint_step", 0)) or 0
    )


def best_reward(row: Mapping[str, Any]) -> float:
    return float(
        row_input(row).get("best_reward_seen", row.get("best_reward_seen", 0.0))
        or 0.0
    )


def state_keys(path: Path) -> set[StateKey]:
    occurrences: dict[tuple[int, int], int] = defaultdict(int)
    keys: set[StateKey] = set()
    for row in jsonl_rows(path):
        base = (goal_id(row), checkpoint_step(row))
        occurrence = occurrences[base]
        occurrences[base] += 1
        key = (*base, occurrence)
        if key in keys:
            raise ValueError(f"Duplicate state key: {key}")
        keys.add(key)
    return keys


def validate_file(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    expected_size = int(spec["bytes"])
    if size != expected_size:
        raise ValueError(f"{path}: {size} bytes; expected {expected_size}")
    digest = sha256(path)
    if digest != str(spec["sha256"]):
        raise ValueError(f"{path}: checksum does not match configs/paper.json")
    result: dict[str, Any] = {"path": str(path), "bytes": size, "sha256": digest}
    if "rows" in spec:
        rows = sum(1 for _ in path.open("rb"))
        expected_rows = int(spec["rows"])
        if rows != expected_rows:
            raise ValueError(f"{path}: {rows} rows; expected {expected_rows}")
        result["rows"] = rows
    return result


def validate_labels(
    path: Path,
    *,
    expected_rows: int,
    reward_mode: str,
    continuations_per_state: int,
    horizon_steps: int,
) -> tuple[dict[str, Any], set[int], set[StateKey]]:
    continuation_counts: Counter[int] = Counter()
    reward_modes: Counter[str] = Counter()
    horizons: Counter[int] = Counter()
    goals: set[int] = set()
    keys: set[StateKey] = set()
    occurrences: dict[tuple[int, int], int] = defaultdict(int)
    row_count = 0
    for row in jsonl_rows(path):
        row_count += 1
        unexpected = set(row) - LABEL_FIELDS
        if unexpected:
            raise ValueError(
                f"{path}: unexpected label fields {sorted(unexpected)}"
            )
        deltas = [float(value) for value in row.get("continuation_deltas", [])]
        continuation_counts[len(deltas)] += 1
        metadata = row.get("metadata", {})
        reward_modes[str(metadata.get("reward_mode", ""))] += 1
        horizons[int(metadata.get("total_horizon_steps", 0) or 0)] += 1
        row_goal = goal_id(row)
        row_step = checkpoint_step(row)
        goals.add(row_goal)
        base = (row_goal, row_step)
        occurrence = occurrences[base]
        occurrences[base] += 1
        key = (*base, occurrence)
        if key in keys:
            raise ValueError(f"{path}: duplicate state key {key}")
        keys.add(key)
        maximum_gain = 1.0 - min(1.0, max(0.0, best_reward(row)))
        if any(
            not math.isfinite(delta)
            or delta < -1e-8
            or delta > maximum_gain + 1e-7
            for delta in deltas
        ):
            raise ValueError(f"{path}: continuation gain outside [0, 1-current_best]")

    if row_count != expected_rows:
        raise ValueError(f"{path}: {row_count} rows; expected {expected_rows}")
    expected_count = Counter({continuations_per_state: row_count})
    if continuation_counts != expected_count:
        raise ValueError(f"{path}: continuation counts {dict(continuation_counts)}")
    if reward_modes != Counter({reward_mode: row_count}):
        raise ValueError(f"{path}: reward modes {dict(reward_modes)}")
    if horizons != Counter({horizon_steps: row_count}):
        raise ValueError(f"{path}: horizons {dict(horizons)}")

    return (
        {
            "rows": row_count,
            "goals": len(goals),
            "continuations_per_state": continuations_per_state,
            "horizon_steps": horizon_steps,
            "reward_mode": reward_mode,
        },
        goals,
        keys,
    )


def validate_inputs(
    config_path: Path = PAPER_CONFIG,
    *,
    include_checkpoints: bool = True,
    include_environment: bool = False,
    webshop_asset_override: Path | None = None,
) -> dict[str, Any]:
    config = load_manifest(config_path)
    protocol = config["protocol"]
    report: dict[str, Any] = {"config": str(config_path), "domains": {}}
    split_goals: dict[str, dict[str, set[int]]] = defaultdict(dict)

    for domain, domain_config in config["domains"].items():
        domain_report: dict[str, Any] = {"labels": {}, "checkpoints": {}}
        label_keys_by_split: dict[str, set[StateKey]] = {}
        for split, file_spec in domain_config["labels"].items():
            path = resolve_artifact_path(config, file_spec)
            validate_file(path, file_spec)
            label_report, goals, keys = validate_labels(
                path,
                expected_rows=int(file_spec["rows"]),
                reward_mode=str(domain_config["reward_mode"]),
                continuations_per_state=int(
                    protocol["monte_carlo_continuations_per_state"]
                ),
                horizon_steps=int(protocol["continuation_horizon_steps"]),
            )
            domain_report["labels"][split] = label_report
            label_keys_by_split[split] = keys
            split_goals[domain][split] = goals

        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = split_goals[domain][left] & split_goals[domain][right]
            if overlap:
                raise ValueError(
                    f"{domain}: {len(overlap)} goals occur in both {left} and {right}"
                )

        for split, file_spec in domain_config["splits"]["goal_ids"].items():
            split_path = resolve_artifact_path(config, file_spec)
            validate_file(split_path, file_spec)
            expected_goals = {
                int(value)
                for value in json.loads(split_path.read_text(encoding="utf-8"))
            }
            if len(expected_goals) != int(file_spec["goals"]):
                raise ValueError(
                    f"{domain}/{split}: split file contains {len(expected_goals)} goals; "
                    f"expected {file_spec['goals']}"
                )
            if split_goals[domain][split] != expected_goals:
                raise ValueError(f"{domain}/{split}: labels do not match the goal split")
        split_metadata = domain_config["splits"]["metadata"]
        validate_file(resolve_artifact_path(config, split_metadata), split_metadata)

        if include_checkpoints:
            for split, file_spec in domain_config["checkpoints"].items():
                path = resolve_artifact_path(config, file_spec)
                file_report = validate_file(path, file_spec)
                missing = label_keys_by_split[split] - state_keys(path)
                if missing:
                    raise ValueError(
                        f"{domain}/{split}: {len(missing)} labeled states are absent "
                        "from the source trajectory"
                    )
                domain_report["checkpoints"][split] = {
                    **file_report,
                    "labeled_states_covered": len(label_keys_by_split[split]),
                }

        if include_environment:
            if "environment_asset" in domain_config:
                path = resolve_artifact_path(
                    config,
                    domain_config["environment_asset"],
                    override=webshop_asset_override,
                )
                domain_report["environment"] = validate_file(
                    path, domain_config["environment_asset"]
                )
            else:
                domain_report["environment"] = {
                    name: validate_file(resolve_artifact_path(config, spec), spec)
                    for name, spec in domain_config["environment_assets"].items()
                }
        report["domains"][domain] = domain_report

    report["status"] = "valid"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--skip-checkpoints", action="store_true")
    parser.add_argument("--include-environment", action="store_true")
    parser.add_argument("--webshop-asset", type=Path)
    args = parser.parse_args()
    report = validate_inputs(
        args.config,
        include_checkpoints=not bool(args.skip_checkpoints),
        include_environment=bool(args.include_environment),
        webshop_asset_override=args.webshop_asset,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

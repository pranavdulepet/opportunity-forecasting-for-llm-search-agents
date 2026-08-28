"""Merge modulo-sharded JSONL files into canonical global order."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


class MergeError(ValueError):
    pass


_SHARD_PATTERNS = (
    re.compile(r"(?:^|[^a-z0-9])shard[_-]?(\d+)[_-]of[_-]?(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|[^a-z0-9])sh(\d+)-of-(\d+)", re.IGNORECASE),
    re.compile(r"(?:^|[^0-9])(\d+)-of-(\d+)(?:[^0-9]|$)", re.IGNORECASE),
)
_SHARD_IDX = re.compile(
    r"(?:^|[^a-z0-9])shard[_-]?idx(?:=|[_-])?(\d+)", re.IGNORECASE
)
_NUM_SHARDS = re.compile(
    r"(?:^|[^a-z0-9])num[_-]?shards(?:=|[_-])?(\d+)", re.IGNORECASE
)


def _metadata_from_name(path: Path) -> tuple[int, int] | None:
    name = path.name
    idx_match = _SHARD_IDX.search(name)
    count_match = _NUM_SHARDS.search(name)
    if idx_match or count_match:
        if not (idx_match and count_match):
            raise MergeError(f"incomplete shard_idx/num_shards metadata in {path}")
        return int(idx_match.group(1)), int(count_match.group(1))
    for pattern in _SHARD_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _expand_paths(specs: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_spec in specs:
        spec = os.fspath(raw_spec)
        if glob.has_magic(spec):
            matches = sorted(Path(match) for match in glob.glob(spec))
            if not matches:
                raise MergeError(f"shard pattern matched no files: {spec}")
            paths.extend(matches)
        else:
            paths.append(Path(spec))
    if not paths:
        raise MergeError("no shard paths provided")
    normalized = [str(path.resolve()) for path in paths]
    if len(set(normalized)) != len(normalized):
        raise MergeError("the same shard path was provided more than once")
    for path in paths:
        if not path.is_file():
            raise MergeError(f"shard is not a file: {path}")
    return paths


def _assign_indices(
    paths: Sequence[Path],
    explicit_indices: Sequence[int] | None,
    num_shards: int | None,
) -> tuple[dict[int, Path], int]:
    metadata = [_metadata_from_name(path) for path in paths]
    encoded = [item for item in metadata if item is not None]

    if explicit_indices is not None:
        indices = [int(index) for index in explicit_indices]
        if len(indices) != len(paths):
            raise MergeError(
                f"received {len(indices)} shard indices for {len(paths)} shard paths"
            )
    elif encoded:
        if len(encoded) != len(paths):
            raise MergeError("cannot mix named and unnamed shards without explicit indices")
        indices = [item[0] for item in encoded]
    else:
        indices = list(range(len(paths)))

    encoded_counts = {item[1] for item in encoded}
    if len(encoded_counts) > 1:
        raise MergeError(f"inconsistent num_shards values: {sorted(encoded_counts)}")
    if num_shards is None:
        count = next(iter(encoded_counts)) if encoded_counts else len(paths)
    else:
        count = int(num_shards)
        if encoded_counts and encoded_counts != {count}:
            raise MergeError(
                f"--num-shards={count} conflicts with filename num_shards="
                f"{next(iter(encoded_counts))}"
            )
    if count <= 0:
        raise MergeError("num_shards must be positive")

    if explicit_indices is not None:
        for path, index, item in zip(paths, indices, metadata):
            if item is not None and item[0] != index:
                raise MergeError(
                    f"explicit shard index {index} conflicts with filename index "
                    f"{item[0]} for {path}"
                )

    by_index: dict[int, Path] = {}
    for index, path in zip(indices, paths):
        if not 0 <= index < count:
            raise MergeError(f"shard index {index} is outside [0, {count})")
        if index in by_index:
            raise MergeError(
                f"duplicate shard index {index}: {by_index[index]} and {path}"
            )
        by_index[index] = path

    missing = sorted(set(range(count)) - set(by_index))
    if missing:
        raise MergeError(f"missing shard indices: {missing}")
    return by_index, count


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise MergeError(f"duplicate JSON object key: {key!r}")
        obj[key] = value
    return obj


def _reject_constant(value: str) -> None:
    raise MergeError(f"non-standard JSON constant: {value}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MergeError(f"{path}:{line_number}: blank JSONL row")
                try:
                    value = json.loads(
                        line,
                        object_pairs_hook=_object_pairs,
                        parse_constant=_reject_constant,
                    )
                except (json.JSONDecodeError, MergeError) as exc:
                    raise MergeError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise MergeError(
                        f"{path}:{line_number}: expected a JSON object, "
                        f"got {type(value).__name__}"
                    )
                rows.append(value)
    except UnicodeDecodeError as exc:
        raise MergeError(f"{path}: invalid UTF-8: {exc}") from exc
    return rows


def _interleave(shards: dict[int, list[dict[str, Any]]], count: int) -> list[dict[str, Any]]:
    total = sum(len(rows) for rows in shards.values())
    for index in range(count):
        expected = max(0, (total + count - 1 - index) // count)
        actual = len(shards[index])
        if actual != expected:
            raise MergeError(
                f"shard {index} has {actual} rows; modulo sharding of {total} rows "
                f"across {count} shards requires {expected}"
            )
    return [
        shards[global_index % count][global_index // count]
        for global_index in range(total)
    ]


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _check_reference(
    rows: Sequence[dict[str, Any]], reference_path: Path
) -> None:
    reference = _read_jsonl(reference_path)
    if len(rows) != len(reference):
        raise MergeError(
            f"reference row count mismatch: merged={len(rows)}, "
            f"reference={len(reference)}"
        )
    for index, (actual, expected) in enumerate(zip(rows, reference)):
        if not _exact_equal(actual, expected):
            raise MergeError(f"reference mismatch at row {index}")


def _canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def merge_data_shards(
    shard_specs: Sequence[str | Path],
    output_path: str | Path,
    *,
    explicit_indices: Sequence[int] | None = None,
    num_shards: int | None = None,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    paths = _expand_paths(shard_specs)
    by_index, count = _assign_indices(paths, explicit_indices, num_shards)
    shard_rows = {index: _read_jsonl(path) for index, path in by_index.items()}
    rows = _interleave(shard_rows, count)

    reference = Path(reference_path) if reference_path is not None else None
    if reference is not None:
        if not reference.is_file():
            raise MergeError(f"reference is not a file: {reference}")
        _check_reference(rows, reference)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            for row in rows:
                handle.write(_canonical_line(row))
                handle.write("\n")
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    return {
        "num_shards": count,
        "output": str(output),
        "reference": str(reference) if reference is not None else None,
        "rows": len(rows),
        "shards": [
            {"index": index, "path": str(by_index[index]), "rows": len(shard_rows[index])}
            for index in range(count)
        ],
        "status": "ok",
    }


def _parse_indices(values: Sequence[str] | None) -> list[int] | None:
    if values is None:
        return None
    parts = [part for value in values for part in value.split(",") if part]
    try:
        return [int(part) for part in parts]
    except ValueError as exc:
        raise MergeError("--shard-indices must contain integers") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="*", help="Shard paths or glob patterns")
    parser.add_argument(
        "--shard-paths", "--shard_paths", nargs="+", default=[], help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--glob", "--shard-glob", dest="globs", action="append", default=[]
    )
    parser.add_argument("-o", "--output", "--output-path", "--output_path", required=True)
    parser.add_argument(
        "--shard-indices", "--shard_indices", nargs="+", help="Indices in input order"
    )
    parser.add_argument("--num-shards", "--num_shards", type=int)
    parser.add_argument("--reference", "--reference-path", "--reference_path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    specs = [*args.shards, *args.shard_paths, *args.globs]
    try:
        report = merge_data_shards(
            specs,
            args.output,
            explicit_indices=_parse_indices(args.shard_indices),
            num_shards=args.num_shards,
            reference_path=args.reference,
        )
    except (MergeError, OSError) as exc:
        print(json.dumps({"error": str(exc), "status": "error"}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

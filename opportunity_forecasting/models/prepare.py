"""Prepare the pinned Qwen model files used by the experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from opportunity_forecasting.manifest import (
    PAPER_CONFIG,
    load_manifest,
    resolve_backbone_path,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_files(root: Path, specs: dict[str, dict[str, object]]) -> None:
    for relative, spec in specs.items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_bytes = path.stat().st_size
        expected_bytes = int(spec["bytes"])
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Size mismatch for {path}: {actual_bytes} != {expected_bytes}"
            )
        actual_hash = sha256(path)
        expected_hash = str(spec["sha256"])
        if actual_hash != expected_hash:
            raise ValueError(
                f"Checksum mismatch for {path}: {actual_hash} != {expected_hash}"
            )


def materialize_file(source: Path, destination: Path, *, symlink: bool) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if symlink:
        destination.symlink_to(source.resolve())
        return
    try:
        os.link(source.resolve(), destination)
    except OSError:
        shutil.copy2(source.resolve(), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Use an already materialized snapshot instead of the Hugging Face cache.",
    )
    parser.add_argument(
        "--link-snapshot",
        action="store_true",
        help="Symlink snapshot files into output; useful on a shared cluster but not relocatable.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    backbone = manifest["protocol"]["backbone"]
    if not isinstance(backbone, dict):
        raise ValueError("The public manifest must define structured backbone provenance")
    output = (args.output_dir or resolve_backbone_path(manifest)).expanduser().resolve()
    official_files = dict(backbone["official_files"])
    if not args.verify_only:
        output.mkdir(parents=True, exist_ok=True)
        if args.snapshot_dir:
            snapshot = args.snapshot_dir.expanduser().resolve()
            if not snapshot.is_dir():
                raise FileNotFoundError(snapshot)
            for source in snapshot.iterdir():
                if source.is_file() or source.is_symlink():
                    materialize_file(
                        source,
                        output / source.name,
                        symlink=bool(args.link_snapshot),
                    )
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise SystemExit("Create the environment from environments/training.yml first") from exc
            snapshot_download(
                repo_id=str(backbone["model_id"]),
                revision=str(backbone["revision"]),
                local_dir=str(output),
                local_files_only=bool(args.local_files_only),
            )
    verify_files(output, official_files)

    provenance = {
        "model_id": backbone["model_id"],
        "revision": backbone["revision"],
        "output_dir": str(output),
        "snapshot_dir": str(args.snapshot_dir.resolve()) if args.snapshot_dir else None,
        "snapshot_linked": bool(args.snapshot_dir and args.link_snapshot),
        "file_sha256": {
            relative: str(spec["sha256"])
            for relative, spec in official_files.items()
        },
    }
    if not args.verify_only:
        (output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "verified" if args.verify_only else "prepared",
                "output_dir": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

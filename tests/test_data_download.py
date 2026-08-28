from __future__ import annotations

import zipfile
from pathlib import Path

from opportunity_forecasting.data.download import prepare_canonical_data, sha256


def test_prepare_canonical_data_extracts_and_verifies_archive(tmp_path: Path) -> None:
    payloads = {
        "data/webshop/labels/train.jsonl": b'{"label": 1}\n',
        "data/webshop/checkpoints/train.jsonl": b'{"state": 1}\n',
        "data/paper_search/labels/train.jsonl": b'{"label": 2}\n',
        "data/paper_search/checkpoints/train.jsonl": b'{"state": 2}\n',
        "data/paper_search/export/queries.jsonl": b'{"query": 1}\n',
    }
    source_root = tmp_path / "source"
    specs = {}
    for relative, payload in payloads.items():
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        specs[relative] = {
            "path": relative,
            "bytes": len(payload),
            "sha256": sha256(path),
        }

    archive = tmp_path / "canonical-data.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in payloads:
            bundle.write(source_root / relative, relative)

    manifest = {
        "canonical_data_archive": {
            "bytes": archive.stat().st_size,
            "sha256": sha256(archive),
            "url": "unused",
            "paths": list(payloads),
        },
        "domains": {
            "webshop": {
                "labels": {"train": specs["data/webshop/labels/train.jsonl"]},
                "checkpoints": {
                    "train": specs["data/webshop/checkpoints/train.jsonl"]
                },
            },
            "paper_search": {
                "labels": {
                    "train": specs["data/paper_search/labels/train.jsonl"]
                },
                "checkpoints": {
                    "train": specs["data/paper_search/checkpoints/train.jsonl"]
                },
                "environment_assets": {
                    "queries": specs["data/paper_search/export/queries.jsonl"]
                },
            },
        },
    }
    output_root = tmp_path / "output"
    report = prepare_canonical_data(
        manifest, output_root=output_root, archive_path=archive
    )
    assert report["status"] == "prepared"
    assert report["files"] == len(payloads)
    assert all(
        (output_root / relative).read_bytes() == payload
        for relative, payload in payloads.items()
    )
    assert prepare_canonical_data(
        manifest, output_root=output_root, archive_path=archive
    )["status"] == "already_verified"

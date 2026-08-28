"""Prepare pinned external dependencies not stored in the repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

from opportunity_forecasting.manifest import PAPER_CONFIG, REPO_ROOT, load_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def verify_file(path: Path, spec: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(spec["bytes"]) or sha256(path) != spec["sha256"]:
        raise ValueError(f"External asset mismatch: {path}")


def download_webshop_data(spec: dict, root: Path) -> None:
    source = spec["data_source"]
    assets = spec["data_assets"]
    for name in ("items_shuffle", "items_ins_v2"):
        asset = assets[name]
        hf_hub_download(
            repo_id=source["repo_id"],
            repo_type=source["repo_type"],
            revision=source["revision"],
            filename=asset["filename"],
            local_dir=root / "data",
        )
        verify_file(REPO_ROOT / asset["path"], asset)
    human = assets["items_human_ins"]
    source_path = root / human["source_path"]
    output_path = REPO_ROOT / human["path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, output_path)
    verify_file(output_path, human)


def setup_webshop(
    manifest: dict,
    *,
    download_data: bool,
    build_index: bool,
    index_threads: int,
) -> None:
    spec = manifest["third_party"]["webshop"]
    root = REPO_ROOT / spec["root"]
    if not (root / ".git").is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", spec["url"], str(root)])
    run(["git", "fetch", "--all", "--tags"], cwd=root)
    run(["git", "checkout", "--detach", spec["commit"]], cwd=root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if commit != spec["commit"]:
        raise ValueError(f"WebShop commit mismatch: {commit}")
    if download_data:
        download_webshop_data(spec, root)
    if build_index:
        for asset in spec["data_assets"].values():
            verify_file(REPO_ROOT / asset["path"], asset)
        run(
            [
                sys.executable,
                "-m",
                "opportunity_forecasting.data.webshop_index",
                "--webshop-root",
                str(root),
                "--threads",
                str(index_threads),
            ],
            cwd=REPO_ROOT,
        )
    print(
        json.dumps(
            {
                "commit": commit,
                "data": "verified" if download_data or build_index else "not_requested",
                "index": "built" if build_index else "not_requested",
                "status": "prepared",
                "webshop_root": str(root),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--download-webshop-data", action="store_true")
    parser.add_argument("--build-webshop-index", action="store_true")
    parser.add_argument("--index-threads", type=int, default=8)
    args = parser.parse_args()
    setup_webshop(
        load_manifest(args.manifest),
        download_data=bool(args.download_webshop_data),
        build_index=bool(args.build_webshop_index),
        index_threads=int(args.index_threads),
    )


if __name__ == "__main__":
    main()

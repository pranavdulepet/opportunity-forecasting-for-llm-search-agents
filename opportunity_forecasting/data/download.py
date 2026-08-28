"""Download and verify the canonical paper dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Mapping

from opportunity_forecasting import REPO_ROOT
from opportunity_forecasting.manifest import PAPER_CONFIG, load_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_file_specs(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    available: dict[str, Mapping[str, Any]] = {}
    for domain in manifest["domains"].values():
        for group in ("labels", "checkpoints"):
            for spec in domain[group].values():
                available[str(spec["path"])] = spec
        for spec in domain.get("environment_assets", {}).values():
            available[str(spec["path"])] = spec
    paths = [str(path) for path in manifest["canonical_data_archive"]["paths"]]
    missing = sorted(set(paths) - set(available))
    if missing:
        raise ValueError(f"Archive paths are absent from the data manifest: {missing}")
    return {path: available[path] for path in paths}


def verify_file(path: Path, spec: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(spec["bytes"])
        and sha256(path) == str(spec["sha256"])
    )


def download_archive(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    partial.replace(destination)


def extract_archive(
    archive: Path,
    specs: Mapping[str, Mapping[str, Any]],
    *,
    output_root: Path,
) -> None:
    expected = set(specs)
    with zipfile.ZipFile(archive) as bundle:
        actual = {info.filename for info in bundle.infolist() if not info.is_dir()}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"Archive members differ from manifest: missing={missing}, extra={extra}"
            )
        for relative, spec in specs.items():
            destination = (output_root / relative).resolve()
            if output_root.resolve() not in destination.parents:
                raise ValueError(f"Unsafe archive path: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            with bundle.open(relative) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            if not verify_file(partial, spec):
                partial.unlink(missing_ok=True)
                raise ValueError(f"Extracted file does not match manifest: {relative}")
            partial.replace(destination)


def prepare_canonical_data(
    manifest: Mapping[str, Any],
    *,
    output_root: Path = REPO_ROOT,
    archive_path: Path | None = None,
) -> dict[str, Any]:
    specs = canonical_file_specs(manifest)
    if specs and all(
        verify_file(output_root / relative, spec)
        for relative, spec in specs.items()
    ):
        return {"files": len(specs), "status": "already_verified"}

    archive_spec = manifest["canonical_data_archive"]
    archive = archive_path
    if archive is None:
        archive = output_root / ".artifacts" / "canonical-data.zip"
        if not archive.is_file() or not verify_file(archive, archive_spec):
            download_archive(str(archive_spec["url"]), archive)
    archive = archive.expanduser().resolve()
    if not verify_file(archive, archive_spec):
        raise ValueError(f"Canonical data archive does not match the manifest: {archive}")

    extract_archive(archive, specs, output_root=output_root)
    if not all(verify_file(output_root / relative, spec) for relative, spec in specs.items()):
        raise ValueError("Canonical data verification failed after extraction")
    return {
        "archive": str(archive),
        "files": len(specs),
        "status": "prepared",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PAPER_CONFIG)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    report = prepare_canonical_data(
        load_manifest(args.manifest),
        output_root=args.output_root.expanduser().resolve(),
        archive_path=args.archive,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

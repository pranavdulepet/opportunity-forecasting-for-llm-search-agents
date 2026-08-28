"""Load the paper configuration and resolve its data and model paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


from opportunity_forecasting import REPO_ROOT
PAPER_CONFIG = REPO_ROOT / "configs" / "paper.json"


def load_manifest(path: Path = PAPER_CONFIG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def resolve_artifact_path(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    override: Path | None = None,
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    path = Path(str(spec["path"])).expanduser()
    return path if path.is_absolute() else repo_root / path


def resolve_backbone_path(
    manifest: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> Path:
    override = os.environ.get("OPPORTUNITY_BASE_MODEL")
    if override:
        return Path(override).expanduser().resolve()
    path = Path(str(manifest["protocol"]["backbone"]["path"])).expanduser()
    return path if path.is_absolute() else repo_root / path


def resolve_model_reference(reference: str, *, repo_root: Path = REPO_ROOT) -> str:
    """Resolve a repository-relative model path while preserving Hub model IDs."""
    value = str(reference).strip()
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = repo_root / path
    if candidate.exists() or value.startswith("runs/"):
        return str(candidate)
    return value

"""Download EDGAR-CORPUS split files from HuggingFace with a local manifest.

The dataset uses *script-based* loading, so the Hub's rows API is unavailable;
the reliable path is to fetch the raw jsonl files directly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag_ingest.config import SPLIT_TO_FILE, Settings
from rag_ingest.logging_setup import get_logger

log = get_logger(__name__)


def _manifest_path(settings: Settings) -> Path:
    return settings.dataset_dir / "manifest.json"


def _read_manifest(settings: Settings) -> dict:
    path = _manifest_path(settings)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"repo_id": settings.edgar_repo_id, "year": settings.edgar_year, "files": {}}


def _write_manifest(settings: Settings, manifest: dict) -> None:
    settings.dataset_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(settings).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def download_split(settings: Settings, split: str, *, force: bool = False) -> Path:
    """Download one split (train/validation/test) and return its local path."""
    from huggingface_hub import hf_hub_download

    filename = f"{settings.edgar_year}/{SPLIT_TO_FILE.get(split, split)}.jsonl"
    target = settings.dataset_dir / filename
    if target.exists() and not force:
        log.info("already present: %s", target)
        return target

    log.info("downloading %s from %s ...", filename, settings.edgar_repo_id)
    resolved = Path(
        hf_hub_download(
            repo_id=settings.edgar_repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(settings.dataset_dir),
        )
    )
    # hf_hub_download preserves the repo path (dataset/2020/test.jsonl); flatten
    # it to dataset/test.jsonl so it matches the shipped layout.
    target = settings.dataset_dir / f"{SPLIT_TO_FILE.get(split, split)}.jsonl"
    if resolved != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved.replace(target)
        resolved = target
    size = resolved.stat().st_size if resolved.exists() else 0

    manifest = _read_manifest(settings)
    manifest["files"][split] = {
        "path": str(resolved.relative_to(settings.dataset_dir)),
        "bytes": size,
        "fetched_at": datetime.now(UTC).isoformat(),
    }
    _write_manifest(settings, manifest)
    log.info("downloaded %s (%.1f MiB)", resolved, size / (1024 * 1024))
    return resolved

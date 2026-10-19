#!/usr/bin/env python
"""Standalone chunking + ingestion runner with tqdm progress bars.

Examples
--------
    # everything (test + validate + train)
    python scripts/ingest.py test validate train

    # just test, quick smoke of 50 documents
    python scripts/ingest.py test --limit 50

    # force CPU (slower) if the GPU is busy
    python scripts/ingest.py test --device cpu

Notes
-----
* Runs the real pipeline: extract -> clean -> chunk -> embed (BGE, fp16) ->
  Postgres/pgvector, one ACID transaction per document.
* Re-running is safe: unchanged documents are skipped via content hash.
* The model is loaded once and reused across all requested splits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tqdm import tqdm  # noqa: E402

from rag_ingest.config import Settings  # noqa: E402
from rag_ingest.domain.models import IngestStats  # noqa: E402
from rag_ingest.logging_setup import configure_logging, get_logger  # noqa: E402
from rag_ingest.migrate import upgrade_to_head  # noqa: E402
from rag_ingest.wiring import build_embedder, build_pipeline  # noqa: E402

log = get_logger("scripts.ingest")


def count_lines(path: Path) -> int:
    """Count records in a jsonl file (for an exact total on the progress bar)."""
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def ingest_split(settings: Settings, split: str, embedder, limit: int | None) -> IngestStats:
    path = settings.dataset_file(split)
    if not path.exists():
        raise FileNotFoundError(f"missing dataset file for split '{split}': {path}")

    total = count_lines(path)
    if limit is not None:
        total = min(total, limit)

    pipeline = build_pipeline(settings, split, embedder=embedder)
    bar = tqdm(
        total=total,
        desc=f"{split:<8}",
        unit="doc",
        dynamic_ncols=True,
        bar_format=(
            "{l_bar}{bar}| {n_fmt}/{total_fmt} "
            "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ),
    )

    def on_progress(stats: IngestStats) -> None:
        bar.n = stats.documents_seen
        bar.set_postfix(
            indexed=stats.documents_indexed,
            skipped=stats.documents_skipped,
            failed=stats.documents_failed,
            chunks=f"{stats.chunks_written:,}",
            refresh=False,
        )
        bar.refresh()

    started = time.perf_counter()
    try:
        stats = pipeline.run(limit=limit, on_progress=on_progress)
    except KeyboardInterrupt:
        bar.close()
        log.warning("interrupted during %s (partial progress is committed per document)", split)
        raise
    elapsed = time.perf_counter() - started
    bar.close()

    rate = stats.documents_seen / elapsed if elapsed else 0.0
    print(
        f"  {split}: {stats.documents_indexed} indexed, "
        f"{stats.documents_skipped} skipped, {stats.documents_failed} failed, "
        f"{stats.chunks_written:,} chunks in {_fmt(elapsed)} "
        f"({rate:.2f} doc/s)"
    )
    print(f"  run id: {pipeline.run_id}\n")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("splits", nargs="+", help="one or more of: test validate train")
    parser.add_argument("--limit", type=int, default=0, help="max documents per split (0 = all)")
    parser.add_argument(
        "--device",
        default=None,
        help="override embedding device, e.g. cuda or cpu (default: settings)",
    )
    parser.add_argument(
        "--no-migrate",
        action="store_true",
        help="skip 'alembic upgrade head' (use if the schema is already current)",
    )
    args = parser.parse_args()

    overrides = {}
    if args.device:
        overrides["embed_device"] = args.device
    settings = Settings(**overrides)
    configure_logging(settings.log_level)

    print(f"device: {settings.embed_device} | batch: {settings.embed_batch_size} | "
          f"chunk target/overlap/max: "
          f"{settings.chunk_target_tokens}/{settings.chunk_overlap_tokens}/{settings.chunk_max_tokens}")

    if not args.no_migrate:
        log.info("applying database migrations (alembic upgrade head)")
        upgrade_to_head(settings)

    limit = args.limit or None

    print("loading embedding model ...")
    embedder = build_embedder(settings)

    grand = IngestStats()
    try:
        for split in args.splits:
            stats = ingest_split(settings, split, embedder, limit)
            grand.documents_seen += stats.documents_seen
            grand.documents_indexed += stats.documents_indexed
            grand.documents_skipped += stats.documents_skipped
            grand.documents_failed += stats.documents_failed
            grand.chunks_written += stats.chunks_written
    except KeyboardInterrupt:
        return 130

    print("TOTAL:", grand.model_dump())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface.

    rag-ingest init-db
    rag-ingest download --split test --split validation
    rag-ingest ingest --split test --limit 25
    rag-ingest status
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from rag_ingest.adapters.sources.edgar_download import download_split
from rag_ingest.adapters.storage.postgres.engine import (
    build_engine,
    build_session_factory,
    ping,
)
from rag_ingest.config import Settings
from rag_ingest.logging_setup import configure_logging
from rag_ingest.migrate import upgrade_to_head
from rag_ingest.wiring import build_settings

app = typer.Typer(help="Enterprise RAG ingestion pipeline (SEC EDGAR 10-K).", no_args_is_help=True)
console = Console()


def _settings() -> Settings:
    settings = build_settings()
    configure_logging(settings.log_level)
    return settings


@app.command("init-db")
def init_db_command() -> None:
    """Apply Alembic migrations up to head (creates tables + pgvector extension)."""
    settings = _settings()
    engine = build_engine(settings.postgres_dsn)
    console.print(f"connected: [dim]{ping(engine)[:80]}[/dim]")
    upgrade_to_head(settings)
    console.print("[green]schema at migration head[/green]")


@app.command()
def download(
    split: Annotated[
        list[str] | None, typer.Option("--split", help="train/validation/test")
    ] = None,
    force: Annotated[bool, typer.Option(help="Re-download even if present")] = False,
) -> None:
    """Download EDGAR-CORPUS jsonl splits into the datasets directory."""
    settings = _settings()
    for name in split or ["test"]:
        path = download_split(settings, name, force=force)
        console.print(f"[green]{name}[/green] -> {path}")


@app.command()
def ingest(
    split: Annotated[str, typer.Option(help="Which split to ingest")] = "test",
    limit: Annotated[int, typer.Option(help="Max documents (0 = no limit)")] = 0,
    run_id: Annotated[str | None, typer.Option(help="Override the run id")] = None,
) -> None:
    """Run the ingestion pipeline for one split."""
    from rag_ingest.wiring import build_pipeline

    settings = _settings()
    pipeline = build_pipeline(settings, split, run_id=run_id)
    stats = pipeline.run(limit=limit or None)
    console.print_json(data=stats.model_dump())
    console.print(f"run id: [cyan]{pipeline.run_id}[/cyan]")


@app.command()
def status() -> None:
    """Show row counts in the system of record."""
    from sqlalchemy import func, select

    from rag_ingest.adapters.storage.postgres.tables import (
        ChunkRow,
        DocumentRow,
        EmbeddingRow,
        IngestRunRow,
    )

    settings = _settings()
    engine = build_engine(settings.postgres_dsn)
    factory = build_session_factory(engine)
    table = Table(title="rag_ingest store")
    table.add_column("table")
    table.add_column("rows", justify="right")
    with factory() as session:
        for name, model in (
            ("documents", DocumentRow),
            ("chunks", ChunkRow),
            ("embeddings", EmbeddingRow),
            ("ingest_runs", IngestRunRow),
        ):
            count = session.scalar(select(func.count()).select_from(model)) or 0
            table.add_row(name, str(count))
    console.print(table)


if __name__ == "__main__":
    app()

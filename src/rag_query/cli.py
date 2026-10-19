"""Command line interface for the query pipeline.

    rag-ask ask "What are Apple's main risk factors?"
    rag-ask ask "how about 2020?" --session demo --token dev-public
    rag-ask ask "risk factors" --cik 320193 --section 1A
    rag-ask ask "risk factors" --stream
    rag-ask feedback <request_id> --rating 1 --token dev-public
    rag-ask serve --port 8080

The service exits non-zero when the request did not produce a normal answer
(including a safe output-guard fallback), so the CLI composes with pipelines.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from rag_query.domain.models import (
    AskRequest,
    AskResult,
    QueryStatus,
    RetrievalFilters,
)
from rag_query.logging import configure_logging
from rag_query.service import QueryService
from rag_query.wiring import (
    build_authenticator,
    build_history_store,
    build_service,
    build_settings,
)

app = typer.Typer(
    help="Enterprise RAG query pipeline (Pipeline 2, steps 1-14).",
    no_args_is_help=True,
)
console = Console()

_STATUS_STYLE = {
    QueryStatus.ANSWERED: "green",
    QueryStatus.ENCODED: "green",
    QueryStatus.NO_CONTEXT: "yellow",
    QueryStatus.NOT_ANSWERABLE: "yellow",
    QueryStatus.REJECTED: "yellow",
    QueryStatus.RATE_LIMITED: "yellow",
    QueryStatus.OUTPUT_BLOCKED: "yellow",
    QueryStatus.UNAUTHORIZED: "red",
    QueryStatus.BLOCKED: "red",
    QueryStatus.ERROR: "red",
}


@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="The question to ask.")],
    token: Annotated[
        str | None,
        typer.Option("--token", "-t", help="Dev token (see RAG_DEV_TOKENS)."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Session id, for conversation history."),
    ] = None,
    cik: Annotated[
        str | None,
        typer.Option("--cik", help="Comma-separated CIK filter(s), e.g. 320193."),
    ] = None,
    year: Annotated[
        str | None,
        typer.Option("--year", help="Comma-separated filing year filter(s), e.g. 2020."),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Comma-separated Item key filter(s), e.g. 1A,7."),
    ] = None,
    vector: Annotated[bool, typer.Option("--vector", help="Include the full embedding.")] = False,
    stream: Annotated[
        bool,
        typer.Option(
            "--stream",
            help="Verified streaming: print the guarded answer as it arrives.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
) -> None:
    """Run one question through the full retrieval-augmented pipeline."""
    if stream and as_json:
        raise typer.BadParameter("--stream and --json are mutually exclusive")

    settings = build_settings()
    configure_logging(settings.log_level)
    service = build_service(settings)

    filters = RetrievalFilters(
        ciks=_split(cik),
        years=_split(year),
        section_keys=tuple(value.upper() for value in _split(section)),
    )
    request = AskRequest(
        query=query,
        session_id=session,
        token=token,
        filters=filters if not filters.is_empty else None,
    )

    if stream:
        _stream(service, request)
        raise typer.Exit(code=0)

    result = service.ask(request, include_vector=vector)
    if as_json:
        console.print_json(data=result.model_dump(mode="json"))
    else:
        _render(result)
    raise typer.Exit(code=0 if result.status.ok else 1)


@app.command()
def feedback(
    request_id: Annotated[str, typer.Argument(help="The request_id returned by ask.")],
    rating: Annotated[
        int,
        typer.Option("--rating", "-r", help="+1 for helpful, -1 for not helpful."),
    ],
    comment: Annotated[str, typer.Option("--comment", "-c", help="Optional comment.")] = "",
    token: Annotated[
        str | None,
        typer.Option("--token", "-t", help="Dev token (see RAG_DEV_TOKENS)."),
    ] = None,
) -> None:
    """Record thumbs-up/down for a previously answered turn."""
    settings = build_settings()
    configure_logging(settings.log_level)
    store = build_history_store(settings)
    if store is None:
        console.print("[red]History storage is disabled (RAG_HISTORY_ENABLED=false).[/red]")
        raise typer.Exit(code=1)

    # Feedback needs only auth + storage; do not load the embedding/LLM stack.
    service = QueryService(
        None,
        settings=settings,
        authenticator=build_authenticator(settings),
        history_store=store,
        feedback_admin_group=settings.feedback_admin_group,
    )
    result = service.feedback(token, request_id, rating, comment)
    style = "green" if result.status == "recorded" else "red"
    console.print(f"[{style}]{result.status}[/{style}]: {result.message}")
    raise typer.Exit(code=0 if result.status == "recorded" else 1)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8080,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Start the FastAPI app (loads the models on startup)."""
    import uvicorn

    if reload:
        uvicorn.run("rag_query.api:create_app", factory=True, host=host, port=port, reload=True)
    else:
        from rag_query.api import create_app

        uvicorn.run(create_app(), host=host, port=port)


def _split(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _stream(service: QueryService, request: AskRequest) -> None:
    final: AskResult | None = None
    for event in service.ask_stream(request):
        if event["event"] == "token":
            console.print(event["data"].get("text", ""), end="", markup=False, highlight=False)
        elif event["event"] == "done":
            final = AskResult.model_validate(event["data"])
    console.print()
    if final is None:
        console.print("[red]The stream ended without a final result.[/red]")
        return
    _render_sources(final)
    _render_diagnostics(final)
    raise typer.Exit(code=0 if final.status.ok else 1)


def _render(result: AskResult) -> None:
    if result.answer:
        console.print(Panel(Markdown(result.answer), title="answer", title_align="left"))
    elif result.message:
        console.print(Panel(result.message, title="message", title_align="left"))

    _render_sources(result)

    table = Table(title=f"query {result.request_id}", show_header=False, title_justify="left")
    table.add_column("field", style="bold", no_wrap=True)
    table.add_column("value", overflow="fold")

    style = _STATUS_STYLE.get(result.status, "white")
    table.add_row("status", f"[{style}]{result.status.value}[/{style}]")
    table.add_row("query", result.query)
    if result.rewritten_query is not None:
        table.add_row("rewritten", result.rewritten_query)
    if result.principal is not None:
        who = f"{result.principal.user_id}  groups={list(result.principal.groups)}"
        table.add_row("principal", who)
    if result.guard is not None:
        detail = result.guard.reason or "ok"
        table.add_row("guard", f"{result.guard.guard}: {detail}")
    if result.route is not None:
        table.add_row("route", f"answerable={result.route.answerable} ({result.route.intent})")
    if result.retrieval is not None:
        retrieval = result.retrieval
        table.add_row(
            "retrieval",
            f"dense={retrieval.dense_candidates} lexical={retrieval.lexical_candidates} "
            f"fused={retrieval.fused_candidates} reranked={retrieval.reranked} "
            f"top={_score(retrieval.top_score)}",
        )
        if not retrieval.filters.is_empty:
            table.add_row("filters", str(retrieval.filters.model_dump(exclude_defaults=True)))
    if result.output_guard is not None:
        verdict = result.output_guard
        state = "pass" if verdict.allowed else "FAIL"
        table.add_row("output_guard", f"{state}: {verdict.reason}")
    if result.message and result.message != result.answer:
        table.add_row("message", result.message)
    if result.turn_id:
        table.add_row("turn_id", result.turn_id)
    if result.embedding_dim is not None:
        norm = f"{result.embedding_norm:.4f}" if result.embedding_norm is not None else "?"
        table.add_row(
            "embedding",
            f"dim={result.embedding_dim} model={result.embedding_model} norm={norm}",
        )
    table.add_row("trace", " -> ".join(result.trace) or "(none)")
    console.print(table)

    _render_diagnostics(result)
    if result.embedding_preview:
        console.print(f"[dim]vector[:8] = {result.embedding_preview}[/dim]")


def _render_sources(result: AskResult) -> None:
    if not result.citations:
        return
    table = Table(title="sources", title_justify="left")
    table.add_column("id", style="bold", no_wrap=True)
    table.add_column("company / filing", overflow="fold")
    table.add_column("section", overflow="fold")
    table.add_column("score", justify="right", no_wrap=True)
    for citation in result.citations:
        company = citation.company or f"CIK {citation.cik}"
        score = citation.rerank_score if citation.rerank_score is not None else citation.score
        table.add_row(
            f"[{citation.source_id}]",
            f"{company}\n{citation.doc_id} ({citation.year})",
            f"Item {citation.section_key}: {citation.section_title}",
            _score(score),
        )
    console.print(table)
    for citation in result.citations:
        console.print(f"[dim]{citation.source_id} {citation.url}[/dim]")


def _render_diagnostics(result: AskResult) -> None:
    if result.warnings:
        console.print(f"[yellow]warnings:[/yellow] {result.warnings}")
    if result.timings_ms:
        timings = "  ".join(f"{k}={v:.1f}ms" for k, v in result.timings_ms.items())
        console.print(f"[dim]timings: {timings}[/dim]")


def _score(value: float | None) -> str:
    return "?" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    app()

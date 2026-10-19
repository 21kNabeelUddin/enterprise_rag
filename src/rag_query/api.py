"""FastAPI wrapper around the same query service the CLI uses.

    uvicorn rag_query.api:create_app --factory --port 8080

Auth is read from the request headers (``X-Dev-Token`` or a ``Bearer`` token);
the models are built once on startup and reused for every request. Blocking
endpoints are plain ``def`` so FastAPI runs the GPU/LLM work in a threadpool.

``/ask/stream`` is verified streaming: the full pipeline (including the output
guard) runs first, then the safe answer is sent as SSE ``token`` events followed
by one ``done`` event carrying the complete :class:`AskResult`.

A dependency-free test console is served at ``/`` (and ``/ui``). It is a single
static HTML file that calls the same endpoints, so it exercises the real
pipeline -- guards, filters, history, citations and verified streaming included.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag_query.config import QuerySettings
from rag_query.domain.models import (
    AskRequest,
    AskResult,
    FeedbackResult,
    QueryStatus,
    RetrievalFilters,
)
from rag_query.logging import configure_logging
from rag_query.service import QueryService
from rag_query.wiring import build_service, build_settings

# The endpoint always returns the full AskResult envelope, but the HTTP status
# tracks the pipeline's terminal state so callers can branch on it without
# parsing the body. ``answered``, ``no_context``, ``output_blocked`` and
# ``not_answerable`` are all successful HTTP exchanges; the body says whether a
# verified answer was produced.
_STATUS_TO_HTTP: dict[QueryStatus, int] = {
    QueryStatus.REJECTED: 400,
    QueryStatus.UNAUTHORIZED: 401,
    QueryStatus.RATE_LIMITED: 429,
    QueryStatus.BLOCKED: 400,
    QueryStatus.ERROR: 500,
}

# Single-file test console, shipped next to this module (see package-data in
# pyproject.toml). Served from disk so editing the HTML needs no restart.
_UI_FILE = Path(__file__).with_name("static") / "index.html"


class AskBody(BaseModel):
    query: str
    session_id: str | None = None
    filters: RetrievalFilters | None = None


class FeedbackBody(BaseModel):
    request_id: str
    rating: int = Field(description="+1 for helpful, -1 for not helpful")
    comment: str = ""


def create_app(
    settings: QuerySettings | None = None,
    service: QueryService | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``service`` is injectable so tests can drive the API with an in-memory
    :class:`QueryService`; in production it is built from ``settings`` on
    startup (loading the embedding model and LLM client once).
    """
    settings = settings or build_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service if service is not None else build_service(settings)
        yield

    app = FastAPI(
        title="Enterprise RAG Query API",
        version="0.2.0",
        description=(
            "Pipeline 2 (steps 1-14): input -> auth -> guard -> route -> rewrite -> encode "
            "-> retrieve -> rerank -> context -> generate -> output guard -> format -> persist."
        ),
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        """Serve the single-file test console."""
        if not _UI_FILE.exists():
            raise HTTPException(status_code=500, detail=f"UI file missing: {_UI_FILE}")
        return FileResponse(_UI_FILE, media_type="text/html")

    @app.get("/health")
    def health() -> dict[str, Any]:
        service: QueryService = app.state.service
        return {"status": "ok", **service.metadata}

    @app.post("/ask", response_model=AskResult)
    def ask(
        body: AskBody,
        response: Response,
        include_vector: bool = False,
        authorization: Annotated[str | None, Header()] = None,
        x_dev_token: Annotated[str | None, Header()] = None,
    ) -> AskResult:
        service: QueryService = app.state.service
        request = AskRequest(
            query=body.query,
            session_id=body.session_id,
            token=_extract_token(authorization, x_dev_token),
            filters=body.filters,
        )
        result = service.ask(request, include_vector=include_vector)
        response.status_code = _STATUS_TO_HTTP.get(result.status, 200)
        if result.status is QueryStatus.UNAUTHORIZED:
            response.headers["WWW-Authenticate"] = "Bearer"
        return result

    @app.post("/ask/stream")
    def ask_stream(
        body: AskBody,
        authorization: Annotated[str | None, Header()] = None,
        x_dev_token: Annotated[str | None, Header()] = None,
    ) -> StreamingResponse:
        service: QueryService = app.state.service
        request = AskRequest(
            query=body.query,
            session_id=body.session_id,
            token=_extract_token(authorization, x_dev_token),
            filters=body.filters,
        )

        def event_stream():  # noqa: ANN202
            for event in service.ask_stream(request):
                yield (
                    f"event: {event['event']}\n"
                    f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/feedback", response_model=FeedbackResult)
    def feedback(
        body: FeedbackBody,
        response: Response,
        authorization: Annotated[str | None, Header()] = None,
        x_dev_token: Annotated[str | None, Header()] = None,
    ) -> FeedbackResult:
        service: QueryService = app.state.service
        result = service.feedback(
            _extract_token(authorization, x_dev_token),
            body.request_id,
            body.rating,
            body.comment,
        )
        response.status_code = {
            "recorded": 200,
            "unauthorized": 401,
            "forbidden": 403,
            "not_found": 404,
            "error": 500,
        }.get(result.status, 500)
        if result.status == "unauthorized":
            response.headers["WWW-Authenticate"] = "Bearer"
        return result

    return app


def _extract_token(authorization: str | None, x_dev_token: str | None) -> str | None:
    if x_dev_token and x_dev_token.strip():
        return x_dev_token.strip()
    if authorization:
        value = authorization.strip()
        if value.lower().startswith("bearer "):
            return value[len("bearer ") :].strip()
        return value
    return None

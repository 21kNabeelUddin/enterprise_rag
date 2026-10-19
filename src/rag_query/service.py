"""Query service: run the graph and shape the result.

The graph speaks in state updates; the service turns one request into
:class:`AskResult`. It also serialises access to the graph with a lock -- the
local stack has a single LLM slot and one GPU, so concurrent inference would
only thrash.

``ask_stream`` is *verified* streaming: the graph runs to completion, the output
guard passes, and only then is the final answer chunked to the caller. Raw model
tokens are never exposed before validation.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import numpy as np

from rag_query.config import QuerySettings
from rag_query.domain.errors import AuthenticationError
from rag_query.domain.models import (
    AskRequest,
    AskResult,
    ContextSummary,
    FeedbackResult,
    QueryStatus,
    RetrievalFilters,
)
from rag_query.logging import get_logger
from rag_query.ports.auth import Authenticator
from rag_query.ports.history import HistoryStore
from rag_query.state import QueryState

log = get_logger(__name__)

_PREVIEW_LEN = 8
_STREAM_CHARS = 32


class QueryService:
    def __init__(
        self,
        graph: Any,
        *,
        settings: QuerySettings | None = None,
        metadata: dict[str, Any] | None = None,
        lock: threading.Lock | None = None,
        authenticator: Authenticator | None = None,
        history_store: HistoryStore | None = None,
        feedback_admin_group: str = "admin",
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._metadata = dict(metadata or {})
        self._lock = lock or threading.Lock()
        self._authenticator = authenticator
        self._history_store = history_store
        self._feedback_admin_group = feedback_admin_group

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def ask(
        self,
        request: AskRequest,
        *,
        include_vector: bool = False,
        request_id: str | None = None,
    ) -> AskResult:
        initial: QueryState = {
            "request_id": request_id or uuid4().hex,
            "raw_query": request.query,
            "session_id": request.session_id,
            "token": request.token,
            "history": list(request.history),
            "filters": request.filters or RetrievalFilters(),
            "timings_ms": {},
            "trace": [],
            "warnings": [],
        }

        with self._lock:
            final = self._graph.invoke(initial)

        result = self._to_result(final, include_vector=include_vector)
        log.info(
            "ask %s status=%s rewrite=%r ms=%.1f",
            result.request_id,
            result.status.value,
            (result.rewritten_query or "")[:60],
            sum(result.timings_ms.values()),
        )
        return result

    def ask_stream(
        self,
        request: AskRequest,
        *,
        request_id: str | None = None,
        chunk_chars: int = _STREAM_CHARS,
    ) -> Iterator[dict[str, Any]]:
        """Run the verified pipeline, then stream the safe answer in chunks.

        This is intentionally not raw token streaming: a token that later fails
        the output guard cannot be unsent. Chunking the verified answer gives a
        responsive UI without ever exposing an unvalidated claim.
        """
        result = self.ask(request, request_id=request_id)
        text = result.answer or result.message or ""
        size = max(1, chunk_chars)
        for offset in range(0, len(text), size):
            yield {"event": "token", "data": {"text": text[offset : offset + size]}}
        yield {"event": "done", "data": result.model_dump(mode="json")}

    def feedback(
        self,
        token: str | None,
        request_id: str,
        rating: int,
        comment: str = "",
    ) -> FeedbackResult:
        """Record thumbs-up/down for a persisted turn.

        Ownership is checked here rather than in the store so the same rule
        applies to the CLI, the API and any future surface.
        """
        if self._authenticator is None:
            return FeedbackResult(status="error", message="Authentication is not configured.")
        try:
            principal = self._authenticator.authenticate(token)
        except AuthenticationError as exc:
            return FeedbackResult(status="unauthorized", message=str(exc))
        except Exception as exc:  # noqa: BLE001
            log.warning("feedback authentication failed: %s", exc)
            return FeedbackResult(status="unauthorized", message="Authentication failed.")

        if self._history_store is None:
            return FeedbackResult(status="error", message="History storage is not enabled.")
        if rating not in (-1, 1):
            return FeedbackResult(status="error", message="Rating must be +1 or -1.")

        try:
            turn = self._history_store.get_turn(request_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("could not load turn %s", request_id)
            return FeedbackResult(status="error", message=f"Could not load turn: {exc}")
        if turn is None:
            return FeedbackResult(status="not_found", message="No turn with that request id.")
        if (
            turn.user_id != principal.user_id
            and self._feedback_admin_group not in principal.groups
        ):
            return FeedbackResult(
                status="forbidden",
                message="You can only rate your own turns.",
            )
        if not turn.turn_id:
            return FeedbackResult(status="error", message="Turn was not persisted.")

        try:
            feedback = self._history_store.save_feedback(
                turn.turn_id,
                principal.user_id,
                rating,
                comment,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("could not save feedback for turn %s", turn.turn_id)
            return FeedbackResult(status="error", message=f"Could not save feedback: {exc}")
        return FeedbackResult(status="recorded", feedback=feedback, message="Feedback recorded.")

    @staticmethod
    def _to_result(state: QueryState, *, include_vector: bool) -> AskResult:
        bundle = state.get("context")
        context_summary = (
            ContextSummary(
                sources=len(bundle.sources),
                neighbors=sum(1 for source in bundle.sources if source.chunk.is_neighbor),
                tokens=bundle.token_count,
                truncated=bundle.truncated,
            )
            if bundle is not None
            else None
        )
        result = AskResult(
            request_id=state.get("request_id", ""),
            status=state.get("status") or QueryStatus.ERROR,
            query=state.get("raw_query") or "",
            normalized_query=state.get("query") or "",
            rewritten_query=state.get("rewritten_query"),
            message=state.get("message"),
            principal=state.get("principal"),
            guard=state.get("guard"),
            route=state.get("route"),
            answer=state.get("answer"),
            citations=list(state.get("citations") or ()),
            turn_id=state.get("turn_id"),
            retrieval=state.get("retrieval_summary"),
            context=context_summary,
            output_guard=state.get("output_guard"),
            generation=state.get("generation"),
            history_loaded=int(state.get("history_loaded") or 0),
            warnings=list(state.get("warnings") or []),
            timings_ms=dict(state.get("timings_ms") or {}),
            trace=tuple(state.get("trace") or ()),
        )

        vector = state.get("query_vector")
        if vector is not None:
            array = np.asarray(vector, dtype=np.float32)
            fingerprint = state.get("fingerprint")
            result.embedding_dim = int(array.shape[0])
            result.embedding_model = fingerprint.model_id if fingerprint else None
            result.embedding_norm = float(np.linalg.norm(array))
            result.embedding_preview = [round(float(x), 6) for x in array[:_PREVIEW_LEN]]
            if include_vector:
                result.embedding = [float(x) for x in array]
        return result

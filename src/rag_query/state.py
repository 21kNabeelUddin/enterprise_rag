"""The LangGraph state threaded through every node.

Nodes return *partial* updates; LangGraph merges them back into this dict.
``timings_ms`` and ``trace`` use reducers so several nodes can contribute
without clobbering each other.

Convention: a node sets ``status`` **only** when it terminates the run early
(rejection/block/etc.). A healthy run leaves ``status`` unset until the encoder
sets :attr:`QueryStatus.ENCODED`; retrieval and later steps keep that marker and
the finalize node promotes it to :attr:`QueryStatus.ANSWERED` on success.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import numpy as np

from rag_ingest.domain.models import ModelFingerprint
from rag_query.domain.models import (
    ChatMessage,
    Citation,
    ContextBundle,
    Generation,
    GuardVerdict,
    Principal,
    QueryStatus,
    RetrievalFilters,
    RetrievalSummary,
    RetrievedChunk,
    RouteDecision,
)


def _merge_dict(a: dict | None, b: dict | None) -> dict:
    out = dict(a or {})
    out.update(b or {})
    return out


def _append(a: list | None, b: list | None) -> list:
    return list(a or []) + list(b or [])


class QueryState(TypedDict, total=False):
    # --- request (step 1 input) ---
    request_id: str
    raw_query: str
    query: str  # normalised
    session_id: str | None
    token: str | None
    history: list[ChatMessage]
    filters: RetrievalFilters

    # --- step 2 authenticate ---
    principal: Principal

    # --- step 3 input guardrail ---
    guard: GuardVerdict

    # --- step 4 router ---
    route: RouteDecision

    # --- step 5 rewriter ---
    rewritten_query: str | None

    # --- step 7 encoder ---
    query_vector: np.ndarray
    embedding_dim: int
    fingerprint: ModelFingerprint

    # --- step 8 retrieval ---
    candidates: list[RetrievedChunk]
    retrieval_summary: RetrievalSummary

    # --- step 9 rerank ---
    rerank_applied: bool

    # --- step 10 context ---
    context: ContextBundle
    history_loaded: int

    # --- steps 11-13 generation / guard / format ---
    generation: Generation
    answer: str | None
    output_guard: GuardVerdict
    output_feedback: str
    generation_attempts: int
    citations: list[Citation]

    # --- step 14 persistence ---
    turn_id: str | None

    # --- control / observability ---
    status: QueryStatus
    message: str | None
    warnings: Annotated[list[str], _append]
    timings_ms: Annotated[dict[str, float], _merge_dict]
    trace: Annotated[list[str], _append]

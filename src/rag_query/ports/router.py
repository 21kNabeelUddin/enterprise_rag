"""Routing port (decide whether a query is answerable)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_query.domain.models import RouteDecision


@runtime_checkable
class QueryRouter(Protocol):
    """Classifies a query as answerable or as small-talk/off-topic/empty.

    A rule-based adapter ships today; an LLM classifier can implement the same
    port later without touching the graph.
    """

    def route(self, query: str) -> RouteDecision: ...

"""The complete Pipeline 2 graph (steps 1-14).

    START
      -> input            (1) normalise / reject
      -> authenticate     (2) token -> principal
      -> history          (14) load recent session turns for the rewriter
      -> input_guard      (3) rate limit + prompt-guard + PII
      -> route            (4) answerable?
      -> rewrite          (5) standalone search query
      -> encode           (7) embed with the ingestion model
      -> retrieve         (8) dense + lexical, RRF, ACL/metadata filters
      -> rerank           (9) Qwen3 cross-encoder, top-k
      -> context          (10) deterministic token-budgeted prompt
      -> generate         (11) grounded answer with citations
      -> output_guard     (12) citation/grounding/PII/leak checks (retry once)
      -> format           (13) resolve citations
      -> finalize         (14) persist turn, promote status

Any node that sets a terminal status short-circuits to ``fallback`` (except an
output-guard failure, which formats a safe answer with sources). HyDE (step 6)
is intentionally absent.

``build_graph`` also supports the legacy steps 1-7 graph by omitting the RAG
nodes; this keeps the early unit tests focused on their own steps.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from rag_query.domain.models import QueryStatus
from rag_query.nodes.base import Node
from rag_query.state import QueryState


def _next_or_fallback(success_target: str):
    """Conditional edge: continue on success, divert to fallback on any status."""

    def _decide(state: QueryState) -> str:
        return "fallback" if state.get("status") else success_target

    return _decide


def _after_encode(state: QueryState) -> str:
    return "fallback" if state.get("status") is QueryStatus.ERROR else "end"


def _after_encode_rag(state: QueryState) -> str:
    return "retrieve" if state.get("status") is QueryStatus.ENCODED else "fallback"


def _continue_or_fallback(success_target: str):
    """Steps 8-11 use ENCODED as the healthy marker, not a status-free state."""

    def _decide(state: QueryState) -> str:
        status = state.get("status")
        if status is None or status is QueryStatus.ENCODED:
            return success_target
        return "fallback"

    return _decide


def _after_output_guard(state: QueryState) -> str:
    status = state.get("status")
    if status is QueryStatus.OUTPUT_BLOCKED:
        return "format"
    if status is not None and status is not QueryStatus.ENCODED:
        return "fallback"
    if state.get("output_feedback"):
        return "generate"
    return "format"


def build_graph(
    *,
    input_node: Node,
    authenticate_node: Node,
    guard_node: Node,
    route_node: Node,
    rewrite_node: Node,
    encode_node: Node,
    fallback_node: Node,
    history_node: Node | None = None,
    retrieve_node: Node | None = None,
    rerank_node: Node | None = None,
    context_node: Node | None = None,
    generate_node: Node | None = None,
    output_guard_node: Node | None = None,
    format_node: Node | None = None,
    finalize_node: Node | None = None,
) -> Any:
    rag_nodes = (
        retrieve_node,
        rerank_node,
        context_node,
        generate_node,
        output_guard_node,
        format_node,
        finalize_node,
    )
    if retrieve_node is not None and any(node is None for node in rag_nodes):
        raise ValueError(
            "The RAG graph requires retrieve, rerank, context, generate, "
            "output_guard, format and finalize nodes."
        )

    builder = StateGraph(QueryState)

    builder.add_node("input", input_node)
    builder.add_node("authenticate", authenticate_node)
    builder.add_node("input_guard", guard_node)
    builder.add_node("route", route_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("encode", encode_node)
    builder.add_node("fallback", fallback_node)

    if history_node is not None:
        builder.add_node("history", history_node)

    if retrieve_node is not None:
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("rerank", rerank_node)
        builder.add_node("context", context_node)
        builder.add_node("generate", generate_node)
        builder.add_node("output_guard", output_guard_node)
        builder.add_node("format", format_node)
        builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "input")
    builder.add_conditional_edges(
        "input", _next_or_fallback("authenticate"),
        {"authenticate": "authenticate", "fallback": "fallback"},
    )

    after_auth = "history" if history_node is not None else "input_guard"
    builder.add_conditional_edges(
        "authenticate", _next_or_fallback(after_auth),
        {after_auth: after_auth, "fallback": "fallback"},
    )
    if history_node is not None:
        builder.add_conditional_edges(
            "history", _next_or_fallback("input_guard"),
            {"input_guard": "input_guard", "fallback": "fallback"},
        )

    builder.add_conditional_edges(
        "input_guard", _next_or_fallback("route"),
        {"route": "route", "fallback": "fallback"},
    )
    builder.add_conditional_edges(
        "route", _next_or_fallback("rewrite"),
        {"rewrite": "rewrite", "fallback": "fallback"},
    )
    builder.add_edge("rewrite", "encode")

    if retrieve_node is None:
        builder.add_conditional_edges(
            "encode", _after_encode,
            {"fallback": "fallback", "end": END},
        )
    else:
        builder.add_conditional_edges(
            "encode", _after_encode_rag,
            {"retrieve": "retrieve", "fallback": "fallback"},
        )
        builder.add_conditional_edges(
            "retrieve", _continue_or_fallback("rerank"),
            {"rerank": "rerank", "fallback": "fallback"},
        )
        builder.add_conditional_edges(
            "rerank", _continue_or_fallback("context"),
            {"context": "context", "fallback": "fallback"},
        )
        builder.add_conditional_edges(
            "context", _continue_or_fallback("generate"),
            {"generate": "generate", "fallback": "fallback"},
        )
        builder.add_conditional_edges(
            "generate", _continue_or_fallback("output_guard"),
            {"output_guard": "output_guard", "fallback": "fallback"},
        )
        builder.add_conditional_edges(
            "output_guard", _after_output_guard,
            {"generate": "generate", "format": "format", "fallback": "fallback"},
        )
        builder.add_edge("format", "finalize")
        builder.add_edge("finalize", END)

    if finalize_node is not None:
        builder.add_edge("fallback", "finalize")
    else:
        builder.add_edge("fallback", END)

    return builder.compile()

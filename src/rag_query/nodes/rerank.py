"""Step 9 -- cross-encoder rerank.

Scores the fused candidates with the local Qwen3 reranker, keeps the top
``top_k``, then optionally expands each hit with adjacent chunks from the same
section. Neighbor expansion happens *after* ranking so it never displaces a
relevance-ranked source.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus
from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.reranker import Reranker
from rag_query.ports.retriever import Retriever
from rag_query.state import QueryState

log = get_logger(__name__)


class RerankNode(Node):
    name = "rerank"

    def __init__(
        self,
        reranker: Reranker,
        *,
        top_k: int = 8,
        expander: Retriever | None = None,
        neighbor_window: int = 0,
    ) -> None:
        self._reranker = reranker
        self._top_k = max(1, top_k)
        self._expander = expander
        self._neighbor_window = max(0, neighbor_window)

    def run(self, state: QueryState) -> dict[str, Any]:
        candidates = list(state.get("candidates") or [])
        if not candidates:
            return {
                "status": QueryStatus.NO_CONTEXT,
                "message": "No candidates were available to rerank.",
            }

        query = state.get("rewritten_query") or state.get("query") or ""
        summary = state.get("retrieval_summary")
        ciks = summary.filters.ciks if summary is not None else ()
        warnings: list[str] = []
        rerank_applied = True
        scored: list[Any] = []
        try:
            # For comparison questions, score a larger pool so the selection
            # step can keep a quota from each company. Single-company questions
            # score only top_k, as before.
            score_k = min(len(candidates), self._top_k * 2)
            if len(ciks) <= 1:
                score_k = min(len(candidates), self._top_k)
            scored = self._reranker.rerank(query, candidates, top_k=score_k)
            ranked = (
                _balanced_top_k(scored, ciks, self._top_k)
                if len(ciks) > 1
                else scored[: self._top_k]
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("reranker failed; keeping retrieval order")
            warnings.append(f"rerank_failed: {exc}")
            rerank_applied = False
            ranked = candidates[: self._top_k]

        if not ranked:
            return {
                "status": QueryStatus.NO_CONTEXT,
                "message": "Reranking produced no usable passages.",
                "warnings": warnings,
            }

        reranked_count = len(scored) if scored else len(ranked)
        if self._expander is not None and self._neighbor_window > 0:
            try:
                ranked = self._expander.expand_neighbors(
                    ranked,
                    window=self._neighbor_window,
                    principal=state.get("principal"),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("neighbor expansion failed: %s", exc)
                warnings.append(f"neighbor_expansion_failed: {exc}")

        if summary is not None:
            summary = summary.model_copy(
                update={
                    "reranked": reranked_count,
                    "expanded": len(ranked),
                    "neighbor_sources": sum(1 for chunk in ranked if chunk.is_neighbor),
                    # Expansion is primary-first, so the top reranked hit stays
                    # at index 0 and its score is the meaningful top score.
                    "top_score": ranked[0].rerank_score
                    if ranked[0].rerank_score is not None
                    else summary.top_score,
                }
            )

        return {
            "candidates": ranked,
            "retrieval_summary": summary,
            "rerank_applied": rerank_applied,
            "warnings": warnings,
        }


def _balanced_top_k(
    scored: list[Any],
    ciks: tuple[str, ...],
    top_k: int,
) -> list[Any]:
    """Pick top_k chunks with a per-company quota, then fill by score."""
    quota = max(1, top_k // max(1, len(ciks)))
    selected: list[Any] = []
    selected_ids: set[str] = set()
    for cik in ciks:
        for chunk in (item for item in scored if item.cik == cik):
            if len([item for item in selected if item.cik == cik]) >= quota:
                break
            if chunk.chunk_id in selected_ids:
                continue
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
    for chunk in scored:
        if len(selected) >= top_k:
            break
        if chunk.chunk_id not in selected_ids:
            selected.append(chunk)
            selected_ids.add(chunk.chunk_id)
    selected.sort(
        key=lambda chunk: chunk.rerank_score if chunk.rerank_score is not None else float("-inf"),
        reverse=True,
    )
    return [
        chunk.model_copy(update={"rank": index + 1})
        for index, chunk in enumerate(selected[:top_k])
    ]

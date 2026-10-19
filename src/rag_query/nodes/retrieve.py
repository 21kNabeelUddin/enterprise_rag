"""Step 8 -- hybrid dense + lexical retrieval.

Builds the effective filter set (explicit request filters plus conservative
metadata extracted from the question), applies the principal's ACL groups, and
returns fused candidates. An empty result is a terminal ``NO_CONTEXT`` rather
than an error: the corpus may simply not cover the question.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import (
    QueryStatus,
    RetrievalFilters,
    RetrievalSummary,
)
from rag_query.domain.query_analysis import MetadataExtractor
from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.retriever import Retriever
from rag_query.state import QueryState

log = get_logger(__name__)

_NO_CONTEXT_MESSAGE = (
    "I couldn't find relevant passages in the SEC 10-K filings for that question."
)


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class RetrieveNode(Node):
    name = "retrieve"

    def __init__(
        self,
        retriever: Retriever,
        *,
        filter_extractor: MetadataExtractor | None = None,
        top_k: int = 40,
        per_company_k: int = 8,
    ) -> None:
        self._retriever = retriever
        self._filter_extractor = filter_extractor
        self._top_k = max(1, top_k)
        self._per_company_k = max(2, per_company_k)

    def run(self, state: QueryState) -> dict[str, Any]:
        principal = state.get("principal")
        vector = state.get("query_vector")
        if principal is None or vector is None:
            return {
                "status": QueryStatus.ERROR,
                "message": "Retrieval requires an authenticated principal and a query vector.",
            }

        raw_query = state.get("query") or ""
        query = state.get("rewritten_query") or raw_query
        # Company names come from the user's own question, plus a concise
        # standalone rewrite (which may legitimately pull a company from
        # history for a follow-up). Long answer-like rewrites are discarded by
        # the rewriter before this point. Cap the merged set so one verbose
        # rewrite cannot widen the filter to the whole directory.
        explicit = state.get("filters") or RetrievalFilters()
        extracted = RetrievalFilters()
        if self._filter_extractor is not None:
            try:
                raw_filters = self._filter_extractor.extract(raw_query)
                rewritten_filters = self._filter_extractor.extract(query)
                if any(cik in raw_query for cik in raw_filters.ciks):
                    # An explicit numeric CIK wins outright.
                    ciks = raw_filters.ciks
                else:
                    ciks = _unique(raw_filters.ciks + rewritten_filters.ciks)[:3]
                extracted = RetrievalFilters(
                    ciks=ciks,
                    years=_unique(raw_filters.years + rewritten_filters.years),
                    section_keys=_unique(
                        raw_filters.section_keys + rewritten_filters.section_keys
                    ),
                    doc_ids=_unique(raw_filters.doc_ids + rewritten_filters.doc_ids),
                )
            except Exception as exc:  # noqa: BLE001 - analysis is best-effort
                log.warning("query metadata extraction failed: %s", exc)
        filters = explicit.merged_with(extracted)

        warnings: list[str] = []
        try:
            if len(filters.ciks) > 1:
                candidates, summary = self._retrieve_by_company(
                    query,
                    vector,
                    principal=principal,
                    filters=filters,
                    warnings=warnings,
                )
            else:
                result = self._retriever.retrieve(
                    query,
                    vector,
                    principal=principal,
                    filters=filters,
                    limit=self._top_k,
                )
                candidates = list(result.candidates)
                top_score = candidates[0].fused_score if candidates else None
                summary = RetrievalSummary(
                    dense_candidates=result.dense_candidates,
                    lexical_candidates=result.lexical_candidates,
                    fused_candidates=result.fused_candidates or len(candidates),
                    reranked=0,
                    top_score=top_score,
                    model_id=getattr(self._retriever, "model_id", ""),
                    lexical_query=result.lexical_query,
                    filters=result.filters or filters,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("retrieval failed")
            return {"status": QueryStatus.ERROR, "message": f"Retrieval failed: {exc}"}

        updates: dict[str, Any] = {
            "candidates": candidates,
            "retrieval_summary": summary,
        }
        if warnings:
            updates["warnings"] = warnings
        if not candidates:
            updates["status"] = QueryStatus.NO_CONTEXT
            updates["message"] = _NO_CONTEXT_MESSAGE
        elif filters.ciks and len(filters.ciks) == 1:
            log.info("retrieval filtered to CIK %s", filters.ciks[0])
        return updates

    def _retrieve_by_company(
        self,
        query: str,
        vector: Any,
        *,
        principal: Any,
        filters: RetrievalFilters,
        warnings: list[str],
    ) -> tuple[list[Any], RetrievalSummary]:
        """Retrieve each company separately, then interleave the rankings.

        A single top-k query with ``cik IN (...)`` lets the company with the
        most similar filing language take every slot. Comparison questions then
        get an answer about one company with citations from that company only.
        Per-company retrieval guarantees every requested company is represented
        before the cross-encoder sees the pool.
        """
        buckets: list[list[Any]] = []
        dense = lexical = fused = 0
        lexical_queries: list[str] = []
        for cik in filters.ciks:
            scoped = filters.model_copy(update={"ciks": (cik,)})
            try:
                result = self._retriever.retrieve(
                    query,
                    vector,
                    principal=principal,
                    filters=scoped,
                    limit=self._per_company_k,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("retrieval failed for CIK %s: %s", cik, exc)
                warnings.append(f"retrieval_failed_for_cik_{cik}: {exc}")
                continue
            buckets.append(list(result.candidates))
            dense += result.dense_candidates
            lexical += result.lexical_candidates
            fused += result.fused_candidates or len(result.candidates)
            if result.lexical_query:
                lexical_queries.append(result.lexical_query)

        if not buckets:
            raise RuntimeError("retrieval failed for every company filter")

        candidates = _interleave(buckets)
        summary = RetrievalSummary(
            dense_candidates=dense,
            lexical_candidates=lexical,
            fused_candidates=fused or len(candidates),
            reranked=0,
            top_score=candidates[0].fused_score if candidates else None,
            model_id=getattr(self._retriever, "model_id", ""),
            lexical_query=" | ".join(dict.fromkeys(lexical_queries)),
            filters=filters,
        )
        return candidates, summary


def _interleave(buckets: list[list[Any]]) -> list[Any]:
    out: list[Any] = []
    for index in range(max((len(bucket) for bucket in buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket):
                out.append(bucket[index])
    return out

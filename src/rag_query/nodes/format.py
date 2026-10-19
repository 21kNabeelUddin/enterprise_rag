"""Step 13 -- format the answer and resolve citations.

The generator emits inline markers (``[S1]``); this node validates them against
the context, maps each source id to a filename/CIK/Item/URL record, and returns a
clean answer. Unknown markers are stripped rather than shown as broken links.
"""

from __future__ import annotations

from typing import Any

from rag_query.domain.citations import (
    build_citation,
    is_no_answer,
    strip_unknown_citations,
)
from rag_query.domain.models import QueryStatus
from rag_query.domain.query_analysis import CompanyDirectory
from rag_query.nodes.base import Node
from rag_query.state import QueryState

_SAFE_OUTPUT_MESSAGE = (
    "I found relevant filings but couldn't produce a fully verified answer. "
    "Please review the cited sources."
)


class FormatAnswerNode(Node):
    name = "format"

    def __init__(
        self,
        companies: CompanyDirectory | None = None,
        *,
        max_snippet_chars: int = 280,
    ) -> None:
        self._companies = companies
        self._max_snippet_chars = max_snippet_chars

    def run(self, state: QueryState) -> dict[str, Any]:
        status = state.get("status")
        bundle = state.get("context")
        sources = bundle.sources if bundle is not None else ()
        valid_ids = {source.source_id for source in sources}

        if status is QueryStatus.OUTPUT_BLOCKED:
            answer = _SAFE_OUTPUT_MESSAGE
        else:
            answer = (state.get("answer") or "").strip()
            if not answer:
                answer = "I couldn't find that in the provided filings."

        answer = strip_unknown_citations(answer, valid_ids).strip()
        citations = [
            build_citation(
                source,
                company=(
                    self._companies.name_for_cik(source.chunk.cik)
                    if self._companies is not None
                    else None
                ),
                snippet_chars=self._max_snippet_chars,
            )
            for source in sources
        ]

        updates: dict[str, Any] = {"answer": answer, "citations": citations}
        if (status is None or status is QueryStatus.ENCODED) and is_no_answer(answer):
            # A partial answer may cite supporting context while still saying
            # the requested fact is absent. The sources stay attached, but the
            # terminal status reflects that the question was not answered.
            updates["status"] = QueryStatus.NO_CONTEXT
            updates["message"] = "I couldn't find that in the provided filings."
        return updates

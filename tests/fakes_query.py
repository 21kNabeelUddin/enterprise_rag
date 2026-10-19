"""Test doubles for the query pipeline.

Together with :mod:`tests.fakes` (which supplies ``FakeEmbedder``) these let the
whole graph run with no GPU, no LLM and no database.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rag_query.domain.errors import AuthenticationError
from rag_query.domain.models import (
    ChatMessage,
    Citation,
    ContextBundle,
    Feedback,
    Generation,
    GuardVerdict,
    Principal,
    QueryStatus,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
    RouteDecision,
    TurnRecord,
)


class FakeLLM:
    def __init__(self, reply: str = "rewritten question", error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(list(messages))
        if self._error is not None:
            raise self._error
        return self._reply


class FakeAuthenticator:
    def __init__(
        self,
        principals: dict[str, Principal] | None = None,
        anonymous: Principal | None = None,
    ) -> None:
        self._principals = principals or {
            "good-token": Principal(user_id="u1", groups=("public",), display_name="U1")
        }
        self._anonymous = anonymous or Principal(user_id="anonymous", groups=("public",))

    def authenticate(self, token: str | None) -> Principal:
        if not token:
            return self._anonymous
        principal = self._principals.get(token)
        if principal is None:
            raise AuthenticationError("unknown token")
        return principal


class FakeGuard:
    def __init__(
        self,
        *,
        name: str = "fake_guard",
        allowed: bool = True,
        reason: str = "ok",
        labels: tuple[str, ...] = (),
        scores: dict[str, float] | None = None,
    ) -> None:
        self._name = name
        self._allowed = allowed
        self._reason = reason
        self._labels = labels
        self._scores = scores or {}

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str) -> GuardVerdict:
        return GuardVerdict(
            allowed=self._allowed,
            guard=self._name,
            reason=self._reason,
            scores=self._scores,
            labels=self._labels,
        )


class FakeRateLimiter:
    def __init__(self, *, allow: bool = True) -> None:
        self._allow = allow
        self.calls = 0

    def allow(self, key: str) -> bool:
        self.calls += 1
        return self._allow


class FakeRouter:
    def __init__(self, *, answerable: bool = True, intent: str = "question") -> None:
        self._answerable = answerable
        self._intent = intent

    def route(self, query: str) -> RouteDecision:
        return RouteDecision(
            answerable=self._answerable,
            intent=self._intent,
            reason="fake routing decision",
        )


class FakeRewriter:
    def __init__(
        self,
        *,
        result: str = "rewritten question",
        error: Exception | None = None,
    ) -> None:
        self._result = result
        self._error = error
        self.seen_history: list[ChatMessage] = []

    def rewrite(self, query: str, history: Sequence[ChatMessage]) -> str:
        self.seen_history = list(history)
        if self._error is not None:
            raise self._error
        return self._result


class FakeVectorProbe:
    def __init__(self, info=None, *, error: Exception | None = None) -> None:  # noqa: ANN001
        self._info = info
        self._error = error

    def describe(self):  # noqa: ANN201
        if self._error is not None:
            raise self._error
        return self._info


# --- steps 8-14 fakes -----------------------------------------------------
def make_chunk(
    index: int = 0,
    *,
    chunk_id: str | None = None,
    doc_id: str = "320193_2020.htm",
    cik: str = "320193",
    section_key: str = "1A",
    text: str | None = None,
    fused_score: float = 0.02,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or f"{doc_id}:{section_key}:{index}",
        doc_id=doc_id,
        cik=cik,
        year="2020",
        section_key=section_key,
        chunk_index=index,
        text=text
        or (
            "Apple faces risks from supply-chain disruptions, cybersecurity "
            "incidents and changes in consumer demand."
        ),
        token_count=20,
        dense_score=0.9,
        lexical_score=0.5,
        fused_score=fused_score,
    )


class FakeRetriever:
    def __init__(
        self,
        chunks: Sequence[RetrievedChunk] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._chunks = list(chunks or [])
        self._error = error
        self.calls: list[dict] = []

    @property
    def model_id(self) -> str:
        return "fake-embedder"

    def retrieve(
        self,
        query: str,
        vector: np.ndarray,
        *,
        principal: Principal,
        filters: RetrievalFilters | None = None,
        limit: int | None = None,
    ) -> RetrievalResult:
        filters = filters or RetrievalFilters()
        self.calls.append(
            {
                "query": query,
                "vector": vector,
                "principal": principal,
                "filters": filters,
                "limit": limit,
            }
        )
        if self._error is not None:
            raise self._error
        if not principal.groups:
            return RetrievalResult(filters=filters)
        kept = [chunk for chunk in self._chunks if _matches(chunk, filters)]
        if limit is not None:
            kept = kept[:limit]
        return RetrievalResult(
            candidates=tuple(kept),
            dense_candidates=len(kept),
            lexical_candidates=len(kept),
            fused_candidates=len(kept),
            lexical_query=query,
            filters=filters,
        )

    def expand_neighbors(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        window: int = 1,
        principal: Principal | None = None,
    ) -> list[RetrievedChunk]:
        if principal is not None and not principal.groups:
            return list(chunks)
        existing = {chunk.chunk_id for chunk in chunks}
        out: list[RetrievedChunk] = []
        for chunk in chunks:
            out.append(chunk)
            for offset in range(1, max(0, window) + 1):
                neighbor_id = f"{chunk.doc_id}:{chunk.section_key}:{chunk.chunk_index + offset}"
                if neighbor_id in existing:
                    continue
                existing.add(neighbor_id)
                out.append(
                    chunk.model_copy(
                        update={
                            "chunk_id": neighbor_id,
                            "chunk_index": chunk.chunk_index + offset,
                            "text": f"Additional context after: {chunk.text}",
                            "is_neighbor": True,
                            "dense_score": None,
                            "lexical_score": None,
                            "rerank_score": None,
                            "rank": None,
                        }
                    )
                )
        return out


class FakeReranker:
    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return True

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        ordered = sorted(
            candidates,
            key=lambda chunk: chunk.fused_score or 0.0,
            reverse=True,
        )
        kept = ordered[: (top_k or len(ordered))]
        return [
            chunk.model_copy(
                update={"rerank_score": round(1.0 - index * 0.1, 3), "rank": index + 1}
            )
            for index, chunk in enumerate(kept)
        ]


class FakeGenerator:
    def __init__(
        self,
        replies: Sequence[str] | str | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        if isinstance(replies, str):
            replies = [replies]
        self._replies = list(replies or ["Apple faces supply-chain risks [S1]."])
        self._error = error
        self.calls: list[tuple[ContextBundle, str]] = []

    def generate(self, bundle: ContextBundle, *, strict_feedback: str = "") -> Generation:
        self.calls.append((bundle, strict_feedback))
        if self._error is not None:
            raise self._error
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        return Generation(
            text=self._replies[index],
            model="fake-generator",
            prompt_tokens=bundle.token_count,
            completion_tokens=12,
            finish_reason="stop",
            attempts=len(self.calls),
        )


class FakeOutputGuard:
    def __init__(
        self,
        verdicts: Sequence[GuardVerdict] | None = None,
        *,
        allowed: bool = True,
        reason: str = "ok",
    ) -> None:
        self._verdicts = list(verdicts or [])
        self._allowed = allowed
        self._reason = reason
        self.calls = 0

    def check(
        self,
        answer: str,
        sources: Sequence[object],
        *,
        query: str,
    ) -> GuardVerdict:
        self.calls += 1
        if self._verdicts:
            return self._verdicts[min(self.calls - 1, len(self._verdicts) - 1)]
        return GuardVerdict(
            allowed=self._allowed,
            guard="fake_output_guard",
            reason=self._reason,
        )


class FakeHistoryStore:
    def __init__(self, messages: Sequence[ChatMessage] | None = None) -> None:
        self.messages = list(messages or [])
        self.turns: dict[str, TurnRecord] = {}
        self.feedback: dict[tuple[str, str], Feedback] = {}
        self._next_turn = 1

    def recent(
        self,
        session_id: str,
        user_id: str,
        *,
        limit: int = 6,
    ) -> list[ChatMessage]:
        return self.messages[-limit:]

    def save_turn(self, turn: TurnRecord) -> str:
        turn_id = turn.turn_id or f"turn-{self._next_turn}"
        self._next_turn += 1
        self.turns[turn.request_id] = turn.model_copy(update={"turn_id": turn_id})
        return turn_id

    def get_turn(self, request_id: str) -> TurnRecord | None:
        return self.turns.get(request_id)

    def save_feedback(
        self,
        turn_id: str,
        user_id: str,
        rating: int,
        comment: str = "",
    ) -> Feedback:
        feedback = Feedback(
            feedback_id=f"fb-{len(self.feedback) + 1}",
            turn_id=turn_id,
            user_id=user_id,
            rating=rating,
            comment=comment,
        )
        self.feedback[(turn_id, user_id)] = feedback
        return feedback


class KeywordEmbedder:
    """Bag-of-keywords embedding: semantically overlapping texts score high.

    ``FakeEmbedder`` hashes whole strings, which is perfect for encoder tests
    but gives orthogonal vectors for a sentence and its source passage.
    """

    def __init__(self, vocabulary: Sequence[str] | None = None) -> None:
        self._vocabulary = list(
            vocabulary
            or (
                "risk",
                "supply",
                "chain",
                "cyber",
                "cybersecurity",
                "privacy",
                "demand",
                "consumer",
                "facility",
                "california",
                "regulatory",
                "climate",
            )
        )
        from rag_ingest.domain.models import ModelFingerprint

        self._fingerprint = ModelFingerprint(
            model_id="keyword-embedder",
            dim=len(self._vocabulary),
            normalize=True,
        )

    @property
    def fingerprint(self):  # noqa: ANN201
        return self._fingerprint

    @property
    def dim(self) -> int:
        return len(self._vocabulary)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            words = text.lower().replace("-", " ").split()
            for word in words:
                for index, keyword in enumerate(self._vocabulary):
                    if keyword in word:
                        out[row, index] += 1.0
            norm = float(np.linalg.norm(out[row])) or 1.0
            out[row] /= norm
        return out


def make_citation(source_id: str = "S1", *, chunk_id: str = "c1") -> Citation:
    return Citation(
        source_id=source_id,
        chunk_id=chunk_id,
        doc_id="320193_2020.htm",
        cik="320193",
        company="Apple Inc.",
        year="2020",
        section_key="1A",
        section_title="Risk Factors",
        chunk_index=0,
        url="https://www.sec.gov/Archives/edgar/data/320193/320193_2020.htm",
        snippet="supply-chain risk",
        score=0.02,
        rerank_score=0.9,
    )


def make_turn(
    request_id: str = "req-1",
    *,
    user_id: str = "u1",
    status: QueryStatus = QueryStatus.ANSWERED,
) -> TurnRecord:
    return TurnRecord(
        request_id=request_id,
        session_id="s1",
        user_id=user_id,
        groups=("public",),
        question="What are Apple's risk factors?",
        answer="Supply-chain risk [S1].",
        status=status,
        citations=(make_citation(),),
        trace=("input", "finalize"),
    )


def _matches(chunk: RetrievedChunk, filters: RetrievalFilters) -> bool:
    if filters.ciks and chunk.cik not in filters.ciks:
        return False
    if filters.years and chunk.year not in filters.years:
        return False
    if filters.section_keys and chunk.section_key not in filters.section_keys:
        return False
    return not (filters.doc_ids and chunk.doc_id not in filters.doc_ids)

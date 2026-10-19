"""Domain models shared by every query-pipeline port.

Pure data: no storage, no ML, no web framework. Keeping these free of I/O is
what lets the whole graph run against fakes in tests and against Postgres +
local models in production without changing a single node.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

ANONYMOUS_USER_ID = "anonymous"


class Principal(BaseModel):
    """The authenticated caller.

    ``groups`` travel with the request and become the ACL filter at retrieval
    (step 8). A principal with no groups is authenticated but sees nothing.
    """

    model_config = ConfigDict(frozen=True)

    user_id: str
    groups: tuple[str, ...] = ()
    display_name: str | None = None
    token_label: str | None = None  # which dev token produced this principal

    @property
    def is_anonymous(self) -> bool:
        return self.user_id == ANONYMOUS_USER_ID


class ChatMessage(BaseModel):
    """One turn of conversation. ``role`` is ``system``/``user``/``assistant``."""

    model_config = ConfigDict(frozen=True)

    role: str
    content: str


class GuardVerdict(BaseModel):
    """The combined outcome of the input (or output) guardrails."""

    model_config = ConfigDict(frozen=True)

    allowed: bool
    guard: str = ""
    reason: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    labels: tuple[str, ...] = ()


class RouteDecision(BaseModel):
    """Whether a query should proceed to retrieval at all."""

    model_config = ConfigDict(frozen=True)

    answerable: bool
    intent: str = "question"
    reason: str = ""


class QueryStatus(StrEnum):
    """Terminal state of one pass through the pipeline.

    ``ENCODED`` remains the success marker for the steps 1-7 partial graph;
    the complete pipeline terminates with ``ANSWERED`` (or a safe fallback).
    """

    ENCODED = "encoded"  # reached step 7 successfully (partial graph)
    ANSWERED = "answered"  # steps 8-14 produced a verified answer
    REJECTED = "rejected"  # malformed input (step 1)
    UNAUTHORIZED = "unauthorized"  # bad token (step 2)
    RATE_LIMITED = "rate_limited"  # over quota (step 3)
    BLOCKED = "blocked"  # input guardrail tripped (step 3)
    NOT_ANSWERABLE = "not_answerable"  # router declined (step 4)
    NO_CONTEXT = "no_context"  # retrieval found nothing usable (step 8)
    OUTPUT_BLOCKED = "output_blocked"  # output guardrail failed after retry (step 12)
    ERROR = "error"  # unrecoverable fault

    @property
    def ok(self) -> bool:
        """True for outcomes a CLI/API caller should treat as a normal reply."""
        return self in {
            QueryStatus.ENCODED,
            QueryStatus.ANSWERED,
            QueryStatus.NOT_ANSWERABLE,
            QueryStatus.NO_CONTEXT,
        }


class StoredVectorInfo(BaseModel):
    """What the index actually contains, used to prove encoder compatibility."""

    model_config = ConfigDict(frozen=True)

    count: int = 0
    model_ids: tuple[str, ...] = ()
    dims: tuple[int, ...] = ()


class RetrievalFilters(BaseModel):
    """Optional metadata constraints for hybrid retrieval.

    Values are OR-ed within a field and AND-ed across fields -- the same
    semantics SQL ``= ANY(...)`` gives us. Empty tuples mean "no filter".
    """

    model_config = ConfigDict(frozen=True)

    ciks: tuple[str, ...] = ()
    years: tuple[str, ...] = ()
    section_keys: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not (self.ciks or self.years or self.section_keys or self.doc_ids)

    def merged_with(self, other: RetrievalFilters | None) -> RetrievalFilters:
        """Fill this request's empty fields from e.g. query analysis.

        Explicit request filters always win; extracted filters only supply the
        dimensions the caller did not specify.
        """
        if other is None:
            return self
        return RetrievalFilters(
            ciks=self.ciks or other.ciks,
            years=self.years or other.years,
            section_keys=self.section_keys or other.section_keys,
            doc_ids=self.doc_ids or other.doc_ids,
        )


class RetrievedChunk(BaseModel):
    """One chunk returned by retrieval or reranking."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    cik: str
    year: str
    section_key: str
    chunk_index: int
    text: str
    token_count: int = 0
    dense_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None
    is_neighbor: bool = False


class RetrievalSummary(BaseModel):
    """Observability for step 8/9: how many candidates each stage saw."""

    model_config = ConfigDict(frozen=True)

    dense_candidates: int = 0
    lexical_candidates: int = 0
    fused_candidates: int = 0
    reranked: int = 0
    expanded: int = 0
    neighbor_sources: int = 0
    top_score: float | None = None
    model_id: str = ""
    lexical_query: str = ""
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)

    def with_filters(self, filters: RetrievalFilters) -> RetrievalSummary:
        return self.model_copy(update={"filters": filters})


class RetrievalResult(BaseModel):
    """Candidates plus the stage counts that produced them."""

    model_config = ConfigDict(frozen=True)

    candidates: tuple[RetrievedChunk, ...] = ()
    dense_candidates: int = 0
    lexical_candidates: int = 0
    fused_candidates: int = 0
    lexical_query: str = ""
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)


class ContextSource(BaseModel):
    """A chunk selected for the prompt, with the citation id it was given."""

    model_config = ConfigDict(frozen=True)

    source_id: str  # S1, S2, ...
    chunk: RetrievedChunk


class ContextBundle(BaseModel):
    """Deterministic prompt material produced by the context builder."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...] = ()
    sources: tuple[ContextSource, ...] = ()
    token_count: int = 0
    truncated: bool = False

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)


class ContextSummary(BaseModel):
    """What actually fit into the generator prompt."""

    model_config = ConfigDict(frozen=True)

    sources: int = 0
    neighbors: int = 0
    tokens: int = 0
    truncated: bool = False


class Citation(BaseModel):
    """A source reference the caller can render and link to."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    chunk_id: str
    doc_id: str
    cik: str
    company: str | None = None
    year: str
    section_key: str
    section_title: str
    chunk_index: int
    url: str
    snippet: str
    score: float | None = None
    rerank_score: float | None = None


class Generation(BaseModel):
    """Metadata about one generator call, including cache behaviour.

    llama.cpp reports prompt-cache reuse separately from total prompt tokens:
    ``cached_tokens`` were served from the KV cache, while
    ``prompt_eval_tokens`` were actually evaluated. The rate is cached/total.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_hit_rate: float = 0.0
    prompt_eval_tokens: int = 0
    prompt_eval_ms: float = 0.0
    decode_ms: float = 0.0
    prompt_eval_tokens_per_s: float = 0.0
    decode_tokens_per_s: float = 0.0
    finish_reason: str = ""
    attempts: int = 1


class TurnRecord(BaseModel):
    """One persisted question/answer turn (step 14)."""

    model_config = ConfigDict(frozen=True)

    turn_id: str | None = None
    request_id: str
    session_id: str | None = None
    user_id: str = ANONYMOUS_USER_ID
    groups: tuple[str, ...] = ()
    question: str = ""
    rewritten_query: str | None = None
    answer: str | None = None
    status: QueryStatus = QueryStatus.ERROR
    citations: tuple[Citation, ...] = ()
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: tuple[str, ...] = ()


class Feedback(BaseModel):
    """A thumbs-up/down attached to a persisted turn."""

    model_config = ConfigDict(frozen=True)

    feedback_id: str | None = None
    turn_id: str
    user_id: str
    rating: int  # +1 or -1
    comment: str = ""


class FeedbackResult(BaseModel):
    """Outcome of a feedback write, mapped to HTTP/CLI status by the caller."""

    model_config = ConfigDict(frozen=True)

    status: str  # recorded | unauthorized | forbidden | not_found | error
    feedback: Feedback | None = None
    message: str = ""


class AskRequest(BaseModel):
    """A single inbound question."""

    query: str
    session_id: str | None = None
    token: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)
    filters: RetrievalFilters | None = None


class AskResult(BaseModel):
    """Everything the pipeline learned about one request."""

    request_id: str
    status: QueryStatus
    query: str
    normalized_query: str = ""
    rewritten_query: str | None = None
    message: str | None = None

    principal: Principal | None = None
    guard: GuardVerdict | None = None
    route: RouteDecision | None = None

    # --- steps 8-14 ---
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    turn_id: str | None = None
    retrieval: RetrievalSummary | None = None
    context: ContextSummary | None = None
    output_guard: GuardVerdict | None = None
    generation: Generation | None = None
    history_loaded: int = 0

    # --- step 7 diagnostics ---
    embedding_dim: int | None = None
    embedding_model: str | None = None
    embedding_norm: float | None = None
    embedding_preview: list[float] = Field(default_factory=list)
    embedding: list[float] | None = None  # full vector, opt-in

    warnings: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    trace: tuple[str, ...] = ()

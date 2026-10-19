"""End-to-end tests for Pipeline 2 steps 8-14 with in-memory ports."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rag_query.api import create_app
from rag_query.config import QuerySettings
from rag_query.domain.context import ContextBuilder
from rag_query.domain.models import (
    AskRequest,
    GuardVerdict,
    Principal,
    QueryStatus,
    RetrievalFilters,
)
from rag_query.domain.query_analysis import CompanyDirectory, MetadataExtractor
from rag_query.graph import build_graph
from rag_query.nodes import (
    AuthenticateNode,
    ContextNode,
    EncodeNode,
    FallbackNode,
    FinalizeNode,
    FormatAnswerNode,
    GenerateNode,
    HistoryNode,
    InputGuardNode,
    InputNode,
    OutputGuardNode,
    RerankNode,
    RetrieveNode,
    RewriteNode,
    RouteNode,
)
from rag_query.service import QueryService
from tests.fakes import CharCounter, FakeEmbedder
from tests.fakes_query import (
    FakeAuthenticator,
    FakeGenerator,
    FakeGuard,
    FakeHistoryStore,
    FakeOutputGuard,
    FakeRateLimiter,
    FakeReranker,
    FakeRetriever,
    FakeRewriter,
    FakeRouter,
    make_chunk,
)


def _companies() -> CompanyDirectory:
    return CompanyDirectory(
        name_to_ciks={"Apple Inc.": ("320193",), "Microsoft Corporation": ("789019",)},
        ticker_to_cik={"AAPL": "320193", "MSFT": "789019"},
        name_by_cik={"320193": "Apple Inc.", "789019": "Microsoft Corporation"},
    )


def _auth() -> FakeAuthenticator:
    return FakeAuthenticator(
        {
            "good-token": Principal(user_id="u1", groups=("public",), display_name="U1"),
            "other-token": Principal(user_id="u2", groups=("public",), display_name="U2"),
            "admin-token": Principal(user_id="admin", groups=("admin",), display_name="Admin"),
        }
    )


def _service(
    *,
    chunks=None,
    generator=None,
    guard=None,
    history=None,
    reranker=None,
    retriever=None,
    authenticator=None,
    rewriter=None,
    neighbor_window: int = 0,
    rerank_top_k: int = 3,
    guard_retries: int = 1,
) -> QueryService:
    companies = _companies()
    history = history if history is not None else FakeHistoryStore()
    rewriter = rewriter or FakeRewriter(result="Apple risk factors")
    retriever = retriever or FakeRetriever(
        chunks
        if chunks is not None
        else [
            make_chunk(0, fused_score=0.03),
            make_chunk(1, fused_score=0.02),
            make_chunk(2, fused_score=0.01),
        ]
    )
    reranker = reranker or FakeReranker()
    generator = generator or FakeGenerator()
    guard = guard or FakeOutputGuard()
    authenticator = authenticator or _auth()

    graph = build_graph(
        input_node=InputNode(max_chars=4000),
        authenticate_node=AuthenticateNode(authenticator),
        history_node=HistoryNode(history, max_turns=6),
        guard_node=InputGuardNode([FakeGuard()], rate_limiter=FakeRateLimiter()),
        route_node=RouteNode(FakeRouter()),
        rewrite_node=RewriteNode(rewriter, enabled=True),
        encode_node=EncodeNode(FakeEmbedder(8), expected_dim=8, check_index=False),
        fallback_node=FallbackNode(),
        retrieve_node=RetrieveNode(
            retriever,
            filter_extractor=MetadataExtractor(companies, corpus_years=("2020",)),
            top_k=40,
        ),
        rerank_node=RerankNode(
            reranker,
            top_k=rerank_top_k,
            expander=retriever,
            neighbor_window=neighbor_window,
        ),
        context_node=ContextNode(
            ContextBuilder(
                CharCounter(),
                token_budget=6000,
                max_sources=8,
                max_history_messages=6,
            )
        ),
        generate_node=GenerateNode(generator),
        output_guard_node=OutputGuardNode(guard, max_retries=guard_retries),
        format_node=FormatAnswerNode(companies),
        finalize_node=FinalizeNode(history, persist_failures=True),
    )
    return QueryService(
        graph,
        settings=QuerySettings(),
        metadata={"pipeline": "test"},
        authenticator=authenticator,
        history_store=history,
        feedback_admin_group="admin",
    )


def test_full_pipeline_answers_with_citations_and_persists() -> None:
    history = FakeHistoryStore()
    generator = FakeGenerator(
        ["Apple faces supply-chain and cybersecurity risks [S1], [S2]."]
    )
    service = _service(
        generator=generator,
        history=history,
        neighbor_window=0,
    )

    result = service.ask(
        AskRequest(query="What are Apple's risk factors in 2020?", token="good-token")
    )

    assert result.status is QueryStatus.ANSWERED
    assert result.answer is not None
    assert "[S1]" in result.answer
    assert result.citations
    assert result.citations[0].source_id == "S1"
    assert result.citations[0].company == "Apple Inc."
    assert result.citations[0].section_title == "Risk Factors"
    assert result.turn_id is not None
    assert result.retrieval is not None
    assert result.retrieval.dense_candidates == 3
    assert result.retrieval.reranked >= 3
    assert result.output_guard is not None and result.output_guard.allowed
    assert result.generation is not None and result.generation.model == "fake-generator"
    assert result.trace == (
        "input",
        "authenticate",
        "history",
        "input_guard",
        "route",
        "rewrite",
        "encode",
        "retrieve",
        "rerank",
        "context",
        "generate",
        "output_guard",
        "format",
        "finalize",
    )
    assert {"retrieve", "rerank", "context", "generate", "output_guard", "format"} <= set(
        result.timings_ms
    )
    saved = history.get_turn(result.request_id)
    assert saved is not None
    assert saved.status is QueryStatus.ANSWERED
    assert saved.answer == result.answer
    assert len(saved.citations) == len(result.citations)
    assert len(result.citations) == 3


def test_partial_refusal_with_citations_is_no_context() -> None:
    generator = FakeGenerator(
        [
            "I couldn't find the current stock price of Apple Inc. (AAPL) on the "
            "NASDAQ in the provided filings. Historical data is available [S1]."
        ]
    )
    service = _service(generator=generator, neighbor_window=0)

    result = service.ask(
        AskRequest(query="What is Apple's share price right now?", token="good-token")
    )

    assert result.status is QueryStatus.NO_CONTEXT
    assert result.answer is not None and "current stock price" in result.answer
    assert result.citations  # the useful supporting sources are still returned
    assert result.message == "I couldn't find that in the provided filings."


def test_neighbor_expansion_adds_adjacent_context_without_losing_primaries() -> None:
    service = _service(neighbor_window=1)
    result = service.ask(AskRequest(query="What are Apple's risks?", token="good-token"))
    assert result.status is QueryStatus.ANSWERED
    assert len(result.citations) == 4
    assert any(citation.chunk_index == 3 for citation in result.citations)
    assert result.citations[0].rerank_score is not None


def test_query_metadata_filters_retrieval_and_explicit_filters_win() -> None:
    chunks = [
        make_chunk(0, cik="320193", text="Apple supply-chain risk."),
        make_chunk(1, cik="789019", doc_id="789019_2020.htm", text="Microsoft risk."),
        make_chunk(2, cik="320193", section_key="7", text="Apple liquidity."),
    ]
    retriever = FakeRetriever(chunks)
    service = _service(chunks=chunks, retriever=retriever)

    extracted = service.ask(
        AskRequest(query="What are Apple's risk factors in 2020?", token="good-token")
    )
    assert extracted.status is QueryStatus.ANSWERED
    first_filters = retriever.calls[0]["filters"]
    assert first_filters.ciks == ("320193",)
    assert first_filters.years == ("2020",)
    assert first_filters.section_keys == ("1A",)
    assert all(citation.cik == "320193" for citation in extracted.citations)

    explicit = service.ask(
        AskRequest(
            query="What are Apple's risk factors?",
            token="good-token",
            filters=RetrievalFilters(ciks=("789019",)),
        )
    )
    assert explicit.status is QueryStatus.ANSWERED
    second_filters = retriever.calls[-1]["filters"]
    assert second_filters.ciks == ("789019",)
    assert all(citation.cik == "789019" for citation in explicit.citations)


def test_multi_company_retrieval_and_rerank_keep_both_sources() -> None:
    chunks = [
        make_chunk(0, cik="320193", doc_id="320193_2020.htm", fused_score=0.9),
        make_chunk(1, cik="320193", doc_id="320193_2020.htm", fused_score=0.8),
        make_chunk(2, cik="789019", doc_id="789019_2020.htm", fused_score=0.2),
        make_chunk(3, cik="789019", doc_id="789019_2020.htm", fused_score=0.1),
    ]
    retriever = FakeRetriever(chunks)
    service = _service(chunks=chunks, retriever=retriever, rerank_top_k=8)
    result = service.ask(
        AskRequest(
            query="Compare Microsoft and Apple risk factors in 2020.",
            token="good-token",
        )
    )

    assert result.status is QueryStatus.ANSWERED
    assert len(retriever.calls) == 2
    assert {call["filters"].ciks for call in retriever.calls} == {
        ("320193",),
        ("789019",),
    }
    assert result.retrieval is not None
    assert result.retrieval.filters.ciks == ("789019", "320193")
    ciks = [citation.cik for citation in result.citations]
    assert ciks.count("320193") >= 2
    assert ciks.count("789019") >= 2


def test_no_context_short_circuits_generation_and_persists() -> None:
    history = FakeHistoryStore()
    generator = FakeGenerator()
    service = _service(chunks=[], generator=generator, history=history)

    result = service.ask(AskRequest(query="What are Apple's risks?", token="good-token"))

    assert result.status is QueryStatus.NO_CONTEXT
    assert result.answer == result.message
    assert result.citations == []
    assert result.retrieval is not None
    assert result.retrieval.dense_candidates == 0
    assert not generator.calls
    assert result.trace[-2:] == ("fallback", "finalize")
    assert history.get_turn(result.request_id) is not None


def test_acl_without_groups_sees_nothing() -> None:
    auth = FakeAuthenticator(
        {"locked-token": Principal(user_id="locked", groups=())},
    )
    service = _service(authenticator=auth)
    result = service.ask(AskRequest(query="What are Apple's risks?", token="locked-token"))
    assert result.status is QueryStatus.NO_CONTEXT
    assert result.answer


def test_output_guard_retries_once_with_feedback() -> None:
    guard = FakeOutputGuard(
        [
            GuardVerdict(
                allowed=False,
                guard="fake_output_guard",
                reason="missing citation",
            ),
            GuardVerdict(allowed=True, guard="fake_output_guard", reason="ok"),
        ]
    )
    generator = FakeGenerator(["An uncited answer.", "A verified answer [S1]."])
    service = _service(guard=guard, generator=generator)

    result = service.ask(AskRequest(query="What are Apple's risks?", token="good-token"))

    assert result.status is QueryStatus.ANSWERED
    assert result.answer == "A verified answer [S1]."
    assert result.generation is not None and result.generation.attempts == 2
    assert len(generator.calls) == 2
    assert generator.calls[0][1] == ""
    assert "missing citation" in generator.calls[1][1]
    assert result.output_guard is not None and result.output_guard.allowed
    assert any("output_guard_retry" in warning for warning in result.warnings)


def test_output_guard_blocks_after_retry_with_safe_answer() -> None:
    guard = FakeOutputGuard(allowed=False, reason="ungrounded")
    generator = FakeGenerator(["Bad one.", "Bad two."])
    history = FakeHistoryStore()
    service = _service(guard=guard, generator=generator, history=history)

    result = service.ask(AskRequest(query="What are Apple's risks?", token="good-token"))

    assert result.status is QueryStatus.OUTPUT_BLOCKED
    assert result.answer is not None
    assert "couldn't produce a fully verified answer" in result.answer
    assert result.citations
    assert len(generator.calls) == 2
    assert result.output_guard is not None and not result.output_guard.allowed
    saved = history.get_turn(result.request_id)
    assert saved is not None and saved.status is QueryStatus.OUTPUT_BLOCKED


def test_history_is_loaded_before_rewrite() -> None:
    from rag_query.domain.models import ChatMessage

    history = FakeHistoryStore(
        [
            ChatMessage(role="user", content="What was Apple's revenue?"),
            ChatMessage(role="assistant", content="It was reported in Item 7 [S1]."),
        ]
    )
    rewriter = FakeRewriter(result="Apple revenue 2020")
    service = _service(history=history, rewriter=rewriter)

    result = service.ask(
        AskRequest(
            query="And its risk factors?",
            session_id="s1",
            token="good-token",
        )
    )

    assert result.status is QueryStatus.ANSWERED
    assert result.history_loaded == 2
    assert [message.content for message in rewriter.seen_history] == [
        "What was Apple's revenue?",
        "It was reported in Item 7 [S1].",
    ]


def test_reranker_failure_keeps_retrieval_order() -> None:
    reranker = FakeReranker(error=RuntimeError("boom"))
    service = _service(reranker=reranker)
    result = service.ask(AskRequest(query="What are Apple's risks?", token="good-token"))
    assert result.status is QueryStatus.ANSWERED
    assert any("rerank_failed" in warning for warning in result.warnings)
    assert result.citations


def test_feedback_ownership_and_validation() -> None:
    history = FakeHistoryStore()
    service = _service(history=history)
    result = service.ask(AskRequest(query="What are Apple's risks?", token="good-token"))

    recorded = service.feedback("good-token", result.request_id, 1, "helpful")
    assert recorded.status == "recorded"
    assert recorded.feedback is not None
    assert recorded.feedback.rating == 1

    forbidden = service.feedback("other-token", result.request_id, -1)
    assert forbidden.status == "forbidden"

    admin = service.feedback("admin-token", result.request_id, 1, "reviewed")
    assert admin.status == "recorded"

    missing = service.feedback("good-token", "does-not-exist", 1)
    assert missing.status == "not_found"

    unauthorized = service.feedback("bad-token", result.request_id, 1)
    assert unauthorized.status == "unauthorized"

    invalid = service.feedback("good-token", result.request_id, 7)
    assert invalid.status == "error"


def test_verified_stream_chunks_the_guarded_answer() -> None:
    service = _service(
        generator=FakeGenerator(["Apple faces verified risks [S1]."])
    )
    request = AskRequest(query="What are Apple's risks?", token="good-token")

    events = list(service.ask_stream(request, chunk_chars=8))

    assert events[-1]["event"] == "done"
    tokens = "".join(
        event["data"]["text"] for event in events if event["event"] == "token"
    )
    assert tokens == "Apple faces verified risks [S1]."
    assert events[-1]["data"]["status"] == QueryStatus.ANSWERED.value


def test_api_ask_stream_and_feedback() -> None:
    settings = QuerySettings()
    service = _service()
    app = create_app(settings=settings, service=service)

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"query": "What are Apple's risk factors?"},
            headers={"X-Dev-Token": "good-token"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "answered"
        request_id = body["request_id"]

        unauthorized = client.post(
            "/ask",
            json={"query": "What are Apple's risk factors?"},
            headers={"X-Dev-Token": "bad-token"},
        )
        assert unauthorized.status_code == 401
        assert unauthorized.headers["www-authenticate"] == "Bearer"

        stream = client.post(
            "/ask/stream",
            json={"query": "What are Apple's risk factors?"},
            headers={"Authorization": "Bearer good-token"},
        )
        assert stream.status_code == 200
        assert "event: token" in stream.text
        assert "event: done" in stream.text
        done_line = next(
            line for line in stream.text.splitlines() if line.startswith("data: ")
            and '"status"' in line
        )
        done = json.loads(done_line[len("data: ") :])
        assert done["status"] == "answered"

        feedback = client.post(
            "/feedback",
            json={"request_id": request_id, "rating": 1, "comment": "good"},
            headers={"X-Dev-Token": "good-token"},
        )
        assert feedback.status_code == 200
        assert feedback.json()["status"] == "recorded"

        wrong_user = client.post(
            "/feedback",
            json={"request_id": request_id, "rating": -1},
            headers={"X-Dev-Token": "other-token"},
        )
        assert wrong_user.status_code == 403

        missing = client.post(
            "/feedback",
            json={"request_id": "nope", "rating": 1},
            headers={"X-Dev-Token": "good-token"},
        )
        assert missing.status_code == 404

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["pipeline"] == "test"


def test_api_serves_the_test_console() -> None:
    app = create_app(settings=QuerySettings(), service=_service())

    with TestClient(app) as client:
        for path in ("/", "/ui"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/html")
            assert "Enterprise RAG" in response.text
            assert "/ask/stream" in response.text

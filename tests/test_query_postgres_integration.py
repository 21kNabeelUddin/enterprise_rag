"""Read-only integration tests for query retrieval, ACLs, neighbours and history.

These tests never truncate corpus tables. They are skipped unless
``RAG_QUERY_TEST_POSTGRES_DSN`` is set explicitly, so a normal unit-test run
cannot touch production data::

    $env:RAG_QUERY_TEST_POSTGRES_DSN = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    pytest tests/test_query_postgres_integration.py -q

History tests write a uniquely-named turn and remove it (and its feedback) in a
``finally`` block. If the process is killed between those steps, delete rows
whose ``request_id`` starts with ``it-query-``.
"""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from rag_ingest.adapters.storage.postgres.engine import build_engine
from rag_query.adapters.storage.postgres.history import PostgresHistoryStore
from rag_query.adapters.storage.postgres.retriever import PostgresHybridRetriever
from rag_query.adapters.storage.postgres.tables import (
    AnswerFeedbackRow,
    QueryTurnRow,
)
from rag_query.adapters.storage.postgres.vector_probe import StoredVectorProbe
from rag_query.domain.models import (
    Citation,
    Principal,
    QueryStatus,
    RetrievalFilters,
    TurnRecord,
)

DSN = os.environ.get("RAG_QUERY_TEST_POSTGRES_DSN")
pytestmark = pytest.mark.skipif(
    not DSN,
    reason="RAG_QUERY_TEST_POSTGRES_DSN not set (read-only query integration tests)",
)

MODEL_ID = "BAAI/bge-large-en-v1.5"
DIM = 1024
PUBLIC = Principal(user_id="it-query-user", groups=("public",))


@pytest.fixture(scope="module")
def engine():
    assert DSN is not None
    return build_engine(DSN, connect_timeout=10)


@pytest.fixture(scope="module")
def retriever(engine) -> PostgresHybridRetriever:
    return PostgresHybridRetriever(
        engine,
        model_id=MODEL_ID,
        dim=DIM,
        dense_k=20,
        lexical_k=20,
        fused_k=10,
        statement_timeout_ms=15_000,
    )


@pytest.fixture(scope="module")
def vector() -> np.ndarray:
    # The vector only needs to be a valid unit vector: these tests assert
    # filter and ACL semantics, not semantic relevance. Using a constant avoids
    # loading BGE for an integration test.
    values = np.ones(DIM, dtype=np.float32)
    return values / np.linalg.norm(values)


def test_filtered_retrieval_hits_real_index(retriever, vector) -> None:
    result = retriever.retrieve(
        "risk factors",
        vector,
        principal=PUBLIC,
        filters=RetrievalFilters(
            ciks=("320193",),
            years=("2020",),
            section_keys=("1A",),
        ),
        limit=6,
    )

    assert result.candidates
    assert len(result.candidates) <= 6
    assert result.dense_candidates > 0
    assert result.lexical_candidates > 0
    assert result.fused_candidates >= len(result.candidates)
    assert result.lexical_query
    assert retriever.model_id == MODEL_ID
    assert all(chunk.cik == "320193" for chunk in result.candidates)
    assert all(chunk.year == "2020" for chunk in result.candidates)
    assert all(chunk.section_key == "1A" for chunk in result.candidates)
    assert all(chunk.fused_score is not None for chunk in result.candidates)


def test_empty_acl_groups_return_no_candidates(retriever, vector) -> None:
    result = retriever.retrieve(
        "risk factors",
        vector,
        principal=Principal(user_id="it-no-groups", groups=()),
        filters=RetrievalFilters(ciks=("320193",), years=("2020",)),
        limit=6,
    )

    assert result.candidates == ()
    assert result.dense_candidates == 0
    assert result.lexical_candidates == 0
    assert result.fused_candidates == 0


def test_neighbor_expansion_preserves_primary_scores_and_acl(retriever, vector) -> None:
    result = retriever.retrieve(
        "risk factors",
        vector,
        principal=PUBLIC,
        filters=RetrievalFilters(
            ciks=("320193",),
            years=("2020",),
            section_keys=("1A",),
        ),
        limit=3,
    )
    primaries = list(result.candidates)
    assert primaries

    expanded = retriever.expand_neighbors(primaries, window=1, principal=PUBLIC)
    assert expanded
    by_id = {chunk.chunk_id: chunk for chunk in expanded}
    assert len(by_id) == len(expanded), "neighbour expansion must not duplicate chunks"

    for primary in primaries:
        same = by_id[primary.chunk_id]
        assert same.is_neighbor is False
        assert same.fused_score == primary.fused_score
        assert same.rank == primary.rank
        assert same.cik == primary.cik
        assert same.doc_id == primary.doc_id
        assert same.section_key == primary.section_key

    neighbors = [chunk for chunk in expanded if chunk.is_neighbor]
    assert neighbors
    for neighbor in neighbors:
        assert neighbor.fused_score is None
        assert any(
            neighbor.doc_id == primary.doc_id
            and neighbor.section_key == primary.section_key
            and abs(neighbor.chunk_index - primary.chunk_index) <= 1
            for primary in primaries
        )

    # A principal in an unrelated group must not receive neighbour text even
    # though it already holds the primary chunks.
    private = Principal(user_id="it-query-private", groups=("private",))
    no_neighbors = retriever.expand_neighbors(primaries, window=1, principal=private)
    assert no_neighbors
    assert all(chunk.is_neighbor is False for chunk in no_neighbors)


def test_vector_probe_reports_the_index_fingerprint(engine) -> None:
    info = StoredVectorProbe(engine, ttl_s=0).describe(force=True)

    assert info.count > 0
    assert info.model_ids == (MODEL_ID,)
    assert info.dims == (DIM,)


def test_history_round_trip_and_feedback_upsert(engine) -> None:
    store = PostgresHistoryStore(engine)
    request_id = f"it-query-{uuid4().hex}"
    session_id = f"it-session-{uuid4().hex}"
    turn = TurnRecord(
        request_id=request_id,
        session_id=session_id,
        user_id=PUBLIC.user_id,
        groups=PUBLIC.groups,
        question="What are Apple's risk factors?",
        rewritten_query="Apple Inc. risk factors 2020",
        answer="Apple describes supply-chain and pandemic risks [S1].",
        status=QueryStatus.ANSWERED,
        citations=(
            Citation(
                source_id="S1",
                chunk_id="320193_2020.htm:1A:13",
                doc_id="320193_2020.htm",
                cik="320193",
                year="2020",
                section_key="1A",
                chunk_index=13,
                section_title="Risk Factors",
                url="https://www.sec.gov/edgar/browse/?CIK=320193",
                snippet="Apple describes supply-chain and pandemic risks.",
            ),
        ),
    )

    turn_id: str | None = None
    try:
        turn_id = store.save_turn(turn)
        loaded = store.get_turn(request_id)
        assert loaded is not None
        assert loaded.turn_id == turn_id
        assert loaded.answer == turn.answer
        assert loaded.citations[0].chunk_id == "320193_2020.htm:1A:13"

        recent = store.recent(session_id, PUBLIC.user_id, limit=6)
        assert [message.role for message in recent] == ["user", "assistant"]
        assert recent[0].content == turn.question
        assert recent[1].content == turn.answer

        first = store.save_feedback(turn_id, PUBLIC.user_id, 1, "helpful")
        second = store.save_feedback(turn_id, PUBLIC.user_id, -1, "changed my mind")
        assert first.feedback_id == second.feedback_id
        assert second.rating == -1

        with Session(engine) as session:
            rows = session.scalars(
                select(AnswerFeedbackRow).where(AnswerFeedbackRow.turn_id == turn_id)
            ).all()
        assert len(rows) == 1
        assert rows[0].rating == -1
        assert rows[0].comment == "changed my mind"
    finally:
        if turn_id is not None:
            with Session(engine) as session:
                session.execute(
                    delete(AnswerFeedbackRow).where(AnswerFeedbackRow.turn_id == turn_id)
                )
                session.execute(delete(QueryTurnRow).where(QueryTurnRow.turn_id == turn_id))
                session.commit()

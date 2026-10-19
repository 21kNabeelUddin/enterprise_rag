"""Postgres history and feedback store (step 14)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from rag_query.adapters.storage.postgres.tables import (
    AnswerFeedbackRow,
    QueryTurnRow,
)
from rag_query.domain.models import (
    ChatMessage,
    Citation,
    Feedback,
    QueryStatus,
    TurnRecord,
)
from rag_query.logging import get_logger

log = get_logger(__name__)


class PostgresHistoryStore:
    def __init__(self, engine: Engine) -> None:
        self._session_factory = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)

    def recent(
        self,
        session_id: str,
        user_id: str,
        *,
        limit: int = 6,
    ) -> list[ChatMessage]:
        if not session_id or limit <= 0:
            return []
        stmt = (
            select(QueryTurnRow.question, QueryTurnRow.answer)
            .where(
                QueryTurnRow.session_id == session_id,
                QueryTurnRow.user_id == user_id,
                QueryTurnRow.status == QueryStatus.ANSWERED.value,
                QueryTurnRow.answer.is_not(None),
            )
            .order_by(QueryTurnRow.created_at.desc())
            .limit(limit)
        )
        with self._session_factory() as session:
            rows = session.execute(stmt).all()

        messages: list[ChatMessage] = []
        for question, answer in reversed(rows):
            if question:
                messages.append(ChatMessage(role="user", content=question))
            if answer:
                messages.append(ChatMessage(role="assistant", content=answer))
        return messages

    def save_turn(self, turn: TurnRecord) -> str:
        values = {
            "request_id": turn.request_id,
            "session_id": turn.session_id,
            "user_id": turn.user_id,
            "groups": list(turn.groups),
            "question": turn.question,
            "rewritten_query": turn.rewritten_query,
            "answer": turn.answer,
            "status": turn.status.value,
            "citations": [citation.model_dump(mode="json") for citation in turn.citations],
            "timings": dict(turn.timings_ms),
            "trace": list(turn.trace),
        }
        stmt = insert(QueryTurnRow).values(**values)
        stmt = stmt.on_conflict_do_update(
            index_elements=[QueryTurnRow.request_id],
            set_={
                "session_id": stmt.excluded.session_id,
                "user_id": stmt.excluded.user_id,
                "groups": stmt.excluded.groups,
                "question": stmt.excluded.question,
                "rewritten_query": stmt.excluded.rewritten_query,
                "answer": stmt.excluded.answer,
                "status": stmt.excluded.status,
                "citations": stmt.excluded.citations,
                "timings": stmt.excluded.timings,
                "trace": stmt.excluded.trace,
            },
        ).returning(QueryTurnRow.turn_id)
        with self._session_factory() as session:
            turn_id = str(session.execute(stmt).scalar_one())
            session.commit()
        log.debug("persisted turn %s (request %s)", turn_id, turn.request_id)
        return turn_id

    def get_turn(self, request_id: str) -> TurnRecord | None:
        with self._session_factory() as session:
            row = session.scalar(
                select(QueryTurnRow).where(QueryTurnRow.request_id == request_id)
            )
            if row is None:
                return None
            return _to_turn(row)

    def save_feedback(
        self,
        turn_id: str,
        user_id: str,
        rating: int,
        comment: str = "",
    ) -> Feedback:
        if rating not in (-1, 1):
            raise ValueError("rating must be +1 or -1")
        turn_uuid = UUID(str(turn_id))
        stmt = insert(AnswerFeedbackRow).values(
            turn_id=turn_uuid,
            user_id=user_id,
            rating=rating,
            comment=comment or "",
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_answer_feedback_turn_user",
            set_={"rating": stmt.excluded.rating, "comment": stmt.excluded.comment},
        ).returning(AnswerFeedbackRow.feedback_id)
        with self._session_factory() as session:
            feedback_id = str(session.execute(stmt).scalar_one())
            session.commit()
        return Feedback(
            feedback_id=feedback_id,
            turn_id=str(turn_id),
            user_id=user_id,
            rating=rating,
            comment=comment or "",
        )


def _to_turn(row: QueryTurnRow) -> TurnRecord:
    citations: list[Citation] = []
    for item in row.citations or []:
        try:
            citations.append(Citation.model_validate(item))
        except Exception:  # noqa: BLE001 - old/partial rows must not break reads
            continue
    try:
        status = QueryStatus(row.status)
    except ValueError:
        status = QueryStatus.ERROR
    return TurnRecord(
        turn_id=str(row.turn_id),
        request_id=row.request_id,
        session_id=row.session_id,
        user_id=row.user_id,
        groups=tuple(row.groups or ()),
        question=row.question or "",
        rewritten_query=row.rewritten_query,
        answer=row.answer,
        status=status,
        citations=tuple(citations),
        timings_ms=dict(row.timings or {}),
        trace=tuple(row.trace or ()),
    )




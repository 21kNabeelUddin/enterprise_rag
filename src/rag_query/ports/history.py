"""Conversation history and feedback storage port (step 14)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_query.domain.models import ChatMessage, Feedback, TurnRecord


@runtime_checkable
class HistoryStore(Protocol):
    """Persist turns and load recent history for the rewriter.

    History is scoped by both ``session_id`` and ``user_id``: a guessed session
    id must never expose another caller's conversation.
    """

    def recent(
        self,
        session_id: str,
        user_id: str,
        *,
        limit: int = 6,
    ) -> list[ChatMessage]: ...

    def save_turn(self, turn: TurnRecord) -> str:
        """Upsert by ``request_id`` and return the persisted ``turn_id``."""
        ...

    def get_turn(self, request_id: str) -> TurnRecord | None: ...

    def save_feedback(
        self,
        turn_id: str,
        user_id: str,
        rating: int,
        comment: str = "",
    ) -> Feedback: ...

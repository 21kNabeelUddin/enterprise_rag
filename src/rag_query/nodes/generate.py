"""Step 11 -- generate a grounded answer from the context bundle."""

from __future__ import annotations

from typing import Any

from rag_query.domain.models import QueryStatus
from rag_query.logging import get_logger
from rag_query.nodes.base import Node
from rag_query.ports.generator import Generator
from rag_query.state import QueryState

log = get_logger(__name__)


class GenerateNode(Node):
    name = "generate"

    def __init__(self, generator: Generator) -> None:
        self._generator = generator

    def run(self, state: QueryState) -> dict[str, Any]:
        bundle = state.get("context")
        if bundle is None or not bundle.sources:
            return {
                "status": QueryStatus.NO_CONTEXT,
                "message": "There is no context available to generate an answer.",
            }

        attempts = int(state.get("generation_attempts") or 0) + 1
        feedback = state.get("output_feedback") or ""
        try:
            generation = self._generator.generate(bundle, strict_feedback=feedback)
        except Exception as exc:  # noqa: BLE001
            log.exception("generation failed")
            return {
                "status": QueryStatus.ERROR,
                "message": f"Answer generation failed: {exc}",
                "generation_attempts": attempts,
            }

        if not (generation.text or "").strip():
            return {
                "status": QueryStatus.ERROR,
                "message": "The generator returned an empty answer.",
                "generation_attempts": attempts,
            }

        return {
            "generation": generation,
            "answer": generation.text,
            "generation_attempts": attempts,
        }

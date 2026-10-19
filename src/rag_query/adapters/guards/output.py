"""Output guardrails (step 12).

Four independent checks run after generation:

* citations  -- every claim is cited and every marker resolves to a real source
* grounding  -- answer sentences are semantically close to the cited passages
* PII        -- the answer does not leak personal data
* leakage    -- the answer does not expose the system prompt or secrets

Each adapter returns a :class:`GuardVerdict`; the composite fails on the first
declining check while preserving scores and labels for observability.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from rag_ingest.ports.embedder import Embedder
from rag_query.domain.citations import extract_citation_ids, is_no_answer
from rag_query.domain.models import ContextSource, GuardVerdict
from rag_query.logging import get_logger

log = get_logger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")
_API_KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20,})\b")

_LEAK_MARKERS = (
    "You are an enterprise analyst answering questions about SEC 10-K filings",
    "Answer ONLY from the supplied context",
    "RAG_DEV_TOKENS",
    "RAG_POSTGRES_DSN",
    "RAG_GUARD_MODEL_DIR",
    "<|im_start|>",
    "<|im_end|>",
)


class CitationOutputGuard:
    name = "output_citations"

    def __init__(self, *, min_citations: int = 1) -> None:
        self._min_citations = max(0, min_citations)

    def check(
        self,
        answer: str,
        sources: Sequence[ContextSource],
        *,
        query: str,
    ) -> GuardVerdict:
        answer = answer or ""
        if not answer.strip():
            return GuardVerdict(allowed=False, guard=self.name, reason="empty answer")

        if is_no_answer(answer):
            return GuardVerdict(
                allowed=True,
                guard=self.name,
                reason="model reported the answer is not in the context",
                labels=("no_answer",),
            )

        valid = {source.source_id for source in sources}
        cited = extract_citation_ids(answer)
        unknown = [source_id for source_id in cited if source_id not in valid]
        if unknown:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason=f"answer cites unknown source(s): {', '.join(unknown)}",
                labels=("unknown_citation",),
            )
        if len(cited) < self._min_citations:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason=(
                    f"answer has {len(cited)} citation(s); "
                    f"at least {self._min_citations} required"
                ),
                labels=("missing_citation",),
            )
        return GuardVerdict(
            allowed=True,
            guard=self.name,
            reason=f"{len(cited)} citation(s) validated",
            scores={"citations": float(len(cited))},
        )


class GroundingOutputGuard:
    name = "output_grounding"

    def __init__(
        self,
        embedder: Embedder,
        *,
        threshold: float = 0.45,
        min_grounded_ratio: float = 0.6,
        min_sentence_chars: int = 20,
    ) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._min_ratio = min_grounded_ratio
        self._min_sentence_chars = max(8, min_sentence_chars)

    def check(
        self,
        answer: str,
        sources: Sequence[ContextSource],
        *,
        query: str,
    ) -> GuardVerdict:
        if is_no_answer(answer):
            return GuardVerdict(
                allowed=True,
                guard=self.name,
                reason="no-answer response",
                labels=("no_answer",),
            )
        if not sources:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason="no sources available to ground the answer",
            )

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SPLIT_RE.split(answer or "")
            if len(sentence.strip()) >= self._min_sentence_chars
        ]
        if not sentences:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason="answer has no checkable sentences",
            )

        all_texts = [source.chunk.text for source in sources]
        try:
            source_vectors = self._embedder.embed(all_texts)
            sentence_vectors = self._embedder.embed(sentences)
        except Exception as exc:  # noqa: BLE001 - never fail the request on infra
            log.warning("grounding check unavailable: %s", exc)
            return GuardVerdict(
                allowed=True,
                guard=self.name,
                reason="grounding checker unavailable",
                labels=("grounding_skipped",),
            )

        source_vectors = _normalise(source_vectors)
        sentence_vectors = _normalise(sentence_vectors)
        per_sentence: list[float] = []
        for sentence, vector in zip(sentences, sentence_vectors, strict=True):
            cited = extract_citation_ids(sentence)
            indices = [
                index
                for index, source in enumerate(sources)
                if source.source_id in cited
            ] or list(range(len(all_texts)))
            similarities = source_vectors[indices] @ vector
            per_sentence.append(float(np.max(similarities)) if len(similarities) else 0.0)

        grounded = [score for score in per_sentence if score >= self._threshold]
        ratio = len(grounded) / len(per_sentence)
        scores = {
            "grounding_mean": round(float(np.mean(per_sentence)), 6),
            "grounding_min": round(float(np.min(per_sentence)), 6),
            "grounding_ratio": round(ratio, 6),
        }
        if ratio < self._min_ratio:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason=(
                    f"{ratio:.0%} of sentences are grounded; "
                    f"{self._min_ratio:.0%} required"
                ),
                scores=scores,
                labels=("ungrounded",),
            )
        return GuardVerdict(
            allowed=True,
            guard=self.name,
            reason=f"{ratio:.0%} of sentences grounded",
            scores=scores,
        )


class PiiOutputGuard:
    name = "output_pii"

    def __init__(self, patterns: dict[str, re.Pattern[str]] | None = None) -> None:
        self._patterns = patterns or {
            "email": _EMAIL_RE,
            "ssn": _SSN_RE,
            "credit_card": _CARD_RE,
            "api_key": _API_KEY_RE,
        }

    def check(
        self,
        answer: str,
        sources: Sequence[ContextSource],
        *,
        query: str,
    ) -> GuardVerdict:
        findings = tuple(
            sorted(
                {
                    name
                    for name, pattern in self._patterns.items()
                    if pattern.search(answer or "")
                }
            )
        )
        if findings:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason=f"answer contains disallowed PII: {', '.join(findings)}",
                scores={name: 1.0 for name in findings},
                labels=findings,
            )
        return GuardVerdict(allowed=True, guard=self.name, reason="no output PII detected")


class LeakageOutputGuard:
    name = "output_leakage"

    def __init__(self, markers: Sequence[str] | None = None) -> None:
        self._markers = tuple(markers or _LEAK_MARKERS)

    def check(
        self,
        answer: str,
        sources: Sequence[ContextSource],
        *,
        query: str,
    ) -> GuardVerdict:
        lowered = (answer or "").lower()
        hits = [marker for marker in self._markers if marker.lower() in lowered]
        if hits:
            return GuardVerdict(
                allowed=False,
                guard=self.name,
                reason="answer appears to expose prompt or configuration text",
                labels=("leakage",),
            )
        return GuardVerdict(
            allowed=True,
            guard=self.name,
            reason="no prompt leakage detected",
        )


class CompositeOutputGuard:
    name = "output"

    def __init__(self, guards: Sequence[object]) -> None:
        self._guards = list(guards)

    @property
    def guards(self) -> tuple[object, ...]:
        return tuple(self._guards)

    def check(
        self,
        answer: str,
        sources: Sequence[ContextSource],
        *,
        query: str,
    ) -> GuardVerdict:
        if not self._guards:
            return GuardVerdict(
                allowed=True,
                guard=self.name,
                reason="no output guards configured",
            )

        verdicts: list[GuardVerdict] = []
        for guard in self._guards:
            try:
                verdicts.append(guard.check(answer, sources, query=query))  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                log.exception("output guard %s failed", getattr(guard, "name", guard))
                verdicts.append(
                    GuardVerdict(
                        allowed=False,
                        guard=getattr(guard, "name", "output_guard"),
                        reason=f"guard error: {exc}",
                        labels=("guard_error",),
                    )
                )

        failing = next((verdict for verdict in verdicts if not verdict.allowed), None)
        scores: dict[str, float] = {}
        labels: list[str] = []
        notes: list[str] = []
        for verdict in verdicts:
            scores.update(verdict.scores)
            labels.extend(label for label in verdict.labels if label not in labels)
            if verdict.reason:
                notes.append(f"{verdict.guard}: {verdict.reason}")

        return GuardVerdict(
            allowed=failing is None,
            guard=failing.guard if failing else self.name,
            reason=failing.reason if failing else "; ".join(notes),
            scores=scores,
            labels=tuple(labels),
        )


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms

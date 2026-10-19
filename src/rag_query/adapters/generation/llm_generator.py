"""Grounded answer generator backed by the shared OpenAI-compatible LLM client."""

from __future__ import annotations

import time

from rag_query.domain.models import ChatMessage, ContextBundle, Generation
from rag_query.logging import get_logger
from rag_query.ports.llm import LLMClient

log = get_logger(__name__)


class LlmAnswerGenerator:
    def __init__(
        self,
        llm: LLMClient,
        *,
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 768,
        max_retries: int = 1,
        retry_sleep_s: float = 0.5,
    ) -> None:
        self._llm = llm
        self._model = model
        self._temperature = temperature
        self._max_tokens = max(64, max_tokens)
        self._max_retries = max(0, max_retries)
        self._retry_sleep_s = max(0.0, retry_sleep_s)

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        bundle: ContextBundle,
        *,
        strict_feedback: str = "",
    ) -> Generation:
        messages = list(bundle.messages)
        if strict_feedback:
            messages.append(
                ChatMessage(
                    role="user",
                    content=(
                        "Your previous answer was rejected by the output guardrail: "
                        f"{strict_feedback}. Regenerate the answer using only the context "
                        "above, cite every factual claim with [S#], and do not add outside "
                        "knowledge."
                    ),
                )
            )

        attempts = 0
        last_error: Exception | None = None
        while attempts <= self._max_retries:
            attempts += 1
            try:
                text = self._llm.chat(
                    messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                usage = getattr(self._llm, "last_usage", {}) or {}
                timings = getattr(self._llm, "last_timings", {}) or {}
                prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                cached_tokens = int(
                    timings.get("cache_n", usage.get("cached_tokens", 0)) or 0
                )
                if prompt_tokens <= 0 and timings:
                    prompt_tokens = int(
                        (timings.get("prompt_n", 0) or 0)
                        + (timings.get("cache_n", 0) or 0)
                    )
                if completion_tokens <= 0 and timings:
                    completion_tokens = int(timings.get("predicted_n", 0) or 0)
                if prompt_tokens > 0:
                    cached_tokens = min(cached_tokens, prompt_tokens)
                prompt_eval_tokens = int(
                    timings.get("prompt_n", max(prompt_tokens - cached_tokens, 0)) or 0
                )
                prompt_eval_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
                decode_ms = float(timings.get("predicted_ms", 0.0) or 0.0)
                prompt_tps = float(timings.get("prompt_per_second", 0.0) or 0.0)
                if prompt_tps <= 0 and prompt_eval_ms > 0 and prompt_eval_tokens > 0:
                    prompt_tps = prompt_eval_tokens / (prompt_eval_ms / 1000.0)
                decode_tps = float(timings.get("predicted_per_second", 0.0) or 0.0)
                if decode_tps <= 0 and decode_ms > 0 and completion_tokens > 0:
                    decode_tps = completion_tokens / (decode_ms / 1000.0)
                return Generation(
                    text=(text or "").strip(),
                    model=self._model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    cache_hit_rate=(
                        cached_tokens / prompt_tokens if prompt_tokens > 0 else 0.0
                    ),
                    prompt_eval_tokens=prompt_eval_tokens,
                    prompt_eval_ms=prompt_eval_ms,
                    decode_ms=decode_ms,
                    prompt_eval_tokens_per_s=prompt_tps,
                    decode_tokens_per_s=decode_tps,
                    finish_reason=str(getattr(self._llm, "last_finish_reason", "") or ""),
                    attempts=attempts,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.warning("generation attempt %d failed: %s", attempts, exc)
                if attempts <= self._max_retries:
                    time.sleep(self._retry_sleep_s)

        raise RuntimeError(
            f"generation failed after {attempts} attempt(s): {last_error}"
        ) from last_error

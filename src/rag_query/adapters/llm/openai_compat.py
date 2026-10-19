"""OpenAI-compatible chat client (targets the local llama.cpp server).

Works with any server that speaks ``POST /v1/chat/completions`` -- llama.cpp,
vLLM, Ollama or a hosted API. Qwen "thinking" is disabled by default via
``chat_template_kwargs`` so the model answers instead of spending the token
budget on reasoning.
"""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from rag_query.domain.models import ChatMessage
from rag_query.logging import get_logger

log = get_logger(__name__)


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        model: str = "local",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout_s: float = 180.0,
        enable_thinking: bool = False,
        cache_prompt: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._enable_thinking = enable_thinking
        self._cache_prompt = cache_prompt
        self._client = client or httpx.Client()
        self._last_usage: dict[str, int] = {}
        self._last_timings: dict[str, float] = {}
        self._last_finish_reason = ""

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def last_usage(self) -> dict[str, int]:
        """Token usage from the most recent successful call, when reported."""
        return dict(self._last_usage)

    @property
    def last_timings(self) -> dict[str, float]:
        """llama.cpp server timings for the most recent call, when reported.

        Includes ``cache_n`` (prompt tokens reused from the KV cache),
        ``prompt_n`` (tokens actually evaluated), ``prompt_ms`` and
        ``predicted_ms``. Other OpenAI-compatible servers return an empty dict.
        """
        return dict(self._last_timings)

    @property
    def last_finish_reason(self) -> str:
        return self._last_finish_reason

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature if temperature is None else temperature,
            "max_tokens": self._max_tokens if max_tokens is None else max_tokens,
            "stream": False,
            "cache_prompt": self._cache_prompt,
        }
        if not self._enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._base_url}/chat/completions"
        log.debug("POST %s (%d messages)", url, len(messages))
        response = self._client.post(url, json=payload, headers=headers, timeout=self._timeout_s)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        self._last_usage = {
            str(key): int(value)
            for key, value in usage.items()
            if isinstance(value, int | float)
        }
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if isinstance(cached, int | float):
            self._last_usage["cached_tokens"] = int(cached)
        timings = data.get("timings") or {}
        self._last_timings = {
            str(key): float(value)
            for key, value in timings.items()
            if isinstance(value, int | float)
        }
        choice = (data.get("choices") or [{}])[0]
        self._last_finish_reason = str(choice.get("finish_reason") or "")
        return choice.get("message", {}).get("content") or ""

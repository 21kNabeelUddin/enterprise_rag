"""LLM client telemetry: prompt-cache and decode metrics (no GPU, no network)."""

from __future__ import annotations

import httpx

from rag_query.adapters.generation.llm_generator import LlmAnswerGenerator
from rag_query.adapters.llm.openai_compat import OpenAICompatibleLLM
from rag_query.domain.models import ChatMessage, ContextBundle


def _response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 19,
                "completion_tokens": 2,
                "total_tokens": 21,
                "prompt_tokens_details": {"cached_tokens": 15},
            },
            "timings": {
                "cache_n": 15,
                "prompt_n": 4,
                "prompt_ms": 740.961,
                "prompt_per_second": 25.64,
                "predicted_n": 2,
                "predicted_ms": 242.265,
                "predicted_per_second": 4.13,
            },
        },
    )


def test_openai_client_exposes_cache_and_timing_fields() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read().decode())
        return _response()

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM("http://test/v1", model="test", client=client)

    assert llm.chat([ChatMessage(role="user", content="hi")]) == "ok"
    assert llm.last_usage["cached_tokens"] == 15
    assert llm.last_timings["cache_n"] == 15
    assert llm.last_timings["prompt_n"] == 4
    assert '"cache_prompt":true' in seen[0]


def test_generator_maps_cache_and_throughput_telemetry() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: _response()))
    llm = OpenAICompatibleLLM("http://test/v1", model="test", client=client)
    bundle = ContextBundle(
        messages=(
            ChatMessage(role="system", content="answer from context"),
            ChatMessage(role="user", content="question"),
        ),
        token_count=19,
    )

    generation = LlmAnswerGenerator(llm, model="test").generate(bundle)

    assert generation.prompt_tokens == 19
    assert generation.completion_tokens == 2
    assert generation.cached_tokens == 15
    assert generation.cache_hit_rate == 15 / 19
    assert generation.prompt_eval_tokens == 4
    assert generation.prompt_eval_ms == 740.961
    assert generation.decode_ms == 242.265
    assert generation.prompt_eval_tokens_per_s == 25.64
    assert generation.decode_tokens_per_s == 4.13


def test_client_can_disable_prompt_cache() -> None:
    payloads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(request.read().decode())
        return _response()

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = OpenAICompatibleLLM(
        "http://test/v1", model="test", cache_prompt=False, client=client
    )
    llm.chat([ChatMessage(role="user", content="hi")])

    assert '"cache_prompt":false' in payloads[0]

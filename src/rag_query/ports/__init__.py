"""Ports (Protocols) the query pipeline depends on.

Nodes are written against these, never against concrete adapters, which is what
keeps the pipeline testable with in-memory fakes and swappable in production.
"""

from __future__ import annotations

from rag_query.ports.auth import Authenticator
from rag_query.ports.generator import Generator
from rag_query.ports.guard import QueryGuard
from rag_query.ports.history import HistoryStore
from rag_query.ports.llm import LLMClient
from rag_query.ports.output_guard import OutputGuard
from rag_query.ports.rate_limit import RateLimiter
from rag_query.ports.reranker import Reranker
from rag_query.ports.retriever import Retriever
from rag_query.ports.rewriter import QueryRewriter
from rag_query.ports.router import QueryRouter
from rag_query.ports.token_counter import TokenCounter

__all__ = [
    "Authenticator",
    "Generator",
    "HistoryStore",
    "LLMClient",
    "OutputGuard",
    "QueryGuard",
    "QueryRewriter",
    "QueryRouter",
    "RateLimiter",
    "Reranker",
    "Retriever",
    "TokenCounter",
]

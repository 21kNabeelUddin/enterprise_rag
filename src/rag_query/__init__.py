"""Enterprise RAG query pipeline (question -> answer).

Pipeline 2 of the system. Steps 1-7 are implemented so far:

    1. Input            -> :mod:`rag_query.nodes.input`
    2. Authenticate     -> :mod:`rag_query.nodes.authenticate`
    3. Input guardrail  -> :mod:`rag_query.nodes.input_guard`
    4. Router           -> :mod:`rag_query.nodes.route`
    5. Query rewriter   -> :mod:`rag_query.nodes.rewrite`
    (6. HyDE            -> intentionally not implemented)
    7. Encoder          -> :mod:`rag_query.nodes.encode`

The pipeline is a LangGraph state machine whose nodes depend only on ports, so
every step is unit-testable without a GPU, an LLM or a database.
"""

from __future__ import annotations

__version__ = "0.1.0"

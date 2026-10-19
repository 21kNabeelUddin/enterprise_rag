"""Nodes: one file per pipeline step.

Each node is a small class with a ``run`` method and a wrapping ``__call__``
that records timing and the execution trace. Nodes depend only on ports, so a
step can be swapped or faked without touching its neighbours.
"""

from __future__ import annotations

from rag_query.nodes.authenticate import AuthenticateNode
from rag_query.nodes.base import Node
from rag_query.nodes.context import ContextNode
from rag_query.nodes.encode import EncodeNode
from rag_query.nodes.fallback import FallbackNode
from rag_query.nodes.finalize import FinalizeNode
from rag_query.nodes.format import FormatAnswerNode
from rag_query.nodes.generate import GenerateNode
from rag_query.nodes.history import HistoryNode
from rag_query.nodes.input import InputNode
from rag_query.nodes.input_guard import InputGuardNode
from rag_query.nodes.output_guard import OutputGuardNode
from rag_query.nodes.rerank import RerankNode
from rag_query.nodes.retrieve import RetrieveNode
from rag_query.nodes.rewrite import RewriteNode
from rag_query.nodes.route import RouteNode

__all__ = [
    "AuthenticateNode",
    "ContextNode",
    "EncodeNode",
    "FallbackNode",
    "FinalizeNode",
    "FormatAnswerNode",
    "GenerateNode",
    "HistoryNode",
    "InputGuardNode",
    "InputNode",
    "Node",
    "OutputGuardNode",
    "RerankNode",
    "RetrieveNode",
    "RewriteNode",
    "RouteNode",
]

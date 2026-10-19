"""Typed errors raised across the query pipeline.

Nodes translate these into a terminal :class:`~rag_query.domain.models.QueryStatus`
instead of letting exceptions escape the graph, so a single bad request never
takes the process down.
"""

from __future__ import annotations


class QueryPipelineError(Exception):
    """Base class for recoverable query-pipeline failures."""


class AuthenticationError(QueryPipelineError):
    """The caller presented a token that could not be resolved to a principal."""


class FingerprintMismatchError(QueryPipelineError):
    """The query encoder does not match the vectors already stored in the index."""

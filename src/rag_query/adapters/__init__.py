"""Concrete adapters behind the query ports.

This is the only layer that knows about specific technologies (transformers,
httpx, SQLAlchemy). Everything above it sees ports.
"""

from __future__ import annotations

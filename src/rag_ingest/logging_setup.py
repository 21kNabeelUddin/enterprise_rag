"""Central logging configuration.

Uses ``rich`` when available for readable console output, otherwise plain
stdlib logging. Call :func:`configure_logging` once at process start.
"""

from __future__ import annotations

import logging

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler: logging.Handler
    try:
        from rich.console import Console
        from rich.logging import RichHandler

        # Logs go to stderr so stdout stays clean for machine-readable output
        # (e.g. ``rag-ask ask --json``).
        handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        fmt = "%(message)s"
    except Exception:  # pragma: no cover - rich is a declared dependency
        handler = logging.StreamHandler()  # already defaults to stderr
        fmt = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"

    logging.basicConfig(
        level=level.upper(),
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[handler],
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

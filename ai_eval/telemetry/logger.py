"""Structured logging to stderr.

Verbosity:
  -q / --quiet    → WARNING and above
  default         → INFO   (human-friendly status to stderr; data still on stdout)
  -v / --verbose  → DEBUG  (single-line JSON events)
  -vv             → DEBUG with extra fields
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

_LOGGER_NAME = "ai_eval"


class _JsonFormatter(logging.Formatter):
    """Single-line JSON formatter used at -v / -vv."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        for key in ("task", "metric", "model", "duration_ms"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


class _PlainFormatter(logging.Formatter):
    """Bare human formatter used at default verbosity."""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


def configure(*, verbosity: int, quiet: bool) -> logging.Logger:
    """Configure the root ai_eval logger and return it."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False

    if quiet:
        level = logging.WARNING
    elif verbosity >= 1:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_JsonFormatter() if verbosity >= 1 else _PlainFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger() -> logging.Logger:
    """Return the configured logger (or a default one if `configure` wasn't called)."""
    return logging.getLogger(_LOGGER_NAME)


__all__ = ["configure", "get_logger"]

"""Build the async instructor client once and cache it.

Mirrors the lazy-import + error-hint pattern of
``ai_eval.inference.slm.client.py:44-49``: litellm/instructor are optional at
import time of this module's *callers* but hard-required at first use. A failure
here raises :class:`JudgeUnavailable` with a ``fix`` hint instead of bubbling a
raw ``ImportError`` so the CLI can render a remediation message.
"""

from __future__ import annotations

from typing import Any


class JudgeUnavailable(Exception):
    """Hard failure of the judge gateway (provider/deps unavailable)."""

    def __init__(self, message: str, *, why: str = "", fix: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.why = why
        self.fix = fix


#: Process-cached async instructor client (built on first use).
_async_client: Any = None


def get_async_client() -> Any:
    """Return the cached async instructor client, building it on first call.

    Raises :class:`JudgeUnavailable` if ``litellm``/``instructor`` are missing.
    """
    global _async_client
    if _async_client is not None:
        return _async_client
    try:
        import instructor
        import litellm
    except ImportError as exc:
        raise JudgeUnavailable(
            "the judge gateway requires `litellm` and `instructor`",
            why="dependencies are not installed",
            fix="pip install litellm instructor (or `ai-evals`)",
        ) from exc
    # Silence litellm's verbose provider banners: it prints "Provider List"
    # links to stdout on first provider lookup, which corrupts the JSON data
    # channel of `doctor`/`judge --ping` when a model is unreachable.
    try:
        litellm.set_verbose = False
        litellm.suppress_debug_messages = True
        import logging as _logging

        for _n in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"):
            _logging.getLogger(_n).setLevel(_logging.WARNING)
    except Exception:
        pass
    _async_client = instructor.from_litellm(litellm.acompletion)
    return _async_client


def reset_client() -> None:
    """Drop the cached client (tests that patch litellm call this)."""
    global _async_client
    _async_client = None


__all__ = ["JudgeUnavailable", "get_async_client", "reset_client"]

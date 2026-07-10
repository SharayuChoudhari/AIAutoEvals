"""Progress indicators that auto-silence when stderr is not a TTY."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console

# Shared stderr console used for status output. Access via `stderr_console()`
# so callers always get the same instance.
_stderr_console = Console(stderr=True, soft_wrap=True)


def stderr_console() -> Console:
    """Return the shared stderr Rich Console."""
    return _stderr_console


@contextmanager
def status(message: str, *, enabled: bool = True) -> Iterator[None]:
    """Show a rich spinner when interactive; otherwise no-op.

    Always safe to call. When `enabled=False` (quiet/non-TTY/JSON) yields silently.
    """
    if not enabled or not sys.stderr.isatty():
        yield
        return
    with stderr_console().status(f"[dim]{message}[/dim]", spinner="dots"):
        yield


__all__ = ["status", "stderr_console"]

"""Progress indicators that auto-silence when stderr is not a TTY."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console

# Shared stderr console used for status output. `force_terminal=None` lets rich
# detect TTY; `no_color` is applied later via theme.
_stderr_console = Console(stderr=True, soft_wrap=True)


def stderr_console() -> Console:
    return _stderr_console


@contextmanager
def status(message: str, *, enabled: bool = True) -> Iterator[None]:
    """Show a rich spinner when interactive; otherwise no-op.

    Always safe to call. When `enabled=False` (quiet/non-TTY/JSON) yields silently.
    """
    if not enabled or not sys.stderr.isatty():
        yield
        return
    with _stderr_console.status(f"[dim]{message}[/dim]", spinner="dots"):
        yield


__all__ = ["status", "stderr_console"]

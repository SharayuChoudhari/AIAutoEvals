"""Advisory file lock around history/run writes.

``fcntl``-based on POSIX; a no-op context manager on non-POSIX (Windows).
Best-effort: if locking fails, the write proceeds unlocked (the history file
is append-friendly and worst-case is a duplicate entry).
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path


@contextlib.contextmanager
def advisory_lock(lock_path: Path) -> Iterator[None]:
    """Acquire an exclusive advisory lock on ``lock_path``.

    On POSIX uses ``fcntl.flock``; on non-POSIX yields without locking. Never
    raises on a lock failure — callers still write (best-effort).
    """
    if not sys.platform.startswith("linux") and sys.platform != "darwin":
        yield
        return
    try:
        import fcntl
    except ImportError:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except OSError:
            # best-effort: proceed unlocked
            pass
        yield
    finally:
        try:
            import fcntl as _f

            _f.flock(fh.fileno(), _f.LOCK_UN)
        except Exception:
            pass
        fh.close()


__all__ = ["advisory_lock"]

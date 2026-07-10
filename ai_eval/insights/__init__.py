"""Insights package: history, diff, root-cause narrative."""

from __future__ import annotations

from .diff import compute_delta
from .history import append_history, export, list_history, load_run, prune
from .root_cause import explain

__all__ = [
    "append_history",
    "compute_delta",
    "explain",
    "export",
    "list_history",
    "load_run",
    "prune",
]

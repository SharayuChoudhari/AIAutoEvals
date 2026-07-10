"""Automated golden-set bootstrapper (Phase 3).

Captures real runtime traces from a user-specified command (e.g. ``pytest``)
into ``eval/golden_set.json`` by monkey-patching the LLM/retrieval frameworks
in a child subprocess via a generated ``sitecustomize.py``.

The parent process:
1. writes a temp ``sitecustomize.py`` that wires the wrappers to a JSONL sink,
2. runs the user command with an augmented ``PYTHONPATH``,
3. reads the sink, dedups, caps, and merges into the golden set.

One golden example per **top-level task invocation**; nested LLM/retrieve calls
fold into ``trace.calls`` (correlated via a thread-local task stack in
:mod:`ai_eval.bootstrap.tracer`).
"""

from __future__ import annotations

from .golden_writer import append
from .tracer import enter_task, exit_task, redact
from .wrappers import install_hooks

__all__ = ["append", "enter_task", "exit_task", "install_hooks", "redact"]

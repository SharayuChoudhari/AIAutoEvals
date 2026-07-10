"""Run execution engine (Phase 4).

Async-native. For each task (filtered), for each golden example:
1. import the task module (inserting ``cwd`` + ``file_path.parent`` on
   ``sys.path``), call the entry symbol with the example's input — measure
   latency. Import/call errors → ``task.errors``, example marked ``error``
   (never a process crash).
2. for each metric in the task: non-judge metrics (``latency_*``) compute
   locally; judge metrics ``await gateway.score(...)`` via the metric's prompt
   builder, parallelized by an ``asyncio`` semaphore.
3. aggregate per-metric scores (mean across examples) and write per-example
   traces to ``traces.ndjson``.

The engine is transport-agnostic: the gateway is injected so tests pass a fake.
"""

from __future__ import annotations

from .engine import execute
from .record import (
    ExampleRecord,
    GitMeta,
    MetricResult,
    RunRecord,
    RunSummary,
    TaskRecord,
)

__all__ = [
    "ExampleRecord",
    "GitMeta",
    "MetricResult",
    "RunRecord",
    "RunSummary",
    "TaskRecord",
    "execute",
]

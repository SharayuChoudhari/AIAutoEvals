"""Pydantic run-record schema (``schema_version: 1``).

Matches design §3.4: ``id, started_at, finished_at, git{sha,branch,dirty},
config_hash, baseline_id, tasks{<name>{examples, metrics{<m>{score,delta,
threshold,status}}, latency_ms{p50,p95}}}, summary{passed,failed,errors},
tags``. Per-task + top-level ``errors`` added for graceful failure surfacing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_eval.config.defaults import SCHEMA_VERSION

MetricStatus = Literal["pass", "fail", "skip", "error"]


class GitMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha: str | None = None
    branch: str | None = None
    dirty: bool | None = None


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    score: float | None = None
    delta: float | None = None
    threshold: float = 0.0
    status: MetricStatus = "skip"
    weight: float = 1.0
    errors: list[str] = Field(default_factory=list)


class ExampleRecord(BaseModel):
    """One golden example's outcome within a task."""

    model_config = ConfigDict(extra="forbid")

    id: str
    status: Literal["pass", "fail", "error"] = "pass"
    score: float | None = None
    latency_ms: float = 0.0
    error: str | None = None
    metric_scores: dict[str, float] = Field(default_factory=dict)
    metric_errors: dict[str, list[str]] = Field(default_factory=dict)
    #: ``"auto"`` marks an auto-seeded example (D6) so the renderer can emit a
    #: one-line notice ("scored against auto-seeded fixtures; run bootstrap for
    #: trustworthy baselines"). ``None`` for real captured/user examples.
    seed: str | None = None
    #: Per-node scores keyed by synthetic ``node_id`` (e.g. ``retrieve_0``)
    #: → ``{metric_name: score}``. Populated only when the task declares
    #: ``node_metrics`` and the example's ``trace.calls`` expose matching nodes
    #: (AGENTS.md §1). Backward-compatible: old ``run.json`` files with no
    #: ``node_scores`` key load unchanged.
    node_scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examples: list[ExampleRecord] = Field(default_factory=list)
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: int = 0
    failed: int = 0
    errors: int = 0
    examples: int = 0


class RunRecord(BaseModel):
    """Top-level run record written to ``.ai-evals/runs/<id>/run.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    id: str
    started_at: float
    finished_at: float | None = None
    git: GitMeta | None = None
    config_hash: str | None = None
    baseline_id: str | None = None
    tasks: dict[str, TaskRecord] = Field(default_factory=dict)
    summary: RunSummary = Field(default_factory=RunSummary)
    tags: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    # opaque extras for debugging (judge tier used, cache stats, etc.)
    extras: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ExampleRecord",
    "GitMeta",
    "MetricResult",
    "MetricStatus",
    "RunRecord",
    "RunSummary",
    "TaskRecord",
]

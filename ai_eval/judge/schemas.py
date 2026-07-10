"""Pydantic schemas for the judge gateway.

``JudgeRequest`` is the input contract handed to :func:`score`; the gateway
serializes the metric's prompt-builder output into ``messages`` before the
underlying transport runs. ``JudgeResponse`` is the validated instructor
response model — the judge returns a single ``score`` plus an optional
``rationale`` and per-dimension ``sub_scores`` (the metric's
``scored_dimension`` selects which sub-score becomes the canonical metric
value at aggregation time).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JudgeRequest(BaseModel):
    """One judge scoring call. The gateway fills ``messages`` from the metric."""

    model_config = ConfigDict(extra="forbid")

    task_name: str
    task_type: str
    metric: str
    example: dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    tier_hint: str | None = None


class JudgeResponse(BaseModel):
    """Validated judge output. ``score`` is in ``[0, 1]``."""

    model_config = ConfigDict(extra="allow")

    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    sub_scores: dict[str, float] = Field(default_factory=dict)
    #: The raw provider payload (truncated) for debugging / `--verbose`.
    raw: Any = None


class JudgeError(BaseModel):
    """A captured failure for the run record (never raised directly)."""

    model_config = ConfigDict(extra="forbid")

    model: str
    message: str
    retryable: bool = True


__all__ = ["JudgeError", "JudgeRequest", "JudgeResponse"]

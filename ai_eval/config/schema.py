"""Pydantic schema for `rubrics.yaml` and related on-disk structures.

Every on-disk JSON/YAML embeds `schema_version: 1`. Bumps require migration in
`config/loader.py`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ProjectType = Literal[
    "rag",
    "tools",
    "tool_calling",
    "agent",
    "chat",
    "rag_and_tools",
    "custom",
]

TaskType = Literal["tool_calling", "rag", "agent", "chat"]


class JudgeConfig(BaseModel):
    """Configured judge models. `default` is required for `run`."""

    model_config = ConfigDict(extra="forbid")

    default: str = Field(..., description="Primary judge LiteLLM model id.")
    regression_check: str | None = Field(
        default=None,
        description="Optional regression judge for `diff --explain`.",
    )
    fallback: list[str] = Field(
        default_factory=list,
        description="Ordered fallbacks if `default` is unreachable.",
    )


class DefaultsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parallel: int = Field(default=4, ge=1, le=64)
    cache: bool = True
    tolerance: float = Field(default=0.02, ge=0.0, le=1.0)


class MetricSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0)
    params: dict[str, Any] = Field(default_factory=dict)


class TaskJudgeOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str | None = None
    regression_check: str | None = None


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    entry: str | None = None
    type: TaskType
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    metrics: list[MetricSpec] = Field(default_factory=list)
    judge_overrides: TaskJudgeOverrides | None = None
    redact: list[str] = Field(default_factory=list)


class RubricsConfig(BaseModel):
    """Top-level schema for `eval/rubrics.yaml`."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1)
    project_type: ProjectType = "custom"
    judge: JudgeConfig
    defaults: DefaultsBlock = Field(default_factory=DefaultsBlock)
    tasks: dict[str, TaskSpec] = Field(default_factory=dict)

    @field_validator("tasks")
    @classmethod
    def _validate_task_names(cls, v: dict[str, TaskSpec]) -> dict[str, TaskSpec]:
        for name in v:
            if not name or not name.replace("_", "").isalnum():
                raise ValueError(
                    f"task name {name!r} must be snake_case alphanumeric"
                )
        return v


__all__ = [
    "DefaultsBlock",
    "JudgeConfig",
    "MetricSpec",
    "ProjectType",
    "RubricsConfig",
    "TaskJudgeOverrides",
    "TaskSpec",
    "TaskType",
]

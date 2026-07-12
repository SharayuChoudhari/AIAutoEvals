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
    "workflow",
    "custom",
]

#: Open per-task type string. The SLM may emit any value from the suggested
#: vocabulary (chat, rag, agent, tool_calling, scoring, extraction,
#: classification, summarization, translation, booking, workflow, other) or a
#: custom one. Existing rule detectors emit the same string values they always
#: did, so this stays backward compatible.
TaskType = str


class JudgeTiering(BaseModel):
    """Optional override of the judge prompt-tier model set.

    When absent, :func:`~ai_eval.judge.tiering.select_tier` uses the built-in
    :data:`~ai_eval.config.defaults.COMPLEX_MODEL_HINTS`. An empty
    ``complex_models`` forces every model to the basic checklist tier.
    """

    model_config = ConfigDict(extra="forbid")

    complex_models: list[str] = Field(default_factory=list)


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
    tiering: JudgeTiering | None = Field(
        default=None,
        description="Optional override of the complex-model tier hint set.",
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

    @field_validator("name")
    @classmethod
    def _validate_registered(cls, v: str) -> str:
        # Imported lazily to avoid a config -> metrics import cycle at module
        # import time (schema is imported very early).
        from ai_eval.metrics.registry import is_registered, is_strict, warn_unknown

        if not is_registered(v):
            if is_strict():
                raise ValueError(
                    f"metric {v!r} is not registered; add it via the "
                    f"'ai_eval.metrics' entry-point group or pick from the "
                    f"built-in set"
                )
            warn_unknown(v)
        return v


class TaskJudgeOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default: str | None = None
    regression_check: str | None = None


class HintTaskSpec(BaseModel):
    """One entry in ``eval/ai-evals.hints.yaml``.

    The opt-in hints file is the escape hatch for orchestration the AST
    detectors can't see (Temporal, Prefect, Airflow, proprietary state loops)
    and the lever to split a single detected entry point into per-intent
    tasks. Hints are additive: when a hint and an AST task share the same
    ``(file_path, entry)`` the AST task wins (hints fill gaps, they don't
    override successful detection).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    file_path: str
    entry: str | None = None
    type: TaskType
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    #: When ``true`` the task is immune to call-graph demotion and
    #: judge-exclusion (plan D1 escape hatch: ``force_task: true`` to
    #: force-include a site the heuristics would drop).
    force_task: bool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.replace("_", "").isalnum():
            raise ValueError(
                f"hint task name {v!r} must be snake_case alphanumeric"
            )
        return v


class HintsFile(BaseModel):
    """Top-level schema for ``eval/ai-evals.hints.yaml``.

    Absent or empty file means "no hints". Each hint becomes a
    :class:`~ai_eval.inference.detectors.base.DetectedTask` with
    ``framework="hint"`` before the rubric engine runs.
    """

    model_config = ConfigDict(extra="forbid")

    tasks: list[HintTaskSpec] = Field(default_factory=list)
    #: Repo-relative path globs whose detected sites are force-excluded as
    #: judge code (plan D1 escape hatch: ``judge_code: [paths]``). Matches
    #: against ``DetectedTask.file_path`` (POSIX glob).
    judge_code: list[str] = Field(default_factory=list)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str
    entry: str | None = None
    type: TaskType
    purpose: str | None = None
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
    #: Which engine produced this file: "rules", "slm", or "hybrid". Absent for
    #: files written before this field existed.
    rubric_engine: str | None = None

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
    "HintTaskSpec",
    "HintsFile",
    "JudgeConfig",
    "JudgeTiering",
    "MetricSpec",
    "ProjectType",
    "RubricsConfig",
    "TaskJudgeOverrides",
    "TaskSpec",
    "TaskType",
]

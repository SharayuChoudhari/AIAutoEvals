"""Turn a flat list of `DetectedTask`s into a `RubricsConfig` ready for disk."""

from __future__ import annotations

import re
from pathlib import Path

from ai_eval.config.defaults import (
    DEFAULT_JUDGE,
    DEFAULT_REGRESSION_JUDGE,
    SCHEMA_VERSION,
)
from ai_eval.config.schema import (
    DefaultsBlock,
    JudgeConfig,
    MetricSpec,
    ProjectType,
    RubricsConfig,
    TaskSpec,
)
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.task_selection import select_tasks

# Default metric set per task type. Matches design §2.6 (Phase 4 metrics).
_DEFAULT_METRICS: dict[str, list[MetricSpec]] = {
    "tool_calling": [
        MetricSpec(name="argument_accuracy", threshold=0.9, weight=1.0),
        MetricSpec(name="hallucination_rate", threshold=0.1, weight=1.0),
    ],
    "rag": [
        MetricSpec(name="context_precision", threshold=0.85, weight=1.0),
        MetricSpec(name="faithfulness", threshold=0.9, weight=1.0),
    ],
    "agent": [
        MetricSpec(name="argument_accuracy", threshold=0.85, weight=1.0),
        MetricSpec(name="hallucination_rate", threshold=0.1, weight=1.0),
    ],
    "chat": [
        MetricSpec(name="hallucination_rate", threshold=0.1, weight=1.0),
    ],
    # task_completion is the only built-in metric native to `workflow`.
    # slot_filling_accuracy was removed from built-ins (it's now a project-local
    # metric in eval/metrics.yaml — see docs/metrics.md). argument_accuracy /
    # hallucination_rate are NOT auto-assigned here — they list tool_calling/
    # agent, not workflow. Users add niche/per-task metrics when wanted.
    "workflow": [
        MetricSpec(name="task_completion", threshold=0.9, weight=1.0),
    ],
}


def _looks_like_rag(task: DetectedTask) -> bool:
    """Defensive RAG check: an explicitly-tagged rag task, or a task whose
    inputs/outputs resemble retrieval (``query`` in inputs, ``documents`` in
    outputs) even when its ``type`` fell through to ``chat`` for an
    undetected framework. Prevents silent downgrade to ``project_type: chat``.
    """
    if task.type == "rag":
        return True
    if "documents" in task.outputs and "query" in task.inputs:
        return True
    return False


def _classify_project_type(tasks: list[DetectedTask]) -> ProjectType:
    if not tasks:
        return "custom"
    rag_count = sum(1 for t in tasks if _looks_like_rag(t))
    agent_count = sum(1 for t in tasks if t.type == "agent")
    tool_count = sum(1 for t in tasks if t.type == "tool_calling")
    workflow_count = sum(1 for t in tasks if t.type == "workflow")
    has_rag = rag_count > 0
    # Workflow tasks are tool-bearing (they drive tools through orchestration),
    # so they join tool_calling/agent in the has_tools signal.
    has_tools = tool_count > 0 or agent_count > 0 or workflow_count > 0
    if has_rag and has_tools:
        return "rag_and_tools"
    if has_tools:
        # An all-workflow repo classifies to the dedicated `workflow` literal
        # rather than collapsing into tool_calling/agent, which would mislabel
        # orchestration-heavy repos that happen to call tools. When workflow
        # tasks coexist with plain tool_calling/agent tasks, fall back to the
        # dominant tool/agent branch so mixed repos keep their existing label.
        non_workflow_tools = tool_count + agent_count
        if non_workflow_tools == 0:
            return "workflow"
        return "tool_calling" if tool_count >= agent_count else "agent"
    if has_rag:
        return "rag"
    return "chat"


def _unique_name(name: str, used: set[str]) -> str:
    if name not in used:
        return name
    counter = 2
    while f"{name}_{counter}" in used:
        counter += 1
    return f"{name}_{counter}"


_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _is_private_entry(entry: str | None) -> bool:
    """True when the entry is a private (``_``-prefixed) method.

    ``Class._method`` → True (private). ``Class.method`` → False (public).
    Bare module-level functions (no ``.``) → False. Private methods are
    internal implementation details, not the complete job the user cares
    about, so they're marked ``top_level=False`` (AGENTS.md §1).
    """
    if entry and "." in entry:
        method = entry.rpartition(".")[2]
        return method.startswith("_")
    return False


def _camel_to_snake(name: str) -> str:
    """``ChatMessageService`` → ``chat_message_service``.

    Inserts an underscore before each uppercase letter (except at the start)
    and lowercases. Handles consecutive capitals by only breaking before a
    capital that follows a lowercase/digit boundary — good enough for
    Python class names (``OpenAI`` → ``open_ai``, ``HTTPClient`` →
    ``http_client``).
    """
    return _CAMEL_BOUNDARY.sub("_", name).lower()


def _collapse_dotted_name(name: str, entry: str | None) -> str:
    """Collapse a dotted ``Class.method`` name to a snake_case-alphanumeric key.

    ``Class.method`` → ``class_method``: CamelCase class names split to
    snake_case and the leading underscore of private methods is dropped so the
    key never begins with ``_``. Bare module-level names (no ``.`` in
    ``entry``) pass through unchanged. The dotted resolvable form stays in
    ``TaskSpec.entry``; only the rubrics dict key is collapsed.
    """
    if entry and "." in entry:
        # ``ConversationWorkflowService._call_model`` →
        # ``conversation_workflow_service_call_model``.
        cls, _, method = entry.rpartition(".")
        cls_part = _camel_to_snake(cls.replace(".", "_"))
        method_part = method.lstrip("_")
        name = f"{cls_part}_{method_part}" if method_part else cls_part
    return name


def _rubric_key_name(task: DetectedTask) -> str:
    """Derive a snake_case-alphanumeric rubrics key from a detected task.

    ``DetectedTask.name`` may be a dotted ``Class.method`` entry (post
    class-body descent) which is correct for ``TaskSpec.entry`` but illegal as
    a rubrics dict key (validated snake_case alphanumeric by
    ``RubricsConfig``). Collapse ``Class.method`` → ``class_method`` so the key
    is schema-valid while ``entry`` keeps the resolvable dotted form. Bare
    module-level names pass through unchanged.
    """
    return _collapse_dotted_name(task.name, task.entry)


def build_rubrics(
    scan: ScanResult,
    *,
    judge_default: str | None = None,
    judge_regression: str | None = None,
    project_root: Path | None = None,
    judge_code_globs: list[str] | None = None,
    force_task_keys: set[tuple[str, str | None]] | None = None,
) -> RubricsConfig:
    """Return a `RubricsConfig` synthesized from the scan result.

    When ``project_root`` is provided, the scan is first filtered through task
    selection (call-graph demotion + judge-exclusion, plan D1/R2). When it is
    ``None`` (e.g. unit tests constructing a ``ScanResult`` directly with no
    files on disk), tasks are used as-is — backward compatible.
    """
    if project_root is not None:
        scan = select_tasks(
            project_root,
            scan,
            judge_code_globs=judge_code_globs,
            force_task_keys=force_task_keys,
        )
    judge = JudgeConfig(
        default=judge_default or DEFAULT_JUDGE,
        regression_check=judge_regression or DEFAULT_REGRESSION_JUDGE,
    )
    used_names: set[str] = set()
    task_specs: dict[str, TaskSpec] = {}
    for task in scan.tasks:
        name = _unique_name(_rubric_key_name(task), used_names)
        used_names.add(name)
        # If the project-level RAG recovery (``_looks_like_rag``) promotes this
        # task's shape to rag, keep per-task metrics consistent with the
        # promoted project_type rather than leaving a chat-typed metric set.
        task_type = "rag" if _looks_like_rag(task) and task.type != "rag" else task.type
        task_specs[name] = TaskSpec(
            file_path=task.file_path,
            entry=task.entry,
            type=task_type,
            inputs=task.inputs,
            outputs=task.outputs,
            metrics=list(_DEFAULT_METRICS.get(task_type, _DEFAULT_METRICS["chat"])),
            top_level=task.top_level and not _is_private_entry(task.entry),
        )
    return RubricsConfig(
        schema_version=SCHEMA_VERSION,
        project_type=_classify_project_type(scan.tasks),
        judge=judge,
        defaults=DefaultsBlock(),
        tasks=task_specs,
    )


__all__ = ["build_rubrics"]

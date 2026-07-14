"""SLM rubric builder — the hybrid engine's classification stage.

Consumes a :class:`~ai_eval.inference.ast_scan.ScanResult` (the AST evidence,
populated by the rule detectors in ``hybrid`` mode, or empty in ``slm`` mode
with the entry-point fallback) and emits a validated
:class:`~ai_eval.config.schema.RubricsConfig`.

Per-task calls: one SLM call per detected task (capped at ``max_tasks``), each
asking the SLM to classify purpose/type/inputs/outputs/metrics from the
enclosing-function snippet + framework hints. The empty-evidence fallback makes
one repo-level call over entry-point files.

Determinism: ``temperature=0`` + an evidence-hash keyed file cache under
``.ai-evals/rubric_cache/``. Hard-fail on SLM unavailability via
:class:`RubricEngineError` (no silent fallback to rules — use
``--rubric-engine rules`` for that).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

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
from ai_eval.inference.slm.cache import ResponseCache, evidence_hash
from ai_eval.inference.slm.client import CompleteFn
from ai_eval.inference.slm.client import complete as _default_complete
from ai_eval.inference.slm.evidence import (
    build_task_evidence,
    select_entry_point_files,
)
from ai_eval.inference.slm.prompts import render, render_few_shot
from ai_eval.inference.slm.validation import ValidationReport, validate_against_scan
from ai_eval.inference.synthesize import _collapse_dotted_name, _unique_name
from ai_eval.metrics.registry import all_names
from ai_eval.metrics.registry import get as get_metric


class RubricEngineError(Exception):
    """Hard failure of the SLM rubric engine (no silent fallback to rules)."""

    def __init__(self, message: str, *, why: str = "", fix: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.why = why
        self.fix = fix


# ---------------------------------------------------------------------------
# SLM response models (validated by instructor before reaching builder logic)
# ---------------------------------------------------------------------------


class _SLMMetric(BaseModel):
    name: str
    threshold: float | None = None
    weight: float | None = None


class _SLMTask(BaseModel):
    """The per-task shape the SLM returns for the classify_task prompt."""

    purpose: str | None = None
    type: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    metrics: list[_SLMMetric] = Field(default_factory=list)


class _SLMTaskNamed(_SLMTask):
    """The named shape for the empty-recover prompt (SLM proposes names)."""

    name: str
    file_path: str
    entry: str | None = None


class _SLMRecovery(BaseModel):
    tasks: list[_SLMTaskNamed] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost-control caps
# ---------------------------------------------------------------------------


@dataclass
class EvidenceCaps:
    max_snippet_chars: int = 1500
    max_tasks: int = 25
    max_tokens: int = 1024
    budget_tokens: int = 0  # 0 = no hard budget enforcement
    retries: int = 2


# ---------------------------------------------------------------------------
# Build entry points
# ---------------------------------------------------------------------------


@dataclass
class BuildStats:
    slm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tasks_dropped: int = 0
    fallback_used: bool = False


def _is_empty_classification(slm: _SLMTask) -> bool:
    """True when the SLM returned a structurally-valid but content-free
    classification: no inputs, no outputs, no metrics, and no purpose.

    This is the silent-failure signature we saw when the SLM (e.g. an
    unavailable ollama endpoint, or a too-weak model returning an empty
    object) produces a response that validates against the all-optional
    ``_SLMTask`` schema yet carries no usable rubric data. Caching such a
    result would poison every subsequent run against the same evidence
    hash; this guard rejects it so we never persist (or trust) an empty
    classification.
    """
    return not slm.inputs and not slm.outputs and not slm.metrics and not slm.purpose


def _fallback_task_spec(task: DetectedTask) -> TaskSpec:
    """Build a TaskSpec from the detector's own evidence when the SLM
    classification is unusable. Reuses the rule engine's default metric set
    for the detector-classified type so the rubric is never silently empty."""
    from ai_eval.inference.synthesize import _DEFAULT_METRICS, _is_private_entry

    task_type = task.type
    metrics = list(_DEFAULT_METRICS.get(task_type, _DEFAULT_METRICS["chat"]))
    return TaskSpec(
        file_path=task.file_path,
        entry=task.entry,
        type=task_type,
        purpose=None,
        inputs=list(task.inputs),
        outputs=list(task.outputs),
        metrics=metrics,
        top_level=task.top_level and not _is_private_entry(task.entry),
    )


def _metric_specs_from_slm(
    slm_metrics: list[_SLMMetric], project_root: Path | None = None
) -> list[MetricSpec]:
    """Convert SLM-proposed metrics into validated MetricSpec, applying
    registry defaults for threshold/weight when the SLM omits them.

    ``project_root`` resolves project-local ``eval/metrics.yaml`` metrics so
    the SLM can propose (and get defaults for) custom metrics the project
    declares."""
    out: list[MetricSpec] = []
    for m in slm_metrics:
        reg = get_metric(m.name, project_root=project_root)
        threshold = (
            m.threshold if m.threshold is not None else (reg.default_threshold if reg else 0.0)
        )
        weight = m.weight if m.weight is not None else (reg.default_weight if reg else 1.0)
        out.append(MetricSpec(name=m.name, threshold=threshold, weight=weight))
    return out


#: Task types that are NOT RAG but that a weak SLM may still emit for a
#: genuine RAG call site. ``chat`` is the common silent-downgrade failure
#: mode; ``other`` is the model punting on classification.
_NON_RAG_DOWNGRADES = {"chat", "other", "summarization", "translation"}


def _looks_like_rag_task(slm: _SLMTask) -> bool:
    """True when the SLM's own response shape indicates RAG, independent of
    the ``type`` it chose. Catches the case where a weak SLM returns the right
    IO (``query`` in inputs, an answer-like output) but a conservative
    ``type: chat``."""
    if "documents" in slm.outputs:
        return True
    return "query" in slm.inputs and any(o in _RAG_OUTPUT_HINTS for o in slm.outputs)


def _apply_rag_downgrade_guard(slm: _SLMTask, task: DetectedTask) -> None:
    """Prevent a weak SLM from silently downgrading a detector-confirmed RAG
    task (or one whose own IO clearly resembles RAG) to ``chat`` / ``other``.

    Mutates ``slm`` in place: when the guard fires it sets ``type = "rag"`` and,
    if the SLM's metrics are not RAG-appropriate, replaces them with the
    registry default RAG set. The SLM's ``purpose`` / ``inputs`` / ``outputs``
    are preserved so its richer description survives.

    Why this is needed: in ``slm`` and ``hybrid`` modes the SLM owns the final
    type, but a small local model frequently returns ``type: chat`` even for a
    ``RetrievalQA`` / ``vectorstore.similarity_search`` call site because the
    snippet is short. Without this guard that flips both the task ``type`` and
    the repo-level ``project_type`` to ``chat`` — the exact regression users
    hit. The ``rules`` engine already has the analogous ``_looks_like_rag``
    recovery in ``synthesize.py``; this mirrors it for the SLM path.
    """
    if slm.type == "rag":
        return
    detector_confirmed_rag = task.type == "rag"
    # Without detector confirmation, require the strong ``documents`` output
    # signal — ``query`` + ``answer`` alone is also a plain chat Q&A shape and
    # would over-promote genuine chat/extraction tasks. The broad answer-like
    # hints are only trusted when the detector already flagged retrieval.
    if detector_confirmed_rag:
        io_resembles_rag = _looks_like_rag_task(slm)
    else:
        io_resembles_rag = "documents" in slm.outputs and "query" in slm.inputs
    if not detector_confirmed_rag and not io_resembles_rag:
        return
    if slm.type not in _NON_RAG_DOWNGRADES:
        # The SLM chose a specific non-RAG type (e.g. extraction, booking). Only
        # override the silent "chat/other" punt, not a deliberate classification.
        return
    slm.type = "rag"
    # Ensure at least one RAG metric is present; if the SLM's metrics are all
    # non-RAG (e.g. only hallucination_rate), swap in the default RAG set so the
    # rubric carries context_precision + faithfulness.
    rag_metric_names = {"context_precision", "faithfulness"}
    if not any(m.name in rag_metric_names for m in slm.metrics):
        slm.metrics = [
            _SLMMetric(name="context_precision"),
            _SLMMetric(name="faithfulness"),
        ]


def _apply_hint_type_pin(slm: _SLMTask, task: DetectedTask) -> None:
    """Pin a hint task's declared ``type`` so the SLM can't override it.

    Per plan Open Question 2: hints are explicit user declarations, so in
    ``hybrid``/``slm`` mode the hint's ``type`` is authoritative — the SLM
    still refines purpose/inputs/outputs/metrics, but it must not reclassify a
    hint-flagged task (``framework == "hint"``) to a different type. Without
    this pin a weak SLM could silently downgrade a declared ``workflow`` hint
    to ``chat``/``other``.

    Only the ``type`` is pinned; the SLM's richer purpose/IO/metrics survive so
    the hint's declaration shapes the rubric without discarding the SLM's
    refinement of the descriptive fields.
    """
    if task.framework == "hint" and slm.type != task.type:
        slm.type = task.type


def _task_spec_from_slm(
    slm: _SLMTask,
    *,
    file_path: str,
    entry: str | None,
    project_root: Path | None = None,
) -> TaskSpec:
    from ai_eval.inference.synthesize import _is_private_entry

    return TaskSpec(
        file_path=file_path,
        entry=entry,
        type=slm.type,
        purpose=slm.purpose,
        inputs=list(slm.inputs),
        outputs=list(slm.outputs),
        metrics=_metric_specs_from_slm(slm.metrics, project_root=project_root),
        top_level=not _is_private_entry(entry),
    )


def _classify_project_type_slm(tasks: list[TaskSpec]) -> ProjectType:
    """Project-level summary reusing the synthesize decision tree, adapted to
    the open ``type: str`` surface."""
    if not tasks:
        return "custom"
    rag_count = sum(1 for t in tasks if t.type == "rag" or _io_looks_rag(t))
    agent_count = sum(1 for t in tasks if t.type == "agent")
    tool_count = sum(1 for t in tasks if t.type in {"tool_calling", "booking"})
    workflow_count = sum(1 for t in tasks if t.type == "workflow")
    has_rag = rag_count > 0
    has_tools = tool_count > 0 or agent_count > 0 or workflow_count > 0
    if has_rag and has_tools:
        return "rag_and_tools"
    if has_tools:
        # All-workflow repo → dedicated `workflow` literal; mixed repos keep
        # the dominant tool/agent label. Mirrors synthesize._classify_project_type.
        non_workflow_tools = tool_count + agent_count
        if non_workflow_tools == 0:
            return "workflow"
        return "tool_calling" if tool_count >= agent_count else "agent"
    if has_rag:
        return "rag"
    return "chat"


#: Output field names that, alongside a ``query`` input, indicate a RAG task
#: even when the SLM typed it ``chat``. Tenant-RAG pipelines frequently emit
#: ``answer`` (not ``documents``) as the final output, so the narrow
#: ``documents``-only check silently downgraded them to ``chat``.
_RAG_OUTPUT_HINTS = {"documents", "answer", "response", "result"}


def _io_looks_rag(task: TaskSpec) -> bool:
    if "documents" in task.outputs:
        return True
    return "query" in task.inputs and any(o in _RAG_OUTPUT_HINTS for o in task.outputs)


def _messages(system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Classify the task described above."},
    ]


def build_rubrics_slm(
    scan: ScanResult,
    *,
    project_root: Path,
    model: str = DEFAULT_JUDGE,
    judge_default: str | None = None,
    judge_regression: str | None = None,
    caps: EvidenceCaps | None = None,
    complete_fn: CompleteFn | None = None,
    cache: ResponseCache | None = None,
    rule_hints: dict[str, str] | None = None,
) -> tuple[RubricsConfig, BuildStats, ValidationReport]:
    """Build a ``RubricsConfig`` from the scan using the SLM.

    ``complete_fn`` is injectable so tests avoid network/ollama. When None the
    real :func:`~ai_eval.inference.slm.client.complete` is used.

    ``rule_hints`` (hybrid mode) maps a task name to the rule engine's
    classified type, surfaced as prior evidence in the SLM prompt — the SLM
    still owns the final type, but sees the rule's opinion. Absent in ``slm``
    mode.

    Returns the rubrics, build stats, and the AST validation report.
    """
    caps = caps or EvidenceCaps()
    do_complete: CompleteFn = complete_fn if complete_fn is not None else _default_complete
    cache = cache if cache is not None else ResponseCache(project_root)
    stats = BuildStats()
    metric_names = sorted(all_names(project_root))
    few_shot = render_few_shot()
    used_names: set[str] = set()
    task_specs: dict[str, TaskSpec] = {}
    rule_hints = rule_hints or {}

    # Approximate token budget enforcement (decision #8). 0 means no cap. We
    # approximate tokens as chars/4 and stop issuing further per-task prompts
    # once the accumulated prompt size is exceeded — bounds total SLM cost on
    # large repos. Cache hits cost nothing (no live call).
    budget_chars = caps.budget_tokens * 4 if caps.budget_tokens > 0 else 0
    spent_chars = 0

    if scan.tasks:
        for task in scan.tasks[: caps.max_tasks]:
            if budget_chars and spent_chars >= budget_chars:
                break
            spec, prompt_chars = _classify_one_task(
                task,
                project_root=project_root,
                model=model,
                caps=caps,
                do_complete=do_complete,
                cache=cache,
                metric_names=metric_names,
                few_shot=few_shot,
                stats=stats,
                rule_hint=rule_hints.get(task.name),
            )
            if spec is None:
                continue
            name = _unique_name(_collapse_dotted_name(task.name, task.entry), used_names)
            used_names.add(name)
            task_specs[name] = spec
            spent_chars += prompt_chars
    else:
        stats.fallback_used = True
        recovered = _recover_empty(
            project_root=project_root,
            model=model,
            caps=caps,
            do_complete=do_complete,
            cache=cache,
            metric_names=metric_names,
            few_shot=few_shot,
            stats=stats,
        )
        for named in recovered:
            name = _unique_name(_collapse_dotted_name(named.name, named.entry), used_names)
            used_names.add(name)
            task_specs[name] = _task_spec_from_slm(
                named,
                file_path=named.file_path,
                entry=named.entry,
                project_root=project_root,
            )

    rubrics = RubricsConfig(
        schema_version=SCHEMA_VERSION,
        project_type=_classify_project_type_slm(list(task_specs.values())),
        judge=JudgeConfig(
            default=judge_default or DEFAULT_JUDGE,
            regression_check=judge_regression or DEFAULT_REGRESSION_JUDGE,
        ),
        defaults=DefaultsBlock(),
        tasks=task_specs,
        rubric_engine="slm",
    )

    rubrics, report = validate_against_scan(rubrics, scan, project_root=project_root)
    stats.tasks_dropped = len(report.dropped_tasks)
    stats.cache_hits = cache.stats.hits
    stats.cache_misses = cache.stats.misses
    return rubrics, stats, report


def _classify_one_task(
    task: DetectedTask,
    *,
    project_root: Path,
    model: str,
    caps: EvidenceCaps,
    do_complete: CompleteFn,
    cache: ResponseCache,
    metric_names: list[str],
    few_shot: str,
    stats: BuildStats,
    rule_hint: str | None = None,
) -> tuple[TaskSpec | None, int]:
    """Classify one task. Returns ``(spec, prompt_chars)``.

    ``prompt_chars`` is the snippet size actually sent to the SLM (0 on a cache
    hit, since no live call is made), so the caller can enforce a token budget.
    """
    evidence = build_task_evidence(project_root, task, max_snippet_chars=caps.max_snippet_chars)
    # Hybrid mode: surface the rule engine's classified type as prior evidence
    # so the SLM can weigh it (it still owns the final type). The hint also
    # enters the cache key so hybrid and slm runs don't share a cache entry.
    if rule_hint:
        evidence.framework_hints.append(f"rule_engine_type={rule_hint}")
    key = evidence_hash(
        snippet=evidence.snippet,
        framework_hints=evidence.framework_hints,
        enclosing_function=evidence.enclosing_function,
        file_path=evidence.file_path,
        model=model,
        template_name="classify_task",
    )
    cached = cache.get(key)
    if cached is not None:
        slm = _SLMTask.model_validate(cached)
        prompt_chars = 0  # cache hit — no live prompt sent
        # A stale cache entry may hold an empty (content-free) classification
        # from a prior failed SLM call. Treat it as a miss and re-run the SLM
        # rather than returning silently-empty rubrics forever.
        if _is_empty_classification(slm):
            stats.cache_hits -= 1
            stats.cache_misses += 1
            cached = None
    if cached is None:
        prompt = render(
            "classify_task",
            snippet=evidence.snippet,
            framework_hints=", ".join(evidence.framework_hints) or "(none)",
            enclosing_function=evidence.enclosing_function or "(module-level)",
            file_path=evidence.file_path,
            metric_names=", ".join(metric_names),
            few_shot=few_shot,
        )
        slm = do_complete(
            model=model,
            messages=_messages(prompt),
            response_model=_SLMTask,
            temperature=0.0,
            max_tokens=caps.max_tokens,
            retries=caps.retries,
        )
        stats.slm_calls += 1
        prompt_chars = len(evidence.snippet)
        # Never persist an empty classification — it would poison the cache
        # and every future run against this evidence hash would return empty
        # inputs/outputs/metrics. Only cache results that carry real content.
        if not _is_empty_classification(slm):
            cache.put(key, slm.model_dump(mode="json"))
    # If the SLM still returned an empty classification (even after the live
    # call), fall back to the detector's own evidence + default metrics so the
    # rubric is usable rather than silently empty.
    if _is_empty_classification(slm):
        return _fallback_task_spec(task), prompt_chars
    _apply_rag_downgrade_guard(slm, task)
    _apply_hint_type_pin(slm, task)
    return _task_spec_from_slm(
        slm, file_path=task.file_path, entry=task.entry, project_root=project_root
    ), prompt_chars


def _recover_empty(
    *,
    project_root: Path,
    model: str,
    caps: EvidenceCaps,
    do_complete: CompleteFn,
    cache: ResponseCache,
    metric_names: list[str],
    few_shot: str,
    stats: BuildStats,
) -> list[_SLMTaskNamed]:
    empty_ev = select_entry_point_files(project_root)
    if not empty_ev.entry_files:
        return []
    fields = empty_ev.to_prompt_fields()
    key = evidence_hash(
        snippet=fields["entry_files"],
        framework_hints=[],
        enclosing_function=None,
        file_path="<repo>",
        model=model,
        template_name="empty_recover",
    )
    cached = cache.get(key)
    if cached is not None:
        recovery = _SLMRecovery.model_validate(cached)
        # A stale cache entry may hold empty (content-free) recovered tasks
        # from a prior failed SLM call. Treat it as a miss and re-run.
        if recovery.tasks and all(
            not t.inputs and not t.outputs and not t.metrics and not t.purpose
            for t in recovery.tasks
        ):
            cached = None
    if cached is None:
        prompt = render(
            "empty_recover",
            entry_files=fields["entry_files"],
            file_tree=fields["file_tree"],
            metric_names=", ".join(metric_names),
            few_shot=few_shot,
        )
        recovery = do_complete(
            model=model,
            messages=_messages(prompt),
            response_model=_SLMRecovery,
            temperature=0.0,
            max_tokens=caps.max_tokens,
            retries=caps.retries,
        )
        stats.slm_calls += 1
        # Only cache non-empty recoveries; an all-empty result would poison
        # the cache and silently yield empty rubrics on every future run.
        if recovery.tasks and any(
            t.inputs or t.outputs or t.metrics or t.purpose for t in recovery.tasks
        ):
            cache.put(key, recovery.model_dump(mode="json"))
    # Drop any recovered tasks that came back content-free — they'd produce
    # silently-empty rubric entries with no inputs/outputs/metrics.
    return [t for t in recovery.tasks if t.inputs or t.outputs or t.metrics or t.purpose]


__all__ = [
    "BuildStats",
    "EvidenceCaps",
    "RubricEngineError",
    "build_rubrics_slm",
]

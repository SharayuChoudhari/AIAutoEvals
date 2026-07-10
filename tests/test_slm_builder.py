"""Tests for the SLM rubric builder (plan task H.17).

The litellm client is never called: tests inject a fake ``complete`` via the
``complete_fn`` parameter of :func:`build_rubrics_slm` (dependency injection,
per the plan's mocking strategy).

Coverage:
- evidence.py: snippet assembly (enclosing-function source) + entry-point
  selection for the empty-evidence fallback.
- build_rubrics_slm with a mocked client for: RAG-with-manual-retrieval,
  booking/tool-call, scoring, extraction, and the empty-evidence fallback.
- validate_against_scan drops hallucinated file_paths / entries.
- cache hit short-circuits the SLM call (identical evidence hash → no second
  call).
- hard-fail (RubricEngineError) when the injected complete raises after
  retries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai_eval.config.schema import JudgeConfig, RubricsConfig, TaskSpec
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.slm.builder import (
    EvidenceCaps,
    RubricEngineError,
    build_rubrics_slm,
)
from ai_eval.inference.slm.cache import ResponseCache
from ai_eval.inference.slm.evidence import (
    build_task_evidence,
    select_entry_point_files,
)
from ai_eval.inference.slm.validation import validate_against_scan

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _task(
    name: str = "t",
    *,
    file_path: str = "src/app.py",
    entry: str | None = "handler",
    type_: str = "chat",
    framework: str = "openai",
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    evidence: list[str] | None = None,
) -> DetectedTask:
    return DetectedTask(
        name=name,
        framework=framework,
        type=type_,
        file_path=file_path,
        entry=entry,
        inputs=inputs or [],
        outputs=outputs or [],
        evidence=evidence or [],
    )


def _scan(tasks: list[DetectedTask]) -> ScanResult:
    return ScanResult(
        files_scanned=max(1, len(tasks)),
        elapsed_seconds=0.0,
        tasks=tasks,
        frameworks_seen={t.framework for t in tasks},
    )


def _write_func(repo: Path, rel: str, func_name: str, body: str) -> None:
    """Write a python file containing a top-level function ``func_name``."""
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"def {func_name}():\n    {body}\n", encoding="utf-8")


class _FakeComplete:
    """Records calls and returns canned response-model instances.

    ``mapping`` maps response_model type -> value (instance) or callable
    ``(messages) -> instance``. A callable value is invoked with the messages
    list so per-call variation (e.g. round-robin) is possible.
    """

    def __init__(self, mapping: dict[type, Any]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._mapping = mapping

    def __call__(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_model: type,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retries: int = 2,
    ) -> Any:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_model": response_model,
            }
        )
        if response_model not in self._mapping:
            raise AssertionError(f"no canned response for {response_model!r}")
        val = self._mapping[response_model]
        if callable(val) and not isinstance(val, type):
            return val(messages)
        return val


def _make_complete(mapping: dict[type, Any]) -> _FakeComplete:
    """Build a fake complete returning ``mapping[response_model]`` per call.

    Values may be instances or callables(messages) -> instance.
    """
    return _FakeComplete(mapping)


# ---------------------------------------------------------------------------
# evidence.py: snippet assembly
# ---------------------------------------------------------------------------


def test_build_task_evidence_uses_enclosing_function_source(tmp_path: Path) -> None:
    _write_func(
        tmp_path,
        "src/app.py",
        "handler",
        "return openai.chat.completions.create(model='gpt-4o-mini')",
    )
    task = _task(file_path="src/app.py", entry="handler", evidence=["openai chat"])
    ev = build_task_evidence(tmp_path, task, max_snippet_chars=1500)
    assert "def handler" in ev.snippet
    assert "openai.chat.completions.create" in ev.snippet
    assert ev.enclosing_function == "handler"
    assert ev.file_path == "src/app.py"
    # framework + first evidence lines surface as hints
    assert "openai" in ev.framework_hints


def test_build_task_evidence_truncates_to_cap(tmp_path: Path) -> None:
    long_body = "x = " + repr("a" * 4000)
    _write_func(tmp_path, "src/big.py", "handler", long_body)
    task = _task(file_path="src/big.py", entry="handler")
    ev = build_task_evidence(tmp_path, task, max_snippet_chars=200)
    assert len(ev.snippet) <= 200
    assert ev.snippet.endswith("...")


def test_build_task_evidence_falls_back_to_evidence_when_no_source(
    tmp_path: Path,
) -> None:
    # entry points at a function not present in the file → fall back to evidence
    task = _task(file_path="src/missing.py", entry="nope", evidence=["sig: openai chat"])
    ev = build_task_evidence(tmp_path, task)
    assert "sig: openai chat" in ev.snippet


# ---------------------------------------------------------------------------
# evidence.py: entry-point selection
# ---------------------------------------------------------------------------


def test_select_entry_point_files_picks_main_and_handlers(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "util.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "handlers").mkdir()
    (tmp_path / "handlers" / "bot.py").write_text("def bot(): ...\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("docs\n", encoding="utf-8")

    ev = select_entry_point_files(tmp_path)
    rels = {r for r, _ in ev.entry_files}
    assert "main.py" in rels
    assert "handlers/bot.py" in rels
    # Non-entry files are not in entry_files but ARE in the file tree.
    assert "src/util.py" in ev.file_tree
    assert "README.md" not in rels


def test_select_entry_point_files_truncates_content(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("x = '" + "a" * 4000 + "'\n", encoding="utf-8")
    ev = select_entry_point_files(tmp_path, max_file_chars=300)
    for _rel, content in ev.entry_files:
        assert len(content) <= 300


# ---------------------------------------------------------------------------
# build_rubrics_slm: per-task classification (mocked client)
# ---------------------------------------------------------------------------


def _slm_task(
    type_: str = "rag",
    purpose: str | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    metrics: list[dict] | None = None,
):
    from ai_eval.inference.slm.builder import _SLMTask

    return _SLMTask(
        type=type_,
        purpose=purpose or "test",
        inputs=inputs or [],
        outputs=outputs or [],
        metrics=metrics or [],
    )


def _slm_metric(name: str):
    from ai_eval.inference.slm.builder import _SLMMetric

    return _SLMMetric(name=name)


def test_build_slm_rag_with_manual_retrieval(tmp_path: Path) -> None:
    """Failure mode #1: RAG misclassified as chat by rules is correctly typed
    'rag' by the SLM."""
    _write_func(
        tmp_path,
        "services/rag.py",
        "tenant_qa",
        "docs = sql_search(query, tenant_id); "
        "return openai.chat.completions.create(model='gpt-4o-mini')",
    )
    task = _task(
        name="tenant_qa",
        file_path="services/rag.py",
        entry="tenant_qa",
        type_="chat",  # rules misclassified
        inputs=["query", "tenant_id"],
        outputs=["documents"],
    )
    scan = _scan([task])

    canned = _slm_task(
        type_="rag",
        purpose="RAG with manual SQL retrieval filtered by tenant",
        inputs=["query", "tenant_id"],
        outputs=["answer"],
        metrics=[_slm_metric("context_precision"), _slm_metric("faithfulness")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})

    rubrics, stats, _report = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert stats.slm_calls == 1
    assert rubrics.rubric_engine == "slm"
    task_spec = rubrics.tasks["tenant_qa"]
    assert task_spec.type == "rag"
    assert task_spec.purpose.startswith("RAG")
    metric_names = {m.name for m in task_spec.metrics}
    assert {"context_precision", "faithfulness"} <= metric_names
    # registry defaults applied for threshold since SLM omitted them
    faith = next(m for m in task_spec.metrics if m.name == "faithfulness")
    assert faith.threshold == 0.9  # registry default


def test_build_slm_guard_promotes_detector_confirmed_rag_over_weak_chat(
    tmp_path: Path,
) -> None:
    """Regression: a weak SLM returning ``type: chat`` for a task the detector
    already classified ``rag`` must NOT silently downgrade the rubric to chat.
    The guard promotes the task back to ``rag`` with RAG metrics."""
    _write_func(
        tmp_path, "services/chat_messages.py", "chat_messages",
        "return vectorstore.similarity_search(query, filter={'tenant_id': tenant_id})",
    )
    task = _task(
        name="chat_messages",
        file_path="services/chat_messages.py",
        entry="chat_messages",
        type_="rag",  # detector correctly classified retrieval
        inputs=["query", "tenant_id"],
        outputs=["documents"],
    )
    scan = _scan([task])
    # Weak SLM punts to chat with only hallucination_rate.
    canned = _slm_task(
        type_="chat",
        purpose="chat over retrieved docs",
        inputs=["query", "tenant_id"],
        outputs=["answer"],
        metrics=[_slm_metric("hallucination_rate")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _report = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    spec = rubrics.tasks["chat_messages"]
    assert spec.type == "rag"
    assert {m.name for m in spec.metrics} == {"context_precision", "faithfulness"}
    # SLM's purpose description survives the guard.
    assert spec.purpose and "retrieved" in spec.purpose
    assert rubrics.project_type == "rag"


def test_build_slm_guard_recovers_rag_from_io_shape_without_detector(
    tmp_path: Path,
) -> None:
    """The guard also fires on IO shape alone (query input + answer output)
    even when the detector itself typed the task ``chat`` — mirrors the rules
    engine's ``_looks_like_rag`` recovery for undetected frameworks."""
    _write_func(tmp_path, "services/rag.py", "tenant_qa",
                "return openai.chat.completions.create(model='gpt-4o-mini')")
    task = _task(
        name="tenant_qa",
        file_path="services/rag.py",
        entry="tenant_qa",
        type_="chat",  # undetected framework → rules engine fallback
        inputs=["query", "tenant_id"],
        outputs=["documents"],
    )
    scan = _scan([task])
    canned = _slm_task(
        type_="chat",
        inputs=["query", "tenant_id"],
        outputs=["documents"],
        metrics=[_slm_metric("hallucination_rate")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert rubrics.tasks["tenant_qa"].type == "rag"
    assert rubrics.project_type == "rag"


def test_build_slm_guard_does_not_promote_chat_qa_without_detector(tmp_path: Path) -> None:
    """A plain chat Q&A shape (query input + answer output, no documents, no
    detector rag flag) must NOT be promoted to rag — that would over-classify
    genuine chat tasks. The broad answer-like hint is only trusted with
    detector confirmation."""
    _write_func(
        tmp_path, "chat/qa.py", "answer",
        "return openai.chat.completions.create(model='gpt-4o-mini')",
    )
    task = _task(name="answer", file_path="chat/qa.py", entry="answer",
                 type_="chat", inputs=["query"], outputs=["answer"])
    scan = _scan([task])
    canned = _slm_task(
        type_="chat", inputs=["query"], outputs=["answer"],
        metrics=[_slm_metric("hallucination_rate")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert rubrics.tasks["answer"].type == "chat"


def test_build_slm_guard_preserves_deliberate_non_rag_classification(
    tmp_path: Path,
) -> None:
    """When the SLM deliberately picks a specific non-RAG type (extraction,
    not a chat/other punt), the guard must NOT override it — the SLM owns the
    type in that case."""
    _write_func(tmp_path, "extract/invoice.py", "parse_invoice",
                "return openai.chat.completions.create(model='gpt-4o-mini')")
    task = _task(
        name="parse_invoice",
        file_path="extract/invoice.py",
        entry="parse_invoice",
        type_="rag",  # detector misfires; SLM correctly says extraction
        inputs=["query", "document_text"],
        outputs=["answer"],
    )
    scan = _scan([task])
    canned = _slm_task(
        type_="extraction",
        inputs=["document_text"],
        outputs=["vendor", "total"],
        metrics=[_slm_metric("extraction_field_accuracy")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert rubrics.tasks["parse_invoice"].type == "extraction"


def test_build_slm_guard_keeps_slm_rag_metrics_when_present(tmp_path: Path) -> None:
    """If the guard promotes type→rag but the SLM already supplied a RAG metric,
    keep the SLM's metrics rather than clobbering with defaults."""
    _write_func(tmp_path, "services/rag.py", "qa", "return vectorstore.similarity_search(q)")
    task = _task(name="qa", file_path="services/rag.py", entry="qa",
                 type_="rag", inputs=["query"], outputs=["documents"])
    scan = _scan([task])
    canned = _slm_task(
        type_="chat",
        inputs=["query"],
        outputs=["answer"],
        metrics=[_slm_metric("context_precision"), _slm_metric("faithfulness"),
                 _slm_metric("hallucination_rate")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    names = {m.name for m in rubrics.tasks["qa"].metrics}
    assert names == {"context_precision", "faithfulness", "hallucination_rate"}


def test_build_slm_booking_tool_call(tmp_path: Path) -> None:
    """Failure mode #2: booking bot with no recognized framework → SLM recovers
    a booking task via per-task classification."""
    _write_func(
        tmp_path,
        "booking/app.py",
        "book_slot",
        "return openai.chat.completions.create(model='gpt-4o-mini', tools=[book_schema])",
    )
    task = _task(
        name="book_slot",
        file_path="booking/app.py",
        entry="book_slot",
        type_="chat",
    )
    scan = _scan([task])
    canned = _slm_task(
        type_="booking",
        purpose="Appointment booking with slot selection",
        inputs=["user_intent"],
        outputs=["booking_confirmation"],
        metrics=[_slm_metric("slot_filling_accuracy"), _slm_metric("task_completion")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert rubrics.tasks["book_slot"].type == "booking"
    # project_type reflects the open vocabulary
    assert rubrics.project_type in {"tool_calling", "agent", "rag_and_tools"}


def test_build_slm_scoring_and_extraction(tmp_path: Path) -> None:
    _write_func(
        tmp_path, "eval/score.py", "grade",
        "return openai.chat.completions.create(model='gpt-4o')",
    )
    _write_func(
        tmp_path, "extract/invoice.py", "parse",
        "return openai.chat.completions.create(model='gpt-4o-mini', "
        "response_format={'type':'json_object'})",
    )
    tasks = [
        _task(name="grade", file_path="eval/score.py", entry="grade", type_="chat"),
        _task(name="parse", file_path="extract/invoice.py", entry="parse", type_="chat"),
    ]
    scan = _scan(tasks)
    from ai_eval.inference.slm.builder import _SLMTask

    responses = [
        _slm_task(
            type_="scoring",
            inputs=["question", "student_answer"],
            metrics=[_slm_metric("scoring_accuracy")],
        ),
        _slm_task(
            type_="extraction",
            inputs=["document_text"],
            outputs=["vendor", "total"],
            metrics=[_slm_metric("extraction_field_accuracy")],
        ),
    ]
    it = iter(responses)
    fake = _make_complete({_SLMTask: lambda msgs: next(it)})
    rubrics, stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert stats.slm_calls == 2
    assert rubrics.tasks["grade"].type == "scoring"
    assert rubrics.tasks["parse"].type == "extraction"


def test_build_slm_respects_max_tasks_cap(tmp_path: Path) -> None:
    _write_func(tmp_path, "src/a.py", "f1", "return 1")
    _write_func(tmp_path, "src/b.py", "f2", "return 2")
    _write_func(tmp_path, "src/c.py", "f3", "return 3")
    tasks = [
        _task(name=f"f{i}", file_path=f"src/{'abc'[i-1]}.py", entry=f"f{i}", type_="chat")
        for i in range(1, 4)
    ]
    scan = _scan(tasks)
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete(
        {_SLMTask: _slm_task(type_="chat", metrics=[_slm_metric("hallucination_rate")])}
    )
    caps = EvidenceCaps(max_tasks=2)
    rubrics, stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake, caps=caps
    )
    assert stats.slm_calls == 2
    assert len(rubrics.tasks) == 2


# ---------------------------------------------------------------------------
# build_rubrics_slm: empty-evidence fallback
# ---------------------------------------------------------------------------


def test_build_slm_empty_evidence_fallback(tmp_path: Path) -> None:
    """Empty scan.tasks → entry-point selection + one repo-level SLM call."""
    (tmp_path / "main.py").write_text(
        "def book_appointment(intent):\n    return 'confirmed'\n", encoding="utf-8"
    )
    scan = _scan([])
    from ai_eval.inference.slm.builder import _SLMRecovery, _SLMTaskNamed

    canned = _SLMRecovery(
        tasks=[
            _SLMTaskNamed(
                name="book_appointment",
                file_path="main.py",
                entry="book_appointment",
                type="booking",
                purpose="book an appointment",
                inputs=["intent"],
                outputs=["confirmation"],
                metrics=[_slm_metric("task_completion")],
            )
        ]
    )
    fake = _make_complete({_SLMRecovery: canned})
    rubrics, stats, _report = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert stats.fallback_used is True
    assert stats.slm_calls == 1
    assert "book_appointment" in rubrics.tasks
    assert rubrics.tasks["book_appointment"].type == "booking"


def test_build_slm_empty_evidence_no_entry_files(tmp_path: Path) -> None:
    """No entry-point files found → empty tasks, no SLM call."""
    (tmp_path / "data.txt").write_text("not python\n", encoding="utf-8")
    scan = _scan([])
    from ai_eval.inference.slm.builder import _SLMRecovery

    fake = _make_complete({_SLMRecovery: _SLMRecovery()})
    rubrics, stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    assert stats.slm_calls == 0
    assert rubrics.tasks == {}
    assert rubrics.project_type == "custom"


# ---------------------------------------------------------------------------
# validate_against_scan: drop hallucinated paths / entries
# ---------------------------------------------------------------------------


def test_validate_drops_hallucinated_file_path(tmp_path: Path) -> None:
    scan = _scan([_task(file_path="src/real.py")])
    rubrics = RubricsConfig(
        judge=JudgeConfig(default="ollama/test:1b"),
        tasks={
            "real": TaskSpec(file_path="src/real.py", type="chat"),
            "ghost": TaskSpec(file_path="src/ghost.py", type="chat"),  # not scanned
        },
    )
    rubrics, report = validate_against_scan(rubrics, scan, project_root=tmp_path)
    assert "real" in rubrics.tasks
    assert "ghost" not in rubrics.tasks
    assert "ghost" in report.dropped_tasks


def test_validate_clears_entry_not_defined_in_file(tmp_path: Path) -> None:
    _write_func(tmp_path, "src/app.py", "real_fn", "return 1")
    scan = _scan([_task(file_path="src/app.py")])
    rubrics = RubricsConfig(
        judge=JudgeConfig(default="ollama/test:1b"),
        tasks={
            "t": TaskSpec(file_path="src/app.py", entry="made_up_fn", type="chat"),
        },
    )
    rubrics, report = validate_against_scan(rubrics, scan, project_root=tmp_path)
    assert rubrics.tasks["t"].entry is None
    assert "t" in report.cleared_entries


def test_validate_warns_on_rag_misclassification(tmp_path: Path) -> None:
    scan = _scan([_task(file_path="src/app.py")])
    rubrics = RubricsConfig(
        judge=JudgeConfig(default="ollama/test:1b"),
        tasks={
            "t": TaskSpec(
                file_path="src/app.py",
                type="chat",
                inputs=["query"],
                outputs=["documents"],
            ),
        },
    )
    import warnings as w

    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        rubrics, report = validate_against_scan(
            rubrics, scan, project_root=tmp_path, warn=True
        )
    assert report.rag_misclassification_warnings
    assert any("resembles RAG" in str(m.message) for m in caught)


# ---------------------------------------------------------------------------
# Cache: hit short-circuits the SLM call
# ---------------------------------------------------------------------------


def test_cache_hit_short_circuits_slm_call(tmp_path: Path) -> None:
    _write_func(
        tmp_path,
        "src/app.py",
        "handler",
        "return openai.chat.completions.create(model='gpt-4o-mini')",
    )
    task = _task(name="handler", file_path="src/app.py", entry="handler")
    scan = _scan([task])
    from ai_eval.inference.slm.builder import _SLMTask

    canned = _slm_task(type_="rag", metrics=[_slm_metric("faithfulness")])
    fake = _make_complete({_SLMTask: canned})

    cache = ResponseCache(tmp_path)
    # First run: cache miss → 1 SLM call.
    r1, s1, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake, cache=cache
    )
    assert s1.slm_calls == 1
    assert s1.cache_misses >= 1
    # Second run: identical evidence hash → cache hit → no new SLM call.
    r2, s2, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake, cache=cache
    )
    assert s2.slm_calls == 0
    assert s2.cache_hits >= 1
    # Determinism: identical input → identical output type.
    assert r1.tasks["handler"].type == r2.tasks["handler"].type == "rag"


def test_cache_disabled_makes_no_hits(tmp_path: Path) -> None:
    _write_func(tmp_path, "src/app.py", "handler", "return 1")
    task = _task(name="handler", file_path="src/app.py", entry="handler")
    scan = _scan([task])
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete(
        {_SLMTask: _slm_task(type_="chat", metrics=[_slm_metric("hallucination_rate")])}
    )
    cache = ResponseCache(tmp_path, enabled=False)
    build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake, cache=cache
    )
    build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake, cache=cache
    )
    # Disabled cache → 2 live SLM calls, no hits/misses recorded.
    assert len(fake.calls) == 2
    assert cache.stats.hits == 0
    assert cache.stats.misses == 0  # disabled cache doesn't record


# ---------------------------------------------------------------------------
# Hard-fail when complete raises after retries
# ---------------------------------------------------------------------------


def test_build_slm_hard_fails_on_complete_error(tmp_path: Path) -> None:
    _write_func(tmp_path, "src/app.py", "handler", "return 1")
    task = _task(name="handler", file_path="src/app.py", entry="handler")
    scan = _scan([task])

    def raising(**kw):
        raise RuntimeError("ollama unreachable")

    with pytest.raises(RuntimeError):
        build_rubrics_slm(
            scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=raising
        )


def test_rubric_engine_error_carries_remediation() -> None:
    err = RubricEngineError("boom", why="conn refused", fix="ollama serve")
    assert err.why == "conn refused"
    assert err.fix == "ollama serve"
    assert "boom" in str(err)


# ---------------------------------------------------------------------------
# Empty-classification guard: reject silently-empty SLM results
# ---------------------------------------------------------------------------


def test_empty_live_slm_falls_back_to_detector_evidence(tmp_path: Path) -> None:
    """When the live SLM returns a content-free classification (no
    inputs/outputs/metrics/purpose), the builder falls back to the detector's
    own evidence + default metrics so the rubric is never silently empty.
    The empty result must NOT be cached."""
    _write_func(
        tmp_path, "services/chat.py", "chat_messages_chain",
        "return openai.chat.completions.create(model='gpt-4o-mini', messages=messages)",
    )
    task = _task(
        name="chat_messages_chain",
        file_path="services/chat.py",
        entry="chat_messages_chain",
        type_="chat",
        inputs=["messages"],
        outputs=["content"],
    )
    scan = _scan([task])
    from ai_eval.inference.slm.builder import _SLMTask

    # Always-empty (but schema-valid) SLM response: no purpose/IO/metrics.
    fake = _make_complete(
        {_SLMTask: _SLMTask(type="chat", purpose=None, inputs=[], outputs=[], metrics=[])}
    )
    cache = ResponseCache(tmp_path)
    rubrics, stats, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b",
        complete_fn=fake, cache=cache,
    )
    assert stats.slm_calls == 1
    spec = rubrics.tasks["chat_messages_chain"]
    # Fell back to detector evidence — not empty.
    assert spec.inputs == ["messages"]
    assert spec.outputs == ["content"]
    assert [m.name for m in spec.metrics] == ["hallucination_rate"]
    # Empty result was not cached (would otherwise poison future runs).
    assert not list((tmp_path / ".ai-evals" / "rubric_cache").glob("*.json"))


def test_stale_empty_cache_is_treated_as_a_miss(tmp_path: Path) -> None:
    """A cache entry holding a content-free classification from a prior failed
    SLM call is rejected (treated as a miss) and a fresh SLM call is made,
    rather than returning silently-empty rubrics forever."""
    _write_func(
        tmp_path, "services/chat.py", "chat_messages_chain",
        "return openai.chat.completions.create(model='gpt-4o-mini', messages=messages)",
    )
    task = _task(
        name="chat_messages_chain",
        file_path="services/chat.py",
        entry="chat_messages_chain",
        type_="chat",
        inputs=["messages"],
        outputs=["content"],
    )
    scan = _scan([task])
    from ai_eval.inference.slm.builder import _SLMTask

    good = _slm_task(
        type_="chat",
        purpose="Chat completion",
        inputs=["messages"],
        outputs=["response"],
        metrics=[_slm_metric("faithfulness")],
    )
    fake = _make_complete({_SLMTask: good})
    cache = ResponseCache(tmp_path)

    # First run: produces a good result and caches it.
    r1, s1, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b",
        complete_fn=fake, cache=cache,
    )
    assert s1.slm_calls == 1
    assert r1.tasks["chat_messages_chain"].inputs == ["messages"]

    # Poison the cache with a content-free entry under the same key.
    from ai_eval.inference.slm.cache import evidence_hash

    ev = build_task_evidence(tmp_path, task, max_snippet_chars=1500)
    key = evidence_hash(
        snippet=ev.snippet, framework_hints=sorted(ev.framework_hints),
        enclosing_function=ev.enclosing_function, file_path=ev.file_path,
        model="ollama/test:1b", template_name="classify_task",
    )
    (tmp_path / ".ai-evals" / "rubric_cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".ai-evals" / "rubric_cache" / f"{key}.json").write_text(
        json.dumps(
            {"type": "chat", "inputs": [], "outputs": [], "metrics": [], "purpose": None}
        ),
        encoding="utf-8",
    )

    # Second run: stale empty entry is rejected → fresh SLM call → good result.
    r2, s2, _ = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b",
        complete_fn=fake, cache=cache,
    )
    assert s2.slm_calls == 1  # re-ran the SLM despite a cache file existing
    assert r2.tasks["chat_messages_chain"].inputs == ["messages"]


def test_build_slm_pins_hint_task_type_over_slm(tmp_path: Path) -> None:
    """A hint-flagged task (framework='hint') pins its declared ``type`` so a
    weak SLM cannot reclassify it. The SLM still refines purpose/inputs/
    outputs/metrics, but ``type`` is authoritative from the hint (plan Open
    Question 2)."""
    _write_func(
        tmp_path, "services/wf.py", "run",
        "return temporal.execute(state)",  # no AST detector sees temporal
    )
    # Hint task: framework="hint", declared type="workflow".
    task = _task(
        name="temporal_workflow",
        file_path="services/wf.py",
        entry="run",
        type_="workflow",
        framework="hint",
        inputs=["state"],
        outputs=["state"],
    )
    scan = _scan([task])
    # Weak SLM downgrades to chat — the pin must override this.
    canned = _slm_task(
        type_="chat",
        purpose="some chat thing",
        inputs=["state"],
        outputs=["reply"],
        metrics=[_slm_metric("hallucination_rate")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _report = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    spec = rubrics.tasks["temporal_workflow"]
    # Type pinned from the hint, NOT the SLM's "chat".
    assert spec.type == "workflow"
    # The SLM's descriptive refinement survives (purpose/inputs/outputs).
    assert spec.purpose == "some chat thing"
    assert spec.inputs == ["state"]
    assert spec.outputs == ["reply"]


def test_build_slm_does_not_pin_non_hint_task_type(tmp_path: Path) -> None:
    """A regular AST-detected task (framework != 'hint') is NOT type-pinned —
    the SLM owns its type. Regression guard so the hint pin is scoped to hints."""
    _write_func(
        tmp_path, "src/app.py", "handler",
        "return openai.chat.completions.create(model='gpt-4o-mini')",
    )
    task = _task(
        name="handler",
        file_path="src/app.py",
        entry="handler",
        type_="chat",
        framework="openai",  # not a hint
    )
    scan = _scan([task])
    canned = _slm_task(
        type_="tool_calling",  # SLM reclassifies — allowed for non-hint tasks
        inputs=["messages"],
        outputs=["tool_calls"],
        metrics=[_slm_metric("argument_accuracy")],
    )
    from ai_eval.inference.slm.builder import _SLMTask

    fake = _make_complete({_SLMTask: canned})
    rubrics, _stats, _report = build_rubrics_slm(
        scan, project_root=tmp_path, model="ollama/test:1b", complete_fn=fake
    )
    spec = rubrics.tasks["handler"]
    # SLM's type survives for a non-hint task.
    assert spec.type == "tool_calling"

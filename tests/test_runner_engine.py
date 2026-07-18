"""Tests for the runner engine: aggregation, latency, import errors, parallelism.

A fake async ``complete_fn`` drives the judge gateway; a fixture task module
provides the entry symbol. No network.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from ai_eval.config.schema import (
    DefaultsBlock,
    JudgeConfig,
    MetricSpec,
    RubricsConfig,
    TaskSpec,
)
from ai_eval.metrics import reset_cache
from ai_eval.runner.engine import execute


@pytest.fixture(autouse=True)
def _reset():
    reset_cache()
    yield
    reset_cache()


def _rubrics(tmp_path: Path, file_path: str) -> RubricsConfig:
    return RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "chat_task": TaskSpec(
                file_path=file_path,
                entry="main",
                type="chat",
                metrics=[
                    MetricSpec(name="hallucination_rate", threshold=0.5),
                ],
            )
        },
    )


def _write_task_module(tmp_path: Path, name: str, body: str) -> str:
    (tmp_path / "src").mkdir(exist_ok=True)
    p = tmp_path / "src" / f"{name}.py"
    p.write_text(body, encoding="utf-8")
    return f"src/{name}.py"


def _fake_complete(score: float = 0.9):
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate({"score": score, "rationale": "ok", "sub_scores": {}})

    return complete_fn


def _run(coro):
    return asyncio.run(coro)


def test_engine_aggregates_metric_scores(tmp_path: Path) -> None:
    fp = _write_task_module(tmp_path, "chat1", "def main(query):\n    return {'answer': query}\n")
    rubrics = _rubrics(tmp_path, fp)
    golden = {
        "chat_task": [
            {"id": "e1", "input": {"query": "a"}, "expected": None, "trace": {"calls": []}},
            {"id": "e2", "input": {"query": "b"}, "expected": None, "trace": {"calls": []}},
        ]
    }
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(0.8),
            run_id="r_test1",
            started_at=time.time(),
        )
    )
    assert "chat_task" in record.tasks
    m = record.tasks["chat_task"].metrics["hallucination_rate"]
    assert m.score == pytest.approx(0.8)
    assert m.status == "pass"  # 0.8 >= 0.5
    assert record.summary.examples == 2
    assert record.summary.passed == 2


def test_engine_records_latency(tmp_path: Path) -> None:
    fp = _write_task_module(
        tmp_path, "slow", "import time\ndef main(q):\n    time.sleep(0.01)\n    return q\n"
    )
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(),
            run_id="r_test2",
            started_at=time.time(),
        )
    )
    lat = record.tasks["chat_task"].latency_ms
    assert lat["p50"] >= 0.0
    assert lat["p95"] >= 0.0


def test_engine_task_import_error_recorded_not_crash(tmp_path: Path) -> None:
    fp = "src/missing.py"  # does not exist
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(),
            run_id="r_test3",
            started_at=time.time(),
        )
    )
    assert record.tasks["chat_task"].errors
    assert record.tasks["chat_task"].examples[0].status == "error"
    assert record.summary.errors == 1


def test_engine_task_call_error_recorded(tmp_path: Path) -> None:
    fp = _write_task_module(tmp_path, "boom", "def main(q):\n    raise ValueError('nope')\n")
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(),
            run_id="r_test4",
            started_at=time.time(),
        )
    )
    assert record.tasks["chat_task"].examples[0].status == "error"
    assert "ValueError" in (record.tasks["chat_task"].examples[0].error or "")


def test_engine_unregistered_metric_raises(tmp_path: Path) -> None:
    from ai_eval.runner.thresholds import MetricNotImplementedError

    fp = _write_task_module(tmp_path, "ok", "def main(q):\n    return q\n")
    rubrics = RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "t": TaskSpec(
                file_path=fp,
                entry="main",
                type="chat",
                metrics=[MetricSpec(name="totally_unknown_metric")],
            )
        },
    )
    golden = {"t": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    with pytest.raises(MetricNotImplementedError):
        _run(
            execute(
                rubrics,
                golden,
                project_root=tmp_path,
                complete_fn=_fake_complete(),
                run_id="r_test5",
                started_at=time.time(),
            )
        )


def test_engine_fresh_repo_no_delta(tmp_path: Path) -> None:
    fp = _write_task_module(tmp_path, "fresh", "def main(q):\n    return q\n")
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            baseline=None,
            complete_fn=_fake_complete(0.9),
            run_id="r_test6",
            started_at=time.time(),
        )
    )
    assert record.tasks["chat_task"].metrics["hallucination_rate"].delta is None


def test_engine_baseline_delta(tmp_path: Path) -> None:
    fp = _write_task_module(tmp_path, "delta", "def main(q):\n    return q\n")
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    baseline = {
        "chat_task": {
            "metrics": {"hallucination_rate": {"score": 0.95, "threshold": 0.5, "status": "pass"}}
        }
    }
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            baseline=baseline,
            complete_fn=_fake_complete(0.8),
            run_id="r_test7",
            started_at=time.time(),
        )
    )
    m = record.tasks["chat_task"].metrics["hallucination_rate"]
    assert m.delta == pytest.approx(-0.15)


def test_engine_parallelism_cap(tmp_path: Path) -> None:
    """With parallel=1, task calls are serialized; with parallel=4 they overlap."""
    fp = _write_task_module(
        tmp_path, "par", "import time\ndef main(q):\n    time.sleep(0.05)\n    return q\n"
    )
    rubrics = _rubrics(tmp_path, fp)
    golden = {
        "chat_task": [
            {"id": f"e{i}", "input": f"q{i}", "expected": None, "trace": {"calls": []}}
            for i in range(4)
        ]
    }
    t0 = time.perf_counter()
    _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=_fake_complete(),
            run_id="r_test8",
            started_at=time.time(),
        )
    )
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=4,
            complete_fn=_fake_complete(),
            run_id="r_test9",
            started_at=time.time(),
        )
    )
    parallel_time = time.perf_counter() - t0
    # parallel should be much faster (4 sleep(0.05) overlapping vs serial ~0.2s).
    # Use a generous ratio to avoid scheduler-jitter flakes.
    assert parallel_time < serial * 0.6, (parallel_time, serial)


def test_engine_judge_failure_marks_example_error(tmp_path: Path) -> None:
    fp = _write_task_module(tmp_path, "jfail", "def main(q):\n    return q\n")
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}

    async def failing(*, model, messages, response_model, temperature=0.0):
        raise RuntimeError("judge down")

    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=failing,
            run_id="r_test10",
            started_at=time.time(),
        )
    )
    assert record.tasks["chat_task"].examples[0].status == "error"
    assert "hallucination_rate" in record.tasks["chat_task"].examples[0].metric_errors


def test_engine_node_scoring_from_trace(tmp_path: Path) -> None:
    """A task with ``node_metrics`` scores each matching trace node and rolls
    the per-node scores into ``ExampleRecord.node_scores`` (AGENTS.md §1).

    The entry point runs once; its internal calls are scored from the captured
    trace, not re-executed. The node-level metric's aggregate is the mean of
    node scores across examples.
    """
    from ai_eval.config.schema import NodeMetric

    fp = _write_task_module(tmp_path, "rag1", "def main(query):\n    return query\n")
    rubrics = RubricsConfig(
        schema_version=1,
        project_type="rag",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "rag_task": TaskSpec(
                file_path=fp,
                entry="main",
                type="rag",
                metrics=[],
                node_metrics=[
                    NodeMetric(
                        node_selector="kind=retrieve",
                        metric=MetricSpec(name="context_precision", threshold=0.5),
                    ),
                ],
            )
        },
    )
    golden = {
        "rag_task": [
            {
                "id": "e1",
                "input": {"query": "a"},
                "expected": None,
                "trace": {
                    "calls": [
                        {
                            "kind": "retrieve",
                            "name": "pgvector.search",
                            "args": {"query": "a"},
                            "result": [{"id": 1, "text": "a doc"}],
                            "latency_ms": 1.0,
                        },
                        {
                            "kind": "llm",
                            "name": "openai.create",
                            "args": {"prompt": "a"},
                            "result": {"answer": "a"},
                            "latency_ms": 2.0,
                        },
                    ]
                },
            }
        ]
    }
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(0.88),
            run_id="r_node1",
            started_at=time.time(),
        )
    )
    ex = record.tasks["rag_task"].examples[0]
    # The retrieve node was scored under node_id "retrieve_0".
    assert "retrieve_0" in ex.node_scores
    assert "context_precision" in ex.node_scores["retrieve_0"]
    assert ex.node_scores["retrieve_0"]["context_precision"] == pytest.approx(0.88)
    # The llm node was NOT scored (no node_metric binds kind=llm).
    assert "llm_0" not in ex.node_scores
    # The task-level aggregate surfaces the node metric.
    m = record.tasks["rag_task"].metrics["context_precision"]
    assert m.score == pytest.approx(0.88)
    assert m.status == "pass"  # 0.88 >= 0.5


def test_engine_node_scoring_no_trace_no_scores(tmp_path: Path) -> None:
    """When a task has ``node_metrics`` but the example has no ``trace.calls``
    (e.g. pre-bootstrap auto-seeded fixtures), node scoring is a no-op and the
    aggregate metric is ``skip``. This is the honest behavior — node scoring
    requires captured traces by construction (AGENTS.md §1)."""
    from ai_eval.config.schema import NodeMetric

    fp = _write_task_module(tmp_path, "rag2", "def main(q):\n    return q\n")
    rubrics = RubricsConfig(
        schema_version=1,
        project_type="rag",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "rag_task": TaskSpec(
                file_path=fp,
                entry="main",
                type="rag",
                metrics=[],
                node_metrics=[
                    NodeMetric(
                        node_selector="kind=retrieve",
                        metric=MetricSpec(name="context_precision", threshold=0.5),
                    ),
                ],
            )
        },
    )
    golden = {
        "rag_task": [
            {"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}},
        ]
    }
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(0.9),
            run_id="r_node2",
            started_at=time.time(),
        )
    )
    ex = record.tasks["rag_task"].examples[0]
    assert ex.node_scores == {}
    m = record.tasks["rag_task"].metrics["context_precision"]
    assert m.status == "skip"


def test_engine_node_selector_name_substring(tmp_path: Path) -> None:
    """The ``name~=`` selector matches by substring on ``call["name"]``, and a
    selector matching multiple nodes scores each independently (node_id is
    ``<kind>_<i>`` so each match gets a distinct rollup key)."""
    from ai_eval.config.schema import NodeMetric

    fp = _write_task_module(tmp_path, "rag3", "def main(query):\n    return query\n")
    rubrics = RubricsConfig(
        schema_version=1,
        project_type="rag",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "rag_task": TaskSpec(
                file_path=fp,
                entry="main",
                type="rag",
                metrics=[],
                node_metrics=[
                    NodeMetric(
                        node_selector="name~=pgvector",
                        metric=MetricSpec(name="context_precision", threshold=0.0),
                    ),
                ],
            )
        },
    )
    golden = {
        "rag_task": [
            {
                "id": "e1",
                "input": {"query": "a"},
                "expected": None,
                "trace": {
                    "calls": [
                        {
                            "kind": "retrieve",
                            "name": "pgvector.orm.op",
                            "args": {},
                            "result": [],
                            "latency_ms": 1.0,
                        },
                        {
                            "kind": "retrieve",
                            "name": "pgvector.search",
                            "args": {},
                            "result": [],
                            "latency_ms": 1.0,
                        },
                        {
                            "kind": "retrieve",
                            "name": "chromadb.query",
                            "args": {},
                            "result": [],
                            "latency_ms": 1.0,
                        },
                    ]
                },
            }
        ]
    }
    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=_fake_complete(0.7),
            run_id="r_node3",
            started_at=time.time(),
        )
    )
    ex = record.tasks["rag_task"].examples[0]
    # Two matches (pgvector.orm.op, pgvector.search); chromadb.query skipped.
    assert "retrieve_0" in ex.node_scores
    assert "retrieve_1" in ex.node_scores
    assert len(ex.node_scores) == 2
    # Aggregate is the mean of the two node scores (both 0.7).
    m = record.tasks["rag_task"].metrics["context_precision"]
    assert m.score == pytest.approx(0.7)


def test_engine_node_metric_unimplemented_raises(tmp_path: Path) -> None:
    """A node metric with no implementation fails fast at run start
    (MetricNotImplementedError), not silently per-example."""
    from ai_eval.config.schema import NodeMetric
    from ai_eval.runner.thresholds import MetricNotImplementedError

    fp = _write_task_module(tmp_path, "rag4", "def main(q):\n    return q\n")
    rubrics = RubricsConfig(
        schema_version=1,
        project_type="rag",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "rag_task": TaskSpec(
                file_path=fp,
                entry="main",
                type="rag",
                metrics=[],
                node_metrics=[
                    NodeMetric(
                        node_selector="kind=retrieve",
                        metric=MetricSpec(name="totally_unknown_node_metric"),
                    ),
                ],
            )
        },
    )
    golden = {
        "rag_task": [
            {
                "id": "e1",
                "input": "q",
                "expected": None,
                "trace": {
                    "calls": [
                        {
                            "kind": "retrieve",
                            "name": "x",
                            "args": {},
                            "result": [],
                            "latency_ms": 1.0,
                        }
                    ]
                },
            },
        ]
    }
    with pytest.raises(MetricNotImplementedError):
        _run(
            execute(
                rubrics,
                golden,
                project_root=tmp_path,
                complete_fn=_fake_complete(),
                run_id="r_node4",
                started_at=time.time(),
            )
        )


def test_engine_old_run_json_loads_without_node_scores(tmp_path: Path) -> None:
    """An old ``run.json`` written before ``node_scores`` existed loads fine
    — the field defaults to empty. Backward-compat for the schema change."""
    from ai_eval.runner.record import ExampleRecord

    ex = ExampleRecord(id="e1", status="pass", metric_scores={"m": 0.9})
    assert ex.node_scores == {}


def test_engine_node_scoring_does_not_fail_example_on_judge_error(
    tmp_path: Path,
) -> None:
    """A node-level judge failure is best-effort: it doesn't mark the example
    as ``error`` (only entry-level metric failures do). Node scores are
    diagnostic signal, not pass/fail gates."""
    from ai_eval.config.schema import NodeMetric

    fp = _write_task_module(tmp_path, "rag5", "def main(q):\n    return q\n")
    rubrics = RubricsConfig(
        schema_version=1,
        project_type="rag",
        judge=JudgeConfig(default="fake/local"),
        defaults=DefaultsBlock(),
        tasks={
            "rag_task": TaskSpec(
                file_path=fp,
                entry="main",
                type="rag",
                metrics=[],
                node_metrics=[
                    NodeMetric(
                        node_selector="kind=retrieve",
                        metric=MetricSpec(name="context_precision", threshold=0.5),
                    ),
                ],
            )
        },
    )
    golden = {
        "rag_task": [
            {
                "id": "e1",
                "input": "q",
                "expected": None,
                "trace": {
                    "calls": [
                        {
                            "kind": "retrieve",
                            "name": "x",
                            "args": {},
                            "result": [],
                            "latency_ms": 1.0,
                        }
                    ]
                },
            },
        ]
    }

    async def failing(*, model, messages, response_model, temperature=0.0):
        raise RuntimeError("judge down")

    record = _run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            complete_fn=failing,
            run_id="r_node5",
            started_at=time.time(),
        )
    )
    ex = record.tasks["rag_task"].examples[0]
    # The node score wasn't recorded (judge failed), but the example is pass
    # (no entry-level metrics to fail, and node failures are best-effort).
    assert ex.status == "pass"
    assert ex.node_scores == {}
    # The aggregate is skip (no node scores collected).
    m = record.tasks["rag_task"].metrics["context_precision"]
    assert m.status == "skip"


def test_render_run_shows_node_scores_block(tmp_path: Path) -> None:
    """The human run render emits a compact per-node block when an example has
    ``node_scores`` (AGENTS.md §1)."""
    import io

    from rich.console import Console

    from ai_eval.cli.render.tables import render_run
    from ai_eval.runner.record import (
        ExampleRecord,
        MetricResult,
        RunRecord,
        RunSummary,
        TaskRecord,
    )

    record = RunRecord(
        id="r1",
        started_at=0.0,
        finished_at=1.0,
        git={"sha": None, "branch": None, "dirty": None},
        config_hash="h",
        baseline_id=None,
        tags=[],
        summary=RunSummary(passed=1, failed=0, errors=0, examples=1),
        tasks={
            "rag_task": TaskRecord(
                examples=[
                    ExampleRecord(
                        id="ex",
                        status="pass",
                        node_scores={
                            "retrieve_0": {"context_precision": 0.88},
                            "llm_0": {"hallucination_rate": 0.05},
                        },
                    )
                ],
                metrics={
                    "context_precision": MetricResult(
                        name="context_precision", score=0.88, status="pass"
                    ),
                },
            ),
        },
    )
    buf = io.StringIO()
    import ai_eval.cli.render.tables as t

    orig = t.stdout_console
    t.stdout_console = lambda *, no_color: Console(file=buf, force_terminal=False, no_color=True)
    try:
        render_run(record, no_color=True)
    finally:
        t.stdout_console = orig
    out = buf.getvalue()
    assert "nodes:" in out
    assert "retrieve_0" in out
    assert "context_precision=0.8800" in out
    assert "llm_0" in out
    assert "hallucination_rate=0.0500" in out


def test_render_run_omits_node_block_when_empty(tmp_path: Path) -> None:
    """Tasks without ``node_scores`` render unchanged — no per-node block."""
    import io

    from rich.console import Console

    from ai_eval.cli.render.tables import render_run
    from ai_eval.runner.record import (
        ExampleRecord,
        MetricResult,
        RunRecord,
        RunSummary,
        TaskRecord,
    )

    record = RunRecord(
        id="r2",
        started_at=0.0,
        finished_at=1.0,
        git={"sha": None, "branch": None, "dirty": None},
        config_hash="h",
        baseline_id=None,
        tags=[],
        summary=RunSummary(passed=1, failed=0, errors=0, examples=1),
        tasks={
            "chat_task": TaskRecord(
                examples=[ExampleRecord(id="ex", status="pass")],
                metrics={
                    "hallucination_rate": MetricResult(
                        name="hallucination_rate", score=0.1, status="pass"
                    ),
                },
            ),
        },
    )
    buf = io.StringIO()
    import ai_eval.cli.render.tables as t

    orig = t.stdout_console
    t.stdout_console = lambda *, no_color: Console(file=buf, force_terminal=False, no_color=True)
    try:
        render_run(record, no_color=True)
    finally:
        t.stdout_console = orig
    out = buf.getvalue()
    assert "nodes:" not in out

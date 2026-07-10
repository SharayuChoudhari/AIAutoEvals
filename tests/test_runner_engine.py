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
        return response_model.model_validate(
            {"score": score, "rationale": "ok", "sub_scores": {}}
        )
    return complete_fn


def _run(coro):
    return asyncio.run(coro)


def test_engine_aggregates_metric_scores(tmp_path: Path) -> None:
    fp = _write_task_module(
        tmp_path, "chat1", "def main(query):\n    return {'answer': query}\n"
    )
    rubrics = _rubrics(tmp_path, fp)
    golden = {
        "chat_task": [
            {"id": "e1", "input": {"query": "a"}, "expected": None, "trace": {"calls": []}},
            {"id": "e2", "input": {"query": "b"}, "expected": None, "trace": {"calls": []}},
        ]
    }
    record = _run(
        execute(
            rubrics, golden, project_root=tmp_path,
            complete_fn=_fake_complete(0.8),
            run_id="r_test1", started_at=time.time(),
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
            rubrics, golden, project_root=tmp_path,
            complete_fn=_fake_complete(), run_id="r_test2", started_at=time.time(),
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
            rubrics, golden, project_root=tmp_path,
            complete_fn=_fake_complete(), run_id="r_test3", started_at=time.time(),
        )
    )
    assert record.tasks["chat_task"].errors
    assert record.tasks["chat_task"].examples[0].status == "error"
    assert record.summary.errors == 1


def test_engine_task_call_error_recorded(tmp_path: Path) -> None:
    fp = _write_task_module(
        tmp_path, "boom", "def main(q):\n    raise ValueError('nope')\n"
    )
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    record = _run(
        execute(
            rubrics, golden, project_root=tmp_path,
            complete_fn=_fake_complete(), run_id="r_test4", started_at=time.time(),
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
                file_path=fp, entry="main", type="chat",
                metrics=[MetricSpec(name="totally_unknown_metric")],
            )
        },
    )
    golden = {"t": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    with pytest.raises(MetricNotImplementedError):
        _run(
            execute(
                rubrics, golden, project_root=tmp_path,
                complete_fn=_fake_complete(), run_id="r_test5", started_at=time.time(),
            )
        )


def test_engine_fresh_repo_no_delta(tmp_path: Path) -> None:
    fp = _write_task_module(tmp_path, "fresh", "def main(q):\n    return q\n")
    rubrics = _rubrics(tmp_path, fp)
    golden = {"chat_task": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]}
    record = _run(
        execute(
            rubrics, golden, project_root=tmp_path, baseline=None,
            complete_fn=_fake_complete(0.9), run_id="r_test6", started_at=time.time(),
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
            rubrics, golden, project_root=tmp_path, baseline=baseline,
            complete_fn=_fake_complete(0.8), run_id="r_test7", started_at=time.time(),
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
            rubrics, golden, project_root=tmp_path, parallel=1,
            complete_fn=_fake_complete(), run_id="r_test8", started_at=time.time(),
        )
    )
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    _run(
        execute(
            rubrics, golden, project_root=tmp_path, parallel=4,
            complete_fn=_fake_complete(), run_id="r_test9", started_at=time.time(),
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
            rubrics, golden, project_root=tmp_path,
            complete_fn=failing, run_id="r_test10", started_at=time.time(),
        )
    )
    assert record.tasks["chat_task"].examples[0].status == "error"
    assert "hallucination_rate" in record.tasks["chat_task"].examples[0].metric_errors

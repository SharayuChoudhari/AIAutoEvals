"""Tests for hybrid golden-set seeding (D6) and the seed:auto renderer notice.

The IO-coupled auto-seed branch was removed (AGENTS.md §1: the ``_Stub`` path
is gone; IO-coupled entry points require ``ai-evals bootstrap`` to capture a
real trace). All top-level tasks now get the pure-LLM shape variants.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_eval.config.schema import JudgeConfig, RubricsConfig, TaskSpec
from ai_eval.scaffold.golden_writer import has_real_captures, write_stub
from ai_eval.scaffold.seeder import SEED_AUTO_MARKER, seed_golden_set


def _rubrics(tasks: dict[str, TaskSpec]) -> RubricsConfig:
    return RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        tasks=tasks,
    )


def test_pure_llm_task_gets_shape_variants(tmp_path: Path) -> None:
    rubrics = _rubrics(
        {
            "summarize": TaskSpec(file_path="s.py", entry="summarize", type="chat"),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    seeded = seed_golden_set(rubrics, golden)
    assert seeded == [("summarize", 5)]
    data = json.loads(golden.read_text(encoding="utf-8"))
    examples = data["tasks"]["summarize"]
    assert len(examples) == 5
    assert all(e["seed"] == SEED_AUTO_MARKER for e in examples)
    labels = {e["id"] for e in examples}
    assert "summarize_auto_empty" in labels
    assert "summarize_auto_unicode" in labels


def test_io_coupled_task_no_longer_auto_seeded(tmp_path: Path) -> None:
    """IO-coupled top-level tasks are no longer auto-seeded (the ``_Stub``
    path is removed). They require ``ai-evals bootstrap`` to capture a real
    trace — so a task with no real captures gets zero auto-seeds."""
    rubrics = _rubrics(
        {
            "svc_process": TaskSpec(file_path="svc.py", entry="Svc.process", type="chat"),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    seeded = seed_golden_set(rubrics, golden)
    # IO-coupled is no longer a seeder concept — all top-level tasks get the
    # pure-LLM shape variants. The harness (D5) may still be written to disk
    # by ``ai-evals init``, but the seeder treats all top-level tasks the same.
    assert seeded == [("svc_process", 5)]


def test_auto_seeds_dont_count_as_real_captures(tmp_path: Path) -> None:
    """A golden set with only auto-seeded examples must NOT be treated as
    having real captures — so re-running init refreshes rather than merges."""
    rubrics = _rubrics(
        {
            "summarize": TaskSpec(file_path="s.py", entry="summarize", type="chat"),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    seed_golden_set(rubrics, golden)
    # 5 auto-seeded examples, zero real captures.
    assert has_real_captures(golden) is False


def test_real_captures_preserved_on_reseed(tmp_path: Path) -> None:
    """Re-seeding preserves real (non-auto) captures and refreshes only the
    auto-seeds."""
    rubrics = _rubrics(
        {
            "chat_task": TaskSpec(file_path="c.py", entry="main", type="chat"),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    # Simulate a real captured example.
    data = json.loads(golden.read_text(encoding="utf-8"))
    data["tasks"]["chat_task"] = [{"id": "real_1", "input": "captured"}]
    golden.write_text(json.dumps(data), encoding="utf-8")
    assert has_real_captures(golden) is True

    seed_golden_set(rubrics, golden)
    data = json.loads(golden.read_text(encoding="utf-8"))
    examples = data["tasks"]["chat_task"]
    # 1 real capture + 5 auto-seeds.
    real = [e for e in examples if e.get("seed") != SEED_AUTO_MARKER]
    auto = [e for e in examples if e.get("seed") == SEED_AUTO_MARKER]
    assert len(real) == 1
    assert real[0]["id"] == "real_1"
    assert len(auto) == 5
    # has_real_captures still true after re-seed.
    assert has_real_captures(golden) is True


def test_reseed_replaces_prior_auto_seeds(tmp_path: Path) -> None:
    """A second seed call replaces prior auto-seeds (doesn't accumulate)."""
    rubrics = _rubrics(
        {
            "summarize": TaskSpec(file_path="s.py", entry="summarize", type="chat"),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    seed_golden_set(rubrics, golden)
    seed_golden_set(rubrics, golden)
    data = json.loads(golden.read_text(encoding="utf-8"))
    assert len(data["tasks"]["summarize"]) == 5  # not 10


def test_renderer_shows_auto_seed_notice(tmp_path: Path) -> None:
    """The human run render emits the one-line auto-seed notice when a task's
    examples are all ``seed: auto``."""
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
            "svc_process": TaskRecord(
                examples=[ExampleRecord(id="ex", status="pass", seed="auto")],
                metrics={
                    "hallucination_rate": MetricResult(name="hallucination_rate", status="skip")
                },
            ),
        },
    )
    import io

    from rich.console import Console

    buf = io.StringIO()
    import ai_eval.cli.render.tables as t

    orig = t.stdout_console
    t.stdout_console = lambda *, no_color: Console(file=buf, force_terminal=False, no_color=True)
    try:
        render_run(record, no_color=True)
    finally:
        t.stdout_console = orig
    out = buf.getvalue()
    assert "auto-seeded fixtures" in out
    assert "bootstrap" in out

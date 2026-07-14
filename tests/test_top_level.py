"""Tests for the top_level task marker and its effect on seeding/running.

Covers AGENTS.md §1 (target the complete job): non-top-level tasks (private
methods, internal DAOs/services) are scaffolded into rubrics.yaml but skipped
by auto-seed and run, with a notice directing the user to ``ai-evals bootstrap``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_eval.config.schema import JudgeConfig, RubricsConfig, TaskSpec
from ai_eval.scaffold.golden_writer import write_stub
from ai_eval.scaffold.seeder import seed_golden_set


def _rubrics(tasks: dict[str, TaskSpec]) -> RubricsConfig:
    return RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        tasks=tasks,
    )


def test_top_level_defaults_true() -> None:
    """A hand-written rubrics.yaml without the field loads with top_level=True."""
    spec = TaskSpec(file_path="s.py", entry="main", type="chat")
    assert spec.top_level is True


def test_seeder_skips_non_top_level_tasks(tmp_path: Path) -> None:
    """Non-top-level tasks get zero auto-seeds; top-level tasks get the full set."""
    rubrics = _rubrics(
        {
            "top_fn": TaskSpec(file_path="a.py", entry="main", type="chat"),
            "internal_dao": TaskSpec(
                file_path="b.py", entry="DAO.search", type="chat", top_level=False
            ),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    seeded = dict(seed_golden_set(rubrics, golden, io_coupled_tasks=set()))
    assert seeded["top_fn"] == 5
    assert seeded["internal_dao"] == 0
    data = json.loads(golden.read_text(encoding="utf-8"))
    assert len(data["tasks"]["top_fn"]) == 5
    assert data["tasks"]["internal_dao"] == []


def test_seeder_preserves_real_captures_for_non_top_level(tmp_path: Path) -> None:
    """Real captures on a non-top-level task are preserved (not wiped)."""
    rubrics = _rubrics(
        {
            "internal": TaskSpec(
                file_path="b.py", entry="DAO.search", type="chat", top_level=False
            ),
        }
    )
    golden = tmp_path / "golden_set.json"
    write_stub(rubrics, golden)
    data = json.loads(golden.read_text(encoding="utf-8"))
    data["tasks"]["internal"] = [{"id": "real_1", "input": "captured"}]
    golden.write_text(json.dumps(data), encoding="utf-8")

    seed_golden_set(rubrics, golden, io_coupled_tasks=set())
    data = json.loads(golden.read_text(encoding="utf-8"))
    # Real capture preserved, no auto-seeds added.
    assert len(data["tasks"]["internal"]) == 1
    assert data["tasks"]["internal"][0]["id"] == "real_1"


def test_private_method_entry_is_non_top_level() -> None:
    """The synthesize layer marks private (_-prefixed) methods as non-top-level."""
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask
    from ai_eval.inference.synthesize import build_rubrics

    scan = ScanResult(
        files_scanned=1,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="Svc.public_method",
                framework="openai",
                type="chat",
                file_path="svc.py",
                entry="Svc.public_method",
            ),
            DetectedTask(
                name="Svc._private_method",
                framework="openai",
                type="chat",
                file_path="svc.py",
                entry="Svc._private_method",
            ),
        ],
        frameworks_seen={"openai"},
    )
    rubrics = build_rubrics(scan)
    assert rubrics.tasks["svc_public_method"].top_level is True
    assert rubrics.tasks["svc_private_method"].top_level is False


def test_bare_function_entry_is_top_level() -> None:
    """Module-level functions (no dot) are always top-level."""
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask
    from ai_eval.inference.synthesize import build_rubrics

    scan = ScanResult(
        files_scanned=1,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="summarize",
                framework="openai",
                type="chat",
                file_path="s.py",
                entry="summarize",
            ),
        ],
        frameworks_seen={"openai"},
    )
    rubrics = build_rubrics(scan)
    assert rubrics.tasks["summarize"].top_level is True


def test_runner_skips_non_top_level_task(tmp_path: Path) -> None:
    """A non-top-level task produces a task-level notice and no examples,
    and does NOT import the entry module."""
    import asyncio
    import time

    from ai_eval.runner.engine import execute

    rubrics = _rubrics(
        {
            "internal": TaskSpec(
                file_path="b.py", entry="DAO.search", type="chat", top_level=False
            ),
        }
    )
    golden = {"internal": [{"id": "e1", "input": "", "seed": "auto"}]}

    # Write the task module with an import sentinel — if the runner imports it,
    # a marker file appears. The skip path must NOT import it.
    (tmp_path / "b.py").write_text(
        "class DAO:\n    def search(self, q):\n        return q\n",
        encoding="utf-8",
    )

    record = asyncio.run(
        execute(
            rubrics,
            golden,
            project_root=tmp_path,
            parallel=1,
            complete_fn=None,
            run_id="r",
            started_at=time.time(),
        )
    )
    task = record.tasks["internal"]
    assert len(task.examples) == 0
    assert any("skipped" in e for e in task.errors)
    assert any("bootstrap" in e for e in task.errors)

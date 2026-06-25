"""Unit tests for golden_writer safety contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_eval.scaffold.golden_writer import has_real_captures, write_stub


def _make_rubrics(task_names: list[str]):
    """Return a minimal RubricsConfig-like object with the given task names."""
    from ai_eval.config.schema import JudgeConfig, RubricsConfig, TaskSpec

    tasks = {
        name: TaskSpec(file_path="src/a.py", type="chat")
        for name in task_names
    }
    return RubricsConfig(
        judge=JudgeConfig(default="ollama/qwen2.5-coder:7b"),
        tasks=tasks,
    )


def test_write_stub_creates_file_when_absent(tmp_path: Path) -> None:
    rubrics = _make_rubrics(["task_a"])
    path = tmp_path / "golden_set.json"
    status = write_stub(rubrics, path)
    assert status == "wrote"
    data = json.loads(path.read_text())
    assert data["tasks"]["task_a"] == []


def test_write_stub_refreshes_empty_existing(tmp_path: Path) -> None:
    rubrics = _make_rubrics(["task_a", "task_b"])
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps({"schema_version": 1, "tasks": {"task_a": []}}))
    status = write_stub(rubrics, path)
    assert status == "refreshed"
    data = json.loads(path.read_text())
    assert "task_b" in data["tasks"]


def test_write_stub_merges_when_captures_exist(tmp_path: Path) -> None:
    rubrics = _make_rubrics(["task_a", "task_b"])
    path = tmp_path / "golden_set.json"
    captures = {"schema_version": 1, "tasks": {"task_a": [{"id": "gs_1", "input": {}}]}}
    path.write_text(json.dumps(captures))

    status = write_stub(rubrics, path)
    assert status == "merged"
    data = json.loads(path.read_text())
    # Existing captures must survive.
    assert len(data["tasks"]["task_a"]) == 1
    # New task key merged in.
    assert data["tasks"]["task_b"] == []


def test_write_stub_overwrite_false_preserves_existing_tasks(tmp_path: Path) -> None:
    """overwrite=False must NOT remove old task keys even if rubrics no longer has them."""
    rubrics = _make_rubrics(["task_b"])  # task_a removed from rubrics
    path = tmp_path / "golden_set.json"
    captures = {"schema_version": 1, "tasks": {"task_a": [{"id": "gs_1", "input": {}}]}}
    path.write_text(json.dumps(captures))

    write_stub(rubrics, path, overwrite=False)
    data = json.loads(path.read_text())
    assert "task_a" in data["tasks"], "existing task with captures must not be dropped"
    assert "task_b" in data["tasks"]


def test_write_stub_overwrite_true_discards_captures(tmp_path: Path) -> None:
    rubrics = _make_rubrics(["task_a"])
    path = tmp_path / "golden_set.json"
    captures = {"schema_version": 1, "tasks": {"task_a": [{"id": "gs_1", "input": {}}]}}
    path.write_text(json.dumps(captures))

    status = write_stub(rubrics, path, overwrite=True)
    assert status == "overwrote"
    data = json.loads(path.read_text())
    assert data["tasks"]["task_a"] == []


def test_has_real_captures_returns_false_for_absent(tmp_path: Path) -> None:
    assert has_real_captures(tmp_path / "nonexistent.json") is False


def test_has_real_captures_returns_false_for_empty_tasks(tmp_path: Path) -> None:
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps({"schema_version": 1, "tasks": {"t": []}}))
    assert has_real_captures(path) is False


def test_has_real_captures_returns_true_when_populated(tmp_path: Path) -> None:
    path = tmp_path / "golden_set.json"
    path.write_text(json.dumps({"schema_version": 1, "tasks": {"t": [{"id": "x"}]}}))
    assert has_real_captures(path) is True

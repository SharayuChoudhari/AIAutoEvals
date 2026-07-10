"""Unit tests for the opt-in hints loader + merge with scan results."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.hints import merge_hints, parse_hints


def _scan(tasks: list[DetectedTask]) -> ScanResult:
    return ScanResult(
        files_scanned=1,
        elapsed_seconds=0.0,
        tasks=tasks,
        frameworks_seen={t.framework for t in tasks} or {"openai"},
    )


def _write_hints(path: Path, tasks: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"tasks": tasks}), encoding="utf-8")


def test_parse_hints_absent_file_is_empty(tmp_path: Path) -> None:
    hints = parse_hints(tmp_path / "missing.yaml")
    assert hints.tasks == []


def test_parse_hints_empty_file_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert parse_hints(path).tasks == []


def test_parse_hints_loads_valid(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {
            "name": "book_appointment",
            "file_path": "services/wf.py",
            "entry": "Wf.run",
            "type": "workflow",
            "inputs": ["message_text"],
            "outputs": ["reply_text"],
        }
    ])
    hints = parse_hints(path)
    assert len(hints.tasks) == 1
    h = hints.tasks[0]
    assert h.name == "book_appointment"
    assert h.type == "workflow"
    assert h.entry == "Wf.run"
    assert h.inputs == ["message_text"]


def test_parse_hints_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_hints(path)


def test_parse_hints_rejects_bad_name(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    _write_hints(path, [{"name": "bad-name!", "file_path": "f.py", "type": "workflow"}])
    with pytest.raises(ValidationError):
        parse_hints(path)


def test_parse_hints_rejects_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "ok", "file_path": "f.py", "type": "workflow", "unknown": 1}
    ])
    with pytest.raises(ValidationError):
        parse_hints(path)


def test_merge_hints_absent_file_returns_scan_unchanged(tmp_path: Path) -> None:
    scan = _scan([DetectedTask(name="a", framework="openai", type="chat", file_path="a.py")])
    out = merge_hints(scan, tmp_path / "missing.yaml")
    assert out.tasks == scan.tasks


def test_merge_hints_empty_tasks_returns_scan_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    _write_hints(path, [])
    scan = _scan([DetectedTask(name="a", framework="openai", type="chat", file_path="a.py")])
    out = merge_hints(scan, path)
    assert out.tasks == scan.tasks


def test_merge_hints_fills_gap_when_no_ast_match(tmp_path: Path) -> None:
    """A hint with a file_path/entry that no AST detector found is appended."""
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "temporal_workflow", "file_path": "wf/run.py", "entry": "run", "type": "workflow"}
    ])
    scan = _scan([
        DetectedTask(
            name="chat_a", framework="openai", type="chat",
            file_path="other.py", entry="other",
        )
    ])
    out = merge_hints(scan, path)
    assert len(out.tasks) == 2
    hint_task = next(t for t in out.tasks if t.framework == "hint")
    assert hint_task.name == "temporal_workflow"
    assert hint_task.type == "workflow"
    assert hint_task.file_path == "wf/run.py"
    assert hint_task.entry == "run"
    assert any("hint task" in e for e in hint_task.evidence)


def test_merge_hints_ast_wins_on_collision(tmp_path: Path) -> None:
    """When a hint and an AST task share (file_path, entry), the AST task
    wins — the hint is dropped, not merged, so we never get two tasks for
    the same entry point."""
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "hint_run", "file_path": "wf.py", "entry": "run", "type": "workflow"}
    ])
    scan = _scan([
        DetectedTask(
            name="ast_run", framework="openai", type="workflow",
            file_path="wf.py", entry="run",
        )
    ])
    out = merge_hints(scan, path)
    # Only the AST task survives.
    assert len(out.tasks) == 1
    assert out.tasks[0].name == "ast_run"
    assert out.tasks[0].framework == "openai"


def test_merge_hints_same_file_different_entries_both_kept(tmp_path: Path) -> None:
    """Two hints on the same file but different entries don't collide."""
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "book", "file_path": "wf.py", "entry": "book_intent", "type": "workflow"},
        {"name": "cancel", "file_path": "wf.py", "entry": "cancel_intent", "type": "workflow"},
    ])
    scan = _scan([])
    out = merge_hints(scan, path)
    assert len(out.tasks) == 2
    assert {t.name for t in out.tasks} == {"book", "cancel"}


def test_merge_hints_per_intent_split_on_one_entry(tmp_path: Path) -> None:
    """The plan's split lever: multiple hints with the same file_path/entry
    but distinct names all stay as separate tasks (the AST-wins rule only
    applies to AST-vs-hint collisions, not hint-vs-hint)."""
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "book_appointment", "file_path": "wf.py", "entry": "run", "type": "workflow"},
        {"name": "cancel_appointment", "file_path": "wf.py", "entry": "run", "type": "workflow"},
    ])
    scan = _scan([])
    out = merge_hints(scan, path)
    assert len(out.tasks) == 2
    assert {t.name for t in out.tasks} == {"book_appointment", "cancel_appointment"}


def test_merge_hints_module_level_entry_none(tmp_path: Path) -> None:
    """A hint with entry=None (module-level) collides with an AST task that
    also has entry=None on the same file_path — AST still wins."""
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "hint_mod", "file_path": "mod.py", "entry": None, "type": "workflow"}
    ])
    scan = _scan([
        DetectedTask(
            name="ast_mod", framework="openai", type="workflow",
            file_path="mod.py", entry=None,
        )
    ])
    out = merge_hints(scan, path)
    assert len(out.tasks) == 1
    assert out.tasks[0].name == "ast_mod"


def test_merge_hints_preserves_scan_metadata(tmp_path: Path) -> None:
    """The merged ScanResult reuses the input's files_scanned/elapsed."""
    path = tmp_path / "h.yaml"
    _write_hints(path, [
        {"name": "h", "file_path": "x.py", "entry": "f", "type": "workflow"}
    ])
    scan = ScanResult(
        files_scanned=42,
        elapsed_seconds=1.5,
        tasks=[],
        frameworks_seen=set(),
    )
    out = merge_hints(scan, path)
    assert out.files_scanned == 42
    assert out.elapsed_seconds == 1.5
    assert len(out.tasks) == 1

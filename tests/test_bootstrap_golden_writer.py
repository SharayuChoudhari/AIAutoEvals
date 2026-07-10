"""Tests for the bootstrap golden_writer: append/replace, safe-merge, max cap."""

from __future__ import annotations

import json
from pathlib import Path

from ai_eval.bootstrap.golden_writer import append, has_real_captures


def _ex(task: str, q: str) -> dict:
    return {"task": task, "input": {"query": q}, "expected": None, "trace": {"calls": []}}


def test_append_creates_new_file(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    summary = append(p, [_ex("t1", "a"), _ex("t1", "b")])
    assert summary["written"] == 2
    data = json.loads(p.read_text())
    assert set(data["tasks"].keys()) == {"t1"}
    assert len(data["tasks"]["t1"]) == 2


def test_append_merges_into_existing(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    append(p, [_ex("t1", "a")])
    append(p, [_ex("t1", "b"), _ex("t2", "z")])
    data = json.loads(p.read_text())
    assert len(data["tasks"]["t1"]) == 2
    assert len(data["tasks"]["t2"]) == 1


def test_append_dedups_by_input(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    append(p, [_ex("t1", "a")])
    summary = append(p, [_ex("t1", "a")])  # duplicate input
    assert summary["written"] == 0
    assert summary["skipped"] == 1
    data = json.loads(p.read_text())
    assert len(data["tasks"]["t1"]) == 1


def test_replace_mode_clears_task(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    append(p, [_ex("t1", "old")])
    # replace all
    append(p, [_ex("t1", "new")], append_mode=False)
    data = json.loads(p.read_text())
    assert [ex["input"]["query"] for ex in data["tasks"]["t1"]] == ["new"]


def test_replace_with_task_filter(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    append(p, [_ex("t1", "a"), _ex("t2", "b")])
    append(p, [_ex("t1", "fresh")], append_mode=False, task_filter="t1")
    data = json.loads(p.read_text())
    # t1 replaced; t2 untouched
    assert [ex["input"]["query"] for ex in data["tasks"]["t1"]] == ["fresh"]
    assert [ex["input"]["query"] for ex in data["tasks"]["t2"]] == ["b"]


def test_max_per_task_drops_oldest(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    examples = [_ex("t1", f"q{i}") for i in range(5)]
    append(p, examples, max_per_task=2)
    data = json.loads(p.read_text())
    assert len(data["tasks"]["t1"]) == 2
    # oldest dropped -> keeps the last two
    assert [ex["input"]["query"] for ex in data["tasks"]["t1"]] == ["q3", "q4"]


def test_task_filter_skips_other_tasks(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    append(p, [_ex("t1", "a"), _ex("t2", "b")], task_filter="t1")
    data = json.loads(p.read_text())
    assert "t2" not in data["tasks"]
    assert len(data["tasks"]["t1"]) == 1


def test_has_real_captures(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    assert not has_real_captures(p)
    append(p, [_ex("t1", "a")])
    assert has_real_captures(p)


def test_corrupt_file_is_replaced(tmp_path: Path) -> None:
    p = tmp_path / "golden_set.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("not json", encoding="utf-8")
    append(p, [_ex("t1", "a")])
    data = json.loads(p.read_text())
    assert len(data["tasks"]["t1"]) == 1

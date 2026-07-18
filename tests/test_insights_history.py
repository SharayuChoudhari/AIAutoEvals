"""Tests for insights.history: append/list/load/prune/export + advisory lock."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_eval.insights.history import (
    append_history,
    export,
    last_run_id,
    list_history,
    load_run,
    prune,
)
from ai_eval.runner.record import GitMeta, RunRecord, RunSummary
from ai_eval.storage.runs import save_run


def _record(rid: str) -> RunRecord:
    return RunRecord(
        id=rid,
        started_at=1000.0,
        finished_at=1001.0,
        git=GitMeta(sha="abc", branch="main", dirty=False),
        summary=RunSummary(passed=1, failed=0, errors=0, examples=1),
        tags=["t"],
    )


def test_append_then_list(tmp_path: Path) -> None:
    save_run(tmp_path, _record("r_a"))
    append_history(tmp_path, _record("r_a"))
    runs = list_history(tmp_path)
    assert len(runs) == 1
    assert runs[0]["id"] == "r_a"


def test_last_run_id(tmp_path: Path) -> None:
    save_run(tmp_path, _record("r_a"))
    append_history(tmp_path, _record("r_a"))
    save_run(tmp_path, _record("r_b"))
    append_history(tmp_path, _record("r_b"))
    assert last_run_id(tmp_path) == "r_b"


def test_load_run_last(tmp_path: Path) -> None:
    save_run(tmp_path, _record("r_a"))
    append_history(tmp_path, _record("r_a"))
    rec = load_run(tmp_path, "last")
    assert rec.id == "r_a"


def test_load_run_by_id(tmp_path: Path) -> None:
    save_run(tmp_path, _record("r_a"))
    append_history(tmp_path, _record("r_a"))
    rec = load_run(tmp_path, "r_a")
    assert rec.id == "r_a"


def test_load_run_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path, "nope")


def test_prune_keeps_recent(tmp_path: Path) -> None:
    for rid in ["r1", "r2", "r3"]:
        save_run(tmp_path, _record(rid))
        append_history(tmp_path, _record(rid))
    removed = prune(tmp_path, keep=1)
    assert removed == 2
    runs = list_history(tmp_path)
    assert len(runs) == 1
    assert runs[0]["id"] == "r3"


def test_export(tmp_path: Path) -> None:
    save_run(tmp_path, _record("r_a"))
    append_history(tmp_path, _record("r_a"))
    out = export(tmp_path, tmp_path / "export.json")
    data = json.loads(out.read_text())
    assert "runs" in data
    assert len(data["runs"]) == 1


def test_history_missing_returns_empty(tmp_path: Path) -> None:
    assert list_history(tmp_path) == []
    assert last_run_id(tmp_path) is None


def test_history_slim_record_shape(tmp_path: Path) -> None:
    save_run(tmp_path, _record("r_a"))
    append_history(tmp_path, _record("r_a"))
    runs = list_history(tmp_path)
    assert "id" in runs[0]
    assert "summary" in runs[0]
    assert "git_sha" in runs[0]
    assert runs[0]["git_sha"] == "abc"

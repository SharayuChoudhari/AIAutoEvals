"""Run history (``.ai-evals/history.json``).

A list of slim records: ``{id, started_at, summary, git.sha, tags}``. Full run
records live under ``.ai-evals/runs/<id>/``. Writes are guarded by an advisory
lock (:mod:`ai_eval.storage.locks`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_eval.config.defaults import SCHEMA_VERSION
from ai_eval.runner.record import RunRecord
from ai_eval.storage.locks import advisory_lock
from ai_eval.storage.runs import load_run as _load_run_record


def _history_path(state_dir: Path) -> Path:
    return state_dir / "history.json"


def _load(state_dir: Path) -> list[dict[str, Any]]:
    p = _history_path(state_dir)
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict) and isinstance(data.get("runs"), list):
        return data["runs"]
    if isinstance(data, list):
        return data
    return []


def _save(state_dir: Path, runs: list[dict[str, Any]]) -> None:
    p = _history_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    with advisory_lock(state_dir / "config.lock"):
        p.write_text(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "runs": runs},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


def _slim(record: RunRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "summary": record.summary.model_dump(mode="json"),
        "git_sha": record.git.sha if record.git else None,
        "tags": list(record.tags),
    }


def append_history(state_dir: Path, record: RunRecord) -> None:
    """Append a slim record for ``record`` to history (advisory-locked)."""
    runs = _load(state_dir)
    runs.append(_slim(record))
    _save(state_dir, runs)


def list_history(state_dir: Path) -> list[dict[str, Any]]:
    """Return all slim history records, newest last."""
    return _load(state_dir)


def last_run_id(state_dir: Path) -> str | None:
    runs = _load(state_dir)
    return runs[-1]["id"] if runs else None


def load_run(state_dir: Path, run_id_or_last: str) -> RunRecord:
    """Load a full run record. ``"last"`` resolves to the most recent."""
    if run_id_or_last == "last":
        rid = last_run_id(state_dir)
        if rid is None:
            raise FileNotFoundError("no runs in history")
    else:
        rid = run_id_or_last
    return _load_run_record(state_dir, rid)


def prune(state_dir: Path, *, keep: int) -> int:
    """Keep only the most recent ``keep`` runs; return the count removed."""
    runs = _load(state_dir)
    if len(runs) <= keep:
        return 0
    removed = runs[:-keep] if keep > 0 else runs
    kept = runs[-keep:] if keep > 0 else []
    _save(state_dir, kept)
    # best-effort: delete run dirs of removed runs
    for r in removed:
        rid = r.get("id")
        if not rid:
            continue
        rd = state_dir / "runs" / rid
        if rd.is_dir():
            import shutil
            shutil.rmtree(rd, ignore_errors=True)
    return len(removed)


def export(state_dir: Path, out_path: Path) -> Path:
    """Write the full history JSON to ``out_path``; return the path."""
    runs = _load(state_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "runs": runs},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return out_path


__all__ = [
    "append_history",
    "export",
    "last_run_id",
    "list_history",
    "load_run",
    "prune",
]

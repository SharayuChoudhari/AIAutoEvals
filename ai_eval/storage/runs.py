"""Run persistence: id generation, save/load run records + traces + judge log.

Layout under ``.ai-eval/runs/<id>/``:
- ``run.json`` — the :class:`~ai_eval.runner.record.RunRecord`
- ``traces.ndjson`` — per-example traces (one JSON object per line)
- ``judge_log.ndjson`` — judge call log (request/response/errors)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ai_eval.runner.record import RunRecord


def new_run_id() -> str:
    """Return a short, unique run id ``r_<sha5>``."""
    h = hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()
    return f"r_{h[:5]}"


def _run_dir(state_dir: Path, run_id: str) -> Path:
    return state_dir / "runs" / run_id


def save_run(
    state_dir: Path,
    record: RunRecord,
    *,
    traces: list[dict[str, Any]] | None = None,
    judge_log: list[dict[str, Any]] | None = None,
) -> Path:
    """Write the run record + optional traces/judge log; return the run dir."""
    rd = _run_dir(state_dir, record.id)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "run.json").write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if traces:
        _write_ndjson(rd / "traces.ndjson", traces)
    if judge_log:
        _write_ndjson(rd / "judge_log.ndjson", judge_log)
    return rd


def _write_ndjson(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, default=str, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def load_run(state_dir: Path, run_id: str) -> RunRecord:
    """Load a run record by id. Raises ``FileNotFoundError`` if absent."""
    path = _run_dir(state_dir, run_id) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"run {run_id!r} not found at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunRecord.model_validate(data)


def load_run_raw(state_dir: Path, run_id: str) -> dict[str, Any]:
    """Load a run record as a raw dict (for diff/baseline lookups)."""
    path = _run_dir(state_dir, run_id) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(f"run {run_id!r} not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["load_run", "load_run_raw", "new_run_id", "save_run"]

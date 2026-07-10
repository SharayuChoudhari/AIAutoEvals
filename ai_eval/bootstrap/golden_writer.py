"""Golden-set append/replace writer for the bootstrapper.

Reuses :mod:`ai_eval.scaffold.golden_writer`'s safe-merge contract:
populated captures are never overwritten without an explicit ``--replace``
(``append=False``). New task keys merge in non-destructively; per-task arrays
are capped by ``--max`` and deduped by input hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_eval.config.defaults import SCHEMA_VERSION
from ai_eval.scaffold.golden_writer import GoldenSetOverwriteError, has_real_captures


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "tasks": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": SCHEMA_VERSION, "tasks": {}}
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), dict):
        return {"schema_version": SCHEMA_VERSION, "tasks": {}}
    return data


def _input_hash(example: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(example.get("input"), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def append(
    path: Path,
    examples: list[dict[str, Any]],
    *,
    append_mode: bool = True,
    max_per_task: int | None = None,
    task_filter: str | None = None,
) -> dict[str, Any]:
    """Merge ``examples`` into the golden set at ``path``.

    Args:
        path: golden_set.json location.
        examples: list of golden-example dicts (each has ``task``, ``input``...).
        append_mode: True → merge (default, ``--append``); False → replace
            (``--replace``). Replace refuses to discard existing real captures
            unless the caller explicitly asked (here ``append_mode=False`` IS
            the explicit ask, so replace proceeds).
        max_per_task: cap per-task array length (drops oldest beyond the cap).
        task_filter: only this task name is written; others untouched.

    Returns a summary dict ``{written, skipped, tasks: {name: count}}``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load(path)
    tasks: dict[str, list[dict[str, Any]]] = existing["tasks"]

    # Replace mode: clear the target task(s) first.
    if not append_mode:
        if task_filter is not None:
            tasks[task_filter] = []
        else:
            tasks = {}

    summary: dict[str, Any] = {"written": 0, "skipped": 0, "tasks": {}}

    by_task: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        tname = ex.get("task") or "unknown"
        if task_filter is not None and tname != task_filter:
            continue
        by_task.setdefault(tname, []).append(ex)

    for tname, exs in by_task.items():
        bucket = tasks.setdefault(tname, [])
        seen = {_input_hash(e) for e in bucket}
        for ex in exs:
            h = _input_hash(ex)
            if h in seen:
                summary["skipped"] += 1
                continue
            seen.add(h)
            bucket.append(ex)
            summary["written"] += 1
        if max_per_task is not None and len(bucket) > max_per_task:
            # drop oldest beyond the cap
            dropped = len(bucket) - max_per_task
            del bucket[:dropped]
            summary["skipped"] += dropped
        summary["tasks"][tname] = len(bucket)

    out = {"schema_version": SCHEMA_VERSION, "tasks": tasks}
    path.write_text(
        json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


__all__ = ["GoldenSetOverwriteError", "append", "has_real_captures"]

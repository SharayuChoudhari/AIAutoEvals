"""Initial-state writers for `eval/golden_set.json`.

`golden_set.json` is **captured user data**, not regenerable scaffold (see
plan §2.3). Writers in this module therefore refuse to overwrite a populated
golden set unless the caller passes an explicit ``overwrite=True``. New task
keys are merged in non-destructively by default.
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_eval.config.defaults import SCHEMA_VERSION
from ai_eval.config.schema import RubricsConfig


def empty_stub(rubrics: RubricsConfig) -> dict:
    """An empty golden set keyed by task — to be filled by `ai-evals bootstrap`."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tasks": {name: [] for name in rubrics.tasks},
    }


def _load_existing(path: Path) -> dict | None:
    """Return the parsed golden set at ``path`` or ``None`` if absent/unparseable."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def has_real_captures(path: Path) -> bool:
    """True if the on-disk golden set contains at least one non-auto-seeded
    capture. Auto-seeded examples (D6, marked ``seed: auto``) don't count as
    real captures — they're regenerable scaffold, not user/captured data, so
    they must not flip the init merge/refresh decision.
    """
    data = _load_existing(path)
    if data is None:
        return False
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        return False
    for v in tasks.values():
        if not isinstance(v, list):
            continue
        for ex in v:
            # A real capture is any non-auto-seeded example.
            if isinstance(ex, dict) and ex.get("seed") != "auto":
                return True
    return False


class GoldenSetOverwriteError(RuntimeError):
    """Raised when a write would destroy populated captures and ``overwrite`` is False."""


def write_stub(
    rubrics: RubricsConfig,
    path: Path,
    *,
    overwrite: bool = False,
) -> str:
    """Write or merge the empty stub at ``path``.

    Contract:
      - If ``path`` does not exist: write the empty stub. Returns ``"wrote"``.
      - If ``path`` exists but contains **no real captures** (every task list is
        empty / missing): rewrite it so new task keys appear. Returns ``"refreshed"``.
      - If ``path`` exists and contains **real captures**:
          - When ``overwrite=False`` (default): **merge** — add stub entries for
            task names that are missing, but leave existing capture arrays
            untouched. Returns ``"merged"``.
          - When ``overwrite=True``: replace with the empty stub, discarding
            captures. Returns ``"overwrote"``.

    The default path is always safe: populated captures are preserved unless the
    caller explicitly opts in with ``overwrite=True``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_stub = empty_stub(rubrics)

    existing = _load_existing(path)
    if existing is None:
        path.write_text(
            json.dumps(new_stub, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return "wrote"

    if not has_real_captures(path):
        path.write_text(
            json.dumps(new_stub, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return "refreshed"

    if overwrite:
        path.write_text(
            json.dumps(new_stub, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return "overwrote"

    # Merge: keep existing capture arrays; add empty lists for any new task names.
    raw_tasks = existing.get("tasks")
    existing_tasks: dict[str, list] = raw_tasks if isinstance(raw_tasks, dict) else {}
    merged_tasks: dict[str, list] = {}
    # Preserve every existing task (including ones removed from rubrics, so the
    # user can re-add them or prune via `ai-evals history`).
    for name, captures in existing_tasks.items():
        merged_tasks[name] = captures if isinstance(captures, list) else []
    for name in rubrics.tasks:
        merged_tasks.setdefault(name, [])

    merged = {
        "schema_version": SCHEMA_VERSION,
        "tasks": merged_tasks,
    }
    # Only write if the merge changed anything; avoid touching mtime / git status.
    if merged != existing:
        path.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return "merged"


__all__ = [
    "GoldenSetOverwriteError",
    "empty_stub",
    "has_real_captures",
    "write_stub",
]

"""Hybrid golden-set seeding (D6).

Pure-LLM tasks get 3-5 shape-varied auto-seeded inputs (empty, short, long,
unicode, boundary) so ``ai-evals run`` produces meaningful scores immediately —
zero authoring, zero environment. IO-coupled tasks get the stub harness (D5)
plus 1 shape-valid example flagged ``seed: auto`` so the pipeline is green
(non-error, non-empty); the run prints a one-line notice directing the user to
``ai-evals bootstrap`` for trustworthy regression baselines.

Auto-seeded examples carry a ``"seed": "auto"`` marker so
:func:`ai_eval.scaffold.golden_writer.has_real_captures` can distinguish them
from real captured traces (auto-seeds don't count as "real captures" for the
init merge/refresh decision).
"""

from __future__ import annotations

import json
from pathlib import Path

from ai_eval.config.schema import RubricsConfig

#: Shape-varied input variants for pure-LLM auto-seeding (plan D6 default).
_PURE_LLM_VARIANTS: list[tuple[str, object]] = [
    ("empty", ""),
    ("short", "hi"),
    ("long", "Please summarize the following text in detail. " * 20),
    ("unicode", "héllo wörld — 日本語 🚀"),
    ("boundary", " "),
]

#: The marker auto-seeded examples carry so they're distinguishable from real
#: captures. ``has_real_captures`` ignores examples with this marker.
SEED_AUTO_MARKER = "auto"


def _is_auto_seeded(example: dict) -> bool:
    return example.get("seed") == SEED_AUTO_MARKER


def _pure_llm_examples(task_name: str) -> list[dict]:
    return [
        {"id": f"{task_name}_auto_{label}", "input": value, "seed": SEED_AUTO_MARKER}
        for label, value in _PURE_LLM_VARIANTS
    ]


def _io_coupled_example(task_name: str) -> list[dict]:
    return [
        {
            "id": f"{task_name}_auto",
            "input": "",
            "seed": SEED_AUTO_MARKER,
        }
    ]


def seed_for_task(task_name: str, *, io_coupled: bool) -> list[dict]:
    """Return the auto-seed examples for one task.

    Pure-LLM tasks get the shape-varied set; IO-coupled tasks get a single
    green-pipeline example (the harness supplies the canned IO).
    """
    if io_coupled:
        return _io_coupled_example(task_name)
    return _pure_llm_examples(task_name)


def _load_existing(path: Path) -> dict:
    if not path.is_file():
        return {"schema_version": 1, "tasks": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "tasks": {}}
    if not isinstance(data, dict) or not isinstance(data.get("tasks"), dict):
        return {"schema_version": 1, "tasks": {}}
    return data


def _strip_auto_seeds(examples: list) -> list:
    """Remove existing auto-seeded examples so re-seeding refreshes them."""
    if not isinstance(examples, list):
        return []
    return [e for e in examples if isinstance(e, dict) and not _is_auto_seeded(e)]


def seed_golden_set(
    rubrics: RubricsConfig,
    path: Path,
    io_coupled_tasks: set[str],
) -> list[tuple[str, int]]:
    """Auto-seed ``golden_set.json`` for every task in ``rubrics``.

    Runs after :func:`golden_writer.write_stub` so the task keys exist. For each
    task: if the task already has real (non-auto) captures, preserve them and
    refresh only the auto-seeds (re-seeding replaces prior auto-seeds). If the
    task has no real captures, fill it with the auto-seed set.

    ``io_coupled_tasks`` — task names classified IO-coupled (D4) get a single
    green-pipeline example; the rest get the pure-LLM shape variants.

    Returns ``[(task_name, num_seeded)]``.
    """
    data = _load_existing(path)
    tasks: dict[str, list] = data.get("tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}
    seeded: list[tuple[str, int]] = []
    for name in rubrics.tasks:
        existing = tasks.get(name, [])
        if not isinstance(existing, list):
            existing = []
        real = _strip_auto_seeds(existing)
        auto = seed_for_task(name, io_coupled=name in io_coupled_tasks)
        tasks[name] = real + auto
        seeded.append((name, len(auto)))
    out = {"schema_version": data.get("schema_version", 1), "tasks": tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seeded


__all__ = ["SEED_AUTO_MARKER", "seed_for_task", "seed_golden_set"]

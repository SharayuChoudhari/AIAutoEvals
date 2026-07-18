"""Hybrid golden-set seeding (D6).

Pure-LLM top-level tasks get 3-5 shape-varied auto-seeded inputs (empty,
short, long, unicode, boundary) so ``ai-evals run`` produces meaningful scores
immediately — zero authoring, zero environment. IO-coupled tasks no longer get
an auto-seeded example (the ``_Stub`` path is removed — AGENTS.md §1): they
require ``ai-evals bootstrap`` to capture a real trace before ``run`` can score
their nodes.

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


def seed_for_task(task_name: str) -> list[dict]:
    """Return the auto-seed examples for one task.

    All top-level tasks get the pure-LLM shape-varied set (the only runnable
    kind now — AGENTS.md §1). IO-coupled entry points are no longer
    auto-seeded: they need ``ai-evals bootstrap`` to capture a real trace.
    """
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
) -> list[tuple[str, int]]:
    """Auto-seed ``golden_set.json`` for every top-level task in ``rubrics``.

    Runs after :func:`golden_writer.write_stub` so the task keys exist. For each
    task: if the task already has real (non-auto) captures, preserve them and
    refresh only the auto-seeds (re-seeding replaces prior auto-seeds). If the
    task has no real captures, fill it with the auto-seed set.

    **Non-top-level tasks** (``top_level=False`` — private methods, IO-coupled
    services, peer-reached survivors) are skipped entirely: their task key
    gets an empty list and the run prints a notice directing the user to
    ``ai-evals bootstrap`` (AGENTS.md §1). This is the "target the complete
    job" contract.

    Returns ``[(task_name, num_seeded)]``.
    """
    data = _load_existing(path)
    tasks: dict[str, list] = data.get("tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}
    seeded: list[tuple[str, int]] = []
    for name, tspec in rubrics.tasks.items():
        existing = tasks.get(name, [])
        if not isinstance(existing, list):
            existing = []
        real = _strip_auto_seeds(existing)
        if not tspec.top_level:
            # Non-top-level (internal) tasks: preserve real captures if any,
            # but do NOT auto-seed. The run will skip with a bootstrap notice.
            tasks[name] = real
            seeded.append((name, 0))
            continue
        auto = seed_for_task(name)
        tasks[name] = real + auto
        seeded.append((name, len(auto)))
    out = {"schema_version": data.get("schema_version", 1), "tasks": tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return seeded


__all__ = ["SEED_AUTO_MARKER", "seed_for_task", "seed_golden_set"]

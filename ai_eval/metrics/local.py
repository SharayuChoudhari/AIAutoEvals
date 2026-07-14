"""Project-local metric registry (``eval/metrics.yaml``).

Loads niche/project-specific metrics declared in the consuming project's
``eval/metrics.yaml`` and merges them into the runtime registry (see
:func:`ai_eval.metrics.registry.load_metrics` /
:func:`ai_eval.metrics.registry.load_judge_metrics`). Built-in metrics always
win on name collision; the local file can only ADD names.

Schema (``eval/metrics.yaml``)::

    metrics:
      - name: slot_filling_accuracy           # required, snake_case
        description: Accuracy of slots filled. # required
        applicable_task_types: [booking, workflow]  # optional
        scored_dimension: slot_filling_accuracy     # optional, defaults to name
        threshold: 0.9                               # optional, default 0.0
        weight: 1.0                                  # optional, default 1.0
        non_judge: false                             # optional, default false
        prompt_template: |                           # optional for non_judge
          Evaluate slot filling for {{input}} vs expected {{expected}}.
          Output: {{output}}.

``prompt_template`` uses simple ``{{name}}`` placeholder substitution
(no Jinja2). Recognized placeholders: ``{{input}}``, ``{{output}}``,
``{{expected}}``, ``{{task_type}}``, ``{{task_purpose}}``, ``{{metric_name}}``.
The template is wrapped into a ``prompt_builder`` closure returning a chat
messages list (system + user) mirroring
:func:`ai_eval.judge.prompts.cot_complex.build` output shape.

See ``docs/metrics.md`` for a copy-paste template re-declaring the six removed
niche metrics (slot_filling_accuracy, translation_bleu, classification_f1,
extraction_field_accuracy, scoring_accuracy, summary_faithfulness).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_eval.metrics.registry import JudgeMetric, Metric

#: The local metrics file path relative to the project root.
_LOCAL_METRICS_REL = Path("eval") / "metrics.yaml"


def _local_path(project_root: Path) -> Path:
    return project_root / _LOCAL_METRICS_REL


def _substitute(template: str, vars_: dict[str, str]) -> str:
    """Replace ``{{name}}`` placeholders in ``template`` with values."""
    out = template
    for key, val in vars_.items():
        out = out.replace("{{" + key + "}}", val)
    return out


def _build_prompt_builder(
    metric_name: str,
    metric_description: str,
    scored_dimension: str,
    template: str,
) -> Any:
    """Return a ``prompt_builder(task_spec, example, output) -> messages`` closure.

    Mirrors the CoT prompt shape (system + user messages). The template fills
    the user message; the system message is the standard strict-judge header.
    """

    def builder(task_spec: Any, example: dict[str, Any], output: Any) -> list[dict[str, str]]:
        task_type = getattr(task_spec, "type", "custom") or "custom"
        task_purpose = getattr(task_spec, "purpose", None) or ""
        inp = example.get("input", example) if isinstance(example, dict) else example
        expected = example.get("expected") if isinstance(example, dict) else None
        vars_: dict[str, str] = {
            "metric_name": metric_name,
            "metric_description": metric_description,
            "scored_dimension": scored_dimension,
            "task_type": str(task_type),
            "task_purpose": str(task_purpose),
            "input": json.dumps(inp, default=str, ensure_ascii=False),
            "output": json.dumps(output, default=str, ensure_ascii=False),
            "expected": json.dumps(expected, default=str, ensure_ascii=False),
        }
        system = (
            "You are a strict evaluation judge. Score the candidate output "
            f"against the criteria for `{metric_name}`. Return a JSON object "
            f"with a `score` in [0, 1], a short `rationale`, and `sub_scores` "
            f"keyed by dimension name (one dimension must be "
            f"`{scored_dimension}`). Be adversarial: only credit what the "
            "evidence supports."
        )
        user = _substitute(template, vars_)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    return builder


def _coerce_entry(raw: dict[str, Any]) -> tuple[Metric, JudgeMetric | None] | None:
    """Parse one YAML entry into a ``(Metric, JudgeMetric|None)`` pair.

    Returns ``None`` on a malformed entry (caller skips it). The ``JudgeMetric``
    is ``None`` when the entry is ``non_judge: true`` or has no
    ``prompt_template`` (name-only registration).
    """
    try:
        name = str(raw["name"])
    except (KeyError, TypeError):
        return None
    description = str(raw.get("description", ""))
    applicable = tuple(raw.get("applicable_task_types", []) or ())
    threshold = float(raw.get("threshold", raw.get("default_threshold", 0.0)))
    weight = float(raw.get("weight", raw.get("default_weight", 1.0)))
    non_judge = bool(raw.get("non_judge", False))
    scored_dimension = str(raw.get("scored_dimension", name))
    prompt_template = raw.get("prompt_template")

    metric = Metric(
        name=name,
        description=description,
        default_threshold=threshold,
        default_weight=weight,
        applicable_task_types=applicable,
    )
    judge: JudgeMetric | None = None
    if not non_judge and isinstance(prompt_template, str) and prompt_template.strip():
        judge = JudgeMetric(
            name=name,
            description=description,
            applicable_task_types=applicable,
            scored_dimension=scored_dimension,
            prompt_builder=_build_prompt_builder(
                name, description, scored_dimension, prompt_template
            ),
            default_threshold=threshold,
            default_weight=weight,
            non_judge=False,
        )
    elif not non_judge:
        # Name-only judge registration (no template) — the judge gateway will
        # treat it as unimplemented at run time (MetricNotImplementedError),
        # but schema validation passes. This mirrors the bare-string entry-point
        # plugin path. Users who want it runnable must supply a prompt_template.
        judge = JudgeMetric(
            name=name,
            description=description,
            applicable_task_types=applicable,
            scored_dimension=scored_dimension,
            prompt_builder=None,
            default_threshold=threshold,
            default_weight=weight,
            non_judge=False,
        )
    return metric, judge


def load_local_metrics(project_root: Path) -> tuple[Metric, ...]:
    """Load ``Metric`` entries from ``eval/metrics.yaml``.

    Returns an empty tuple when the file is absent, unreadable, or empty.
    Malformed entries are silently skipped (best-effort, never crashes the run).
    """
    path = _local_path(project_root)
    if not path.is_file():
        return ()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception):
        return ()
    if not isinstance(data, dict):
        return ()
    raw_metrics = data.get("metrics")
    if not isinstance(raw_metrics, list):
        return ()
    out: list[Metric] = []
    for entry in raw_metrics:
        if not isinstance(entry, dict):
            continue
        parsed = _coerce_entry(entry)
        if parsed is None:
            continue
        out.append(parsed[0])
    return tuple(out)


def load_local_judge_metrics(project_root: Path) -> tuple[JudgeMetric, ...]:
    """Load ``JudgeMetric`` entries (those with a prompt) from ``eval/metrics.yaml``.

    Returns an empty tuple when the file is absent or no entries have a
    runnable judge prompt.
    """
    path = _local_path(project_root)
    if not path.is_file():
        return ()
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, Exception):
        return ()
    if not isinstance(data, dict):
        return ()
    raw_metrics = data.get("metrics")
    if not isinstance(raw_metrics, list):
        return ()
    out: list[JudgeMetric] = []
    for entry in raw_metrics:
        if not isinstance(entry, dict):
            continue
        parsed = _coerce_entry(entry)
        if parsed is None:
            continue
        judge = parsed[1]
        if judge is not None:
            out.append(judge)
    return tuple(out)


__all__ = ["load_local_judge_metrics", "load_local_metrics"]

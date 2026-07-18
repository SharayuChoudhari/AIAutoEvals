"""Binary checklist judge prompt builder for local/small models.

Asks Yes/No per criterion and derives a ``score`` = fraction of Yes answers.
``sub_scores`` carries each criterion; the metric's ``scored_dimension`` is the
roll-up key (``"overall"`` by default). Simpler prompt → fewer tokens → works
on 7B-class local models that flake on free-form CoT.
"""

from __future__ import annotations

import json
from typing import Any


def build(
    *,
    metric_name: str,
    metric_description: str,
    scored_dimension: str,
    task_type: str,
    task_purpose: str | None,
    input: Any,
    output: Any,
    expected: Any = None,
    checks: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    """Return a chat messages list for the binary checklist tier."""
    checks = checks or (metric_description or f"output satisfies {metric_name}",)
    system = (
        "You are a strict evaluation judge using a binary checklist. For each "
        'check, answer `yes` or `no`. Return JSON: {"score": float in [0,1] '
        '(fraction of yes), "rationale": str, "sub_scores": {check: 1.0 if '
        "yes else 0.0}}. The roll-up dimension is named by `scored_dimension`."
    )
    user_lines = [
        f"metric: {metric_name}",
        f"scored_dimension: {scored_dimension}",
        f"task_type: {task_type}",
    ]
    if task_purpose:
        user_lines.append(f"task_purpose: {task_purpose}")
    user_lines.append("input: " + json.dumps(input, default=str, ensure_ascii=False))
    user_lines.append("output: " + json.dumps(output, default=str, ensure_ascii=False))
    if expected is not None:
        user_lines.append("expected: " + json.dumps(expected, default=str, ensure_ascii=False))
    user_lines.append("checks:")
    for i, c in enumerate(checks, 1):
        user_lines.append(f"  {i}. {c}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


__all__ = ["build"]

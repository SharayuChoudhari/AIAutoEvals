"""Frontier-tier chain-of-thought judge prompt builder.

Used for complex models (GPT-4 class). The prompt asks the judge to reason step
by step, emit per-dimension sub-scores, and a final aggregate ``score`` in
``[0, 1]``. The metric's ``scored_dimension`` names which sub-score the runner
should lift into the metric value (e.g. ``hallucination_rate`` uses
``"unsupported_claims"``).
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
) -> list[dict[str, str]]:
    """Return a chat messages list for the complex CoT tier."""
    system = (
        "You are a strict evaluation judge. Score the candidate output against "
        "the criteria using chain-of-thought reasoning. Return a JSON object "
        "with a `score` in [0, 1], a short `rationale`, and `sub_scores` keyed "
        f"by dimension name (one dimension must be `{scored_dimension}`). "
        "Be adversarial: only credit what the evidence supports."
    )
    user_lines = [
        f"metric: {metric_name}",
        f"criteria: {metric_description}",
        f"scored_dimension: {scored_dimension}",
        f"task_type: {task_type}",
    ]
    if task_purpose:
        user_lines.append(f"task_purpose: {task_purpose}")
    user_lines.append("input: " + json.dumps(input, default=str, ensure_ascii=False))
    user_lines.append("output: " + json.dumps(output, default=str, ensure_ascii=False))
    if expected is not None:
        user_lines.append(
            "expected: " + json.dumps(expected, default=str, ensure_ascii=False)
        )
    user_lines.append(
        "Produce JSON: {\"score\": float, \"rationale\": str, "
        "\"sub_scores\": {\"" + scored_dimension + "\": float}}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_lines)},
    ]


__all__ = ["build"]

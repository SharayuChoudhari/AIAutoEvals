"""Root-cause narrative for regressing metrics (``diff --explain``).

One judge call per regressing ``(metric, example)``: the regression judge
compares the current vs. baseline example output and explains *why* the metric
dropped. The narrative is rendered into the diff table.
"""

from __future__ import annotations

import json
from typing import Any


async def explain(
    *,
    metric_name: str,
    current_output: Any,
    baseline_output: Any,
    current_score: float | None,
    baseline_score: float | None,
    example: dict[str, Any],
    regression_judge_model: str,
    complete_fn=None,
    fallback_models: tuple[str, ...] = (),
) -> str:
    """Return a short narrative explaining why ``metric_name`` regressed.

    Falls back to a deterministic stub when no judge transport is available
    (e.g. ``complete_fn=None`` and no provider), so ``diff --explain`` never
    crashes on a missing judge — it just produces a terse text delta.
    """
    from ai_eval.judge.gateway import score as judge_score
    from ai_eval.judge.schemas import JudgeRequest

    messages = _build_messages(
        metric_name=metric_name,
        current_output=current_output,
        baseline_output=baseline_output,
        current_score=current_score,
        baseline_score=baseline_score,
        example=example,
    )
    try:
        resp, errors = await judge_score(
            JudgeRequest(
                task_name="diff_explain",
                task_type="custom",
                metric=metric_name,
                example=example,
                output=current_output,
            ),
            model=regression_judge_model,
            prompt_messages=messages,
            scored_dimension="score",
            complete_fn=complete_fn,
            fallback_models=fallback_models,
        )
    except Exception as exc:
        return _fallback(metric_name, current_score, baseline_score, str(exc))

    if resp is None:
        return _fallback(
            metric_name, current_score, baseline_score, "; ".join(e.message for e in errors)
        )
    return resp.rationale or _fallback(metric_name, current_score, baseline_score, None)


def _build_messages(
    *,
    metric_name: str,
    current_output: Any,
    baseline_output: Any,
    current_score: float | None,
    baseline_score: float | None,
    example: dict[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You explain why an evaluation metric regressed between two runs. "
        "Compare the current and baseline outputs, identify the concrete "
        "difference that caused the score drop, and return a short rationale."
    )
    user = "\n".join(
        [
            f"metric: {metric_name}",
            f"current_score: {current_score}",
            f"baseline_score: {baseline_score}",
            "current_output: " + json.dumps(current_output, default=str, ensure_ascii=False),
            "baseline_output: " + json.dumps(baseline_output, default=str, ensure_ascii=False),
            "example: " + json.dumps(example, default=str, ensure_ascii=False),
            'Return JSON: {"score": 0.0, "rationale": "<one paragraph>"}',
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _fallback(
    metric_name: str,
    current_score: float | None,
    baseline_score: float | None,
    err: str | None,
) -> str:
    delta = None
    if current_score is not None and baseline_score is not None:
        delta = current_score - baseline_score
    parts = [f"metric {metric_name!r} regressed"]
    if delta is not None:
        parts.append(f"by {delta:+.3f} ({baseline_score:.3f} → {current_score:.3f})")
    if err:
        parts.append(f"(regression judge unavailable: {err})")
    return " ".join(parts) + "."


__all__ = ["explain"]

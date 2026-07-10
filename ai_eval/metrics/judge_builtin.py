"""Built-in judge metrics: prompt builders + scored dimensions.

A :class:`~ai_eval.metrics.registry.JudgeMetric` couples a metric name to its
judge prompt builder (``(task_spec, example, output) -> messages``) and the
canonical sub-score dimension the runner lifts into the metric value.

These four ship in-code; ``latency_p50`` / ``latency_p95`` are non-judge and
live in ``ai_eval/runner/metrics/latency.py`` (computed by the engine).
``context_precision`` is "partial" per design §2.6: a deterministic first pass
over retrieved docs, with the judge as a tiebreaker — implemented as a judge
metric whose prompt builder includes the deterministic signal.
"""

from __future__ import annotations

import json
from typing import Any

from ai_eval.judge.prompts.checklist_basic import build as build_checklist
from ai_eval.judge.prompts.cot_complex import build as build_cot
from ai_eval.metrics.registry import JudgeMetric


def _input_payload(example: dict[str, Any]) -> Any:
    """Return the example's input (either ``input`` dict or the whole example)."""
    if "input" in example and isinstance(example["input"], dict):
        return example["input"]
    return example


def _expected_payload(example: dict[str, Any]) -> Any:
    return example.get("expected")


def _task_fields(task_spec: Any) -> tuple[str, str | None]:
    """Extract ``(task_type, purpose)`` from a TaskSpec or duck-typed object."""
    task_type = getattr(task_spec, "type", "custom") or "custom"
    purpose = getattr(task_spec, "purpose", None)
    return task_type, purpose


def _argument_accuracy(
    task_spec: Any, example: dict[str, Any], output: Any, *, tier: str = "complex"
) -> list[dict[str, str]]:
    task_type, purpose = _task_fields(task_spec)
    inp = _input_payload(example)
    expected = _expected_payload(example)
    common = dict(
        metric_name="argument_accuracy",
        metric_description=(
            "Accuracy of tool/function call arguments vs. expected arguments. "
            "Score 1.0 for exact match, partial credit for correct keys/values."
        ),
        scored_dimension="argument_accuracy",
        task_type=task_type,
        task_purpose=purpose,
        input=inp,
        output=output,
        expected=expected,
    )
    if tier == "basic":
        return build_checklist(
            checks=(
                "all expected arguments present",
                "argument values match expected",
                "no extra/invalid arguments",
            ),
            **common,
        )
    return build_cot(**common)


def _hallucination_rate(
    task_spec: Any, example: dict[str, Any], output: Any, *, tier: str = "complex"
) -> list[dict[str, str]]:
    task_type, purpose = _task_fields(task_spec)
    inp = _input_payload(example)
    expected = _expected_payload(example)
    common = dict(
        metric_name="hallucination_rate",
        metric_description=(
            "Rate of unsupported claims in the output relative to grounded "
            "context. Score = 1 - (unsupported_claims / total_claims)."
        ),
        scored_dimension="unsupported_claims",
        task_type=task_type,
        task_purpose=purpose,
        input=inp,
        output=output,
        expected=expected,
    )
    if tier == "basic":
        return build_checklist(
            checks=(
                "every claim is supported by context",
                "no fabricated entities/dates/numbers",
                "no contradictions with context",
            ),
            **common,
        )
    return build_cot(**common)


def _faithfulness(
    task_spec: Any, example: dict[str, Any], output: Any, *, tier: str = "complex"
) -> list[dict[str, str]]:
    task_type, purpose = _task_fields(task_spec)
    inp = _input_payload(example)
    expected = _expected_payload(example)
    common = dict(
        metric_name="faithfulness",
        metric_description=(
            "Faithfulness of the answer to the retrieved context. Score 1.0 when "
            "the answer is fully entailed by the context."
        ),
        scored_dimension="faithfulness",
        task_type=task_type,
        task_purpose=purpose,
        input=inp,
        output=output,
        expected=expected,
    )
    if tier == "basic":
        return build_checklist(
            checks=(
                "answer is entailed by context",
                "no information added beyond context",
                "answer addresses the query",
            ),
            **common,
        )
    return build_cot(**common)


def _context_precision(
    task_spec: Any, example: dict[str, Any], output: Any, *, tier: str = "complex"
) -> list[dict[str, str]]:
    """Partial metric: deterministic signal first, judge tiebreak (design §2.6).

    The deterministic signal is the fraction of retrieved chunks that contain
    the query terms (computed here over the example's trace); the judge decides
    relevance for the ambiguous remainder.
    """
    task_type, purpose = _task_fields(task_spec)
    inp = _input_payload(example)
    trace = example.get("trace") or {}
    calls = trace.get("calls", []) if isinstance(trace, dict) else []
    retrieved = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        if call.get("kind") in {"retrieve", "vector_search"}:
            retrieved.append(call.get("result"))
    det_signal = _deterministic_precision(inp, retrieved)
    common = dict(
        metric_name="context_precision",
        metric_description=(
            "Precision of retrieved context relevance to the query. "
            f"Deterministic signal (term-overlap) = {det_signal:.3f}; judge "
            "resolves ambiguous chunks. Score in [0, 1]."
        ),
        scored_dimension="context_precision",
        task_type=task_type,
        task_purpose=purpose,
        input={**(_input_payload(example) if isinstance(inp, dict) else {"input": inp}),
               "_retrieved": retrieved,
               "_deterministic_signal": det_signal},
        output=output,
    )
    if tier == "basic":
        return build_checklist(
            checks=(
                "retrieved context matches the query",
                "no irrelevant chunks in top results",
                "context suffices to answer",
            ),
            **common,
        )
    return build_cot(**common)


def _deterministic_precision(query: Any, retrieved: list[Any]) -> float:
    """Term-overlap heuristic: fraction of retrieved chunks sharing query terms."""
    if not retrieved:
        return 1.0
    qtext = json.dumps(query, default=str, ensure_ascii=False).lower()
    qterms = {w for w in qtext.split() if len(w) > 2}
    if not qterms:
        return 1.0
    hits = 0
    for chunk in retrieved:
        ctext = json.dumps(chunk, default=str, ensure_ascii=False).lower()
        if any(t in ctext for t in qterms):
            hits += 1
    return hits / len(retrieved)


def _task_completion(
    task_spec: Any, example: dict[str, Any], output: Any, *, tier: str = "complex"
) -> list[dict[str, str]]:
    task_type, purpose = _task_fields(task_spec)
    inp = _input_payload(example)
    expected = _expected_payload(example)
    common = dict(
        metric_name="task_completion",
        metric_description=(
            "Whether the end-to-end task (e.g. booking, agent workflow) "
            "completed successfully against the expected outcome. Score 1.0 "
            "for full completion, partial credit for incomplete but plausible "
            "progress."
        ),
        scored_dimension="task_completion",
        task_type=task_type,
        task_purpose=purpose,
        input=inp,
        output=output,
        expected=expected,
    )
    if tier == "basic":
        return build_checklist(
            checks=(
                "output reaches the expected terminal state",
                "no required step is skipped or aborted",
                "side effects (booking, confirmation) match expected",
            ),
            **common,
        )
    return build_cot(**common)


def _slot_filling_accuracy(
    task_spec: Any, example: dict[str, Any], output: Any, *, tier: str = "complex"
) -> list[dict[str, str]]:
    task_type, purpose = _task_fields(task_spec)
    inp = _input_payload(example)
    expected = _expected_payload(example)
    common = dict(
        metric_name="slot_filling_accuracy",
        metric_description=(
            "Accuracy of slots/arguments filled for booking or workflow tasks "
            "vs. the expected slots. Score 1.0 for exact match, partial credit "
            "per correctly filled slot."
        ),
        scored_dimension="slot_filling_accuracy",
        task_type=task_type,
        task_purpose=purpose,
        input=inp,
        output=output,
        expected=expected,
    )
    if tier == "basic":
        return build_checklist(
            checks=(
                "all required slots are present in the output",
                "slot values match the expected values",
                "no extra or invalid slots filled",
            ),
            **common,
        )
    return build_cot(**common)


BUILTIN_JUDGE_METRICS: tuple[JudgeMetric, ...] = (
    JudgeMetric(
        name="argument_accuracy",
        description="Accuracy of tool/function call arguments vs. expected.",
        applicable_task_types=("tool_calling", "agent"),
        scored_dimension="argument_accuracy",
        prompt_builder=_argument_accuracy,
    ),
    JudgeMetric(
        name="hallucination_rate",
        description="Rate of unsupported claims in the output relative to grounded context.",
        applicable_task_types=("chat", "rag", "agent", "tool_calling"),
        scored_dimension="unsupported_claims",
        prompt_builder=_hallucination_rate,
    ),
    JudgeMetric(
        name="context_precision",
        description="Precision of retrieved context relevance to the query (partial: det + judge).",
        applicable_task_types=("rag",),
        scored_dimension="context_precision",
        prompt_builder=_context_precision,
    ),
    JudgeMetric(
        name="faithfulness",
        description="Faithfulness of the answer to the retrieved context.",
        applicable_task_types=("rag", "summarization"),
        scored_dimension="faithfulness",
        prompt_builder=_faithfulness,
    ),
    JudgeMetric(
        name="task_completion",
        description="Whether the end-to-end task (e.g. booking, agent) completed successfully.",
        applicable_task_types=("booking", "workflow", "agent"),
        scored_dimension="task_completion",
        prompt_builder=_task_completion,
    ),
    JudgeMetric(
        name="slot_filling_accuracy",
        description="Accuracy of slots/arguments filled for booking or workflow tasks.",
        applicable_task_types=("booking", "workflow"),
        scored_dimension="slot_filling_accuracy",
        prompt_builder=_slot_filling_accuracy,
    ),
)


__all__ = ["BUILTIN_JUDGE_METRICS"]

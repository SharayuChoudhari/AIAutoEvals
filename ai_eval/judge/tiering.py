"""Model-id → prompt tier selection.

Frontier models (GPT-4 class, Claude 3 Opus/Sonnet, Gemini 1.5 Pro, o1) get the
chain-of-thought prompt tier; everything else (local small models, gpt-3.5,
haiku, etc.) gets the binary checklist tier. The hint list is a set of plain
case-insensitive substrings matched against the LiteLLM model id.

Users override/extend the set via ``judge.tiering.complex_models`` in
``rubrics.yaml`` (see :class:`~ai_eval.config.schema.JudgeConfig`).
"""

from __future__ import annotations

from ai_eval.config.defaults import COMPLEX_MODEL_HINTS

#: The two tiers a judge prompt can target.
TIERS = ("complex", "basic")


def select_tier(
    model_id: str,
    *,
    complex_models: tuple[str, ...] | None = None,
) -> str:
    """Return ``"complex"`` or ``"basic"`` for ``model_id``.

    ``complex_models`` (when provided) replaces the built-in
    :data:`~ai_eval.config.defaults.COMPLEX_MODEL_HINTS`; an empty tuple forces
    everything to ``basic``.
    """
    hints = complex_models if complex_models is not None else COMPLEX_MODEL_HINTS
    lowered = model_id.lower()
    for hint in hints:
        if hint and hint.lower() in lowered:
            return "complex"
    return "basic"


__all__ = ["TIERS", "select_tier"]

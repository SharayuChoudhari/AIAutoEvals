"""Judge prompt builders.

One module per tier. Each builder takes ``(task_spec, example, output)`` and
returns a chat-style ``messages`` list (the same shape litellm accepts). The
metric's ``scored_dimension`` selects which field of the validated
:class:`~ai_eval.judge.schemas.JudgeResponse` becomes the canonical metric
score.
"""

from __future__ import annotations

from .checklist_basic import build as build_checklist_basic
from .cot_complex import build as build_cot_complex

__all__ = ["build_checklist_basic", "build_cot_complex"]

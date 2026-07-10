"""Model-agnostic judge gateway.

Async-native transport layer for LLM-as-judge scoring. This is intentionally a
**separate code path** from the sync SLM rubric client in
``ai_eval.inference.slm.client``: that client serves the init-time rubric
builder (one-shot classification), while this gateway serves the Phase 4 runner
(many parallel judge calls per example/metric). Do not refactor the two into a
shared client — they have different lifecycles, error contracts, and test
injection points.

Public surface:
- :class:`JudgeRequest` / :class:`JudgeResponse` / :class:`JudgeError` (pydantic)
- :func:`score` — async, injectable transport
- :func:`ping` — reachability probe
"""

from __future__ import annotations

from .gateway import JudgeUnavailable, ping, score
from .schemas import JudgeError, JudgeRequest, JudgeResponse
from .tiering import select_tier

__all__ = [
    "JudgeError",
    "JudgeRequest",
    "JudgeResponse",
    "JudgeUnavailable",
    "ping",
    "score",
    "select_tier",
]

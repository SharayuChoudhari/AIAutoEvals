"""Latency metrics computed by the engine (no judge call).

``latency_p50`` and ``latency_p95`` are registered in the metric registry as
``non_judge=True``; the engine computes them from the per-example latency
samples and writes them directly into the task record's ``latency_ms`` dict.
"""

from __future__ import annotations

from collections.abc import Sequence


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile of ``values`` (0..100)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def compute_latencies(samples: Sequence[float]) -> dict[str, float]:
    """Return ``{"p50": ..., "p95": ...}`` from per-example latency samples (ms)."""
    if not samples:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
    }


__all__ = ["compute_latencies", "percentile"]

"""Per-metric delta computation between two runs."""

from __future__ import annotations

from typing import Any


def compute_delta(
    current: dict[str, Any], baseline: dict[str, Any] | None
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Return ``{task: {metric: {score, delta, baseline_score}}}``.

    Re-derives deltas for display. A missing baseline metric → ``delta=None``.
    """
    out: dict[str, dict[str, dict[str, float | None]]] = {}
    cur_tasks = current.get("tasks") or {}
    base_tasks = (baseline or {}).get("tasks") or {}
    for tname, cur_task in cur_tasks.items():
        cur_metrics = (cur_task or {}).get("metrics") or {}
        base_metrics = (base_tasks.get(tname) or {}).get("metrics") or {}
        row: dict[str, dict[str, float | None]] = {}
        for mname, cur_m in cur_metrics.items():
            cur_score = cur_m.get("score") if isinstance(cur_m, dict) else None
            base_score = None
            bm = base_metrics.get(mname)
            if isinstance(bm, dict):
                base_score = bm.get("score")
            delta = None
            if cur_score is not None and base_score is not None:
                delta = cur_score - base_score
            row[mname] = {
                "score": cur_score,
                "baseline_score": base_score,
                "delta": delta,
            }
        out[tname] = row
    return out


__all__ = ["compute_delta"]

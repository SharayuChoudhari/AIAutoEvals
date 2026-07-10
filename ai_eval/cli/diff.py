"""`ai-eval diff` — show regression diff vs. a baseline run.

Loads current + baseline runs, computes per-metric deltas, and renders a
metric-level Δ table. With ``--explain`` and a configured regression judge,
emits a root-cause narrative per regressing metric (one judge call each).
"""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from ai_eval.cli.app import (
    EXIT_OK,
    EXIT_USAGE,
    GlobalOptions,
    OutputFormat,
)
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_diff
from ai_eval.config.loader import load_resolved
from ai_eval.insights.diff import compute_delta
from ai_eval.storage.paths import resolve_paths
from ai_eval.storage.runs import load_run_raw


def diff_command(
    ctx: typer.Context,
    against: str = typer.Option("last", "--against"),
    task: str | None = typer.Option(None, "--task"),
    metric: str | None = typer.Option(None, "--metric"),
    limit: int = typer.Option(10, "--limit", min=1),
    explain: bool = typer.Option(True, "--explain/--no-explain"),
) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd)

    # "current" = the most recent run; "against" = the baseline.
    from ai_eval.insights.history import last_run_id

    current_id = last_run_id(paths.state_dir)
    if current_id is None:
        typer.echo("error: no runs found (run `ai-eval run` first)", err=True)
        raise typer.Exit(code=EXIT_USAGE)
    try:
        current = load_run_raw(paths.state_dir, current_id)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc

    baseline: dict[str, Any] | None = None
    if against == "last":
        # "last" against the current means the run before current.
        from ai_eval.insights.history import list_history

        runs = list_history(paths.state_dir)
        if len(runs) >= 2:
            baseline_id = runs[-2]["id"]
            try:
                baseline = load_run_raw(paths.state_dir, baseline_id)
            except FileNotFoundError:
                baseline = None
    elif against != "none":
        try:
            baseline = load_run_raw(paths.state_dir, against)
        except FileNotFoundError as exc:
            typer.echo(f"error: baseline run {against!r} not found", err=True)
            raise typer.Exit(code=EXIT_USAGE) from exc

    deltas = compute_delta(current, baseline)

    # filter
    if task:
        deltas = {t: v for t, v in deltas.items() if t == task}
    if metric:
        deltas = {
            t: {m: v for m, v in v.items() if m == metric} for t, v in deltas.items()
        }

    narratives: dict[str, str] = {}
    if explain and baseline is not None:
        narratives = asyncio.run(
            _explain_regressions(
                opts, current, baseline, deltas, metric_filter=metric, limit=limit
            )
        )

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "current_run": current_id,
                "baseline_run": against,
                "deltas": deltas,
                "narratives": narratives,
            }
        )
    else:
        render_diff(deltas, narratives, no_color=opts.no_color, limit=limit)

    exit_code = EXIT_OK
    any_regressed = any(
        (d := row.get("delta")) is not None and d < 0
        for t in deltas.values()
        for row in t.values()
    )
    if any_regressed:
        # informational exit 0 unless --fail-on-regression; but diff has no such
        # flag wired in Phase 4 surface (per stub it's --explain). Keep exit 0.
        pass
    raise typer.Exit(code=exit_code)


async def _explain_regressions(
    opts: GlobalOptions,
    current: dict[str, Any],
    baseline: dict[str, Any],
    deltas: dict[str, dict[str, dict[str, Any]]],
    *,
    metric_filter: str | None,
    limit: int,
) -> dict[str, str]:
    from ai_eval.insights.root_cause import explain

    resolved = load_resolved(project_root=opts.cwd, config_path=opts.config_path)
    regression_model = (resolved.data.get("judge") or {}).get("regression_check")
    if not regression_model:
        return {}

    out: dict[str, str] = {}
    count = 0
    for tname, metrics in deltas.items():
        for mname, row in metrics.items():
            if metric_filter and mname != metric_filter:
                continue
            delta = row.get("delta")
            if delta is None or delta >= 0:
                continue
            if count >= limit:
                break
            count += 1
            key = f"{tname}.{mname}"
            cur_examples = (current.get("tasks", {}).get(tname) or {}).get("examples", [])
            base_examples = (baseline.get("tasks", {}).get(tname) or {}).get("examples", [])
            cur_out = cur_examples[0] if cur_examples else None
            base_out = base_examples[0] if base_examples else None
            out[key] = await explain(
                metric_name=mname,
                current_output=cur_out,
                baseline_output=base_out,
                current_score=row.get("score"),
                baseline_score=row.get("baseline_score"),
                example=cur_out or {},
                regression_judge_model=regression_model,
            )
    return out


__all__ = ["diff_command"]

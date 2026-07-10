"""`ai-evals run` — execute evaluations against the current code.

Loads resolved config, resolves the golden set, resolves the baseline
(``last``/``none``/run-id), runs the async engine with the judge gateway, saves
the run record + history, and renders per the effective format.

Exit codes: 0 ok, 1 runner error, 2 usage, 3 regression/threshold breach under
``--fail-on-regression``.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import typer

from ai_eval.cli.app import (
    EXIT_GENERAL,
    EXIT_OK,
    EXIT_REGRESSION,
    EXIT_USAGE,
    GlobalOptions,
    OutputFormat,
)
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_run
from ai_eval.config.loader import load_resolved
from ai_eval.insights.history import append_history, last_run_id
from ai_eval.runner.engine import config_hash, execute
from ai_eval.runner.git import collect_git_meta
from ai_eval.runner.thresholds import MetricNotImplementedError
from ai_eval.storage.paths import resolve_paths
from ai_eval.storage.runs import new_run_id, save_run


def run_command(
    ctx: typer.Context,
    task: list[str] = typer.Option([], "--task", help="Limit to task(s) (repeatable)."),
    judge: str | None = typer.Option(None, "--judge", help="Override judge.default."),
    baseline: str = typer.Option("last", "--baseline", help="Baseline run id or `last` or `none`."),
    fail_on_regression: bool = typer.Option(
        False,
        "--fail-on-regression",
        help="Exit 3 if any metric regresses past tolerance.",
    ),
    tolerance: float = typer.Option(0.02, "--tolerance", min=0.0, max=1.0),
    parallel: int = typer.Option(4, "--parallel", min=1, max=64),
    no_cache: bool = typer.Option(False, "--no-cache"),
    save: bool = typer.Option(True, "--save/--no-save"),
    tag: list[str] = typer.Option([], "--tag", help="Attach tag(s) (repeatable)."),
) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd)
    paths.ensure_state()

    resolved = load_resolved(project_root=opts.cwd, config_path=opts.config_path)
    try:
        rubrics = resolved.as_rubrics()
    except Exception as exc:
        typer.echo(f"error: rubrics.yaml invalid: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc

    golden_path = paths.golden_set_json
    if not golden_path.is_file():
        typer.echo(
            f"error: golden set not found at {golden_path}", err=True
        )
        typer.echo("  fix: run `ai-evals bootstrap -- <cmd>` to capture traces", err=True)
        raise typer.Exit(code=EXIT_USAGE)
    try:
        golden_set = json.loads(golden_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        typer.echo(f"error: golden set unreadable: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc
    golden_tasks = golden_set.get("tasks", {}) if isinstance(golden_set, dict) else {}

    # Resolve baseline.
    baseline_id: str | None = None
    baseline_tasks: dict[str, Any] | None = None
    if baseline != "none":
        try:
            if baseline == "last":
                bid = last_run_id(paths.state_dir)
                if bid is not None:
                    from ai_eval.storage.runs import load_run_raw
                    base_record = load_run_raw(paths.state_dir, bid)
                    baseline_id = bid
                    baseline_tasks = base_record.get("tasks")
            else:
                from ai_eval.storage.runs import load_run_raw
                base_record = load_run_raw(paths.state_dir, baseline)
                baseline_id = baseline
                baseline_tasks = base_record.get("tasks")
        except FileNotFoundError:
            # baseline run missing -> treat as fresh
            baseline_id = None
            baseline_tasks = None

    started_at = time.time()
    run_id = new_run_id()
    chash = config_hash(rubrics)

    try:
        record = asyncio.run(
            execute(
                rubrics,
                golden_tasks,
                project_root=opts.cwd,
                parallel=parallel,
                cache_on=not no_cache,
                task_filter=tuple(task),
                judge_override=judge,
                tolerance=tolerance,
                fail_on_regression=fail_on_regression,
                baseline=baseline_tasks,
                run_id=run_id,
                started_at=started_at,
                tags=list(tag),
                config_hash_value=chash,
                baseline_id=baseline_id,
            )
        )
    except MetricNotImplementedError as exc:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(f"  why:  {exc.why}", err=True)
        typer.echo(f"  fix:  {exc.fix}", err=True)
        raise typer.Exit(code=EXIT_GENERAL) from exc
    except Exception as exc:
        typer.echo(f"error: run failed: {type(exc).__name__}: {exc}", err=True)
        raise typer.Exit(code=EXIT_GENERAL) from exc

    record.git = collect_git_meta(opts.cwd)

    if save:
        save_run(paths.state_dir, record)
        append_history(paths.state_dir, record)

    # Determine exit code.
    exit_code = EXIT_OK
    if fail_on_regression:
        any_fail = any(
            m.status == "fail"
            for t in record.tasks.values()
            for m in t.metrics.values()
        )
        if any_fail:
            exit_code = EXIT_REGRESSION
    if record.summary.errors > 0 and exit_code == EXIT_OK:
        # errors don't force non-zero unless a metric failed; surface in output.
        pass

    # Render.
    if opts.effective_format == OutputFormat.JSON:
        json_dump(record.model_dump(mode="json"))
    elif opts.effective_format == OutputFormat.TSV:
        _render_tsv(record)
    else:
        render_run(record, no_color=opts.no_color)

    raise typer.Exit(code=exit_code)


def _render_tsv(record) -> None:
    lines = ["task\tmetric\tscore\tdelta\tthreshold\tstatus"]
    for tname, t in record.tasks.items():
        for mname, m in t.metrics.items():
            delta = "" if m.delta is None else f"{m.delta:+.4f}"
            score = "" if m.score is None else f"{m.score:.4f}"
            lines.append(
                f"{tname}\t{mname}\t{score}\t{delta}\t{m.threshold:.4f}\t{m.status}"
            )
    typer.echo("\n".join(lines))


__all__ = ["run_command"]

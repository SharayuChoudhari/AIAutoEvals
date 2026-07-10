"""`ai-eval history` — list, show, prune, export past runs."""

from __future__ import annotations

import json

import typer

from ai_eval.cli.app import (
    EXIT_OK,
    EXIT_USAGE,
    GlobalOptions,
    OutputFormat,
)
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_history
from ai_eval.insights.history import export as export_history
from ai_eval.insights.history import list_history, load_run, prune
from ai_eval.storage.paths import resolve_paths


def history_command(
    ctx: typer.Context,
    list_runs: bool = typer.Option(True, "--list/--no-list"),
    show: str | None = typer.Option(None, "--show"),
    prune_runs: bool = typer.Option(False, "--prune"),
    keep: int = typer.Option(50, "--keep", min=1),
    export: str | None = typer.Option(None, "--export"),
) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd)

    if export:
        export_history(paths.state_dir, __import__("pathlib").Path(export))
        typer.echo(f"exported history to {export}", err=True)
        raise typer.Exit(code=EXIT_OK)

    if prune_runs:
        removed = prune(paths.state_dir, keep=keep)
        if opts.effective_format == OutputFormat.JSON:
            json_dump({"removed": removed, "kept": keep})
        else:
            typer.echo(f"removed {removed} run(s); kept {keep}")
        raise typer.Exit(code=EXIT_OK)

    if show:
        try:
            record = load_run(paths.state_dir, show)
        except FileNotFoundError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=EXIT_USAGE) from exc
        if opts.effective_format == OutputFormat.JSON:
            json_dump(record.model_dump(mode="json"))
        else:
            typer.echo(json.dumps(record.model_dump(mode="json"), indent=2))
        raise typer.Exit(code=EXIT_OK)

    # default: --list
    runs = list_history(paths.state_dir)
    if opts.effective_format == OutputFormat.JSON:
        json_dump({"runs": runs})
    else:
        render_history(runs, no_color=opts.no_color)
    raise typer.Exit(code=EXIT_OK)


__all__ = ["history_command"]

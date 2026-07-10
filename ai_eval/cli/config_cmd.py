"""`ai-evals config` — inspect resolved configuration."""

from __future__ import annotations

import typer

from ai_eval.cli.app import EXIT_OK, GlobalOptions, OutputFormat
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_config
from ai_eval.config.loader import load_resolved


def config_command(
    ctx: typer.Context,
    show_path: bool = typer.Option(
        False,
        "--path",
        help="Print the resolved rubrics.yaml path and exit.",
    ),
    set_kv: str | None = typer.Option(
        None,
        "--set",
        help="key=value to write back to rubrics.yaml (Phase 2; not yet implemented).",
    ),
) -> None:
    opts: GlobalOptions = ctx.obj
    resolved = load_resolved(project_root=opts.cwd, config_path=opts.config_path)

    if show_path:
        path = resolved.rubrics_path or (opts.cwd / "eval" / "rubrics.yaml")
        typer.echo(str(path))
        raise typer.Exit(code=EXIT_OK)

    if set_kv is not None:
        typer.echo(
            "error: `--set` is reserved for Phase 2; not yet implemented",
            err=True,
        )
        raise typer.Exit(code=2)

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "rubrics_path": str(resolved.rubrics_path) if resolved.rubrics_path else None,
                "user_config_path": (
                    str(resolved.user_config_path) if resolved.user_config_path else None
                ),
                "merged": resolved.data,
                "sources": resolved.sources,
            }
        )
    else:
        render_config(resolved.data, resolved.sources, no_color=opts.no_color)

    raise typer.Exit(code=EXIT_OK)


__all__ = ["config_command"]

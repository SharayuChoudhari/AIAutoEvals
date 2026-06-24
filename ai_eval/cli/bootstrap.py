"""`ai-eval bootstrap` — placeholder for Phase 3.

Validates the `-- <cmd>` contract from day one so the error message is correct
even before the runtime tracer lands.
"""

from __future__ import annotations

import typer

from ai_eval.cli.app import EXIT_USAGE
from ai_eval.cli.stubs import not_yet


def bootstrap_command(
    ctx: typer.Context,
    task: str | None = typer.Option(None, "--task", help="Capture only one task."),
    max_captures: int = typer.Option(50, "--max", min=1),
    append: bool = typer.Option(True, "--append/--replace"),
    no_instrument: bool = typer.Option(False, "--no-instrument"),
    out: str | None = typer.Option(None, "--out", help="Override golden set path."),
    runtime_cmd: list[str] = typer.Argument(
        None,
        metavar="-- <command> [args...]",
        help="The user runtime to trace (everything after `--`).",
    ),
) -> None:
    if not runtime_cmd:
        typer.echo("error: missing runtime command", err=True)
        typer.echo("  what: no `-- <command>` provided", err=True)
        typer.echo("  why:  bootstrap requires an explicit runtime to trace", err=True)
        typer.echo("  fix:  ai-eval bootstrap -- pytest -q", err=True)
        raise typer.Exit(code=EXIT_USAGE)
    not_yet("Phase 3", "bootstrap", "docs/phases/phase-3.md")


__all__ = ["bootstrap_command"]

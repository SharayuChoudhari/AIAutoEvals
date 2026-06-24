"""`ai-eval diff` — placeholder for Phase 4."""

from __future__ import annotations

import typer

from ai_eval.cli.stubs import not_yet


def diff_command(
    ctx: typer.Context,
    against: str = typer.Option("last", "--against"),
    task: str | None = typer.Option(None, "--task"),
    metric: str | None = typer.Option(None, "--metric"),
    limit: int = typer.Option(10, "--limit", min=1),
    explain: bool = typer.Option(True, "--explain/--no-explain"),
) -> None:
    not_yet("Phase 4", "diff", "docs/phases/phase-4.md")


__all__ = ["diff_command"]

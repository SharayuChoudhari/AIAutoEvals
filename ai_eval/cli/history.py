"""`ai-eval history` — placeholder for Phase 4."""

from __future__ import annotations

import typer

from ai_eval.cli.stubs import not_yet


def history_command(
    ctx: typer.Context,
    list_runs: bool = typer.Option(True, "--list/--no-list"),
    show: str | None = typer.Option(None, "--show"),
    prune: bool = typer.Option(False, "--prune"),
    keep: int = typer.Option(50, "--keep", min=1),
    export: str | None = typer.Option(None, "--export"),
) -> None:
    not_yet("Phase 4", "history", "docs/phases/phase-4.md")


__all__ = ["history_command"]

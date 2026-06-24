"""`ai-eval report` — placeholder for Phase 4."""

from __future__ import annotations

import typer

from ai_eval.cli.stubs import not_yet


def report_command(
    ctx: typer.Context,
    run_id: str = typer.Argument("last"),
    out: str | None = typer.Option(None, "--out"),
    web: bool = typer.Option(False, "--web"),
) -> None:
    not_yet("Phase 4", "report", "docs/phases/phase-4.md")


__all__ = ["report_command"]

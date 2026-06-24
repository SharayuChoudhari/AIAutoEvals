"""`ai-eval judge` — placeholder for Phase 2."""

from __future__ import annotations

import typer

from ai_eval.cli.stubs import not_yet


def judge_command(
    ctx: typer.Context,
    list_models: bool = typer.Option(False, "--list"),
    ping: str | None = typer.Option(None, "--ping"),
    prompt: str | None = typer.Option(None, "--prompt"),
    model: str | None = typer.Option(None, "--model"),
) -> None:
    not_yet("Phase 2", "judge", "docs/phases/phase-2.md")


__all__ = ["judge_command"]

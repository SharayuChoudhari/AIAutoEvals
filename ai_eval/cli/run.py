"""`ai-eval run` — placeholder for Phase 4."""

from __future__ import annotations

import typer

from ai_eval.cli.stubs import not_yet


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
    not_yet("Phase 4", "run", "docs/phases/phase-4.md")


__all__ = ["run_command"]

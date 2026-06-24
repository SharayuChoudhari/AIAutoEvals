"""Shared helpers for commands that are reserved for later phases.

Each future phase ships its own module (`cli/run.py`, etc.) — these stubs only
exist so the top-level help surface is complete from v0.1 and users get a
useful error rather than 'unknown command'.
"""

from __future__ import annotations

import sys

import typer

from ai_eval import __version__
from ai_eval.cli.app import EXIT_OK, EXIT_USAGE


def not_yet(phase: str, command: str, see: str) -> None:
    """Raise the canonical 'phase not yet implemented' usage error."""
    typer.echo(f"error: `ai-eval {command}` is reserved for {phase}", err=True)
    typer.echo(f"  what: command stub — body lands in {phase}", err=True)
    typer.echo("  why:  v0.1.0 ships Phase 1 only (init / analyze / doctor / config)", err=True)
    typer.echo(f"  fix:  see {see} or `ai-eval --help`", err=True)
    raise typer.Exit(code=EXIT_USAGE)


def version_command() -> None:
    """Print the version and exit."""
    py = f"python {sys.version_info.major}.{sys.version_info.minor}"
    typer.echo(f"ai-eval {__version__} ({py})")
    raise typer.Exit(code=EXIT_OK)


__all__ = ["not_yet", "version_command"]

"""Human-format renderers for terminal output."""

from __future__ import annotations

from typing import Iterable

from rich.console import Console
from rich.table import Table

from ai_eval.cli.render.theme import FAIL, INFO, PASS, state_glyph


def stdout_console(*, no_color: bool) -> Console:
    """Rich console for the data channel (stdout)."""
    return Console(no_color=no_color, soft_wrap=True, highlight=False)


def render_init_summary(
    *,
    files_scanned: int,
    elapsed_seconds: float,
    written: list[tuple[str, str]],          # (relative_path, status)
    tasks: list[tuple[str, str, str]],       # (name, type, file_path)
    next_command: str,
    no_color: bool,
) -> None:
    """Render the human-form summary for `ai-eval init` per design §1.2."""
    console = stdout_console(no_color=no_color)
    console.print(
        f"{state_glyph(PASS, no_color=no_color)} scanned {files_scanned} files "
        f"in {elapsed_seconds:.1f}s"
    )
    if tasks:
        console.print(
            f"{state_glyph(PASS, no_color=no_color)} detected {len(tasks)} AI task(s)"
        )
        for name, kind, path in tasks:
            console.print(f"  - [cyan]{name}[/cyan]  ({kind})   {path}")
    else:
        console.print(
            f"{state_glyph(INFO, no_color=no_color)} no AI tasks detected; "
            f"writing a stub rubrics.yaml"
        )
    for rel_path, status in written:
        glyph = state_glyph(PASS if status != "skipped" else INFO, no_color=no_color)
        console.print(f"{glyph} {status} {rel_path}")
    console.print(f"next: [cyan]{next_command}[/cyan]")


def render_dry_run_summary(
    *,
    files_scanned: int,
    tasks: list[tuple[str, str, str]],
    would_write: Iterable[str],
    no_color: bool,
) -> None:
    """Render `ai-eval init --dry-run` output."""
    console = stdout_console(no_color=no_color)
    console.print(
        f"{state_glyph(INFO, no_color=no_color)} dry-run: scanned {files_scanned} files"
    )
    console.print(
        f"{state_glyph(INFO, no_color=no_color)} would detect {len(tasks)} AI task(s)"
    )
    for name, kind, path in tasks:
        console.print(f"  - [cyan]{name}[/cyan]  ({kind})   {path}")
    for path in would_write:
        console.print(
            f"{state_glyph(INFO, no_color=no_color)} would write {path}"
        )


def render_doctor(checks: list[tuple[str, bool, str]], *, no_color: bool) -> None:
    """Render the `doctor` checklist."""
    console = stdout_console(no_color=no_color)
    table = Table(show_header=True, header_style="bold")
    table.add_column("check", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail")
    for name, ok, detail in checks:
        glyph = state_glyph(PASS if ok else FAIL, no_color=no_color)
        table.add_row(name, glyph, detail)
    console.print(table)


def render_config(merged: dict, sources: dict[str, str], *, no_color: bool) -> None:
    """Render the merged config with source provenance per key."""
    console = stdout_console(no_color=no_color)
    table = Table(show_header=True, header_style="bold")
    table.add_column("key")
    table.add_column("value")
    table.add_column("source")

    def walk(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            dotted = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                walk(v, dotted)
            else:
                table.add_row(dotted, repr(v), sources.get(dotted, "builtin"))

    walk(merged)
    console.print(table)


__all__ = [
    "render_config",
    "render_doctor",
    "render_dry_run_summary",
    "render_init_summary",
    "stdout_console",
]

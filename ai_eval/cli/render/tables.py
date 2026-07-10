"""Human-format renderers for terminal output."""

from __future__ import annotations

from collections.abc import Iterable

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


def render_judge_list(
    rows: list[tuple[str, str, bool, str]], *, no_color: bool
) -> None:
    """Render `judge --list`. ``rows`` = (role, model, reachable, detail)."""
    console = stdout_console(no_color=no_color)
    table = Table(show_header=True, header_style="bold")
    table.add_column("role", no_wrap=True)
    table.add_column("model")
    table.add_column("status", no_wrap=True)
    table.add_column("detail")
    for role, mdl, ok, detail in rows:
        glyph = state_glyph(PASS if ok else FAIL, no_color=no_color)
        table.add_row(role, mdl, glyph, detail)
    console.print(table)


def render_judge_prompt(model: str, response, *, no_color: bool) -> None:
    """Render `judge --prompt` output (the validated JudgeResponse)."""
    console = stdout_console(no_color=no_color)
    console.print(f"model: [cyan]{model}[/cyan]")
    console.print(f"score: {response.score}")
    if response.rationale:
        console.print(f"rationale: {response.rationale}")
    if response.sub_scores:
        parts = ", ".join(f"{k}={v}" for k, v in response.sub_scores.items())
        console.print(f"sub_scores: {parts}")


def _fmt_num(v: float | None, prec: int = 4) -> str:
    return "" if v is None else f"{v:.{prec}f}"


def render_run(record, *, no_color: bool) -> None:
    """Render a run record as a per-task metric table with a Δ column."""
    console = stdout_console(no_color=no_color)
    s = record.summary
    glyph = state_glyph(PASS if s.errors == 0 and s.failed == 0 else FAIL, no_color=no_color)
    console.print(
        f"{glyph} run {record.id} — "
        f"{s.passed} passed, {s.failed} failed, {s.errors} errors "
        f"({s.examples} examples)"
    )
    for tname, t in record.tasks.items():
        console.print(f"\n[cyan]{tname}[/cyan]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("metric", no_wrap=True)
        table.add_column("score", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("threshold", justify="right")
        table.add_column("status", no_wrap=True)
        for mname, m in t.metrics.items():
            delta = "—" if m.delta is None else f"{m.delta:+.4f}"
            st = state_glyph(PASS if m.status == "pass" else FAIL, no_color=no_color)
            table.add_row(mname, _fmt_num(m.score), delta, f"{m.threshold:.4f}", st)
        if t.latency_ms:
            table.add_row(
                "latency_ms",
                f"p50={t.latency_ms.get('p50', 0):.1f}",
                "",
                "",
                state_glyph(INFO, no_color=no_color),
            )
        console.print(table)
        errs = [e for e in t.errors]
        if errs:
            for e in errs[:5]:
                console.print(f"  {state_glyph(FAIL, no_color=no_color)} {e}")


def render_diff(
    deltas: dict, narratives: dict, *, no_color: bool, limit: int = 10
) -> None:
    """Render per-metric Δ table + regression narratives."""
    console = stdout_console(no_color=no_color)
    table = Table(show_header=True, header_style="bold")
    table.add_column("task", no_wrap=True)
    table.add_column("metric", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("baseline", justify="right")
    table.add_column("Δ", justify="right")
    table.add_column("status", no_wrap=True)
    count = 0
    for tname, metrics in deltas.items():
        for mname, row in metrics.items():
            if count >= limit:
                break
            count += 1
            delta = "—" if row.get("delta") is None else f"{row['delta']:+.4f}"
            d = row.get("delta")
            st = (
                state_glyph(FAIL, no_color=no_color)
                if d is not None and d < 0
                else state_glyph(PASS, no_color=no_color)
            )
            table.add_row(
                tname, mname,
                _fmt_num(row.get("score")),
                _fmt_num(row.get("baseline_score")),
                delta, st,
            )
    console.print(table)
    if narratives:
        console.print("\n[bold]regression narratives[/bold]")
        for key, narrative in narratives.items():
            console.print(f"  [cyan]{key}[/cyan]: {narrative}")


def render_history(runs: list[dict], *, no_color: bool) -> None:
    """Render the run history list."""
    console = stdout_console(no_color=no_color)
    if not runs:
        console.print(f"{state_glyph(INFO, no_color=no_color)} no runs recorded")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", no_wrap=True)
    table.add_column("started_at")
    table.add_column("examples", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("git")
    table.add_column("tags")
    for r in runs:
        s = r.get("summary", {}) or {}
        import datetime as _dt
        ts = r.get("started_at")
        when = (
            _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else ""
        )
        table.add_row(
            r.get("id", "?"),
            when,
            str(s.get("examples", 0)),
            str(s.get("passed", 0)),
            str(s.get("failed", 0)),
            (r.get("git_sha") or "")[:8],
            ",".join(r.get("tags", []) or []),
        )
    console.print(table)


def render_report(record: dict, *, no_color: bool) -> None:
    """Render a past run record (dict form) as a human report."""
    console = stdout_console(no_color=no_color)
    rid = record.get("id", "?")
    s = record.get("summary", {}) or {}
    console.print(f"[bold]run {rid}[/bold]")
    console.print(
        f"  examples: {s.get('examples', 0)}  "
        f"passed: {s.get('passed', 0)}  "
        f"failed: {s.get('failed', 0)}  "
        f"errors: {s.get('errors', 0)}"
    )
    for tname, t in (record.get("tasks") or {}).items():
        console.print(f"\n[cyan]{tname}[/cyan]")
        for mname, m in (t.get("metrics") or {}).items():
            score = "" if m.get("score") is None else f"{m['score']:.4f}"
            delta = "—" if m.get("delta") is None else f"{m['delta']:+.4f}"
            console.print(f"  {mname}: score={score} Δ={delta} status={m.get('status')}")


__all__ = [
    "render_config",
    "render_diff",
    "render_doctor",
    "render_dry_run_summary",
    "render_history",
    "render_init_summary",
    "render_judge_list",
    "render_judge_prompt",
    "render_report",
    "render_run",
    "stdout_console",
]

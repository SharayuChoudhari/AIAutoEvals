"""`ai-eval report` — render a past run as human / json / md / html."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ai_eval.cli.app import (
    EXIT_OK,
    EXIT_USAGE,
    GlobalOptions,
    OutputFormat,
)
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_report
from ai_eval.insights.history import last_run_id
from ai_eval.storage.paths import resolve_paths
from ai_eval.storage.runs import load_run_raw


def report_command(
    ctx: typer.Context,
    run_id: str = typer.Argument("last"),
    out: str | None = typer.Option(None, "--out"),
    web: bool = typer.Option(False, "--web"),
) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd)

    rid = run_id
    if rid == "last":
        rid = last_run_id(paths.state_dir) or "last"
    try:
        record = load_run_raw(paths.state_dir, rid)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc

    if web:
        html = _render_html(record)
        out_path = Path(out) if out else (paths.state_dir / f"{record['id']}.html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        typer.echo(f"file://{out_path.resolve()}")
        raise typer.Exit(code=EXIT_OK)

    if opts.effective_format == OutputFormat.JSON:
        if out:
            Path(out).write_text(
                json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            typer.echo(f"wrote {out}", err=True)
        else:
            json_dump(record)
    else:
        render_report(record, no_color=opts.no_color)
        if out:
            md = _render_md(record)
            Path(out).write_text(md, encoding="utf-8")
            typer.echo(f"wrote {out}", err=True)
    raise typer.Exit(code=EXIT_OK)


def _render_md(record: dict) -> str:
    lines = [f"# Run {record.get('id')}", ""]
    summary = record.get("summary", {})
    lines.append(
        f"- examples: {summary.get('examples', 0)}  "
        f"passed: {summary.get('passed', 0)}  "
        f"failed: {summary.get('failed', 0)}  "
        f"errors: {summary.get('errors', 0)}"
    )
    lines.append("")
    for tname, t in record.get("tasks", {}).items():
        lines.append(f"## {tname}")
        lines.append("| metric | score | delta | threshold | status |")
        lines.append("|---|---|---|---|---|")
        for mname, m in (t.get("metrics") or {}).items():
            score = "" if m.get("score") is None else f"{m['score']:.4f}"
            delta = "" if m.get("delta") is None else f"{m['delta']:+.4f}"
            lines.append(
                f"| {mname} | {score} | {delta} | {m.get('threshold', 0):.4f} | {m.get('status')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _render_html(record: dict) -> str:
    md = _render_md(record)
    body = (
        md.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>\n")
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>ai-eval run {record.get('id')}</title></head>"
        f"<body><pre>{body}</pre></body></html>"
    )


__all__ = ["report_command"]

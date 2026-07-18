"""`ai-evals bootstrap` — capture real runtime traces into a golden set.

Runs the user's command (everything after ``--``) in a child subprocess with a
generated ``sitecustomize.py`` that monkey-patches the LLM/retrieval
frameworks and records traces to a JSONL sink. After the child exits, the
parent reads the sink, dedups, caps by ``--max``, applies ``--task`` and
``--append``/``--replace`` filters, and merges into ``eval/golden_set.json``.

Exit codes:
- ``0`` captures saved (child exit code ignored — captures are flushed either way)
- ``1`` child crashed (captures still flushed) OR a writer error
- ``2`` usage error (missing command, ``--no-instrument`` rejected)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import typer

from ai_eval.bootstrap import golden_writer as bootstrap_gw
from ai_eval.bootstrap.sitecustomize import build_child_env, write_sitecustomize
from ai_eval.cli.app import (
    EXIT_GENERAL,
    EXIT_OK,
    EXIT_USAGE,
    GlobalOptions,
    OutputFormat,
)
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.config.loader import load_resolved
from ai_eval.storage.paths import resolve_paths


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
    opts: GlobalOptions = ctx.obj

    if not runtime_cmd:
        typer.echo("error: missing runtime command", err=True)
        typer.echo("  what: no `-- <command>` provided", err=True)
        typer.echo("  why:  bootstrap requires an explicit runtime to trace", err=True)
        typer.echo("  fix:  ai-evals bootstrap -- pytest -q", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    if no_instrument:
        typer.echo("error: `--no-instrument` is reserved for Phase 5 (OTEL)", err=True)
        typer.echo("  what: OpenTelemetry bootstrap path not yet implemented", err=True)
        typer.echo("  fix:  omit the flag (instrumentation is on by default)", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    paths = resolve_paths(opts.cwd)
    resolved = load_resolved(project_root=opts.cwd, config_path=opts.config_path)
    try:
        rubrics = resolved.as_rubrics()
    except Exception as exc:
        typer.echo(f"error: rubrics.yaml invalid: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE) from exc

    # Build task specs for the shim (file_path/entry/name per task).
    task_specs: list[dict] = []
    for name, spec in rubrics.tasks.items():
        if task is not None and name != task:
            continue
        task_specs.append(
            {
                "name": name,
                "file_path": spec.file_path,
                "entry": spec.entry or "main",
            }
        )

    golden_path = Path(out) if out else paths.golden_set_json

    with tempfile.TemporaryDirectory(prefix="ai-evals-bootstrap-") as tmp:
        shim_dir = Path(tmp)
        sink_path = shim_dir / "captures.jsonl"
        # ensure sink exists so the child's makedirs is a no-op
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        sink_path.touch()

        write_sitecustomize(
            shim_dir,
            sink_path=sink_path,
            task_specs=task_specs,
            redact=True,
        )
        child_env = build_child_env(
            dict(os.environ),
            shim_dir,
            sink_path,
            task_specs,
            redact=True,
        )

        if opts.effective_format != OutputFormat.JSON and not opts.quiet:
            typer.echo(
                f"running: {' '.join(runtime_cmd)} (capturing traces...)",
                err=True,
            )

        completed = subprocess.run(
            runtime_cmd,
            env=child_env,
            cwd=str(opts.cwd),
        )
        child_exit = completed.returncode

        # Read the sink regardless of child exit code.
        examples: list[dict] = []
        if sink_path.is_file():
            for line in sink_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        summary = bootstrap_gw.append(
            golden_path,
            examples,
            append_mode=append,
            max_per_task=max_captures,
            task_filter=task,
        )

        if opts.effective_format == OutputFormat.JSON:
            json_dump(
                {
                    "golden_set_path": str(golden_path),
                    "child_exit_code": child_exit,
                    "captured": summary["written"],
                    "skipped": summary["skipped"],
                    "tasks": summary["tasks"],
                }
            )
        elif not opts.quiet:
            typer.echo(
                f"saved {summary['written']} capture(s), "
                f"{summary['skipped']} skipped -> {golden_path}",
                err=True,
            )

    # Captures flushed either way; exit 1 if the child crashed so CI notices.
    if child_exit != 0:
        raise typer.Exit(code=EXIT_GENERAL)
    raise typer.Exit(code=EXIT_OK)


__all__ = ["bootstrap_command"]

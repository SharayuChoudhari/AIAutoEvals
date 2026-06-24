"""`ai-eval init` — first-run scaffold.

Per design §1.2:
- runs inference, writes `eval/rubrics.yaml`, `eval/golden_set.json` stub,
  `eval/tests.py`, and ensures `.ai-eval/` state dir exists.
- flags: --force, --dry-run, --judge-default, --judge-regression, --frameworks,
  --include, --exclude.
- exit codes: 0 ok, 1 write failed, 2 invalid flags.
- stdout: human summary table OR JSON object when --format json.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from ai_eval.cli.app import EXIT_GENERAL, EXIT_OK, EXIT_USAGE, GlobalOptions, OutputFormat
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_dry_run_summary, render_init_summary
from ai_eval.config.defaults import DEFAULT_JUDGE, DEFAULT_REGRESSION_JUDGE
from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.synthesize import build_rubrics
from ai_eval.scaffold import golden_writer, rubrics_writer, tests_writer
from ai_eval.scaffold.gitignore_patch import ensure_gitignored
from ai_eval.storage.paths import resolve_paths
from ai_eval.telemetry.logger import get_logger
from ai_eval.telemetry.progress import status


def _split_csv(value: str | None) -> list[str]:
    return [s.strip() for s in (value or "").split(",") if s.strip()]


def _err(message: str, *, what: str, why: str, fix: str) -> None:
    """Per design §1.7: every error has what/why/fix."""
    typer.echo(f"error: {message}", err=True)
    typer.echo(f"  what: {what}", err=True)
    typer.echo(f"  why:  {why}", err=True)
    typer.echo(f"  fix:  {fix}", err=True)


def init_command(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        help="Overwrite existing eval/ files.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print what would be written; no writes.",
    ),
    judge_default: str = typer.Option(
        DEFAULT_JUDGE,
        "--judge-default",
        help="Default judge model id (e.g. ollama/qwen2.5-coder:7b).",
        show_default=True,
    ),
    judge_regression: str = typer.Option(
        DEFAULT_REGRESSION_JUDGE,
        "--judge-regression",
        help="Regression judge model id.",
        show_default=True,
    ),
    frameworks: str = typer.Option(
        "",
        "--frameworks",
        help="Comma list to bias detection (e.g. openai,langchain).",
    ),
    include: list[str] = typer.Option(
        [],
        "--include",
        help="Glob to include (repeatable).",
    ),
    exclude: list[str] = typer.Option(
        [],
        "--exclude",
        help="Glob to exclude (repeatable).",
    ),
) -> None:
    opts: GlobalOptions = ctx.obj
    log = get_logger()
    paths = resolve_paths(opts.cwd, eval_dir=None)

    # Pre-flight: check write conflicts before doing work.
    targets = [paths.rubrics_yaml, paths.golden_set_json, paths.tests_py]
    existing = [p for p in targets if p.exists()]
    if existing and not (force or dry_run):
        rels = ", ".join(str(p.relative_to(opts.cwd)) for p in existing)
        _err(
            "eval/ files already exist",
            what=f"would overwrite: {rels}",
            why="init is a fresh-scaffold command",
            fix="re-run with --force, or use `ai-eval analyze` to merge",
        )
        raise typer.Exit(code=EXIT_USAGE)

    framework_filter = _split_csv(frameworks) or None

    with status(f"scanning {opts.cwd}…", enabled=not opts.quiet):
        scan = scan_repo(
            opts.cwd,
            include=include,
            exclude=exclude,
            framework_filter=framework_filter,
        )

    log.debug("scan complete: %d files, %d tasks", scan.files_scanned, len(scan.tasks))

    rubrics = build_rubrics(
        scan,
        judge_default=judge_default,
        judge_regression=judge_regression,
    )

    tasks_view = [
        (name, spec.type, spec.file_path) for name, spec in rubrics.tasks.items()
    ]

    if dry_run:
        if opts.effective_format == OutputFormat.JSON:
            json_dump(
                {
                    "dry_run": True,
                    "files_scanned": scan.files_scanned,
                    "elapsed_seconds": round(scan.elapsed_seconds, 3),
                    "would_write": [
                        str(paths.rubrics_yaml.relative_to(opts.cwd)),
                        str(paths.golden_set_json.relative_to(opts.cwd)),
                        str(paths.tests_py.relative_to(opts.cwd)),
                    ],
                    "tasks": [
                        {"name": n, "type": t, "file_path": p} for n, t, p in tasks_view
                    ],
                }
            )
        else:
            render_dry_run_summary(
                files_scanned=scan.files_scanned,
                tasks=tasks_view,
                would_write=[
                    str(paths.rubrics_yaml.relative_to(opts.cwd)),
                    str(paths.golden_set_json.relative_to(opts.cwd)),
                    str(paths.tests_py.relative_to(opts.cwd)),
                ],
                no_color=opts.no_color,
            )
        raise typer.Exit(code=EXIT_OK)

    # Write phase.
    written: list[tuple[str, str]] = []
    try:
        rubrics_writer.write(rubrics, paths.rubrics_yaml)
        written.append((str(paths.rubrics_yaml.relative_to(opts.cwd)), "wrote"))

        if paths.golden_set_json.exists() and not force:
            written.append((str(paths.golden_set_json.relative_to(opts.cwd)), "skipped"))
        else:
            golden_writer.write_stub(rubrics, paths.golden_set_json)
            written.append((str(paths.golden_set_json.relative_to(opts.cwd)), "wrote"))

        tests_writer.write(paths.tests_py)
        written.append((str(paths.tests_py.relative_to(opts.cwd)), "wrote"))

        paths.ensure_state()
        written.append((str(paths.state_dir.relative_to(opts.cwd)), "ensured"))

        if ensure_gitignored(opts.cwd / ".gitignore"):
            written.append((".gitignore", "updated"))
    except OSError as exc:
        _err(
            "failed to write scaffold files",
            what=str(exc),
            why="permission denied or disk full",
            fix=f"check write permissions on {opts.cwd}",
        )
        raise typer.Exit(code=EXIT_GENERAL) from exc

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "files_scanned": scan.files_scanned,
                "elapsed_seconds": round(scan.elapsed_seconds, 3),
                "written": [{"path": p, "status": s} for p, s in written],
                "tasks": [
                    {"name": n, "type": t, "file_path": p} for n, t, p in tasks_view
                ],
                "next": "ai-eval bootstrap -- pytest",
            }
        )
    else:
        render_init_summary(
            files_scanned=scan.files_scanned,
            elapsed_seconds=scan.elapsed_seconds,
            written=written,
            tasks=tasks_view,
            next_command="ai-eval bootstrap -- pytest",
            no_color=opts.no_color,
        )
    sys.stdout.flush()
    raise typer.Exit(code=EXIT_OK)


__all__ = ["init_command"]

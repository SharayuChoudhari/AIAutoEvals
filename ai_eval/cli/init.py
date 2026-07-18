"""`ai-evals init` — first-run scaffold.

Per design §1.2:
- runs inference, writes `eval/rubrics.yaml`, `eval/golden_set.json` stub,
  `eval/tests.py`, and ensures `.ai-evals/` state dir exists.
- flags: --force, --dry-run, --judge-default, --judge-regression, --frameworks,
  --include, --exclude.
- exit codes: 0 ok, 1 write failed, 2 invalid flags.
- stdout: human summary table OR JSON object when --format json.

`--force` governs regenerable scaffold (rubrics.yaml, tests.py).
`eval/golden_set.json` is captured user data — it is always preserved unless
the user explicitly passes `--reset-golden`.
"""

from __future__ import annotations

import sys

import typer

from ai_eval.cli.app import EXIT_GENERAL, EXIT_OK, EXIT_USAGE, GlobalOptions, OutputFormat
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_dry_run_summary, render_init_summary
from ai_eval.cli.rubric_engine import (
    VALID_ENGINES,
    build_with_engine,
    fail_engine_error,
    make_caps,
)
from ai_eval.config.defaults import DEFAULT_JUDGE, DEFAULT_REGRESSION_JUDGE
from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.hints import merge_hints
from ai_eval.scaffold import golden_writer, rubrics_writer, tests_writer
from ai_eval.scaffold.gitignore_patch import ensure_gitignored
from ai_eval.scaffold.harness_writer import write_harnesses
from ai_eval.scaffold.hints_writer import write_hints_template
from ai_eval.scaffold.seeder import seed_golden_set
from ai_eval.storage.paths import resolve_paths
from ai_eval.telemetry.logger import get_logger
from ai_eval.telemetry.progress import status

#: Default rubric engine. `hybrid` runs the rule detectors for grounded
#: evidence, then the SLM for classification. Use `rules` to skip the SLM.
DEFAULT_RUBRIC_ENGINE = "hybrid"


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
        help=(
            "Overwrite existing rubrics.yaml and tests.py. Never touches captured golden-set data."
        ),
    ),
    reset_golden: bool = typer.Option(
        False,
        "--reset-golden",
        help="Also reset golden_set.json to an empty stub, discarding all captured traces.",
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
    rubric_engine: str = typer.Option(
        DEFAULT_RUBRIC_ENGINE,
        "--rubric-engine",
        help=(
            "Rubric generation engine: rules (rule-only), slm (SLM-only), hybrid (rules then SLM)."
        ),
    ),
    rubric_model: str = typer.Option(
        DEFAULT_JUDGE,
        "--rubric-model",
        help="SLM model id (litellm format) for slm/hybrid engines.",
        show_default=True,
    ),
    rubric_max_snippet_chars: int = typer.Option(
        1500,
        "--rubric-max-snippet-chars",
        help="Max chars of the enclosing-function snippet sent to the SLM per task.",
        show_default=True,
    ),
    rubric_max_tasks: int = typer.Option(
        25,
        "--rubric-max-tasks",
        help="Max number of detected tasks sent to the SLM (caps cost).",
        show_default=True,
    ),
    rubric_budget_tokens: int = typer.Option(
        0,
        "--rubric-budget-tokens",
        help="Hard total prompt token budget (0 = no cap).",
        show_default=True,
    ),
) -> None:
    opts: GlobalOptions = ctx.obj
    log = get_logger()
    paths = resolve_paths(opts.cwd, eval_dir=None)

    if rubric_engine not in VALID_ENGINES:
        _err(
            "invalid --rubric-engine",
            what=f"got {rubric_engine!r}",
            why=f"--rubric-engine must be one of: {', '.join(VALID_ENGINES)}",
            fix=f"re-run with --rubric-engine {'|'.join(VALID_ENGINES)}",
        )
        raise typer.Exit(code=EXIT_USAGE)

    # Pre-flight: only scaffold files (rubrics.yaml, tests.py) trigger --force.
    # golden_set.json is captured user data — handled separately below.
    scaffold_targets = [paths.rubrics_yaml, paths.tests_py]
    existing_scaffold = [p for p in scaffold_targets if p.exists()]

    # Dry-run: classify what the real run would actually do.
    scaffold_conflicts = existing_scaffold and not force
    golden_has_captures = golden_writer.has_real_captures(paths.golden_set_json)

    if dry_run:
        would_write: list[str] = []
        requires_force: list[str] = []
        for p in scaffold_targets:
            if p.exists() and not force:
                requires_force.append(str(p.relative_to(opts.cwd)))
            else:
                would_write.append(str(p.relative_to(opts.cwd)))

        if not paths.golden_set_json.exists():
            would_write.append(str(paths.golden_set_json.relative_to(opts.cwd)))
        elif golden_has_captures:
            if reset_golden:
                would_write.append(
                    f"{paths.golden_set_json.relative_to(opts.cwd)} (RESET — discards captures)"
                )
            else:
                would_write.append(
                    f"{paths.golden_set_json.relative_to(opts.cwd)} (merge — new task keys added)"
                )
        else:
            would_write.append(str(paths.golden_set_json.relative_to(opts.cwd)))

        # Run inference even in dry-run so we can show task counts.
        framework_filter = _split_csv(frameworks) or None
        with status(f"scanning {opts.cwd}…", enabled=not opts.quiet):
            scan = scan_repo(
                opts.cwd,
                include=include,
                exclude=exclude,
                framework_filter=framework_filter,
            )
        # Merge opt-in hint tasks before the rubric engine runs so both rules
        # and SLM/hybrid see them. A malformed hints file is an error even in
        # dry-run (the user would hit it on the real run too).
        try:
            scan = merge_hints(scan, paths.hints_yaml)
        except Exception as exc:
            _err(
                "failed to load hints file",
                what=str(exc).splitlines()[0] if str(exc) else "hints file error",
                why="eval/ai-evals.hints.yaml is malformed or violates the schema",
                fix="fix the hints file, or remove it to skip hints",
            )
            raise typer.Exit(code=EXIT_GENERAL) from exc
        try:
            result = build_with_engine(
                engine=rubric_engine,
                scan=scan,
                project_root=opts.cwd,
                model=rubric_model,
                judge_default=judge_default,
                judge_regression=judge_regression,
                caps=make_caps(rubric_max_snippet_chars, rubric_max_tasks, rubric_budget_tokens),
                hints_path=paths.hints_yaml,
            )
        except typer.Exit:
            raise
        except Exception as exc:
            fail_engine_error(exc)
            return  # unreachable; fail_engine_error exits
        rubrics = result.rubrics
        tasks_view = [(n, s.type, s.file_path) for n, s in rubrics.tasks.items()]

        if opts.effective_format == OutputFormat.JSON:
            json_dump(
                {
                    "schema_version": rubrics.schema_version,
                    "dry_run": True,
                    "files_scanned": scan.files_scanned,
                    "elapsed_seconds": round(scan.elapsed_seconds, 3),
                    "would_write": would_write,
                    "requires_force": requires_force,
                    "rubric_engine": rubrics.rubric_engine,
                    "tasks": [{"name": n, "type": t, "file_path": p} for n, t, p in tasks_view],
                }
            )
        else:
            render_dry_run_summary(
                files_scanned=scan.files_scanned,
                tasks=tasks_view,
                would_write=would_write,
                no_color=opts.no_color,
            )
            if requires_force:
                typer.echo(
                    f"  note: the following already exist and require --force: "
                    f"{', '.join(requires_force)}",
                    err=True,
                )
        raise typer.Exit(code=EXIT_OK)

    # Real-run pre-flight.
    if scaffold_conflicts:
        rels = ", ".join(str(p.relative_to(opts.cwd)) for p in existing_scaffold)
        _err(
            "eval/ scaffold files already exist",
            what=f"would overwrite: {rels}",
            why="init is a fresh-scaffold command",
            fix="re-run with --force, or use `ai-evals analyze` to merge rubrics changes",
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
    try:
        scan = merge_hints(scan, paths.hints_yaml)
    except Exception as exc:
        _err(
            "failed to load hints file",
            what=str(exc).splitlines()[0] if str(exc) else "hints file error",
            why="eval/ai-evals.hints.yaml is malformed or violates the schema",
            fix="fix the hints file, or remove it to skip hints",
        )
        raise typer.Exit(code=EXIT_GENERAL) from exc

    log.debug("scan complete: %d files, %d tasks", scan.files_scanned, len(scan.tasks))

    try:
        result = build_with_engine(
            engine=rubric_engine,
            scan=scan,
            project_root=opts.cwd,
            model=rubric_model,
            judge_default=judge_default,
            judge_regression=judge_regression,
            caps=make_caps(rubric_max_snippet_chars, rubric_max_tasks, rubric_budget_tokens),
            hints_path=paths.hints_yaml,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        fail_engine_error(exc)
        return  # unreachable; fail_engine_error exits
    rubrics = result.rubrics
    tasks_view = [(n, s.type, s.file_path) for n, s in rubrics.tasks.items()]

    # Write phase.
    written: list[tuple[str, str]] = []
    try:
        rubrics_writer.write(rubrics, paths.rubrics_yaml)
        written.append((str(paths.rubrics_yaml.relative_to(opts.cwd)), "wrote"))

        # golden_set.json: always preserve real captures unless --reset-golden.
        golden_status = golden_writer.write_stub(
            rubrics,
            paths.golden_set_json,
            overwrite=reset_golden,
        )
        written.append((str(paths.golden_set_json.relative_to(opts.cwd)), golden_status))

        tests_writer.write(paths.tests_py)
        written.append((str(paths.tests_py.relative_to(opts.cwd)), "wrote"))

        # IO-coupled tasks get a stub harness (D5): monkey-patches their
        # self.<dao>.<method>() reads with canned fixtures so `run` is green
        # without a live DB/HTTP backend. Region-split; regenerable wiring is
        # AST-hash-gated, fixtures preserved across regenerations. The
        # returned IO-coupled names are not used for seeding anymore (the
        # ``_Stub`` path is removed — AGENTS.md §1): IO-coupled entry points
        # require ``ai-evals bootstrap`` to capture a real trace.
        harness_written, _io_coupled_names = write_harnesses(
            rubrics, paths.eval_dir, project_root=opts.cwd
        )
        for hname, hstatus in harness_written:
            written.append((str(paths.eval_dir.relative_to(opts.cwd) / hname), hstatus))

        # Hybrid golden-set seeding (D6): auto-seed shape-varied inputs for
        # top-level pure-LLM tasks so ``run`` produces meaningful scores
        # immediately. IO-coupled entry points are no longer auto-seeded (the
        # ``_Stub`` path is removed — AGENTS.md §1): they require
        # ``ai-evals bootstrap`` to capture a real trace. Runs after
        # write_stub so the task keys exist; auto-seeds are marked
        # ``seed: auto`` so they don't count as real captures for the next
        # init's merge decision.
        if rubrics.tasks:
            seed_golden_set(rubrics, paths.golden_set_json)

        # Emit a commented-out hints template on first init only — never
        # overwrite a user-edited hints file (re-runs preserve edits, mirroring
        # the golden-set preservation rule).
        hints_status = write_hints_template(paths.hints_yaml)
        if hints_status is not None:
            written.append((str(paths.hints_yaml.relative_to(opts.cwd)), hints_status))

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
                "rubric_engine": rubrics.rubric_engine,
                "tasks": [{"name": n, "type": t, "file_path": p} for n, t, p in tasks_view],
                "next": "ai-evals bootstrap -- pytest",
            }
        )
    else:
        render_init_summary(
            files_scanned=scan.files_scanned,
            elapsed_seconds=scan.elapsed_seconds,
            written=written,
            tasks=tasks_view,
            next_command="ai-evals bootstrap -- pytest",
            no_color=opts.no_color,
        )
    sys.stdout.flush()
    raise typer.Exit(code=EXIT_OK)


__all__ = ["init_command"]

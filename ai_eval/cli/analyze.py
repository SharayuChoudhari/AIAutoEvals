"""`ai-eval analyze` — re-scan and propose a merged rubrics.yaml.

Default mode is `--dry-run`. With `--write`, applies the chosen `--merge-strategy`.
"""

from __future__ import annotations

import difflib
from enum import Enum
from pathlib import Path

import typer
import yaml

from ai_eval.cli.app import EXIT_GENERAL, EXIT_OK, EXIT_USAGE, GlobalOptions, OutputFormat
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import stdout_console
from ai_eval.config.defaults import DEFAULT_JUDGE, DEFAULT_REGRESSION_JUDGE
from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.synthesize import build_rubrics
from ai_eval.scaffold import rubrics_writer
from ai_eval.storage.paths import resolve_paths


class MergeStrategy(str, Enum):
    KEEP = "keep"              # keep existing values on conflict
    OVERWRITE = "overwrite"    # take new values on conflict
    PROMPT = "prompt"          # ask (forbidden under --no-input)


def _err(message: str, *, what: str, why: str, fix: str) -> None:
    typer.echo(f"error: {message}", err=True)
    typer.echo(f"  what: {what}", err=True)
    typer.echo(f"  why:  {why}", err=True)
    typer.echo(f"  fix:  {fix}", err=True)


def _merge_dicts(
    existing: dict,
    incoming: dict,
    strategy: MergeStrategy,
    conflicts: list[str],
    prefix: str = "",
) -> dict:
    out = dict(existing)
    for key, new_value in incoming.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in out:
            out[key] = new_value
            continue
        old_value = out[key]
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            out[key] = _merge_dicts(old_value, new_value, strategy, conflicts, dotted)
            continue
        if old_value == new_value:
            continue
        conflicts.append(dotted)
        if strategy == MergeStrategy.OVERWRITE:
            out[key] = new_value
        # KEEP / PROMPT: leave existing in place (PROMPT bails out earlier).
    return out


def analyze_command(
    ctx: typer.Context,
    write: bool = typer.Option(
        False,
        "--write",
        help="Apply the merge (default is dry-run).",
    ),
    merge_strategy: MergeStrategy = typer.Option(
        MergeStrategy.KEEP,
        "--merge-strategy",
        help="How to resolve conflicts.",
        case_sensitive=False,
    ),
    show_diff: bool = typer.Option(
        False,
        "--diff",
        help="Print a unified yaml diff.",
    ),
) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd, eval_dir=None)

    if not paths.rubrics_yaml.is_file():
        _err(
            "no rubrics.yaml to analyze",
            what=f"{paths.rubrics_yaml} does not exist",
            why="analyze re-syncs an existing rubrics.yaml; nothing to sync yet",
            fix="run `ai-eval init` first",
        )
        raise typer.Exit(code=EXIT_USAGE)

    if merge_strategy == MergeStrategy.PROMPT and opts.no_input:
        _err(
            "interactive merge prompted under --no-input",
            what="--merge-strategy prompt requires user input",
            why="--no-input is set (or CI=true auto-set it)",
            fix="re-run with --merge-strategy keep or --merge-strategy overwrite",
        )
        raise typer.Exit(code=EXIT_USAGE)

    existing = yaml.safe_load(paths.rubrics_yaml.read_text(encoding="utf-8")) or {}

    scan = scan_repo(opts.cwd)
    rubrics = build_rubrics(
        scan,
        judge_default=existing.get("judge", {}).get("default") or DEFAULT_JUDGE,
        judge_regression=existing.get("judge", {}).get("regression_check")
        or DEFAULT_REGRESSION_JUDGE,
    )
    incoming = rubrics_writer.to_dict(rubrics)

    conflicts: list[str] = []
    merged = _merge_dicts(existing, incoming, merge_strategy, conflicts)

    old_yaml = yaml.safe_dump(existing, sort_keys=False) if existing else ""
    new_yaml = yaml.safe_dump(merged, sort_keys=False)

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "wrote": write,
                "merge_strategy": merge_strategy.value,
                "conflicts": conflicts,
                "files_scanned": scan.files_scanned,
                "tasks": [
                    {"name": n, "type": s.type, "file_path": s.file_path}
                    for n, s in rubrics.tasks.items()
                ],
            }
        )
    else:
        console = stdout_console(no_color=opts.no_color)
        if show_diff:
            diff = difflib.unified_diff(
                old_yaml.splitlines(),
                new_yaml.splitlines(),
                fromfile="rubrics.yaml (current)",
                tofile="rubrics.yaml (proposed)",
                lineterm="",
            )
            for line in diff:
                console.print(line, highlight=False)
        console.print(
            f"- conflicts: {len(conflicts)}  strategy: {merge_strategy.value}  "
            f"will{'' if write else ' NOT'} write"
        )

    if merge_strategy == MergeStrategy.PROMPT and conflicts and not opts.assume_yes:
        _err(
            "merge conflicts require explicit strategy",
            what=f"{len(conflicts)} conflicting keys",
            why="--merge-strategy prompt without -y",
            fix="re-run with --merge-strategy keep|overwrite or pass --yes",
        )
        raise typer.Exit(code=EXIT_GENERAL)

    if write:
        try:
            paths.rubrics_yaml.write_text(new_yaml, encoding="utf-8")
        except OSError as exc:
            _err(
                "failed to write rubrics.yaml",
                what=str(exc),
                why="permission denied or disk full",
                fix=f"check write permissions on {paths.rubrics_yaml}",
            )
            raise typer.Exit(code=EXIT_GENERAL) from exc

    raise typer.Exit(code=EXIT_OK)


__all__ = ["analyze_command"]

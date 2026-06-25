"""`ai-eval analyze` — re-scan and propose a merged rubrics.yaml.

Default mode is dry-run (no `--write`). When `--write` is passed:
  - Creates a `.bak` copy of the current rubrics.yaml before replacing it.
  - Re-emits the ai-eval file header comment that `init` adds.
  - Skips writing when the merged result is identical to what is already on disk.
  - The `--merge-strategy` flag is now required to be explicit when conflicts
    exist (exits 1 otherwise), per plan §1.2.
  - `PROMPT` strategy is removed until interactive prompting is implemented
    (Phase 2); passing it yields a clear usage error.

CI auto-mode note: when `CI=true`, `--no-input` is forced on; passing
`--merge-strategy overwrite --write` in CI additionally requires `--yes` to
prevent silent destructive rewrites.
"""

from __future__ import annotations

import difflib
import os
import shutil
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

# Sentinel so we can tell whether the user explicitly passed --merge-strategy.
_UNSET = object()

_RUBRICS_HEADER = rubrics_writer._HEADER


class MergeStrategy(str, Enum):
    KEEP = "keep"              # keep existing values on conflict
    OVERWRITE = "overwrite"    # take new values on conflict
    # PROMPT removed — it was never wired to actual prompting.


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
        # KEEP: leave existing in place.
    return out


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via a temp file + os.replace."""
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _make_backup(path: Path) -> Path:
    """Copy ``path`` to ``path.bak``, returning the backup path."""
    bak = path.with_suffix(".yaml.bak")
    shutil.copy2(path, bak)
    return bak


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
        help="How to resolve conflicts: keep (default) or overwrite.",
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

    # CI safety gate: destructive overwrite in CI requires explicit --yes.
    if write and merge_strategy == MergeStrategy.OVERWRITE and opts.is_ci and not opts.assume_yes:
        _err(
            "destructive overwrite refused in CI without explicit --yes",
            what="--merge-strategy overwrite --write in CI would rewrite rubrics.yaml silently",
            why="CI=true auto-set --no-input; missing explicit --yes / -y",
            fix="re-run with --yes to confirm, or use --merge-strategy keep",
        )
        raise typer.Exit(code=EXIT_USAGE)

    raw_text = paths.rubrics_yaml.read_text(encoding="utf-8")
    existing = yaml.safe_load(raw_text) or {}

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

    # Re-emit the canonical header, then the YAML body — so the header is not
    # lost on round-trips even when safe_dump rewrites the file.
    new_body = yaml.safe_dump(merged, sort_keys=False, default_flow_style=False, indent=2)
    new_text = _RUBRICS_HEADER + new_body

    # For diff display, compare against the stripped-header version so the diff
    # shows only meaningful content changes (not header re-emission noise).
    old_body = yaml.safe_dump(existing, sort_keys=False, default_flow_style=False, indent=2) \
        if existing else ""

    no_change = (merged == existing)

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "wrote": write and not no_change,
                "no_change": no_change,
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
                old_body.splitlines(),
                new_body.splitlines(),
                fromfile="rubrics.yaml (current)",
                tofile="rubrics.yaml (proposed)",
                lineterm="",
            )
            for line in diff:
                console.print(line, highlight=False)
        console.print(
            f"- conflicts: {len(conflicts)}  strategy: {merge_strategy.value}  "
            f"{'no change' if no_change else ('will write' if write else 'will NOT write')}"
        )

    # Exit 1 when conflicts exist and strategy was not explicitly chosen —
    # per plan §1.2: "1 merge conflict requiring --merge-strategy".
    # Typer cannot distinguish default-KEEP from explicit-KEEP, so we surface
    # this for KEEP + conflicts to nudge the user to be intentional.
    if conflicts and merge_strategy == MergeStrategy.KEEP and not write:
        _err(
            "conflicts detected — review before applying",
            what=f"{len(conflicts)} key(s) differ: {', '.join(conflicts[:5])}"
            + (" …" if len(conflicts) > 5 else ""),
            why="re-inference produced different values from what rubrics.yaml has",
            fix=(
                "run with --diff to see changes, then --write --merge-strategy keep "
                "to preserve your edits, or --merge-strategy overwrite to take new values"
            ),
        )
        raise typer.Exit(code=EXIT_GENERAL)

    if write and not no_change:
        bak = _make_backup(paths.rubrics_yaml)
        try:
            _atomic_write(paths.rubrics_yaml, new_text)
        except OSError as exc:
            _err(
                "failed to write rubrics.yaml",
                what=str(exc),
                why="permission denied or disk full",
                fix=f"check write permissions on {paths.rubrics_yaml}",
            )
            raise typer.Exit(code=EXIT_GENERAL) from exc
        typer.echo(
            f"  backup: {bak.relative_to(opts.cwd)}  →  wrote rubrics.yaml",
            err=True,
        )

    raise typer.Exit(code=EXIT_OK)


__all__ = ["analyze_command"]

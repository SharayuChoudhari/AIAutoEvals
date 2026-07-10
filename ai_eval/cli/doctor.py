"""`ai-evals doctor` — read-only environment check.

Per plan §1.2 and §2.12: doctor is **always safe, read-only**. It never
creates directories, never writes files, and never modifies any state.

Exit codes (plan §1.8):
  0 — all checks pass
  1 — any required check fails (so CI can gate on `ai-evals doctor`)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import typer

from ai_eval.cli.app import EXIT_GENERAL, EXIT_OK, GlobalOptions, OutputFormat
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_doctor
from ai_eval.config.loader import load_resolved
from ai_eval.storage.paths import resolve_paths


def _check_python() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    return ok, f"python {major}.{minor} ({'>=3.10 ok' if ok else 'need >=3.10'})"


def _check_module(name: str, *, optional: bool = False) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    if spec is not None:
        return True, "installed"
    label = "not installed (optional)" if optional else "not installed"
    # Optional deps are always ok=True so they don't affect exit code.
    return optional, label


def _check_writable(path: Path) -> tuple[bool, str]:
    """Non-mutating writability probe.

    - If ``path`` already exists: check ``os.access`` for write permission.
    - If ``path`` does not exist: check that the nearest existing ancestor is
      writable (which implies we can create ``path``).

    Never creates directories or files.
    """
    target = path
    while not target.exists():
        parent = target.parent
        if parent == target:
            return False, f"no ancestor of {path} exists"
        target = parent
    if os.access(target, os.W_OK):
        return True, str(path)
    return False, f"not writable: {target}"


def _check_rubrics(cwd: Path, config_path: Path | None) -> tuple[bool, str]:
    """Return (ok, detail).

    - Absent rubrics.yaml → ok=False ("not initialised" category, required).
    - Present but invalid → ok=False ("invalid schema" category, required).
    - Present and valid   → ok=True.
    """
    resolved = load_resolved(project_root=cwd, config_path=config_path)
    if resolved.rubrics_path is None:
        return False, "not found — run `ai-evals init`"
    try:
        resolved.as_rubrics()
    except Exception as exc:
        return False, f"invalid: {exc}"
    return True, str(resolved.rubrics_path)


def _check_provider_creds() -> tuple[bool, str]:
    """At least one LLM provider credential must be present for `run` to work."""
    candidates = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "OLLAMA_HOST",
    ]
    found = [k for k in candidates if os.environ.get(k)]
    if found:
        return True, "set: " + ", ".join(found)
    return False, "no provider creds found (set OPENAI_API_KEY or run `ollama serve`)"


def _check_judge_gateway(
    cwd: Path, config_path: Path | None
) -> tuple[bool, str]:
    """Best-effort ping of ``judge.default``. Optional/required=False.

    A failure here is informational — it warns without forcing exit 1 unless
    every other provider check also failed. The ping is timeout-bounded and
    never raises.
    """
    resolved = load_resolved(project_root=cwd, config_path=config_path)
    judge = (resolved.data.get("judge") or {}) if isinstance(resolved.data, dict) else {}
    model = judge.get("default")
    if not model:
        return False, "judge.default not configured (run `ai-evals init`)"
    try:
        import asyncio

        from ai_eval.judge.gateway import ping

        # Short timeout: doctor must stay fast and read-only. A longer reachability
        # probe belongs to `ai-evals judge --ping`.
        ok, detail = asyncio.run(ping(model, timeout=5.0))
        return ok, f"{model}: {detail}"
    except Exception as exc:
        return False, f"judge ping unavailable: {type(exc).__name__}: {exc}"


def doctor_command(ctx: typer.Context) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd, eval_dir=None)

    # (name, ok, detail, required)
    checks: list[tuple[str, bool, str, bool]] = []

    def chk(name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        checks.append((name, ok, detail, required))

    chk("python version", *_check_python())
    chk("pydantic", *_check_module("pydantic"))
    chk("typer", *_check_module("typer"))
    chk("rich", *_check_module("rich"))
    chk("yaml", *_check_module("yaml"))
    chk("litellm", *_check_module("litellm", optional=True), required=False)
    chk("instructor", *_check_module("instructor", optional=True), required=False)
    chk("state dir writable", *_check_writable(paths.state_dir))
    chk("rubrics.yaml", *_check_rubrics(opts.cwd, opts.config_path))
    chk("provider credentials", *_check_provider_creds())
    chk(
        "judge gateway reachable",
        *_check_judge_gateway(opts.cwd, opts.config_path),
        required=False,
    )

    # Plan §1.2 / §1.8: exit 1 if any REQUIRED check fails (so CI can gate).
    # Optional checks (litellm, instructor, judge gateway) warn but never force
    # exit 1 — their failure doesn't block `init`/`analyze` or `--rubric-engine
    # rules` runs.
    any_failed = any(not ok for _, ok, _, req in checks if req)

    render_checks = [(name, ok, detail) for name, ok, detail, _ in checks]

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "ok": not any_failed,
                "checks": [
                    {"name": n, "ok": ok, "detail": detail, "required": req}
                    for n, ok, detail, req in checks
                ],
            }
        )
    else:
        render_doctor(render_checks, no_color=opts.no_color)

    raise typer.Exit(code=EXIT_GENERAL if any_failed else EXIT_OK)


__all__ = ["doctor_command"]

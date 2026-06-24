"""`ai-eval doctor` — read-only environment check."""

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
    return ok, f"python {major}.{minor} ({'>=3.10' if ok else 'too old'})"


def _check_module(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    return (spec is not None, "installed" if spec else "not installed")


def _check_writable(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, str(path)
    except OSError as exc:
        return False, f"{path}: {exc}"


def _check_rubrics(cwd: Path, config_path: Path | None) -> tuple[bool, str]:
    resolved = load_resolved(project_root=cwd, config_path=config_path)
    if resolved.rubrics_path is None:
        return False, "no rubrics.yaml — run `ai-eval init`"
    try:
        resolved.as_rubrics()
    except Exception as exc:
        return False, f"invalid rubrics.yaml: {exc}"
    return True, str(resolved.rubrics_path)


def _check_provider_creds() -> tuple[bool, str]:
    """Soft-check provider creds: pass if at least one of common keys is set."""
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
    return False, "no provider creds found (set OPENAI_API_KEY or run ollama serve)"


def doctor_command(ctx: typer.Context) -> None:
    opts: GlobalOptions = ctx.obj
    paths = resolve_paths(opts.cwd, eval_dir=None)

    checks: list[tuple[str, bool, str]] = []
    checks.append(("python version", *_check_python()))
    checks.append(("pydantic", *_check_module("pydantic")))
    checks.append(("typer", *_check_module("typer")))
    checks.append(("rich", *_check_module("rich")))
    checks.append(("yaml", *_check_module("yaml")))
    checks.append(("litellm (optional)", *_check_module("litellm")))
    checks.append(("instructor (optional)", *_check_module("instructor")))
    checks.append(("state dir writable", *_check_writable(paths.state_dir)))
    checks.append(("rubrics.yaml", *_check_rubrics(opts.cwd, opts.config_path)))
    checks.append(("provider credentials", *_check_provider_creds()))

    # Optional checks (litellm, instructor, creds, rubrics-not-yet-init) don't fail doctor.
    required = {"python version", "pydantic", "typer", "rich", "yaml", "state dir writable"}
    all_required_ok = all(ok for name, ok, _ in checks if name in required)

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "ok": all_required_ok,
                "checks": [
                    {"name": n, "ok": ok, "detail": detail} for n, ok, detail in checks
                ],
            }
        )
    else:
        render_doctor(checks, no_color=opts.no_color)

    raise typer.Exit(code=EXIT_OK if all_required_ok else EXIT_GENERAL)


__all__ = ["doctor_command"]

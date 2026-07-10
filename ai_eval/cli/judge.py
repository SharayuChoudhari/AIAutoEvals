"""`ai-eval judge` — inspect and test the configured judge models.

Three read-only modes, all honoring ``--format json``:
- ``--list`` (default) — show ``judge.default`` / ``regression_check`` /
  ``fallback`` with a best-effort reachability ping.
- ``--ping <model>`` — one-shot reachability probe of a single model id.
- ``--prompt <text> --model <m>`` — one-shot completion via the gateway.

Reachability pings are timeout-bounded (default 10s) and never raise; failures
appear as ``unreachable`` rows / JSON entries.
"""

from __future__ import annotations

import asyncio
from typing import Any

import typer

from ai_eval.cli.app import EXIT_GENERAL, EXIT_OK, EXIT_USAGE, GlobalOptions, OutputFormat
from ai_eval.cli.render.json_out import dump as json_dump
from ai_eval.cli.render.tables import render_judge_list, render_judge_prompt
from ai_eval.config.loader import load_resolved

JUDGE_PING_TIMEOUT = 10.0


def judge_command(
    ctx: typer.Context,
    list_models: bool = typer.Option(False, "--list"),
    ping: str | None = typer.Option(None, "--ping"),
    prompt: str | None = typer.Option(None, "--prompt"),
    model: str | None = typer.Option(None, "--model"),
) -> None:
    opts: GlobalOptions = ctx.obj
    resolved = load_resolved(project_root=opts.cwd, config_path=opts.config_path)

    judge_cfg: dict[str, Any] = resolved.data.get("judge", {}) or {}
    default_model = model or judge_cfg.get("default")
    fallback_models = list(judge_cfg.get("fallback", []) or [])
    regression_model = judge_cfg.get("regression_check")

    # Exactly one mode; default to --list when none given.
    modes = sum(bool(x) for x in (list_models, ping is not None, prompt is not None))
    if modes > 1:
        typer.echo(
            "error: --list, --ping, and --prompt are mutually exclusive", err=True
        )
        raise typer.Exit(code=EXIT_USAGE)
    if prompt is not None and model is None and default_model is None:
        typer.echo(
            "error: --prompt requires --model or a configured judge.default",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    if ping is not None:
        _run_ping(opts, ping)
        return
    if prompt is not None:
        _run_prompt(opts, default_model, prompt, fallback_models)
        return
    _run_list(opts, default_model, regression_model, fallback_models)


def _run_list(
    opts: GlobalOptions,
    default_model: str | None,
    regression_model: str | None,
    fallback_models: list[str],
) -> None:
    targets: list[tuple[str, str]] = []
    if default_model:
        targets.append(("default", default_model))
    if regression_model:
        targets.append(("regression_check", regression_model))
    for i, m in enumerate(fallback_models):
        targets.append((f"fallback[{i}]", m))
    if not targets:
        typer.echo("error: no judge models configured (run `ai-eval init`)", err=True)
        raise typer.Exit(code=EXIT_GENERAL)

    rows = [(role, mdl, *_ping_sync(mdl)) for role, mdl in targets]

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "models": [
                    {"role": role, "model": mdl, "reachable": ok, "detail": detail}
                    for role, mdl, ok, detail in rows
                ],
            }
        )
    else:
        render_judge_list(rows, no_color=opts.no_color)
    raise typer.Exit(code=EXIT_OK)


def _run_ping(opts: GlobalOptions, model: str) -> None:
    ok, detail = _ping_sync(model)
    if opts.effective_format == OutputFormat.JSON:
        json_dump({"model": model, "reachable": ok, "detail": detail})
    else:
        glyph = "ok" if ok else "fail"
        typer.echo(f"{glyph} {model}  {detail}")
    raise typer.Exit(code=EXIT_OK if ok else EXIT_GENERAL)


def _run_prompt(
    opts: GlobalOptions,
    model: str | None,
    prompt_text: str,
    fallback_models: list[str],
) -> None:
    from ai_eval.judge.gateway import score
    from ai_eval.judge.schemas import JudgeRequest

    assert model is not None  # checked by caller
    request = JudgeRequest(
        task_name="cli_prompt", task_type="custom", metric="cli_prompt", example={}
    )
    messages = [
        {"role": "system", "content": "Respond helpfully."},
        {"role": "user", "content": prompt_text},
    ]
    try:
        response, errors = asyncio.run(
            score(
                request,
                model=model,
                prompt_messages=messages,
                scored_dimension="score",
                fallback_models=tuple(fallback_models),
            )
        )
    except Exception as exc:
        typer.echo(f"error: judge call failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_GENERAL) from exc

    if response is None:
        typer.echo(
            "error: all judge models failed: "
            + "; ".join(f"{e.model}: {e.message}" for e in errors),
            err=True,
        )
        raise typer.Exit(code=EXIT_GENERAL)

    if opts.effective_format == OutputFormat.JSON:
        json_dump(
            {
                "model": model,
                "response": _response_to_dict(response),
                "errors": [e.model_dump() for e in errors],
            }
        )
    else:
        render_judge_prompt(model, response, no_color=opts.no_color)
    raise typer.Exit(code=EXIT_OK)


def _response_to_dict(response) -> dict[str, Any]:
    return response.model_dump(mode="json")


def _ping_sync(model: str) -> tuple[bool, str]:
    """Run the async ping in a fresh event loop; never raises."""
    from ai_eval.judge.gateway import ping as _ping

    try:
        return asyncio.run(_ping(model, timeout=JUDGE_PING_TIMEOUT))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


__all__ = ["judge_command"]

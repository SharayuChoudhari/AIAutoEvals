"""CLI tests for `ai-evals judge` with a fake transport.

The CLI calls the gateway's async ``ping``/``score`` which default to the real
instructor client. We patch ``ai_eval.judge.gateway._default_complete`` with a
fake async callable so no network is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli.app import app


@pytest.fixture
def fake_complete(monkeypatch: pytest.MonkeyPatch):
    """Patch the gateway's default transport with a canned async fn.

    Returns a mutable holder so individual tests can swap responses.
    """
    holder = {"responses": {}, "raise_default": False}

    async def _fake(*, model, messages, response_model, temperature=0.0):
        if holder["raise_default"]:
            raise RuntimeError("unreachable")
        if model in holder["responses"]:
            val = holder["responses"][model]
            if isinstance(val, Exception):
                raise val
            return response_model.model_validate(val)
        # default: for ping -> ok True; for score -> 0.9
        try:
            return response_model.model_validate({"ok": True})
        except Exception:
            return response_model.model_validate(
                holder["responses"].get("__default__", {"score": 0.9, "rationale": "ok"})
            )

    import ai_eval.judge.gateway as gw

    monkeypatch.setattr(gw, "_default_complete", _fake)
    return holder


def _rubrics(tmp_path: Path, default: str = "fake/local-model") -> None:
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_type": "custom",
                "judge": {
                    "default": default,
                    "regression_check": "fake/regression",
                    "fallback": ["fake/fb1"],
                },
                "defaults": {"parallel": 4, "cache": True, "tolerance": 0.02},
                "tasks": {},
            }
        ),
        encoding="utf-8",
    )


def test_judge_list_json(runner: CliRunner, tmp_path: Path, fake_complete, clean_env) -> None:
    _rubrics(tmp_path)
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "judge", "--list"])
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    roles = [m["role"] for m in payload["models"]]
    assert "default" in roles and "regression_check" in roles
    assert "fallback[0]" in roles
    # all reachable via the fake transport (ping returns ok=True)
    assert all(m["reachable"] for m in payload["models"])


def test_judge_list_human(runner: CliRunner, tmp_path: Path, fake_complete, clean_env) -> None:
    _rubrics(tmp_path)
    result = runner.invoke(app, ["-C", str(tmp_path), "judge", "--list"])
    assert result.exit_code == 0
    assert "fake/local-model" in result.output


def test_judge_ping_ok(runner: CliRunner, tmp_path: Path, fake_complete, clean_env) -> None:
    _rubrics(tmp_path)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "judge", "--ping", "fake/local"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["reachable"] is True


def test_judge_ping_fail(runner: CliRunner, tmp_path: Path, fake_complete, clean_env) -> None:
    _rubrics(tmp_path)
    fake_complete["raise_default"] = True
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "judge", "--ping", "fake/local"]
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["reachable"] is False


def test_judge_prompt_json(runner: CliRunner, tmp_path: Path, fake_complete, clean_env) -> None:
    _rubrics(tmp_path)
    fake_complete["responses"]["fake/local-model"] = {
        "score": 0.8,
        "rationale": "good",
        "sub_scores": {"score": 0.8},
    }
    result = runner.invoke(
        app,
        [
            "-C",
            str(tmp_path),
            "--format",
            "json",
            "judge",
            "--prompt",
            "say hi",
            "--model",
            "fake/local-model",
        ],
    )
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    assert payload["model"] == "fake/local-model"
    assert payload["response"]["score"] == 0.8


def test_judge_prompt_human(runner: CliRunner, tmp_path: Path, fake_complete, clean_env) -> None:
    _rubrics(tmp_path)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "judge", "--prompt", "hi", "--model", "fake/local"]
    )
    assert result.exit_code == 0
    assert "score" in result.output.lower()


def test_judge_list_with_defaults_only(
    runner: CliRunner, tmp_path: Path, fake_complete, clean_env
) -> None:
    # no rubrics.yaml -> falls back to built-in judge.default
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "judge", "--list"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    roles = [m["role"] for m in payload["models"]]
    assert "default" in roles


def test_judge_mutually_exclusive(
    runner: CliRunner, tmp_path: Path, fake_complete, clean_env
) -> None:
    _rubrics(tmp_path)
    result = runner.invoke(app, ["-C", str(tmp_path), "judge", "--list", "--ping", "x"])
    assert result.exit_code == 2

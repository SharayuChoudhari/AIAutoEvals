"""Tests for `version`, `doctor`, `config`, stubs, and CI auto-mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli.app import app


def test_version(runner: CliRunner, clean_env: None) -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "ai-eval" in result.output


def test_version_flag(runner: CliRunner, clean_env: None) -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ai-eval" in result.output


def test_doctor_runs_json(runner: CliRunner, tmp_path: Path, clean_env: None) -> None:
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "doctor"]
    )
    # Exit may be 1 (no rubrics, no provider creds). JSON must be valid.
    assert result.exit_code in (0, 1)
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert any(c["name"] == "python version" for c in payload["checks"])


def test_doctor_does_not_create_ai_eval_dir(
    runner: CliRunner, tmp_path: Path, clean_env: None
) -> None:
    """Doctor is read-only: it must not create .ai-eval/ as a side-effect."""
    state_dir = tmp_path / ".ai-eval"
    assert not state_dir.exists()
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "doctor"])
    assert not state_dir.exists(), "doctor must not create .ai-eval/"


def test_doctor_exits_1_when_rubrics_absent(
    runner: CliRunner, tmp_path: Path, clean_env: None
) -> None:
    """Plan §1.2: doctor exits 1 if any check fails (so CI can gate)."""
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "doctor"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert not payload["ok"]
    rubrics_check = next(c for c in payload["checks"] if c["name"] == "rubrics.yaml")
    assert rubrics_check["ok"] is False


def test_doctor_exits_0_when_all_required_pass(
    runner: CliRunner, tmp_path: Path, clean_env: None
) -> None:
    """Create a valid rubrics.yaml and set a fake provider env; doctor should pass."""
    from ai_eval.config.defaults import DEFAULT_JUDGE
    import yaml as _yaml

    rubrics = {
        "schema_version": 1,
        "judge": {"default": DEFAULT_JUDGE},
        "defaults": {"parallel": 4, "cache": True, "tolerance": 0.02},
        "tasks": {},
    }
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        _yaml.safe_dump(rubrics), encoding="utf-8"
    )
    # also create .ai-eval/ so the writable check passes
    (tmp_path / ".ai-eval").mkdir()

    result = runner.invoke(
        app,
        ["-C", str(tmp_path), "--format", "json", "doctor"],
        env={"OPENAI_API_KEY": "sk-fake"},
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


def test_config_print(runner: CliRunner, tmp_path: Path, clean_env: None) -> None:
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "config"]
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert "merged" in payload
    assert payload["merged"]["judge"]["default"].startswith("ollama/")


def test_run_is_stubbed_with_phase_error(
    runner: CliRunner, tmp_path: Path, clean_env: None
) -> None:
    result = runner.invoke(app, ["-C", str(tmp_path), "run"])
    assert result.exit_code == 2
    assert "Phase 4" in (result.stderr or result.output)


def test_bootstrap_requires_explicit_command(
    runner: CliRunner, tmp_path: Path, clean_env: None
) -> None:
    result = runner.invoke(app, ["-C", str(tmp_path), "bootstrap"])
    assert result.exit_code == 2
    assert "missing runtime command" in (result.stderr or result.output)


def test_ci_auto_mode_forces_json(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CI", "true")
    result = runner.invoke(app, ["-C", str(tmp_path), "config"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "merged" in payload

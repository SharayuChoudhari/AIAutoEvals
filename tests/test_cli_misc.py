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


def test_doctor_runs(runner: CliRunner, tmp_path: Path, clean_env: None) -> None:
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "doctor"]
    )
    # Exit code may be 1 (no rubrics yet) — that's expected. JSON must be valid.
    assert result.exit_code in (0, 1)
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert any(c["name"] == "python version" for c in payload["checks"])


def test_config_print(runner: CliRunner, tmp_path: Path, clean_env: None) -> None:
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "config"]
    )
    assert result.exit_code == 0, result.stderr or result.output
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
    # Output should be valid JSON, not a rich table.
    payload = json.loads(result.stdout)
    assert "merged" in payload

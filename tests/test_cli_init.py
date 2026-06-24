"""End-to-end coverage for `ai-eval init` on a tiny fixture repo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ai_eval.cli.app import app


def _run(runner: CliRunner, cwd: Path, *args: str):
    return runner.invoke(app, ["-C", str(cwd), *args])


def test_init_dry_run_json(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    result = _run(runner, tiny_repo, "--format", "json", "init", "--dry-run")
    assert result.exit_code == 0, result.output + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["dry_run"] is True
    assert payload["files_scanned"] >= 3
    names = {t["name"] for t in payload["tasks"]}
    assert {"customer_support_agent", "knowledge_base_retriever", "chat_chain"} <= names
    # Nothing was actually written.
    assert not (tiny_repo / "eval" / "rubrics.yaml").exists()


def test_init_writes_full_scaffold(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    result = _run(runner, tiny_repo, "--format", "json", "init")
    assert result.exit_code == 0, result.stderr or result.output

    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    golden_path = tiny_repo / "eval" / "golden_set.json"
    tests_path = tiny_repo / "eval" / "tests.py"
    state_dir = tiny_repo / ".ai-eval"
    gitignore = tiny_repo / ".gitignore"

    assert rubrics_path.is_file()
    assert golden_path.is_file()
    assert tests_path.is_file()
    assert state_dir.is_dir()
    assert ".ai-eval/" in gitignore.read_text(encoding="utf-8")

    rubrics = yaml.safe_load(rubrics_path.read_text(encoding="utf-8"))
    assert rubrics["schema_version"] == 1
    assert rubrics["judge"]["default"].startswith("ollama/")
    assert "customer_support_agent" in rubrics["tasks"]

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["schema_version"] == 1
    assert set(golden["tasks"]) == set(rubrics["tasks"])

    # tests.py is the bundled template (not empty).
    assert "ai_eval" in tests_path.read_text(encoding="utf-8")


def test_init_refuses_without_force_when_files_exist(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    (tiny_repo / "eval").mkdir()
    (tiny_repo / "eval" / "rubrics.yaml").write_text("dummy: true\n", encoding="utf-8")

    result = _run(runner, tiny_repo, "init")
    assert result.exit_code == 2, result.output
    assert "already exist" in result.stderr
    assert "--force" in result.stderr


def test_init_force_overwrites(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    (tiny_repo / "eval").mkdir()
    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    rubrics_path.write_text("schema_version: 0\n", encoding="utf-8")

    result = _run(runner, tiny_repo, "init", "--force")
    assert result.exit_code == 0, result.stderr or result.output
    body = rubrics_path.read_text(encoding="utf-8")
    assert "schema_version: 1" in body

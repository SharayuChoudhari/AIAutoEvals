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


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------

def test_init_dry_run_json(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    result = _run(runner, tiny_repo, "--format", "json", "init", "--dry-run")
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["dry_run"] is True
    assert payload["files_scanned"] >= 3
    names = {t["name"] for t in payload["tasks"]}
    assert {"customer_support_agent", "knowledge_base_retriever", "chat_chain"} <= names
    # Nothing was actually written.
    assert not (tiny_repo / "eval" / "rubrics.yaml").exists()


def test_init_dry_run_shows_requires_force_when_files_exist(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """dry-run must accurately reflect that a real run would require --force."""
    (tiny_repo / "eval").mkdir()
    (tiny_repo / "eval" / "rubrics.yaml").write_text("schema_version: 0\n", encoding="utf-8")

    result = _run(runner, tiny_repo, "--format", "json", "init", "--dry-run")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    # rubrics.yaml must appear in requires_force, not would_write.
    assert any("rubrics.yaml" in p for p in payload["requires_force"])
    # tests.py doesn't exist yet, so it should be in would_write.
    assert any("tests.py" in p for p in payload["would_write"])


# ---------------------------------------------------------------------------
# full scaffold write
# ---------------------------------------------------------------------------

def test_init_writes_full_scaffold(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    result = _run(runner, tiny_repo, "--format", "json", "init")
    assert result.exit_code == 0, result.stderr or result.stdout

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

    assert "ai_eval" in tests_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# --force scope: only scaffold files; golden_set captures preserved
# ---------------------------------------------------------------------------

def test_init_refuses_when_scaffold_files_exist_without_force(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    (tiny_repo / "eval").mkdir()
    (tiny_repo / "eval" / "rubrics.yaml").write_text("dummy: true\n", encoding="utf-8")

    result = _run(runner, tiny_repo, "init")
    assert result.exit_code == 2, result.stdout
    assert "already exist" in result.stderr
    assert "--force" in result.stderr


def test_init_force_rewrites_rubrics_but_preserves_captures(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """--force overwrites rubrics/tests but never wipes golden_set captures."""
    (tiny_repo / "eval").mkdir()
    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    golden_path = tiny_repo / "eval" / "golden_set.json"

    rubrics_path.write_text("schema_version: 0\n", encoding="utf-8")
    # Simulate real captures in golden_set.json.
    captures = {
        "schema_version": 1,
        "tasks": {"customer_support_agent": [{"id": "gs_abc", "input": {"messages": []}}]},
    }
    golden_path.write_text(json.dumps(captures), encoding="utf-8")

    result = _run(runner, tiny_repo, "init", "--force")
    assert result.exit_code == 0, result.stderr or result.stdout

    # rubrics.yaml should be rewritten.
    assert "schema_version: 1" in rubrics_path.read_text(encoding="utf-8")
    # golden_set.json must still contain the real captures.
    golden_after = json.loads(golden_path.read_text(encoding="utf-8"))
    assert len(golden_after["tasks"]["customer_support_agent"]) == 1, (
        "real captures were wiped by --force — this is the data-loss bug"
    )


def test_init_reset_golden_discards_captures(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """--reset-golden is the explicit opt-in needed to discard captures."""
    (tiny_repo / "eval").mkdir()
    golden_path = tiny_repo / "eval" / "golden_set.json"
    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    rubrics_path.write_text("schema_version: 0\n", encoding="utf-8")
    captures = {
        "schema_version": 1,
        "tasks": {"customer_support_agent": [{"id": "gs_abc", "input": {}}]},
    }
    golden_path.write_text(json.dumps(captures), encoding="utf-8")

    result = _run(runner, tiny_repo, "init", "--force", "--reset-golden")
    assert result.exit_code == 0
    golden_after = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden_after["tasks"].get("customer_support_agent") == [], (
        "captures should be reset to [] when --reset-golden is passed"
    )


def test_init_force_overwrites_rubrics_only(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """Regression: --force must rewrite rubrics.yaml even without golden_set present."""
    (tiny_repo / "eval").mkdir()
    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    rubrics_path.write_text("schema_version: 0\n", encoding="utf-8")

    result = _run(runner, tiny_repo, "init", "--force")
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "schema_version: 1" in rubrics_path.read_text(encoding="utf-8")

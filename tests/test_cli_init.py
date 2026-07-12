"""End-to-end coverage for `ai-evals init` on a tiny fixture repo."""

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
    # --rubric-engine rules reproduces the pre-SLM snapshot without ollama.
    result = _run(
        runner, tiny_repo, "--format", "json", "init", "--dry-run",
        "--rubric-engine", "rules",
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["dry_run"] is True
    assert payload["files_scanned"] >= 3
    names = {t["name"] for t in payload["tasks"]}
    assert {"customer_support_agent", "knowledge_base_retriever", "chat_chain"} <= names
    assert payload["rubric_engine"] == "rules"
    # Nothing was actually written.
    assert not (tiny_repo / "eval" / "rubrics.yaml").exists()


def test_init_dry_run_shows_requires_force_when_files_exist(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """dry-run must accurately reflect that a real run would require --force."""
    (tiny_repo / "eval").mkdir()
    (tiny_repo / "eval" / "rubrics.yaml").write_text("schema_version: 0\n", encoding="utf-8")

    result = _run(
        runner, tiny_repo, "--format", "json", "init", "--dry-run",
        "--rubric-engine", "rules",
    )
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
    result = _run(
        runner, tiny_repo, "--format", "json", "init", "--rubric-engine", "rules"
    )
    assert result.exit_code == 0, result.stderr or result.stdout

    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    golden_path = tiny_repo / "eval" / "golden_set.json"
    tests_path = tiny_repo / "eval" / "tests.py"
    state_dir = tiny_repo / ".ai-evals"
    gitignore = tiny_repo / ".gitignore"

    assert rubrics_path.is_file()
    assert golden_path.is_file()
    assert tests_path.is_file()
    assert state_dir.is_dir()
    assert ".ai-evals/" in gitignore.read_text(encoding="utf-8")

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

    result = _run(runner, tiny_repo, "init", "--force", "--rubric-engine", "rules")
    assert result.exit_code == 0, result.stderr or result.stdout

    # rubrics.yaml should be rewritten.
    assert "schema_version: 1" in rubrics_path.read_text(encoding="utf-8")
    # golden_set.json must still contain the real captures (D6 also auto-seeds
    # shape variants, so check the real capture survived, not the total count).
    golden_after = json.loads(golden_path.read_text(encoding="utf-8"))
    task_examples = golden_after["tasks"]["customer_support_agent"]
    real = [e for e in task_examples if e.get("seed") != "auto"]
    assert len(real) == 1 and real[0]["id"] == "gs_abc", (
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

    result = _run(
        runner, tiny_repo, "init", "--force", "--reset-golden",
        "--rubric-engine", "rules",
    )
    assert result.exit_code == 0
    golden_after = json.loads(golden_path.read_text(encoding="utf-8"))
    # --reset-golden discards real captures; D6 then auto-seeds, so the task
    # list has auto-seeds but no real captures.
    task_examples = golden_after["tasks"].get("customer_support_agent", [])
    real = [e for e in task_examples if e.get("seed") != "auto"]
    assert real == [], (
        "real captures should be discarded when --reset-golden is passed"
    )


def test_init_force_overwrites_rubrics_only(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """Regression: --force must rewrite rubrics.yaml even without golden_set present."""
    (tiny_repo / "eval").mkdir()
    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    rubrics_path.write_text("schema_version: 0\n", encoding="utf-8")

    result = _run(runner, tiny_repo, "init", "--force", "--rubric-engine", "rules")
    assert result.exit_code == 0, result.stderr or result.stdout
    assert "schema_version: 1" in rubrics_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# --rubric-engine slm (mocked client) and hard-fail on unreachable model
# ---------------------------------------------------------------------------


def test_init_slm_engine_writes_rubric_with_mocked_client(
    runner: CliRunner, tiny_repo: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--rubric-engine slm with a mocked SLM client writes a rubric whose tasks
    carry the SLM-classified type (no ollama needed)."""
    from ai_eval.inference.slm import builder as slm_builder
    from ai_eval.inference.slm.builder import _SLMMetric, _SLMRecovery, _SLMTask

    def fake_complete(**kw):
        rm = kw["response_model"]
        if rm is _SLMTask:
            return _SLMTask(
                type="rag",
                purpose="retrieval-augmented QA",
                inputs=["query"],
                outputs=["answer"],
                metrics=[_SLMMetric(name="faithfulness")],
            )
        if rm is _SLMRecovery:
            return _SLMRecovery()
        raise AssertionError(f"unexpected response_model {rm!r}")

    # The builder resolves the default client via the module-level name
    # `_default_complete`; patching that name redirects the CLI path.
    monkeypatch.setattr(slm_builder, "_default_complete", fake_complete)

    result = _run(
        runner, tiny_repo, "--format", "json", "init", "--rubric-engine", "slm"
    )
    assert result.exit_code == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["rubric_engine"] == "slm"
    # The SLM re-typed at least one detected task as 'rag'.
    types = {t["type"] for t in payload["tasks"]}
    assert "rag" in types

    rubrics_path = tiny_repo / "eval" / "rubrics.yaml"
    assert rubrics_path.is_file()
    rubrics = yaml.safe_load(rubrics_path.read_text(encoding="utf-8"))
    assert rubrics["rubric_engine"] == "slm"
    assert any(t["type"] == "rag" for t in rubrics["tasks"].values())


def test_init_slm_engine_hard_fails_when_model_unreachable(
    runner: CliRunner, tiny_repo: Path, clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the SLM is unavailable after retries, init must hard-fail (exit 1)
    with an actionable remediation hint — never silently fall back to rules."""
    from ai_eval.inference.slm import builder as slm_builder

    def raising(**kw):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(slm_builder, "_default_complete", raising)

    result = _run(runner, tiny_repo, "init", "--rubric-engine", "slm")
    assert result.exit_code == 1, result.stdout
    stderr = result.stderr or ""
    # Remediation hint points the user at the rules fallback / ollama.
    assert "--rubric-engine rules" in stderr
    # No rubrics.yaml was written on failure.
    assert not (tiny_repo / "eval" / "rubrics.yaml").exists()


def test_init_rejects_invalid_rubric_engine(
    runner: CliRunner, tiny_repo: Path, clean_env: None
) -> None:
    """An unknown --rubric-engine value is a usage error (exit 2)."""
    result = _run(runner, tiny_repo, "init", "--rubric-engine", "magic")
    assert result.exit_code == 2
    assert "invalid --rubric-engine" in (result.stderr or "")

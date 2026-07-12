"""CLI tests for `ai-evals run` with a fake judge transport."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ai_eval.cli.app import app


def _fake_complete_factory(monkeypatch: pytest.MonkeyPatch, score: float = 0.9):
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate(
            {"score": score, "rationale": "ok", "sub_scores": {}}
        )
    import ai_eval.judge.gateway as gw
    monkeypatch.setattr(gw, "_default_complete", complete_fn)


def _setup_repo(tmp_path: Path, *, metric_threshold: float = 0.5,
                task_file: str = "src/chat.py") -> None:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / task_file).write_text(
        textwrap.dedent(
            """
            def main(query):
                return {'answer': query}
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_type": "chat",
                "judge": {"default": "fake/local"},
                "defaults": {"parallel": 2, "cache": True, "tolerance": 0.02},
                "tasks": {
                    "chat_task": {
                        "file_path": task_file,
                        "entry": "main",
                        "type": "chat",
                        "metrics": [
                            {"name": "hallucination_rate", "threshold": metric_threshold}
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval" / "golden_set.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tasks": {
                    "chat_task": [
                        {"id": "e1", "input": {"query": "hello"},
                         "expected": None, "trace": {"calls": []}}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)


def test_run_json_pass(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run"]
    )
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    assert "tasks" in payload
    m = payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]
    assert m["score"] == 0.9
    assert m["status"] == "pass"


def test_run_human(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    result = runner.invoke(app, ["-C", str(tmp_path), "run"])
    assert result.exit_code == 0
    assert "chat_task" in result.output


def test_run_tsv(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "tsv", "run"])
    assert result.exit_code == 0
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("task\tmetric")
    assert any("hallucination_rate" in ln for ln in lines[1:])


def test_run_fail_on_regression_exit_3(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    # score 0.3 < threshold 0.9 -> fail -> exit 3
    _fake_complete_factory(monkeypatch, score=0.3)
    _setup_repo(tmp_path, metric_threshold=0.9)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run", "--fail-on-regression"]
    )
    assert result.exit_code == 3


def test_run_pass_without_fail_flag_when_metric_low(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    # Without --fail-on-regression, a below-threshold metric -> status fail
    # but exit 0 (informational). Actually threshold breach -> status fail; the
    # plan says --fail-on-regression triggers exit 3. Without it, exit 0.
    _fake_complete_factory(monkeypatch, score=0.3)
    _setup_repo(tmp_path, metric_threshold=0.9)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    m = payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]
    assert m["status"] == "fail"


def test_run_fresh_repo_no_delta(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run"]
    )
    payload = json.loads(result.stdout)
    m = payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]
    assert m["delta"] is None


def test_run_unregistered_metric_exit_1(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "bad.py").write_text("def main(q):\n    return q\n", encoding="utf-8")
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_type": "chat",
                "judge": {"default": "fake/local"},
                "defaults": {"parallel": 2, "cache": True, "tolerance": 0.02},
                "tasks": {
                    "t": {
                        "file_path": "src/bad.py", "entry": "main", "type": "chat",
                        "metrics": [{"name": "totally_unknown_metric", "threshold": 0.5}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval" / "golden_set.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {
            "t": [{"id": "e1", "input": "q", "expected": None, "trace": {"calls": []}}]
        }}),
        encoding="utf-8",
    )
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run"]
    )
    assert result.exit_code == 1
    assert "totally_unknown_metric" in (result.stderr or result.output)


def test_run_missing_golden_set(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        json.dumps({"schema_version": 1, "project_type": "chat",
                    "judge": {"default": "fake/local"},
                    "defaults": {"parallel": 2, "cache": True, "tolerance": 0.02},
                    "tasks": {}}),
        encoding="utf-8",
    )
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)
    result = runner.invoke(app, ["-C", str(tmp_path), "run"])
    assert result.exit_code == 2
    assert "golden set" in (result.stderr or result.output).lower()


def test_run_saves_to_history(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    history = json.loads((tmp_path / ".ai-evals" / "history.json").read_text())
    assert len(history["runs"]) == 1


def test_run_empty_golden_set_marks_metrics_skip(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    """A task whose golden array is empty → 0 examples → metric status `skip`,
    never `fail`. The human render must show a skip glyph + a bootstrap hint,
    not a misleading `✗ fail`."""
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    # Overwrite golden set with an empty capture array for the task.
    (tmp_path / "eval" / "golden_set.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {"chat_task": []}}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    m = payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]
    assert m["status"] == "skip"
    assert m["score"] is None
    assert payload["summary"]["examples"] == 0


def test_run_empty_golden_set_human_warns_and_skips(
    runner: CliRunner, tmp_path: Path, monkeypatch, clean_env
) -> None:
    _fake_complete_factory(monkeypatch, score=0.9)
    _setup_repo(tmp_path, metric_threshold=0.5)
    (tmp_path / "eval" / "golden_set.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {"chat_task": []}}),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "human", "--no-color", "run"])
    assert result.exit_code == 0
    out = result.output
    assert "0 examples" in out
    assert "no examples ran" in out
    assert "bootstrap" in out
    # skip glyph label must appear, not a bare fail
    assert "skip" in out

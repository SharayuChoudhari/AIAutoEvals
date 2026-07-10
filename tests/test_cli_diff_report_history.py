"""CLI tests for `ai-evals diff`, `report`, `history`."""

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
            {"score": score, "rationale": "narrative here", "sub_scores": {}}
        )
    import ai_eval.judge.gateway as gw
    monkeypatch.setattr(gw, "_default_complete", complete_fn)


def _setup_repo(tmp_path: Path, *, score: float = 0.9) -> None:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "chat.py").write_text(
        textwrap.dedent("def main(query):\n    return {'answer': query}\n"),
        encoding="utf-8",
    )
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1, "project_type": "chat",
                "judge": {"default": "fake/local", "regression_check": "fake/reg"},
                "defaults": {"parallel": 2, "cache": True, "tolerance": 0.02},
                "tasks": {
                    "chat_task": {
                        "file_path": "src/chat.py", "entry": "main", "type": "chat",
                        "metrics": [{"name": "hallucination_rate", "threshold": 0.5}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "eval" / "golden_set.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {
            "chat_task": [{"id": "e1", "input": {"query": "hello"},
                           "expected": None, "trace": {"calls": []}}]
        }}),
        encoding="utf-8",
    )
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)


def _run_once(runner, tmp_path, score):
    _fake_complete_factory(monkeypatch := pytest.MonkeyPatch(), score=score)
    monkeypatch.setenv  # noqa
    # use a fresh monkeypatch per call
    import ai_eval.judge.gateway as gw

    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate(
            {"score": score, "rationale": "n", "sub_scores": {}}
        )
    gw._default_complete = complete_fn
    return runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])


@pytest.fixture
def fake_judge(monkeypatch: pytest.MonkeyPatch):
    """Patch the judge transport with a configurable fake (auto-restored)."""
    holder = {"score": 0.9}

    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate(
            {"score": holder["score"], "rationale": "ok", "sub_scores": {}}
        )
    import ai_eval.judge.gateway as gw
    monkeypatch.setattr(gw, "_default_complete", complete_fn)
    return holder


# --- history ---

def test_history_list_empty(
    runner: CliRunner, tmp_path: Path, clean_env
) -> None:
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "history"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["runs"] == []


def test_history_list_after_run(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "history"]
    )
    payload = json.loads(result.stdout)
    assert len(payload["runs"]) == 1


def test_history_show(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    runs = json.loads(
        (tmp_path / ".ai-evals" / "history.json").read_text()
    )["runs"]
    rid = runs[-1]["id"]
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "history", "--show", rid]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["id"] == rid


def test_history_export(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    out_path = str(tmp_path / "hist.json")
    result = runner.invoke(
        app, ["-C", str(tmp_path), "history", "--export", out_path]
    )
    assert result.exit_code == 0
    data = json.loads(Path(out_path).read_text())
    assert len(data["runs"]) == 1


def test_history_prune(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "history", "--prune", "--keep", "1"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["removed"] == 1


# --- diff ---

def test_diff_after_two_runs(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    fake_judge["score"] = 0.9
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run", "--no-cache"])
    fake_judge["score"] = 0.6
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run", "--no-cache"])

    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "diff", "--against", "last", "--no-explain"]
    )
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    deltas = payload["deltas"]["chat_task"]["hallucination_rate"]
    assert deltas["delta"] is not None
    assert deltas["delta"] < 0


def test_diff_no_runs_errors(
    runner: CliRunner, tmp_path: Path, clean_env
) -> None:
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)
    result = runner.invoke(app, ["-C", str(tmp_path), "diff"])
    assert result.exit_code == 2


# --- report ---

def test_report_json(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "report", "last"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "tasks" in payload


def test_report_web(
    runner: CliRunner, tmp_path: Path, fake_judge, clean_env
) -> None:
    _setup_repo(tmp_path)
    runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "run"])
    result = runner.invoke(
        app, ["-C", str(tmp_path), "report", "last", "--web"]
    )
    assert result.exit_code == 0
    assert result.output.startswith("file://")


def test_report_missing_run(
    runner: CliRunner, tmp_path: Path, clean_env
) -> None:
    (tmp_path / ".ai-evals").mkdir(exist_ok=True)
    result = runner.invoke(app, ["-C", str(tmp_path), "report", "nope"])
    assert result.exit_code == 2

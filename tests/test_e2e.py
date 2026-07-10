"""E2E: init → bootstrap → run → diff on a fixture repo.

Uses a fake async judge transport and a runtime script that exercises the
bootstrap wrappers, so no network/ollama is touched. The `tiny_repo` fixture
isn't used directly because init's SLM engine needs ollama; we use
`--rubric-engine rules` to stay hermetic, then a hand-written task module
matches the rubrics task shape for run.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ai_eval.cli.app import app


def _fake_complete(monkeypatch: pytest.MonkeyPatch, score: float = 0.9):
    async def complete_fn(*, model, messages, response_model, temperature=0.0):
        return response_model.model_validate(
            {"score": score, "rationale": "ok", "sub_scores": {}}
        )
    import ai_eval.judge.gateway as gw
    monkeypatch.setattr(gw, "_default_complete", complete_fn)


_RUNTIME_SNIPPET = textwrap.dedent(
    """
    import json, os, sys, types
    mod = types.ModuleType("openai")
    r = types.ModuleType("openai.resources")
    c = types.ModuleType("openai.resources.chat")
    cc = types.ModuleType("openai.resources.chat.completions")
    class Completions:
        def create(self, **kw):
            return {"choices": [{"message": {"content": "hi"}}]}
    cc.Completions = Completions
    c.completions = cc
    r.chat = c
    mod.resources = r
    sys.modules.update({
        "openai": mod, "openai.resources": r,
        "openai.resources.chat": c, "openai.resources.chat.completions": cc,
    })
    from ai_eval.bootstrap.wrappers import install_hooks
    from ai_eval.bootstrap.tracer import configure_sink, enter_task, exit_task
    install_hooks(redact_fields=True)
    sink = os.environ["AI_EVAL_BOOTSTRAP_SINK"]
    def _sink(ex):
        with open(sink, "a") as fh:
            fh.write(json.dumps(ex, default=str)); fh.write("\\n")
    configure_sink(sink=_sink, redact_fields=["*"])
    enter_task("chat_task", {"query": "what is the return policy"})
    Completions().create(model="m", messages=[{"role": "user", "content": "hi"}])
    exit_task({"answer": "30 days"})
    """
)


def _make_repo(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "chat.py").write_text(
        textwrap.dedent(
            """
            from openai import OpenAI

            def main(query):
                client = OpenAI()
                resp = client.chat.completions.create(
                    model='gpt-4o-mini', messages=[{'role': 'user', 'content': query}]
                )
                return {'answer': resp['choices'][0]['message']['content']}
            """
        ),
        encoding="utf-8",
    )


def _install_fake_openai():
    """Install a stub openai in sys.modules so the task module imports + runs."""
    import sys
    import types
    mod = types.ModuleType("openai")

    class _Completions:
        def create(self, **kw):
            return {"choices": [{"message": {"content": "30 days"}}]}

    class _Chat:
        completions = _Completions()

    class OpenAI:
        def __init__(self, *a, **kw):
            pass

        @property
        def chat(self):
            return _Chat()

    mod.OpenAI = OpenAI
    sys.modules["openai"] = mod


def test_e2e_init_bootstrap_run_diff(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env
) -> None:
    _make_repo(tmp_path)

    # 1. init with rules engine (no ollama). --yes is a global flag.
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--yes", "--format", "json", "init",
              "--rubric-engine", "rules", "--force"]
    )
    assert result.exit_code == 0, result.stderr or result.output
    rubrics_path = tmp_path / "eval" / "rubrics.yaml"
    assert rubrics_path.is_file()

    # Patch the rubrics so the task entry/file match our fixture module + add a metric.
    rubrics = yaml.safe_load(rubrics_path.read_text())
    # ensure judge.default is the fake model so run's gateway uses it.
    rubrics["judge"] = {"default": "fake/local", "regression_check": "fake/reg", "fallback": []}
    # Ensure our chat task targets src/chat.py:main with hallucination_rate metric.
    rubrics["tasks"] = {
        "chat_task": {
            "file_path": "src/chat.py",
            "entry": "main",
            "type": "chat",
            "inputs": ["query"],
            "outputs": ["answer"],
            "metrics": [{"name": "hallucination_rate", "threshold": 0.5}],
        }
    }
    rubrics_path.write_text(yaml.safe_dump(rubrics), encoding="utf-8")

    # 2. bootstrap via a runtime snippet that emits a capture for chat_task.
    result = runner.invoke(
        app,
        ["-C", str(tmp_path), "--format", "json", "bootstrap", "--",
         sys.executable, "-c", _RUNTIME_SNIPPET],
    )
    assert result.exit_code == 0, result.stderr or result.output
    golden = json.loads((tmp_path / "eval" / "golden_set.json").read_text())
    assert len(golden["tasks"]["chat_task"]) == 1

    # 3. run (first run = baseline, fresh repo).
    _install_fake_openai()
    _fake_complete(monkeypatch, score=0.9)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run", "--no-cache"]
    )
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    m = payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]
    assert m["score"] == 0.9
    assert m["delta"] is None  # fresh repo, no baseline

    # 4. run again with a lower score -> regression, diff shows negative delta.
    _fake_complete(monkeypatch, score=0.4)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run", "--no-cache", "--fail-on-regression"]
    )
    assert result.exit_code == 3  # below threshold 0.5
    payload = json.loads(result.stdout)
    assert payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]["status"] == "fail"

    # 5. diff against last.
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "diff", "--against", "last", "--no-explain"]
    )
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    d = payload["deltas"]["chat_task"]["hallucination_rate"]
    assert d["delta"] is not None
    assert d["delta"] < 0


def test_e2e_fresh_repo_no_delta_exit_0(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env
) -> None:
    _make_repo(tmp_path)
    runner.invoke(
        app, ["-C", str(tmp_path), "--yes", "--format", "json", "init",
              "--rubric-engine", "rules", "--force"]
    )
    rubrics_path = tmp_path / "eval" / "rubrics.yaml"
    rubrics = yaml.safe_load(rubrics_path.read_text())
    rubrics["judge"] = {"default": "fake/local"}
    rubrics["tasks"] = {
        "chat_task": {
            "file_path": "src/chat.py", "entry": "main", "type": "chat",
            "inputs": ["query"], "outputs": ["answer"],
            "metrics": [{"name": "hallucination_rate", "threshold": 0.5}],
        }
    }
    rubrics_path.write_text(yaml.safe_dump(rubrics), encoding="utf-8")
    (tmp_path / "eval" / "golden_set.json").write_text(
        json.dumps({"schema_version": 1, "tasks": {
            "chat_task": [{"id": "e1", "input": {"query": "hi"},
                           "expected": None, "trace": {"calls": []}}]
        }}),
        encoding="utf-8",
    )
    _fake_complete(monkeypatch, score=0.9)
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "run", "--no-cache"]
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["tasks"]["chat_task"]["metrics"]["hallucination_rate"]["delta"] is None

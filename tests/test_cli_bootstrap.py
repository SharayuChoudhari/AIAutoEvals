"""E2E test for `ai-evals bootstrap`.

The runtime command is a Python snippet that installs a fake ``openai`` module,
calls the bootstrap wrappers directly, and emits captures to the sink path
(read from the env var the CLI set). This exercises the full parent→child→sink
plumbing and the golden-writer merge without requiring the real openai or the
sitecustomize auto-import path.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from typer.testing import CliRunner

from ai_eval.cli.app import app


def _rubrics(tmp_path: Path) -> None:
    (tmp_path / "eval").mkdir(exist_ok=True)
    (tmp_path / "eval" / "rubrics.yaml").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_type": "chat",
                "judge": {"default": "fake/local"},
                "defaults": {"parallel": 4, "cache": True, "tolerance": 0.02},
                "tasks": {
                    "chat_task": {
                        "file_path": "src/chat.py",
                        "entry": "main",
                        "type": "chat",
                        "metrics": [{"name": "hallucination_rate", "threshold": 0.1}],
                    }
                },
            }
        ),
        encoding="utf-8",
    )


_RUNTIME_SNIPPET = textwrap.dedent(
    """
    import json, os, sys, types
    # fake openai with the path the wrapper patches
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
    out = []
    def _sink(ex):
        out.append(ex)
        with open(sink, "a") as fh:
            fh.write(json.dumps(ex, default=str)); fh.write("\\n")
    configure_sink(sink=_sink, redact_fields=["*"])
    enter_task("chat_task", {"query": "hello"})
    Completions().create(model="m", messages=[{"role": "user", "content": "hi"}])
    exit_task({"answer": "hi"})
    """
)


def test_bootstrap_e2e_populates_golden_set(runner: CliRunner, tmp_path: Path, clean_env) -> None:
    _rubrics(tmp_path)
    cmd = [sys.executable, "-c", _RUNTIME_SNIPPET]
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "bootstrap", "--", *cmd])
    assert result.exit_code == 0, result.stderr or result.output
    payload = json.loads(result.stdout)
    assert payload["captured"] >= 1
    golden = json.loads((tmp_path / "eval" / "golden_set.json").read_text())
    assert "chat_task" in golden["tasks"]
    assert len(golden["tasks"]["chat_task"]) == 1
    ex = golden["tasks"]["chat_task"][0]
    assert ex["input"] == {"query": "hello"}
    assert ex["trace"]["calls"][0]["kind"] == "llm"


def test_bootstrap_missing_command(runner: CliRunner, tmp_path: Path, clean_env) -> None:
    result = runner.invoke(app, ["-C", str(tmp_path), "bootstrap"])
    assert result.exit_code == 2
    assert "missing runtime command" in (result.stderr or result.output)


def test_bootstrap_no_instrument_rejected(runner: CliRunner, tmp_path: Path, clean_env) -> None:
    _rubrics(tmp_path)
    result = runner.invoke(app, ["-C", str(tmp_path), "bootstrap", "--no-instrument", "--", "true"])
    assert result.exit_code == 2
    assert "Phase 5" in (result.stderr or result.output)


def test_bootstrap_child_failure_exit_1(runner: CliRunner, tmp_path: Path, clean_env) -> None:
    _rubrics(tmp_path)
    # runtime exits non-zero but still flushes nothing -> exit 1
    result = runner.invoke(
        app,
        [
            "-C",
            str(tmp_path),
            "--format",
            "json",
            "bootstrap",
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(3)",
        ],
    )
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["child_exit_code"] == 3
    assert payload["captured"] == 0


def test_bootstrap_replace_mode(runner: CliRunner, tmp_path: Path, clean_env) -> None:
    _rubrics(tmp_path)
    # seed an existing capture
    from ai_eval.bootstrap.golden_writer import append

    append(
        tmp_path / "eval" / "golden_set.json",
        [
            {
                "task": "chat_task",
                "input": {"query": "old"},
                "expected": None,
                "trace": {"calls": []},
            }
        ],
    )
    cmd = [sys.executable, "-c", _RUNTIME_SNIPPET]
    result = runner.invoke(
        app, ["-C", str(tmp_path), "--format", "json", "bootstrap", "--replace", "--", *cmd]
    )
    assert result.exit_code == 0, result.stderr or result.output
    golden = json.loads((tmp_path / "eval" / "golden_set.json").read_text())
    queries = [ex["input"]["query"] for ex in golden["tasks"]["chat_task"]]
    assert "old" not in queries  # replaced
    assert "hello" in queries


def test_bootstrap_does_not_modify_rubrics_top_level(
    runner: CliRunner, tmp_path: Path, clean_env
) -> None:
    """AGENTS.md §1 contract: bootstrap is capture-only. It must never edit
    ``rubrics.yaml`` — in particular it must not flip ``top_level`` flags.
    Users re-run ``ai-evals init`` to regenerate ``top_level`` from the
    signature-inspection + Layer 3 demotion pipeline."""
    _rubrics(tmp_path)
    # Inject a non-top-level task into rubrics.yaml before bootstrap.
    rubrics_path = tmp_path / "eval" / "rubrics.yaml"
    rubrics_data = json.loads(rubrics_path.read_text())
    rubrics_data["tasks"]["internal_dao"] = {
        "file_path": "src/dao.py",
        "entry": "DAO.search",
        "type": "chat",
        "top_level": False,
        "metrics": [{"name": "hallucination_rate", "threshold": 0.1}],
    }
    rubrics_path.write_text(json.dumps(rubrics_data), encoding="utf-8")
    mtime_before = rubrics_path.stat().st_mtime_ns

    cmd = [sys.executable, "-c", _RUNTIME_SNIPPET]
    result = runner.invoke(app, ["-C", str(tmp_path), "--format", "json", "bootstrap", "--", *cmd])
    assert result.exit_code == 0, result.stderr or result.output

    # rubrics.yaml must be byte-for-byte untouched (mtime unchanged).
    assert rubrics_path.stat().st_mtime_ns == mtime_before
    after = json.loads(rubrics_path.read_text())
    assert after["tasks"]["internal_dao"]["top_level"] is False
    assert after["tasks"]["chat_task"].get("top_level", True) is True

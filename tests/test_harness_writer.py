"""Tests for IO-coupled classification (D4) and harness codegen (D5)."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.config.schema import JudgeConfig, RubricsConfig, TaskSpec
from ai_eval.inference.callgraph import build_call_graph
from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.scaffold.harness_writer import (
    HarnessSpec,
    IOAttr,
    classify_io_coupled,
    render_harness,
    write_harnesses,
)


def _rubrics(tasks: dict[str, TaskSpec]) -> RubricsConfig:
    return RubricsConfig(
        schema_version=1,
        project_type="chat",
        judge=JudgeConfig(default="fake/local"),
        tasks=tasks,
    )


def _contexts(root: Path, tasks: list[DetectedTask]) -> dict:
    scan = ScanResult(
        files_scanned=len(tasks), elapsed_seconds=0.0,
        tasks=tasks, frameworks_seen=set(),
    )
    _, ctx = build_call_graph(root, scan)
    return ctx


# ---------------------------------------------------------------------------
# D4 classification
# ---------------------------------------------------------------------------

def test_io_coupled_classifies_dao_read(tmp_path: Path) -> None:
    """A method reading ``self.dao.search(...)`` where ``self.dao`` is assigned
    a DAO (not an LLM client / graph) in ``__init__`` is IO-coupled."""
    (tmp_path / "svc.py").write_text(
        "from layers.dao import DocumentVectorDAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DocumentVectorDAO()\n"
        "        self.client = OpenAI()\n"
        "    def process(self, q):\n"
        "        docs = self.dao.search(q)\n"
        "        return docs\n",
        encoding="utf-8",
    )
    task = TaskSpec(file_path="svc.py", entry="Svc.process", type="chat")
    task_dt = DetectedTask(name="svc_process", framework="unknown", type="chat",
                           file_path="svc.py", entry="Svc.process")
    ctx = _contexts(tmp_path, [task_dt])
    attrs = classify_io_coupled(task, ctx)
    assert len(attrs) == 1
    assert attrs[0].attr == "dao"
    assert attrs[0].method == "search"
    assert attrs[0].ctor_name == "DocumentVectorDAO"


def test_pure_llm_no_self_attr_read_is_not_io_coupled(tmp_path: Path) -> None:
    """A module-level function calling OpenAI inline (no ``self.<attr>`` reads)
    is pure-LLM — empty attrs list."""
    (tmp_path / "summarize.py").write_text(
        "from openai import OpenAI\n"
        "def summarize(text):\n"
        "    return OpenAI().chat.completions.create(input=text)\n",
        encoding="utf-8",
    )
    task = TaskSpec(file_path="summarize.py", entry="summarize", type="chat")
    task_dt = DetectedTask(name="summarize", framework="unknown", type="chat",
                           file_path="summarize.py", entry="summarize")
    ctx = _contexts(tmp_path, [task_dt])
    assert classify_io_coupled(task, ctx) == []


def test_llm_client_attr_not_stubbed(tmp_path: Path) -> None:
    """``self.client`` assigned to ``OpenAI()`` is NOT stubbed (LLM clients run
    real). Only the DAO read counts as IO-coupled."""
    (tmp_path / "svc.py").write_text(
        "from openai import OpenAI\n"
        "from layers.dao import DAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.client = OpenAI()\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        docs = self.dao.search(q)\n"
        "        return self.client.chat.completions.create(input=docs)\n",
        encoding="utf-8",
    )
    task = TaskSpec(file_path="svc.py", entry="Svc.process", type="chat")
    task_dt = DetectedTask(name="svc_process", framework="unknown", type="chat",
                           file_path="svc.py", entry="Svc.process")
    ctx = _contexts(tmp_path, [task_dt])
    attrs = classify_io_coupled(task, ctx)
    # Only dao.search; the OpenAI client read is not a stub target.
    assert {a.attr for a in attrs} == {"dao"}


def test_graph_object_attr_not_stubbed(tmp_path: Path) -> None:
    """``self.graph`` assigned to a compiled StateGraph is NOT stubbed (graphs
    orchestrate the LLM and must run real)."""
    (tmp_path / "wf.py").write_text(
        "from langgraph.graph import StateGraph\n"
        "class WF:\n"
        "    def __init__(self):\n"
        "        self.graph = StateGraph(dict).compile()\n"
        "    def run(self, state):\n"
        "        return self.graph.invoke(state)\n",
        encoding="utf-8",
    )
    task = TaskSpec(file_path="wf.py", entry="WF.run", type="workflow")
    task_dt = DetectedTask(name="wf_run", framework="unknown", type="workflow",
                           file_path="wf.py", entry="WF.run")
    ctx = _contexts(tmp_path, [task_dt])
    assert classify_io_coupled(task, ctx) == []


# ---------------------------------------------------------------------------
# D5 harness codegen
# ---------------------------------------------------------------------------

def test_render_harness_parses_and_has_regions() -> None:
    spec = HarnessSpec(
        task_name="svc_process", entry="Svc.process", file_path="svc.py",
        attrs=[IOAttr(attr="dao", method="search", ctor_name="DAO")],
        body_hash="abc123def456",
    )
    content = render_harness(spec)
    ast.parse(content)  # valid Python
    assert "# AUTO-GENERATED DO NOT EDIT" in content
    assert "# ast-hash: abc123def456" in content
    assert "# FIXTURE DATA — edit freely" in content
    assert "('dao', 'search')" in content


def test_write_harnesses_creates_file_for_io_coupled_task(tmp_path: Path) -> None:
    (tmp_path / "svc.py").write_text(
        "from layers.dao import DAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    rubrics = _rubrics({
        "svc_process": TaskSpec(file_path="svc.py", entry="Svc.process", type="chat"),
    })
    written, _ = write_harnesses(rubrics, eval_dir, project_root=tmp_path)
    assert any(status == "wrote" for _, status in written)
    harness_path = eval_dir / "_harness_svc_process.py"
    assert harness_path.is_file()
    content = harness_path.read_text(encoding="utf-8")
    ast.parse(content)
    assert "('dao', 'search')" in content


def test_write_harnesses_skips_pure_llm_task(tmp_path: Path) -> None:
    (tmp_path / "summarize.py").write_text(
        "from openai import OpenAI\n"
        "def summarize(text):\n"
        "    return OpenAI().chat.completions.create(input=text)\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    rubrics = _rubrics({
        "summarize": TaskSpec(file_path="summarize.py", entry="summarize", type="chat"),
    })
    written, _ = write_harnesses(rubrics, eval_dir, project_root=tmp_path)
    # No harness file generated for pure-LLM task.
    assert written == []
    assert not (eval_dir / "_harness_summarize.py").exists()


def test_write_harnesses_preserves_region2_on_regen(tmp_path: Path) -> None:
    """Re-running write_harnesses with a changed body hash preserves the
    user-edited region 2 (fixtures). Region 1 is regenerated."""
    (tmp_path / "svc.py").write_text(
        "from layers.dao import DAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    rubrics = _rubrics({
        "svc_process": TaskSpec(file_path="svc.py", entry="Svc.process", type="chat"),
    })
    write_harnesses(rubrics, eval_dir, project_root=tmp_path)
    harness_path = eval_dir / "_harness_svc_process.py"
    # User edits region 2 fixtures.
    content = harness_path.read_text(encoding="utf-8")
    user_content = content.replace(
        "('dao', 'search'): {}", "('dao', 'search'): [{'id': 1, 'text': 'doc'}]"
    )
    harness_path.write_text(user_content, encoding="utf-8")
    # Change the method body → hash changes → region 1 regenerated.
    (tmp_path / "svc.py").write_text(
        "from layers.dao import DAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        docs = self.dao.search(q)\n"
        "        return docs\n",
        encoding="utf-8",
    )
    written, _ = write_harnesses(rubrics, eval_dir, project_root=tmp_path)
    assert any(status == "refreshed" for _, status in written)
    refreshed = harness_path.read_text(encoding="utf-8")
    # Region 2 fixture edit preserved.
    assert "[{'id': 1, 'text': 'doc'}]" in refreshed


def test_write_harnesses_skips_when_hash_unchanged(tmp_path: Path) -> None:
    """When the body hash is unchanged, the harness file is not rewritten."""
    (tmp_path / "svc.py").write_text(
        "from layers.dao import DAO\n"
        "class Svc:\n"
        "    def __init__(self):\n"
        "        self.dao = DAO()\n"
        "    def process(self, q):\n"
        "        return self.dao.search(q)\n",
        encoding="utf-8",
    )
    eval_dir = tmp_path / "eval"
    rubrics = _rubrics({
        "svc_process": TaskSpec(file_path="svc.py", entry="Svc.process", type="chat"),
    })
    write_harnesses(rubrics, eval_dir, project_root=tmp_path)
    harness_path = eval_dir / "_harness_svc_process.py"
    mtime_before = harness_path.stat().st_mtime_ns
    written, _ = write_harnesses(rubrics, eval_dir, project_root=tmp_path)
    assert any(status == "skipped" for _, status in written)
    assert harness_path.stat().st_mtime_ns == mtime_before

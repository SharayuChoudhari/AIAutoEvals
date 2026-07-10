"""Unit tests for the LangGraph workflow detector."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.detectors.langgraph import LangGraphDetector
from ai_eval.inference.signatures import collect_imports, find_callable_defs, iter_calls


def _parse(source: str) -> tuple[ast.AST, list]:
    tree = ast.parse(source)
    return tree, collect_imports(tree)


def test_langgraph_matches_on_langgraph_import(tmp_path: Path) -> None:
    source = "from langgraph.graph import StateGraph\n"
    tree, imports = _parse(source)
    assert LangGraphDetector().matches(tree, imports)


def test_langgraph_does_not_match_langchain_import(tmp_path: Path) -> None:
    """langchain is a different framework string — LangGraph detector must
    not match on it (LangChainDetector owns that family)."""
    source = "from langchain_openai import ChatOpenAI\n"
    tree, imports = _parse(source)
    assert not LangGraphDetector().matches(tree, imports)


def test_langgraph_stategraph_invoke_emits_workflow(tmp_path: Path) -> None:
    path = tmp_path / "wf.py"
    source = (
        "from langgraph.graph import END, StateGraph\n"
        "\n"
        "def run(state):\n"
        "    graph = StateGraph(dict)\n"
        "    graph.add_node('step', lambda x: x)\n"
        "    compiled = graph.compile()\n"
        "    return compiled.invoke(state)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "workflow"
    assert tasks[0].entry == "run"
    assert tasks[0].framework == "langgraph"
    assert tasks[0].inputs == ["state"]
    assert tasks[0].outputs == ["state"]
    # Emit on the invoke call, not add_node.
    assert any("invoke" in e for e in tasks[0].evidence)
    assert all("add_node" not in e for e in tasks[0].evidence)


def test_langgraph_self_graph_invoke(tmp_path: Path) -> None:
    """The appointment_helper pattern: ``self.graph.invoke(state)`` where the
    graph was built in __init__ (a different enclosing def). The detector
    emits on the invoke call in run(); construction lives in _build_graph but
    compile is construction evidence in the SAME def as invoke only when
    present. Here we model the compiled graph stored on self, so the run def
    has an invoke but no construction — to avoid a false positive we require
    construction in the same def, mirroring the plan's gate. The real
    appointment_helper repo is covered by the e2e fixture (it builds AND
    invokes in separate defs, so hints fill that gap)."""
    path = tmp_path / "svc.py"
    source = (
        "from langgraph.graph import StateGraph\n"
        "\n"
        "class Svc:\n"
        "    def run(self, state):\n"
        "        return self.graph.invoke(state)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    # No construction evidence (StateGraph/compile) in the run def — no emit.
    # This is the deliberate false-positive guard: graph.invoke alone is too
    # generic. The plan notes construction must be in the SAME enclosing def.
    assert tasks == []


def test_langgraph_compile_fallback_when_no_invoke(tmp_path: Path) -> None:
    """When a def builds and compiles a graph but never invokes it in the same
    def, emit on the compile() call (the graph's public surface)."""
    path = tmp_path / "builder.py"
    source = (
        "from langgraph.graph import StateGraph\n"
        "\n"
        "def build_graph():\n"
        "    graph = StateGraph(dict)\n"
        "    graph.add_node('s', lambda x: x)\n"
        "    return graph.compile()\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "workflow"
    assert tasks[0].entry == "build_graph"
    assert any("compile" in e for e in tasks[0].evidence)


def test_langgraph_invoke_without_construction_is_not_emitted(tmp_path: Path) -> None:
    """A bare ``x.invoke(...)`` with no StateGraph/compile in the same def
    must not emit — graph.invoke is too common a method name. The langgraph
    import prefix gates matches(), but we still require construction evidence."""
    path = tmp_path / "nope.py"
    source = (
        "from langgraph.graph import END\n"
        "\n"
        "def run(thing, state):\n"
        "    return thing.invoke(state)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert tasks == []


def test_langgraph_stategraph_without_invoke_or_compile_not_emitted(tmp_path: Path) -> None:
    """Only StateGraph(...) + add_node, no run/compile — not an entry point."""
    path = tmp_path / "partial.py"
    source = (
        "from langgraph.graph import StateGraph\n"
        "\n"
        "def setup():\n"
        "    graph = StateGraph(dict)\n"
        "    graph.add_node('a', lambda x: x)\n"
        "    graph.add_node('b', lambda x: x)\n"
        "    return graph\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert tasks == []


def test_langgraph_dedups_per_enclosing_def(tmp_path: Path) -> None:
    """Multiple invoke calls in one def emit one task."""
    path = tmp_path / "wf.py"
    source = (
        "from langgraph.graph import StateGraph\n"
        "\n"
        "def run(state):\n"
        "    g = StateGraph(dict)\n"
        "    g.add_node('a', lambda x: x)\n"
        "    c = g.compile()\n"
        "    c.invoke(state)\n"
        "    return c.invoke(state)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].entry == "run"


def test_langgraph_ainvoke_emits(tmp_path: Path) -> None:
    """async graph.ainvoke is also a run verb."""
    path = tmp_path / "awf.py"
    source = (
        "from langgraph.graph import StateGraph\n"
        "\n"
        "async def run(state):\n"
        "    g = StateGraph(dict)\n"
        "    g.add_node('a', lambda x: x)\n"
        "    c = g.compile()\n"
        "    return await c.ainvoke(state)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = LangGraphDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "workflow"
    assert any("ainvoke" in e for e in tasks[0].evidence)


def test_scan_repo_finds_langgraph_workflow(tmp_path: Path) -> None:
    (tmp_path / "wf.py").write_text(
        (
            "from langgraph.graph import END, StateGraph\n"
            "\n"
            "def run(state):\n"
            "    graph = StateGraph(dict)\n"
            "    graph.add_node('step', lambda x: x)\n"
            "    compiled = graph.compile()\n"
            "    return compiled.invoke(state)\n"
        ),
        encoding="utf-8",
    )
    result = scan_repo(tmp_path)
    assert any(t.type == "workflow" for t in result.tasks)
    wf = next(t for t in result.tasks if t.type == "workflow")
    assert wf.framework == "langgraph"
    assert wf.entry == "run"

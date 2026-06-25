"""Detector + synthesize unit tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.detectors.chromadb import ChromaDBDetector
from ai_eval.inference.detectors.langchain import LangChainDetector
from ai_eval.inference.detectors.openai_chat import OpenAIChatDetector
from ai_eval.inference.detectors.openai_tools import OpenAIToolsDetector
from ai_eval.inference.signatures import (
    OPENAI_TOOL_KWARGS,
    collect_imports,
    find_callable_defs,
    has_openai_tool_kwarg,
    is_openai_completions_create,
    iter_calls,
)
from ai_eval.inference.synthesize import build_rubrics


def _parse(source: str) -> tuple[ast.AST, list]:
    tree = ast.parse(source)
    return tree, collect_imports(tree)


# ---------------------------------------------------------------------------
# Shared predicates in signatures.py
# ---------------------------------------------------------------------------

def test_openai_tool_kwargs_constant_is_complete() -> None:
    """Ensure the shared partition constant contains all expected kwarg names."""
    assert "tools" in OPENAI_TOOL_KWARGS
    assert "functions" in OPENAI_TOOL_KWARGS
    assert "tool_choice" in OPENAI_TOOL_KWARGS


def test_is_openai_completions_create_chained_receiver() -> None:
    source = "client.chat.completions.create(messages=[])\n"
    tree = ast.parse(source)
    calls = list(iter_calls(tree))
    assert len(calls) == 1
    assert is_openai_completions_create(calls[0])


def test_has_openai_tool_kwarg_true_and_false() -> None:
    tree_with = ast.parse("f(tools=[])\n")
    tree_without = ast.parse("f(messages=[])\n")
    assert has_openai_tool_kwarg(list(iter_calls(tree_with))[0])
    assert not has_openai_tool_kwarg(list(iter_calls(tree_without))[0])


# ---------------------------------------------------------------------------
# Detector integration
# ---------------------------------------------------------------------------

def test_openai_tools_detector_picks_up_tools_kwarg(tmp_path: Path) -> None:
    path = tmp_path / "agent.py"
    source = (
        "from openai import OpenAI\n"
        "def support_agent(msgs):\n"
        "    return OpenAI().chat.completions.create(messages=msgs, tools=[])\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tools = OpenAIToolsDetector()
    chat = OpenAIChatDetector()

    assert tools.matches(tree, imports)
    tools_tasks = tools.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    chat_tasks = chat.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)

    assert len(tools_tasks) == 1
    assert tools_tasks[0].type == "tool_calling"
    assert tools_tasks[0].entry == "support_agent"
    # Plain-chat detector must NOT also claim this call.
    assert chat_tasks == []


def test_chat_detector_picks_up_non_tool_call(tmp_path: Path) -> None:
    path = tmp_path / "chat.py"
    source = (
        "from openai import OpenAI\n"
        "def chat_only(msgs):\n"
        "    return OpenAI().chat.completions.create(messages=msgs)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = OpenAIChatDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "chat"


def test_chromadb_detector_emits_rag_task(tmp_path: Path) -> None:
    path = tmp_path / "rag.py"
    source = (
        "import chromadb\n"
        "def lookup(q):\n"
        "    col = chromadb.Client().get_collection('kb')\n"
        "    return col.query(query_texts=[q])\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = ChromaDBDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"


def test_langchain_detector_invoke(tmp_path: Path) -> None:
    path = tmp_path / "chain.py"
    source = (
        "from langchain_openai import ChatOpenAI\n"
        "def chat_chain(p):\n"
        "    return ChatOpenAI().invoke(p)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = LangChainDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].framework == "langchain"


# ---------------------------------------------------------------------------
# Full scan_repo
# ---------------------------------------------------------------------------

def test_scan_repo_finds_all_tasks(tiny_repo: Path) -> None:
    result = scan_repo(tiny_repo)
    assert result.files_scanned >= 3
    names = {t.name for t in result.tasks}
    assert "customer_support_agent" in names
    assert "knowledge_base_retriever" in names
    assert "chat_chain" in names
    assert {"openai", "chromadb", "langchain"}.issubset(result.frameworks_seen)


def test_synthesize_classifies_project_type(tiny_repo: Path) -> None:
    result = scan_repo(tiny_repo)
    rubrics = build_rubrics(result)
    assert rubrics.project_type == "rag_and_tools"
    assert "customer_support_agent" in rubrics.tasks
    assert rubrics.tasks["customer_support_agent"].metrics


def test_synthesize_resolves_name_collisions() -> None:
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask

    scan = ScanResult(
        files_scanned=2,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(name="run", framework="openai", type="chat",
                         file_path="a.py", entry="run"),
            DetectedTask(name="run", framework="openai", type="chat",
                         file_path="b.py", entry="run"),
        ],
        frameworks_seen={"openai"},
    )
    rubrics = build_rubrics(scan)
    assert set(rubrics.tasks) == {"run", "run_2"}


def test_ignored_directories_not_scanned(tmp_path: Path) -> None:
    """Files inside .venv must never be enumerated."""
    venv = tmp_path / ".venv" / "lib" / "python3" / "site-packages"
    venv.mkdir(parents=True)
    (venv / "openai_stub.py").write_text(
        "from openai import OpenAI\n"
        "def f(): return OpenAI().chat.completions.create(messages=[])\n",
        encoding="utf-8",
    )
    src = tmp_path / "src"
    src.mkdir()
    (src / "real.py").write_text(
        "from openai import OpenAI\n"
        "def real(): return OpenAI().chat.completions.create(messages=[])\n",
        encoding="utf-8",
    )
    result = scan_repo(tmp_path)
    assert {t.name for t in result.tasks} == {"real"}


def test_no_duplicate_tasks_from_entrypoint_overlap(tiny_repo: Path) -> None:
    """Detectors registered via entry-points must not double-fire against builtins."""
    result = scan_repo(tiny_repo)
    names = [t.name for t in result.tasks]
    # No task name should appear more than once.
    for name in names:
        assert names.count(name) == 1, f"task {name!r} detected more than once"

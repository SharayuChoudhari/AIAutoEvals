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
from ai_eval.inference.signatures import collect_imports
from ai_eval.inference.synthesize import build_rubrics


def _parse(source: str) -> tuple[ast.AST, list]:
    tree = ast.parse(source)
    return tree, collect_imports(tree)


def test_openai_tools_detector_picks_up_tools_kwarg(tmp_path: Path) -> None:
    path = tmp_path / "agent.py"
    source = (
        "from openai import OpenAI\n"
        "def support_agent(msgs):\n"
        "    return OpenAI().chat.completions.create(messages=msgs, tools=[])\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)

    tools = OpenAIToolsDetector()
    chat = OpenAIChatDetector()

    assert tools.matches(tree, imports)
    tools_tasks = tools.extract(tree, imports, path, tmp_path)
    chat_tasks = chat.extract(tree, imports, path, tmp_path)

    assert len(tools_tasks) == 1
    assert tools_tasks[0].type == "tool_calling"
    assert tools_tasks[0].entry == "support_agent"
    # Plain-chat detector must NOT also claim this call (avoids double counting).
    assert chat_tasks == []


def test_chat_detector_skips_when_tools_present(tmp_path: Path) -> None:
    path = tmp_path / "chat.py"
    source = (
        "from openai import OpenAI\n"
        "def chat_only(msgs):\n"
        "    return OpenAI().chat.completions.create(messages=msgs)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)

    tasks = OpenAIChatDetector().extract(tree, imports, path, tmp_path)
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
    tasks = ChromaDBDetector().extract(tree, imports, path, tmp_path)
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
    tasks = LangChainDetector().extract(tree, imports, path, tmp_path)
    assert len(tasks) == 1
    assert tasks[0].framework == "langchain"


def test_scan_repo_finds_all_tasks(tiny_repo: Path) -> None:
    result = scan_repo(tiny_repo)
    assert result.files_scanned >= 3
    names = {t.name for t in result.tasks}
    assert "customer_support_agent" in names
    assert "knowledge_base_retriever" in names
    assert "chat_chain" in names
    # Frameworks seen should be the unique set, not per-task.
    assert {"openai", "chromadb", "langchain"}.issubset(result.frameworks_seen)


def test_synthesize_classifies_project_type(tiny_repo: Path) -> None:
    result = scan_repo(tiny_repo)
    rubrics = build_rubrics(result)
    # rag + tool_calling → rag_and_tools
    assert rubrics.project_type == "rag_and_tools"
    # All synthesized tasks must validate.
    assert "customer_support_agent" in rubrics.tasks
    assert rubrics.tasks["customer_support_agent"].metrics  # has default metrics


def test_synthesize_resolves_name_collisions() -> None:
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask

    scan = ScanResult(
        files_scanned=2,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="run", framework="openai", type="chat",
                file_path="a.py", entry="run",
            ),
            DetectedTask(
                name="run", framework="openai", type="chat",
                file_path="b.py", entry="run",
            ),
        ],
        frameworks_seen={"openai"},
    )
    rubrics = build_rubrics(scan)
    assert set(rubrics.tasks) == {"run", "run_2"}


def test_ignored_directories(tmp_path: Path) -> None:
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

"""Detector + synthesize unit tests."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.detectors.chromadb import ChromaDBDetector
from ai_eval.inference.detectors.langchain import LangChainDetector
from ai_eval.inference.detectors.openai_chat import OpenAIChatDetector
from ai_eval.inference.detectors.openai_tools import OpenAIToolsDetector
from ai_eval.inference.detectors.pgvector import PGVectorDetector
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


def test_pgvector_detector_psycopg2_cosine(tmp_path: Path) -> None:
    path = tmp_path / "retriever.py"
    source = (
        "import psycopg2\n"
        "def find_similar(conn, embedding):\n"
        "    cur = conn.cursor()\n"
        "    cur.execute('SELECT id FROM docs ORDER BY embedding <-> %s LIMIT 5', [embedding])\n"
        "    return cur.fetchall()\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = PGVectorDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"
    assert tasks[0].framework == "pgvector"
    assert tasks[0].entry == "find_similar"


def test_pgvector_detector_asyncpg_inner_product(tmp_path: Path) -> None:
    path = tmp_path / "search.py"
    source = (
        "import asyncpg\n"
        "async def search(pool, vec):\n"
        "    return await pool.fetch('SELECT * FROM items ORDER BY vec <#> $1', vec)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = PGVectorDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"


def test_pgvector_detector_direct_package(tmp_path: Path) -> None:
    path = tmp_path / "lookup.py"
    source = (
        "from pgvector.psycopg2 import register_vector\n"
        "import psycopg2\n"
        "def lookup(conn, q_vec):\n"
        "    cur = conn.cursor()\n"
        "    cur.execute('SELECT body FROM chunks ORDER BY embedding <=> %s LIMIT 3', [q_vec])\n"
        "    return cur.fetchall()\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = PGVectorDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"
    assert tasks[0].framework == "pgvector"


def test_pgvector_detector_sqlalchemy_no_vector_no_task(tmp_path: Path) -> None:
    path = tmp_path / "users.py"
    source = (
        "from sqlalchemy import create_engine, text\n"
        "def get_users(engine):\n"
        "    with engine.connect() as conn:\n"
        "        return conn.execute(text('SELECT * FROM users'))\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = PGVectorDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert tasks == []


def test_pgvector_detector_sqlalchemy_orm_vector_op_is_rag(tmp_path: Path) -> None:
    """Regression: pgvector used through the SQLAlchemy/SQLModel ORM
    (``Model.column.op('<=>')(...)``) was previously undetected because the
    operator lives in a separate ``op`` call, not in a ``cur.execute(...)``
    SQL string. This is the dominant pattern in ORM-based pgvector repos."""
    path = tmp_path / "document_vectors_dao.py"
    source = (
        "from sqlalchemy import select\n"
        "from sqlalchemy.ext.declarative import declarative_base\n"
        "from sqlalchemy.orm import Session\n"
        "from pgvector.sqlalchemy import Vector\n"
        "\n"
        "def search_similar_vectors(session, query_embedding):\n"
        "    distance = DocumentVector.embedding.op('<=>')(query_embedding)\n"
        "    stmt = select(DocumentVector).order_by(distance).limit(5)\n"
        "    return session.execute(stmt).scalars().all()\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = PGVectorDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"
    assert tasks[0].framework == "pgvector"
    assert tasks[0].entry == "search_similar_vectors"
    assert any("orm vector operator" in e for e in tasks[0].evidence)


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
    assert tasks[0].type == "chat"
    assert tasks[0].name == "chat_chain"


def test_langchain_retrievalqa_chain_is_rag(tmp_path: Path) -> None:
    path = tmp_path / "eval.py"
    source = (
        "from langchain.chains import RetrievalQA\n"
        "from langchain_openai import ChatOpenAI\n"
        "\n"
        "def single_eval(query):\n"
        "    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=None)\n"
        "    return qa.invoke({'query': query})\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = LangChainDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"
    # Classified rag via the ``RetrievalQA`` import signal; the emitted call is
    # the chain invoke, not a direct retrieval call site.
    assert tasks[0].name == "single_eval"
    assert tasks[0].inputs == ["query"]
    assert tasks[0].outputs == ["documents"]
    assert tasks[0].evidence[0] == "langchain qa.invoke at eval.py:6"


def test_langchain_chat_before_rag_is_rag_order_independent(tmp_path: Path) -> None:
    path = tmp_path / "mixed.py"
    source = (
        "from langchain_pinecone import PineconeVectorStore\n"
        "from langchain_openai import ChatOpenAI\n"
        "\n"
        "def mixed(query):\n"
        "    out = ChatOpenAI().invoke(query)\n"
        "    vs = PineconeVectorStore.from_existing_index('kb')\n"
        "    return vs.similarity_search(query)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = LangChainDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    types = {t.type for t in tasks}
    assert types == {"rag"}
    assert any("retriever call" in e for e in tasks[0].evidence)


def test_langchain_tenant_retriever_invoke_is_rag(tmp_path: Path) -> None:
    path = tmp_path / "services.py"
    source = (
        "from langchain_community.vectorstores import FAISS\n"
        "\n"
        "def chat_messages(query, tenant_id):\n"
        "    vectorstore = FAISS.load_local('index')\n"
        "    retriever = vectorstore.as_retriever(\n"
        "        search_kwargs={'filter': {'tenant_id': tenant_id}}\n"
        "    )\n"
        "    return retriever.invoke(query)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = LangChainDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "rag"
    assert tasks[0].name == "chat_messages"
    # The emitted task must point at the actual retrieval call
    # (``retriever.invoke``), not the ``as_retriever`` builder.
    assert tasks[0].evidence[0] == "langchain retriever.invoke at services.py:8"


def test_langchain_agent_still_precedence_over_retriever(tmp_path: Path) -> None:
    path = tmp_path / "agent.py"
    source = (
        "from langchain.agents import AgentExecutor, create_react_agent\n"
        "from langchain_community.vectorstores import FAISS\n"
        "\n"
        "def run_agent(query):\n"
        "    retriever = FAISS.load_local('i').as_retriever()\n"
        "    agent = create_react_agent(None, [retriever], None)\n"
        "    return AgentExecutor(agent=agent).invoke({'input': query})\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)
    tasks = LangChainDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    types = {t.type for t in tasks}
    assert "agent" in types
    assert "rag" not in types


def test_classify_project_type_recovers_rag_from_documents_output() -> None:
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask

    scan = ScanResult(
        files_scanned=1,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="eval",
                framework="custom",
                type="chat",
                file_path="eval.py",
                entry="eval",
                inputs=["query"],
                outputs=["documents"],
            ),
        ],
        frameworks_seen={"custom"},
    )
    rubrics = build_rubrics(scan)
    assert rubrics.project_type == "rag"


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
    assert "tenant_retriever" in names
    assert {"openai", "chromadb", "langchain", "pgvector"}.issubset(result.frameworks_seen)


def test_synthesize_classifies_project_type(tiny_repo: Path) -> None:
    result = scan_repo(tiny_repo)
    rubrics = build_rubrics(result)
    assert rubrics.project_type == "rag_and_tools"
    assert "customer_support_agent" in rubrics.tasks
    assert rubrics.tasks["customer_support_agent"].metrics


def test_synthesize_classifies_tenant_rag_repo(tenant_rag_repo: Path) -> None:
    result = scan_repo(tenant_rag_repo)
    rubrics = build_rubrics(result)
    assert rubrics.project_type == "rag"
    names = {t.name for t in result.tasks}
    assert "single_eval" in names
    assert "chat_messages" in names
    assert {t.type for t in result.tasks} == {"rag"}
    rag_metrics = {m.name for m in rubrics.tasks["single_eval"].metrics}
    assert "context_precision" in rag_metrics
    assert "faithfulness" in rag_metrics


def test_synthesize_resolves_name_collisions() -> None:
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask

    scan = ScanResult(
        files_scanned=2,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="run", framework="openai", type="chat", file_path="a.py", entry="run"
            ),
            DetectedTask(
                name="run", framework="openai", type="chat", file_path="b.py", entry="run"
            ),
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


def test_langchain_detects_chained_vectorstore_similarity_search(tmp_path: Path) -> None:
    """An inline-chained ``VectorStore.from_existing_index(...).similarity_search(...)``
    must be recognized as a retrieval call site (not just via the import flag)."""
    (tmp_path / "svc.py").write_text(
        (
            "from langchain_pinecone import PineconeVectorStore\n"
            "\n"
            "def lookup(query, tenant_id):\n"
            "    return PineconeVectorStore.from_existing_index('kb').similarity_search(\n"
            "        query, filter={'tenant_id': tenant_id}\n"
            "    )\n"
        ),
        encoding="utf-8",
    )
    result = scan_repo(tmp_path)
    assert {t.name for t in result.tasks} == {"lookup"}
    task = result.tasks[0]
    assert task.type == "rag"
    # Evidence should point at the similarity_search retrieval call site.
    assert any("similarity_search" in ev for ev in task.evidence)


def test_synthesize_promotes_chat_shaped_rag_to_rag_metrics() -> None:
    """A task typed ``chat`` but shaped like RAG (query/documents IO) must
    get rag metrics, not hallucination_rate, when recovered by _looks_like_rag."""
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask

    scan = ScanResult(
        files_scanned=1,
        elapsed_seconds=0.0,
        tasks=[
            DetectedTask(
                name="f",
                framework="langchain",
                type="chat",
                file_path="f.py",
                entry="f",
                inputs=["query"],
                outputs=["documents"],
                evidence=[],
            )
        ],
        frameworks_seen={"langchain"},
    )
    rubrics = build_rubrics(scan)
    assert rubrics.project_type == "rag"
    metrics = {m.name for m in rubrics.tasks["f"].metrics}
    assert metrics == {"context_precision", "faithfulness"}


def test_no_duplicate_tasks_from_entrypoint_overlap(tiny_repo: Path) -> None:
    """Detectors registered via entry-points must not double-fire against builtins."""
    result = scan_repo(tiny_repo)
    names = [t.name for t in result.tasks]
    # No task name should appear more than once.
    for name in names:
        assert names.count(name) == 1, f"task {name!r} detected more than once"

"""Shared pytest fixtures for ai-evals tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    # Click 8.2+ removed `mix_stderr`; stderr is captured separately by default
    # and exposed as `result.stderr` on the new CliRunner.
    return CliRunner()


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip AI_EVAL_* env + CI markers so tests are reproducible."""
    for key in list(os.environ):
        if key.startswith("AI_EVAL_") or key == "CI":
            monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A minimal fake repo with one openai chat task and one langchain rag chain."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "support.py").write_text(
        (
            "from openai import OpenAI\n"
            "\n"
            "def customer_support_agent(messages):\n"
            "    client = OpenAI()\n"
            "    return client.chat.completions.create(\n"
            "        model='gpt-4o-mini', messages=messages, tools=[]\n"
            "    )\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "rag.py").write_text(
        (
            "import chromadb\n"
            "\n"
            "def knowledge_base_retriever(query):\n"
            "    client = chromadb.Client()\n"
            "    collection = client.get_collection('kb')\n"
            "    return collection.query(query_texts=[query], n_results=4)\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "chat.py").write_text(
        (
            "from langchain_openai import ChatOpenAI\n"
            "\n"
            "def chat_chain(prompt):\n"
            "    model = ChatOpenAI()\n"
            "    return model.invoke(prompt)\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "src" / "retriever.py").write_text(
        (
            "import psycopg2\n"
            "\n"
            "def tenant_retriever(conn, query_embedding, tenant_id):\n"
            "    cur = conn.cursor()\n"
            "    cur.execute(\n"
            "        'SELECT id, content FROM docs WHERE tenant_id = %s '\n"
            "        'ORDER BY embedding <-> %s LIMIT 5',\n"
            "        [tenant_id, query_embedding],\n"
            "    )\n"
            "    return cur.fetchall()\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "ignored.py").write_text(
        "# this file lives under tests/ but isn't ignored by default scan\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tenant_rag_repo(tmp_path: Path) -> Path:
    """A minimal repo mirroring a LangChain tenant-based RAG system.

    Two modules shaped like the user's reported misclassification:
    ``evaluations/single_eval.py`` (RetrievalQA chain) and
    ``services/chat_messages.py`` (Pinecone vectorstore similarity search
    with a tenant filter).
    """
    (tmp_path / "evaluations").mkdir()
    (tmp_path / "evaluations" / "single_eval.py").write_text(
        (
            "from langchain.chains import RetrievalQA\n"
            "from langchain_openai import ChatOpenAI\n"
            "\n"
            "def single_eval(query):\n"
            "    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=None)\n"
            "    return qa.invoke({'query': query})\n"
        ),
        encoding="utf-8",
    )
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "chat_messages.py").write_text(
        (
            "from langchain_pinecone import PineconeVectorStore\n"
            "\n"
            "def chat_messages(query, tenant_id):\n"
            "    vectorstore = PineconeVectorStore.from_existing_index('kb')\n"
            "    return vectorstore.similarity_search(\n"
            "        query, filter={'tenant_id': tenant_id}\n"
            "    )\n"
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def appointment_workflow_repo(tmp_path: Path) -> Path:
    """A minimal repo mirroring the appointment_helper shape: an OpenAI
    Responses call with tools inside a graph-run method, plus a separate
    graph builder. The Responses detector fires on `_call_model` (tools=
    kwarg); the LangGraph detector requires construction + invoke in the
    SAME def, so the graph split across two methods is intentionally not
    auto-detected (hints fill that gap)."""
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "conversation_workflow_service.py").write_text(
        (
            "import os\n"
            "from typing import Any\n"
            "\n"
            "\n"
            "class ConversationWorkflowService:\n"
            "    def __init__(self):\n"
            "        self.client = self._build_openai_client()\n"
            "        self.graph = self._build_graph()\n"
            "\n"
            "    def run(self, state):\n"
            "        return self.graph.invoke(state)\n"
            "\n"
            "    def _build_openai_client(self):\n"
            "        from openai import OpenAI\n"
            "        return OpenAI(api_key=os.getenv('OPENAI_API_KEY'))\n"
            "\n"
            "    def _build_graph(self):\n"
            "        from langgraph.graph import END, StateGraph\n"
            "        graph = StateGraph(dict)\n"
            "        graph.add_node('call_model', self._call_model)\n"
            "        graph.set_entry_point('call_model')\n"
            "        graph.add_edge('call_model', END)\n"
            "        return graph.compile()\n"
            "\n"
            "    def _call_model(self, state):\n"
            "        response = self.client.responses.create(\n"
            "            model='gpt-5', input=state['messages'], tools=self._tool_schemas()\n"
            "        )\n"
            "        state['tool_calls'] = [\n"
            "            i for i in response.output if getattr(i, 'type', '') == 'function_call'\n"
            "        ]\n"
            "        return state\n"
            "\n"
            "    def _tool_schemas(self):\n"
            "        return [{'type': 'function', 'name': 'book_appointment'}]\n"
        ),
        encoding="utf-8",
    )
    return tmp_path

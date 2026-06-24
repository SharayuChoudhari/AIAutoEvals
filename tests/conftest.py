"""Shared pytest fixtures for ai-eval tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

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
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "ignored.py").write_text(
        "# this file lives under tests/ but isn't ignored by default scan\n",
        encoding="utf-8",
    )
    return tmp_path

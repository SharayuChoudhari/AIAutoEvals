"""Tests for bootstrap wrappers: capture shape, redaction, correlation.

Fake ``openai``/``langchain``/``chromadb`` modules are installed in a clean
``sys.modules`` so the patches apply without the real frameworks.
"""

from __future__ import annotations

import sys
import types

import pytest

from ai_eval.bootstrap import wrappers
from ai_eval.bootstrap.tracer import (
    configure_sink,
    enter_task,
    exit_task,
    get_tracer,
    record_call,
)


@pytest.fixture(autouse=True)
def _reset_tracer_each():
    t = get_tracer()
    t.reset()
    t._local.__dict__.clear()
    yield
    t.reset()
    t._local.__dict__.clear()


def _install_fake_openai_chat():
    """Install a minimal fake openai.resources.chat.completions.Completions."""
    mod = types.ModuleType("openai")
    sub_resources = types.ModuleType("openai.resources")
    sub_chat = types.ModuleType("openai.resources.chat")
    sub_completions = types.ModuleType("openai.resources.chat.completions")

    class Completions:
        def create(self, **kwargs):
            return {"id": "chatcmpl-1", "choices": [{"message": {"content": "hi"}}]}

    sub_completions.Completions = Completions
    sub_chat.completions = sub_completions
    sub_resources.chat = sub_chat
    mod.resources = sub_resources

    sys.modules["openai"] = mod
    sys.modules["openai.resources"] = sub_resources
    sys.modules["openai.resources.chat"] = sub_chat
    sys.modules["openai.resources.chat.completions"] = sub_completions
    return Completions


def _install_fake_chromadb():
    mod = types.ModuleType("chromadb")
    sub_api = types.ModuleType("chromadb.api")

    class Collection:
        def query(self, **kwargs):
            return {"documents": [["doc1", "doc2"]]}

    sub_api.Collection = Collection
    mod.api = sub_api
    sys.modules["chromadb"] = mod
    sys.modules["chromadb.api"] = sub_api
    return Collection


def test_install_hooks_patches_openai_chat() -> None:
    Completions = _install_fake_openai_chat()
    orig_create = Completions.create
    status = wrappers.install_hooks()
    assert status["openai.chat"] == "patched"
    # The class method is now wrapped (different callable)
    assert Completions.create is not orig_create


def test_install_hooks_absent_framework_marked() -> None:
    # pinecone not installed in the fake env
    for mod_name in list(sys.modules):
        if mod_name.startswith("pinecone"):
            del sys.modules[mod_name]
    status = wrappers.install_hooks()
    assert status["pinecone.query"] == "absent"


def test_patched_call_records_into_active_task() -> None:
    Completions = _install_fake_openai_chat()
    wrappers.install_hooks()

    captured: list[dict] = []
    configure_sink(sink=captured.append, redact_fields=None)

    enter_task("my_task", {"query": "hello"})
    c = Completions()
    result = c.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    example = exit_task()

    assert example is not None
    assert example["task"] == "my_task"
    assert example["input"] == {"query": "hello"}
    assert len(example["trace"]["calls"]) == 1
    call = example["trace"]["calls"][0]
    assert call["kind"] == "llm"
    assert call["name"] == "openai.chat.completions.create"
    assert call["result"] == result
    assert call["latency_ms"] >= 0.0
    assert captured == [example]


def test_chromadb_patched() -> None:
    Collection = _install_fake_chromadb()
    wrappers.install_hooks()
    enter_task("rag", {"query": "q"})
    Collection().query(query_texts=["q"], n_results=2)
    example = exit_task()
    assert example is not None
    assert example["trace"]["calls"][0]["kind"] == "retrieve"


def test_redaction_masks_secret_keys() -> None:
    from ai_eval.bootstrap.tracer import redact
    out = redact({"api_key": "sk-secret", "other": "keep"})
    assert out["api_key"] == "***"
    assert out["other"] == "keep"


def test_redaction_masks_env_style_string() -> None:
    from ai_eval.bootstrap.tracer import redact
    out = redact("OPENAI_API_KEY=sk-secret123 extra")
    assert "sk-secret123" not in out
    assert "OPENAI_API_KEY=***" in out


def test_nested_calls_fold_into_one_example() -> None:
    Completions = _install_fake_openai_chat()
    wrappers.install_hooks()
    enter_task("outer", {"q": 1})
    Completions().create(model="m", messages=[])
    Completions().create(model="m", messages=[])
    example = exit_task()
    assert example is not None
    assert len(example["trace"]["calls"]) == 2


def test_dedup_per_task_by_input() -> None:
    Completions = _install_fake_openai_chat()
    wrappers.install_hooks()
    captured: list[dict] = []
    configure_sink(sink=captured.append, redact_fields=None)

    for _ in range(3):
        enter_task("t", {"x": 1})
        Completions().create(model="m", messages=[])
        exit_task()

    # Only the first is emitted; the next two are deduped.
    assert len(captured) == 1


def test_record_call_no_active_task_is_noop() -> None:
    # No enter_task -> record_call drops silently
    record_call("llm", "x", [], None, 1.0)
    assert get_tracer().current is None

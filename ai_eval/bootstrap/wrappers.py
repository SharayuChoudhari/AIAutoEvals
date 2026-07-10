"""Monkey-patch LLM/retrieval frameworks to capture runtime traces.

Each patch is **independently skippable**: if the framework isn't installed
(``ImportError``) or its API shape differs (``AttributeError``), the patch is
skipped and the framework is recorded as "not patched" in capture metadata.
This keeps the bootstrapper robust across user environments that have only a
subset of the frameworks.

Each patched call records ``{kind, name, args, result, latency_ms}`` into the
active task's trace via :func:`ai_eval.bootstrap.tracer.record_call`.

Redaction is controlled by the tracer's ``redact_fields`` flag (set via
:func:`configure_sink`); when enabled, args/results are masked before capture.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ai_eval.bootstrap.tracer import get_tracer, record_call

#: Frameworks we attempted to patch + whether we succeeded. Populated by
#: :func:`install_hooks` so the capture metadata records what was observable.
PATCH_STATUS: dict[str, str] = {}


def _wrap(orig: Callable, *, kind: str, name: str) -> Callable:
    """Return a wrapper that times + records ``orig``'s call."""

    def _sync(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig(*args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        record_call(kind, name, args=list(args), result=result, latency_ms=latency_ms)
        return result

    async def _async(*args, **kwargs):
        t0 = time.perf_counter()
        result = await orig(*args, **kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        record_call(kind, name, args=list(args), result=result, latency_ms=latency_ms)
        return result

    if _is_coroutine(orig):
        return _async
    return _sync


def _is_coroutine(fn: Any) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(fn)


def _patch_openai_chat() -> bool:
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        return False
    orig = getattr(Completions, "create", None)
    if orig is None:
        return False
    Completions.create = _wrap(orig, kind="llm", name="openai.chat.completions.create")  # type: ignore[method-assign]
    return True


def _patch_openai_responses() -> bool:
    try:
        from openai.resources.responses import Responses
    except ImportError:
        return False
    orig = getattr(Responses, "create", None)
    if orig is None:
        return False
    Responses.create = _wrap(orig, kind="llm", name="openai.responses.create")  # type: ignore[method-assign]
    return True


def _patch_langchain_chat() -> bool:
    try:
        from langchain_core.language_models import BaseChatModel  # type: ignore[import-not-found]
    except ImportError:
        return False
    orig = getattr(BaseChatModel, "invoke", None)
    if orig is None:
        return False
    BaseChatModel.invoke = _wrap(orig, kind="llm", name="langchain.chat.invoke")  # type: ignore[method-assign]
    return True


def _patch_chromadb_query() -> bool:
    try:
        from chromadb.api import Collection  # type: ignore[import-not-found]
    except ImportError:
        return False
    orig = getattr(Collection, "query", None)
    if orig is None:
        return False
    Collection.query = _wrap(orig, kind="retrieve", name="chromadb.collection.query")  # type: ignore[method-assign]
    return True


def _patch_pinecone_query() -> bool:
    try:
        from pinecone.data_index import Index  # type: ignore[import-not-found]
    except ImportError:
        return False
    orig = getattr(Index, "query", None)
    if orig is None:
        return False
    Index.query = _wrap(orig, kind="retrieve", name="pinecone.index.query")  # type: ignore[method-assign]
    return True


def _patch_pinecone_similarity_search() -> bool:
    try:
        from pinecone import PineconeVectorStore  # type: ignore[import-not-found]
    except ImportError:
        return False
    orig = getattr(PineconeVectorStore, "similarity_search", None)
    if orig is None:
        return False
    PineconeVectorStore.similarity_search = _wrap(  # type: ignore[method-assign]
        orig, kind="retrieve", name="pinecone.similarity_search"
    )
    return True


_PATCHERS = [
    ("openai.chat", _patch_openai_chat),
    ("openai.responses", _patch_openai_responses),
    ("langchain.chat", _patch_langchain_chat),
    ("chromadb.query", _patch_chromadb_query),
    ("pinecone.query", _patch_pinecone_query),
    ("pinecone.similarity_search", _patch_pinecone_similarity_search),
]


def install_hooks(
    *,
    redact_fields: bool = True,
) -> dict[str, str]:
    """Install all available framework patches; return ``{framework: status}``.

    Each patch is independently skippable on ``ImportError``/``AttributeError``.
    ``status`` is ``"patched"`` or ``"absent"``. The caller records this into
    capture metadata so failures are diagnosable.
    """
    PATCH_STATUS.clear()
    from ai_eval.bootstrap.tracer import configure_sink
    configure_sink(
        sink=_noop_sink, redact_fields=[] if redact_fields else None
    )
    # Re-enable redaction flag on the tracer (configure_sink sets redact_fields)
    if redact_fields:
        get_tracer().redact_fields = ["*"]

    for name, fn in _PATCHERS:
        try:
            ok = fn()
        except (ImportError, AttributeError):
            ok = False
        PATCH_STATUS[name] = "patched" if ok else "absent"
    return dict(PATCH_STATUS)


def _noop_sink(example: dict) -> None:
    """Default no-op sink; the real sink is wired by the sitecustomize shim."""


__all__ = ["PATCH_STATUS", "install_hooks"]

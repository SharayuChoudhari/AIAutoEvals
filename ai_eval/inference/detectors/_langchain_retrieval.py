"""Retrieval / vectorstore heuristics shared by the LangChain detector.

Extracted from ``langchain.py`` so the detector class stays comparable in
size to its sibling detectors. These constants and pure helpers identify
LangChain-family vectorstore / retriever usage from imports and call sites.
"""

from __future__ import annotations

import ast

from ai_eval.inference.signatures import ImportInfo, attr_chain

#: Retrieval-chain constructors / base classes imported from langchain.
RETRIEVAL_CHAIN_NAMES = {
    "RetrievalQA",
    "ConversationalRetrievalChain",
    "BaseRetrievalQA",
    "RetrievalChain",
    "BaseRetriever",
}

#: Vector-store provider packages whose module path lacks the literal
#: ``vectorstores`` (e.g. ``langchain_pinecone``), so the bare ``vectorstores``
#: substring check misses them.
VECTORSTORE_PROVIDER_PREFIXES = (
    "langchain_pinecone",
    "langchain_weaviate",
    "langchain_chroma",
    "langchain_milvus",
    "langchain_qdrant",
    "langchain_pgvector",
)

#: Vector-store / retriever class-name hints bound from the provider packages
#: above or from ``langchain_community.vectorstores``.
VECTORSTORE_NAME_HINTS = (
    "VectorStore",
    "FAISS",
    "Pinecone",
    "Weaviate",
    "Chroma",
    "Milvus",
    "Qdrant",
    "PGVector",
)

#: Method names that indicate a vectorstore similarity-search call site.
VECTORSTORE_SEARCH_METHODS = {
    "similarity_search",
    "similarity_search_with_score",
    "similarity_search_by_vector",
    "max_marginal_relevance_search",
    "similarity_search_with_relevance_scores",
}

#: Method names that indicate an actual retrieval call site. ``as_retriever``
#: is intentionally excluded — it is a retriever *factory* that performs no
#: retrieval; the import-based signal covers it, and the real retrieval call
#: is the subsequent ``.invoke()`` / ``get_relevant_documents()``.
RETRIEVER_METHODS = {"get_relevant_documents"}


def retrieval_chain_imported(imports: list[ImportInfo]) -> bool:
    """True if a known retrieval-chain constructor is imported."""
    for info in imports:
        if info.name in RETRIEVAL_CHAIN_NAMES:
            return True
        if info.name.endswith("RetrievalQA") or info.name.endswith("RetrievalChain"):
            return True
    return False


def vectorstore_provider_imported(imports: list[ImportInfo]) -> bool:
    """True if a vector-store provider package is imported."""
    for info in imports:
        for prefix in VECTORSTORE_PROVIDER_PREFIXES:
            if info.module == prefix or info.module.startswith(prefix + "."):
                return True
    return False


def vectorstore_imported(imports: list[ImportInfo]) -> bool:
    return any("vectorstores" in info.module for info in imports) or any(
        info.name in VECTORSTORE_NAME_HINTS for info in imports
    )


def receiver_is_vectorstore_like(
    chain: list[str], call: ast.Call | None = None
) -> bool:
    """Heuristic: does the receiver name look like a vectorstore binding?

    Matches a local binding whose name hints at a vectorstore (e.g.
    ``vectorstore``, ``store``, ``index``, ``retriever``, ``db``), or any
    attribute chain carrying a vectorstore name hint.

    When the receiver is itself a call (chain head ``<call>``) — e.g. an
    inline-chained ``VectorStore.from_existing_index(...).similarity_search(...)`` —
    descend into the receiver call's own attribute chain and treat any
    vectorstore name hint there (e.g. ``PineconeVectorStore``) as vectorstore-like.
    """
    head = chain[0] if chain else ""
    if head in {"<call>", "<sub>"}:
        # The receiver is itself a call (e.g. an inline-chained
        # ``VectorStore.from_existing_index(...).similarity_search(...)``).
        # Descend into that receiver call's own attribute chain and look for a
        # vectorstore name hint there (e.g. ``PineconeVectorStore``).
        if call is None or not isinstance(call.func, ast.Attribute):
            return False
        receiver = call.func.value
        if not isinstance(receiver, ast.Call):
            return False
        receiver_func = receiver.func
        if not isinstance(receiver_func, ast.Attribute):
            return False
        inner = attr_chain(receiver_func)
        return any(hint in part for part in inner for hint in VECTORSTORE_NAME_HINTS)
    if head in {"vectorstore", "store", "index", "retriever", "db", "vs"}:
        return True
    if any(hint in head for hint in VECTORSTORE_NAME_HINTS):
        return True
    # Head already checked above; scan the rest of the chain.
    return any(
        hint in part for part in chain[1:] for hint in VECTORSTORE_NAME_HINTS
    )


def is_retriever_call_site(chain: list[str], call: ast.Call | None = None) -> bool:
    """True when an already-computed ``chain`` looks like a retrieval call site."""
    if not chain:
        return False
    last = chain[-1]
    if last in RETRIEVER_METHODS:
        return True
    if last in VECTORSTORE_SEARCH_METHODS and receiver_is_vectorstore_like(
        chain, call
    ):
        return True
    return False


__all__ = [
    "RETRIEVAL_CHAIN_NAMES",
    "RETRIEVER_METHODS",
    "VECTORSTORE_NAME_HINTS",
    "VECTORSTORE_PROVIDER_PREFIXES",
    "VECTORSTORE_SEARCH_METHODS",
    "is_retriever_call_site",
    "receiver_is_vectorstore_like",
    "retrieval_chain_imported",
    "vectorstore_imported",
    "vectorstore_provider_imported",
]

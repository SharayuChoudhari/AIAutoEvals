# Plan: Fix misclassification of tenant-based RAG as `chat` in `ai-eval init`

## Problem

`ai-eval init` generated a `rubrics.yaml` with `project_type: chat` for what is a
tenant-based RAG system. The two detected tasks both came out as `type: chat`:

- `single_eval_chain`  (`evaluations/single_eval.py`)
- `chat_messages_chain` (`services/chat_messages.py`)

Both task names carry the `_chain` suffix, which is emitted **only** by
`LangChainDetector` (`ai_eval/inference/detectors/langchain.py:67`:
`name = entry or f"{file_path.stem}_chain"`). So the LangChain detector is the
one that fired, and it fell through to the `chat` branch
(`langchain.py:77-79`).

### Root cause

`LangChainDetector.extract` classifies a chain invoke/run call as `rag` only
when its `retriever` heuristic is true (`langchain.py:53-55`):

```python
retriever = any(info.name == "as_retriever" for info in imports) or any(
    "vectorstores" in info.module for info in imports
)
```

This heuristic is **import-based** and misses the dominant tenant-RAG patterns:

1. **RetrievalQA / conversational retrieval chains** built via
   `from langchain.chains import RetrievalQA` (no `as_retriever` import, no
   `vectorstores` module).
2. **Vector store retrievers invoked directly** — `vectorstore.similarity_search(...)`
   or `retriever.invoke(...)` where the import is e.g.
   `from langchain_community.vectorstores import FAISS` (module contains
   `vectorstores`, so this *sometimes* works) but a Pinecone/Weaviate/etc.
   import path like `langchain_pinecone import PineconeVectorStore` does **not**
   contain the literal `vectorstores`.
3. **Tenant-scoped retrieval** where the tenant filter is applied in SQL/Python
   before/after the vector call — the detector has no notion of multi-tenancy
   and treats the whole function as a plain chat chain because it sees a
   `.invoke()` on a chain whose imports don't match the retriever heuristic.

Because both tasks classified as `chat`, `_classify_project_type`
(`synthesize.py:43-55`) sees `has_rag=False` and falls through to `return
"chat"`.

## Scope

Fix detection so tenant-based RAG built on LangChain is classified as `rag`
(not `chat`), producing the correct `project_type` and the RAG default metric
set (`context_precision`, `faithfulness`) instead of `hallucination_rate`.

This plan is intentionally **LangChain-focused** because the evidence (the
`_chain` suffix in both task names) points squarely at `LangChainDetector`.
The OpenAI / ChromaDB / pgvector detectors already handle their own RAG paths
and are not the culprit here. A secondary hardening for the
`project_type` fallback is included.

## Changes

### 1. Broaden the LangChain retriever heuristic
**File:** `ai_eval/inference/detectors/langchain.py`

Replace the import-only `retriever` check with a combined import + call-site
heuristic so retrieval is detected even when the tenant filter lives in user
code:

- **Import signals (additive):**
  - current: `info.name == "as_retriever"` or `"vectorstores" in info.module`
  - add: module/name in a known set of retrieval-chain constructors and
    retriever base classes, e.g.
    `RetrievalQA`, `ConversationalRetrievalChain`, `BaseRetrievalQA`,
    `RetrievalChain`, and any `*RetrievalQA` / `*RetrievalChain` suffix match.
  - add: vector-store provider packages whose module path does not contain the
    literal `vectorstores`, e.g. `langchain_pinecone`, `langchain_weaviate`,
    `langchain_chroma`, `langchain_milvus`, `langchain_qdrant`, `langchain_pgvector`.
- **Call-site signals (new):** a call whose attribute chain ends in a retrieval
  method should mark the enclosing task as `rag` regardless of imports:
  - `as_retriever(...)`
  - `similarity_search(...)`, `similarity_search_with_score(...)`,
    `max_marginal_relevance_search(...)`, `selectivity(...)` on any receiver
    whose chain contains a known vectorstore hint.
  - `get_relevant_documents(...)` / `ainvoke` on a retriever-shaped receiver.
- Keep the existing agentic branch taking precedence over the retriever branch
  (agents may also retrieve).

### 2. Capture retrieval evidence
When the new call-site heuristic fires, append an evidence string
(`langchain retriever call ... at rel:lineno`) to `DetectedTask.evidence` so
`ai-eval doctor` / future explain output can justify the `rag` classification.

### 3. Normalize RAG task naming for retrieval chains
For tasks classified `rag` via the new heuristic, prefer
`entry or f"{file_path.stem}_retriever"` (matching ChromaDB/pgvector naming)
instead of the generic `_chain` suffix. This makes tenant RAG tasks
distinguishable from chat chains in the generated rubrics. Keep `_chain` for
the genuine `chat` fall-through so the suffix remains a reliable signal.

### 4. Harden `_classify_project_type` fallback
**File:** `ai_eval/inference/synthesize.py`

When at least one task has `inputs`/`outputs` resembling retrieval
(e.g. `"query"` in inputs and `"documents"` in outputs) but its `type` is
`chat`, do not silently downgrade to `project_type: chat`. Add a minimal
guard: if any `chat` task's `outputs` contain `documents`, treat it as `rag`
for project-type purposes (defensive — should be rare after fix #1, but
prevents silent misclassification for undetected frameworks).

### 5. Tests
**File:** `tests/test_detectors.py` (and `tests/conftest.py` fixtures)

Add cases mirroring the user's tenant-RAG shapes:

- `test_langchain_retrievalqa_chain_is_rag`: a `RetrievalQA.from_chain_type(...)`
  invoke → single task, `type == "rag"`, RAG metrics.
- `test_langchain_vectorstore_provider_similarity_search_is_rag`: e.g.
  `from langchain_pinecone import PineconeVectorStore` then
  `vectorstore.similarity_search(query, filter={"tenant_id": tid})` → `rag`.
- `test_langchain_tenant_retriever_invoke_is_rag`: retriever built from a
  vectorstore with a tenant `search_kwargs` filter, `.invoke(query)` → `rag`.
- `test_classify_project_type_recovers_rag_from_documents_output`: unit test
  for the synthesize guard in change #4.
- Extend `tiny_repo` (or add a `tenant_rag_repo` fixture) with a LangChain
  tenant-RAG module and assert `build_rubrics(...).project_type == "rag"` (or
  `"rag_and_tools"` when combined with the existing agent task).

### 6. Docs / changelog note (optional, only if project convention)
No README change required unless the user wants one; detection improvements
are internal. Skip unless requested.

## Validation

- `pytest tests/test_detectors.py -q` — all old + new detector tests pass.
- `pytest tests/test_cli_init.py -q` — init dry-run still classifies
  `tiny_repo` as `rag_and_tools`.
- `ruff check ai_eval/inference tests` and `mypy ai_eval/inference` (or the
  project's configured lint/typecheck) — clean.
- Manual: build a tiny throwaway repo matching the user's
  `evaluations/single_eval.py` + `services/chat_messages.py` shapes and run
  `ai-eval init --dry-run`; confirm `project_type: rag` and RAG metrics.

## Out of scope

- Multi-tenant first-class modeling (a `tenant_id` input field, per-tenant
  golden sets) is a larger feature; this plan only ensures tenant RAG is
  *detected as RAG*. Surfacing tenant scoping in the rubric is a follow-up.
- New detectors for frameworks other than LangChain.
- Phase 2-4 judging/run behavior.

## Risk / notes

- Broadening the retriever heuristic could over-classify a hybrid
  chat+retrieval chain as `rag` when the user thinks of it as a chat agent.
  Mitigation: agentic signals still take precedence, and the call-site
  heuristic keys off concrete retrieval method names, not just imports.
- Name change (`_chain` → `_retriever`) shifts task keys in the rubric;
  acceptable for a fresh `init`/`analyze --force` run, but document it.

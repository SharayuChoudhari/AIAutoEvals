# Plan: PGVector Detector

## Goal

Add a `PGVectorDetector` so that `ai-eval init` correctly classifies projects using pgvector (via psycopg2, psycopg3, asyncpg, SQLAlchemy, or the `pgvector` Python package) as `type: rag` with the correct metrics (`context_precision`, `faithfulness`) instead of falling back to `chat` + `hallucination_rate`.

---

## Files to Change

### 1. NEW — `ai_eval/inference/detectors/pgvector.py`

Model after `chromadb.py`. Two-stage detection:

**`matches()`** — returns `True` if any of these import prefixes are present:
- `psycopg2`
- `psycopg` (covers psycopg3)
- `asyncpg`
- `sqlalchemy`
- `pgvector`

**`extract()`** — for every `ast.Call` in the file:
1. Compute `attr_chain(call.func)`.
2. Check if the last element of the chain is in the set of DB execution methods:
   `{"execute", "fetch", "fetchrow", "fetchval", "fetchone", "fetchall", "query", "search", "similarity_search"}`.
3. For calls matching step 2, scan **all `ast.Constant` string arguments** (positional and keyword) for the presence of pgvector distance operator substrings: `"<->"`, `"<=>"`, `"<#>"`.
   - For imports of `pgvector` directly (no SQL string needed), `.query()` or `.search()` alone is sufficient signal.
4. If a vector operator is found (or it's a direct `pgvector` import + query/search call), emit a `DetectedTask` with:
   - `type="rag"`
   - `framework="pgvector"`
   - `inputs=["query", "embedding"]`
   - `outputs=["documents", "distances"]`
   - `evidence=[f"pgvector {'.'.join(chain)} at {rel}:{call.lineno}"]`

**Deduplication:** same `seen: set[str]` pattern as `ChromaDBDetector` — one task per enclosing function name per file.

**SQLAlchemy breadth concern:** `matches()` will fire on any SQLAlchemy file, but `extract()` only emits a task when a vector operator string is found in a SQL argument — safe fallback to zero tasks for non-vector SQLAlchemy usage.

---

### 2. EDIT — `ai_eval/inference/ast_scan.py`

Add import and register in `builtin_detectors()`:

```python
# Add import at top with other detectors:
from ai_eval.inference.detectors.pgvector import PGVectorDetector

# In builtin_detectors(), insert before OpenAIChatDetector:
def builtin_detectors() -> list[Detector]:
    return [
        OpenAIToolsDetector(),
        PGVectorDetector(),        # ← new, before generic chat detector
        OpenAIChatDetector(),
        LangChainDetector(),
        ChromaDBDetector(),
    ]
```

Placement before `OpenAIChatDetector` ensures that a file importing both `openai` and `psycopg2` (typical tenantwise RAG: retrieve from PG, generate with OpenAI) gets the correct RAG task emitted by `PGVectorDetector` in addition to the chat task — not instead of, since they operate on different call sites.

---

### 3. EDIT — `tests/test_detectors.py`

Add four new test cases mirroring the style of `test_chromadb_detector_emits_rag_task`:

**`test_pgvector_detector_psycopg2_cosine(tmp_path)`**
```python
source = (
    "import psycopg2\n"
    "def find_similar(conn, embedding):\n"
    "    cur = conn.cursor()\n"
    "    cur.execute('SELECT id FROM docs ORDER BY embedding <-> %s LIMIT 5', [embedding])\n"
    "    return cur.fetchall()\n"
)
```
Assert: 1 task, `type == "rag"`, `framework == "pgvector"`, `entry == "find_similar"`.

**`test_pgvector_detector_asyncpg_inner_product(tmp_path)`**
```python
source = (
    "import asyncpg\n"
    "async def search(pool, vec):\n"
    "    return await pool.fetch('SELECT * FROM items ORDER BY vec <#> $1', vec)\n"
)
```
Assert: 1 task, `type == "rag"`.

**`test_pgvector_detector_direct_package(tmp_path)`**
```python
source = (
    "from pgvector.psycopg2 import register_vector\n"
    "import psycopg2\n"
    "def lookup(conn, q_vec):\n"
    "    cur = conn.cursor()\n"
    "    cur.execute('SELECT body FROM chunks ORDER BY embedding <=> %s LIMIT 3', [q_vec])\n"
    "    return cur.fetchall()\n"
)
```
Assert: 1 task, `type == "rag"`, `framework == "pgvector"`.

**`test_pgvector_detector_sqlalchemy_no_vector_no_task(tmp_path)`**
```python
source = (
    "from sqlalchemy import create_engine, text\n"
    "def get_users(engine):\n"
    "    with engine.connect() as conn:\n"
    "        return conn.execute(text('SELECT * FROM users'))\n"
)
```
Assert: 0 tasks emitted (plain SQLAlchemy without vector operators must not fire).

---

### 4. EDIT — `tests/conftest.py`

Add a pgvector file to the `tiny_repo` fixture so `scan_repo` integration tests cover the new detector:

```python
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
```

Update assertions in `test_scan_repo_finds_all_tasks`:
```python
assert "tenant_retriever" in names
assert "pgvector" in result.frameworks_seen
```

`test_synthesize_classifies_project_type` needs no change — `project_type` is already asserted as `"rag_and_tools"`, which still holds.

---

## Implementation Order

1. `ai_eval/inference/detectors/pgvector.py` — new detector
2. `ai_eval/inference/ast_scan.py` — register detector
3. `tests/conftest.py` — extend fixture
4. `tests/test_detectors.py` — add unit + integration tests
5. Run `pytest tests/test_detectors.py` to verify

---

## What This Does Not Change

- `synthesize.py` — `_DEFAULT_METRICS["rag"]` already returns `context_precision` + `faithfulness`. Once tasks are correctly typed as `rag`, the right metrics flow automatically.
- `schema.py` — `"pgvector"` does not need to be a `ProjectType`; the project type `"rag"` is correct and sufficient.
- LangChain PGVector wrapper (`langchain_community.vectorstores.PGVector`) — already caught by `LangChainDetector`'s `"vectorstores" in info.module` check. No change needed there.

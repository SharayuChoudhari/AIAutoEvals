# Plan: Fix ai-evals auto-seed call failures on real Basic_RAG repo

## Problem statement

After the 0.1.4 release, `ai-evals run` against the real Basic_RAG repo
(`/Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG`) fails every auto-seeded
example with `TypeError` / `AttributeError`. There are **two independent
problems**, and both must be fixed for `ai-evals run` to produce the
green-pipeline (non-error) result the D6 auto-seed contract promises.

---

## Problem A — Stale install (PRIMARY BLOCKER)

### Symptom
The exact same `TypeError` errors appear after editing the local engine source
and running `uv run ai-evals run` in Basic_RAG. The fix is not loaded.

### Root cause
Basic_RAG's venv holds the **published `dot-ai-evals==0.1.4` wheel** as a
regular (non-editable) install:

```
uv pip show dot-ai-evals   (run inside Basic_RAG)
  Name: dot-ai-evals
  Version: 0.1.4
  Location: /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG/.venv/lib/python3.13/site-packages
```

Basic_RAG depends on it via `pyproject.toml`:
```toml
"dot-ai-evals>=0.1.4",
```

The local AIAutoEvals source has `version = "0.1.4"` (same string), so uv sees
"already satisfied" and skips reinstall. A prior `uv pip install -e .` either
ran in the wrong venv or against the wrong package name, leaving the stale
wheel in place.

### Verification (already confirmed)
```bash
cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
uv run python -c "import ai_eval.runner.engine as e; print(e.__file__); print(hasattr(e,'_build_call_args'))"
# → /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG/.venv/lib/python3.13/site-packages/ai_eval/runner/engine.py
# → False   ← stale wheel, fix NOT present
```
```bash
cd /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals
uv run python -c "import ai_eval.runner.engine as e; print(e.__file__); print(hasattr(e,'_build_call_args'))"
# → /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals/ai_eval/runner/engine.py
# → True    ← local source HAS the fix
```

### Fix
Editable-install the local source into Basic_RAG's venv so future edits are
live without reinstalling:

```bash
cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
uv pip install -e /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals --reinstall-package dot-ai-evals
```

Then re-verify the engine loads from the source path:
```bash
cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
uv run python -c "import ai_eval.runner.engine as e; print(e.__file__)"
# MUST print: /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals/ai_eval/runner/engine.py
```

If it still prints the site-packages path, the editable install didn't take —
check `uv pip show dot-ai-evals` shows an editable/`-e` location, and that no
`ai_eval` dir shadow exists on `sys.path` ahead of the editable install.

---

## Problem B — Scalar-binding too aggressive for object-typed params

### Symptom (appears only after Problem A is fixed)
For `chat_message_service_process_query`, the error changes from a missing-arg
`TypeError` to:
```
AttributeError: 'str' object has no attribute 'chat_id'
```

### Root cause
`ai_eval/runner/engine.py::_build_call_args` (added in the 0.1.4→unreleased
fix for multi-arg methods) binds the auto-seed scalar `""` to the **first
required param regardless of its type annotation**. For methods whose first
required param is object-typed, the method body does attribute access on a
bare string and crashes.

### Affected real Basic_RAG methods

| Task entry | Signature | Current binding | Result |
|---|---|---|---|
| `DocumentVectorDAO.search_similar_vectors` | `(self, query_embedding, user_id, limit=5, ...)` | scalar → `query_embedding` | OK-ish (stub fills `user_id`) |
| `SingleQueryEvaluator.evaluate_single` | `(self, question: str, retrieved_contexts: List[str], answer: str, ...)` | scalar → `question` (str) | OK |
| `ChatMessageService._create_workflow` | `(self)` | no args (zero-param path) | OK |
| `ChatMessageService.process_query` | `(self, query_request: ChatQueryRequest)` | scalar → `query_request` | **FAILS** — `query_request.chat_id` on a str |

### What `_Stub` does (context for the fix)
`_Stub` (`ai_eval/runner/engine.py:49`) is a permissive stand-in: any
attribute access returns a new `_Stub`, and calling it returns a `_Stub`. So
`_Stub().chat_id` → `_Stub()` (truthy), `_Stub().query.strip()` → `_Stub()`.
It lets object-typed method bodies run without `AttributeError`, fulfilling
the D6 "green pipeline" intent until the user runs `ai-evals bootstrap` for
real baselines.

### Current binding logic (the buggy part)
In `_build_call_args`, after collecting `slots` (non-self positional params):
```python
required = [p for p in slots if p.default is inspect.Parameter.empty]
scalar_idx = -1
if required:
    scalar_idx = next(
        (i for i, p in enumerate(required) if p.annotation is str),
        0,   # ← BUG: falls back to index 0 even when that param is object-typed
    )
```
The `0` fallback binds the scalar to an object-typed param.

### Recommended fix — heuristic binding rule
Replace the fallback so the scalar binds **only** when the param is plausibly
string-like:

1. If any required param is annotated `str` → bind the scalar to the first
   `str`-annotated required param; all other required params get `_placeholder_for`.
2. Else if **all** required params are **unannotated** → bind the scalar to the
   first (treat unannotated as string-like, preserving the `process(self, q)`
   single-arg fast-path contract).
3. Else (some required param has a non-str annotation, e.g.
   `query_request: ChatQueryRequest`) → **do not bind the scalar**; give every
   non-str required param a `_placeholder_for` (which yields `_Stub` for
   object-typed/unannotated-here params). The seed string is effectively
   dropped for that example (acceptable for a green-pipeline sanity run).

Pseudocode for the new `scalar_idx` decision:
```python
required = [p for p in slots if p.default is inspect.Parameter.empty]
str_idxs = [i for i, p in enumerate(required) if p.annotation is str]
all_unannotated = all(p.annotation is inspect.Parameter.empty for p in required)

if str_idxs:
    scalar_idx = str_idxs[0]                      # case 1
elif all_unannotated and required:                # case 2
    scalar_idx = 0
else:
    scalar_idx = -1                               # case 3: no scalar binding
```
The downstream loop already handles `scalar_idx == -1` (no slot gets the
scalar; every required param gets a placeholder) — verify that path produces
all-placeholders and no scalar.

### Why this preserves existing tests
- `test_build_call_args_multi_required_str_binds_to_first_str_param`
  (`evaluate_single`-shape) → case 1, scalar → `question`. ✓
- `test_build_call_args_no_str_param_binds_scalar_to_first_required`
  (`search_similar_vectors`-shape, all unannotated) → case 2, scalar →
  `query_embedding`. ✓
- `test_build_call_args_zero_user_params_returns_empty` (`_create_workflow`)
  → unchanged (no slots). ✓
- `test_dotted_entry_required_non_self_arg_constructs`
  (`process(self, q)` unannotated) → case 2, scalar → `q`. ✓
- `test_engine_multi_arg_method_runs_from_scalar_auto_seed`
  (`evaluate_single(self, question: str, retrieved_contexts, answer: str)`)
  → `question` is str → case 1. ✓ (note: `retrieved_contexts` is unannotated
  but `all_unannotated` is False because `question`/`answer` are str → case 1
  wins, scalar → `question`, others get `_Stub`/`""`.)

### New test to add
Mirror `ChatMessageService.process_query`:
```python
def test_engine_object_typed_single_param_gets_stub_not_scalar(tmp_path):
    # process_query(self, query_request: ChatQueryRequest) with input=""
    # → query_request is a _Stub, body's .chat_id access doesn't raise.
    (tmp_path / "svc.py").write_text(
        "class ChatQueryRequest:\n"
        "    def __init__(self, chat_id='', query=''): ...\n"
        "class Svc:\n"
        "    def __init__(self, session):\n"
        "        self.session = session\n"
        "    async def process_query(self, query_request: ChatQueryRequest):\n"
        "        cid = query_request.chat_id   # would crash on a bare str\n"
        "        return {'chat_id': cid}\n",
        encoding="utf-8",
    )
    rubrics = _rubrics({"svc_pq": TaskSpec(file_path="svc.py",
        entry="Svc.process_query", type="chat", metrics=[])})
    golden = {"svc_pq": [{"id": "e1", "input": "", "seed": "auto"}]}
    record = _run(execute(rubrics, golden, project_root=tmp_path, parallel=1,
        complete_fn=None, run_id="obj", started_at=time.time()))
    task = record.tasks["svc_pq"]
    assert all(e.status != "error" for e in task.examples), task.errors
    assert not any("has no attribute" in (e.error or "") for e in task.examples)
```

---

## Implementation order (in a new session)

1. **Apply the Problem B engine fix** in
   `ai_eval/runner/engine.py::_build_call_args` (the `scalar_idx` decision
   above). Add the new object-typed test to
   `tests/test_runner_dotted_entry.py`.
2. **Run the local test suite** from the AIAutoEvals repo:
   ```bash
   cd /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals
   uv run pytest tests/test_runner_dotted_entry.py tests/test_runner_engine.py tests/test_integration_basic_rag.py -q
   uv run ruff check ai_eval/runner/engine.py tests/test_runner_dotted_entry.py
   uv run ruff format ai_eval/runner/engine.py tests/test_runner_dotted_entry.py
   uv run mypy ai_eval/runner/engine.py
   ```
   All must pass. Expected: 362+ tests green (332 existing + the new test).
3. **Apply the Problem A install fix** — editable-install into Basic_RAG's venv:
   ```bash
   cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
   uv pip install -e /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals --reinstall-package dot-ai-evals
   uv run python -c "import ai_eval.runner.engine as e; print(e.__file__, hasattr(e,'_build_call_args'))"
   ```
   Must print the source path and `True`.
4. **Re-run on Basic_RAG**:
   ```bash
   cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
   uv run ai-evals run
   ```
   Expected: errors drop substantially. The auto-seeded pure-LLM tasks
   (`evaluate_single`, `_create_workflow`) should produce non-error results
   (status `skip` with the "auto-seeded fixtures" notice). `process_query`
   should also no longer raise `AttributeError` — it may still fail later in
   its body (async/langgraph/DB) but that's a per-task IO concern outside this
   fix's scope; the D6 contract directs users to `ai-evals bootstrap` for
   trustworthy baselines on IO-coupled tasks.

---

## Boundaries / out of scope

- This fix does **not** make auto-seeded runs produce meaningful *scores* —
  that requires `ai-evals bootstrap` to capture real traces. Auto-seeds are a
  green-pipeline sanity check only.
- IO-coupled tasks (`search_similar_vectors`, `process_query`) may still error
  deeper in their bodies (DB/langgraph) even with correct arg binding — the
  harness stubs only cover `self.<attr>.<method>()` reads classified by D4.
  Improving harness coverage is a separate concern.
- No changes to the seeder, harness writer, rubrics schema, or CLI.

## Risks

- **Editable-install shadowing:** if Basic_RAG has a local `ai_eval/` dir or
  an extra `sys.path` entry, it can shadow the editable install. The
  verification one-liner in step 3 catches this.
- **numpy yank warning:** `uv run` may print
  `warning: numpy==2.4.0 is yanked`. Harmless for this fix; only affects
  runtime if Basic_RAG's `evaluations` package imports numpy-heavy deps that
  fail to load (which previously caused the transient `__init__ missing config`
  errors on the first run). If that recurs, pin a non-yanked numpy in
  Basic_RAG's deps.
- **Heuristic false negatives:** an unannotated object-typed param with
  *other* unannotated params hits case 2 and binds the scalar to it, which
  could still crash on attribute access. This is an acceptable trade-off:
  unannotated code is ambiguous, and the alternative (strict str-only binding)
  breaks the existing single-arg fast-path contract. Document this in the
  function docstring.

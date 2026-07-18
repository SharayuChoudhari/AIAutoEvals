# Plan: Demote IO-coupled internal methods + add FastAPI entry-point detector

## Problem

`ai-evals run` against a FastAPI backend fails with `TypeError: <Class>.__init__() missing 1 required positional argument: 'session'` for detected internal methods like `DocumentVectorDAO.search_similar_vectors`, `SingleQueryEvaluator.evaluate_single`, `ChatMessageService.process_query`. Bootstrapping the backend does not help because `bootstrap` is capture-only and never touches `top_level`.

## Root cause (verified)

The `top_level: bool` flag on `TaskSpec` (default `True`, `ai_eval/config/schema.py:201`) gates auto-seed + run. It is set by only two weak mechanisms:

1. **`_is_private_entry`** (`ai_eval/inference/synthesize.py:106-117`) — catches only `_`-prefixed methods. This is why `ChatMessageService._create_workflow` is correctly skipped while the three failing public methods slip through.
2. **Layer 3 peer-reached demotion** (`ai_eval/inference/task_selection.py:275-286`) — fires only when another *detected* task calls the method. There is **no FastAPI detector**, so route handlers are never detected, so no call-graph edge exists to demote the DAO/Service/Evaluator methods.

Gaps confirmed across the flow:
- No `__init__`-signature inspection at synthesis time. The signature is only checked at runtime in `engine.py:_build_call_args` / `engine.py:244` (`instance = cls()`), by which point the task is already auto-seeded and being called.
- `classify_io_coupled` (`ai_eval/scaffold/harness_writer.py:123-172`) correctly identifies these tasks as IO-coupled, but `ai_eval/cli/init.py:327` binds its result to `_io_coupled_names` and never uses it.
- The SLM path `ai_eval/inference/slm/builder.py:278` (`_task_spec_from_slm`) sets `top_level=not _is_private_entry(entry)`, ignoring `task.top_level` and silently undoing any Layer 3 demotion in SLM/hybrid mode.
- `ai-evals bootstrap` (`ai_eval/cli/bootstrap.py:39-164`) writes only `golden_set.json`; it never re-runs `classify_io_coupled`, never edits `rubrics.yaml`, never flips `top_level`. So re-running `bootstrap` cannot heal stale rubrics.

## Decisions (confirmed with user)

1. **Detection mechanism**: `__init__` signature inspection at synthesis time. Demote `Class.method` to `top_level=False` when the class's `__init__` has required params (after `self`) that are not str-bindable. No class-name heuristic (false-positive prone). Also wire `classify_io_coupled` into `top_level` and fix the SLM path so Layer 3 demotion survives.
2. **Apply / migrate**: Detection-time only. Bootstrap stays capture-only (AGENTS.md §1 contract preserved). Users with stale rubrics re-run `ai-evals init` (which preserves user edits via its existing merge path) to regenerate `top_level` flags.
3. **Entry-point scope**: Add a FastAPI detector (`@app.get/post/put/delete`, `@router.get/post/...`) that emits route handlers as `top_level=True` entry points, built on a new `WebServiceDetector` base so Flask/Sanic/Tornado can be added later. Combined with the demotion fix, the DAO/Service/Evaluator methods the handler calls get demoted (Layer 3 peer-reach now has an edge because the handler is a detected task) and `run` has a real end-to-end entry point to score from the captured trace.

## Affected files (source)

### Demotion fix — `__init__` signature inspection

- `ai_eval/inference/signatures.py` — add `class_init_requires_nonstr_args(tree, class_name) -> bool`. Parse the class's `__init__` `args` node: return `True` if any positional-or-kw arg (after `self`) has no default AND an annotation that is not in a str-bindable allow-list (`str`, `Optional[str]`, `str | None`, `Any`, unannotated-with-default-`= ""`). Mirror the existing `find_callable_defs` top-level-class walk. Pure-AST, no import/execution. Export from `__all__`.
- `ai_eval/inference/synthesize.py` — in `build_rubrics` (around line 200-208), change the `top_level=` expression from `task.top_level and not _is_private_entry(task.entry)` to also consult signature inspection: `task.top_level and not _is_private_entry(task.entry) and not _requires_io_args(task, ...)`. The inspection needs the file AST; since `build_rubrics` receives only `ScanResult` (no parsed trees), add a thin helper that re-parses `task.file_path` under `project_root` when provided (mirrors how `task_selection.select_tasks` is already called with `project_root`). Cache parsed trees per file to avoid re-parsing across tasks in the same scan. Fallback to `True` (no demotion) when the file can't be parsed or `__init__` can't be found — preserves over-promotion-safe degradation.
- `ai_eval/inference/slm/builder.py` — fix `_task_spec_from_slm` (line 261-279) and `_fallback_task_spec` (line 141-158): both must accept the originating `DetectedTask` and propagate `task.top_level` (mirror Path A/B: `task.top_level and not _is_private_entry(task.entry)`), so Layer 3 demotion survives the SLM path. `_task_spec_from_slm` currently takes only `_SLMTask`; change its signature to also take `task: DetectedTask` and pass `task.top_level`. Update call sites at `builder.py:405` and `builder.py:506-508`.

### Wire `classify_io_coupled` into `top_level`

- `ai_eval/cli/init.py` (line 327-329) — stop discarding `io_coupled_names`. After `write_harnesses`, for each task name in the set that does **not** have a corresponding harness file written (true IO-coupled, no canned reads available), set `rubrics.tasks[name].top_level = False` before `rubrics_writer.write`. This is a second-pass safety net: the signature-inspection demotion is primary; this catches cases the signature heuristic misses (e.g. `__init__` with all defaults but runtime IO dependency). Note the harness file written set is `harness_written` (list of `(name, status)`); a task is "IO-coupled without harness" iff its name is in `io_coupled_names` but not in `{n for n, _ in harness_written}`.
- `ai_eval/scaffold/harness_writer.py` — `write_harnesses` already returns `(written, io_coupled_names)`; no change needed. Just stop binding the second element to `_`-prefixed.

### FastAPI detector

- `ai_eval/inference/detectors/web.py` — **new file**. `WebServiceDetector` base class (subclass of `Detector`) with a `_ROUTE_DECORATORS: tuple[str, ...]` class attribute and a `framework` attribute. `matches` returns `True` when the file imports the framework's app/router class (overridable). `extract` walks `defs` for `FunctionDef`/`AsyncFunctionDef` whose decorator list contains an `ast.Call`/`ast.Attribute` whose trailing name is in `_ROUTE_DECORATORS` (e.g. `get`, `post`, `put`, `delete`, `patch`). Emits one `DetectedTask` per route handler with `entry=<handler name>` (bare module-level function), `type="chat"` (the LLM-bearing default; a RAG route will be reclassified by `_looks_like_rag` in synthesize if its IO resembles retrieval), `inputs=["query"]`, `outputs=["response"]`, `evidence=[f"fastapi route {method} at {rel}:{lineno}"]`, `top_level=True` (the default — route handlers ARE the end-to-end entry points). Dedup by entry name within a file.
- `ai_eval/inference/detectors/fastapi.py` — **new file**. `FastAPIDetector(WebServiceDetector)` with `framework = "fastapi"`, `_ROUTE_DECORATORS = ("get", "post", "put", "delete", "patch")`, `matches` checking `has_import_prefix(imports, "fastapi")`. Thin subclass — most logic in the base.
- `ai_eval/inference/ast_scan.py` — add `FastAPIDetector` to `builtin_detectors()` (line 79-87). Place it early in the ordering (before `OpenAIChatDetector`) so route handlers are claimed as top-level entries; the internal DAO/Service methods they call are still detected by `PGVectorDetector`/`OpenAIChatDetector` and get demoted by Layer 3 + signature inspection.
- Call-graph resolution: `callgraph.py:_resolve_self_attr_call` already resolves `self.<attr>.<m>(...)` when `<attr>` is assigned in `__init__` to a repo-imported constructor. A FastAPI route that constructs `ChatMessageService(self.session)` in its body (not `__init__` — routes are functions, not classes) won't be resolved by the current `__init__`-only tracking. **Decision: do NOT extend callgraph scope in this plan.** The signature-inspection demotion is the safety net that catches the DAO/Service methods regardless of whether a call edge exists. Layer 3 will still fire for the class-based case (a service whose `__init__` constructs the DAO).

## Tests to add / update

- `tests/test_top_level.py` — extend:
  - `test_dao_init_requires_session_is_demoted`: a `class DocumentVectorDAO: def __init__(self, session): ...; def search_similar_vectors(self, q): ...` scanned via `build_rubrics` with `project_root` → `top_level is False`.
  - `test_dao_init_with_str_default_stays_top_level`: `def __init__(self, query: str = "")` → `top_level is True` (no false demotion).
  - `test_dao_init_no_args_stays_top_level`: `def __init__(self): ...` → `top_level is True`.
  - `test_evaluator_init_requires_config_is_demoted`: `SingleQueryEvaluator.__init__(self, config)` → demoted.
  - `test_slm_path_preserves_layer3_demotion`: a `DetectedTask` with `top_level=False` (simulating Layer 3) passed through `_task_spec_from_slm` → resulting `TaskSpec.top_level is False` (regression test for `builder.py:278`).
- `tests/test_detectors.py` (or new `tests/test_fastapi_detector.py`) — add:
  - `test_fastapi_detector_finds_post_route`: a `main.py` with `from fastapi import FastAPI; app = FastAPI(); @app.post("/chat")\nasync def chat(req): ...` → one `DetectedTask` with `entry="chat"`, `top_level=True`.
  - `test_fastapi_detector_finds_router_route`: `@router.get("/items")` shape.
  - `test_fastapi_detector_ignores_non_route`: a plain `def helper():` with no route decorator → no tasks.
  - `test_fastapi_detector_ignores_non_fastapi_file`: a file without `fastapi` import → `matches` returns `False`.
- `tests/test_task_selection.py` — add:
  - `test_fastapi_route_demotes_dao_it_calls`: when a detected FastAPI route handler calls `self.dao.search(...)` (class-based route) the DAO task is peer-reached and demoted. This exercises Layer 3 with a real FastAPI edge.
- `tests/test_init_wiring.py` (new or extend existing init tests) — add:
  - `test_io_coupled_without_harness_is_demoted`: a task in `io_coupled_names` with no harness file written gets `top_level=False` after `init` runs. Use a fake `write_harnesses` returning `(empty, {"some_task"})`.
- `tests/test_bootstrap.py` (if exists) — add a regression test confirming `bootstrap` does **not** modify `rubrics.yaml`'s `top_level` (locks the AGENTS.md §1 contract).

## Validation plan (mandatory gates per AGENTS.md)

```bash
uv run ruff format ai_eval tests
uv run ruff check ai_eval tests
uv run mypy ai_eval
uv run pytest -q
```

All four must pass. Specifically run:
- `uv run pytest -q tests/test_top_level.py tests/test_detectors.py tests/test_task_selection.py` — demotion + detector coverage.
- `uv run pytest -q tests/test_synthesize_workflow_metrics.py` — ensure the synthesize change didn't break existing metric assignment.

## Risks / edge cases

- **Over-demotion**: a legitimately-runnable top-level class method whose `__init__` takes a non-str arg (e.g. a LangGraph workflow `def __init__(self, graph: CompiledStateGraph)`) would be demoted. **Mitigation**: the str-bindable allow-list should also pass through types in `_GRAPH_TYPES` / `_LLM_CLIENT_TYPES` (reuse from `harness_writer.py`) — these are the types the runner/harness already know how to construct. If unsure, fall back to `True` (over-promotion-safe, per `callgraph.py` R2 contract). Document the allow-list in the `class_init_requires_nonstr_args` docstring.
- **force_task escape hatch**: a user who explicitly wants to run an IO-coupled internal method can already use `force_task: true` in hints (immune to demotion per `task_selection.py:237-239`). The signature-inspection demotion must also respect this — but since it runs in `build_rubrics` (after `select_tasks` has already applied force_task immunity by keeping the task in `final_kept`), and force_task tasks already have `top_level=True` from the detector, the demotion must check `force_task_keys` before demoting. `build_rubrics` receives `force_task_keys` (line 171) — pass it through to the demotion check.
- **FastAPI dependency-injected routes**: FastAPI routes typically receive their service via DI (`def chat(req, svc: ChatMessageService = Depends(...)`), not `self.dao`. The signature-inspection demotion still catches the service/DAO methods directly (they require `session`/`config`), so `run` skips them with the bootstrap notice and the FastAPI route handler is the scored entry point. This is the intended AGENTS.md §1 flow: the route runs once (during `bootstrap`), its internal calls are scored from `example["trace"]["calls"]` via `node_metrics`. The FastAPI detector gives `run` a real top-level entry to score; users author `node_metrics` after a first `bootstrap` reveals real `call.kind`/`call.name` values.
- **Bootstrap wrappers gap**: `ai_eval/bootstrap/wrappers.py:131-138` (`_PATCHERS`) has no pgvector/SQLAlchemy patcher, so `DocumentVectorDAO` calls during bootstrap won't be captured into `example["trace"]["calls"]` as `kind=retrieve`. This is a **separate, pre-existing gap** — out of scope for this plan (the user's immediate blocker is the `TypeError` on `run`, not missing trace capture). Flag in the plan's "Out of scope" section.

## Out of scope

- Adding pgvector/SQLAlchemy/asyncpg capture patchers to `bootstrap/wrappers.py` (separate gap; the `TypeError` blocker is fixed by the demotion + FastAPI detector).
- Extending `callgraph.py` beyond `__init__`-only attribute tracking (R2+ scope).
- Flask/Sanic/Tornado/Django detectors (the `WebServiceDetector` base makes these additive later).
- Runtime self-heal in `engine.py` (the `TypeError` path at `engine.py:507-518` stays as the last-resort diagnostic; with the demotion fix it should rarely fire).
- Editing `bootstrap` to heal rubrics (preserves AGENTS.md §1 contract; users re-run `init`).

## Ordered task list

1. Add `class_init_requires_nonstr_args(tree, class_name) -> bool` to `ai_eval/inference/signatures.py`. Reuse `_LLM_CLIENT_TYPES`/`_GRAPH_TYPES` from `harness_writer.py` (or move them to a shared `ai_eval/inference/types.py` constant module to avoid a circular import — `signatures.py` is in `inference/`, `harness_writer.py` is in `scaffold/`; importing scaffold from inference is wrong direction). Add to `__all__`.
2. In `ai_eval/inference/synthesize.py:build_rubrics`, apply signature-inspection demotion. Add a small per-file AST cache (dict keyed by `file_path`) inside `build_rubrics` so multiple tasks in the same file don't re-parse. Respect `force_task_keys` (skip demotion for forced tasks). Keep the `_is_private_entry` check (it's a cheap short-circuit).
3. Fix `ai_eval/inference/slm/builder.py`: `_task_spec_from_slm` and `_fallback_task_spec` take `task: DetectedTask` and use `task.top_level and not _is_private_entry(task.entry)`. Update call sites at lines 405 and 506-508. (The signature-inspection demotion from step 2 runs in `build_rubrics`, which is the rules path; the SLM path relies on `task.top_level` being correctly set by `select_tasks` Layer 3 + the detector. The SLM path does NOT re-run signature inspection — that's acceptable because `select_tasks` already ran and set `task.top_level`. The fix here is only to stop `_task_spec_from_slm` from clobbering `task.top_level` with `not _is_private_entry(entry)`.)
4. In `ai_eval/cli/init.py:327`, stop discarding `io_coupled_names`. After `write_harnesses`, demote tasks in `io_coupled_names` that have no harness file to `top_level=False` before writing rubrics.
5. Create `ai_eval/inference/detectors/web.py` with `WebServiceDetector` base.
6. Create `ai_eval/inference/detectors/fastapi.py` with `FastAPIDetector(WebServiceDetector)`.
7. Register `FastAPIDetector()` in `ai_eval/inference/ast_scan.py:builtin_detectors()` (early in the list, before `OpenAIChatDetector`).
8. Add tests (see "Tests to add / update").
9. Run the four mandatory gates. Fix any failures. Do not declare done with a known failing gate.

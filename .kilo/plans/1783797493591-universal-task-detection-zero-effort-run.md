# Plan: Universal task detection & zero-effort run

## Context

`ai-evals init` currently mis-promotes internal nodes (DAOs, shared helpers) and
judge harnesses (RAGAS-style evaluators) as tasks, and cannot resolve class-method
entry points. Result: `ai-evals bootstrap` captures nothing (it wraps a non-existent
`main`) and `ai-evals run` shows 0 examples with misleading `fail` rows.

Verified root causes (both universal, not codebase-specific):

1. **No task-boundary selection.** Detectors emit a task per framework-call site
   with no cross-file reachability view. Internal nodes (`search_similar_vectors`,
   reached only via `process_query`) and judge harnesses (`SingleQueryEvaluator`,
   an LLM-as-judge loop) get promoted as tasks.
2. **Entry resolution is top-level-only.** `find_callable_defs` /
   `enclosing_def_name` (`ai_eval/inference/signatures.py`) only walk top-level
   functions, never `ClassDef` bodies. Any task whose entry is a class method
   resolves to `entry=None` → defaults to `main` → 0 captures. `_wrap_entry`
   (`ai_eval/bootstrap/sitecustomize.py`) compounds this: it does
   `getattr(mod, entry)` + `setattr(mod, entry, ...)` and can't wrap/rebind a
   method or handle `async`.

Basic_RAG (`/Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG`) is a verification
case, not the specification. Every rule below is keyed off structural signals, not
Basic_RAG patterns.

## Goal

From `ai-evals init` to a non-empty `ai-evals run` with zero manual
rubrics/harness/fixture authoring, for any Python repo using a detected AI
framework — correct task set, resolvable entries, runnable bodies, and an
honestly-labeled golden set.

## Decisions (all universal)

### D1 — Task selection = P2 reachability roots + judge-exclusion

Build a cross-file call graph among detected framework-call sites (see R2). A site
is a **task** iff no other detected site reaches it transitively (it is a root).
Reached sites are demoted to internal nodes (not emitted as tasks).

**Judge-exclusion** (second layer): exclude a detected site iff
`J1 ∨ (J2 ∧ J3) ∨ J5`:
- **J1** — enclosing module imports a known judge framework (`ragas`, `deepeval`,
  `trulens`, `langchain.evaluation`, `openai.evals`, `promptfoo`, `inspect_ai`).
- **J2** — file lives under `evaluations/`, `eval/`, `tests/`, `benchmarks/`,
  `metrics/`.
- **J3** — enclosing callable returns a score-shaped object (a `float`/`int` field
  named `score`/`rating`/`faithfulness`/`precision`, or a class whose name
  contains `Evaluation`/`Metric`). Bare `Result` suffix is NOT sufficient (would
  false-positive on `SearchResult`).
- **J5** — site is inside `ai-evals`' own generated `eval/tests.py`.

J4 (RAGAS `question`+`answer`+`contexts` input signature) adds confidence but is
not gating. Policy is disjunctive-broad (low false-negative) because silently
dropping a real task is the worst failure for a zero-config tool. Overridable via
`ai-evals.hints.yaml`: `judge_code: [paths]` to force-exclude;
`force_task: true` to force-include.

Route/CLI-boundary detection (P3) is explicitly **out of scope** (later refinement
for better naming only, not required for task-set correctness).

### R2 — Call-graph resolution strategy

Edges between **detected framework-call sites only** (not arbitrary functions).
Resolve `self.<attr>.<m>(...)` by: tracking `<attr>` to its `__init__` assignment
(`SomeClass(...)`), resolving `SomeClass`'s import to a file, checking whether `<m>`
(or its body) contains a detected site. Same for module-level `fn(...)` calls via
imports. Purely AST-local, no execution.

Degrades **safely to over-promotion** when `self.<attr>` isn't set in `__init__`
(lazy init, properties, dynamic dispatch `getattr(self, name)()`): no edge drawn →
both sites become roots → never wrongly demotes a real task. `__init__`-only attr
tracking is the agreed scope (R2, not R2+).

Edge case soundness: if a public surface delegates to a helper that makes the LLM
call (the surface itself has no detected framework call), the surface is not a
root and the helper alone is promoted — correct under P2 (the helper is the true
task surface).

### D2 — Entry format = dotted `Class.method`

`find_callable_defs` descends into `ClassDef` bodies (collects methods).
`enclosing_def_name` returns `Class.method` when the enclosing def is a method,
bare name for module-level functions. All detectors emit the qualified name.

### D3 — Bootstrap wrapping resolves dotted + async entries

`_wrap_entry` (`sitecustomize.py`) resolves `Class.method` via
`getattr(getattr(mod, cls), method)`, wraps the function (sync or async — detect
via `inspect.iscoroutinefunction`), and rebinds on the **class** (not the module)
so instances pick up the wrapper.

### D4 — IO-coupled classification

A task is **IO-coupled** (→ harness codegen fires) iff its resolved method body
contains a read call through `self.<attr>` where `<attr>` is assigned in
`__init__` to something that is NOT a known LLM-client type
(`OpenAI`/`Anthropic`/`ChatOpenAI`/`Ollama`/`HuggingFaceHub`/…) and NOT a known
framework graph object (compiled LangGraph/LangChain chain). LLM clients and
compiled graphs run real (need a key, not a stub); DAOs/session/files/HTTP-clients/
queues get stubbed. Reuses R2's `__init__` attr tracking — no new analysis pass.

- `process_query` (reads `self.chat_dao`/`self.company_dao`/`self.document_vector_dao`/
  `self.session`, LLM is inside the compiled graph) → IO-coupled ✓
- Module-level `summarize(text)` calling `OpenAI().chat...` inline → pure-LLM ✓
  (no `self.<attr>` read, client is real)

### D5 — Stub-harness codegen

When `init`/`analyze` detects an IO-coupled task, AST-walk the resolved callable
body, collect every `self.<init-assigned-attr>.<m>(...)` read call, and emit
`eval/_harness_<task>.py` that monkey-patches each such class's methods to return
schema-shaped canned objects (built from the ORM/Pydantic models the call sites
reference) and no-ops session mutators. The rule is "stub every
`self.<init-assigned-attr>.<m>(...)` in the body," generalizing to any IO shape
(HTTP clients, file handles, queue producers), not just DB DAOs.

**Lifecycle (region-split):**
- Region 1 `# AUTO-GENERATED DO NOT EDIT` — wiring (class/method resolution,
  monkey-patch setup, no-op session). Regenerated wholesale on `init`/`analyze`,
  but only if the task method body's AST hash changed since last generation (hash
  stored in a sidecar comment).
- Region 2 `# FIXTURE DATA — edit freely` — canned return objects. Written only
  on first generation; never overwritten thereafter.
- On load, the harness best-effort-validates that each patched method's return
  shape still matches the call-site AST and **warns** (not fails) on drift.

Mirrors `golden_set.json`'s "preserve captured/user data" contract.

### D6 — Golden-set source = hybrid

- **Pure-LLM tasks:** `init` auto-seeds 3–5 shape-varied inputs (empty, short,
  long, unicode, boundary) into `golden_set.json`. `run` produces meaningful
  scores immediately — zero authoring, zero environment. (LLM runs on real
  inputs; faithfulness/hallucination are scorable.)
- **IO-coupled tasks:** `init` generates the stub harness (D5) AND auto-seeds 1
  shape-valid example flagged `seed: auto` so the pipeline is **green**
  (non-error, non-empty). The run prints a one-line notice: "IO-coupled task
  `<name>` scored against auto-seeded fixtures; run `ai-evals bootstrap --
  <runtime>` for trustworthy regression baselines."
- `ai-evals bootstrap -- <runtime>` remains the **trustworthy** path for both
  task shapes; it replaces auto-seeded examples with captured traces (deduped,
  per the existing writer contract). Auto-seed is the zero-effort green-pipeline
  path; bootstrap is the trustworthy-regression path.

## Ordered tasks

1. **Class-body entry resolution** — `signatures.py`: `find_callable_defs`
   descends into `ClassDef` bodies; `enclosing_def_name` returns `Class.method`.
   Update all detectors to emit dotted entries. Existing detector tests must
   still pass (additive change).

2. **Cross-file call-graph builder (R2)** — new module (e.g.
   `ai_eval/inference/callgraph.py`): resolve `self.<attr>.<m>()` and module
   `fn()` calls across files via `__init__` attr tracking + import resolution.
   Edges between detected framework-call sites only.

3. **Task selection (P2 + judge-exclusion)** — in
   `ai_eval/inference/synthesize.py`: run the call-graph builder over detected
   sites, demote reached sites, apply `J1 ∨ (J2 ∧ J3) ∨ J5` exclusion. Add
   `judge_code` / `force_task` to the `ai-evals.hints.yaml` schema.

4. **Bootstrap wrapper for dotted + async entries** —
   `ai_eval/bootstrap/sitecustomize.py` `_wrap_entry`: dotted resolution,
   `iscoroutinefunction` detection, rebind on class.

5. **IO-coupled classification + harness codegen (D4/D5)** — classify via
   `self.<attr>` read of non-LLM-client/non-graph type (reuses R2 attr tracking);
   AST-walk body; emit `eval/_harness_<task>.py` (region-split, AST-hash-gated,
   drift-warn). New module e.g. `ai_eval/scaffold/harness_writer.py`.

6. **Hybrid golden-set seeding (D6)** — extend
   `ai_eval/scaffold/golden_writer.py` / a new seeder: auto-seed pure-LLM tasks
   with shape variants; IO-coupled tasks get harness + 1 auto-seed flagged
   `seed: auto`. Extend the run renderer (built on the `SKIP`/`seed` glyph work
   already in `ai_eval/cli/render/tables.py`) to show the `seed: auto` notice.

7. **Run-engine integration** — `ai_eval/runner/engine.py` `_import_entry` /
   `_call_entry`: resolve dotted `Class.method`, construct instance with auto-faked
   constructor args (primitives, `None`-defaulted) for pure-LLM; load the harness
   for IO-coupled; `status: skip (io-coupled, harness missing)` if harness absent.

8. **Tests** — synthetic multi-file fixtures per edge case: shared helper called
   by 2 endpoints (helper demoted, both endpoints promoted); framework-object
   entry (LlamaIndex `.aquery()` — root, nothing reaches it); judge harness
   (J2∧J3 excluded); async method (D3 wraps); IO-coupled classification (D4);
   orphan-no-root repo (clean exit, "no public AI task surface" message). Plus a
   Basic_RAG-shape integration test asserting: DAO demoted, endpoint is sole
   task, dotted entry resolvable, harness generated, run produces non-empty
   `seed: auto` result.

## Failure modes (addressed)

- Unseen framework whose call site is a framework object method (LlamaIndex
  `.aquery()`) → P2 treats as root (nothing reaches it) → task.
- Shared retrieval helper called by 2 endpoints → helper demoted, both endpoints
  promoted.
- Eval/judge harness (RAGAS, custom scorer) → judge-exclusion drops it.
- Task whose entry is an async method → D3 wraps async correctly.
- IO-coupled task with no harness generated yet → D4 skips cleanly, no confusing
  `main` traceback.
- Repo with zero detectable roots (pure library) → `init` reports "no public AI
  task surface found; consider hints" and exits cleanly.
- Dynamic dispatch / lazy init → R2 degrades to over-promotion (safe).
- Unconventional judge code missed by J2∧J3 → `judge_code` hints escape hatch.
- Harness drift between regenerated wiring and preserved fixtures → load-time
  warn (not fail).

## Out of scope (explicit)

- Route/CLI-boundary detection (P3) — later refinement for naming only.
- Semantically-meaningful auto-fixtures for IO-coupled tasks — logically
  impossible without domain knowledge; D6 is honest about this (`seed: auto`
  flag + bootstrap notice).
- Graphify integration — the AST is in-hand; the cross-file call graph is built
  natively, not via an external dependency.
- Fully-autonomous init→run with no `bootstrap` — the runtime command remains the
  agreed manual step for trustworthy regression baselines.

## Validation

- Existing detector + threshold tests must pass (class-descent is additive).
- New call-graph tests cover each P2 edge case above.
- Basic_RAG-shape integration test (fixtures, not the real repo) asserts the
  motivating case: DAO demoted, endpoint sole task, dotted entry resolvable,
  harness generated, non-empty `seed: auto` run.

## Implementation-detail items (left to implementer, not architectural)

- Exact auto-seed variant set (empty/short/long/unicode/boundary is a default).
- `hints.yaml` schema additions for `judge_code` / `force_task`.
- `seed: auto` flag propagation into the renderer (extends the `SKIP` glyph
  already added in `ai_eval/cli/render/tables.py` / `theme.py`).
- Known-LLM-client-type allowlist contents (`OpenAI`, `ChatOpenAI`, `Anthropic`,
  `ChatAnthropic`, `Ollama`, `HuggingFaceHub`, `HuggingFacePipeline`,
  `ChatGoogleGenerativeAI`, compiled `StateGraph`/`RunnableSequence`).
- Harness drift-detection's exact validation depth.

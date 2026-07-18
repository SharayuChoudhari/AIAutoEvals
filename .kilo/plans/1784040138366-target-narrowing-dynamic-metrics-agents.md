# Plan: Narrow auto-seed target + dynamic metric registry split + AGENTS.md

## Goal

Three workstreams addressing the user's three concerns, superseding the
prior `1784035702649-auto-seed-call-failures-fix.md` Problem B heuristic:

1. **AGENTS.md** — project-wide rules every coding agent must follow.
2. **Narrow auto-seed/run target** to public top-level entries (the
   "complete job"), instead of every detected method. Replaces the
   `_build_call_args` scalar-binding heuristic — object-typed-first-param
   internal methods are no longer auto-seeded, so the heuristic is moot.
3. **Metric registry split** — minimal high-frequency built-ins +
   project-local `eval/metrics.yaml` for niche/custom metrics. Remove
   niche metrics from built-in code.

The Problem A stale-install fix (editable install into Basic_RAG) is
still required and unchanged from the prior plan; it is listed under
rollout but is not a source-code change.

---

## Workstream 1 — AGENTS.md

Create `/Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals/AGENTS.md`
at repo root. Contents (sections):

### Stack & tooling
- Python >=3.10, `uv` for env/runner, `hatchling` build backend.
- Lint: `uv run ruff check .` and `uv run ruff format .` (line-length 100,
  py310 target; ignores in `pyproject.toml`).
- Typecheck: `uv run mypy ai_eval`.
- Tests: `uv run pytest -q` (testpaths `tests`, timeout 60s).
- An agent making code changes MUST run ruff + mypy + pytest before
  declaring a task done.

### Code conventions
- No comments unless explicitly requested. Docstrings on public
  functions/classes/modules (module docstrings already pervasive — match
  the existing style).
- Frozen dataclasses for value objects (`@dataclass(frozen=True)`).
  Pydantic `BaseModel` with `ConfigDict(extra="forbid")` for schema
  types — adding a field is a schema change requiring a test update.
- `from __future__ import annotations` at the top of every module.
- Match existing import grouping (stdlib / third-party / `ai_eval`).

### Architectural contracts (do not violate)
- **Target the complete job.** `ai-evals run` auto-seeds/runs only
  public top-level entries (module-level functions or public methods on
  the detector-identified LLM/agent/RAG class). Internal DAOs, services,
  private (`_`-prefixed) methods, and IO-coupled tasks without a harness
  are scaffolded into `rubrics.yaml` but skipped by auto-seed/run with a
  notice directing the user to `ai-evals bootstrap`. Do NOT add
  arg-binding heuristics to make internal methods callable from a scalar
  auto-seed — that path is closed by design.
- **Metric registry split.** Built-in metrics are the high-frequency set
  only (see Workstream 3 list). Niche/project-specific metrics live in
  the consuming project's `eval/metrics.yaml`, loaded and merged at
  runtime. Do not add niche metrics back into
  `ai_eval/metrics/registry.py` or `judge_builtin.py`.
- **Seeder/harness/judge/registry boundaries.** Seeder produces
  shape-varied inputs for pure-LLM tasks; harness monkey-patches
  `self.<dao>.<method>()` reads for IO-coupled tasks; judge gateway is
  transport-injected (`complete_fn`); registry is the single validation
  surface consulted by `MetricSpec` and `assert_metric_implemented`.
- **One-release deprecation window** for unknown metric names
  (`warn_unknown`); `AI_EVAL_STRICT_METRICS=1` enforces hard rejection.

### Workflow
- Branch from `main`; commit messages match repo style (imperative,
  short subject).
- Never commit secrets. Never force-push. Only commit when asked.
- Editable-install verification step (see Rollout) when testing against
  a real downstream repo like Basic_RAG.

---

## Workstream 2 — Narrow auto-seed/run target

### Schema change
In `ai_eval/config/schema.py`, add to `TaskSpec`:
```python
top_level: bool = True
```
Default `True` preserves backward compat for hand-written rubrics.yaml.
`extra="forbid"` is already set, so this is an additive, opt-in field.

### Detector marker
In `ai_eval/inference/ast_scan.py`, `DetectedTask` gains the same
`top_level: bool` field. Each detector's `extract(...)` sets it:
- `True` for the public LLM/agent/RAG entrypoint — a module-level
  function, OR a public (non-`_`-prefixed) method on the class the
  detector identifies as the LLM entrypoint (the class that owns the
  `litellm.completion` / `ChatOpenAI(...).invoke` / agent `.astream`
  call the detector matched on).
- `False` for internal classes/methods: DAOs
  (`DocumentVectorDAO.search_similar_vectors`), evaluators
  (`SingleQueryEvaluator.evaluate_single`), private methods
  (`_create_workflow`), and service methods whose first non-self param
  is object-typed AND the class is not the detector's LLM entrypoint.

The scaffold writer (`ai_eval/scaffold/`) propagates `DetectedTask.top_level`
→ `TaskSpec.top_level` when emitting `rubrics.yaml`.

### Seeder change
In `ai_eval/scaffold/seeder.py::seed_golden_set`, skip auto-seeding
tasks where `rubrics.tasks[name].top_level is False`. Those task keys
remain in `golden_set.json` as empty lists (or omitted) and the run
prints one consolidated notice:
```
skipped auto-seed for N internal task(s); run `ai-evals bootstrap` for
trustworthy baselines: <names>
```

### Runner change
In `ai_eval/runner/engine.py::execute`, for each task with
`tspec.top_level is False`: do not import/call the entry; append a
single `ExampleRecord(status="skip")` per the existing skip shape, and
record a task-level notice. No `_build_call_args` changes are needed —
the internal methods are never called.

### Remove the now-moot heuristic
The prior plan's Problem B (`_build_call_args` scalar_idx fallback)
is **not applied**. `_build_call_args` stays as-is (it still handles
the legitimate pure-LLM multi-arg case like `evaluate_single` when such
a task is genuinely top-level). No new object-typed test from the prior
plan is added.

### Tests
- `test_seeder_skips_non_top_level_tasks`: rubrics with one
  `top_level=True` and one `top_level=False` task → only the top-level
  one gets auto-seeds in `golden_set.json`.
- `test_runner_skips_non_top_level_task`: a `top_level=False` task
  produces `status="skip"` examples and does not import the entry
  module (assert via a sentinel import side-effect).
- `test_top_level_default_true`: a hand-written rubrics.yaml without the
  field loads with `top_level=True` for all tasks.

---

## Workstream 3 — Metric registry split

### Trim built-ins
In `ai_eval/metrics/registry.py::BUILTIN_METRICS`, keep only:
- `hallucination_rate`
- `context_precision`
- `faithfulness`
- `argument_accuracy`
- `task_completion`

Remove: `scoring_accuracy`, `extraction_field_accuracy`,
`classification_f1`, `summary_faithfulness`, `translation_bleu`,
`slot_filling_accuracy`.

In `ai_eval/metrics/judge_builtin.py::BUILTIN_JUDGE_METRICS`, remove
the `_slot_filling_accuracy` builder and its `JudgeMetric` entry. Keep
`argument_accuracy`, `hallucination_rate`, `context_precision`,
`faithfulness`, `task_completion` (latency pair stays in
`registry.py::BUILTIN_JUDGE_METRICS_TUPLE`).

### Project-local metrics loader
Add `ai_eval/metrics/local.py` with:
```python
def load_local_metrics(project_root: Path) -> tuple[Metric, ...]: ...
def load_local_judge_metrics(project_root: Path) -> tuple[JudgeMetric, ...]: ...
```
Reads `eval/metrics.yaml` (schema):
```yaml
metrics:
  - name: slot_filling_accuracy
    description: Accuracy of slots/arguments filled for booking tasks.
    applicable_task_types: [booking, workflow]
    scored_dimension: slot_filling_accuracy
    prompt_template: |      # Jinja2-free; a literal prompt with {{input}}, {{output}}, {{expected}} placeholders
      Evaluate slot filling...
  - name: classification_f1
    description: Macro-F1 across classification labels.
    non_judge: false
```
The loader builds a `JudgeMetric` per entry, wrapping `prompt_template`
into a `prompt_builder` closure that substitutes placeholders into a
chat message list (mirroring `build_cot` output shape). Merge order in
`load_metrics` / `load_judge_metrics`:
1. built-ins
2. project-local (`eval/metrics.yaml`) — wins on name collision with a
   built-in? No: built-ins win; local can only ADD names. Document.
3. entry-point plugins (existing dedup by `seen` set).

`load_metrics`/`load_judge_metrics` need a `project_root` param.
Thread it through callers: `is_registered`, `get`, `get_judge_metric`,
`assert_metric_implemented`, and the engine's per-task
`metric_impls` construction in `execute`. The schema validator in
`config/schema.py` currently calls `is_registered` without a root —
make `project_root` optional there (validation stays lenient via
`warn_unknown`), and the run-time `assert_metric_implemented` gets the
real root.

### Documentation
- `docs/metrics.md` (new): the built-in set, the local-registry schema,
  and a copy-paste template re-declaring the six removed niche metrics
  (with their prior prompt-builder intent described as a template).
- `AGENTS.md` (Workstream 1) cross-references this.

### Tests
- `test_local_metrics_yaml_loads`: an `eval/metrics.yaml` with
  `slot_filling_accuracy` → `get_judge_metric("slot_filling_accuracy")`
  returns it; `all_names()` includes it.
- `test_builtin_niche_metrics_removed`: `is_registered("translation_bleu")`
  is `False` with no local file.
- `test_local_metric_runs_in_engine`: a rubrics task using a
  local-only metric produces a non-`error` metric result via a fake
  `complete_fn`.
- Update existing tests that referenced removed built-ins to use the
  local-file path or switch to a retained built-in.

---

## Rollout order (implementation session)

1. Write `AGENTS.md` (Workstream 1). No code, no test impact.
2. Workstream 3 registry trim + local loader + docs. Run:
   `uv run pytest -q && uv run ruff check . && uv run ruff format . && uv run mypy ai_eval`
3. Workstream 2 schema field + detector marker + seeder/runner skip +
   tests. Same lint/type/test gate.
4. Problem A (unchanged from prior plan): editable-install into
   Basic_RAG and re-run:
   ```bash
   cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
   uv pip install -e /Users/sharayuchoudhari/Dev/Projects/AIAutoEvals/AIAutoEvals --reinstall-package dot-ai-evals
   uv run python -c "import ai_eval.runner.engine as e; print(e.__file__, hasattr(e,'_build_call_args'))"
   uv run ai-evals run
   ```
   Expected: internal tasks (`process_query`, `search_similar_vectors`)
   are skipped with the bootstrap notice; top-level pure-LLM tasks run
   without `TypeError`/`AttributeError`.

---

## Boundaries / out of scope
- Auto-seeds remain a green-pipeline sanity check, not meaningful
  scores. Real baselines need `ai-evals bootstrap`.
- No changes to the harness writer, rubrics schema beyond the additive
  `top_level` field, or CLI command surface.
- No Jinja2 dependency — `prompt_template` uses simple `{{name}}`
  substitution implemented in-house.
- The `ai_eval.metrics` entry-point plugin path is unchanged; local
  registry is an additional layer, not a replacement.

## Risks
- **Detector top-level classification ambiguity:** a class that both
  holds the LLM call AND has internal helper methods — the helper
  methods would be `top_level=True` by the "same class as LLM
  entrypoint" rule. Mitigate: also require the method be public
  (non-`_`-prefixed) AND not classified IO-coupled by D4. Document the
  rule in `AGENTS.md` and the detector docstring.
- **Removing niche built-ins breaks existing downstream rubrics.yaml**
  that reference them without a local file. Mitigate: `warn_unknown`
  one-release window + the `docs/metrics.md` copy-paste template + a
  console hint in `assert_metric_implemented`'s error pointing to the
  template.
- **`project_root` threading** touches several modules; risk of a
  missed call site. Mitigate: make the param optional with
  `project_root: Path | None = None` everywhere, and fall back to
  built-ins-only when `None` (preserves the schema-validator path).

# AGENTS.md

This file governs how AI coding agents work in this repository. Every agent
operating on this codebase MUST read and follow these rules. Violating them
breaks invariants the runtime depends on.

## Stack & tooling

- **Language:** Python >=3.10. Use `from __future__ import annotations` at the
  top of every module.
- **Env/runner:** `uv` for virtualenv management and script execution
  (`uv run ...`). Never call bare `python` or `pip` directly.
- **Build backend:** `hatchling` (see `pyproject.toml`).
- **Lint:** `uv run ruff check .` (line-length 100, target py310; ignored
  rules documented in `pyproject.toml` under `[tool.ruff.lint]`).
- **Format:** `uv run ruff format .`
- **Typecheck:** `uv run mypy ai_eval`
- **Tests:** `uv run pytest -q` (testpaths `tests`, per-test timeout 60s).

### Mandatory verification before declaring a task done

An agent that changes any file under `ai_eval/` or `tests/` MUST run, in order:

```bash
uv run ruff format ai_eval tests
uv run ruff check ai_eval tests
uv run mypy ai_eval
uv run pytest -q
```

All four must pass. If a command fails, fix the issue and re-run — do not
declare the task done with a known failing gate.

## Code conventions

- **No comments** unless the user explicitly requests them. Module, class, and
  public-function docstrings are expected (match the existing verbose style —
  they explain *why*, not just *what*).
- **Value objects:** `@dataclass(frozen=True)` (see `ai_eval/metrics/registry.py`).
- **Schema types:** Pydantic `BaseModel` with `ConfigDict(extra="forbid")`.
  Adding a field to a schema model is a schema change — add or update the
  corresponding test. New fields default to a backward-compatible value so
  existing on-disk files (`rubrics.yaml`, `golden_set.json`) still load.
- **Import grouping:** stdlib / third-party / `ai_eval.*`, separated by blank
  lines. Mirror the import order in the file you are editing.
- **Async-native runner:** the execution engine (`ai_eval/runner/engine.py`) is
  async. Sync work (module import, task call) is offloaded via
  `asyncio.to_thread` so the event loop is never blocked.

## Architectural contracts (do not violate)

### 1. Target the complete job

`ai-evals run` runs only the **end-to-end entry point** per use case (the
deepest reachable root) and scores every internal node it calls from the
captured trace. One entry point is run; its internal calls are scored from
`example["trace"]["calls"]`, not re-executed.

- A module-level function, OR
- A public (non-`_`-prefixed) method on the class the detector identified as
  the LLM/agent/RAG entrypoint.

Internal classes/methods — DAOs (`DocumentVectorDAO.search_similar_vectors`),
evaluators (`SingleQueryEvaluator.evaluate_single`), private methods
(`_create_workflow`), and peer-reached survivors (a root called by another
surviving root) — are scaffolded into `rubrics.yaml` (so users can opt into
them) but are **skipped by auto-seed and run** with a notice directing the
user to `ai-evals bootstrap` for trustworthy baselines.

**Node→metric binding** is declared via `node_metrics: list[NodeMetric]` on the
entry task. Each `NodeMetric` is `{node_selector, metric: MetricSpec}`. The
runner walks `example["trace"]["calls"]`, matches nodes by selector
(`kind=retrieve`, `name~=pgvector`, `name=exact`, `call_index=0`), and scores
each match with the bound metric. Per-node scores roll into
`ExampleRecord.node_scores` (keyed by synthetic `node_id` like `retrieve_0`);
the task-level aggregate is the weighted mean of node scores across examples.
Author `node_metrics` after a first `ai-evals bootstrap` reveals real
`call.kind` / `call.name` values — `init` writes `node_metrics: []`.

**`_Stub`/`_fake_call_args` are gone.** Dotted `Class.method` entries
construct with no args (or a harness). An IO-coupled entry point whose
`__init__` requires real args and has no harness fails with a clear
bootstrap-directed error (not a `RecursionError`/`AttributeError` from a
stub). The harness writer (D5) stays: it monkey-patches `self.<dao>.<method>()`
reads so the entry run is green without a live DB/HTTP backend.

**The auto-seed contract** now applies only to pure-LLM entry points (all
top-level tasks get the 5 shape-varied inputs). IO-coupled entry points are
no longer auto-seeded (the `_Stub`-driven single green-pipeline example is
removed): they require `ai-evals bootstrap` to capture a real trace before
`run` can score their nodes.

**Do NOT add arg-binding heuristics** to make internal methods callable from a
scalar auto-seed. That path is closed by design — the `_build_call_args`
scalar-binding logic exists only for legitimate top-level pure-LLM entries of
varying arity, and a required non-str param it can't bind raises a clear
`TypeError` directing to bootstrap.

The `top_level: bool` field on `TaskSpec` (default `True` for backward compat)
controls this. The detector/synthesize layer sets it (including the Layer 3
peer-reached demotion in `task_selection.py`); the seeder and runner honor it.

### 2. Metric registry split

Built-in metrics are the **high-frequency set only**:

- `hallucination_rate`
- `context_precision`
- `faithfulness`
- `argument_accuracy`
- `task_completion`
- `latency_p50`, `latency_p95` (non-judge, computed by the runner)

Niche/project-specific metrics (`slot_filling_accuracy`,
`translation_bleu`, `classification_f1`, `extraction_field_accuracy`,
`scoring_accuracy`, `summary_faithfulness`, and any custom metric) live in the
consuming project's `eval/metrics.yaml`, loaded and merged at runtime by
`ai_eval/metrics/local.py`.

**Do NOT add niche metrics back into** `ai_eval/metrics/registry.py` or
`ai_eval/metrics/judge_builtin.py`. If a metric is only useful for a specific
use case, it belongs in the project-local registry file. See
`docs/metrics.md` for the schema and a copy-paste template.

Merge order in `load_metrics` / `load_judge_metrics`:
1. Built-ins (always win on name collision).
2. Project-local `eval/metrics.yaml` (can only ADD names, never override
   built-ins).
3. Entry-point plugins (existing dedup by `seen` set).

### 3. Seeder / harness / judge / registry boundaries

- **Seeder** (`ai_eval/scaffold/seeder.py`): produces shape-varied inputs
  (`""`, `"hi"`, long, unicode, boundary) for pure-LLM top-level tasks. Skips
  non-top-level tasks entirely. IO-coupled entry points are no longer
  auto-seeded (the `_Stub` path is removed): they require
  `ai-evals bootstrap` to capture a real trace before `run` can score their
  nodes.
- **Harness** (`ai_eval/scaffold/harness_writer.py`): monkey-patches
  `self.<dao>.<method>()` reads for IO-coupled entry points so the entry run
  is green without a live DB/HTTP backend. Region-split: auto-generated wiring
  vs. user-editable fixture data. The harness is now for entry-point
  construction only (it supplies canned reads so a bare `cls()` works).
- **Judge gateway** (`ai_eval/judge/gateway.py`): transport-injected via
  `complete_fn`. Tests pass a fake; production uses LiteLLM. The engine never
  makes a direct network call.
- **Registry** (`ai_eval/metrics/registry.py`): the single validation surface
  consulted by `MetricSpec` (schema-load time, lenient — warns) and
  `assert_metric_implemented` (run time, strict — raises
  `MetricNotImplementedError`, exit 1). Node metrics are validated on the same
  surface so a typo'd `node_metrics` name fails fast at run start.

### 4. One-release deprecation window

Unknown metric names emit `DeprecationWarning` via `warn_unknown` rather than
being rejected outright. `AI_EVAL_STRICT_METRICS=1` enforces hard rejection
now. This lets existing `rubrics.yaml` files survive a registry trim while
`run` still fails fast on metrics it cannot compute.

## Workflow

- Branch from `main`. Commit messages are imperative, short subject
  (e.g. `trim niche metrics from built-in registry`).
- **Never commit secrets** or API keys. Never force-push. Only commit when the
  user explicitly asks.
- When testing against a downstream repo (e.g. Basic_RAG), verify the editable
  install is live before drawing conclusions:
  ```bash
  uv run python -c "import ai_eval.runner.engine as e; print(e.__file__)"
  ```
  The printed path must be the local source tree, not a site-packages wheel.
  If it prints a `.venv/.../site-packages/...` path, the editable install is
  stale — reinstall with `--reinstall-package dot-ai-evals`.

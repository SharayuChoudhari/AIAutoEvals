# Plan: Integrate Phases 2, 3, and 4 of `ai-eval`

Full implementation of the model-agnostic judge gateway (Phase 2), golden-set
bootstrapper (Phase 3), and runner + insights dashboard (Phase 4), replacing the
current stubs in `ai_eval/cli/{judge,bootstrap,run,diff,report,history}.py`.

Companion design docs:
- `.kilo/plans/New_idea.md` (vision, 4-phase roadmap)
- `.kilo/plans/ai-eval-cli-and-system-design.md` (CLI surface + system design)

All decisions below were confirmed with the user. The plan is ordered so each
task is independently shippable.

---

## Locked design decisions

1. **Scope:** full Phases 2, 3, and 4 in one plan (not a vertical slice).
2. **Judge gateway transport:** true **async** via `litellm.acompletion` +
   `instructor.from_litellm(litellm.acompletion)`. Tests inject a fake async
   `acomplete` callable (mirrors the existing SLM `complete_fn` injection pattern
   in `ai_eval/inference/slm/client.py`).
3. **Metric model:** a metric is `(name, judge_prompt_builder, scored_dimension)`
   — dynamic, tied to the detected task's behavior. Each judge metric owns its
   prompt builder receiving `(task_spec, example, output) -> messages`. One
   judge call per (example, metric), parallelized by semaphore.
4. **Unregistered metric at `run` time:** **hard-fail exit 1**. Reconciles with
   the schema validator in `config/schema.py` that only warns unless
   `AI_EVAL_STRICT_METRICS=1`: `init`/`analyze` stay lenient; `run` enforces
   strictly. Documented as a run-time gate in `runner/thresholds.py`.
5. **Task invocation in `run`:** re-import `file_path`, call `entry` symbol with
   each golden example's `input`. Import/call errors → task-level `errors` in the
   run record, not a process crash.
6. **No-baseline behavior:** fresh repo (no prior run / `--baseline none`) →
   `Δ` column shows `—`, no regressions computed, run saved as the new baseline.
   `--fail-on-regression` only triggers on thresholds breached, never deltas.
7. **Bootstrap capture granularity:** one golden example per **top-level task
   invocation**; nested LLM/retrieve calls fold into `trace.calls`. Correlation
   via a thread-local task stack in the wrapper.
8. **Wrapper install:** generated `sitecustomize.py` written to a temp dir
   prepended to `PYTHONPATH` of the child subprocess; cleaned up after the run.
9. **Git metadata:** best-effort `subprocess` git calls; `git: null` if non-repo
   or any command fails.
10. **`diff --explain`:** metric-level Δ table + regression-judge "why it failed"
    narrative, **one judge call per regressing (metric, example)**.

---

## Phase 2 — Model-agnostic gateway & extraction

### New modules

- `ai_eval/judge/__init__.py`
- `ai_eval/judge/schemas.py` — Pydantic:
  - `JudgeRequest{ task_name, task_type, metric, example, output, tier_hint }`
  - `JudgeResponse{ score: float, rationale: str, sub_scores: dict[str,float], raw: Any }`
  - `JudgeError{ model, message, retryable }`
- `ai_eval/judge/gateway.py`
  - `async def score(request, *, model, complete_fn=None, temperature=0.0) -> JudgeResponse`
  - Default `complete_fn` wraps `litellm.acompletion` via instructor; tests pass
    a fake async callable.
  - Retry-with-fallback: `judge.default` → `judge.fallback[0]` → … → `JudgeError`
    collected into the run record.
  - Delegates cache lookup/write to `judge.cache`.
- `ai_eval/judge/instructor_glue.py` — builds the async instructor client once;
  `get_async_client()` caches it. ImportError → `JudgeUnavailable` with fix hint
  (mirrors `inference/slm/client.py:44-49`).
- `ai_eval/judge/tiering.py` — `select_tier(model_id) -> "complex"|"basic"`
  using `COMPLEX_MODEL_HINTS` from `config/defaults.py:34`. Override via
  `judge.tiering.complex_models` in rubrics.yaml (extend `JudgeConfig` schema).
- `ai_eval/judge/prompts/__init__.py`
- `ai_eval/judge/prompts/cot_complex.py` — frontier-tier CoT prompt builder.
- `ai_eval/judge/prompts/checklist_basic.py` — binary checklist (Yes/No) for
  local/small models.
- `ai_eval/judge/cache.py` — `JudgeCache`:
  - key = `sha256(model_id || prompt_text || schema_id)`
  - read/write `.ai-eval/cache/judge/<sha>.json`; `--no-cache` skips both.
  - content-addressed, deterministic (temperature=0 default).

### Dynamic metric registry (`ai_eval/metrics/`)

Extend the existing `ai_eval/metrics/registry.py`:
- Add `JudgeMetric` dataclass: `name, description, applicable_task_types,
  prompt_builder: Callable[[TaskSpec, Example, Any], list[dict]], scored_dimension: str`.
- Built-ins ship `argument_accuracy`, `hallucination_rate`, `context_precision`,
  `faithfulness` (each with a prompt builder + scored dimension), plus the
  non-judge `latency_p50`, `latency_p95` (computed by runner, no judge).
- Custom metrics loaded from `eval/metrics/<name>.py` (eval-dir scan) AND the
  `ai_eval.metrics` entry-point group (existing loader reused).
- `is_registered` / `get` unchanged surface; new `get_judge_metric(name)`.
- **Run-time strict gate:** `runner/thresholds.py` calls `get_judge_metric` and
  raises `MetricNotImplementedError` (exit 1) if a metric in rubrics.yaml has no
  implementation — even when init-time validation only warned.

### CLI wiring

- `ai_eval/cli/judge.py` — replace `not_yet`:
  - `--list` — show configured `judge.default`, `regression_check`, `fallback`
    with reachability (best-effort ping).
  - `--ping <model>` — one-shot reachability probe.
  - `--prompt <text> --model <m>` — one-shot completion via gateway, print
    response; honors `--format json`.
- `ai_eval/cli/doctor.py` — add a `judge gateway reachable` check that pings
  `judge.default` (read-only, timeout-bounded; failure is `required=False` so it
  warns without forcing exit 1 unless all providers are down).
- `ai_eval/config/schema.py` — extend `JudgeConfig` with optional
  `tiering: {complex_models: list[str]} | None` (extra="forbid" preserved).
- `ai_eval/cli/app.py` — no registration changes (commands already registered).

### Phase 2 tests

- `tests/test_judge_gateway.py` — fake async `acomplete`, assert tier selection,
  cache hit/miss, fallback chain, error collection.
- `tests/test_judge_tiering.py` — model-id → tier mapping table.
- `tests/test_judge_cache.py` — content addressing, `--no-cache` bypass.
- `tests/test_judge_metrics.py` — each built-in metric's prompt builder output
  and `scored_dimension`.
- `tests/test_cli_judge.py` — `--list`, `--ping`, `--prompt` snapshot tests
  (human + json), fake transport.
- Update `tests/test_cli_misc.py` for the new doctor check.

---

## Phase 3 — Automated golden-set bootstrapper

### New modules

- `ai_eval/bootstrap/__init__.py`
- `ai_eval/bootstrap/wrappers.py`
  - `install_hooks(capture_sink, redact_fields)` — monkey-patches:
    - `openai.resources.chat.completions.Completions.create`
    - `langchain_core.language_models.BaseChatModel.invoke` (best-effort; absent
      → skip)
    - `chromadb.api.Collection.query` and `pinecone.data_index.Index.query` /
      `.similarity_search` (absent → skip)
  - Each patch records `{kind, name, args, result, latency_ms}` into the active
    task's trace via the thread-local stack in `tracer.py`.
  - Redaction: strip strings matching `(?i)(api[_-]?key|bearer|secret|token)` and
    any `OPENAI_API_KEY=…`-style values from `args`/`result`.
- `ai_eval/bootstrap/tracer.py`
  - Thread-local task stack: `enter_task(task_name, input)` /
    `exit_task(output)`. On exit, flush one golden example
    `{id, captured_at, input, expected?, trace:{calls:[...]}, tags}` to the
    capture sink.
  - `expected` is populated only if the task fn returns a dict with an
    `expected` key or the wrapped test asserted (best-effort; usually empty).
  - Dedup by `sha256(json.dumps(input, sort_keys=True))` per task.
- `ai_eval/bootstrap/golden_writer.py`
  - `append(path, examples, *, append=True, max_per_task)` — merge into
    `eval/golden_set.json` honoring `--append/--replace` and `--max`. Reuses
    `ai_eval.scaffold.golden_writer`'s safe-merge contract (no overwrite of real
    captures without `--replace`).
- `ai_eval/bootstrap/sitecustomize.py` — generated bootstrap shim written to a
  temp dir; imports `ai_eval.bootstrap.wrappers.install_hooks` and wires the
  capture sink (an append-only JSONL sink the parent reads after the child
  exits). The temp dir is prepended to `PYTHONPATH` in the child env so Python
  auto-imports `sitecustomize` at interpreter startup, before any user import.

### CLI wiring (`ai_eval/cli/bootstrap.py`)

Replace `not_yet`:
- Validate `-- <command>` (already done).
- Resolve paths; read rubrics to know task `file_path`/`entry`/`redact` per
  task.
- Generate temp `sitecustomize.py` + sink path; build env with `PYTHONPATH`
  prepend and sink-path env var.
- `subprocess.run(runtime_cmd, env=instrumented, cwd=opts.cwd)`.
- After child exits: read the sink JSONL, correlate to tasks, dedup, cap by
  `--max`, apply `--task` filter, `--append`/`--replace`, write via
  `golden_writer.append`.
- Exit `0` (captures saved) / `1` (child crashed — still flush captures) / `2`
  (usage). Live `[N/max] captured <task>` counter to stderr (rich progress,
  silent when piped).
- Honor `--no-instrument`? **Out of scope** (OTEL deferred); reject the flag
  with exit 2 + a "Phase 5" message.

### Phase 3 tests

- `tests/test_bootstrap_wrappers.py` — fake `openai`/`langchain`/`chromadb`
  modules installed in a clean sys.modules; assert capture shape, redaction,
  thread-local correlation.
- `tests/test_bootstrap_tracer.py` — nested calls fold into one example; dedup;
  `--max` cap.
- `tests/test_bootstrap_golden_writer.py` — append/replace semantics, safe-merge
  with existing captures (reuse `test_golden_writer.py` patterns).
- `tests/test_cli_bootstrap.py` — e2e: fixture repo with a tiny `pytest` that
  calls a patched openai fake; assert `eval/golden_set.json` populated.

---

## Phase 4 — Run execution & insights dashboard

### New modules

- `ai_eval/runner/__init__.py`
- `ai_eval/runner/engine.py`
  - `async def execute(rubrics, golden_set, *, gateway, parallel, cache_on,
    task_filter, judge_override, progress) -> RunRecord`
  - For each task (filtered), for each golden example:
    1. Import `file_path` (insert `cwd`/`file_path.parent` on `sys.path`),
       call `entry(**example.input)` — measure latency. Import/call errors →
       `task.errors`, example marked `error` (not crash).
    2. For each metric in the task: if non-judge (`latency_*`) compute locally;
       else `await gateway.score(...)` via the metric's prompt builder.
    3. Aggregate per-metric scores (mean across examples), write per-example
       trace to `traces.ndjson`.
  - Concurrency: `asyncio` semaphore = `parallel` (CLI default from
    `defaults.parallel`, `--parallel` overrides).
- `ai_eval/runner/metrics/`
  - `argument_accuracy.py`, `hallucination_rate.py`, `context_precision.py`,
    `faithfulness.py` — each exports `JudgeMetric` with prompt builder +
    `scored_dimension`. `context_precision` is "partial" (deterministic first,
    judge tiebreak) per design §2.6.
  - `latency.py` — `latency_p50`, `latency_p95` computed by engine; no judge.
- `ai_eval/runner/thresholds.py`
  - `evaluate_metric(metric_spec, score, baseline_score, tolerance,
    fail_on_regression) -> {score, delta, threshold, status}`
  - **Unregistered metric → raise `MetricNotImplementedError` (exit 1)** even
    if init-time validation only warned.
  - Fresh-repo: `baseline_score=None` → `delta=None`, `status` from threshold
    only.
- `ai_eval/insights/__init__.py`
- `ai_eval/insights/history.py`
  - `append_history(state_dir, slim_record)` — append to
    `.ai-eval/history.json` (list of `{id, started_at, summary, git.sha, tags}`).
  - `list_history`, `load_run(state_dir, run_id|last)`, `prune(keep)`,
    `export(path)`.
- `ai_eval/insights/diff.py`
  - `compute_delta(current, baseline) -> per-metric Δ` (already partly computed
    by thresholds; this re-derives for display).
- `ai_eval/insights/root_cause.py`
  - `async def explain(metric, current_example, baseline_example, *,
    regression_judge, complete_fn) -> narrative:str` — one judge call per
    regressing (metric, example).
- `ai_eval/storage/runs.py`
  - `new_run_id()` (e.g. `r_<sha5>`), `save_run(state_dir, record, traces,
    judge_log)`, `load_run(state_dir, run_id)`.
- `ai_eval/storage/locks.py`
  - `fcntl`-based advisory lock on `.ai-eval/config.lock` around history writes
    (context manager). Best-effort on non-POSIX.

### Run record (Pydantic, `schema_version: 1`)

Matches design §3.4: `id, started_at, finished_at, git{sha,branch,dirty},
config_hash, baseline_id, tasks{<name>{examples, metrics{<m>{score,delta,
threshold,status}}, latency_ms{p50,p95}}}, summary{passed,failed,errors},
tags`. Add `errors` per task + top-level.

### CLI wiring

- `ai_eval/cli/run.py` — replace `not_yet`:
  - Load resolved config; validate rubrics; resolve golden set.
  - Resolve baseline (`last`/`none`/run-id) via `insights.history`.
  - `asyncio.run(runner.engine.execute(...))` with injected gateway (default
    async client; `--no-cache` toggles cache).
  - Git metadata via `subprocess` (best-effort, `null` on failure).
  - Save run to history + `runs/<id>/` unless `--no-save`.
  - Render per `effective_format`:
    - human: rich table (design §1.2 sample) with Δ column + failures block.
    - json: stable schema (run record).
    - tsv: pipe-safe tab-delimited.
  - Exit `0`/`3` (regression or threshold breach under `--fail-on-regression`)
    /`1` (runner error)/`2` (usage).
- `ai_eval/cli/diff.py` — replace `not_yet`:
  - Load current + baseline runs; filter `--task`/`--metric`/`--limit`.
  - Metric-level Δ table; if regression judge configured and `--explain`,
    `await insights.root_cause.explain(...)` per regressing metric, print
    narrative.
  - Exit `0` (informational) unless `--fail-on-regression` (exit 3 if any
    regression).
- `ai_eval/cli/report.py` — replace `not_yet`:
  - Load run (`last` or id); render human/json/md via `--format`; `--out` writes
    file. `--web` writes a minimal HTML and prints a `file://` URL (out of scope
    to open browser; design says print URL only).
- `ai_eval/cli/history.py` — replace `not_yet`:
  - `--list` (default), `--show <id>`, `--prune --keep N`, `--export <file>`.

### Renderers (`ai_eval/cli/render/`)

- Extend `tables.py` with `render_run(record, no_color)`, `render_diff(...)`,
  `render_history(...)`, `render_report(...)`. Reuse `theme.py` state glyphs.
- `json_out.py` already generic (`dump`).

### Phase 4 tests

- `tests/test_runner_engine.py` — fake gateway + fixture golden set; assert
  metric aggregation, latency, task import error handling, parallelism cap.
- `tests/test_runner_metrics.py` — each metric's `compute`/prompt builder.
- `tests/test_runner_thresholds.py` — delta/status, fresh-repo no-Δ,
  `--fail-on-regression`, unregistered-metric hard-fail (exit 1).
- `tests/test_insights_history.py` — append/list/load/prune/export; advisory
  lock.
- `tests/test_insights_diff.py` — Δ computation.
- `tests/test_insights_root_cause.py` — fake regression judge narrative.
- `tests/test_cli_run.py`, `test_cli_diff.py`, `test_cli_report.py`,
  `test_cli_history.py` — snapshot tests (human + json), fake transport.
- `tests/test_e2e.py` — `init → bootstrap → run → diff` on `tiny_repo` fixture
  using a fake async `acomplete`; asserts golden set populated, run record
  written, regression detected when a metric is forced below threshold.

---

## Cross-cutting

- **Exit codes** stay as `app.py` constants (`EXIT_OK=0`, `EXIT_GENERAL=1`,
  `EXIT_USAGE=2`, `EXIT_REGRESSION=3`).
- **No new runtime deps** — `litellm`, `instructor`, `anyio`, `rich`, `pydantic`
  already core. Async uses stdlib `asyncio` + `anyio.to_thread` only where needed
  for sync glue (none, since gateway is async-native).
- **README** — update the status line to reflect Phase 2-4 and remove the "all
  other commands are stubs" note.
- **Lint/typecheck:** `ruff` + `mypy` via existing `pyproject.toml` config.
  `mypy` may need `ai_eval.judge`, `ai_eval.bootstrap`, `ai_eval.runner`,
  `ai_eval.insights` added if a mypy.ini/scoped config exists (none currently —
  default behavior).

---

## Risks & mitigations

- **Async gateway diverges from sync SLM client** → two client code paths. Keep
  `inference/slm/client.py` sync (it serves a different, init-time purpose); do
  not refactor it. Document the split in `judge/gateway.py` docstring.
- **Hard-fail on unregistered metrics** may break existing `rubrics.yaml` files
  that relied on the warning path. Mitigation: strictness is a **run-time-only**
  gate in `runner/thresholds.py`; `init`/`analyze`/schema validation stay
  lenient. The error message points users to register the metric or remove it.
- **Bootstrap wrapper import-time patching** is fragile if frameworks change
  APIs. Mitigation: wrap each patch in try/except + `skip` on `ImportError`/
  `AttributeError`; record which frameworks were patched into the capture
  metadata so failures are diagnosable.
- **Re-importing user task code in `run`** can have side effects. Mitigation:
  import errors are task-level, not process-level; document that `run` imports
  the task module and recommend `if __name__ == "__main__"` guards; the golden
  set + entry symbol keep the surface minimal.

---

## Out of scope (deferred)

- OpenTelemetry bootstrap (`--no-instrument` path).
- Web dashboard server (`--web` only writes static HTML + prints a URL).
- Streaming run output (`--stream`, NDJSON).
- `config --set` write-back.
- Telemetry opt-in pings.
- SQLite history backend (stays JSON per design decision #2).

---

## Validation checklist (run before considering done)

1. `ruff check .` clean.
2. `mypy ai_eval` clean (or unchanged from baseline).
3. `pytest -q` — all existing + new tests pass.
4. `ai-eval doctor` reports the new judge-gateway check.
5. `ai-eval judge --list` / `--ping` / `--prompt` work with a real Ollama judge
   (manual smoke) and the fake transport in tests.
6. On `tiny_repo`: `ai-eval init` → `ai-eval bootstrap -- pytest` (with a fake
   openai) → `ai-eval run` → `ai-eval diff --against last` produces a run
   record, golden set, and a regression narrative (e2e test).
7. `ai-eval run --fail-on-regression` exits `3` when a metric is forced below
   threshold in the e2e fixture; exits `0` otherwise.
8. `ai-eval run` on a fresh repo (no history) shows `—` Δ and exits `0`.
9. Unregistered metric in `rubrics.yaml` → `ai-eval run` exits `1` with a clear
   fix hint, while `ai-eval init`/`analyze` still only warn.

# Plan: Trace-replay node scoring (score all nodes of one end-to-end entry point)

## Goal

Replace the current "many runnable top-level tasks, each scored by its own
re-executed output" model with: **one end-to-end entry point per use case is
run; every internal node it calls is scored from the captured trace**. This
removes the `_Stub`-driven auto-seed crashes (DAO not iterable, evaluator
recursion, `'str' has no attribute 'chat_id'`) at their root, because internal
nodes are no longer constructed/called by the runner — only the entry point is,
and its internal calls are scored from `example["trace"]["calls"]`.

This is an additive capability on top of the existing trace primitive: the
built-in `_context_precision` metric (`ai_eval/metrics/judge_builtin.py:140`)
already reads `example["trace"]["calls"]`, so the runner already threads the
trace into prompt builders. The gap is: (1) entry-point selection, (2) an
explicit node→metric binding, (3) per-node scoring + rollup in the runner.

## Root-cause recap (why this fixes the failures)

| Failing task | Current behavior | Under this plan |
|---|---|---|
| `DocumentVectorDAO.search_similar_vectors` | Detected as a `rag` root, `top_level=True`, auto-seeded, `_Stub()` session → `'_Stub' object is not iterable` | Demoted (called by `process_query`); scored as a `retrieve` node inside `process_query`'s trace |
| `SingleQueryEvaluator.evaluate_single` | `top_level=True`, auto-seeded → `RecursionError` (recurses through stubbed deps) | Demoted (judge/eval code or internal); never auto-run |
| `ChatMessageService.process_query` | `top_level=True`, scalar `""` bound to `query_request` → `'str' has no attribute 'chat_id'` | **The** selected entry point; runs for real with captured traces |
| `ChatMessageService._create_workflow` | Already `top_level=False` | Unchanged (demoted) |

## Decisions (locked)

1. **Entry-point selection** — call-graph: keep only the *deepest reachable
   root*; demote any surviving root that is called by another surviving root
   (transitive). Automatic; no new user config. Reuses
   `ai_eval/inference/callgraph.py::compute_roots`.
2. **Node→metric binding** — new optional `node_metrics: list[NodeMetric]` on
   `TaskSpec`. `NodeMetric = {node_selector, metric: MetricSpec}`. Runner
   walks `example["trace"]["calls"]`, matches nodes, scores each. Explicit,
   survives in `rubrics.yaml`, backward-compatible (default `[]`).
3. **Node score rollup** — per-node rows + aggregate. New optional
   `node_scores: dict[str, dict[str, float]]` on `ExampleRecord` keyed by
   `node_id` (e.g. `retrieve_0`) → `{metric_name: score}`.
   `TaskRecord.metrics` keeps the weighted-mean aggregate per metric.
4. **`_Stub` removal** — delete `_Stub`, `_placeholder_for`, `_fake_call_args`
   (engine.py:48-109) and the IO-coupled auto-seed path
   (seeder.py `_io_coupled_example` / `seed_for_task` `io_coupled` branch).
   Internal nodes are never constructed/called by the runner. Harness writer
   stays (entry points may still be IO-coupled and need canned reads).

---

## Implementation steps

### Step 1 — Schema changes (`ai_eval/config/schema.py`, `ai_eval/runner/record.py`)

Add `NodeMetric` and `node_metrics` to `TaskSpec`; add `node_scores` to
`ExampleRecord`. All new fields default to empty so existing on-disk
`rubrics.yaml` / `golden_set.json` / `run.json` load unchanged.

`ai_eval/config/schema.py`:
```python
class NodeMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_selector: str   # "kind=retrieve" | "name~=pgvector" | "call_index=0"
    metric: MetricSpec

class TaskSpec(BaseModel):
    ...
    node_metrics: list[NodeMetric] = Field(default_factory=list)
```

`ai_eval/runner/record.py`:
```python
class ExampleRecord(BaseModel):
    ...
    node_scores: dict[str, dict[str, float]] = Field(default_factory=dict)
```

Selector grammar (kept tiny, documented in the NodeMetric docstring):
- `kind=<x>` — match `call["kind"] == x` (e.g. `kind=retrieve`, `kind=llm`)
- `name~=<substr>` — substring match on `call["name"]`
- `name=<exact>` — exact match on `call["name"]`
- `call_index=<n>` — the n-th call (0-based)

A selector may match multiple nodes; each match is scored independently and
gets a synthetic `node_id` = `<kind>_<i>` (e.g. `retrieve_0`,
`retrieve_1`).

### Step 2 — Entry-point selection (`ai_eval/inference/task_selection.py`)

After the existing `compute_roots` demotion + judge-exclusion, add a **third
layer** that demotes any survivor that is itself called by another survivor.
Concretely: among `kept` tasks, build the call edges again and keep only roots
not reached by any other kept root. The survivors are the end-to-end entries;
the demoted ones get `top_level=False` propagated (synthesize already does the
private-method demotion; extend it to also honor a new `reached_by_peer` flag
set here).

Update `select_tasks` to set `DetectedTask.top_level = False` on
peer-reached tasks before returning (the dataclass field already exists;
synthesize.py:207 already propagates `task.top_level`). No new field needed on
`DetectedTask`.

Edge case: if multiple disjoint use cases exist (e.g. a chat service + a
separate ingestion service), each keeps its own deepest root — the rule is
"demote if reached by a *kept peer*", not "keep exactly one globally".

### Step 3 — Synthesize/SLM propagation (`ai_eval/inference/synthesize.py`, `ai_eval/inference/slm/builder.py`)

No `top_level` logic change beyond Step 2's flag — synthesize.py:207 already
does `task.top_level and not _is_private_entry(task.entry)`. The
`_is_private_entry` check stays (private methods stay demoted regardless).

For `node_metrics`: the rule/SLM engines do NOT auto-populate it (that's a
user/SLM-authored decision per use case). `init` writes an empty `node_metrics:
[]` (or omits it via the default). Document in `docs/metrics.md` that users
add node metrics after a bootstrap captures real traces (so they know the
actual `call["kind"]`/`call["name"]` values to select on).

### Step 4 — Runner: per-node scoring (`ai_eval/runner/engine.py`)

In `_run_example`, after the entry call produces `output` and before/alongside
the existing per-metric judge loop, add a node-scoring pass when
`tspec.node_metrics` is non-empty AND `example.get("trace")` has calls:

```python
node_scores: dict[str, dict[str, float]] = {}
calls = (example.get("trace") or {}).get("calls", [])
for nm in tspec.node_metrics:
    matches = _select_nodes(calls, nm.node_selector)  # returns [(node_id, call), ...]
    for node_id, call in matches:
        node_example = {**example, "input": call.get("args"), "expected": None,
                        "trace": {"calls": [call]}}
        # reuse the existing judge_score path with nm.metric
        score = await _score_one_metric(nm.metric, node_example, call.get("result"), ...)
        node_scores.setdefault(node_id, {})[nm.metric.name] = score
```

`_select_nodes` implements the selector grammar from Step 1. The per-node
judge call reuses `judge_score` + the metric's `prompt_builder` unchanged —
the builder already accepts `(task_spec, example, output)` and the trace is
already in `example`. `_score_one_metric` is a small refactor extracting the
per-metric judge loop body (currently inline at engine.py:515-545) into a
callable so both the entry-level and node-level paths share it.

`node_scores` is set on the returned `ExampleRecord`.

Aggregate: extend `_aggregate_metrics` so that for a metric name present in
`node_metrics`, the aggregate score is the weighted mean of that metric's
node scores across all examples (mirrors the existing mean-over-examples
logic, just sourcing from `node_scores` instead of `metric_scores`).

### Step 5 — Remove `_Stub` + IO-coupled auto-seed

`ai_eval/runner/engine.py`:
- Delete `_Stub` (48-62), `_placeholder_for` (65-81), `_fake_call_args`
  (84-109).
- `_import_entry`: for dotted `Class.method` entries, construct the instance.
  Constructor args now come ONLY from the harness (if present) or a no-arg
  `cls()` fallback. Drop the `_fake_call_args(sig)` path. If construction
  fails with `TypeError` (required args, no harness), record an
  `error`-status ExampleRecord with a clear message directing to
  `ai-evals bootstrap` (the entry point is IO-coupled and needs a harness).
- `_build_call_args` stays but is simplified: scalar binding is only for
  legitimate top-level pure-LLM entries (the only runnable kind now). Remove
  the object-typed-param `scalar_idx` workaround from the prior plan — it's
  moot because object-typed-param entries are no longer top-level.

`ai_eval/scaffold/seeder.py`:
- Delete `_io_coupled_example` and the `io_coupled` param/branch of
  `seed_for_task`. All runnable (top-level) tasks get the pure-LLM shape
  variants; non-top-level tasks get 0 (existing behavior).
- `seed_golden_set` drops the `io_coupled_tasks` param. Update callers
  (`ai_eval/cli/init.py:344-346`).

`ai_eval/scaffold/harness_writer.py`: unchanged (entry points may still be
IO-coupled — e.g. `process_query` reads `self.chat_dao.*` — and need canned
fixtures so the entry run is green without a live DB).

### Step 6 — Render (`ai_eval/cli/render/tables.py`)

After the existing per-task metric table, if any example has `node_scores`,
print a compact per-node block:
```
  nodes:
    retrieve_0  context_precision=0.8800  faithfulness=0.9100
    llm_0       hallucination_rate=0.0500
```
Keep it opt-in (only when `node_scores` non-empty) so existing tasks render
unchanged. JSON output (`render/json_out.py`) already serializes the whole
record, so `node_scores` flows through with no change.

### Step 7 — AGENTS.md §1 update

Rewrite §1 ("Target the complete job") to describe the new model:
- `ai-evals run` runs only the **end-to-end entry point** per use case (the
  deepest reachable root).
- Internal nodes are **scored from the captured trace**, not re-executed.
  Node→metric binding is declared via `node_metrics` on the entry task.
- `_Stub`/`_fake_call_args` are gone; IO-coupled entry points need a harness
  (the existing D5 harness writer) or a real backend.
- The "auto-seed + run for a green pipeline" contract now applies only to
  pure-LLM entry points. IO-coupled entry points require `ai-evals bootstrap`
  to capture a real trace before `run` can score their nodes.

Update §3 (Seeder/harness boundaries) to drop the IO-coupled auto-seed
description and note the harness is now for entry-point construction only.

---

## Validation

Per AGENTS.md, after touching `ai_eval/` or `tests/`:
```bash
uv run ruff format ai_eval tests
uv run ruff check ai_eval tests
uv run mypy ai_eval
uv run pytest -q
```

### Targeted tests to add/update

- `tests/test_top_level.py`: add a test that a root called by another
  surviving root is demoted (deepest-root selection). Mirror the existing
  `test_compute_roots_demotes_reached_helper` shape but at the
  `select_tasks` level.
- `tests/test_runner_engine.py` (new or existing): a task with
  `node_metrics=[NodeMetric(node_selector="kind=retrieve",
  metric=MetricSpec(name="context_precision",...))]` and a golden example
  carrying `trace.calls=[{kind:"retrieve",...}]` → assert
  `example.node_scores["retrieve_0"]["context_precision"]` is set and the
  task-level aggregate is the mean.
- Update `tests/test_runner_dotted_entry.py`: remove the `_Stub`/
  `_fake_call_args` tests (the symbols are deleted). Replace with a test that
  a dotted entry whose constructor requires args and has no harness fails
  with a clear bootstrap-directed error (not a `RecursionError`/`TypeError`
  from a stub).
- `tests/test_seeder.py`: remove the `io_coupled` branch coverage; assert
  all top-level tasks get the 5 pure-LLM variants.

### End-to-end check against Basic_RAG

After implementation, with the editable install live (per AGENTS.md workflow):
```bash
cd /Users/sharayuchoudhari/Dev/Projects/RAG/Basic_RAG
uv run python -c "import ai_eval.runner.engine as e; print(e.__file__)"
# must print the AIAutoEvals source path
uv run ai-evals init --rubric-engine rules   # regenerate rubrics.yaml
```
Expected `rubrics.yaml` deltas:
- `chat_message_service_process_query`: `top_level: true` (the entry point).
- `document_vector_d_a_o_search_similar_vectors`,
  `single_query_evaluator_evaluate_single`,
  `chat_message_service_create_workflow*`: `top_level: false`.

After a `bootstrap` that captures a `process_query` trace containing a
`retrieve` call, `run` should score `context_precision` against the retrieve
node (via `node_metrics`) with no `_Stub`/`RecursionError`/`AttributeError`.

## Risks / open questions

- **Deepest-root selection false positives**: a repo with two equally-deep
  roots (two independent use cases) keeps both — intended. A repo where the
  "real" entry is a thin controller calling a fat service may demote the
  controller and keep the service; the escape hatch is the existing
  `force_task: true` hint. Document this.
- **Node selector fragility**: `name~=` depends on the captured `call.name`
  (framework-wrangler-defined, e.g. `pgvector.orm.op`). If the user writes a
  selector before bootstrapping, they guess the name. Mitigation: document
  that `node_metrics` is authored *after* a first bootstrap reveals real
  `call.kind`/`call.name` values; `init` writes `node_metrics: []`.
- **No-op for repos without traces**: if a user runs `ai-evals run` before
  `bootstrap`, entry points have no `trace.calls`, so `node_metrics` produces
  no node scores (status `skip`). The existing "auto-seeded fixtures" notice
  covers this. This is the honest behavior — node scoring requires captured
  traces by construction.
- **Schema migration**: `node_metrics`/`node_scores` default to empty, so
  no on-disk file migration is needed. `run.json` written by older versions
  loads fine (Pydantic ignores unknown keys only if `extra="ignore"`; these
  schemas use `extra="forbid"` — but new fields are *absent* in old files,
  which is allowed since they have defaults). Confirm by loading an existing
  `run.json` in a test.

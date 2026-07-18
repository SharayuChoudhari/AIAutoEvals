# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog; adheres to Semantic Versioning.

## [0.1.7] - 2026-07-19
### Added
- **Trace-replay node scoring** (AGENTS.md §1): `ai-evals run` now runs only
  the end-to-end entry point per use case and scores every internal node it
  calls from the captured trace (`example["trace"]["calls"]`), not
  re-executed. Node→metric binding is declared via the new
  `node_metrics: list[NodeMetric]` field on `TaskSpec`. Each `NodeMetric` is
  `{node_selector, metric: MetricSpec}`; the runner walks the trace, matches
  nodes by selector (`kind=retrieve`, `name~=pgvector`, `name=exact`,
  `call_index=0`), and scores each match. Per-node scores roll into the new
  `ExampleRecord.node_scores` field (keyed by synthetic `node_id` like
  `retrieve_0`); the task-level aggregate is the weighted mean of node scores
  across examples. `init` writes `node_metrics: []` — author entries after a
  first `ai-evals bootstrap` reveals real `call.kind`/`call.name` values. See
  `docs/metrics.md` for the selector grammar.
- **Deepest-root selection** (Layer 3 in `task_selection.py`): among surviving
  tasks, demotes any that is itself called by another surviving peer to
  `top_level=False`. Only the end-to-end entry point per use case is run;
  internal nodes are scored from the trace. Independent use cases each keep
  their own deepest root.
- Per-node render block: `ai-evals run` (human format) emits a compact
  `nodes:` section with per-node `metric=score` rows when an example has
  `node_scores`. Opt-in — tasks without `node_metrics` render unchanged.

### Changed
- **`_Stub`/`_fake_call_args` removed** (`ai_eval/runner/engine.py`): dotted
  `Class.method` entries now construct with no args (or a harness). An
  IO-coupled entry point whose `__init__` requires real args and has no
  harness fails with a clear bootstrap-directed `TypeError` (not a
  `RecursionError`/`AttributeError` from a stub). The harness writer (D5)
  stays — it monkey-patches `self.<dao>.<method>()` reads for entry-point
  construction only.
- **IO-coupled auto-seed removed** (`ai_eval/scaffold/seeder.py`): the
  `_Stub`-driven single green-pipeline example is gone. All top-level tasks
  now get the 5 pure-LLM shape variants; IO-coupled entry points require
  `ai-evals bootstrap` to capture a real trace before `run` can score their
  nodes. `seed_golden_set` no longer takes the `io_coupled_tasks` param.
- Refactored the per-metric judge loop into `_score_one_metric` (shared by
  the entry-level and node-level judge passes) so both use identical
  prompt-building and gateway-call semantics.
- Node metrics are validated on the same registry surface as entry-level
  metrics, so a typo'd `node_metrics` name fails fast at run start
  (`MetricNotImplementedError`, exit 1).

### Fixed
- Auto-seed no longer attempts to bind scalar inputs to object-typed params on
  internal methods (the `_build_call_args` scalar-binding crash path is closed
  by skipping non-top-level tasks entirely; a required non-str param it can't
  bind raises a clear `TypeError` directing to `ai-evals bootstrap`).
- Eliminated the `_Stub`-driven crashes (`'_Stub' object is not iterable`,
  `RecursionError` from stubbed deps, `'str' has no attribute 'chat_id'`) by
  removing the stub construction path entirely. Internal nodes are now scored
  from the captured trace, never constructed/called by the runner.

## [0.1.6] - 2026-07-14
### Added
- `AGENTS.md` governing how AI coding agents operate on this repo (stack,
  conventions, architectural contracts, workflow).
- `top_level: bool` field on `TaskSpec`/`DetectedTask` — auto-seed and `run`
  now target only public top-level entries (the complete job). Internal
  methods (DAOs, private `_*` methods, IO-coupled services) are scaffolded
  into `rubrics.yaml` but skipped by auto-seed/run with a notice directing
  users to `ai-evals bootstrap`.
- Project-local metric registry: `eval/metrics.yaml` loader
  (`ai_eval/metrics/local.py`) with `{{placeholder}}` prompt templates. Merged
  into `load_metrics`/`load_judge_metrics` (built-ins win on collision; local
  can only add names). See `docs/metrics.md` for the schema and a copy-paste
  template.

### Changed
- Trimmed built-in metric set to the high-frequency five
  (`hallucination_rate`, `context_precision`, `faithfulness`,
  `argument_accuracy`, `task_completion`) plus the non-judge latency pair.
  Removed niche metrics (`slot_filling_accuracy`, `translation_bleu`,
  `classification_f1`, `extraction_field_accuracy`, `scoring_accuracy`,
  `summary_faithfulness`) from built-ins — re-declare them in the consuming
  project's `eval/metrics.yaml`.
- `workflow` task-type default metrics reduced to `task_completion` only.
- SLM prompts (`few_shot.yaml`, `classify_task.txt`) updated to reference
  built-in metrics only.
- `assert_metric_implemented` error message now points to `eval/metrics.yaml`
  and `docs/metrics.md`.

## [0.1.4] - 2026-07-14
### Fixed
- `ai-evals` CLI failed to check local imports due to incorrect current working directory handling.

## [0.1.0] - 2026-07-10
### Added
- Phase 1: static analysis & scaffolding (`ai-evals init`, `analyze`, `doctor`, `config`).
- Phase 2: model-agnostic judge gateway via LiteLLM + Instructor (`ai-evals judge`).
- Phase 3: automated golden-set bootstrapper (`ai-evals bootstrap`).
- Phase 4: run execution & insights (`ai-evals run`, `diff`, `report`, `history`).

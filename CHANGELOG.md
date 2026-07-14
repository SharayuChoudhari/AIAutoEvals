# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog; adheres to Semantic Versioning.

## [Unreleased]
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

### Fixed
- Auto-seed no longer attempts to bind scalar inputs to object-typed params on
  internal methods (the `_build_call_args` scalar-binding crash path is closed
  by skipping non-top-level tasks entirely).

## [0.1.4] - 2026-07-14
### Fixed
- `ai-evals` CLI failed to check local imports due to incorrect current working directory handling.

## [0.1.0] - 2026-07-10
### Added
- Phase 1: static analysis & scaffolding (`ai-evals init`, `analyze`, `doctor`, `config`).
- Phase 2: model-agnostic judge gateway via LiteLLM + Instructor (`ai-evals judge`).
- Phase 3: automated golden-set bootstrapper (`ai-evals bootstrap`).
- Phase 4: run execution & insights (`ai-evals run`, `diff`, `report`, `history`).

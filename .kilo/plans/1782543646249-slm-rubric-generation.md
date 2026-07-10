# Plan: SLM-powered rubric generation (hybrid AST + SLM)

## Goal
Replace the brittle rule-only rubric classification with a hybrid pipeline where the AST scan
gathers grounded evidence and a local SLM classifies tasks, infers type/inputs/outputs, and picks
metrics from an extensible registry. Must fix the two reported failure modes:

1. **RAG misclassified as `chat`** — projects using unrecognised retrieval (raw `openai.chat`
   with manual retrieval, non-LangChain/non-pgvector/non-ChromaDB stores) fall through to `chat`.
2. **Appointment-booking chatbot detected as nothing** — no recognised framework imports → empty
   task list → `custom` project type with zero tasks.

## Background / current state (from codebase exploration)
- Stack: Python >=3.10, hatchling, deps in `pyproject.toml` (`typer`, `rich`, `pydantic>=2.6`,
  `pyyaml`, `anyio`). Optional `judge` extras already declared: `litellm>=1.40`, `instructor>=1.3`,
  `httpx>=0.27` — **not installed/wired yet**.
- Rule-based pipeline (fully in `ai_eval/inference/`):
  - `ast_scan.py:248 scan_repo()` walks repo, parses `.py`, runs detectors.
  - Detectors under `ai_eval/inference/detectors/`: `openai_chat`, `openai_tools`, `langchain`,
    `_langchain_retrieval`, `pgvector`, `chromadb`. Shared matchers in `signatures.py`.
  - `synthesize.py:80 build_rubrics()` classifies project type via `_classify_project_type`
    (`synthesize.py:54`) and attaches fixed `_DEFAULT_METRICS` per the 4 task types.
  - `rubrics_writer.py:25 to_yaml()` / `:36 write()` emit `eval/rubrics.yaml`.
- Schema (`config/schema.py`): `RubricsConfig`, `TaskSpec` with
  `type: Literal["tool_calling","rag","agent","chat"]`, `inputs`, `outputs`, `metrics`.
- Defaults (`config/defaults.py`): `DEFAULT_JUDGE="ollama/qwen2.5-coder:7b"`,
  `DEFAULT_REGRESSION_JUDGE="openai/gpt-4o-mini"`.
- **No LLM client exists.** `cli/judge.py`, `cli/run.py`, `cli/bootstrap.py` are stubs; `judge/`
  package is referenced but unimplemented.
- Entry points: `ai-eval init` (`cli/init.py:47`) and `ai-eval analyze` (`cli/analyze.py:100`)
  drive rubric generation; `analyze` merges via `_merge_dicts` (`analyze.py:56`) with
  `--merge-strategy {keep,overwrite}`.
- Tests: `tests/test_detectors.py`, `tests/test_cli_init.py`, `tests/conftest.py` (`tiny_repo`,
  `tenant_rag_repo` fixtures).

## Decisions (confirmed with user)
1. **SLM role**: Hybrid — AST scan gathers evidence; SLM consumes `ScanResult` + snippets to emit
   the final `RubricsConfig`.
2. **Client + model**: litellm-based client; default `ollama/qwen2.5-coder:7b`; overridable via
   `--rubric-model` flag / config. Install the `judge` extras.
3. **Integration**: New `--rubric-engine {rules,slm,hybrid}` flag (default `hybrid`) on `init` and
   `analyze`. `rules` preserves current behavior; `slm` skips rule classification; `hybrid` runs
   rules for evidence then SLM for classification.
4. **Output validation**: `instructor` forces SLM JSON into the existing Pydantic `RubricsConfig`.
5. **Evidence scope**: AST tasks + per-task code snippets + framework hints. Empty-evidence
   fallback scans entry-point files (main.py/app.py/handlers/*agent*/*route*).
6. **Task types**: Open — `task_type: str` + new `purpose: str | None` free-text field. Drop the
   4-value Literal. Lets SLM express scoring/rating, document→JSON extraction, classification,
   summarization, translation, etc.
7. **Metrics**: Extensible metric registry. Built-in metrics ship in code; third parties extend
   via `ai_eval.metrics` entry-point group. SLM may only pick registered metric names; unknown
   names are rejected by validation.
8. **Cost control**: One SLM call per AST task (snippet + framework hints) + one repo-level call
   for empty-evidence fallback. Cap snippet chars per task, cap tasks sent, enforce total prompt
   budget. Configurable via flags (`--rubric-max-snippet-chars`, `--rubric-max-tasks`,
   `--rubric-budget-tokens`).
9. **Determinism**: temperature=0, retry on validation failure (instructor retries), response
   cache under `.ai-eval/rubric_cache/` keyed by evidence hash for cheap reproducible re-runs.
10. **Prompt assets**: Bundled prompt templates under `ai_eval/inference/prompts/`
    (`classify_task.txt`, `empty_recover.txt`, plus few-shot examples YAML). Versioned with code.
11. **SLM failure mode**: Hard fail when `--rubric-engine=slm|hybrid` is selected and the SLM is
    unavailable (no ollama, no key, network error after retries). Clear error with remediation
    hint; does NOT silently fall back to rules. (`rules` engine still available.)
12. **Merge semantics**: `analyze` reuses existing `_merge_dicts` keep/overwrite; SLM output is
    just another `RubricsConfig` source merged in.

## Tasks (ordered)

### A. Dependencies + scaffolding
1. `pyproject.toml`: move `litellm`, `instructor`, `httpx` from optional `judge` extras into the
   core dependencies (or into a new `slm` extra that `init`/`analyze` require when engine≠rules).
   Decide: simplest is to add to core deps since hybrid is the default.
2. Create `ai_eval/inference/slm/` package:
   - `client.py` — litellm-backed client wrapper (single function:
     `complete(model, messages, *, temperature=0, max_tokens, retries)` returning parsed instructor
     object).
   - `prompts.py` — loader for bundled prompt templates from `ai_eval/inference/prompts/`.
   - `cache.py` — evidence-hash keyed response cache under `.ai-eval/rubric_cache/`.
   - `evidence.py` — builds per-task evidence dict (snippet, framework hints, imports/calls)
     from `ScanResult`; entry-point selection for empty-evidence fallback.

### B. Metric registry
3. Create `ai_eval/metrics/registry.py`:
   - `Metric` dataclass: `name`, `description`, `default_threshold`, `default_weight`,
     `applicable_task_types` (optional hint list, not enforced).
   - Built-in metrics (reuse existing 4 + add): `hallucination_rate`, `context_precision`,
     `faithfulness`, `argument_accuracy`, `scoring_accuracy`, `extraction_field_accuracy`,
     `classification_f1`, `summary_faithfulness`, `translation_bleu`,
     `task_completion` (for booking/agent), `slot_filling_accuracy`.
   - Plugin loading via `ai_eval.metrics` entry-point group (mirror `load_entrypoint_detectors`
     in `ast_scan.py:79`).
   - `is_registered(name)` / `get(name)` / `all_names()`; validator hook for `MetricSpec.name`.
4. Add entry-point group `ai_eval.metrics` to `pyproject.toml` `[project.entry-points]` (even if
   empty initially, documents the contract).

### C. Schema changes (`ai_eval/config/schema.py`)
5. `TaskSpec`: change `type: Literal[...]` → `type: str`; add `purpose: str | None = None`.
   Keep backward compat: existing rule detectors that set `type="chat"` etc. still valid.
6. `MetricSpec`: add validator that `name` must be in the metric registry (registered or built-in).
   Validation should be skippable for rule-engine-produced metrics? No — register all existing 4
   metrics so both engines validate identically.
7. `RubricsConfig`: ensure `project_type` still a Literal but broaden to include the open
   `task_type` story (project_type stays as-is for now; it's a summary). Add
   `rubric_engine: str | None = None` field to record which engine produced this file.

### D. SLM rubric builder
8. `ai_eval/inference/slm/builder.py` — `build_rubrics_slm(scan, *, model, evidence_caps) -> RubricsConfig`:
   - For each `DetectedTask` in `scan.tasks`: build evidence, call SLM via instructor into a
     `TaskSpec`-shaped pydantic model (with `type: str`, `purpose`, `inputs`, `outputs`,
     `metrics`). Per-task call, capped.
   - Empty-evidence fallback (`scan.tasks` empty): select entry-point files via
     `evidence.select_entry_point_files(repo)`; one SLM call to propose tasks
     (`empty_recover.txt` prompt).
   - Compose into `RubricsConfig` using existing `JudgeConfig` + `DefaultsBlock` defaults
     (reuse `defaults.py`).
9. `ai_eval/inference/slm/validation.py` — `validate_against_scan(rubrics, scan)`:
   - Reject any `TaskSpec` whose `file_path` not in scanned file set → drop + warn.
   - Reject `entry` not found in that file's defs → drop entry, keep task with warning.
   - Enforce metric registry membership (delegates to schema validator).
   - Re-run `_looks_like_rag` (`synthesize.py:41`) as a sanity sanity check: if a task's
     inputs/outputs clearly indicate RAG but `type` says `chat`, warn (do not auto-fix — SLM owns
     the type now).

### E. Prompt templates
10. `ai_eval/inference/prompts/classify_task.txt` — system prompt: given a code snippet + framework
    hints + imports + the enclosing function name, classify the task's purpose, type (open string
    from a suggested vocabulary: chat, rag, agent, tool_calling, scoring, extraction, classification,
    summarization, translation, booking/workflow, other), inputs, outputs, and pick metrics from
    the provided registry name list.
11. `ai_eval/inference/prompts/empty_recover.txt` — given entry-point file contents + repo file
    tree, propose tasks.
12. `ai_eval/inference/prompts/few_shot.yaml` — 4–6 worked examples covering RAG (raw openai+manual
    retrieval), booking/tool-calling, scoring, document→JSON extraction, classification,
    summarization.

### F. Wire into CLI
13. `cli/init.py` and `cli/analyze.py`: add `--rubric-engine {rules,slm,hybrid}` (default
    `hybrid`), `--rubric-model` (default from `defaults.py`), and the cap flags
    (`--rubric-max-snippet-chars`, `--rubric-max-tasks`, `--rubric-budget-tokens`).
14. Branch logic in `init.py` write path (~line 180–199):
    - `rules`: existing `build_rubrics(scan)`.
    - `slm`: `build_rubrics_slm(scan, ...)` (skip rule classification but still run AST scan for
      evidence).
    - `hybrid`: run rules to populate `scan.tasks`; then `build_rubrics_slm` using that evidence.
    - On SLM unavailability after retries: raise a `RubricEngineError` with remediation message
      (start ollama / set API key / use `--rubric-engine rules`). Hard fail.
15. `analyze.py` (~line 144): after SLM-produced `RubricsConfig`, reuse `_merge_dicts` with
    `--merge-strategy` against existing `rubrics.yaml`.

### G. Config resolution
16. `config/loader.py`: extend allowlist (`:73`) and resolution order (`:131`) for
    `AI_EVAL_RUBRIC_ENGINE`, `AI_EVAL_RUBRIC_MODEL`, and the cap vars.

### H. Tests
17. `tests/test_slm_builder.py`:
    - Unit test `evidence.py` snippet assembly + entry-point selection.
    - Mock the litellm client (inject a fake `complete` returning canned `TaskSpec` JSON) and
      assert `build_rubrics_slm` produces expected `RubricsConfig` for: RAG-with-manual-retrieval,
      booking/tool-call, scoring, extraction, empty-evidence fallback.
    - Assert `validate_against_scan` drops hallucinated file_paths/entries.
    - Assert cache hit short-circuits the SLM call (evidence-hash identical → no second call).
    - Assert hard-fail when `complete` raises after retries.
18. `tests/test_metric_registry.py`: built-in registry contents; entry-point plugin loading;
    `MetricSpec` validation rejects unknown metric names.
19. Extend `tests/test_cli_init.py`: `--rubric-engine rules` reproduces old snapshots;
    `--rubric-engine slm` with mocked client writes rubric; missing model → hard-fail exit code.
20. Update `tests/conftest.py` fixtures if schema change breaks `tiny_repo`/`tenant_rag_repo`
    assertions (Literal→str is backward compatible, so likely no change).

### I. Docs
21. Update `README.md` and `docs/how-it-works.html` (already open) to describe the hybrid engine,
    the open task_type + metric registry, flag usage, and ollama setup. (Only if user confirms
    docs are in scope — default: minimal README addition.)

## Risks / validation
- **Schema backward compat**: changing `type` Literal→str must not break existing rule detectors
  (they emit the same string values). Validate by running existing `tests/test_detectors.py`.
- **Metric registry gating**: registering the 4 existing metrics keeps rule-engine output valid;
  any existing `rubrics.yaml` without registered metrics will now fail validation — document a
  migration / loosen validator to allow unknown names with a deprecation warning for one release.
- **Cost**: per-task SLM calls can be slow on large repos; caps + cache mitigate. Validate with a
  repo of ~20 tasks and assert total SLM calls ≤ max-tasks + 1.
- **Hallucination**: AST-bound validation is the guardrail. Validate with a test that injects a
  fake SLM response referencing a non-existent file and asserts the task is dropped.
- **Determinism**: temperature=0 + cache gives reproducibility; assert two `analyze` runs on the
  same repo produce identical `rubrics.yaml` when cache is warm.
- **ollama availability**: hard-fail path must give an actionable message. Validate the message
  in a test that fakes connection failure.

## Open questions (non-blocking, for follow-up)
- Should `project_type` (repo-level summary) also become open/free-text, or stay the Literal?
  Current plan: keep Literal; the open `task_type` per-task is the dynamic surface.
- Should the SLM also be allowed to propose *new* metrics (and emit their definitions), or only
  pick from the registry? Current plan: registry-only; new metrics require a plugin entry point.
- Streaming/progress UI for many-task SLM runs via `rich`. Out of scope for this plan.

# Make the ai-eval pipeline detect workflows (file structure + hints)

## Context & root cause

`ai-eval init`/`analyze` produce **empty** `tasks:` for repos whose orchestration
isn't `chat.completions.create(tools=...)`. Two concrete gaps (verified against
`appointment_helper`):

1. **Responses API undetected.** `is_openai_completions_create`
   (`ai_eval/inference/signatures.py:130`) matches only `<x>.chat.completions.create`
   and legacy `ChatCompletion.create`. The workflow calls
   `client.responses.create(tools=...)` (`conversation_workflow_service.py:232,240`),
   so neither `OpenAIToolsDetector` nor `OpenAIChatDetector` fires.
2. **LangGraph undetected.** `LangChainDetector` matches `langchain*` imports +
   chain verbs (`invoke`/`run`). The workflow uses `langgraph.graph.StateGraph`,
   `add_node`, `graph.invoke` (`:78,82,59`) — a different package + graph shape.

`scan_repo` (`ai_eval/inference/ast_scan.py:248`) runs only AST call-pattern
detectors; there is no concept of `type: workflow` as a detected task, and
`synthesize._DEFAULT_METRICS` (`ai_eval/inference/synthesize.py:22`) has no
`workflow` entry (though the metric registry already lists
`task_completion`/`slot_filling_accuracy` as applicable to `workflow`,
`ai_eval/metrics/registry.py:94,100`).

### Goal

Detect **any** workflow with minimal false negatives, not just the
appointment_helper case. Strategy: automatic AST detection for the common
frameworks (OpenAI Responses, LangGraph) + an **opt-in hints file** as the
escape hatch for novel/proprietary orchestration. File-structure heuristics are
deliberately **not** used as rule triggers (too brittle, low precision).

## Resolved decisions

- **Signals:** AST detectors + opt-in `eval/ai-eval.hints.yaml`. No
  file-structure rule triggers. (Directory hints may be passed as passive
  evidence to the SLM engine only — out of scope unless trivial.)
- **Detected `type`:** `workflow` for both new detectors.
- **Granularity:** AST emits **one task per orchestration entry point**. The
  hints file is the lever to split into per-intent tasks when a user wants finer
  granularity. Multiple intents (book/cancel/…) under one entry point live as
  separate golden captures under that one task — matching the existing
  hand-authored appointment_helper rubric structure.
- **`type: workflow` metric defaults** (added to `synthesize._DEFAULT_METRICS`):
  `task_completion` (0.9, w1.0) + `slot_filling_accuracy` (0.9, w1.0), the two
  metrics native to `workflow`. Users (or the hand-authored rubric) add
  `argument_accuracy`/`hallucination_rate` per-task; the detector keeps
  defaults minimal and native to avoid the applicability mismatch noted in the
  prior plan's Risks.
- **Out of scope:** the Phase-4 `run` runner, modifying app source, a
  file-structure rules engine, SLM prompt changes for directory hints.

## Files to change

1. `ai_eval/inference/signatures.py` — add Responses-API predicates.
2. `ai_eval/inference/detectors/openai_responses.py` — **new** detector.
3. `ai_eval/inference/detectors/langgraph.py` — **new** detector.
4. `ai_eval/inference/ast_scan.py` — register the two new detectors
   (`builtin_detectors`), ordering them before the chat detector so tools/workflow
   take precedence (mirrors the existing `OpenAIToolsDetector`-before-chat rule).
5. `ai_eval/inference/synthesize.py` — add the `workflow` entry to
   `_DEFAULT_METRICS` and teach `_classify_project_type` to treat `workflow`
   tasks as tool-bearing (so `project_type` becomes `tool_calling`/`agent`/a
   new `workflow` value — see Open question 1).
6. `ai_eval/inference/hints.py` — **new** hints loader + merge with scan tasks.
7. `ai_eval/config/schema.py` — **new** `HintsFile`/`HintTaskSpec` pydantic
   models (extra="forbid"); wire into `analyze`/`init`.
8. `ai_eval/cli/analyze.py` and `ai_eval/cli/init.py` — load hints (if present),
   merge hint tasks with `scan.tasks`, dedup by `entry`/`file_path`.
9. `ai_eval/scaffold/init_writer.py` (or equivalent) — emit a commented-out
  `eval/ai-eval.hints.yaml` template on `init`.
10. Tests under `tests/` for the two detectors, the hints loader/merge, and a
    repo-level scan over a fixture mirroring `appointment_helper`'s shape.

## Detector specs

### `OpenAIResponsesDetector` (`openai_responses.py`, framework `openai`)

- `matches`: `has_import_prefix(imports, "openai")` (same as siblings).
- `extract`: iterate `_calls`; match the **Responses API** shape via a new
  `is_openai_responses_create(call)` predicate in `signatures.py`:
  `attr_chain(call.func)[-1] == "create"` **and**
  `attr_chain(call.func)[-2] == "responses"` (i.e. `<client>.responses.create`).
- **Partition key with siblings:** if `has_openai_tool_kwarg(call)` (tools/functions/tool_choice)
  → emit `type="workflow"` (tool-bearing Responses call); else fall through
  (`OpenAIChatDetector` still owns non-tool completions; keep these mutually
  exclusive by kwarg presence, exactly as chat/tools are partitioned today).
- One task per enclosing def (`enclosing_def_name`), `seen`-deduped. `name =
  entry or f"{file_path.stem}_workflow"`. `inputs=["messages","tools"]`,
  `outputs=["output","tool_calls"]`. `evidence=["openai responses.create(tools=...)
  at {rel}:{call.lineno}"]`.

### `LangGraphDetector` (`langgraph.py`, framework `langgraph`)

- `matches`: `has_import_prefix(imports, "langgraph")`.
- `extract`: detect graph construction — a `StateGraph(...)` call OR
  `add_node(...)` calls followed by `.compile()`/`.invoke()`/`.ainvoke()` on the
  resulting object. Emit **one** task per enclosing def that contains a
  `graph.invoke`/`ainvoke`/`stream`/`astream` call (the entry point that runs
  the workflow). `type="workflow"`, `name = entry or f"{file_path.stem}_workflow"`,
  `inputs=["state"]`, `outputs=["state"]`,
  `evidence=["langgraph graph.invoke at {rel}:{call.lineno}"]`.
- Only the `.invoke`/`.compile` call site emits (not every `add_node`), so one
  entry point → one task. If no invoke call is found but a `.compile()` exists,
  fall back to emitting on `.compile()` (the graph's public surface).

### `synthesize._DEFAULT_METRICS["workflow"]`

```python
"workflow": [
    MetricSpec(name="task_completion", threshold=0.9, weight=1.0),
    MetricSpec(name="slot_filling_accuracy", threshold=0.9, weight=1.0),
],
```

### `_classify_project_type`

Count `workflow` tasks alongside `tool_calling`/`agent` as tool-bearing
(`has_tools = tool_count>0 or agent_count>0 or workflow_count>0`). If a repo is
**all** workflow tasks and nothing else, set `project_type = "custom"` (the
existing fallback for `workflow`-heavy repos; see Open question 1 for whether to
add a dedicated `workflow` ProjectType literal).

## Hints file design (`eval/ai-eval.hints.yaml`)

Opt-in, additive escape hatch for anything AST can't see (Temporal, Prefect,
Airflow, proprietary state loops, or to split one entry point into per-intent
tasks). Shape (pydantic, `extra="forbid"`):

```yaml
# Optional; absence means "no hints".
tasks:
  - name: whatsapp_book_appointment        # snake_case, required
    file_path: services/conversation_workflow_service.py  # required
    entry: ConversationWorkflowService.run # optional
    type: workflow                          # required (any registered TaskType)
    inputs: [patient_phone_number, message_text]   # optional
    outputs: [reply_text, intent, appointment_id]   # optional
    # To split one entry point into multiple intents, declare additional hint
    # tasks with the same file_path/entry but distinct names; the merge keeps
    # each as a separate TaskSpec.
```

### Merge semantics (`hints.py`)

- Load hints file if present (`resolve_paths` gains a `hints_yaml` property →
  `eval_dir / "ai-eval.hints.yaml"`).
- Each hint → a `DetectedTask` (`framework="hint"`) appended to `scan.tasks`
  **before** `build_with_engine`, so both `rules` and `slm`/`hybrid` see them.
- **Dedup:** if a hint and an AST task share the same `(file_path, entry)`,
  the AST task wins on auto-detection; the hint wins on `type`/`name` only if
  the AST detector didn't fire for that entry (i.e. hints fill gaps, they don't
  override successful detection). This keeps appointment_helper's auto-detected
  task from colliding with an accidental hint.
- `init` writes a commented-out template; `analyze` merges (preserving existing
  hints on `--merge-strategy keep`, the default).

## Detector ordering in `builtin_detectors`

```python
OpenAIResponsesDetector(),    # NEW — before chat (workflow takes precedence on tools=)
OpenAIToolsDetector(),        # existing
PGVectorDetector(),            # existing
LangGraphDetector(),           # NEW
OpenAIChatDetector(),          # existing
LangChainDetector(),           # existing
ChromaDBDetector(),            # existing
```

Ordering rationale: Responses-with-tools emits `workflow` before the chat
detector can claim it (mirrors existing tools-before-chat precedence).
LangGraph is independent (different framework string) but placed before
LangChain in case a repo imports both.

## Validation

1. **Unit tests** — `tests/test_openai_responses_detector.py`: synthetic ASTs for
   `responses.create(tools=...)` (→ workflow), `responses.create(...)` without
   tools (→ falls through to chat, not emitted here), and `chat.completions.create`
   (must NOT be claimed by the new detector). `tests/test_langgraph_detector.py`:
   `StateGraph`+`add_node`+`graph.invoke` → workflow; plain chain `invoke` →
   not emitted (owned by LangChainDetector).
2. **Hints tests** — `tests/test_hints_merge.py`: hints fill a gap (no AST
   match → hint task appears); AST wins on collision with same `(file_path,
   entry)`; per-intent split via multiple hints on one entry.
3. **Schema tests** — `tests/test_synthesize_workflow_metrics.py`:
   `_DEFAULT_METRICS["workflow"]` returns the two native metrics; a workflow-only
   repo classifies to `custom` (or the new literal if added).
4. **End-to-end on the real repo** — from `appointment_helper`:
   `uv run --cwd <AIAutoEvals> ai-eval --cwd <appointment_helper> analyze --diff
   --merge-strategy keep --rubric-engine rules`. Expect a non-empty `tasks` dict
   containing `conversation_workflow_service` (or the hint-named task) with
   `type: workflow`. Confirms the appointment_helper case now auto-detects.
5. **Regression** — run the existing detector test suite to ensure no detector
   ordering or partition regressions (chat/tools mutual exclusion holds).
6. **No false positives** — a fixture of a plain FastAPI endpoint with a
   `responses.create` *without* `tools=` must **not** produce a `workflow` task
   (stays `chat`-owned or unemitted).

## Risks

- **Responses-vs-chat partition drift:** the three OpenAI detectors (chat,
  tools, responses) must stay mutually exclusive. Mitigation: a shared
  `OPENAI_TOOL_KWARGS` check (already in `signatures.py:127`) — responses uses
  the same predicate as tools. Add a test asserting exactly one of the three
  fires per call.
- **LangGraph false positives:** `graph.invoke` is a common method name.
  Mitigation: require the `langgraph` import prefix (already gates `matches`)
  AND a `StateGraph(...)`/`.compile()` construction in the same enclosing def,
  not just any `.invoke()`.
- **Hints collide with hand-authored rubric:** the appointment_helper rubric is
  hand-populated. Hints must not overwrite it. Mitigation: hints only *add*
  tasks to the scan; `analyze`'s existing merge (`_merge_dicts`, KEEP default)
  already preserves user-edited `tasks`. Document that hints and the rubric
  are complementary, not conflicting, and that populated rubric entries win.
- **Metric applicability:** emitting only `task_completion` +
  `slot_filling_accuracy` (both native to `workflow`) avoids the
  `argument_accuracy`/`hallucination_rate` applicability mismatch the prior
  plan flagged. Users can add the cross-applicable metrics per-task.
- **`type: workflow` not in `ProjectType` literal:** `ProjectType`
  (`schema.py:13`) lists `custom`, `tool_calling`, `agent`, etc. but not
  `workflow`. `TaskType` is `str` (open), so per-task `type: workflow` is fine;
  only `_classify_project_type`'s repo-level value needs a decision (Open
  question 1).

## Open questions

1. **Repo-level `project_type` for an all-workflow repo.** `_classify_project_type`
   (`synthesize.py:54`) has no `workflow` outcome. Easiest: count workflow as
   tool-bearing → `tool_calling` (or `agent` if agentic). Cleaner: add
   `"workflow"` to the `ProjectType` Literal. **Recommendation:** add the
   literal `workflow` to `ProjectType` and return it when all tool-bearing
   tasks are `workflow`-typed (small, semantically correct change). Confirm at
   implementation time; if blocked, fall back to `custom`.
2. **SLM engine integration for hints.** In `hybrid`/`slm` mode the SLM owns
   final `type`. Hints should be treated as **authoritative** (not overridden by
   the SLM) since they're explicit user declarations. Confirm whether
   `build_rubrics_slm` (`ai_eval/inference/slm/builder.py`) can accept
   hint-flagged tasks as fixed-type; if not, hints set the type and the SLM only
   refines purpose/inputs/outputs/metrics. **Recommendation:** hints pin
   `type`; SLM refines the rest. Resolve during implementation of `hints.py` ↔
   `builder.py` wiring.
3. **Hints file name/location.** Chosen `eval/ai-eval.hints.yaml` (sibling of
   `rubrics.yaml`). Confirm vs. an alternative like embedding hints inside
   `rubrics.yaml` itself. **Recommendation:** separate file — keeps
   `rubrics.yaml` as the eval *config* (metrics/judges) and the hints file as
   the *detection* declaration; cleaner separation and the merge logic stays
   simple.

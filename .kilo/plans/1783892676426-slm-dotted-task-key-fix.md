# Fix: SLM rubric engine emits non-snake_case task keys

## Problem

`ai-evals analyze --rubric-engine slm` (and hybrid) fails with:

```
1 validation error for RubricsConfig
tasks
  Value error, task name 'DocumentVectorDAO.search_similar_vectors' must be snake_case alphanumeric
```

The SLM rubric engine builds the `RubricsConfig.tasks` dict keys from
`DetectedTask.name` verbatim. For class methods the detectors emit a dotted
`Class.method` name (correct for `TaskSpec.entry`, illegal as a rubrics key).
`RubricsConfig._validate_task_names` (`ai_eval/config/schema.py:185-193`)
rejects any key that isn't snake_case alphanumeric, so `build_rubrics_slm`
raises before producing rubrics.

The rules engine already solves this: `_rubric_key_name` in
`ai_eval/inference/synthesize.py:119-139` collapses `Class.method` →
`class_method` (CamelCase class → snake_case, leading `_` on private methods
stripped), keeping the dotted form only in `TaskSpec.entry`. The SLM engine
imports `_unique_name` from synthesize but **not** `_rubric_key_name`, so the
collapse never runs on the SLM path.

## Root cause — two failure sites in `ai_eval/inference/slm/builder.py`

- **Line 384** — main task loop:
  `name = _unique_name(task.name, used_names)` — `task` is a `DetectedTask`
  whose `.name` may be dotted.
- **Line 401** — empty-recover loop:
  `name = _unique_name(named.name, used_names)` — `named` is an
  `_SLMTaskNamed` whose `.name` is free-form SLM-proposed text that may also
  contain dots.

Neither site applies the dotted-name collapse, so dotted names reach
`RubricsConfig` and fail validation.

## Fix

Extract a pure collapse helper from `_rubric_key_name` and reuse it from both
engines. One canonical rule; rules-engine behavior unchanged.

### 1. `ai_eval/inference/synthesize.py`

- Extract the body of `_rubric_key_name` into a pure helper:

  ```python
  def _collapse_dotted_name(name: str, entry: str | None) -> str:
      """Collapse a dotted ``Class.method`` name to a snake_case-alphanumeric
      rubrics key. Bare module-level names pass through unchanged."""
      if entry and "." in entry:
          cls, _, method = entry.rpartition(".")
          cls_part = _camel_to_snake(cls.replace(".", "_"))
          method_part = method.lstrip("_")
          name = f"{cls_part}_{method_part}" if method_part else cls_part
      return name
  ```

- Rewrite `_rubric_key_name(task: DetectedTask) -> str` to delegate:

  ```python
  def _rubric_key_name(task: DetectedTask) -> str:
      return _collapse_dotted_name(task.name, task.entry)
  ```

  Keep its existing docstring (it documents the contract for callers). Export
  `_collapse_dotted_name` is private (underscore); import it explicitly in
  builder.

### 2. `ai_eval/inference/slm/builder.py`

- Update the import (line 50):

  ```python
  from ai_eval.inference.synthesize import _collapse_dotted_name, _unique_name
  ```

- **Line 384** (main loop):

  ```python
  key_name = _collapse_dotted_name(task.name, task.entry)
  name = _unique_name(key_name, used_names)
  ```

- **Line 401** (empty-recover loop):

  ```python
  key_name = _collapse_dotted_name(named.name, named.entry)
  name = _unique_name(key_name, used_names)
  ```

  `named.entry` is the `_SLMTaskNamed.entry` field (already present), which is
  the correct analogue of `DetectedTask.entry`.

No change to `_unique_name`, no change to `RubricsConfig` validation, no change
to the rules-engine call path.

## Why this is correct

- `TaskSpec.entry` is unchanged — the resolvable dotted form survives in the
  spec, matching the rules-engine contract (dotted `entry`, snake_case key).
  `test_runner_dotted_entry.py` and `test_synthesize_workflow_metrics.py`
  already assert the dotted-entry contract; the SLM path now matches it.
- The collapse is deterministic and idempotent: a bare snake_case name
  (no `.` in entry, or entry is None) passes through unchanged, so
  non-dotted SLM output is unaffected.
- `_unique_name` still runs after the collapse, so dedup collisions across
  two methods that collapse to the same key are still resolved with the
  `_2`, `_3` suffix scheme.

## Validation

1. **Reproduce-then-fix test** — add to `tests/test_slm_builder.py`:

   ```python
   def test_slm_collapses_dotted_class_method_key(tmp_path, monkeypatch):
       # DetectedTask with dotted name + entry, like a class method
       scan = _scan([_task(name="DocumentVectorDAO.search_similar_vectors",
                           entry="DocumentVectorDAO.search_similar_vectors",
                           file_path="src/dao.py")])
       # inject a stub complete_fn returning a chat classification
       ...
       rubrics, stats, report = build_rubrics_slm(
           scan, project_root=tmp_path, complete_fn=stub, ...)
       assert "document_vector_dao_search_similar_vectors" in rubrics.tasks
       spec = rubrics.tasks["document_vector_dao_search_similar_vectors"]
       assert spec.entry == "DocumentVectorDAO.search_similar_vectors"
   ```

   The test must fail before the fix (raising the
   `task name '...' must be snake_case alphanumeric` validation error) and
   pass after.

2. **Empty-recover dotted-name test** — cover the `builder.py:401` path with a
   stub `complete_fn` whose `_SLMRecovery.tasks[0]` has a dotted `name` and a
   matching dotted `entry`; assert the collapsed key appears and `entry` is
   preserved. This guards the second site the original bug could recur at.

3. **No regression on rules engine** — run existing tests:
   `pytest tests/test_synthesize_workflow_metrics.py tests/test_runner_dotted_entry.py -q`
   (these exercise `_rubric_key_name` / dotted entry) plus
   `pytest tests/test_slm_builder.py -q`.

4. **Full SLM suite** — `pytest tests/test_slm_builder.py tests/test_callgraph.py tests/test_integration_basic_rag.py -q`.

5. **Manual smoke** — re-run the failing command against the original repo:
   `ai-evals analyze --rubric-engine slm` (and `hybrid`) should now produce a
   `rubrics.yaml` whose `tasks:` keys are snake_case and whose entries retain
   dotted `entry` values.

## Out of scope

- No relaxation of `RubricsConfig._validate_task_names` — keep strict validation
  as the safety net; the fix is to stop emitting illegal keys, not to accept
  them.
- No change to `ai-evals.hints.yaml` `HintTaskSpec._validate_name` — hints are
  author-authored and already validated; not on the SLM emit path.
- No change to `_unique_name` semantics.

## Files touched

- `ai_eval/inference/synthesize.py` — extract `_collapse_dotted_name`, delegate
  from `_rubric_key_name`.
- `ai_eval/inference/slm/builder.py` — import `_collapse_dotted_name`, apply at
  lines 384 and 401 (both call sites).
- `tests/test_slm_builder.py` — add the two reproduce-then-fix tests above.

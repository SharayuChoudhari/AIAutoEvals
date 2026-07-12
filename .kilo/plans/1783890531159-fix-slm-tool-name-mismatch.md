# Fix: "Tool name does not match" on `ai-evals init` (weak ollama SLM)

## Problem

`ai-evals init` (default `--rubric-engine hybrid`, default model
`ollama/qwen2.5-coder:7b`) fails with `Tool name does not match` and exits 1.

### Root cause (confirmed)

- `ai_eval/inference/slm/client.py:52` builds the instructor client with the
  default mode (`Mode.TOOLS`), which registers a single function-call tool
  whose name is derived from the Pydantic response model (`_SLMTask` for the
  per-task classify call, `_SLMRecovery` for the empty-evidence call).
- The 7B model is too weak for strict OpenAI-style tool calling: instead of
  echoing the registered tool name, it hallucinated a semantically-named tool
  `_create_workflow` (drawn from prompt content about workflows).
- ollama / instructor rejects the mismatched name with `Tool name does not
  match`; instructor retries (`failed_attempts`); after `caps.retries` (2)
  failures, `client.py:62` wraps the exception into `RubricEngineError`, and
  `fail_engine_error` (`rubric_engine.py:142`) exits 1.
- This is a **model-capability failure**, not a logic bug — but the engine
  turns it into a hard, all-or-nothing failure with only a generic
  "use `--rubric-engine rules`" hint.

### Evidence

- `<completion>` shows ollama returned
  `tool_calls=[...name='_create_workflow'...]` with `finish_reason='tool_calls'`.
- `<exception>Tool name does not match</exception>` follows, inside
  `<failed_attempts>` — instructor's retry loop.
- The string "Tool name does not match" is NOT present in the repo, instructor
  1.15.3, or litellm source — it is produced by the ollama server / provider
  layer when the emitted tool name is not in the registered tools list.
- `client.py:54-61` calls `client.chat.completions.create(...)` with no `mode=`
  override, so `from_litellm(litellm.completion)` (default `Mode.TOOLS`) wins.

## Decision

Switch the instructor client to **`Mode.JSON_SCHEMA`** (`json_schema_mode`).

- Sends a strict JSON schema via `response_format` instead of a function-call
  tool, so there is **no tool name for the model to mismatch**.
- ollama + qwen2.5-coder support `response_format={"type":"json_object"}` and
  json_schema; litellm maps `Mode.JSON_SCHEMA` to the provider-appropriate
  structured-output call.
- Keeps the single SLM call per task (no extra round-trips), preserves the
  existing Pydantic validation contract, and stays compatible with cloud
  providers (OpenAI, Anthropic, etc.) that litellm routes through.

Rejected alternatives (see Q&A):
- *Per-task fallback to rules on mismatch*: still wastes the failed call and
  silently degrades rubrics.
- *Retry with Mode.JSON on mismatch*: more complex, needs mismatch detection.
- *Better error message only*: no robustness gain.

## Affected boundaries

- `ai_eval/inference/slm/client.py` — primary change.
- `ai_eval/inference/slm/builder.py` — no change (response models unchanged).
- `ai_eval/inference/slm/prompts.py` + prompt templates — no change.
- `ai_eval/cli/rubric_engine.py`, `ai_eval/cli/init.py` — no change (error
  surfacing path stays as the last-resort fallback).
- Tests: `tests/test_slm_builder.py`, `tests/test_cli_init.py`.

## Data flow (unchanged except the wire format)

```
build_rubrics_slm
  -> _classify_one_task / _recover_empty
       -> do_complete(model, messages, response_model=_SLMTask|_SLMRecovery, ...)
            -> complete(...)                       [client.py]
                 -> instructor.from_litellm(litellm.completion, mode=Mode.JSON_SCHEMA)
                 -> client.chat.completions.create(model, messages,
                        response_format=<json_schema derived from response_model>)
                 -> instructor validates JSON -> returns _SLMTask instance
       -> cache.put / fallback guards (unchanged)
```

## Implementation steps

1. **`ai_eval/inference/slm/client.py`**
   - Import `instructor.Mode`.
   - In `complete(...)`, construct the client with `mode=instructor.Mode.JSON_SCHEMA`:
     `instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON_SCHEMA)`.
   - Keep the existing `client.chat.completions.create(...)` call and all
     kwargs (`response_model`, `temperature`, `max_tokens`, `max_retries`).
   - Leave the `RubricEngineError` wrapping in the `except Exception` block
     intact — genuine provider/schema failures still hard-fail with the same
     remediation hint.
   - Add a module-level constant `_INSTRUCTOR_MODE =
     instructor.Mode.JSON_SCHEMA` so tests and future tuning can reference it.

2. **No prompt changes.** The prompt already says "Return ONLY the structured
   object" — `Mode.JSON_SCHEMA` reinforces this at the wire layer. The
   `classify_task.txt` / `empty_recover` templates and `few_shot.yaml` are
   untouched.

3. **Tests**
   - `tests/test_slm_builder.py`: existing fake `complete_fn` injectors bypass
     the real client, so they keep passing unchanged. Add one new test that
     asserts the *real* `complete` builds its instructor client with
     `Mode.JSON_SCHEMA` (monkeypatch `instructor.from_litellm` to capture the
     `mode=` kwarg, then call `complete(...)` with a stubbed
     `litellm.completion`).
   - `tests/test_cli_init.py`: the existing
     `test_init_slm_engine_hard_fails_when_model_unreachable` still holds
     (provider failure path unchanged). No new CLI test required — the mode
     switch is exercised at the client unit level.

4. **Validation**
   - `uv run pytest tests/test_slm_builder.py tests/test_cli_init.py -q`
   - `uv run ruff check ai_eval/inference/slm/client.py`
   - `uv run mypy ai_eval/inference/slm/client.py`
   - Manual smoke (if ollama available):
     `ai-evals init --rubric-engine hybrid` against a repo with an OpenAI
     chat call site; confirm rubrics.yaml is written and no
     "Tool name does not match" appears.

## Failure modes & rollback

- If a provider does not support `Mode.JSON_SCHEMA`, instructor/litellm raises
  a structured-output error, which is still caught by the existing
  `except Exception` → `RubricEngineError` path and surfaced with the
  "use `--rubric-engine rules`" hint. No silent corruption.
- Rollback: revert `client.py` to the default-mode construction (one line).

## Out of scope

- Changing the default model away from `ollama/qwen2.5-coder:7b`.
- Adding a `--rubric-mode` CLI flag (the mode is an engine-internal detail).
- Tightening `fail_engine_error` messaging for the tool-name-mismatch
  signature (now structurally impossible under JSON_SCHEMA; if it recurs via
  a different provider quirk, the generic hint still applies).

# Fix: IO-coupled dotted-task examples fail on `ai-evals run`

## Problem

After `ai-evals init` (which generates `golden_set.json` with auto-seeded
examples), `ai-evals run` fails every auto-seeded example for IO-coupled
`Class.method` task entries. Three distinct failure modes appear:

```
✗ single_query_evaluator_evaluate_single_auto_*:
    TypeError: SingleQueryEvaluator.__init__() missing 1 required positional argument: 'config'
✗ chat_message_service_create_workflow_auto_*:
    TypeError: ChatMessageService.__init__() missing 1 required positional argument: 'session'
✗ document_vector_d_a_o_search_similar_vectors_auto:
    ModuleNotFoundError: No module named 'layers'
```

The per-task metric table shows `status: - skip` for every judge metric and
`p50=0.0` latency — no example ever ran; every example errored before the
task method executed, so there are zero scores to aggregate. The auto-seed
notice ("scored against auto-seeded fixtures; run `ai-evals bootstrap …`")
prints correctly, but the examples themselves never execute.

Confirmed by reproduction: the app these tasks come from runs normally, so
`layers` *is* importable in the app's normal runtime — the failures are
specific to `ai-evals run`'s environment setup.

## Root cause — three bugs in `ai_eval/runner/engine.py`

All three are on the dotted `Class.method` instance-construction path.

### Bug 1 — `sys.path` not set up before the harness loads the task module

`ModuleNotFoundError: No module named 'layers'`.

`ai-evals` is an installed console script (`ai-evals = "ai_eval.cli.app:main"`
in `pyproject.toml`), so unlike `python script.py`, the current working
directory is **not** automatically on `sys.path`. The task module
(`svc.py`) does `from layers.dao import DocumentVectorDAO` at module top
level; `layers` lives under the project root and is importable only when
the project root is on `sys.path`.

`_run_example` (engine.py:354-364) runs, in order:
1. `_load_harness(tname, project_root)` — line 359 — which calls the
   harness `install()` → `_load_task_module()` → `spec.loader.exec_module`
   (generated code, harness_writer.py:249-259). This executes the task
   module top-level imports.
2. `_import_entry(tspec, project_root)` — line 363 — which is where
   `sys.path` gets `cwd` + `file_path.parent` inserted (engine.py:138-140).

So the harness loads the task module **before** `sys.path` is set up, and
`from layers.dao import …` fails. The generated harness `_load_task_module`
does no `sys.path` manipulation of its own, so it can't recover.

Reproduced: with `PYTHONSAFEPATH=1` (mimicking a console-script entrypoint
where cwd ∉ sys.path), `exec_module` on a file with
`from layers.dao import …` raises `ModuleNotFoundError`; inserting the
project root onto `sys.path` first makes the same load succeed.

### Bug 2 — `self` not excluded in `_fake_call_args` (engine.py:49-81)

`TypeError: … missing 1 required positional argument: 'config'`.

`inspect.signature(cls.__init__)` returns the signature **including
`self`** as the first parameter. `_fake_call_args` does not skip `self`,
so it fabricates an arg for it. For `SingleQueryEvaluator.__init__(self,
config)` (both params unannotated):

- fabricated `args = [None, None]` (one for `self`, one for `config`)
- `cls(None, None)` → Python adds the real `self` implicitly → 3 args to a
  2-parameter `__init__` → `TypeError: takes 2 positional arguments but 3
  were given`
- the `except TypeError` fallback at engine.py:175-177 runs `cls()` (no
  args) → `TypeError: missing 1 required positional argument: 'config'`

That fallback message is exactly what the run output shows. The inline
comment at engine.py:172 ("Drop 'self' from the fake-args construction")
states the intent but `_fake_call_args` never implements it.

### Bug 3 — required non-primitive args become `None`

Even after Bug 2 is fixed, params annotated with a custom class
(`config: Config`, `session: Session`) or unannotated required params get
`None` (engine.py:79-80). The constructor or method body accesses
attributes on them (`self.session.add(...)`, `config.model`) →
`AttributeError`/`TypeError`. Reproduced: `ChatMessageService(None)` →
`create_workflow` calls `self.session.add(...)` → `AttributeError`.

### Why existing tests didn't catch any of this

`tests/test_runner_dotted_entry.py` only uses `__init__(self)` or
`__init__(self, name='')` (with a default), and puts the DAO class in the
**same file** as the service (no cross-module import). With Bug 2, the
single fabricated `self`→`None` is passed positionally as `name`, which
works by accident; Bug 1 never triggers because there's no
`from <pkg> import …`; Bug 3 never triggers because there's no required
non-primitive arg. No test exercises the real-repo shape: cross-package
imports + a `__init__` with a required non-`self` parameter.

## Scope of fix

The run engine's dotted-task path in `ai_eval/runner/engine.py`. The
auto-seed notice stays (correct by design). The harness (D5) codegen and
the seeder (D6) are unchanged — the harness-generated
`_load_task_module` keeps working once `sys.path` is correct; no harness
file regeneration needed. The fix makes auto-seeded IO-coupled examples
actually *execute* (green pipeline) instead of erroring pre-execution.

Out of scope: relaxing metric thresholds; changing the bootstrap capture
flow; touching `_import_entry`'s harness-reuse module-scanning logic
(engine.py:147-160) — that correctly reuses the patched module so D5
monkey-patches survive; injecting fake modules into `sys.modules`
(rejected — masks real import errors and isn't needed once `sys.path` is
correct).

## Fix

All changes in `ai_eval/runner/engine.py`.

### 1. Extract `sys.path` setup into a helper; call it before the harness load

Extract the existing `sys.path` insertion block (engine.py:138-140) into a
small helper, e.g. `_ensure_task_on_syspath(file_path, cwd)`, and:

- Call it at the **start of `_load_harness`** (before `install()`), so the
  harness's `_load_task_module()` → `exec_module` sees the project root on
  `sys.path` and the task module's top-level `from layers.dao import …`
  resolves. `_load_harness` receives `cwd` already (engine.py:359 passes
  `project_root`); it needs the task's `file_path` too — pass `tspec` (or
  just `file_path`) into `_load_harness`.
- Keep the existing call inside `_import_entry` (now via the helper) so
  bare-fn entries and the dotted path both stay covered. Idempotent
  (guarded by `if p not in sys.path`), so calling it twice is harmless.

This resolves Bug 1 without changing the generated harness code: the
harness's `_load_task_module` works once `sys.path` is correct, and
`_import_entry`'s existing module-reuse scan (engine.py:147-153) still
finds the already-loaded patched module.

### 2. `_fake_call_args` — skip `self` (engine.py:49-81)

Skip the first parameter whose name is `self` (the bound-signature case).
`inspect.signature(cls.__init__)` always lists `self` first for a method;
`inspect.signature(cls)` would drop it but walks the MRO and returns
`*args/**kwargs` for many classes (dataclasses, pydantic models), so keep
`cls.__init__` and skip `self` explicitly by name. Keep the existing
`VAR_POSITIONAL`/`VAR_KEYWORD` and has-default skips.

This implements the already-present comment at engine.py:172.

### 3. `_fake_call_args` — fabricate a `_Stub` for required non-primitive args

For a required (no-default) non-`self` param whose annotation is not a
primitive (`int`/`float`/`bool`/`str`), fabricate a `_Stub()` instead of
`None`. This covers custom-class annotations (`config: Config`), string
annotations (`from __future__ import annotations`), typing constructs, and
unannotated params (`session`) uniformly — no annotation resolution or
`typing.get_type_hints` needed.

Do **not** attempt to construct real instances of annotated types (the
"case 2" idea). Rationale: it adds side-effect risk (a zero-arg `Config()`
could open files/connect), requires annotation resolution that fails on
string/typing constructs, and gives no benefit for the green-pipeline goal
— the D5 harness patches the IO reads, and the `_Stub` only needs to
survive `__init__` and absorb attribute/call access in the method body.
Reproduced: `ChatMessageService(_Stub())` → `create_workflow('hello')`
returns `"workflow:hello"` (the `self.session.add(...)` call returns a
`_Stub`, harmless).

Keep the `except TypeError: cls()` fallback (engine.py:175-177) as a last
resort for metaclasses where the signature lies; with Bug 2 fixed it
rarely triggers.

### 4. Add module-private `_Stub` helper

```python
class _Stub:
    """Permissive stand-in for a required non-primitive constructor arg.

    Any attribute access or call returns a new _Stub so a task class's
    __init__ / method body that touches ``config.x`` or ``session.add(...)``
    doesn't raise. Used only for required non-primitive params (Bug 3)."""
    __slots__ = ()
    def __getattr__(self, name): return _Stub()
    def __call__(self, *a, **kw): return _Stub()
```

Deliberately not `unittest.mock.Mock` — no test-framework dependency, no
assertions, no import into the run path.

## Why this is correct

- **Bug 1**: the only reason `layers` is unimportable under `ai-evals run`
  is the missing `sys.path` entry; the app runs normally, so the package
  exists. Setting up `sys.path` before the harness load (not just before
  `_import_entry`) is the minimal fix and doesn't touch harness codegen.
- **Bug 2**: `self` must never be a fabricated arg — it's supplied
  implicitly by `cls(...)`. Skipping it by name is correct for
  `inspect.signature(cls.__init__)`.
- **Bug 3**: the D5 harness patches `self.<dao>.<method>()` reads to
  return canned fixtures, so the method body's external IO never runs
  regardless of what `config`/`session` are. The `_Stub` just needs to not
  blow up during `__init__` and any attribute/call access — which it
  guarantees by construction. Reproduced end-to-end.
- **Existing tests stay green**: `__init__(self)` / `__init__(self,
  name='')` → after Bug 2 fix, `self` is skipped and `name` has a default
  → no args fabricated → `cls()` → unchanged. Same-file DAO classes → Bug
  1's `sys.path` fix is a no-op (paths already absent or present
  idempotently). No required non-primitive args → Bug 3's `_Stub` path
  isn't exercised.

## Validation

1. **Reproduce-then-fix tests** — add to
   `tests/test_runner_dotted_entry.py` (must FAIL before the fix, PASS
   after). Each writes the task file under `tmp_path` and runs `execute`
   with `parallel=1`:

   - `test_dotted_entry_required_non_self_arg_constructs`: class
     `__init__(self, config)` with `config` unannotated, method returns a
     string. Assert no example errors, no "missing positional argument".
     (Pins Bug 2 + 3; mirrors `SingleQueryEvaluator`.)
   - `test_dotted_entry_stub_arg_absorbs_method_body_access`: class
     `__init__(self, session)`, method body calls
     `self.session.add(len(q))` then returns a string. Assert no example
     errors (the `_Stub` absorbs `session.add`). (Pins Bug 3; mirrors
     `ChatMessageService`.)
   - `test_dotted_entry_cross_package_import_loads`: a `layers/dao.py`
     subpackage under `tmp_path` with `class DocumentVectorDAO`, and a
     `svc.py` that does `from layers.dao import DocumentVectorDAO` at top
     level + `__init__(self, session)` + a `process` method reading
     `self.dao.search(q)`. Generate the harness via `render_harness` (as
     the existing `test_harness_loaded_for_io_coupled_dotted_entry` does)
     with a canned fixture for `('dao', 'search')`. Assert no example
     errors, no `ModuleNotFoundError`, and the canned fixture is returned.
     (Pins Bug 1 end-to-end; mirrors `document_vector_d_a_o`.) Note: this
     test must run with cwd-not-auto-on-sys.path semantics — use
     `monkeypatch.setattr(sys, "path", [...])` to drop `tmp_path` from
     `sys.path` before `execute`, so the bug actually triggers
     pre-fix (otherwise cwd-on-path masks it, as the existing tests do).

2. **`self`-exclusion unit test** — `__init__(self, a, b)` with two
   required non-self params; assert the constructed instance receives
   exactly two fabricated args (two `_Stub`s), not three. Pins Bug 2
   directly at the `_fake_call_args` level (can be a direct unit test of
   `_fake_call_args` + `_import_entry`, not a full `execute`).

3. **No regression on existing dotted tests** —
   `pytest tests/test_runner_dotted_entry.py -q` stays green.

4. **Harness/seeder interaction** —
   `pytest tests/test_harness_writer.py tests/test_seeder.py -q`
   unchanged.

5. **Full runner suite** —
   `pytest tests/test_runner_engine.py tests/test_runner_dotted_entry.py
   tests/test_runner_thresholds.py tests/test_integration_basic_rag.py -q`.

6. **Lint/type** — `ruff check ai_eval/runner/engine.py
   tests/test_runner_dotted_entry.py` and `mypy ai_eval/runner/engine.py`.

7. **Manual smoke** — in the original target repo, re-run `ai-evals run`
   after `ai-evals init`. Expect: auto-seeded IO-coupled examples now
   execute (latency > 0, no construction-time `TypeError` /
   `ModuleNotFoundError`). The auto-seed notice still prints (correct —
   these are still auto-seeds until `bootstrap` captures real traces).

## Files touched

- `ai_eval/runner/engine.py` — extract `_ensure_task_on_syspath` helper;
  call it at the start of `_load_harness` (Bug 1); fix `_fake_call_args`
  to skip `self` (Bug 2) and fabricate `_Stub` for required non-primitive
  args (Bug 3); add module-private `_Stub`.
- `tests/test_runner_dotted_entry.py` — add the three reproduce-then-fix
   tests + the `self`-exclusion unit test.

## Risks / trade-offs

- **`_Stub` masking real bugs**: a `_Stub` silently absorbs misuse, so a
  genuinely broken task could look green. Acceptable for auto-seeds (the
  notice already directs users to `bootstrap` for trustworthy baselines);
  real captures replace auto-seeds and exercise real objects. No change
  for captured examples.
- **`sys.path` mutation is global**: inserting `cwd`/`file_path.parent`
  onto `sys.path` is already done by `_import_entry`; the fix just does it
  earlier (before the harness load). It's idempotent and the existing
  behavior for bare-fn entries is unchanged. No new path entries beyond
  what `_import_entry` already adds.
- **`_load_harness` signature change**: it gains the task's `file_path`
  (or `tspec`) so it can call `_ensure_task_on_syspath`. Internal helper,
  only called from `_run_example` (engine.py:359) — single call site, low
  blast radius.

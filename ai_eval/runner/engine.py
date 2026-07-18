"""Run execution engine.

Async-native. Imports the user's task module, calls the entry symbol per golden
example, measures latency, runs judge metrics via the injected gateway, and
aggregates into a :class:`~ai_eval.runner.record.RunRecord`.

Concurrency: ``asyncio`` semaphore = ``parallel``. Judge calls for one example
run concurrently across metrics; examples within a task run concurrently up to
the semaphore.

Transport injection: ``gateway.score`` is the async judge gateway; tests pass a
fake ``complete_fn`` so no network is touched.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

from ai_eval.config.schema import MetricSpec, RubricsConfig, TaskSpec
from ai_eval.judge.cache import JudgeCache
from ai_eval.judge.gateway import score as judge_score
from ai_eval.judge.schemas import JudgeRequest
from ai_eval.runner.metrics.latency import compute_latencies
from ai_eval.runner.record import (
    ExampleRecord,
    MetricResult,
    RunRecord,
    RunSummary,
    TaskRecord,
)
from ai_eval.runner.thresholds import assert_metric_implemented, evaluate_metric


def config_hash(rubrics: RubricsConfig) -> str:
    payload = json.dumps(rubrics.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _build_call_args(fn, input_: Any) -> tuple[tuple, dict]:
    """Build ``(args, kwargs)`` to call ``fn`` with ``input_``.

    Signature-aware so the auto-seed convention (a scalar string input) works
    for entries of any arity:

    * ``input_`` is a ``dict``  → unpacked as kwargs (rich/golden-set case).
    * ``input_`` is a ``list``/``tuple`` → unpacked as positional args.
    * ``input_`` is a scalar (the D6 auto-seed case) → inspect ``fn``'s
      signature: map the scalar onto the first str-typed required param (or
      the first required param if none is str-typed). Required non-scalar
      params cannot be fabricated anymore (the ``_Stub`` path is removed —
      AGENTS.md §1), so a signature with required non-str params that the
      scalar doesn't bind to is a misconfiguration: the entry is IO-coupled
      or object-typed and should be demoted via ``node_metrics`` + bootstrap,
      not auto-seeded. Params with defaults are left to apply. If the
      signature can't be introspected (builtins/C callables) the scalar is
      passed as a single positional arg (legacy fast path).
    """
    if isinstance(input_, dict):
        return (), dict(input_)
    if isinstance(input_, (list, tuple)):
        return tuple(input_), {}
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return (input_,), {}

    params = list(sig.parameters.values())
    # User-supplied positional-or-keyword params (skip self, *args, **kwargs).
    slots: list[inspect.Parameter] = []
    for p in params:
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if p.name == "self":
            continue
        slots.append(p)

    if not slots:
        # No user params (e.g. ``_create_workflow(self)``) — don't pass the
        # scalar; calling ``fn("")`` would raise "too many positional args".
        return (), {}

    # Required params (no default) are the ones we must fill. Decide which of
    # them the scalar binds to: the first str-typed required param if any
    # (mirrors the auto-seed "single string input" intent), else the first
    # required param. Only that one is filled; the entry is expected to be a
    # legitimate pure-LLM top-level entry of arity 1 (the only runnable kind
    # now — AGENTS.md §1). Multi-required-param entries with non-str params
    # are IO-coupled/object-typed and should be demoted, not auto-seeded.
    required = [p for p in slots if p.default is inspect.Parameter.empty]
    scalar_idx = -1
    if required:
        scalar_idx = next(
            (i for i, p in enumerate(required) if p.annotation is str),
            0,
        )

    args: list = []
    filled_required = 0
    for p in slots:
        if p.default is not inspect.Parameter.empty:
            # Once we hit a defaulted param, stop appending — any later
            # required params would be a Python-level signature error anyway
            # (required-after-default is invalid), so this is safe.
            break
        if filled_required == scalar_idx:
            args.append(input_)
        else:
            # No _Stub fallback: a required param the scalar doesn't bind to
            # means the entry is not a pure-LLM top-level entry. Surface a
            # clear TypeError so the runner records a bootstrap-directed error
            # (AGENTS.md §1) instead of silently fabricating a placeholder.
            raise TypeError(
                f"cannot auto-bind required param {p.name!r} (annotation "
                f"{p.annotation!r}) from scalar input; this entry is "
                f"IO-coupled or object-typed — demote it via node_metrics "
                f"and run `ai-evals bootstrap` to capture a real trace"
            )
        filled_required += 1
    return tuple(args), {}


def _ensure_task_on_syspath(file_path: Path, cwd: Path) -> None:
    """Insert ``cwd`` and the task file's parent onto ``sys.path``.

    Idempotent (guarded by ``if p not in sys.path``), so calling it more than
    once (e.g. before the harness load *and* inside ``_import_entry``) is
    harmless. Needed for dotted entries where the task module does
    ``from <pkg> import ...`` against a subpackage under the project root —
    ``ai-evals`` runs as an installed console script, so the cwd isn't
    auto-added to ``sys.path`` (Bug 1).
    """
    for p in (str(cwd), str(file_path.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_harness(task_name: str, cwd: Path, file_path: Path) -> bool:
    """Import and install the stub harness for an IO-coupled task (D5/D7).

    Looks for ``eval/_harness_<safe_task_name>.py`` and calls its ``install()``
    so the task's ``self.<dao>.<method>()`` reads return canned fixtures. Returns
    True if a harness was installed, False if none exists (caller decides whether
    to skip). Safe to call repeatedly: harness install is idempotent.

    Sets up ``sys.path`` *before* loading the harness so the harness's
    ``_load_task_module`` → ``exec_module`` can resolve the task module's
    top-level cross-package imports (Bug 1).
    """
    import re

    _ensure_task_on_syspath(file_path, cwd)
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", task_name)
    harness_path = cwd / "eval" / f"_harness_{safe}.py"
    if not harness_path.is_file():
        return False
    # Include the cwd in the module name so different repos (same task name)
    # get distinct harness module instances — a prior test's harness must not
    # be reused (it patched a different target module instance).
    import hashlib

    cwd_hash = hashlib.sha1(str(cwd).encode()).hexdigest()[:8]
    mod_name = f"_ai_eval_harness_{safe}_{cwd_hash}"
    # Clear any stale harness/target modules from a prior run so install()
    # patches the freshly-loaded target for THIS repo.
    for stale in (mod_name, "_ai_eval_harness_target"):
        sys.modules.pop(stale, None)
    spec = importlib.util.spec_from_file_location(mod_name, harness_path)
    if spec is None or spec.loader is None:
        return False
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    install = getattr(mod, "install", None)
    if callable(install):
        install()
    return True


def _import_entry(task_spec: TaskSpec, cwd: Path):
    """Import ``task_spec.file_path`` and resolve its ``entry`` symbol.

    For dotted ``Class.method`` entries (D2/D7): constructs an instance of the
    class. Constructor args now come ONLY from the harness (if present — the
    harness monkey-patches ``self.<dao>.<method>()`` reads to return canned
    fixtures) or a no-arg ``cls()`` fallback. The ``_Stub``/``_fake_call_args``
    path is removed (AGENTS.md §1): if construction fails with ``TypeError``
    (required args, no harness), the caller records an ``error``-status
    ExampleRecord with a bootstrap-directed message — the entry point is
    IO-coupled and needs a harness or a real backend.

    Inserts ``cwd`` and the file's parent onto ``sys.path`` first. Returns the
    callable (a bound method for dotted entries), or raises.
    """
    file_path = Path(task_spec.file_path)
    if not file_path.is_absolute():
        file_path = (cwd / task_spec.file_path).resolve()
    module_name = "_ai_eval_task_" + file_path.stem
    # Insert cwd + file parent on sys.path so relative imports in the task work.
    _ensure_task_on_syspath(file_path, cwd)
    # Dotted entries may have a harness that loaded + patched the target
    # module under a different name; reuse that patched module so the
    # monkey-patches survive. Bare ``fn`` entries always load fresh (fast path:
    # no sys.modules scan).
    entry = task_spec.entry or "main"
    mod = None
    if "." in entry:
        target = str(file_path)
        for existing in list(sys.modules.values()):
            ef = getattr(existing, "__file__", None)
            if ef and str(Path(ef).resolve()) == target:
                mod = existing
                break
    if mod is None:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load module from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # Dotted ``Class.method``: resolve the class, construct an instance with
    # no args (the harness, if installed, has patched the DAO reads so a
    # bare ``cls()`` works for IO-coupled entries whose ``__init__`` only
    # needs a session/dao). If ``__init__`` requires real args and no harness
    # supplies them, this raises ``TypeError`` — surfaced as a
    # bootstrap-directed error by the runner (AGENTS.md §1). Bare ``fn``:
    # return the function.
    if "." in entry:
        cls_name, _, method_name = entry.rpartition(".")
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise AttributeError(f"class {cls_name!r} not found in {file_path}")
        instance = cls()
        fn = getattr(instance, method_name, None)
        if fn is None:
            raise AttributeError(f"method {method_name!r} not found on {cls_name!r} in {file_path}")
        return fn
    fn = getattr(mod, entry, None)
    if fn is None:
        raise AttributeError(f"entry symbol {entry!r} not found in {file_path}")
    return fn


def _call_entry(fn, input_: Any):
    """Call the task entry symbol with ``input_``.

    Signature-aware (see :func:`_build_call_args`): a scalar auto-seed input is
    mapped onto the entry's first suitable required param, with placeholders
    fabricated for the remaining required params, so entries of any arity
    (zero-arg, single-arg, multi-arg) run without ``TypeError``. A ``dict``
    input is unpacked as kwargs; a ``list``/``tuple`` as positional args.

    Handles ``async def`` entries by running them in a dedicated event loop
    (the handover flagged returning a raw coroutine as a silent garbage-score
    failure).
    """
    args, kwargs = _build_call_args(fn, input_)
    result = fn(*args, **kwargs)
    # Async entries return a coroutine — await it in a fresh loop. This runs
    # inside asyncio.to_thread (a worker thread), so a nested event loop is
    # safe here (the outer loop is on the main thread).
    if inspect.isawaitable(result):
        import asyncio as _asyncio

        try:
            loop = _asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            return _asyncio.run(result)  # type: ignore[arg-type]
        # We're somehow inside a running loop on this thread — schedule it.
        return loop.run_until_complete(result)  # type: ignore[arg-type]
    return result


async def execute(
    rubrics: RubricsConfig,
    golden_set: dict[str, list[dict]],
    *,
    project_root: Path,
    complete_fn=None,
    parallel: int = 4,
    cache_on: bool = True,
    task_filter: tuple[str, ...] = (),
    judge_override: str | None = None,
    tolerance: float = 0.02,
    fail_on_regression: bool = False,
    baseline: dict[str, Any] | None = None,
    run_id: str,
    started_at: float,
    tags: list[str] | None = None,
    config_hash_value: str | None = None,
    baseline_id: str | None = None,
) -> RunRecord:
    """Execute all tasks and return a :class:`RunRecord`.

    ``complete_fn`` is the async judge transport (fake in tests). ``baseline``
    is the prior run's tasks dict (for delta computation); ``None`` = fresh.
    """
    cache = JudgeCache(project_root / ".ai-evals", enabled=cache_on)
    sem = asyncio.Semaphore(parallel)
    tasks_out: dict[str, TaskRecord] = {}
    default_model = judge_override or rubrics.judge.default
    fallback = tuple(rubrics.judge.fallback)
    complex_models = (
        tuple(rubrics.judge.tiering.complex_models) if rubrics.judge.tiering is not None else None
    )

    task_names = list(rubrics.tasks.keys())
    if task_filter:
        task_names = [n for n in task_names if n in task_filter]

    for tname in task_names:
        tspec = rubrics.tasks[tname]
        record = TaskRecord()
        # Non-top-level tasks (internal DAOs/services, private methods,
        # IO-coupled sub-workflows) are skipped by design (AGENTS.md §1):
        # scaffolded into rubrics.yaml but not auto-seeded/run. Record a
        # task-level notice directing the user to `ai-evals bootstrap`.
        if not tspec.top_level:
            record.errors.append(
                f"{tname}: skipped (non-top-level internal task); "
                f"run `ai-evals bootstrap` for trustworthy baselines"
            )
            tasks_out[tname] = record
            continue
        examples = golden_set.get(tname, [])
        # Validate all metrics are implemented up-front (run-time strict gate).
        # ``project_root`` resolves project-local ``eval/metrics.yaml`` metrics.
        metric_impls = {}
        for mspec in tspec.metrics:
            metric_impls[mspec.name] = assert_metric_implemented(
                mspec.name, project_root=project_root
            )
        # Node metrics (AGENTS.md §1) are validated on the same surface so a
        # typo'd metric name in ``node_metrics`` fails fast at run start, not
        # silently per-example. The cache on ``tspec`` is primed here too.
        if tspec.node_metrics:
            _node_metric_impls(tspec, project_root)

        latencies: list[float] = []

        async def run_one(example: dict, tspec=tspec, tname=tname, metric_impls=metric_impls):
            async with sem:
                return await _run_example(
                    example,
                    tspec=tspec,
                    tname=tname,
                    default_model=default_model,
                    fallback=fallback,
                    complex_models=complex_models,
                    complete_fn=complete_fn,
                    cache=cache,
                    project_root=project_root,
                    metric_impls=metric_impls,
                )

        results = await asyncio.gather(*(run_one(ex) for ex in examples), return_exceptions=True)
        for idx, res in enumerate(results):
            ex = examples[idx] if idx < len(examples) else {}
            ex_id = ex.get("id") or f"{tname}_{idx}"
            if isinstance(res, Exception):
                record.errors.append(f"{ex_id}: {type(res).__name__}: {res}")
                record.examples.append(
                    ExampleRecord(
                        id=ex_id,
                        status="error",
                        error=str(res),
                        seed=ex.get("seed"),
                    )
                )
                continue
            assert isinstance(res, ExampleRecord)
            ex_rec = res
            record.examples.append(ex_rec)
            if ex_rec.status == "error":
                record.errors.append(f"{ex_id}: {ex_rec.error}")
            latencies.append(ex_rec.latency_ms)

        # Aggregate per-metric scores across examples (mean of non-error).
        record.metrics = _aggregate_metrics(
            tspec, record.examples, baseline, tname, tolerance, fail_on_regression
        )
        record.latency_ms = compute_latencies(
            [e.latency_ms for e in record.examples if e.status != "error"]
        )
        tasks_out[tname] = record

    summary = _summarize(tasks_out)
    return RunRecord(
        id=run_id,
        started_at=started_at,
        finished_at=time.time(),
        config_hash=config_hash_value or config_hash(rubrics),
        baseline_id=baseline_id,
        tasks=tasks_out,
        summary=summary,
        tags=list(tags or []),
        extras={
            "parallel": parallel,
            "cache_on": cache_on,
            "cache_stats": {"hits": cache.stats.hits, "misses": cache.stats.misses},
        },
    )


def _select_nodes(calls: list, selector: str) -> list[tuple[str, dict]]:
    """Match trace nodes against a selector clause.

    Returns ``[(node_id, call_dict), ...]`` in call order. ``node_id`` is a
    synthetic ``<kind>_<i>`` (e.g. ``retrieve_0``) so the per-node rollup is
    keyed consistently across examples. A selector may match zero, one, or
    many nodes — each match is scored independently.

    Selector grammar (single clause, comma-free):

    * ``kind=<x>`` — ``call["kind"] == x`` (e.g. ``kind=retrieve``).
    * ``name=<exact>`` — exact match on ``call["name"]``.
    * ``name~=<substr>`` — substring match on ``call["name"]``.
    * ``call_index=<n>`` — the n-th call (0-based), regardless of kind/name.

    Unknown clauses or malformed selectors match nothing (the node pass is a
    no-op, surfaced as ``skip`` at aggregate time). This is the honest
    behavior: a selector authored against a stale trace shape doesn't crash
    the run, it just scores no nodes.
    """
    matches: list[tuple[str, dict]] = []
    if not selector or "=" not in selector:
        return matches
    key, _, value = selector.partition("=")
    key = key.strip()
    value = value.strip()
    kind_counters: dict[str, int] = {}
    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            continue
        if key == "call_index":
            try:
                if int(value) != i:
                    continue
            except ValueError:
                continue
        elif key == "kind":
            if call.get("kind") != value:
                continue
        elif key == "name":
            if call.get("name") != value:
                continue
        elif key == "name~":
            name = call.get("name")
            if not isinstance(name, str) or value not in name:
                continue
        else:
            continue
        kind_val = call.get("kind")
        kind = kind_val if isinstance(kind_val, str) else "node"
        idx = kind_counters.get(kind, 0)
        kind_counters[kind] = idx + 1
        node_id = f"{kind}_{idx}"
        matches.append((node_id, call))
    return matches


async def _run_example(
    example: dict,
    *,
    tspec: TaskSpec,
    tname: str,
    default_model: str,
    fallback: tuple[str, ...],
    complex_models: tuple[str, ...] | None,
    complete_fn,
    cache: JudgeCache,
    project_root: Path,
    metric_impls: dict,
) -> ExampleRecord:
    ex_id = example.get("id") or f"{tname}_{hash(str(example.get('input'))) & 0xFFFF:x}"
    input_ = example.get("input")
    seed = example.get("seed")  # "auto" for D6 auto-seeded examples
    t0 = time.perf_counter()
    output: Any = None
    try:
        # IO-coupled tasks: install the stub harness (D5/D7) before importing
        # the entry so the instance's self.<dao>.<method>() reads return canned
        # fixtures. No-op for pure-LLM tasks (no harness file on disk).
        if "." in (tspec.entry or ""):
            _fp = Path(tspec.file_path)
            if not _fp.is_absolute():
                _fp = (project_root / tspec.file_path).resolve()
            await asyncio.to_thread(_load_harness, tname, project_root, _fp)
        # Offload the (sync) task import + call to a thread so the event loop
        # isn't blocked — parallel examples can then overlap. The import is
        # memoized-ish per module name, and the call is the user's own code.
        fn = await asyncio.to_thread(_import_entry, tspec, project_root)
        output = await asyncio.to_thread(_call_entry, fn, input_)
    except TypeError as exc:
        # Construction or call failed on a required arg (no _Stub fallback —
        # AGENTS.md §1). The entry is IO-coupled or object-typed: it needs a
        # harness (``ai-evals init`` writes one for detected IO reads) or a
        # real backend. Direct the user to bootstrap rather than leaving a
        # bare ``TypeError: missing N required positional arguments``.
        msg = str(exc)
        is_arg_error = (
            "required positional argument" in msg
            or ("missing" in msg.lower() and "argument" in msg.lower())
            or "cannot auto-bind required param" in msg
        )
        hint = (
            " — this entry is IO-coupled or object-typed; run "
            "`ai-evals bootstrap` to capture a real trace, or add a "
            "harness in eval/_harness_<task>.py"
            if is_arg_error
            else ""
        )
        return ExampleRecord(
            id=ex_id,
            status="error",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {msg}{hint}",
            seed=seed,
        )
    except Exception as exc:
        return ExampleRecord(
            id=ex_id,
            status="error",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"{type(exc).__name__}: {exc}",
            seed=seed,
        )
    latency_ms = (time.perf_counter() - t0) * 1000.0

    metric_scores: dict[str, float] = {}
    metric_errors: dict[str, list[str]] = {}
    overall_status = "pass"

    for mspec in tspec.metrics:
        impl = metric_impls[mspec.name]
        if getattr(impl, "non_judge", False):
            continue  # latency computed separately
        score, err = await _score_one_metric(
            tspec=tspec,
            tname=tname,
            mspec=mspec,
            impl=impl,
            example=example,
            output=output,
            default_model=default_model,
            fallback=fallback,
            complex_models=complex_models,
            complete_fn=complete_fn,
            cache=cache,
        )
        if err is not None:
            metric_errors[mspec.name] = err
            overall_status = "error"
        elif score is not None:
            metric_scores[mspec.name] = score

    # Node-scoring pass (AGENTS.md §1): walk the entry's captured trace and
    # score each node matching a declared ``node_metrics`` binding. The entry
    # runs once; its internal calls are scored from the trace, not
    # re-executed. No-op when the task has no ``node_metrics`` or the example
    # has no ``trace.calls`` (e.g. pre-bootstrap auto-seeded fixtures).
    node_scores: dict[str, dict[str, float]] = {}
    if tspec.node_metrics:
        trace = example.get("trace") or {}
        calls = trace.get("calls", []) if isinstance(trace, dict) else []
        node_metric_impls = _node_metric_impls(tspec, project_root)
        for nm in tspec.node_metrics:
            if nm.metric.name not in node_metric_impls:
                continue
            impl = node_metric_impls[nm.metric.name]
            if getattr(impl, "non_judge", False):
                continue
            for node_id, call in _select_nodes(calls, nm.node_selector):
                node_example = {
                    **example,
                    "input": call.get("args"),
                    "expected": None,
                    "trace": {"calls": [call]},
                }
                score, _err = await _score_one_metric(
                    tspec=tspec,
                    tname=tname,
                    mspec=nm.metric,
                    impl=impl,
                    example=node_example,
                    output=call.get("result"),
                    default_model=default_model,
                    fallback=fallback,
                    complex_models=complex_models,
                    complete_fn=complete_fn,
                    cache=cache,
                )
                if score is not None:
                    node_scores.setdefault(node_id, {})[nm.metric.name] = score

    # Determine example pass/fail from its metrics' threshold evaluations.
    if overall_status != "error":
        for mspec in tspec.metrics:
            impl = metric_impls[mspec.name]
            if getattr(impl, "non_judge", False):
                continue
            ev = evaluate_metric(
                mspec.name,
                metric_scores.get(mspec.name),
                threshold=mspec.threshold,
            )
            if ev.status == "fail":
                overall_status = "fail"
                break

    return ExampleRecord(
        id=ex_id,
        status=overall_status,  # type: ignore[arg-type]
        latency_ms=latency_ms,
        metric_scores=metric_scores,
        metric_errors=metric_errors,
        seed=seed,
        node_scores=node_scores,
    )


async def _score_one_metric(
    *,
    tspec: TaskSpec,
    tname: str,
    mspec: MetricSpec,
    impl: Any,
    example: dict,
    output: Any,
    default_model: str,
    fallback: tuple[str, ...],
    complex_models: tuple[str, ...] | None,
    complete_fn,
    cache: JudgeCache,
) -> tuple[float | None, list[str] | None]:
    """Score one metric for one (example, output) pair.

    Shared between the entry-level judge pass and the per-node judge pass so
    both use identical prompt-building and gateway-call semantics. Returns
    ``(score, errors)``: ``errors`` is ``None`` on success, a list of messages
    on failure (gateway returned no response or raised). The caller maps
    non-``None`` errors into ``metric_errors`` / the example's ``error``
    status at the entry level; node-level errors are dropped (a failing node
    metric doesn't fail the example — node scores are best-effort signal).
    """
    request = JudgeRequest(
        task_name=tname,
        task_type=tspec.type,
        metric=mspec.name,
        example=example,
        output=output,
    )
    try:
        messages = impl.prompt_builder(tspec, example, output)  # type: ignore[arg-type]
        response, errors = await judge_score(
            request,
            model=default_model,
            prompt_messages=messages,
            scored_dimension=impl.scored_dimension,
            complete_fn=complete_fn,
            fallback_models=fallback,
            cache=cache,
            complex_models=complex_models,
        )
        if response is None:
            return None, [e.message for e in errors]
        return response.score, None
    except Exception as exc:
        return None, [f"{type(exc).__name__}: {exc}"]


def _node_metric_impls(tspec: TaskSpec, project_root: Path) -> dict[str, Any]:
    """Resolve judge implementations for the metrics named in ``node_metrics``.

    Cached on ``tspec`` via a private attribute so the run-time strict gate
    (``assert_metric_implemented``) is only paid once per task. Raises
    :class:`~ai_eval.runner.thresholds.MetricNotImplementedError` if a node
    metric has no implementation — surfaced at run start, not mid-example.
    """
    cached = getattr(tspec, "_node_metric_impls_cache", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    impls: dict[str, Any] = {}
    for nm in tspec.node_metrics:
        if nm.metric.name in impls:
            continue
        impls[nm.metric.name] = assert_metric_implemented(nm.metric.name, project_root=project_root)
    try:
        object.__setattr__(tspec, "_node_metric_impls_cache", impls)
    except (AttributeError, TypeError):
        pass
    return impls


def _aggregate_metrics(
    tspec: TaskSpec,
    examples: list[ExampleRecord],
    baseline: dict[str, Any] | None,
    tname: str,
    tolerance: float,
    fail_on_regression: bool,
) -> dict[str, MetricResult]:
    out: dict[str, MetricResult] = {}
    base_task = (baseline or {}).get(tname) if baseline else None
    base_metrics: dict[str, Any] = {}
    if isinstance(base_task, dict):
        bm = base_task.get("metrics")
        if isinstance(bm, dict):
            base_metrics = bm
    # Entry-level metrics: mean over examples (existing behavior).
    for mspec in tspec.metrics:
        scores = [e.metric_scores[mspec.name] for e in examples if mspec.name in e.metric_scores]
        if mspec.name in ("latency_p50", "latency_p95"):
            # latency computed by engine; placeholder result for display
            out[mspec.name] = MetricResult(
                name=mspec.name, threshold=mspec.threshold, status="skip"
            )
            continue
        if not scores:
            out[mspec.name] = MetricResult(
                name=mspec.name, threshold=mspec.threshold, status="skip"
            )
            continue
        mean_score = sum(scores) / len(scores)
        base_score = None
        bm = base_metrics.get(mspec.name)
        if isinstance(bm, dict) and bm.get("score") is not None:
            base_score = float(bm["score"])
        ev = evaluate_metric(
            mspec.name,
            mean_score,
            threshold=mspec.threshold,
            baseline_score=base_score,
            tolerance=tolerance,
            fail_on_regression=fail_on_regression,
        )
        out[mspec.name] = MetricResult(
            name=mspec.name,
            score=ev.score,
            delta=ev.delta,
            threshold=ev.threshold,
            status=ev.status,
            weight=mspec.weight,
        )
    # Node-level metrics: weighted mean of that metric's node scores across
    # all examples (AGENTS.md §1). Sourced from ``node_scores`` per example —
    # a node metric only present in ``node_metrics`` (not ``metrics``) still
    # gets an aggregate row so the run report surfaces it.
    for nm in tspec.node_metrics:
        if nm.metric.name in out:
            continue  # already aggregated from entry-level metric_scores
        if nm.metric.name in ("latency_p50", "latency_p95"):
            out[nm.metric.name] = MetricResult(
                name=nm.metric.name, threshold=nm.metric.threshold, status="skip"
            )
            continue
        node_scores_flat: list[float] = []
        for e in examples:
            for _node_id, scores_for_node in e.node_scores.items():
                v = scores_for_node.get(nm.metric.name)
                if v is not None:
                    node_scores_flat.append(v)
        if not node_scores_flat:
            out[nm.metric.name] = MetricResult(
                name=nm.metric.name, threshold=nm.metric.threshold, status="skip"
            )
            continue
        mean_score = sum(node_scores_flat) / len(node_scores_flat)
        base_score = None
        bm = base_metrics.get(nm.metric.name)
        if isinstance(bm, dict) and bm.get("score") is not None:
            base_score = float(bm["score"])
        ev = evaluate_metric(
            nm.metric.name,
            mean_score,
            threshold=nm.metric.threshold,
            baseline_score=base_score,
            tolerance=tolerance,
            fail_on_regression=fail_on_regression,
        )
        out[nm.metric.name] = MetricResult(
            name=nm.metric.name,
            score=ev.score,
            delta=ev.delta,
            threshold=ev.threshold,
            status=ev.status,
            weight=nm.metric.weight,
        )
    return out


def _summarize(tasks: dict[str, TaskRecord]) -> RunSummary:
    passed = failed = errors = examples = 0
    for t in tasks.values():
        for ex in t.examples:
            examples += 1
            if ex.status == "error":
                errors += 1
            elif ex.status == "fail":
                failed += 1
            else:
                passed += 1
    return RunSummary(passed=passed, failed=failed, errors=errors, examples=examples)


__all__ = ["config_hash", "execute"]

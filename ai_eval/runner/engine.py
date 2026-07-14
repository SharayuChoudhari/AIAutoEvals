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

from ai_eval.config.schema import RubricsConfig, TaskSpec
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


class _Stub:
    """Permissive stand-in for a required non-primitive constructor arg.

    Any attribute access or call returns a new _Stub so a task class's
    __init__ / method body that touches ``config.x`` or ``session.add(...)``
    doesn't raise. Used only for required non-primitive params (Bug 3).
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> _Stub:
        return _Stub()

    def __call__(self, *a, **kw) -> _Stub:
        return _Stub()


def _placeholder_for(param: inspect.Parameter) -> Any:
    """Fabricate a plausible value for one required (no-default) param.

    Primitives map to their zero value; everything else (unknown annotations,
    class-typed params, unannotated) gets a permissive ``_Stub`` so method
    bodies that touch ``config.x`` / ``session.add(...)`` don't raise.
    """
    ann = param.annotation
    if ann is int:
        return 0
    if ann is float:
        return 0.0
    if ann is bool:
        return False
    if ann is str:
        return ""
    return _Stub()


def _fake_call_args(sig: inspect.Signature) -> tuple[tuple, dict]:
    """Build plausible positional + keyword args for a callable from its
    inspect.Signature: primitives default to ``""``/``0``/``0.0``/``False``,
    required non-primitive params get a ``_Stub`` (so ``__init__`` / method
    bodies that touch ``config.x`` or ``session.add(...)`` don't raise), and
    params with defaults are left to apply. ``self`` is skipped (it's supplied
    implicitly by ``cls(...)``). Used to construct task-class instances for
    dotted ``Class.method`` entries (pure-LLM path; IO-coupled tasks get the
    harness monkey-patches installed first).
    """
    args: list = []
    kwargs: dict = {}
    for param in sig.parameters.values():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # `self` is supplied implicitly by `cls(...)` — never fabricate it.
        if param.name == "self":
            continue
        if param.default is not inspect.Parameter.empty:
            # Has a real default — let it apply.
            continue
        args.append(_placeholder_for(param))
    return tuple(args), kwargs


def _build_call_args(fn, input_: Any) -> tuple[tuple, dict]:
    """Build ``(args, kwargs)`` to call ``fn`` with ``input_``.

    Signature-aware so the auto-seed convention (a scalar string input) works
    for entries of any arity:

    * ``input_`` is a ``dict``  → unpacked as kwargs (rich/golden-set case).
    * ``input_`` is a ``list``/``tuple`` → unpacked as positional args.
    * ``input_`` is a scalar (the D6 auto-seed case) → inspect ``fn``'s
      signature: map the scalar onto the first str-typed required param (or
      the first required param if none is str-typed), and fabricate
      placeholders for the remaining required params. Params with defaults are
      left to apply. If the signature can't be introspected (builtins/C
      callables) the scalar is passed as a single positional arg (legacy
      fast path).

    This is what makes ``evaluate_single(self, question, retrieved_contexts,
    answer)`` and ``_create_workflow(self)`` callable from a ``""`` auto-seed
    without ``TypeError: missing N required positional arguments``.
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
    # required param. The remaining required params get fabricated placeholders;
    # params with defaults are left to apply (not appended positionally, so
    # Python fills them — but we must still append placeholders for required
    # params that come *after* a defaulted one to keep positional order valid).
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
            args.append(_placeholder_for(p))
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
    class with auto-faked constructor args (primitives / None-defaulted) so the
    method can be called with a ``self``. For IO-coupled tasks the harness
    monkey-patches are installed first (``eval/_harness_<task>.py``) so the
    instance's DAO/session reads return canned fixtures.

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
    # faked args, return the bound method. Bare ``fn``: return the function.
    if "." in entry:
        cls_name, _, method_name = entry.rpartition(".")
        cls = getattr(mod, cls_name, None)
        if cls is None:
            raise AttributeError(f"class {cls_name!r} not found in {file_path}")

        try:
            sig = inspect.signature(cls.__init__)
            # Drop 'self' from the fake-args construction.
            args, kwargs = _fake_call_args(sig)
            instance = cls(*args, **kwargs)
        except TypeError:
            # Fallback: try a no-arg construction.
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
                metric_errors[mspec.name] = [e.message for e in errors]
                overall_status = "error"
            else:
                metric_scores[mspec.name] = response.score
        except Exception as exc:
            metric_errors[mspec.name] = [f"{type(exc).__name__}: {exc}"]
            overall_status = "error"

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
    )


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

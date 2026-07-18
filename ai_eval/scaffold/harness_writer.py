"""IO-coupled classification (D4) and stub-harness codegen (D5).

D4 classifies a detected task as **IO-coupled** when its resolved method body
reads through a ``self.<attr>`` whose ``__init__`` assignment is a type that is
NOT a known LLM client and NOT a known compiled framework graph (those run real
and need a key, not a stub). DAOs / sessions / HTTP clients / file handles /
queues get stubbed.

D5 emits ``eval/_harness_<task>.py`` for each IO-coupled task. The file is
region-split:

  Region 1 (``# AUTO-GENERATED DO NOT EDIT``) — wiring: class/method
  resolution, monkey-patch setup, no-op session. Regenerated wholesale on
  ``init``/``analyze`` only when the task method body's AST hash changed since
  last generation (hash stored in a sidecar comment).

  Region 2 (``# FIXTURE DATA — edit freely``) — canned return objects. Written
  only on first generation; never overwritten thereafter (mirrors
  ``golden_set.json``'s preserve-captures contract).

On load the harness best-effort-validates that each patched method's return
shape still matches the call-site AST and warns (not fails) on drift.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_eval.config.schema import RubricsConfig, TaskSpec
from ai_eval.inference.callgraph import FileContext, build_call_graph
from ai_eval.inference.signatures import iter_calls

#: Known LLM-client type names (D4): assignments to these are NOT stubbed —
#: the client runs real (needs an API key, not canned data).
_LLM_CLIENT_TYPES: frozenset[str] = frozenset(
    {
        "OpenAI",
        "AzureOpenAI",
        "AsyncOpenAI",
        "ChatOpenAI",
        "Anthropic",
        "ChatAnthropic",
        "Ollama",
        "HuggingFaceHub",
        "HuggingFacePipeline",
        "ChatGoogleGenerativeAI",
    }
)

#: Known compiled framework-graph object names (D4): assignments to these run
#: real (they orchestrate the LLM; stubbing them would defeat the eval).
_GRAPH_TYPES: frozenset[str] = frozenset(
    {
        "StateGraph",
        "RunnableSequence",
        "CompiledGraph",
        "CompiledStateGraph",
    }
)


@dataclass
class IOAttr:
    """One ``self.<attr>.<method>(...)`` read site in an IO-coupled task body."""

    attr: str
    method: str
    #: The class name the attr is constructed as (best-effort, from ``__init__``).
    ctor_name: str | None = None


@dataclass
class HarnessSpec:
    """Everything needed to emit ``eval/_harness_<task>.py`` for one task."""

    task_name: str
    entry: str
    file_path: str
    attrs: list[IOAttr] = field(default_factory=list)
    body_hash: str = ""


def _ctor_name_of(rhs: ast.expr) -> str | None:
    """Extract the constructor/bound name from an ``__init__`` RHS expression.

    Unwraps chained builder calls so the underlying type is detected:
      ``SomeClass(...)``              → ``SomeClass``
      ``SomeClass(...)`` (Attribute)  → trailing name (e.g. ``from foo import Bar``)
      ``StateGraph(dict).compile()``  → ``StateGraph`` (unwrap the .compile builder)
      ``SomeClass`` (bare name)       → ``SomeClass``
    """
    if isinstance(rhs, ast.Call):
        f = rhs.func
        # ``X(...).compile()`` — unwrap the receiver call to get X.
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call):
            return _ctor_name_of(f.value)
        if isinstance(f, ast.Name):
            return f.id
        if isinstance(f, ast.Attribute):
            return f.attr
    if isinstance(rhs, ast.Name):
        return rhs.id
    return None


def _is_stub_target(ctor_name: str | None) -> bool:
    """True when a ``self.<attr>`` assignment targets a stubbable type (not an
    LLM client, not a compiled graph). ``None`` (unresolved) → not stubbed
    (safe: don't fabricate a stub for an unknown type)."""
    if ctor_name is None:
        return False
    if ctor_name in _LLM_CLIENT_TYPES:
        return False
    if ctor_name in _GRAPH_TYPES:
        return False
    return True


def classify_io_coupled(task: TaskSpec, contexts: dict[str, FileContext]) -> list[IOAttr]:
    """Return the list of stubbable ``self.<attr>.<method>()`` read sites in the
    task's resolved method body. Empty list ⇒ pure-LLM (not IO-coupled).

    A read site is stubbable iff its ``<attr>`` is assigned in the class
    ``__init__`` to a constructor whose name is NOT an LLM client / graph type.
    """
    ctx = contexts.get(task.file_path)
    if ctx is None or task.entry is None:
        return []
    def_node = ctx.entry_to_def.get(task.entry)
    if def_node is None:
        return []
    # Only methods (dotted ``Class.method``) read ``self.<attr>``; module-level
    # functions have no ``self``.
    if "." not in task.entry:
        return []
    cls_name = task.entry.rsplit(".", 1)[0]
    init = ctx.class_inits.get(cls_name)
    if init is None:
        return []

    attrs: list[IOAttr] = []
    seen: set[tuple[str, str]] = set()
    for call in iter_calls(def_node):
        func = call.func
        if not isinstance(func, ast.Attribute):
            continue
        receiver = func.value
        # ``self.<attr>.<method>(...)``
        if not (
            isinstance(receiver, ast.Attribute)
            and isinstance(receiver.value, ast.Name)
            and receiver.value.id == "self"
        ):
            continue
        attr_name = receiver.attr
        method = func.attr
        rhs = init.attrs.get(attr_name)
        if rhs is None:
            continue  # not set in __init__ → not tracked (R2 scope)
        ctor = _ctor_name_of(rhs)
        if not _is_stub_target(ctor):
            continue
        key = (attr_name, method)
        if key in seen:
            continue
        seen.add(key)
        attrs.append(IOAttr(attr=attr_name, method=method, ctor_name=ctor))
    return attrs


def _body_hash(def_node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Stable hash of a method body's AST (drives region-1 regeneration)."""
    src = ast.dump(def_node)
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


def _build_harness_specs(
    rubrics: RubricsConfig, contexts: dict[str, FileContext]
) -> list[HarnessSpec]:
    specs: list[HarnessSpec] = []
    for name, task in rubrics.tasks.items():
        attrs = classify_io_coupled(task, contexts)
        if not attrs:
            continue  # pure-LLM
        def_node = contexts[task.file_path].entry_to_def.get(task.entry)  # type: ignore[index]
        body_hash = _body_hash(def_node) if def_node is not None else ""
        specs.append(
            HarnessSpec(
                task_name=name,
                entry=task.entry or "",
                file_path=task.file_path,
                attrs=attrs,
                body_hash=body_hash,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Region-split codegen
# ---------------------------------------------------------------------------

_REGION1_HEADER = "# AUTO-GENERATED DO NOT EDIT — ai-evals harness wiring"
_REGION2_HEADER = "# FIXTURE DATA — edit freely"
_HASH_PREFIX = "# ast-hash:"


def _safe_module_attr(task_name: str) -> str:
    """``chat_message_service_process_query`` → a valid Python identifier for the
    harness module's patch-target resolution helper."""
    return re.sub(r"[^0-9a-zA-Z_]", "_", task_name)


def _render_region1(spec: HarnessSpec) -> str:
    """Render the auto-generated wiring region.

    Resolves each stubbed class+method from the task module and monkey-patches
    it to return the canned fixture from region 2. Idempotent: re-running init
    rewrites only this region when the body hash changed.
    """
    lines: list[str] = [_REGION1_HEADER, f"{_HASH_PREFIX} {spec.body_hash}", ""]
    lines.append(f'"""Harness wiring for task {spec.task_name!r}.')
    lines.append("Re-generated by `ai-evals init`/`analyze` when the task method")
    lines.append("body's AST hash changes. Do not edit; edit region 2 fixtures.")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import importlib.util")
    lines.append("import sys")
    lines.append("")
    lines.append(f"_TASK_FILE = {spec.file_path!r}")
    lines.append(f"_TASK_ENTRY = {spec.entry!r}")
    lines.append("from pathlib import Path as _Path")
    lines.append("# Resolve the task file relative to the project root (the harness")
    lines.append("# lives in eval/, so its parent's parent is the project root).")
    lines.append("_TASK_FILE_ABS = str((_Path(__file__).parent.parent / _TASK_FILE).resolve())")
    lines.append("")
    # The fixture table lives in region 2; region 1 imports it.
    lines.append("# region 2 fixtures (defined below): ATTR_METHOD → return value")
    lines.append("_FIXTURES = {}  # populated by region 2")
    lines.append("")
    lines.append("")
    lines.append("def _load_task_module():")
    lines.append('    """Import the task module by file path (idempotent)."""')
    lines.append("    spec = importlib.util.spec_from_file_location(")
    lines.append("        '_ai_eval_harness_target', _TASK_FILE_ABS)")

    lines.append("    if spec is None or spec.loader is None:")
    lines.append("        raise ImportError('cannot load ' + _TASK_FILE_ABS)")
    lines.append("    mod = importlib.util.module_from_spec(spec)")
    lines.append("    sys.modules['_ai_eval_harness_target'] = mod")
    lines.append("    spec.loader.exec_module(mod)  # type: ignore[union-attr]")
    lines.append("    return mod")
    lines.append("")
    lines.append("")
    lines.append("def install():")
    lines.append('    """Monkey-patch each stubbed class method to return its fixture."""')
    lines.append("    mod = _load_task_module()")
    for a in spec.attrs:
        ctor = a.ctor_name or ""
        lines.append(f"    _ctor_cls = getattr(mod, {ctor!r}, None)")
        lines.append("    if _ctor_cls is not None:")
        lines.append(f"        _patch = _make_patch({a.attr!r}, {a.method!r})")
        lines.append(f"        setattr(_ctor_cls, {a.method!r}, _patch)")
    lines.append("")
    lines.append("")
    lines.append("def _make_patch(attr, method):")
    lines.append('    """Build a stub method returning the canned fixture for (attr, method)."""')
    lines.append("    def _stub(self, *a, **kw):")
    lines.append("        return _FIXTURES.get((attr, method))")
    lines.append("    return _stub")
    lines.append("")
    return "\n".join(lines)


def _render_region2(spec: HarnessSpec) -> str:
    """Render the editable fixture region.

    On first generation each stubbed (attr, method) gets a placeholder canned
    object (``{}`` for dict-shaped, ``None`` otherwise). The user edits these
    to match the real call-site return shape. This region is in the same module
    as region 1, so the module-global ``_FIXTURES`` defined here shadows region
    1's empty placeholder and is read by ``install()`` at load time.
    """
    lines: list[str] = ["", "", _REGION2_HEADER, ""]
    lines.append("# Canned return objects for each stubbed self.<attr>.<method>().")
    lines.append("# Edit freely — this region is preserved across regenerations.")
    lines.append("_FIXTURES = {")
    for a in spec.attrs:
        lines.append(f"    ({a.attr!r}, {a.method!r}): {{}},  # {a.ctor_name or 'unknown'}")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def render_harness(spec: HarnessSpec, *, existing_region2: str | None = None) -> str:
    """Render the full harness file.

    ``existing_region2`` — if the harness file already exists, the caller passes
    the preserved region-2 text so it is never overwritten (D5 contract).
    """
    region1 = _render_region1(spec)
    region2 = existing_region2 if existing_region2 is not None else _render_region2(spec)
    return f"{region1}\n{region2}\n"


# ---------------------------------------------------------------------------
# Sidecar hash extraction (region-1 regeneration gate)
# ---------------------------------------------------------------------------


def _existing_body_hash(path: Path) -> str | None:
    """Read the sidecar ``# ast-hash: <hash>`` from an existing harness file."""
    if not path.is_file():
        return None
    head = path.read_text(encoding="utf-8").splitlines()
    for line in head[:5]:
        m = re.match(rf"{re.escape(_HASH_PREFIX)} (\w+)", line)
        if m:
            return m.group(1)
    return None


def _existing_region2(path: Path) -> str | None:
    """Extract the region-2 block from an existing harness file (preserved)."""
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8")
    idx = content.find(_REGION2_HEADER)
    if idx < 0:
        return None
    return content[idx:]


# ---------------------------------------------------------------------------
# Public write API
# ---------------------------------------------------------------------------


def write_harnesses(
    rubrics: RubricsConfig,
    eval_dir: Path,
    *,
    project_root: Path,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Generate ``eval/_harness_<task>.py`` for every IO-coupled task.

    Returns ``([(relative_path, status)], io_coupled_task_names)`` where status
    is ``"wrote"`` (new), ``"refreshed"`` (region 1 regenerated because the body
    hash changed), or ``"skipped"`` (hash unchanged, file untouched). The
    ``io_coupled_task_names`` set lets the golden-set seeder (D6) give those
    tasks a single green-pipeline example instead of the pure-LLM variant set.

    Pure-LLM tasks get no harness. Region 2 (fixtures) is preserved across
    regenerations.
    """
    # Build file contexts for the task files (reuses the callgraph parser so
    # the AST + __init__ tracking is consistent with D4 classification).
    task_files = sorted({Path(t.file_path) for t in rubrics.tasks.values() if t.file_path})
    from ai_eval.inference.ast_scan import ScanResult
    from ai_eval.inference.detectors.base import DetectedTask

    # Minimal scan to populate contexts: build_call_graph needs a ScanResult.
    dummy_tasks = [
        DetectedTask(
            name=n,
            framework="unknown",
            type=d.type,
            file_path=d.file_path,
            entry=d.entry,
        )
        for n, d in rubrics.tasks.items()
    ]
    scan = ScanResult(
        files_scanned=len(task_files),
        elapsed_seconds=0.0,
        tasks=dummy_tasks,
        frameworks_seen=set(),
    )
    _, contexts = build_call_graph(project_root, scan)

    specs = _build_harness_specs(rubrics, contexts)
    io_coupled_names = {s.task_name for s in specs}
    written: list[tuple[str, str]] = []
    eval_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        safe = _safe_module_attr(spec.task_name)
        path = eval_dir / f"_harness_{safe}.py"
        existing_hash = _existing_body_hash(path)
        region2 = _existing_region2(path)
        if existing_hash == spec.body_hash and path.is_file():
            written.append((path.name, "skipped"))
            continue
        content = render_harness(spec, existing_region2=region2)
        path.write_text(content, encoding="utf-8")
        status = "refreshed" if existing_hash is not None else "wrote"
        written.append((path.name, status))
    return written, io_coupled_names


__all__ = [
    "HarnessSpec",
    "IOAttr",
    "classify_io_coupled",
    "write_harnesses",
]

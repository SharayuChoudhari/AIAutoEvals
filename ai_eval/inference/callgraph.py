"""Cross-file call-graph resolution among detected framework-call sites (R2).

The graph's *nodes* are detected task sites (``DetectedTask``s); an edge
``A → B`` means task A's enclosing method body reaches task B's enclosing
method body via a resolvable call. Resolution is purely AST-local — no
execution — and uses two strategies:

1. ``self.<attr>.<m>(...)`` inside method ``A``: track ``<attr>`` to its
   ``__init__`` assignment (``SomeClass(...)`` or a module-level ``fn()``
   import), resolve ``SomeClass``'s import to a file, then check whether that
   file's class method ``<m>`` (or its body transitively) is the enclosing
   method of task B.
2. Module-level ``fn(...)`` calls: resolve ``fn``'s import to a file, check
   whether that file's function ``fn`` is the enclosing method of task B.

Edges are drawn **between detected sites only**. When an ``__init__`` attr
can't be resolved (lazy init, property, dynamic dispatch), no edge is drawn —
both endpoints stay roots. This is the agreed ``R2`` (not ``R2+``) scope:
``__init__``-only attribute tracking, degrading safely to over-promotion.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

from ai_eval.inference.ast_scan import ScanResult
from ai_eval.inference.detectors.base import DetectedTask
from ai_eval.inference.signatures import (
    ImportInfo,
    collect_imports,
    find_callable_defs,
    iter_calls,
)


@dataclass(frozen=True)
class SiteKey:
    """Identity of one detected site: its file + enclosing entry.

    Two tasks with the same ``(file_path, entry)`` are the same site (detectors
    dedup by entry within a file, so collisions are rare; we key on entry
    rather than task name because name is the rubrics key, not the resolver).
    """

    file_path: str
    entry: str | None

    @classmethod
    def of(cls, task: DetectedTask) -> SiteKey:
        return cls(file_path=task.file_path, entry=task.entry)


@dataclass
class ClassInitAttrs:
    """``self.<attr>`` assignments found in a class ``__init__``.

    Maps the attribute name to the RHS expression so callers can tell whether
    it's a constructor call (``SomeClass(...)``) or something else.
    """

    cls_name: str
    attrs: dict[str, ast.expr] = field(default_factory=dict)


@dataclass
class FileContext:
    """Parsed context for one file, indexed for call-graph resolution."""

    path: Path
    #: repo-root-relative POSIX path (the dict key in ``build_call_graph``'s
    #: contexts map), so resolvers can emit same-file targets without re-deriving.
    rel_path: str
    tree: ast.AST
    imports: list[ImportInfo]
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef]
    #: ``dotted entry → def node`` for quick lookup of enclosing methods.
    entry_to_def: dict[str | None, ast.FunctionDef | ast.AsyncFunctionDef]
    #: ``class name → __init__ attr assignments`` for ``self.<attr>`` tracking.
    class_inits: dict[str, ClassInitAttrs]


def _dotted_module(root: Path, file_path: Path) -> str:
    """Best-effort dotted module name for ``file_path`` relative to ``root``.

    ``services/chat_messages.py`` → ``services.chat_messages``. Used to match
    against ``from services.chat_messages import ...`` style imports. Package
    ``__init__.py`` files map to their package (``services/__init__.py`` →
    ``services``).
    """
    rel = file_path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _collect_class_inits(
    tree: ast.AST,
) -> dict[str, ClassInitAttrs]:
    """For each top-level class, record ``self.<attr> = <expr>`` in ``__init__``.

    Only ``__init__`` is scanned (R2 scope). Assignments outside ``__init__``
    (lazy init in other methods, class-level attrs) are deliberately ignored —
    that's the over-promotion-safe degradation boundary.
    """
    out: dict[str, ClassInitAttrs] = {}
    if not isinstance(tree, ast.Module):
        return out
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        init_attrs = ClassInitAttrs(cls_name=node.name)
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if child.name.rsplit(".", 1)[-1] != "__init__":
                continue
            for stmt in ast.walk(child):
                if not isinstance(stmt, ast.Assign):
                    continue
                if len(stmt.targets) != 1:
                    continue
                target = stmt.targets[0]
                # ``self.<attr> = <expr>``
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    init_attrs.attrs[target.attr] = stmt.value
        if init_attrs.attrs:
            out[node.name] = init_attrs
    return out


def _build_file_contexts(
    root: Path, files: list[Path]
) -> tuple[dict[str, FileContext], dict[str, str]]:
    """Parse every file once and build per-file contexts plus a module index.

    Returns ``(file_path_str → FileContext, dotted_module → file_path_str)``.
    The module index is keyed by the dotted module name derived from the file
    path, so import resolution can map a dotted module to a file.
    """
    contexts: dict[str, FileContext] = {}
    module_index: dict[str, str] = {}
    for fpath in files:
        try:
            source = fpath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(fpath))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        rel = fpath.relative_to(root).as_posix()
        defs = find_callable_defs(tree)
        entry_to_def: dict[str | None, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        for d in defs:
            entry_to_def[d.name] = d
        contexts[rel] = FileContext(
            path=fpath,
            rel_path=rel,
            tree=tree,
            imports=collect_imports(tree),
            defs=defs,
            entry_to_def=entry_to_def,
            class_inits=_collect_class_inits(tree),
        )
        module_index[_dotted_module(root, fpath)] = rel
    return contexts, module_index


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------


def _resolve_imported_name(
    imports: list[ImportInfo], module_index: dict[str, str], name: str
) -> str | None:
    """Resolve a locally-bound ``name`` to a file path via the import table.

    Handles ``from pkg.mod import SomeClass`` (name bound = ``SomeClass``) and
    ``import pkg.mod`` (name bound = ``pkg``). Returns the file path (relative
    POSIX) of the defining module, or ``None`` when it's a stdlib/third-party
    module not in the repo.
    """
    for info in imports:
        local = info.alias or info.name.split(".")[0]
        if local != name:
            continue
        # ``from pkg.mod import X [as name]`` → defining module is ``pkg.mod``.
        if info.module and (info.module in module_index):
            return module_index[info.module]
        # ``import pkg.mod`` → the whole dotted module is the source; if it's a
        # repo package, resolve to its file.
        if info.name in module_index:
            return module_index[info.name]
    return None


def _resolve_self_attr_call(
    call: ast.Call,
    ctx: FileContext,
    contexts: dict[str, FileContext],
    module_index: dict[str, str],
) -> tuple[str, str] | None:
    """Resolve a ``self.<attr>.<m>(...)`` call to ``(target_file, method_name)``.

    Returns ``None`` when ``<attr>`` isn't set in any ``__init__`` visible from
    this class (the safe over-promotion degradation). When resolved, returns
    the file where ``<m>`` would be defined and the method's bare name.
    """
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None
    # Want ``self.<attr>.<m>``: the receiver of ``<m>`` is ``self.<attr>``.
    receiver = func.value
    if not isinstance(receiver, ast.Attribute):
        return None
    if not (isinstance(receiver.value, ast.Name) and receiver.value.id == "self"):
        return None
    attr_name = receiver.attr
    method = func.attr
    # Find the class enclosing this call to look up its __init__ attrs. The
    # call's enclosing def is a dotted ``Class.method``; the class segment is
    # the prefix before the last dot.
    # We don't have the enclosing def here, but we can scan all class_inits in
    # the file for one that assigns ``self.<attr> = SomeClass(...)``. If a class
    # assigns the attr to a constructor, resolve the constructor's import.
    for init in ctx.class_inits.values():
        rhs = init.attrs.get(attr_name)
        if rhs is None:
            continue
        # ``SomeClass(...)`` constructor call.
        if isinstance(rhs, ast.Call):
            ctor = rhs.func
            ctor_name: str | None = None
            if isinstance(ctor, ast.Name):
                ctor_name = ctor.id
            elif isinstance(ctor, ast.Attribute):
                ctor_name = ctor.attr
            if ctor_name is None:
                continue
            target_file = _resolve_imported_name(ctx.imports, module_index, ctor_name)
            if target_file is not None:
                return target_file, method
        # ``self.<attr> = SomeClass`` (bare name binding) — resolve import.
        elif isinstance(rhs, ast.Name):
            target_file = _resolve_imported_name(ctx.imports, module_index, rhs.id)
            if target_file is not None:
                return target_file, method
    return None


def _resolve_module_fn_call(
    call: ast.Call,
    ctx: FileContext,
    module_index: dict[str, str],
) -> tuple[str, str] | None:
    """Resolve a bare ``fn(...)`` (or ``pkg.fn(...)``) call to its file + name.

    Returns ``(target_file, fn_name)`` when ``fn`` is imported from a repo
    module; ``None`` otherwise (builtins, third-party).
    """
    func = call.func
    name: str | None = None
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        # ``pkg.fn(...)`` — resolve the package to a file, fn is the method.
        pkg = func.value.id
        target_file = _resolve_imported_name(ctx.imports, module_index, pkg)
        if target_file is not None:
            return target_file, func.attr
    if name is None:
        return None
    target_file = _resolve_imported_name(ctx.imports, module_index, name)
    if target_file is not None:
        return target_file, name
    # Same-file bare-name call: ``fn`` isn't imported, but it may be a
    # module-level def in this file. Resolve to the current file so the
    # edge-matcher can find a detected site with that entry here.
    if name in ctx.entry_to_def:
        return ctx.rel_path, name
    return None


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_call_graph(
    root: Path, scan: ScanResult
) -> tuple[dict[SiteKey, set[SiteKey]], dict[str, FileContext]]:
    """Build the cross-file call graph among detected sites.

    Returns ``(edges, file_contexts)`` where ``edges[A] = {B, C, ...}`` means
    task A reaches tasks B and C. ``file_contexts`` is returned so downstream
    steps (judge-exclusion, IO-coupled classification) reuse the parsed ASTs
    without re-parsing.

    Only detected sites are nodes. A non-detected helper reachable from a
    detected site is *not* a node — it's just an edge hop toward another
    detected site (or a dead end, dropping the edge).
    """
    # Parse the files that contain detected sites (bounded set).
    task_files = sorted({Path(t.file_path) for t in scan.tasks if t.file_path})
    abs_task_files = [
        (root / str(p)) if not str(p).startswith("/") else Path(str(p)) for p in task_files
    ]
    contexts, module_index = _build_file_contexts(root, abs_task_files)

    # Index tasks by site key for O(1) reachability checks.
    sites_by_file_entry: dict[SiteKey, DetectedTask] = {}
    for t in scan.tasks:
        sites_by_file_entry[SiteKey.of(t)] = t

    edges: dict[SiteKey, set[SiteKey]] = {SiteKey.of(t): set() for t in scan.tasks}

    for task in scan.tasks:
        src_key = SiteKey.of(task)
        ctx = contexts.get(task.file_path)
        if ctx is None or task.entry is None:
            continue
        # Only methods (dotted ``Class.method``) have ``self.<attr>`` reads;
        # module-level functions can only reach via bare ``fn()`` calls.
        enclosing_def = ctx.entry_to_def.get(task.entry)
        if enclosing_def is None:
            continue
        for call in iter_calls(enclosing_def):
            # Strategy 1: ``self.<attr>.<m>(...)``
            target = _resolve_self_attr_call(call, ctx, contexts, module_index)
            if target is None:
                # Strategy 2: bare ``fn(...)`` or ``pkg.fn(...)``
                target = _resolve_module_fn_call(call, ctx, module_index)
            if target is None:
                continue
            target_file, target_name = target
            # The target method's enclosing entry is ``Class.target_name`` if
            # it's a method, or bare ``target_name`` if module-level. Match any
            # detected site in ``target_file`` whose entry equals ``target_name``
            # (module fn) or ends in ``.target_name`` (method).
            for key, other in sites_by_file_entry.items():
                if key.file_path != target_file or other is task:
                    continue
                if _entry_matches(other.entry, target_name):
                    edges[src_key].add(key)
    return edges, contexts


def _entry_matches(entry: str | None, target_name: str) -> bool:
    """True if ``entry`` is the bare ``target_name`` or a method ``*.target_name``."""
    if entry is None:
        return False
    if entry == target_name:
        return True
    return entry.rsplit(".", 1)[-1] == target_name and "." in entry


def compute_roots(
    edges: dict[SiteKey, set[SiteKey]], tasks: list[DetectedTask]
) -> tuple[list[DetectedTask], list[DetectedTask]]:
    """Split tasks into ``(roots, reached)``.

    A task is a **root** iff no other detected site reaches it transitively.
    Reached sites are demoted to internal nodes. Transitive closure is computed
    over the edge set so indirect reachability (A→B→C) demotes both B and C.
    """
    all_keys = [SiteKey.of(t) for t in tasks]
    key_to_task = {SiteKey.of(t): t for t in tasks}

    # Transitive closure of reachability via iterative forward traversal.
    reached: set[SiteKey] = set()
    frontier: list[SiteKey] = []
    for key in all_keys:
        frontier.extend(edges.get(key, ()))
    while frontier:
        node = frontier.pop()
        if node in reached:
            continue
        reached.add(node)
        frontier.extend(edges.get(node, ()))

    roots = [key_to_task[k] for k in all_keys if k not in reached]
    reached_tasks = [key_to_task[k] for k in all_keys if k in reached]
    return roots, reached_tasks


__all__ = [
    "ClassInitAttrs",
    "FileContext",
    "SiteKey",
    "build_call_graph",
    "compute_roots",
]

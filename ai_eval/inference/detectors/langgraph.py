"""Detect LangGraph workflows (StateGraph + graph.invoke/.compile)."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.detectors.base import DetectedTask, Detector
from ai_eval.inference.signatures import (
    ImportInfo,
    attr_chain,
    enclosing_def_name,
    find_callable_defs,
    has_import_prefix,
    iter_calls,
)

#: Methods on a compiled graph that run the workflow end-to-end. We emit one
#: task per enclosing def that contains such a call (the public entry point).
_GRAPH_INVOKE_METHODS = {"invoke", "ainvoke", "stream", "astream"}

#: Method that finalizes a graph builder into a runnable. Used both as a
#: construction signal (paired with StateGraph) and as an emit fallback when
#: no invoke call exists in the enclosing def.
_COMPILE_METHOD = "compile"

#: Constructor call that begins a LangGraph graph. ``StateGraph`` may be
#: imported from ``langgraph.graph`` (or aliased); we match on the trailing
#: name only after the langgraph import prefix has gated ``matches``.
_STATEGRAPH_NAMES = {"StateGraph"}


def _is_stategraph_construction(chain: list[str]) -> bool:
    return bool(chain) and chain[-1] in _STATEGRAPH_NAMES


def _is_graph_run_or_compile(chain: list[str]) -> tuple[bool, bool]:
    """Return ``(is_run, is_compile)`` for a graph method call chain.

    ``is_run`` covers invoke/ainvoke/stream/astream (the entry-point verbs);
    ``is_compile`` covers ``.compile()``. Both are computed independently so
    callers can distinguish a run call (the preferred emit site) from a
    compile-only fallback.
    """
    if not chain:
        return False, False
    last = chain[-1]
    return last in _GRAPH_INVOKE_METHODS, last == _COMPILE_METHOD


class LangGraphDetector(Detector):
    """Workflow detector for LangGraph.

    Emits **one** task per enclosing def that runs a compiled graph
    (``graph.invoke``/``ainvoke``/``stream``/``astream``). To avoid false
    positives on the generic ``.invoke()`` method name, the enclosing function
    must also show construction evidence: a ``StateGraph(...)`` call or a
    ``.compile()`` call. When no run call exists but a ``.compile()`` does,
    we emit on the compile call (the graph's public surface).

    Independent of :class:`LangChainDetector` (different framework string),
    but registered before it in case a repo imports both families.
    """

    framework = "langgraph"

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        return has_import_prefix(imports, "langgraph")

    def extract(
        self,
        tree: ast.AST,
        imports: list[ImportInfo],
        file_path: Path,
        project_root: Path,
        *,
        calls: list[ast.Call] | None = None,
        defs: list[ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    ) -> list[DetectedTask]:
        _calls = calls if calls is not None else list(iter_calls(tree))
        _defs = defs if defs is not None else find_callable_defs(tree)
        rel = file_path.relative_to(project_root).as_posix()

        # Group candidate calls by enclosing def so we can require construction
        # evidence alongside a run call (order-independent, mirroring the
        # LangChain per-function dedup pattern). Each entry records the call,
        # its chain, and flags for run/compile/stategraph.
        by_func: dict[str | None, list[tuple[ast.Call, list[str], bool, bool, bool]]] = {}
        for call in _calls:
            chain = attr_chain(call.func)
            if not chain:
                continue
            is_run, is_compile = _is_graph_run_or_compile(chain)
            is_stategraph = _is_stategraph_construction(chain)
            if not is_run and not is_compile and not is_stategraph:
                continue
            entry = enclosing_def_name(call, _defs)
            by_func.setdefault(entry, []).append((call, chain, is_run, is_compile, is_stategraph))

        out: list[DetectedTask] = []
        seen: set[str] = set()
        for entry, items in by_func.items():
            has_construction = any(
                it[4] or it[3]
                for it in items  # stategraph or compile
            )
            # Prefer to emit on a run call when construction evidence exists in
            # the same function. Fall back to a compile call (still gated by
            # construction — compile is itself construction). Skip functions
            # with only an add-node-style StateGraph call and no run/compile,
            # or a bare invoke with no construction signal (likely not a graph).
            run_call = next((it for it in items if it[2]), None)
            if run_call is not None and has_construction:
                emit_call, emit_chain, _, _, _ = run_call
            else:
                compile_call = next((it for it in items if it[3]), None)
                if compile_call is None:
                    continue
                emit_call, emit_chain, _, _, _ = compile_call
            name = entry or f"{file_path.stem}_workflow"
            if name in seen:
                continue
            seen.add(name)
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type="workflow",
                    file_path=rel,
                    entry=entry,
                    inputs=["state"],
                    outputs=["state"],
                    evidence=[f"langgraph {'.'.join(emit_chain)} at {rel}:{emit_call.lineno}"],
                )
            )
        return out


__all__ = ["LangGraphDetector"]

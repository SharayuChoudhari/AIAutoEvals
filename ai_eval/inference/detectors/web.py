"""Web-service detector base class.

Provides the shared route-handler extraction logic for web frameworks that
register endpoints via decorators (``@app.get(...)``, ``@router.post(...)``,
…). Each route handler is emitted as one :class:`DetectedTask` with
``top_level=True`` — route handlers ARE the end-to-end entry points the user
cares about (AGENTS.md §1). The internal DAO/service/evaluator methods a
handler calls are demoted separately by:

1. Layer 3 peer-reach (``task_selection._peer_reached_keys``) when the
   handler is a detected task and an AST call edge resolves to the service.
2. Signature-inspection demotion (``synthesize._requires_io_args``) when the
   service's ``__init__`` requires a non-str-bindable arg (``session``,
   ``config``, …).

Subclasses (FastAPI, and later Flask/Sanic/Tornado/Django) only set
``framework`` and ``_ROUTE_DECORATORS`` and override :meth:`matches` with the
framework's import check. Most logic lives here so the route-walk is
consistent across web frameworks.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.detectors.base import DetectedTask, Detector
from ai_eval.inference.signatures import (
    ImportInfo,
    find_callable_defs,
)


class WebServiceDetector(Detector):
    """Base for decorator-registered route-handler detectors.

    Subclasses set:
      * ``framework`` — the framework string surfaced in ``frameworks_seen``
        and used by ``--frameworks`` filter.
      * ``_ROUTE_DECORATORS`` — the trailing decorator method names that mark
        a route registration (e.g. ``("get", "post", "put", "delete", "patch")``
        for FastAPI).

    and override :meth:`matches` to gate on the framework's import. The
    default :meth:`extract` walks module-level ``def``/``async def`` nodes for
    a decorator list containing ``@<receiver>.<route_method>(...)`` where
    ``<route_method>`` is in ``_ROUTE_DECORATORS``. The handler's bare
    function name becomes the task ``entry`` and ``name`` (route handlers are
    module-level functions, not class methods).
    """

    framework: str = "web"
    _ROUTE_DECORATORS: tuple[str, ...] = ()

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        # Subclasses override with the framework-specific import check.
        return False

    def _decorator_route_method(self, dec: ast.expr) -> str | None:
        """Return the route method name if ``dec`` is a ``@x.<route>(...)``
        decorator whose ``<route>`` is in ``_ROUTE_DECORATORS``.

        Handles both the call form (``@app.get("/path")`` — an ``ast.Call``
        whose ``func`` is an ``ast.Attribute``) and the bare attribute form
        (rare: ``@app.get`` without a call). Returns ``None`` for any other
        shape so non-route decorators (``@staticmethod``, ``@lru_cache``,
        custom markers) are skipped.
        """
        node: ast.expr = dec
        if isinstance(node, ast.Call):
            node = node.func
        if isinstance(node, ast.Attribute) and node.attr in self._ROUTE_DECORATORS:
            return node.attr
        return None

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
        # ``calls`` is unused here (route detection walks decorators, not
        # call sites) but the signature matches the Detector contract so
        # ``scan_repo`` can pass pre-computed lists without special-casing.
        del calls
        _defs = defs if defs is not None else find_callable_defs(tree)
        # Only module-level defs can be route handlers: ``find_callable_defs``
        # also returns top-level class methods, but FastAPI route handlers are
        # always module-level functions (``@app.get`` decorates a ``def`` at
        # module scope, not a method body). Filter to bare-name defs.
        rel = file_path.relative_to(project_root).as_posix()
        out: list[DetectedTask] = []
        seen: set[str] = set()
        for fn in _defs:
            # Skip dotted ``Class.method`` entries — route handlers are
            # module-level functions (no ``.`` in the name).
            if "." in fn.name:
                continue
            route_method = None
            for dec in fn.decorator_list:
                route_method = self._decorator_route_method(dec)
                if route_method is not None:
                    break
            if route_method is None:
                continue
            name = fn.name
            if name in seen:
                continue
            seen.add(name)
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type="chat",
                    file_path=rel,
                    entry=name,
                    inputs=["query"],
                    outputs=["response"],
                    evidence=[f"{self.framework} route {route_method} at {rel}:{fn.lineno}"],
                    top_level=True,
                )
            )
        return out


__all__ = ["WebServiceDetector"]

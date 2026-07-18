"""Detect FastAPI route handlers as top-level entry points.

FastAPI registers routes via decorators on module-level functions::

    from fastapi import FastAPI
    app = FastAPI()

    @app.post("/chat")
    async def chat(req): ...

    # or via an APIRouter:
    from fastapi import APIRouter
    router = APIRouter()

    @router.get("/items")
    async def list_items(): ...

Each route handler is the end-to-end entry point the user cares about
(AGENTS.md §1): ``ai-evals run`` executes the handler once per golden example
and scores its internal calls (DAO/service/evaluator methods) from the
captured trace via ``node_metrics``. The handler is emitted with
``top_level=True``; the internal methods it calls are demoted separately by
Layer 3 peer-reach + signature inspection.

The detector is a thin subclass of :class:`WebServiceDetector` — the route
walk, dedup, and task emission live in the base so Flask/Sanic/Tornado can
reuse them later.
"""

from __future__ import annotations

import ast

from ai_eval.inference.detectors.web import WebServiceDetector
from ai_eval.inference.signatures import ImportInfo, has_import_prefix


class FastAPIDetector(WebServiceDetector):
    """FastAPI route-handler detector.

    ``matches`` gates on a ``fastapi`` import prefix (covers both
    ``from fastapi import FastAPI`` and ``from fastapi import APIRouter``).
    ``extract`` (inherited from :class:`WebServiceDetector`) walks the module's
    top-level ``def``/``async def`` nodes for a decorator whose trailing
    attribute is one of the HTTP verbs FastAPI registers (``get``/``post``/
    ``put``/``delete``/``patch``).
    """

    framework = "fastapi"
    _ROUTE_DECORATORS = ("get", "post", "put", "delete", "patch")

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        del tree  # import-gated only, mirroring OpenAI/LangChain detectors
        return has_import_prefix(imports, "fastapi")


__all__ = ["FastAPIDetector"]

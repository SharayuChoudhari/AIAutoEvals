"""Detect pgvector retriever usage via psycopg2/3, asyncpg, SQLAlchemy, or pgvector."""

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

_DB_EXEC_METHODS = {
    "execute",
    "fetch",
    "fetchrow",
    "fetchval",
    "fetchone",
    "fetchall",
    "query",
    "search",
    "similarity_search",
}

_VECTOR_OPERATORS = ("<->", "<=>", "<#>")


def _call_str_args(call: ast.Call) -> list[str]:
    """Collect the string-valued positional and keyword arguments of a call."""
    out: list[str] = []
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value)
    for kw in call.keywords:
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            out.append(kw.value.value)
    return out


def _has_vector_operator(call: ast.Call) -> bool:
    """Return True if any string argument of call contains a pgvector distance operator."""
    return any(any(op in s for op in _VECTOR_OPERATORS) for s in _call_str_args(call))


def _is_orm_vector_op_call(chain: list[str], call: ast.Call) -> bool:
    """Detect the SQLAlchemy/SQLModel ORM vector-distance pattern::

        Model.column.op("<=>")(query_vec)

    The ``op`` call carries the pgvector operator as a string argument, and is
    not itself a DB exec method. This is the dominant pattern in repos that
    use pgvector through an ORM (SQLModel/SQLAlchemy) instead of raw SQL, where
    the ``<=>`` operator never appears in a ``cur.execute(...)`` string.
    """
    if not chain or chain[-1] != "op":
        return False
    return _has_vector_operator(call)


class PGVectorDetector(Detector):
    framework = "pgvector"

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        return (
            has_import_prefix(imports, "psycopg2")
            or has_import_prefix(imports, "psycopg")
            or has_import_prefix(imports, "asyncpg")
            or has_import_prefix(imports, "sqlalchemy")
            or has_import_prefix(imports, "pgvector")
        )

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
        has_direct_pgvector = has_import_prefix(imports, "pgvector")
        out: list[DetectedTask] = []
        seen: set[str] = set()
        for call in _calls:
            chain = attr_chain(call.func)
            if not chain:
                continue
            method = chain[-1]
            # Two vector-retrieval shapes:
            #   (a) a DB exec method whose own string arg carries a pgvector
            #       operator (raw SQL via psycopg/asyncpg), or — for a direct
            #       pgvector import — a query/search/similarity_search call.
            #   (b) the SQLAlchemy/SQLModel ORM shape where the operator lives
            #       in a separate ``Model.column.op("<=>")(...)`` call that is
            #       never an exec method. Catch it here so ORM-based pgvector
            #       retrieval is detected without raw SQL.
            is_orm_op = _is_orm_vector_op_call(chain, call)
            if not is_orm_op and method not in _DB_EXEC_METHODS:
                continue
            if is_orm_op:
                is_vector_call = True
            elif has_direct_pgvector and method in {"query", "search", "similarity_search"}:
                # For direct pgvector imports, query/search alone is sufficient
                # signal; the operator need not appear in a SQL string.
                is_vector_call = True
            else:
                is_vector_call = _has_vector_operator(call)
            if not is_vector_call:
                continue
            entry = enclosing_def_name(call, _defs)
            name = entry or f"{file_path.stem}_retriever"
            if name in seen:
                continue
            seen.add(name)
            evidence_chain = ".".join(chain)
            evidence = [f"pgvector {evidence_chain} at {rel}:{call.lineno}"]
            if is_orm_op:
                evidence.append(f"pgvector orm vector operator at {rel}:{call.lineno}")
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type="rag",
                    file_path=rel,
                    entry=entry,
                    inputs=["query", "embedding"],
                    outputs=["documents", "distances"],
                    evidence=evidence,
                )
            )
        return out


__all__ = ["PGVectorDetector"]

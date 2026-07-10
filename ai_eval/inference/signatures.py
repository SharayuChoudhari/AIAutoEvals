"""Reusable import + call-pattern matchers used by detectors."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ImportInfo:
    """Normalized record for one `import` / `from ... import` statement."""

    module: str             # the fully-qualified module being imported
    name: str               # symbol bound in the local namespace
    alias: str | None       # the actual local name (e.g. `np` for `numpy as np`)


def collect_imports(tree: ast.AST) -> list[ImportInfo]:
    """Walk an AST and return every import in a normalized form."""
    out: list[ImportInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(
                    ImportInfo(
                        module=alias.name,
                        name=alias.name.split(".")[0],
                        alias=alias.asname,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                out.append(
                    ImportInfo(
                        module=module,
                        name=alias.name,
                        alias=alias.asname,
                    )
                )
    return out


def has_import_prefix(imports: list[ImportInfo], prefix: str) -> bool:
    """True if any import's module starts with `prefix` (dotted)."""
    return any(
        info.module == prefix or info.module.startswith(prefix + ".")
        for info in imports
    )


def attr_chain(node: ast.AST) -> list[str]:
    """Return the dotted attribute chain of `node`.

    Examples:
      ``a.b.c``                → ``['a', 'b', 'c']``
      ``OpenAI().chat.create`` → ``['<call>', 'chat', 'create']``
      ``foo[0].bar``           → ``['<sub>', 'bar']``

    The leading sentinel makes it easy for detectors to match a trailing suffix
    (e.g. ``chain[-3:] == ['chat', 'completions', 'create']``) without caring
    about the receiver expression. Returns ``[]`` only for non-attribute nodes
    where no trailing names can be identified.
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    elif isinstance(current, ast.Call):
        parts.append("<call>")
    elif isinstance(current, ast.Subscript):
        parts.append("<sub>")
    else:
        # Unknown receiver; only return the trailing attribute path (if any).
        if not parts:
            return []
    return list(reversed(parts))


def iter_calls(tree: ast.AST):
    """Yield every `ast.Call` node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def find_callable_defs(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return all top-level `def` / `async def` in a module (no nested defs)."""
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(node)
    return defs


def enclosing_def_name(
    call: ast.AST,
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> str | None:
    """Find the function def that lexically contains `call`.

    Pragmatic: matches by line range. Good enough for repo-scan heuristics.
    """
    line = getattr(call, "lineno", None)
    if line is None:
        return None
    for fn in defs:
        end = getattr(fn, "end_lineno", None) or fn.lineno
        if fn.lineno <= line <= end:
            return fn.name
    return None


# ---------------------------------------------------------------------------
# OpenAI-specific predicates (shared between openai_chat and openai_tools
# detectors so they stay in sync as the partition key between them).
# ---------------------------------------------------------------------------

#: Kwarg names that indicate a chat.completions.create call is doing tool/
#: function calling.  Both detectors MUST use this same constant so they
#: remain mutually exclusive: chat skips calls where this is True; tools
#: skips calls where this is False.
OPENAI_TOOL_KWARGS: frozenset[str] = frozenset({"tools", "functions", "tool_choice"})


def is_openai_completions_create(call: ast.Call) -> bool:
    """True for ``<any>.chat.completions.create(...)`` or legacy
    ``<any>.ChatCompletion.create(...)`` regardless of receiver expression."""
    chain = attr_chain(call.func)
    if len(chain) >= 3 and chain[-3:] == ["chat", "completions", "create"]:
        return True
    if len(chain) >= 2 and chain[-2:] == ["ChatCompletion", "create"]:
        return True
    return False


def is_openai_responses_create(call: ast.Call) -> bool:
    """True for ``<any>.responses.create(...)`` regardless of receiver expression.

    The receiver is typically ``client``, ``self.client``, or ``OpenAI()``
    (yielding the ``<call>`` sentinel), so we match only on the trailing two
    attribute names. This is deliberately distinct from
    :func:`is_openai_completions_create`, which targets the
    ``chat.completions.create`` / legacy ``ChatCompletion.create`` suffixes, so
    the two predicates never overlap.
    """
    chain = attr_chain(call.func)
    return len(chain) >= 2 and chain[-2:] == ["responses", "create"]


def has_openai_tool_kwarg(call: ast.Call) -> bool:
    """True when the call uses at least one tool-calling kwarg."""
    return any(kw.arg in OPENAI_TOOL_KWARGS for kw in call.keywords)


__all__ = [
    "OPENAI_TOOL_KWARGS",
    "ImportInfo",
    "attr_chain",
    "collect_imports",
    "enclosing_def_name",
    "find_callable_defs",
    "has_import_prefix",
    "has_openai_tool_kwarg",
    "is_openai_completions_create",
    "is_openai_responses_create",
    "iter_calls",
]

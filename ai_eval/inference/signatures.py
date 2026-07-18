"""Reusable import + call-pattern matchers used by detectors."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ai_eval.inference.types import RUNNABLE_INIT_TYPES


@dataclass(frozen=True)
class ImportInfo:
    """Normalized record for one `import` / `from ... import` statement."""

    module: str  # the fully-qualified module being imported
    name: str  # symbol bound in the local namespace
    alias: str | None  # the actual local name (e.g. `np` for `numpy as np`)


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
    return any(info.module == prefix or info.module.startswith(prefix + ".") for info in imports)


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
    """Return module-level `def`/`async def` plus direct methods of top-level
    classes (``ClassDef`` bodies). Nested defs inside methods (closures) are
    intentionally excluded so they never produce false enclosing matches.

    Each returned node carries its qualified name in ``node.name``: a bare
    function name for module-level defs, or ``Class.method`` for methods. The
    caller can recover the unqualified name via ``name.rsplit(".", 1)[-1]``.
    """
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    if isinstance(tree, ast.Module):
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(node)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Rewrite the node's name to the dotted ``Class.method``
                        # form. This is the contract every detector relies on:
                        # ``enclosing_def_name`` returns this name verbatim and
                        # detectors emit it as ``DetectedTask.entry``.
                        child.name = f"{node.name}.{child.name}"
                        defs.append(child)
    return defs


def enclosing_def_name(
    call: ast.AST,
    defs: list[ast.FunctionDef | ast.AsyncFunctionDef],
) -> str | None:
    """Find the function/method def that lexically contains `call`.

    Pragmatic: matches by line range. For methods the returned name is the
    dotted ``Class.method`` form produced by :func:`find_callable_defs`.
    Good enough for repo-scan heuristics.

    When a call lies inside multiple defs (a method body containing a nested
    closure), the **smallest enclosing** def wins so closures like LangGraph's
    ``retrieve_node``/``generate_node`` never shadow their host method.
    """
    line = getattr(call, "lineno", None)
    if line is None:
        return None
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    best_span = -1
    for fn in defs:
        end = getattr(fn, "end_lineno", None) or fn.lineno
        if fn.lineno <= line <= end:
            span = end - fn.lineno
            # Prefer the tightest enclosing def (smallest line span). This
            # picks the method over any nested closure sharing the same range.
            if best is None or span < best_span:
                best = fn
                best_span = span
    return best.name if best is not None else None


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


# ---------------------------------------------------------------------------
# __init__-signature inspection — synthesis-time demotion of IO-coupled
# class methods (AGENTS.md §1). Pure-AST; no import/execution.
# ---------------------------------------------------------------------------


def _annotation_name(node: ast.expr | None) -> str | None:
    """Flatten an annotation AST node to its dotted source-form string.

    Returns the textual form a developer would type:
      ``str``                  → ``"str"``
      ``Optional[str]``        → ``"Optional[str]"``
      ``str | None``           → ``"str | None"``
      ``Session``              → ``"Session"``
      ``asyncpg.Connection``   → ``"Connection"`` (trailing attr only, so it
                                  matches the allow-list by class name)

    ``None`` when ``node`` is absent (an unannotated param) or has a shape we
    don't bother stringifying (e.g. ``Callable[...]``). The demotion logic in
    :func:`class_init_requires_nonstr_args` treats an unannotated required
    param as NOT runnable (``cls()`` would fail on it), so ``None`` here
    correctly falls through to the demotion branch.
    """
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        # ``None`` (the singleton) and string-literal annotations
        # (``"str"`` forward refs) — render as their Python literal so a
        # ``str | None`` union's right side yields ``"None"`` and the full
        # union string ``"str | None"`` matches the allow-list.
        return repr(node.value) if node.value is None else str(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        # Use only the trailing attribute name so ``asyncpg.Connection``
        # normalizes to ``Connection`` (matched against the bare-name
        # allow-list; matches by class name, not import path).
        return node.attr
    if isinstance(node, ast.Subscript):
        # ``Optional[str]`` / ``list[str]`` → render as ``Optional[str]``.
        value = _annotation_name(node.value)
        slc = _annotation_name(node.slice)
        if value is None:
            return None
        return f"{value}[{slc}]" if slc is not None else f"{value}[?]"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        # ``str | None`` (PEP 604). Reconstruct the full union string so the
        # allow-list can match ``"str | None"`` exactly.
        left = _annotation_name(node.left)
        right = _annotation_name(node.right)
        if left is None or right is None:
            return None
        return f"{left} | {right}"
    return None


def class_init_requires_nonstr_args(tree: ast.AST, class_name: str) -> bool:
    """Return ``True`` when the class's ``__init__`` has a required positional
    or keyword arg (after ``self``) whose annotation is NOT in the
    str-bindable / runner-known allow-list (:data:`RUNNABLE_INIT_TYPES`).

    This is the synthesis-time signal that a ``Class.method`` task is
    IO-coupled (its construction needs a real backend arg like ``session``,
    ``config``, ``db``) and should be demoted to ``top_level=False`` so the
    seeder and runner skip it (AGENTS.md §1). The runner constructs dotted
    entries with a bare ``cls()`` (no args), so ANY required ``__init__`` param
    breaks construction — unless its annotation is a type the harness/runner
    already know how to build (LLM client, compiled graph) or could bind from
    the auto-seed scalar (``str`` / ``Any`` shapes).

    Pure-AST: no import or execution of the class.

    Demotion contract (over-promotion-safe):

    * **Required param (no default) with no annotation** → demotes (``True``).
      The classic IO-coupled signature: ``def __init__(self, session): ...``.
      ``cls()`` fails with ``TypeError: missing required positional argument``.
    * **Required param with annotation in the allow-list** (``str``,
      ``Optional[str]``, ``str | None``, ``Any``, LLM-client names,
      compiled-graph names) → runnable, ``False``. These are the types the
      runner/harness already know how to construct
      (see :mod:`ai_eval.inference.types`), or that a future runner path
      could bind from the scalar auto-seed. Over-demotion here would drop
      legitimate LangGraph / LangChain entries whose ``__init__`` takes a
      compiled graph or LLM client.
    * **Required param with annotation NOT in the allow-list** (``Session``,
      ``Config``, ``Connection``, …) → demotes (``True``). The harness
      doesn't construct these; ``cls()`` fails.
    * **Param with a default** → never required, never demotes (``False``).
    * **No ``__init__`` defined** (inherits ``object.__init__``) → ``False``.
      A class with no custom ``__init__`` takes no required args.
    * **No class named ``class_name`` in ``tree``** → ``False`` (can't tell,
      don't demote).

    The allow-list is :data:`ai_eval.inference.types.RUNNABLE_INIT_TYPES`:
    the union of LLM-client names, compiled-graph names, and the str/Any
    shapes the engine's ``_build_call_args`` binds from a scalar auto-seed.
    """
    if not isinstance(tree, ast.Module):
        return False
    cls_node: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cls_node = node
            break
    if cls_node is None:
        return False
    init_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for child in cls_node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name == "__init__":
                init_node = child
                break
    if init_node is None:
        return False
    args = init_node.args
    # Build the (arg, default) pairs. Positional-or-keyword args are aligned
    # with the trailing ``args.defaults`` list; kw-only args with
    # ``kw_defaults`` (``None`` sentinel = no default).
    pos_args = list(args.posonlyargs) + list(args.args)
    pos_defaults = list(args.defaults)
    # The last ``len(pos_defaults)`` positional args have the defaults.
    num_no_default_pos = len(pos_args) - len(pos_defaults)
    pos_pairs: list[tuple[ast.arg, ast.expr | None]] = []
    for i, a in enumerate(pos_args):
        default = pos_defaults[i - num_no_default_pos] if i >= num_no_default_pos else None
        pos_pairs.append((a, default))
    kw_pairs: list[tuple[ast.arg, ast.expr | None]] = []
    for a, d in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        kw_pairs.append((a, d))
    for arg, default in pos_pairs + kw_pairs:
        if arg.arg == "self":
            continue
        if default is not None:
            continue
        ann = _annotation_name(arg.annotation)
        # Required param. Demote unless its annotation is in the runnable
        # allow-list (LLM client / graph / str / Any). An unannotated required
        # param (``session``) demotes: ``cls()`` can't satisfy it and the
        # harness only patches method reads, not ``__init__`` args.
        if ann in RUNNABLE_INIT_TYPES:
            continue
        return True
    return False


__all__ = [
    "OPENAI_TOOL_KWARGS",
    "ImportInfo",
    "attr_chain",
    "class_init_requires_nonstr_args",
    "collect_imports",
    "enclosing_def_name",
    "find_callable_defs",
    "has_import_prefix",
    "has_openai_tool_kwarg",
    "is_openai_completions_create",
    "is_openai_responses_create",
    "iter_calls",
]

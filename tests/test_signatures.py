"""Unit tests for the class-body entry resolution contract (plan D2/Step 1).

``find_callable_defs`` descends into ``ClassDef`` bodies (collecting methods as
dotted ``Class.method`` names) but excludes nested closures inside methods.
``enclosing_def_name`` returns the dotted name for methods and picks the
tightest enclosing def when a call sits inside a method that itself contains a
nested closure.
"""

from __future__ import annotations

import ast

from ai_eval.inference.signatures import enclosing_def_name, find_callable_defs


def _defs(source: str) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return find_callable_defs(ast.parse(source))


def test_find_callable_defs_collects_module_level_functions() -> None:
    defs = _defs("def a():\n    pass\n\n\ndef b():\n    pass\n")
    names = [fn.name for fn in defs]
    assert names == ["a", "b"]


def test_find_callable_defs_descends_into_class_body() -> None:
    """Methods of top-level classes are collected with dotted ``Class.method``
    names so detectors emit resolvable entries (plan D2)."""
    defs = _defs(
        "class ChatService:\n"
        "    def process_query(self, q):\n"
        "        return q\n"
        "    async def stream(self, q):\n"
        "        yield q\n"
    )
    names = [fn.name for fn in defs]
    assert names == ["ChatService.process_query", "ChatService.stream"]


def test_find_callable_defs_excludes_nested_closures() -> None:
    """A closure defined inside a method is NOT collected as a callable def —
    it would otherwise produce false enclosing matches (LangGraph
    ``retrieve_node``/``generate_node`` inside a method body)."""
    defs = _defs(
        "class Workflow:\n"
        "    def run(self, state):\n"
        "        def retrieve_node(s):\n"
        "            return s\n"
        "        graph = build_graph(retrieve_node)\n"
        "        return graph.invoke(state)\n"
    )
    names = [fn.name for fn in defs]
    # Only the method, not the inner closure.
    assert names == ["Workflow.run"]


def test_find_callable_defs_handles_nested_classes_top_level_only() -> None:
    """A class nested inside a method is not descended into (out of scope for
    R2's top-level ``__init__`` attr tracking). Only the enclosing method and
    top-level defs are returned."""
    defs = _defs(
        "def build():\n"
        "    class Inner:\n"
        "        def m(self):\n"
        "            return 1\n"
        "    return Inner\n"
    )
    names = [fn.name for fn in defs]
    assert names == ["build"]


def test_enclosing_def_name_returns_dotted_for_method() -> None:
    """A call inside a method resolves to the dotted ``Class.method`` form."""
    source = (
        "class Svc:\n"
        "    def process_query(self, q):\n"
        "        return client.chat.completions.create(messages=q)\n"
    )
    tree = ast.parse(source)
    defs = find_callable_defs(tree)
    # The create(...) call is the only Call node.
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    assert enclosing_def_name(call, defs) == "Svc.process_query"


def test_enclosing_def_name_returns_bare_name_for_module_function() -> None:
    source = "def summarize(text):\n    return client.chat.completions.create(input=text)\n"
    tree = ast.parse(source)
    defs = find_callable_defs(tree)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    assert enclosing_def_name(call, defs) == "summarize"


def test_enclosing_def_name_picks_tightest_enclosing_def() -> None:
    """When a call sits inside a method body that contains a nested closure,
    the method (smallest span) wins over any broader match. This guards against
    closures shadowing their host method.

    Here the call lives inside the method ``run``; even though a nested def
    ``retrieve_node`` exists in the same body, ``retrieve_node`` is excluded
    from ``defs`` (closures aren't collected), so ``run`` is the sole match.
    """
    source = (
        "class Workflow:\n"
        "    def run(self, state):\n"
        "        def retrieve_node(s):\n"
        "            return s\n"
        "        return graph.invoke(state)\n"
    )
    tree = ast.parse(source)
    defs = find_callable_defs(tree)
    # The graph.invoke(...) call.
    call = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "invoke"
    )
    assert enclosing_def_name(call, defs) == "Workflow.run"


def test_enclosing_def_name_returns_none_for_module_level_call() -> None:
    """A call at module scope (no enclosing def) returns ``None`` so detectors
    fall back to the file-stem naming convention."""
    source = "result = client.chat.completions.create(messages=[])\n"
    tree = ast.parse(source)
    defs = find_callable_defs(tree)
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    assert enclosing_def_name(call, defs) is None


def test_enclosing_def_name_async_method() -> None:
    """``async def`` methods resolve identically to sync methods (D3 needs the
    dotted entry to wrap async entries)."""
    source = (
        "class Svc:\n"
        "    async def astream(self, q):\n"
        "        return await client.chat.completions.create(messages=q)\n"
    )
    tree = ast.parse(source)
    defs = find_callable_defs(tree)
    names = [fn.name for fn in defs]
    assert names == ["Svc.astream"]
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call))
    assert enclosing_def_name(call, defs) == "Svc.astream"

"""Unit tests for the OpenAI Responses-API workflow detector + predicate."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.ast_scan import scan_repo
from ai_eval.inference.detectors.openai_chat import OpenAIChatDetector
from ai_eval.inference.detectors.openai_responses import OpenAIResponsesDetector
from ai_eval.inference.detectors.openai_tools import OpenAIToolsDetector
from ai_eval.inference.signatures import (
    collect_imports,
    find_callable_defs,
    is_openai_completions_create,
    is_openai_responses_create,
    iter_calls,
)


def _parse(source: str) -> tuple[ast.AST, list]:
    tree = ast.parse(source)
    return tree, collect_imports(tree)


def test_is_openai_responses_create_matches_client_responses(tmp_path: Path) -> None:
    source = (
        "from openai import OpenAI\n"
        "def f():\n"
        "    client = OpenAI()\n"
        "    return client.responses.create(input=[])\n"
    )
    tree = ast.parse(source)
    calls = list(iter_calls(tree))
    responses_calls = [c for c in calls if is_openai_responses_create(c)]
    assert len(responses_calls) == 1
    # The completions predicate must NOT also match a responses.create call.
    assert not any(is_openai_completions_create(c) for c in responses_calls)


def test_is_openai_responses_create_does_not_match_completions() -> None:
    source = "client.chat.completions.create(messages=[])\n"
    tree = ast.parse(source)
    calls = list(iter_calls(tree))
    assert not any(is_openai_responses_create(c) for c in calls)


def test_responses_detector_tools_kwarg_emits_workflow(tmp_path: Path) -> None:
    path = tmp_path / "wf.py"
    source = (
        "from openai import OpenAI\n"
        "\n"
        "def book_appointment(messages):\n"
        "    client = OpenAI()\n"
        "    return client.responses.create(\n"
        "        model='gpt-5', input=messages, tools=[]\n"
        "    )\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    responses = OpenAIResponsesDetector()
    assert responses.matches(tree, imports)
    tasks = responses.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tasks) == 1
    assert tasks[0].type == "workflow"
    assert tasks[0].entry == "book_appointment"
    assert tasks[0].framework == "openai"
    assert tasks[0].inputs == ["messages", "tools"]
    assert tasks[0].outputs == ["output", "tool_calls"]
    assert any("responses.create(tools=...)" in e for e in tasks[0].evidence)


def test_responses_detector_no_tools_kwarg_does_not_emit(tmp_path: Path) -> None:
    """A non-tool Responses call must fall through to the chat detector —
    the three OpenAI detectors stay mutually exclusive by kwarg presence."""
    path = tmp_path / "chat.py"
    source = (
        "from openai import OpenAI\n"
        "\n"
        "def plain_chat(messages):\n"
        "    client = OpenAI()\n"
        "    return client.responses.create(model='gpt-5', input=messages)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    responses = OpenAIResponsesDetector()
    tasks = responses.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert tasks == []

    # And the chat detector should NOT claim a responses.create call either
    # (it only matches chat.completions.create). This leaves the non-tool
    # responses call unemitted by design — chat owns completions, not responses.
    chat = OpenAIChatDetector()
    chat_tasks = chat.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert chat_tasks == []


def test_responses_detector_does_not_claim_completions(tmp_path: Path) -> None:
    """A chat.completions.create(tools=...) call must be owned by the tools
    detector, never the responses detector."""
    path = tmp_path / "agent.py"
    source = (
        "from openai import OpenAI\n"
        "\n"
        "def support_agent(messages):\n"
        "    client = OpenAI()\n"
        "    return client.chat.completions.create(\n"
        "        model='gpt-4o-mini', messages=messages, tools=[]\n"
        "    )\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    responses = OpenAIResponsesDetector()
    tasks = responses.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert tasks == []

    tools = OpenAIToolsDetector()
    tools_tasks = tools.extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    assert len(tools_tasks) == 1
    assert tools_tasks[0].type == "tool_calling"


def test_responses_detector_dedups_per_enclosing_def(tmp_path: Path) -> None:
    """Two responses.create(tools=...) calls in the same function emit ONE task."""
    path = tmp_path / "wf.py"
    source = (
        "from openai import OpenAI\n"
        "\n"
        "def run(msgs):\n"
        "    client = OpenAI()\n"
        "    a = client.responses.create(model='gpt-5', input=msgs, tools=[])\n"
        "    b = client.responses.create(model='gpt-5', input=a, tools=[])\n"
        "    return b\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = OpenAIResponsesDetector().extract(
        tree, imports, path, tmp_path, calls=calls, defs=defs
    )
    assert len(tasks) == 1
    assert tasks[0].entry == "run"


def test_responses_detector_module_level_emits_fallback_name(tmp_path: Path) -> None:
    """A module-level responses.create call has no enclosing def; the name
    falls back to ``<stem>_workflow``."""
    path = tmp_path / "service.py"
    source = (
        "from openai import OpenAI\n"
        "client = OpenAI()\n"
        "result = client.responses.create(model='gpt-5', input=[], tools=[])\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = OpenAIResponsesDetector().extract(
        tree, imports, path, tmp_path, calls=calls, defs=defs
    )
    assert len(tasks) == 1
    assert tasks[0].name == "service_workflow"
    assert tasks[0].entry is None


def test_responses_detector_self_client_receiver(tmp_path: Path) -> None:
    """The appointment_helper pattern: ``self.client.responses.create(tools=...)``
    where the receiver is an attribute, not a bare Name. attr_chain yields a
    leading Name (``self``), so the trailing two-name check still fires.

    Note: ``_call_model`` is a *method* inside a class. ``find_callable_defs``
    returns only top-level defs (documented behavior shared by all detectors),
    so ``entry`` is ``None`` here — the name falls back to ``<stem>_workflow``.
    This matches how the existing tools/chat detectors behave on methods."""
    path = tmp_path / "svc.py"
    source = (
        "from openai import OpenAI\n"
        "\n"
        "class Svc:\n"
        "    def _call_model(self, state):\n"
        "        return self.client.responses.create(\n"
        "            model='gpt-5', input=state['messages'], tools=self._tool_schemas()\n"
        "        )\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tasks = OpenAIResponsesDetector().extract(
        tree, imports, path, tmp_path, calls=calls, defs=defs
    )
    assert len(tasks) == 1
    assert tasks[0].type == "workflow"
    assert tasks[0].name == "svc_workflow"
    assert tasks[0].entry is None  # method, not a top-level def


def test_responses_detector_does_not_fire_without_openai_import(tmp_path: Path) -> None:
    """matches() gates on the openai import prefix — a bare SDK-less file
    should not be scanned even if the call shape happens to match."""
    path = tmp_path / "no_import.py"
    source = (
        "def f(client):\n"
        "    return client.responses.create(input=[], tools=[])\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    responses = OpenAIResponsesDetector()
    assert not responses.matches(tree, imports)


def test_three_way_openai_mutual_exclusion(tmp_path: Path) -> None:
    """Exactly one of the three OpenAI detectors fires per call (plan Risks):
      - completions.create(tools=...)        → tools detector (tool_calling)
      - completions.create(...) (no tools)   → chat detector (chat)
      - responses.create(tools=...)          → responses detector (workflow)
      - responses.create(...) (no tools)     → none of the three
    No call is claimed by two detectors, and the responses detector never
    claims a completions call (and vice versa).
    """
    path = tmp_path / "svc.py"
    # One of each shape in distinct functions so dedup keeps them separate.
    source = (
        "from openai import OpenAI\n"
        "\n"
        "def f_tools(msgs):\n"
        "    return OpenAI().chat.completions.create(messages=msgs, tools=[])\n"
        "\n"
        "def f_chat(msgs):\n"
        "    return OpenAI().chat.completions.create(messages=msgs)\n"
        "\n"
        "def f_responses_tools(msgs):\n"
        "    return OpenAI().responses.create(input=msgs, tools=[])\n"
        "\n"
        "def f_responses_plain(msgs):\n"
        "    return OpenAI().responses.create(input=msgs)\n"
    )
    path.write_text(source, encoding="utf-8")
    tree, imports = _parse(source)
    calls = list(iter_calls(tree))
    defs = find_callable_defs(tree)

    tools = OpenAIToolsDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    chat = OpenAIChatDetector().extract(tree, imports, path, tmp_path, calls=calls, defs=defs)
    responses = OpenAIResponsesDetector().extract(
        tree, imports, path, tmp_path, calls=calls, defs=defs
    )

    # Each detector claims only its partition.
    assert {t.entry for t in tools} == {"f_tools"}
    assert {t.entry for t in chat} == {"f_chat"}
    assert {t.entry for t in responses} == {"f_responses_tools"}
    # The plain responses call is owned by nobody (chat only matches completions).
    assert "f_responses_plain" not in {t.entry for t in tools + chat + responses}

    # Names don't collide across detectors for the same file — each entry is
    # claimed by exactly one detector, so the union has 3 distinct entries.
    all_entries = [t.entry for t in tools + chat + responses]
    assert sorted(all_entries) == ["f_chat", "f_responses_tools", "f_tools"]


def test_scan_repo_finds_responses_workflow(tmp_path: Path) -> None:
    """End-to-end: a repo with a responses.create(tools=...) call auto-detects
    a workflow task, and project_type classifies to `workflow`."""
    (tmp_path / "wf.py").write_text(
        (
            "from openai import OpenAI\n"
            "\n"
            "def run(msgs):\n"
            "    client = OpenAI()\n"
            "    return client.responses.create(model='gpt-5', input=msgs, tools=[])\n"
        ),
        encoding="utf-8",
    )
    result = scan_repo(tmp_path)
    assert any(t.type == "workflow" for t in result.tasks)
    wf = next(t for t in result.tasks if t.type == "workflow")
    assert wf.entry == "run"
    assert wf.framework == "openai"

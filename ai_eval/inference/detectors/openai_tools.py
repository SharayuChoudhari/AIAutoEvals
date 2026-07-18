"""Detect OpenAI tool / function calling (chat.completions.create with tools=...)."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.detectors.base import DetectedTask, Detector
from ai_eval.inference.signatures import (
    ImportInfo,
    enclosing_def_name,
    find_callable_defs,
    has_import_prefix,
    has_openai_tool_kwarg,
    is_openai_completions_create,
    iter_calls,
)


class OpenAIToolsDetector(Detector):
    framework = "openai"

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        return has_import_prefix(imports, "openai")

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
        out: list[DetectedTask] = []
        seen: set[str] = set()
        for call in _calls:
            if not is_openai_completions_create(call):
                continue
            if not has_openai_tool_kwarg(call):
                # OpenAIChatDetector owns non-tool calls.
                continue
            entry = enclosing_def_name(call, _defs)
            name = entry or f"{file_path.stem}_tools"
            if name in seen:
                continue
            seen.add(name)
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type="tool_calling",
                    file_path=rel,
                    entry=entry,
                    inputs=["messages", "tools"],
                    outputs=["tool_calls", "content"],
                    evidence=[f"openai chat.completions.create(tools=...) at {rel}:{call.lineno}"],
                )
            )
        return out


__all__ = ["OpenAIToolsDetector"]

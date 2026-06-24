"""Detect OpenAI tool / function calling (chat.completions.create with tools=...)."""

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


def _is_chat_completions_create(call: ast.Call) -> bool:
    chain = attr_chain(call.func)
    if len(chain) >= 3 and chain[-3:] == ["chat", "completions", "create"]:
        return True
    if len(chain) >= 2 and chain[-2:] == ["ChatCompletion", "create"]:
        return True
    return False


def _has_tools(call: ast.Call) -> bool:
    return any(kw.arg in {"tools", "functions", "tool_choice"} for kw in call.keywords)


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
    ) -> list[DetectedTask]:
        defs = find_callable_defs(tree)
        rel = file_path.relative_to(project_root).as_posix()
        out: list[DetectedTask] = []
        seen: set[str] = set()
        for call in iter_calls(tree):
            if not _is_chat_completions_create(call):
                continue
            if not _has_tools(call):
                continue
            entry = enclosing_def_name(call, defs)
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
                    evidence=[
                        f"openai chat.completions.create(tools=...) at {rel}:{call.lineno}"
                    ],
                )
            )
        return out


__all__ = ["OpenAIToolsDetector"]

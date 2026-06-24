"""Detect LangChain usage (chat models, chains, agents, retrievers)."""

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

_AGENTIC_HINTS = {"AgentExecutor", "create_react_agent", "create_tool_calling_agent"}
_RETRIEVER_HINTS = {"as_retriever", "invoke"}  # retriever.invoke + vectorstore.as_retriever


def _is_langchain_imports(imports: list[ImportInfo]) -> bool:
    return (
        has_import_prefix(imports, "langchain")
        or has_import_prefix(imports, "langchain_core")
        or has_import_prefix(imports, "langchain_community")
        or has_import_prefix(imports, "langchain_openai")
    )


class LangChainDetector(Detector):
    framework = "langchain"

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        return _is_langchain_imports(imports)

    def extract(
        self,
        tree: ast.AST,
        imports: list[ImportInfo],
        file_path: Path,
        project_root: Path,
    ) -> list[DetectedTask]:
        defs = find_callable_defs(tree)
        rel = file_path.relative_to(project_root).as_posix()
        agentic = any(
            info.name in _AGENTIC_HINTS or info.module.endswith("agents")
            for info in imports
        )
        retriever = any(info.name == "as_retriever" for info in imports) or any(
            "vectorstores" in info.module for info in imports
        )

        out: list[DetectedTask] = []
        seen: set[str] = set()
        for call in iter_calls(tree):
            chain = attr_chain(call.func)
            if not chain:
                continue
            last = chain[-1]
            # Treat `.invoke(...)` or `.ainvoke(...)` on a langchain object as a task.
            if last not in {"invoke", "ainvoke", "run", "arun", "stream", "astream"}:
                continue
            entry = enclosing_def_name(call, defs)
            name = entry or f"{file_path.stem}_chain"
            if name in seen:
                continue
            seen.add(name)
            if agentic:
                kind = "agent"
                inputs, outputs = ["input"], ["output", "intermediate_steps"]
            elif retriever:
                kind = "rag"
                inputs, outputs = ["query"], ["documents"]
            else:
                kind = "chat"
                inputs, outputs = ["input"], ["output"]
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type=kind,
                    file_path=rel,
                    entry=entry,
                    inputs=inputs,
                    outputs=outputs,
                    evidence=[f"langchain {'.'.join(chain)} at {rel}:{call.lineno}"],
                )
            )
        return out


__all__ = ["LangChainDetector"]

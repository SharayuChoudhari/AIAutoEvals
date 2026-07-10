"""Detect LangChain usage (chat models, chains, agents, retrievers)."""

from __future__ import annotations

import ast
from pathlib import Path

from ai_eval.inference.detectors._langchain_retrieval import (
    is_retriever_call_site,
    retrieval_chain_imported,
    vectorstore_imported,
    vectorstore_provider_imported,
)
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

#: Chain invoke / run / stream verbs — the calls a chat chain or agent emits.
_INVOKE_METHODS = {"invoke", "ainvoke", "run", "arun", "stream", "astream"}


def _is_langchain_imports(imports: list[ImportInfo]) -> bool:
    if (
        has_import_prefix(imports, "langchain")
        or has_import_prefix(imports, "langchain_core")
        or has_import_prefix(imports, "langchain_community")
        or has_import_prefix(imports, "langchain_openai")
    ):
        return True
    # Vector-store provider packages (langchain_pinecone, langchain_weaviate,
    # ...) whose top-level prefix is not one of the bare ``langchain`` modules
    # above but are still LangChain-family integrations.
    return vectorstore_provider_imported(imports)


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
        *,
        calls: list[ast.Call] | None = None,
        defs: list[ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
    ) -> list[DetectedTask]:
        _calls = calls if calls is not None else list(iter_calls(tree))
        _defs = defs if defs is not None else find_callable_defs(tree)
        rel = file_path.relative_to(project_root).as_posix()
        agentic = any(
            info.name in _AGENTIC_HINTS or info.module.endswith("agents")
            for info in imports
        )
        retriever_imports = (
            any(info.name == "as_retriever" for info in imports)
            or vectorstore_imported(imports)
            or retrieval_chain_imported(imports)
            or vectorstore_provider_imported(imports)
        )

        # Pre-classify each candidate call once. ``kind`` is decided per
        # enclosing function from the *union* of its candidate calls (rag if
        # any retrieval signal is present, agent if agentic, else chat), so it
        # no longer depends on source order.
        candidates: list[tuple[ast.Call, list[str], str | None, bool]] = []
        for call in _calls:
            chain = attr_chain(call.func)
            if not chain:
                continue
            last = chain[-1]
            # Compute the cheap test first and only run the retriever heuristic
            # when this isn't already an invoke/run/stream call.
            invoke_call = last in _INVOKE_METHODS
            retriever_call = (
                is_retriever_call_site(chain, call) if not invoke_call else False
            )
            # Emit on a chain invoke/run/stream call OR a direct retrieval
            # call site (similarity_search / get_relevant_documents) whose
            # method name isn't an invoke verb.
            if not invoke_call and not retriever_call:
                continue
            entry = enclosing_def_name(call, _defs)
            candidates.append((call, chain, entry, retriever_call))

        # Group candidates by enclosing function. For each function pick one
        # representative call to emit, so the per-name dedup is order-
        # independent: for a rag function prefer the actual retrieval call
        # site (so evidence/line points at the retrieval, not an unrelated
        # chat invoke that happened to come first); otherwise keep the first
        # candidate in source order.
        by_func: dict[str | None, list[tuple[ast.Call, list[str], bool]]] = {}
        for call, chain, entry, retriever_call in candidates:
            by_func.setdefault(entry, []).append((call, chain, retriever_call))

        # Per-function rag flag: rag if any candidate in that function is a
        # retrieval call site (the union over calls, order-independent).
        func_is_rag: dict[str | None, bool] = {
            entry: (any(rc for _, _, rc in items) or retriever_imports)
            for entry, items in by_func.items()
        }

        out: list[DetectedTask] = []
        seen: set[str] = set()
        for entry, items in by_func.items():
            is_rag = func_is_rag[entry]
            if agentic:
                kind = "agent"
                inputs, outputs = ["input"], ["output", "intermediate_steps"]
                name = entry or f"{file_path.stem}_chain"
            elif is_rag:
                kind = "rag"
                inputs, outputs = ["query"], ["documents"]
                name = entry or f"{file_path.stem}_retriever"
            else:
                kind = "chat"
                inputs, outputs = ["input"], ["output"]
                name = entry or f"{file_path.stem}_chain"
            if name in seen:
                continue
            seen.add(name)
            # Representative call: for rag, the retrieval call site; else first.
            rep = next((c for c, _, rc in items if rc), None)
            if rep is None:
                rep, rep_chain, _ = items[0]
            else:
                rep_chain = next(ch for c, ch, rc in items if c is rep)
            evidence = [f"langchain {'.'.join(rep_chain)} at {rel}:{rep.lineno}"]
            if is_rag and any(rc for _, _, rc in items):
                rc_call = next(c for c, _, rc in items if rc)
                rc_chain = next(ch for c, ch, rc in items if rc)
                evidence.append(
                    f"langchain retriever call {'.'.join(rc_chain)} at {rel}:{rc_call.lineno}"
                )
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type=kind,
                    file_path=rel,
                    entry=entry,
                    inputs=inputs,
                    outputs=outputs,
                    evidence=evidence,
                )
            )
        return out


__all__ = ["LangChainDetector"]

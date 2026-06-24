"""Detect ChromaDB retriever usage."""

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


class ChromaDBDetector(Detector):
    framework = "chromadb"

    def matches(self, tree: ast.AST, imports: list[ImportInfo]) -> bool:
        return has_import_prefix(imports, "chromadb")

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
            chain = attr_chain(call.func)
            if not chain:
                continue
            if chain[-1] not in {"query", "get"}:
                continue
            # Bias toward names that include "collection" or "client" to reduce noise.
            receiver_hint = ".".join(chain[:-1]).lower()
            if not any(h in receiver_hint for h in ("collection", "chroma", "vectorstore")):
                # Still accept if chromadb is imported and the call shape is right.
                pass
            entry = enclosing_def_name(call, defs)
            name = entry or f"{file_path.stem}_retriever"
            if name in seen:
                continue
            seen.add(name)
            out.append(
                DetectedTask(
                    name=name,
                    framework=self.framework,
                    type="rag",
                    file_path=rel,
                    entry=entry,
                    inputs=["query"],
                    outputs=["documents", "distances"],
                    evidence=[f"chromadb {'.'.join(chain)} at {rel}:{call.lineno}"],
                )
            )
        return out


__all__ = ["ChromaDBDetector"]

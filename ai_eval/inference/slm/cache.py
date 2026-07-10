"""Evidence-hash keyed response cache for SLM rubric calls.

Caches the per-task classification result under
``.ai-evals/rubric_cache/<hash>.json`` keyed by a canonical hash of the evidence
(the task's enclosing function source, its imports, and the model id). A cache
hit short-circuits the SLM call entirely, giving reproducible, free re-runs.

The cache is purely an optimization; a miss or corruption falls back to a live
SLM call. Storing JSON (not pickled objects) keeps the cache greppable and
survives instructor/litellm upgrades.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eval.config.defaults import DEFAULT_STATE_DIR


def _cache_dir(project_root: Path) -> Path:
    return project_root / DEFAULT_STATE_DIR / "rubric_cache"


def evidence_hash(
    *,
    snippet: str,
    framework_hints: list[str],
    enclosing_function: str | None,
    file_path: str,
    model: str,
    template_name: str,
) -> str:
    """SHA-256 over a canonical JSON serialization of the evidence + model."""
    payload = json.dumps(
        {
            "snippet": snippet,
            "framework_hints": sorted(framework_hints),
            "enclosing_function": enclosing_function,
            "file_path": file_path,
            "model": model,
            "template": template_name,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class ResponseCache:
    """File-backed JSON cache keyed by ``evidence_hash``."""

    def __init__(self, project_root: Path, *, enabled: bool = True) -> None:
        self._root = project_root
        self._enabled = enabled
        self.stats = CacheStats()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        path = _cache_dir(self._root) / f"{key}.json"
        if not path.is_file():
            self.stats.misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return data

    def put(self, key: str, value: dict[str, Any]) -> None:
        if not self._enabled:
            return
        path = _cache_dir(self._root) / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via a unique temp file + os.replace, so a crash or a
        # concurrent `put` on the same key never leaves a half-written file
        # that would force a live SLM call every run until overwritten.
        tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
        try:
            tmp.write_text(
                json.dumps(value, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def clear(self) -> int:
        """Remove every cached entry; return the count removed."""
        d = _cache_dir(self._root)
        if not d.is_dir():
            return 0
        n = 0
        for p in d.glob("*.json"):
            try:
                p.unlink()
                n += 1
            except OSError:
                pass
        return n


__all__ = ["CacheStats", "ResponseCache", "evidence_hash"]

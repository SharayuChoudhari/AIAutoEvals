"""Content-addressed judge-response cache.

Key = ``sha256(model_id || prompt_text || schema_id)``. Stored as JSON under
``.ai-evals/cache/judge/<sha>.json``. Deterministic at ``temperature=0`` (the
default), so a cache hit short-circuits the judge call entirely.

``--no-cache`` (``enabled=False``) skips both reads and writes; the cache
object still exists so callers don't have to branch on it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eval.config.defaults import SCHEMA_VERSION


def cache_key(model_id: str, prompt_text: str, *, schema_id: str) -> str:
    """SHA-256 over the canonical content triple."""
    payload = json.dumps(
        {"model": model_id, "prompt": prompt_text, "schema": schema_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class JudgeCacheStats:
    hits: int = 0
    misses: int = 0


class JudgeCache:
    """File-backed JSON cache for judge responses."""

    def __init__(self, root: Path, *, enabled: bool = True) -> None:
        self._root = root
        self._enabled = enabled
        self.stats = JudgeCacheStats()

    @property
    def dir(self) -> Path:
        return self._root / "cache" / "judge"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        path = self.dir / f"{key}.json"
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
        path = self.dir / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
        try:
            tmp.write_text(
                json.dumps(
                    {"schema_version": SCHEMA_VERSION, **value},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise


__all__ = ["JudgeCache", "JudgeCacheStats", "cache_key"]

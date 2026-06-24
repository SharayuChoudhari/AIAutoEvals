"""Stable JSON output for `--format json`.

Every payload includes `schema_version: 1`. Bumps require migration notes in the
release.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ai_eval.config.defaults import SCHEMA_VERSION


def dump(payload: dict[str, Any]) -> None:
    """Write a single JSON document to stdout (data channel)."""
    body = {"schema_version": SCHEMA_VERSION, **payload}
    sys.stdout.write(json.dumps(body, indent=2, sort_keys=True, ensure_ascii=False))
    sys.stdout.write("\n")
    sys.stdout.flush()


__all__ = ["dump"]

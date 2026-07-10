"""Tests for the judge response cache (content addressing + --no-cache)."""

from __future__ import annotations

from pathlib import Path

from ai_eval.judge.cache import JudgeCache, cache_key


def test_put_then_get(tmp_path: Path) -> None:
    c = JudgeCache(tmp_path, enabled=True)
    k = cache_key("m", "p", schema_id="v1")
    assert c.get(k) is None
    assert c.stats.misses == 1
    c.put(k, {"score": 0.9, "rationale": "ok"})
    got = c.get(k)
    assert got is not None
    assert got["score"] == 0.9
    assert c.stats.hits == 1


def test_disabled_skips_both(tmp_path: Path) -> None:
    c = JudgeCache(tmp_path, enabled=False)
    k = cache_key("m", "p", schema_id="v1")
    assert c.get(k) is None
    c.put(k, {"score": 0.9})
    # nothing written
    assert not (c.dir / f"{k}.json").is_file()
    assert c.stats.hits == 0
    assert c.stats.misses == 0


def test_corrupt_entry_treated_as_miss(tmp_path: Path) -> None:
    c = JudgeCache(tmp_path, enabled=True)
    k = cache_key("m", "p", schema_id="v1")
    c.dir.mkdir(parents=True, exist_ok=True)
    (c.dir / f"{k}.json").write_text("not json", encoding="utf-8")
    assert c.get(k) is None
    assert c.stats.misses == 1
    assert c.stats.hits == 0


def test_dir_under_state_cache_judge(tmp_path: Path) -> None:
    c = JudgeCache(tmp_path)
    assert c.dir == tmp_path / "cache" / "judge"

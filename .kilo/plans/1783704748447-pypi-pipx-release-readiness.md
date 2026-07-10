# Plan: Prepare `ai-eval` for PyPI/pipx as `ai-evals`

Goal: make the package installable via `pipx install ai-evals` and `pip install ai-evals`, with a clean sdist/wheel, a tag-triggered OIDC release pipeline (TestPyPI first, then PyPI), and all naming consistent.

---

## Decisions (locked)

| Decision | Choice |
|---|---|
| PyPI distribution name | `ai-evals` (free; `ai-eval` is taken by HoneyHive) |
| Python import name | `ai_eval` (unchanged — rename is too risky, dist != import name is standard) |
| CLI command name | `ai-evals` (entry point + all user-facing strings) |
| On-disk state dir | `.ai-evals/` (no migration; breaking change accepted) |
| Hints file | `eval/ai-evals.hints.yaml` |
| User-global config | `~/.config/ai-evals/config.yaml` |
| Env var prefix | `AI_EVAL_*` (stays — matches the unchanged import name `ai_eval`) |
| sdist bloat fix | `[tool.hatch.build.targets.sdist]` exclude in pyproject.toml |
| LICENSE | Add MIT file |
| Version source | Single-source in `pyproject.toml`; `__init__.py` reads via `importlib.metadata` |
| CHANGELOG | Add `CHANGELOG.md` (Keep a Changelog format) |
| Release | OIDC Trusted Publishing via GitHub Actions, triggered by tag push |
| Dry run | TestPyPI publish first, verify, then real PyPI |

---

## Tasks

### 1. sdist cleanup (pyproject.toml)

Add an sdist exclude list so the tarball no longer ships `.kilo/` (worktrees, node_modules, plans), `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`.

In `pyproject.toml`, add:

```toml
[tool.hatch.build.targets.sdist]
exclude = [
  ".kilo",
  ".venv",
  "venv",
  ".mypy_cache",
  ".ruff_cache",
  ".pytest_cache",
  "dist",
  "tests",
]
```

Notes:
- `tests/` excluded from sdist is optional but conventional for a CLI tool; keep tests in the repo, just not in the published tarball. If you'd rather ship tests, drop `"tests"` from the list.
- The wheel (`[tool.hatch.build.targets.wheel]`) already only ships `ai_eval/` — leave it as-is.

Validation: `uv build && tar tzf dist/ai_evals-*.tar.gz | grep -c '\\.kilo'` must print `0`.

### 2. Rename distribution + CLI command to `ai-evals`

#### 2a. pyproject.toml
- `name = "ai-eval"` → `name = "ai-evals"`
- `[project.scripts]`: `ai-eval = "ai_eval.cli.app:main"` → `ai-evals = "ai_eval.cli.app:main"`
- `[tool.hatch.build.targets.wheel]`: `packages = ["ai_eval"]` — unchanged (import name stays).

#### 2b. CLI app name + version output
`ai_eval/cli/app.py`:
- `name="ai-eval"` → `name="ai-evals"`
- `help="ai-eval — ..."` → `help="ai-evals — ..."`
- `f"ai-eval {__version__} ({py})"` → `f"ai-evals {__version__} ({py})"`

`ai_eval/cli/stubs.py`:
- All `typer.echo` strings referencing `ai-eval` → `ai-evals` (the reserved-command hint + version output).

#### 2c. All user-facing CLI strings (typer.echo)
Scan every `ai_eval/cli/*.py` for `ai-eval` in strings shown to users (error messages, hints, `fix=`, `why=`) and replace with `ai-evals`. Files with matches:
- `cli/init.py` (next-command hints, hint-file error strings)
- `cli/analyze.py` (fix hint, hints file mention)
- `cli/bootstrap.py` (fix hint, tempdir prefix)
- `cli/run.py` (fix hint)
- `cli/diff.py` (error message)
- `cli/judge.py` (error message)
- `cli/doctor.py` (fix hints)
- `cli/report.py` (HTML title)
- `cli/render/tables.py` (docstrings — cosmetic)

Also update the stale extra name in error strings:
- `ai_eval/inference/slm/client.py:49` — `"pip install 'ai-eval[judge]'..."` → `"pip install ai-evals..."` (the `judge` extra is empty; drop the `[judge]` suffix).
- `ai_eval/judge/instructor_glue.py:44` — `` `ai-eval[judge]` `` → `ai-evals`.

#### 2d. State directory + paths rename
- `ai_eval/config/defaults.py`: `DEFAULT_STATE_DIR: str = ".ai-eval"` → `".ai-evals"`.
- `ai_eval/storage/paths.py`: `hints_yaml` property returns `self.eval_dir / "ai-eval.hints.yaml"` → `"ai-evals.hints.yaml"`. Update the docstring.
- `ai_eval/config/loader.py`: `user_config_path()` returns `base / "ai-eval" / "config.yaml"` → `base / "ai-evals" / "config.yaml"`. Update docstring line 7.
- `ai_eval/scaffold/gitignore_patch.py`: the patch string `"\n# ai-eval: ...\n.ai-eval/\n"` → `"\n# ai-evals: ...\n.ai-evals/\n"`, and the idempotency check `".ai-eval/"` → `".ai-evals/"`.
- `ai_eval/inference/ast_scan.py:52`: `".ai-eval"` (in the ignore list) → `".ai-evals"`.

#### 2e. Docstrings referencing old paths (cosmetic but keep accurate)
Update docstring mentions of `.ai-eval/` and `eval/ai-eval.hints.yaml` in:
- `ai_eval/storage/runs.py`, `ai_eval/insights/history.py`, `ai_eval/runner/record.py`
- `ai_eval/config/schema.py` (lines 107, 137)
- `ai_eval/inference/hints.py`, `ai_eval/inference/slm/builder.py`, `ai_eval/inference/slm/cache.py`, `ai_eval/judge/cache.py`
- `ai_eval/storage/paths.py` (ensure_state docstring)

#### 2f. Tests
Update all test assertions and fixtures using `ai-eval` / `.ai-eval`:
- `tests/test_cli_misc.py` — `assert "ai-eval" in result.output` → `"ai-evals"`, and `.ai-eval` dir assertions.
- `tests/test_cli_init.py` — `.ai-eval` state dir assertion + gitignore assertion.
- `tests/test_cli_run.py`, `tests/test_cli_diff_report_history.py`, `tests/test_slm_builder.py` — `.ai-eval` → `.ai-evals`.
- `tests/test_appointment_workflow_fixture.py` — `ai-eval.hints.yaml` → `ai-evals.hints.yaml`.
- `tests/test_cli_bootstrap.py`, `tests/test_cli_judge.py`, `tests/test_cli_diff_report_history.py` — docstring/comment mentions (cosmetic).

### 3. Version single-sourcing

`ai_eval/__init__.py` — replace the hardcoded `__version__ = "0.1.0"` with a runtime read from installed metadata:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ai-evals")
except PackageNotFoundError:  # editable/source checkout without install
    __version__ = "0.0.0+unknown"
```

Note: `version("ai-evals")` uses the **distribution name**, not the import name.

The single source of truth for the real version becomes `pyproject.toml`'s `version = "..."`.

### 4. Add LICENSE file

Create `LICENSE` with the standard MIT text. Copyright holder: match `git log` author (Sharayu Choudhari / SharayuChoudhari) + year 2026.

Optionally update `pyproject.toml` to the SPDX form so PyPI shows the file:
```toml
license = "MIT"
license-files = ["LICENSE"]
```
(Keep `license = { text = "MIT" }` only if you must support old tooling; SPDX is preferred on modern PyPI.)

### 5. CHANGELOG.md

Create `CHANGELOG.md` in Keep a Changelog format:

```markdown
# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog; adheres to Semantic Versioning.

## [Unreleased]

## [0.1.0] - 2026-07-10
### Added
- Phase 1: static analysis & scaffolding (`ai-evals init`, `analyze`, `doctor`, `config`).
- Phase 2: model-agnostic judge gateway via LiteLLM + Instructor (`ai-evals judge`).
- Phase 3: automated golden-set bootstrapper (`ai-evals bootstrap`).
- Phase 4: run execution & insights (`ai-evals run`, `diff`, `report`, `history`).
```

### 6. README updates

`README.md` — update all install/usage examples:
- `pipx install ai-eval` → `pipx install ai-evals`
- `pip install -e ".[dev]"` — unchanged (dev install still works)
- All `ai-eval <command>` → `ai-evals <command>` (init, bootstrap, run, diff, doctor, judge, config, history, report, analyze)
- `.ai-eval/` path mentions → `.ai-evals/`
- `eval/ai-eval.hints.yaml` mention → `eval/ai-evals.hints.yaml`
- Status line: keep `v0.1.0 — Phases 1–4`.

### 7. docs/how-it-works.html updates

Rename all `ai-eval` → `ai-evals` and `.ai-eval/` → `.ai-evals/` in the rendered HTML (paths, diagrams, config-order section). The `AI_EVAL_*` env var references stay unchanged (import name is `ai_eval`).

### 8. .gitignore

`.gitignore` line 24-25:
```
# ai-eval local state (eval/ is checked in; .ai-eval/ is local cache)
.ai-eval/
```
→
```
# ai-evals local state (eval/ is checked in; .ai-evals/ is local cache)
.ai-evals/
```

### 9. uv.lock

After renaming the project in `pyproject.toml`, run `uv lock` to regenerate `uv.lock` with the new package name (`name = "ai-evals"` on line 11).

### 10. GitHub Actions release workflow (OIDC Trusted Publishing)

Create `.github/workflows/release.yml`:

Triggers:
- On push of tags matching `v*` → publish to **PyPI**.
- A separate `workflow_dispatch` with an input → publish to **TestPyPI**.

Job (Ubuntu, Python 3.12):
1. Checkout.
2. Install `uv`.
3. `uv build` (produces sdist + wheel).
4. `uv run twine check dist/*` (install `twine` as a transient dep, or use `uvx twine check dist/*`).
5. Publish:
   - TestPyPI path: `uv publish --publish-url https://test.pypi.org/legacy/ --trusted-publishing always`
   - PyPI path: `uv publish --trusted-publishing always`
   (Both use OIDC — no API token secret required.)

PyPI Trusted Publisher setup (one-time, manual, before first release):
1. Create the `ai-evals` project on PyPI (`https://pypi.org/manage/account/publishing/`).
2. Add a Trusted Publisher: repo `SharayuChoudhari/AIAutoEvals`, workflow `.github/workflows/release.yml`, environment `pypi`.
3. Repeat for TestPyPI (`https://test.pypi.org/manage/account/publishing/`) with environment `testpypi`.

Add GitHub environments `pypi` and `testpypi` in repo settings (Settings → Environments).

### 11. First-release validation steps (manual, after implementation)

1. `uv lock` then `uv build` — confirm sdist has no `.kilo/`, wheel has only `ai_eval/`.
2. `pip install -e ".[dev]"` then `pytest -q --timeout=30` — all green after rename.
3. `ruff check .` clean; `mypy ai_eval` (baseline only).
4. `ai-evals --version` prints `ai-evals 0.1.0 (...)`.
5. In a scratch repo: `ai-evals init`, confirm `.ai-evals/` is created (not `.ai-eval/`), gitignore patched, `eval/ai-evals.hints.yaml` written.
6. Tag-triggered TestPyPI publish → `pipx install --index-url https://test.pypi.org/simple/ ai-evals` → `ai-evals --help` works.
7. Real PyPI publish: push `v0.1.0` tag → `pipx install ai-evals` → verify.

---

## Risk / failure modes

- **Rename drift:** ~94 string references across the codebase. Mitigation: a final `rg -n 'ai-eval' ai_eval/ tests/ README.md docs/` (excluding `ai_eval/` import paths) must return only `AI_EVAL_` env-var hits + intentional `ai_eval` import names.
- **OIDC not configured before first tag release** → the workflow will fail with a 403. Mitigation: set up the Trusted Publisher on PyPI + TestPyPI **before** pushing any tag.
- **Import name mismatch:** `importlib.metadata.version("ai-evals")` uses the distribution name, not `ai_eval`. Getting this wrong silently returns `0.0.0+unknown`. The plan specifies the distribution name explicitly.
- **Existing users' `.ai-eval/` data** is orphaned by the state-dir rename (accepted per decision — no migration).
- **`uv.lock` stale** after rename → `uv lock` regenerates it.

---

## Out of scope

- Renaming the Python import package `ai_eval` → `ai_evals` (too invasive; dist != import is standard).
- Migrating existing `.ai-eval/` data to `.ai-evals/`.
- Renaming the `AI_EVAL_*` env var prefix.
- Code signing / GPG verification of distributions.
- PyPI project URL / long-description rendering tweaks beyond README (already used as `readme`).

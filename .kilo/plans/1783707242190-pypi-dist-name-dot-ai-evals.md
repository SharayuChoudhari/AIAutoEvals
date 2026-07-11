# Plan: Rename PyPI distribution `ai-evals` → `dot-ai-evals`

Goal: change only the **PyPI distribution name** from `ai-evals` to `dot-ai-evals` to avoid collision/similarity with the taken `ai-eval` (HoneyHive) and `evals` (OpenAI) packages. The CLI command, import name, on-disk paths, and env vars stay unchanged.

---

## Decisions (locked)

| Decision | Choice |
|---|---|
| PyPI distribution name | `dot-ai-evals` (free — 404 confirmed on https://pypi.org/simple/dot-ai-evals/) |
| CLI command word | `ai-evals` (unchanged — `ai-evals init`, `ai-evals run`, etc.) |
| Python import name | `ai_eval` (unchanged) |
| On-disk state dir | `.ai-evals/` (unchanged) |
| Hints file | `eval/ai-evals.hints.yaml` (unchanged) |
| User-global config | `~/.config/ai-evals/config.yaml` (unchanged) |
| Env var prefix | `AI_EVAL_*` (unchanged — matches import name `ai_eval`) |
| Version source | `importlib.metadata.version("dot-ai-evals")` in `__init__.py` |

---

## Tasks

### 1. pyproject.toml — distribution name

`pyproject.toml:6`:
- `name = "ai-evals"` → `name = "dot-ai-evals"`

Leave everything else unchanged:
- `[project.scripts]` entry point stays `ai-evals = "ai_eval.cli.app:main"` (the CLI command word).
- `[tool.hatch.build.targets.wheel]` `packages = ["ai_eval"]` stays (import name).
- `[tool.hatch.build.targets.sdist]` exclude list stays.
- `authors = [{ name = "ai-evals contributors" }]` — optional cosmetic; leave as-is (it's the contributor label, not the dist name).

### 2. __init__.py — version metadata lookup

`ai_eval/__init__.py:9`:
- `__version__ = version("ai-evals")` → `__version__ = version("dot-ai-evals")`

The `version()` call uses the **distribution name**, not the import name. Getting this wrong silently falls through to the `PackageNotFoundError` fallback (`0.0.0+unknown`).

### 3. README.md — install command

`README.md:11`:
- `pipx install ai-evals` → `pipx install dot-ai-evals`

The Quick start usage examples (lines 22-28: `ai-evals init`, `ai-evals bootstrap -- pytest -q`, etc.) stay **as-is** — those are the CLI command, unchanged. Only the `pipx install <dist-name>` line changes.

### 4. uv.lock — regenerate

Run `uv lock`. The lockfile currently has `name = "ai-evals"` on line 11; `uv lock` will regenerate it as `name = "dot-ai-evals"`.

---

## Validation (run in this order)

1. `uv lock` — confirm `rg -n 'name = "dot-ai-evals"' uv.lock` returns line 11.
2. `rm -rf dist && uv build` — confirm output files are named `dot_ai_evals-0.1.0.tar.gz` and `dot_ai_evals-0.1.0-py3-none-any.whl`.
3. `uvx twine check dist/*` — both PASSED.
4. `uv pip install -e ".[dev]"` — confirms it installs as `dot-ai-evals`.
5. `uv run python -c "import ai_eval; print(ai_eval.__version__)"` — prints `0.1.0` (proves `version("dot-ai-evals")` resolves; if it prints `0.0.0+unknown`, the dist name in `__init__.py` is wrong).
6. `uv run ai-evals --version` — prints `ai-evals 0.1.0 (python X.Y)` (CLI command unchanged).
7. `uv run pytest -q --timeout=30` — all green.
8. `uv run ruff check .` — clean.
9. sdist cleanliness unchanged: `tar tzf dist/dot_ai_evals-0.1.0.tar.gz | grep -c '\.kilo'` prints `0`.

---

## Out of scope

- The CLI command word `ai-evals` (stays).
- The import name `ai_eval` (stays).
- The state dir `.ai-evals/`, hints file `ai-evals.hints.yaml`, config path `~/.config/ai-evals/` (all stay).
- The `AI_EVAL_*` env var prefix (stays — matches import name).
- Any change to `.github/workflows/release.yml` (it's name-agnostic — uses `uv publish`).
- Any change to `docs/how-it-works.html` (the "ai-evals" references there are the CLI command word, not the dist name).

---

## Risks / failure modes

- **Wrong version() dist name**: if `__init__.py` keeps `version("ai-evals")` after renaming pyproject to `dot-ai-evals`, `__version__` silently becomes `0.0.0+unknown`. Mitigation: validation step 5 catches this explicitly.
- **Stale uv.lock**: if `uv lock` isn't run, the lockfile still says `ai-evals` and `uv build`/installs may use the old name. Mitigation: run `uv lock` first (task 4, validation step 1).
- **PyPI Trusted Publisher**: the release workflow's Trusted Publisher must be set up for the `dot-ai-evals` project (not `ai-evals`) on PyPI/TestPyPI before the first tag. This is a manual step outside the code.

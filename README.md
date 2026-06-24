# ai-eval

Zero-config, repo-aware, model-agnostic AI evaluation.

> Status: **v0.1.0 — Phase 1** (static analysis + scaffolding).
> See `.kilo/plans/New_idea.md` for the vision and `.kilo/plans/ai-eval-cli-and-system-design.md` for the full system design.

## Install

```bash
pipx install ai-eval         # recommended
# or, for development:
pip install -e ".[dev,judge]"
```

## Quick start

```bash
ai-eval init                       # scan repo, scaffold eval/
ai-eval bootstrap -- pytest -q     # capture real traces (Phase 3)
ai-eval run                        # score the suite (Phase 4)
ai-eval diff --against last        # explain regressions (Phase 4)
```

Run `ai-eval --help` to see all commands.

## Phase 1 capabilities (this release)

- `ai-eval init` — scans a Python repo, detects OpenAI / LangChain / ChromaDB usage,
  writes `eval/rubrics.yaml`, `eval/tests.py`, `eval/golden_set.json`, and `.ai-eval/`.
- `ai-eval analyze` — re-runs detection, proposes a merged `rubrics.yaml`.
- `ai-eval doctor` — environment + config sanity check.
- `ai-eval config --print` — show resolved configuration with source per key.
- All other commands are stubs that exit `2` with a clear "Phase X" message.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | general / runtime error |
| 2 | usage error (bad flags, `--no-input` hit a prompt, command not yet implemented) |
| 3 | evaluation failed thresholds (Phase 4 `run --fail-on-regression`) |

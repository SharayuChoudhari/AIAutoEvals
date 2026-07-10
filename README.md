# ai-evals

Zero-config, repo-aware, model-agnostic AI evaluation.

> Status: **v0.1.0 — Phases 1–4** (static analysis, scaffolding, judge gateway, golden-set bootstrapper, run execution & insights).
> See `.kilo/plans/New_idea.md` for the vision and `.kilo/plans/ai-evals-cli-and-system-design.md` for the full system design.

## Install

```bash
pipx install ai-evals         # recommended
# or, for development:
pip install -e ".[dev]"
```

> `litellm`, `instructor`, and `anyio` are core dependencies — no extra `judge`
> extra is needed to use the judge gateway.

## Quick start

```bash
ai-evals init                       # scan repo, scaffold eval/
ai-evals bootstrap -- pytest -q     # capture real runtime traces
ai-evals run                        # score the suite against the golden set
ai-evals diff --against last        # show metric regressions + narratives
```

Run `ai-evals --help` to see all commands.

## Ollama setup (default judge)

The default judge is the local model `ollama/qwen2.5-coder:7b`. Follow these steps
to set it up:

1. **Install Ollama** from <https://ollama.com>.
2. **Start the local server** (default port `11434`):

   ```bash
   ollama serve
   ```

3. **Pull the default judge model**:

   ```bash
   ollama pull qwen2.5-coder:7b
   ```

4. **Run as normal** — `ai-evals init`, `ai-evals analyze`, etc. use the local judge
   automatically.

If Ollama is unavailable, either set a cloud API key (e.g. `OPENAI_API_KEY` for the
regression judge `openai/gpt-4o-mini`) or fall back to the rules-based rubric
engine (detection/scaffolding only — no judge calls):

```bash
ai-evals init --rubric-engine rules
```

Set a different judge with `--judge-default <model>` (e.g.
`ollama/llama3.2:3b` or `openai/gpt-4o-mini`). Override the judge at run time
with `ai-evals run --judge <model>`.

## Capabilities

### Phase 1 — detection & scaffolding

- `ai-evals init` — scans a Python repo, detects OpenAI / LangChain / ChromaDB usage,
  writes `eval/rubrics.yaml`, `eval/tests.py`, `eval/golden_set.json`, and `.ai-evals/`.
- `ai-evals analyze` — re-runs detection, proposes a merged `rubrics.yaml`.
- `ai-evals doctor` — environment + config sanity check (incl. a judge-gateway probe;
  the probe is optional and never forces exit 1 on its own).
- `ai-evals config --print` — show resolved configuration with source per key.

### Phase 2 — model-agnostic judge gateway

- `ai-evals judge --list` — show configured judges (`default` / `regression_check` /
  `fallback[*]`) with a best-effort reachability ping.
- `ai-evals judge --ping <model>` — one-shot reachability probe of a single model id.
- `ai-evals judge --prompt <text> --model <m>` — one-shot judge completion.
- Async transport via `litellm.acompletion` + `instructor`; content-addressed
  cache under `.ai-evals/cache/judge/` (skipped with `--no-cache`). Two prompt
  tiers — CoT (complex/frontier models) and binary checklist (local/small models)
  — selected from `config/defaults.py:COMPLEX_MODEL_HINTS` (overridable via
  `judge.tiering.complex_models` in `rubrics.yaml`).
- Built-in judge metrics: `argument_accuracy`, `hallucination_rate`,
  `context_precision` (partial: deterministic signal + judge tiebreak),
  `faithfulness`. Non-judge `latency_p50` / `latency_p95` are computed by the
  runner. Custom metrics load from the `ai_eval.metrics` entry-point group.

### Phase 3 — automated golden-set bootstrapper

- `ai-evals bootstrap -- <command>` — captures real runtime traces (one golden
  example per top-level task invocation; nested LLM/retrieve calls fold into
  `trace.calls`) by monkey-patching OpenAI / LangChain / ChromaDB / Pinecone in
  a child subprocess via a generated `sitecustomize.py`. Each framework patch is
  independently skippable when the framework isn't installed.
- Merges into `eval/golden_set.json` with `--append`/`--replace`, `--max`, and
  `--task` filters. Captures are flushed even if the child crashes. Secrets in
  captured args/results are redacted.
- `--no-instrument` is reserved for a future OTEL path and currently exits 2.

### Phase 4 — run execution & insights

- `ai-evals run` — re-imports task modules, calls the entry symbol per golden
  example, scores each metric via the judge gateway, aggregates, and saves a
  run record (`.ai-evals/runs/<id>/run.json`) + history. `--fail-on-regression`
  exits `3` on threshold breaches. Fresh repos (no baseline) show `—` for Δ and
  exit 0. An unregistered metric hard-fails with exit 1 and a fix hint, even
  though `init`/`analyze` only warn.
- `ai-evals diff --against last` — per-metric Δ table; `--explain` emits a
  regression-judge narrative per regressing (metric, example) (one judge call
  each). Falls back to a deterministic text delta when the regression judge is
  unavailable, so it never crashes.
- `ai-evals report [last]` — render a past run as human / json / markdown
  (`--out <file>`); `--web` writes a minimal HTML file and prints a `file://` URL.
- `ai-evals history [--list|--show <id>|--prune --keep N|--export <file>]` —
  manage `.ai-evals/history.json` (slim records; full runs under `runs/<id>/`).

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | general / runtime error |
| 2 | usage error (bad flags, missing `--` command, `--no-input` hit a prompt) |
| 3 | evaluation failed thresholds (`run --fail-on-regression`) |

## Development

```bash
pip install -e ".[dev]"          # pytest, ruff, mypy, pytest-timeout, syrupy
ruff check .                     # lint (must be clean)
mypy ai_eval                      # typecheck (pre-existing baseline errors only)
pytest -q --timeout=30            # full suite (fake judge transport — no network)
```

The test suite uses a fake async judge transport and stubbed frameworks, so it
runs fully offline in ~3s. The end-to-end test
(`tests/test_e2e.py`) exercises `init → bootstrap → run → diff` against a
scratch repo and asserts the golden set is populated, a run record is written,
and a regression delta is detected.

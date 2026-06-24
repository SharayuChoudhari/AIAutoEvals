# ai-eval — CLI UX + End-to-End System Design

Companion plan to `.kilo/plans/New_idea.md`. That file fixes the vision (zero-config, repo-aware, model-agnostic AI evaluation). This plan locks down:

1. The **CLI surface** (commands, flags, output, prompts, help, errors) using the `cli-ux-designer` skill.
2. The **end-to-end system** (modules, data contracts, storage, control flow, extension points, runtime, packaging).

Scope: design only. No code is written here. Implementation follows the 4 phases in `New_idea.md`.

---

## 0. Design Principles (from `cli-ux-designer`)

These are non-negotiable rules every command in this tool must honor.

- **Reasonable defaults, easy overrides.** Zero-config first run must work. Every default has a flag override.
- **Max 2 levels of nesting.** `ai-eval <command> [flags]`. No `ai-eval group sub action`.
- **Terminal-first.** No required web step. URLs printed at the end of operations are fine; `--web` opens browser when relevant.
- **TTY-aware output.** Colors + tables when stdout is a TTY; plain tab-delimited or JSON when piped.
- **Every prompt has a flag.** Interactive by default for exploration; non-interactive with `--yes` / `--no-input` / explicit flags for CI.
- **Errors:** to stderr, exit codes `0` ok / `1` general / `2` usage / `3` evaluation-failed-threshold. Always include "what / why / how to fix".
- **Data to stdout, status to stderr.** Pipe-safe.
- **Sentence case** everywhere. No title case in help/output.
- **Color only enhances meaning.** State is also conveyed by icon + label.
- **Help on every level.** `--help` works at root and on every subcommand.
- **Flag naming is consistent.** Long form always; common shorthand documented. No mixing `--no-color` with `--disable-colors`.

---

## 1. CLI Surface

### 1.1 Top-level help (the contract)

```
ai-eval — zero-config AI evaluation for your repo.

Usage:
  ai-eval <command> [flags]

Core commands:
  init        Scan the repo and scaffold eval/ (rubrics + tests).
  analyze     Re-run repo inference and refresh rubrics.yaml.
  bootstrap   Capture real runtime traces into a golden set.
  run         Execute evaluations against the current code.
  diff        Show regression diff vs. a baseline run.
  report      Render a past run as a human or JSON report.
  judge       Inspect / test the configured judge models.
  doctor      Verify environment, providers, and config.

Additional commands:
  config      Print or edit resolved configuration.
  history     List, prune, or export past runs.
  version     Print the version and exit.

Flags:
  -C, --cwd <dir>           Run as if invoked from <dir>.
      --config <file>       Path to rubrics.yaml (default: eval/rubrics.yaml).
      --format <fmt>        human | json | tsv  (auto: human on TTY, tsv on pipe)
      --no-color            Disable ANSI colors.
      --quiet               Suppress progress; errors still print.
      --verbose             Verbose logs (repeatable: -vv for debug).
  -y, --yes                 Assume "yes" on all confirmations.
      --no-input            Fail instead of prompting.
  -h, --help                Show help for a command.
      --version             Show version.

Learn more:
  ai-eval <command> --help
  Docs: https://ai-eval.dev/docs   (printed only; never auto-opened)

Examples:
  $ ai-eval init
  $ ai-eval bootstrap -- pytest -q
  $ ai-eval run --task customer_support_agent
  $ ai-eval diff --against last --fail-on-regression
```

### 1.2 Command spec

Each command below lists: purpose, args, flags, prompts, exit codes, stdout/stderr contract, example.

#### `ai-eval init`

- **Purpose:** First-run scaffold. Runs inference engine, writes `eval/rubrics.yaml`, `eval/golden_set.json` (empty stub), `eval/tests.py`, and `.ai-eval/` state dir.
- **Args:** none.
- **Flags:**
  - `--force` overwrite existing eval/ files.
  - `--dry-run` print what would be written; no writes.
  - `--judge-default <model>` set default judge (e.g. `ollama/qwen2.5-coder:7b`).
  - `--judge-regression <model>` set regression judge (e.g. `openai/gpt-4o-mini`).
  - `--frameworks <list>` comma list to bias detection (`langchain,openai,llamaindex,...`).
  - `--include <glob>` / `--exclude <glob>` (repeatable).
- **Prompts (interactive only):**
  - Confirm detected frameworks: `Detected: openai, chromadb. Use these? [Y/n]`
  - Confirm default judge model when none provided.
- **Exit codes:** `0` ok, `1` write failed, `2` invalid flags.
- **Stdout:** human summary table (files written, tasks detected). With `--format json`: a `{ "written": [...], "tasks": [...] }` object.
- **Stderr:** progress spinner "Scanning repo… 312 files".
- **Example output:**
  ```
  ✓ Scanned 312 files in 0.8s
  ✓ Detected 2 AI tasks
    • customer_support_agent  (tool_calling)   src/agents/support.py
    • knowledge_base_retriever (rag)           src/rag/kb.py
  ✓ Wrote eval/rubrics.yaml
  ✓ Wrote eval/golden_set.json  (empty — run `ai-eval bootstrap`)
  ✓ Wrote eval/tests.py
  Next: ai-eval bootstrap -- pytest
  ```

#### `ai-eval analyze`

- **Purpose:** Re-run AST inference, merge into existing `rubrics.yaml` (preserves user edits where possible).
- **Args:** none.
- **Flags:** `--write` (default `--dry-run`), `--merge-strategy keep|overwrite|prompt`, `--diff` print yaml diff.
- **Exit codes:** `0` no change, `0` changes proposed (with `--dry-run`), `1` merge conflict requiring `--merge-strategy`.

#### `ai-eval bootstrap`

- **Purpose:** Capture real runtime traces into `eval/golden_set.json`. Two modes.
- **Args:** `-- <command> [args...]` (everything after `--` is the user's runtime, e.g. `pytest -q`, `python app.py`).
- **Flags:**
  - `--task <name>` capture only one task.
  - `--max <N>` cap captures (default 50).
  - `--append | --replace` (default `--append`).
  - `--no-instrument` use OTEL only (requires user setup).
  - `--out <path>` override golden set location.
- **Prompts:** none (it just runs).
- **Exit codes:** `0` captures saved, `1` runtime crashed, `2` usage error.
- **Stdout:** count of captures per task. Stderr shows live counter `[12/50] captured customer_support_agent`.

#### `ai-eval run`

- **Purpose:** Execute the eval suite.
- **Flags:**
  - `--task <name>` (repeatable) limit tasks.
  - `--judge <model>` override `judge.default`.
  - `--baseline <run-id|last|none>` baseline source (default `last`).
  - `--fail-on-regression` exit `3` if any metric regressed past tolerance.
  - `--tolerance <0..1>` regression sensitivity (default 0.02).
  - `--parallel <N>` concurrent judges (default `min(4, cpus)`).
  - `--no-cache` skip judge response cache.
  - `--save / --no-save` write run to history (default save).
  - `--tag <label>` (repeatable) attach tags (e.g. `--tag pr-482`).
- **Prompts:** none. Long-running: live progress in stderr.
- **Exit codes:** `0` all pass, `3` regression / threshold breach, `1` runner error, `2` usage.
- **Stdout (human):**
  ```
  Run 2026-06-25T04-11Z  (id: r_8f3a)
  baseline: r_8e21  Δ vs baseline shown in column 4
  
  task                          metric                 score   Δ      status
  customer_support_agent        argument_accuracy      0.93   +0.01   ✓
  customer_support_agent        hallucination_rate     0.04   -0.01   ✓
  knowledge_base_retriever      context_precision      0.81   -0.07   ✗  below 0.85
  knowledge_base_retriever      faithfulness           0.96   +0.00   ✓
  
  Summary: 3 passed, 1 failed   • latency p50 1.2s (+120ms)
  Failures:
    knowledge_base_retriever.context_precision dropped 0.07 below threshold.
    → ai-eval diff --task knowledge_base_retriever --against last
  ```
- **Stdout (--format json):** stable schema (see §3.4).

#### `ai-eval diff`

- **Purpose:** Show why a run regressed. Calls the regression judge on paired traces.
- **Args:** none.
- **Flags:**
  - `--against <run-id|last|main>` (default `last`).
  - `--task <name>` focus.
  - `--metric <name>` focus.
  - `--limit <N>` traces to show (default 10).
  - `--explain` ask judge for "why it failed" narrative (default on if regression judge configured).
- **Exit codes:** `0` always (diff is informational unless `--fail-on-regression`).
- **Output:** trace-pair view, indented body, color-coded.

#### `ai-eval report`

- **Purpose:** Render a stored run.
- **Args:** `<run-id|last>`.
- **Flags:** `--format human|json|md`, `--out <file>`, `--web` (write html and print URL via local file://).

#### `ai-eval judge`

- **Purpose:** Sanity-check configured judges. Subverbs are not subcommands (anti-pattern); they're flags.
- **Flags:** `--list` (show configured models + reachability), `--ping <model>`, `--prompt <text> --model <m>` for a one-shot.

#### `ai-eval doctor`

- **Purpose:** Verify the environment. Always safe, read-only.
- **Output:** checklist with ✓/✗/!  for: Python version, LiteLLM install, provider creds, local model reachability, write perms on `.ai-eval/`, rubrics.yaml validity.
- **Exit:** `0` all pass, `1` any fail (so CI can gate).

#### `ai-eval config`

- **Purpose:** Inspect resolved config.
- **Flags:** `--print` (default), `--path` (print file path only), `--set <key=value>` (writes back to rubrics.yaml with confirm).

#### `ai-eval history`

- **Purpose:** Manage `.ai-eval/history.json`.
- **Flags:** `--list` (default), `--show <run-id>`, `--prune --keep <N>`, `--export <file>`.

#### `ai-eval version`

- Prints `ai-eval X.Y.Z (python A.B, litellm C.D)`.

### 1.3 Flag conventions

| Convention | Rule |
|---|---|
| Long form | `--kebab-case`, always available |
| Short form | One char, only for top-5 hottest flags (`-y`, `-C`, `-h`, `-v`, `-q`) |
| Negation | `--no-<flag>` (never `--disable-<flag>`) |
| Repeatable | Documented in help with `(repeatable)` |
| Mutually exclusive | `{--append \| --replace}` shown in usage |
| Value syntax | `<required>` angle brackets, `[optional]` square brackets |

### 1.4 Output rules

- **Human (TTY):** Rich tables, colors, icons, indented sections.
- **Pipe (non-TTY):** `--format tsv` auto-applied. No colors. Tabs as delimiters. No spinners — replaced by silent progress.
- **`--format json`:** Stable, versioned schema (`"schema_version": 1`). One JSON document per command (NDJSON for `run` stream when `--stream`).
- **Status to stderr:** spinners, progress counters, "Wrote …".
- **Data to stdout:** the report itself.

### 1.5 Color + icon palette

| State | Color | Icon |
|---|---|---|
| Pass / improvement | green | `✓` |
| Fail / regression | red | `✗` |
| Warning / threshold near | yellow | `!` |
| Info / metadata | gray | `-` |
| Identifier (run-id, task, branch) | cyan | — |
| Link / URL | blue, underline | — |

Rule: never communicate state with color alone — icon + label always present.

### 1.6 Prompts

| Prompt type | When | Always also a flag |
|---|---|---|
| `[Y/n]` yes/no | confirm overwrite, confirm judge model | `--yes`, `--force` |
| Radio | select judge from candidates on first init | `--judge-default <model>` |
| Multi-select | select tasks to keep after detection | `--task <name>` (repeatable) |
| Free text | none (avoid) | n/a |

`--no-input` fails fast with exit `2` if any prompt is reached.

### 1.7 Error model

Every error printed to stderr follows:

```
error: <one-line summary>
  what: <what went wrong>
  why:  <root cause if known>
  fix:  <suggested next command or doc link>
```

Example:
```
error: judge model unreachable
  what: could not connect to ollama at http://localhost:11434
  why:  connection refused
  fix:  start ollama (`ollama serve`) or set judge.default in eval/rubrics.yaml
```

### 1.8 Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | general / runtime error |
| 2 | usage error (bad flags, missing args, `--no-input` hit a prompt) |
| 3 | evaluation failed thresholds (`run --fail-on-regression` and below-threshold metrics) |

---

## 2. End-to-End System Design

### 2.1 Module map (Python package layout)

```
ai_eval/
├── __init__.py
├── __main__.py                # `python -m ai_eval`
├── cli/                       # Typer app + per-command modules
│   ├── app.py                 # root Typer, global flags, TTY detection
│   ├── init.py
│   ├── analyze.py
│   ├── bootstrap.py
│   ├── run.py
│   ├── diff.py
│   ├── report.py
│   ├── judge.py
│   ├── doctor.py
│   ├── config_cmd.py
│   ├── history.py
│   └── render/                # Rich renderers; one per command
│       ├── tables.py
│       ├── json_out.py
│       └── theme.py
├── inference/                 # Phase 1 — repo awareness
│   ├── ast_scan.py            # walks repo, ast.parse per file
│   ├── detectors/             # one detector per framework
│   │   ├── base.py            # Detector ABC
│   │   ├── openai_chat.py
│   │   ├── openai_tools.py
│   │   ├── langchain.py
│   │   ├── llamaindex.py
│   │   ├── chromadb.py
│   │   ├── pinecone.py
│   │   └── pydantic_ai.py
│   ├── signatures.py          # import + call-pattern matchers
│   └── synthesize.py          # produces TaskSpec list → rubrics.yaml
├── scaffold/                  # writes eval/* files
│   ├── rubrics_writer.py
│   ├── tests_writer.py        # generates eval/tests.py stub
│   └── templates/
├── judge/                     # Phase 2 — model-agnostic
│   ├── gateway.py             # LiteLLM facade
│   ├── prompts/
│   │   ├── cot_complex.py     # frontier-tier CoT
│   │   └── checklist_basic.py # local-tier binary
│   ├── schemas.py             # Pydantic models for judge I/O
│   ├── instructor_glue.py     # forces JSON via Instructor
│   ├── cache.py               # content-addressed judge cache
│   └── tiering.py             # picks prompt tier from model id
├── bootstrap/                 # Phase 3
│   ├── tracer.py              # sys.settrace / OTEL hook
│   ├── wrappers.py            # framework-specific monkey-patches
│   └── golden_writer.py
├── runner/                    # Phase 4
│   ├── engine.py              # task loop, parallelism
│   ├── metrics/               # one module per metric
│   │   ├── argument_accuracy.py
│   │   ├── hallucination_rate.py
│   │   ├── context_precision.py
│   │   └── faithfulness.py
│   └── thresholds.py
├── insights/
│   ├── history.py             # read/write .ai-eval/history.json
│   ├── diff.py                # Δ computation
│   └── root_cause.py          # regression-judge driver
├── config/
│   ├── loader.py              # rubrics.yaml + env + flags merge
│   ├── schema.py              # Pydantic schema for rubrics.yaml
│   └── defaults.py
├── storage/
│   ├── paths.py               # .ai-eval/ resolution
│   ├── runs.py                # run record CRUD
│   └── locks.py               # advisory file lock for concurrent runs
└── telemetry/
    ├── logger.py              # structured logging to stderr
    └── progress.py            # rich progress; silent when piped
```

### 2.2 Data flow (sequence)

```
ai-eval init
  └─► inference.ast_scan.walk()
        └─► detectors.* run on each AST
              └─► synthesize.build_task_specs()
                    └─► scaffold.rubrics_writer.write(eval/rubrics.yaml)
                    └─► scaffold.tests_writer.write(eval/tests.py)
                    └─► storage.paths.ensure(.ai-eval/)

ai-eval bootstrap -- pytest
  └─► bootstrap.wrappers.install()        # monkey-patches openai/langchain/...
  └─► subprocess.run(["pytest"], env=instrumented)
        └─► wrappers capture (input, output, tool_calls) per task
              └─► bootstrap.golden_writer.append(eval/golden_set.json)

ai-eval run
  └─► config.loader.load()
  └─► storage.runs.new_run_id()
  └─► runner.engine.execute(tasks, golden_set)
        └─► for each task, for each example:
              ├─► invoke the user's task (re-import file_path)
              ├─► judge.gateway.score(prompt_tier(model), example, output)
              │     └─► judge.cache.lookup → litellm.completion → instructor.parse
              └─► runner.metrics.<metric>.compute(...)
        └─► insights.diff.compute_delta(current, baseline)
        └─► storage.runs.save(run_record)
  └─► cli.render.tables.render_run(run_record)

ai-eval diff --against last
  └─► storage.runs.load(current, baseline)
  └─► insights.root_cause.explain(pair) via regression_judge
  └─► cli.render.tables.render_diff(...)
```

### 2.3 File layout the tool creates in the user's repo

```
<repo-root>/
├── eval/
│   ├── rubrics.yaml          # editable; source of truth for tasks + thresholds
│   ├── golden_set.json       # captured I/O examples per task
│   └── tests.py              # pytest-style entry that calls ai-eval run
└── .ai-eval/                 # tool state — gitignore by default
    ├── history.json          # all runs (rolling, prunable)
    ├── runs/
    │   └── r_8f3a/
    │       ├── record.json   # full run record (see §3.4)
    │       ├── traces.ndjson # per-example traces
    │       └── judge.log     # raw judge calls for audit
    ├── cache/
    │   └── judge/<sha256>.json  # judge response cache
    └── config.lock           # advisory lock for concurrent runs
```

`init` writes a `.gitignore` line `.ai-eval/` and a comment `# eval/ is checked in; .ai-eval/ is local cache`.

### 2.4 Core data contracts (Pydantic, versioned)

All on-disk JSON/YAML embeds `"schema_version": 1`. Bumps require migration in `config/loader.py`.

#### `rubrics.yaml` schema (top-level)

```yaml
schema_version: 1
project_type: rag_and_tools          # enum: rag | tools | tool_calling | agent | rag_and_tools | custom
judge:
  default: <litellm-model-id>        # required
  regression_check: <litellm-model-id>  # optional; enables `diff --explain`
  fallback: [<model-id>, ...]        # optional ordered fallbacks
defaults:
  parallel: 4
  cache: true
  tolerance: 0.02
tasks:
  <task-name>:                       # snake_case, unique
    file_path: <relative-path>       # required
    entry: <python-symbol>           # optional; default is module.main
    type: tool_calling | rag | agent | chat
    inputs: [<key>, ...]             # field names in golden_set examples
    outputs: [<key>, ...]
    metrics:
      - name: <metric-id>
        threshold: <0..1>
        weight: 1.0
        params: { ... }              # metric-specific
    judge_overrides:                 # optional per-task override
      default: <model-id>
```

#### `golden_set.json` schema

```json
{
  "schema_version": 1,
  "tasks": {
    "<task-name>": [
      {
        "id": "gs_<sha8>",
        "captured_at": "2026-06-25T04:11:00Z",
        "input": { "...": "..." },
        "expected": { "...": "..." },     // optional; populated if test asserts
        "trace": {
          "calls": [
            { "kind": "llm|tool|retrieve", "name": "...", "args": {}, "result": {} }
          ]
        },
        "tags": ["from:pytest::test_support_happy_path"]
      }
    ]
  }
}
```

#### Run record (`runs/<id>/record.json`)

```json
{
  "schema_version": 1,
  "id": "r_8f3a",
  "started_at": "...", "finished_at": "...",
  "git": { "sha": "...", "branch": "...", "dirty": true },
  "config_hash": "sha256:...",
  "baseline_id": "r_8e21",
  "tasks": {
    "customer_support_agent": {
      "examples": 47,
      "metrics": {
        "argument_accuracy":  { "score": 0.93, "delta": 0.01, "threshold": 0.9, "status": "pass" },
        "hallucination_rate": { "score": 0.04, "delta": -0.01, "threshold": 0.1, "status": "pass" }
      },
      "latency_ms": { "p50": 1200, "p95": 3100 }
    }
  },
  "summary": { "passed": 3, "failed": 1, "errors": 0 },
  "tags": ["pr-482"]
}
```

#### `history.json`

Append-only list of slim records (id, started_at, summary, git.sha, tags). Full data lives in `runs/<id>/`.

### 2.5 The judge gateway (model-agnostic core)

`judge.gateway.score(...)` is the single seam between the tool and any LLM.

- Input: `JudgeRequest{ task, example, output, metric, tier_hint }`.
- Selects prompt tier:
  - **Complex tier** (CoT) when model id matches an allowlist of frontier models (regex on `gpt-4*`, `claude-3.*`, `gemini-1.5-pro*`, override via `judge.tiering.complex_models`).
  - **Basic tier** (binary checklist) otherwise (default for local models).
- Renders prompt → calls `litellm.completion(..., response_format=...)` through `instructor` for strict Pydantic parsing.
- Wraps in retry-with-fallback: `judge.default` → `judge.fallback[0]` → … → fail (collected into run record as `errors`).
- All responses are cached by `sha256(model_id || prompt_text || schema_id)`.
- Output: `JudgeResponse{ score: float, rationale: str, sub_scores: dict, raw: any }`.

### 2.6 Metric model

Each metric is a Python module exporting:

```python
def compute(examples: list[Example], outputs: list[Any], judge: Judge, params: dict) -> MetricResult: ...
```

Built-in metrics (Phase 4):

| Metric | Type | Needs judge? | Notes |
|---|---|---|---|
| `argument_accuracy` | tool_calling | yes | judge compares emitted tool args to expected/structural correctness |
| `hallucination_rate` | rag / chat | yes | judge inspects output vs. retrieved context |
| `context_precision` | rag | partial | deterministic + judge tiebreak |
| `faithfulness` | rag / chat | yes | judge grounds claims in context |
| `latency_p50` / `latency_p95` | any | no | measured by runner |

Custom metrics live under `eval/metrics/<name>.py` and are picked up by `runner.metrics` via entry-point scan; `rubrics.yaml` references them by name.

### 2.7 Detectors (extension point)

Each detector subclasses `inference.detectors.base.Detector`:

```python
class Detector(ABC):
    framework: str          # "openai", "langchain", ...
    def matches(self, module_ast, imports) -> bool: ...
    def extract(self, module_ast, file_path) -> list[TaskSpec]: ...
```

Built-in set ships with Phase 1: `openai_chat`, `openai_tools`, `langchain`, `llamaindex`, `chromadb`, `pinecone`, `pydantic_ai`. Third parties add new detectors via Python entry-point group `ai_eval.detectors`.

### 2.8 Bootstrap instrumentation strategy

Two strategies, chosen automatically:

1. **Monkey-patch (default).** On `ai-eval bootstrap`, before the subprocess starts, install a `sitecustomize.py` pointer in the env that patches:
   - `openai.OpenAI.chat.completions.create`
   - `langchain_core.language_models.BaseChatModel.invoke`
   - `chromadb.Collection.query`, `pinecone.Index.query`
   - any user-registered framework adapter
   Each patch wraps the call to record `{input, output, latency_ms, tool_calls}` then delegates.
2. **OpenTelemetry (`--no-instrument`).** Read `OTEL_EXPORTER_OTLP_ENDPOINT` and ingest spans matching known semantic conventions; map to the same trace shape. Used when the user already has OTEL in place.

Captures are deduped by `sha256(input_canonical_json)` per task and capped by `--max`.

### 2.9 Concurrency, caching, determinism

- **Parallelism:** `asyncio` + `litellm.acompletion`; semaphore = `--parallel`.
- **Cache:** content-addressed; `--no-cache` disables read+write.
- **Determinism:** runner sets `temperature=0` on judges by default; user can override per-task via `judge_overrides`.
- **Locks:** `.ai-eval/config.lock` (fcntl advisory) so two `run`s don't corrupt history.

### 2.10 Configuration resolution order

Highest wins:

1. CLI flag
2. Env var (`AI_EVAL_<UPPER_SNAKE>`)
3. `eval/rubrics.yaml`
4. `~/.config/ai-eval/config.yaml` (user-global defaults; e.g. API keys, default judge)
5. Built-in defaults in `config/defaults.py`

`ai-eval config --print` shows the merged view annotated with source per key.

### 2.11 Logging + telemetry

- Structured JSON logs to stderr at `--verbose`. Single-line per event, fields `ts`, `level`, `event`, `task`, `metric`, `model`, `duration_ms`.
- No outbound telemetry by default. Opt-in `--telemetry on` (or `AI_EVAL_TELEMETRY=on`) for anonymous version pings only — design hook only; not implemented in Phase 1–4.

### 2.12 Security + safety

- Never write outside `<cwd>/eval/`, `<cwd>/.ai-eval/`, or `~/.config/ai-eval/` without `--out`.
- Never read files matched by `.gitignore` or `--exclude` during inference.
- Redact env-var-looking strings (`OPENAI_API_KEY=…`, `Bearer …`) from captured traces.
- `bootstrap` honors a per-task `redact: [field, …]` list in `rubrics.yaml`.
- All judge prompts include a "do not execute tools or fetch URLs" preamble.

### 2.13 Packaging + distribution

- Package name: `ai-eval` (PyPI). Entry point: `[project.scripts] ai-eval = "ai_eval.cli.app:main"`.
- Python `>=3.10`.
- Runtime deps: `typer`, `rich`, `pydantic>=2`, `instructor`, `litellm`, `pyyaml`, `httpx`, `anyio`.
- Dev deps: `pytest`, `pytest-asyncio`, `mypy`, `ruff`.
- Single binary not required; `pipx install ai-eval` is the recommended path. `uv tool install ai-eval` is documented.

### 2.14 Test strategy

- **Unit:** per detector (golden-fixture repos under `tests/fixtures/repos/*`), per metric, per renderer (snapshot via `syrupy`).
- **Integration:** end-to-end `init → bootstrap → run → diff` on a fixture repo using a stub LiteLLM transport.
- **Contract:** JSON schema tests for `rubrics.yaml`, `golden_set.json`, run record.
- **CLI:** Typer `CliRunner` snapshot tests for every command's `--help` and `--format json` output.

### 2.15 Failure & edge-case matrix

| Situation | Behavior |
|---|---|
| No `eval/` and user runs `run` | exit 2, "fix: ai-eval init" |
| `rubrics.yaml` schema_version newer than tool | exit 1, suggest `pip install -U ai-eval` |
| Judge model unreachable, fallback succeeds | warn on stderr, run proceeds |
| All judges unreachable | exit 1, partial run record saved with `errors` |
| Repo is not a git repo | run still works; `git` block in record is `null` |
| Bootstrap subprocess exits non-zero | captures still flushed; exit 1 with original code preserved as `--exit-from-child` |
| Golden set empty for a task | task marked `skipped` with reason, not failed |
| Detector finds zero tasks | `init` writes a stub `rubrics.yaml` with a TODO and prints next-step doc URL |
| `--format json` while interactive | json to stdout, no spinner, no color, regardless of TTY |

---

## 3. Mapping to the 4 phases in `New_idea.md`

| Phase | Deliverables from this design |
|---|---|
| **Phase 1 — Static analysis & scaffolding** | `cli/app.py` + `cli/init.py` + `cli/analyze.py`; `inference/*` with `openai`, `langchain`, `chromadb` detectors; `scaffold/*` writing `rubrics.yaml`, `tests.py`, `golden_set.json` stub; `config/schema.py`. Exit-code contract live for `init`/`analyze`. Snapshot tests for `--help`. |
| **Phase 2 — Model-agnostic gateway & extraction** | `judge/gateway.py`, `judge/prompts/{cot_complex,checklist_basic}.py`, `judge/schemas.py`, `judge/instructor_glue.py`, `judge/tiering.py`, `judge/cache.py`. `cli/judge.py` (`--list`, `--ping`, `--prompt`). `doctor` checks providers. |
| **Phase 3 — Golden set bootstrapper** | `bootstrap/wrappers.py` per framework, `bootstrap/tracer.py`, `bootstrap/golden_writer.py`. `cli/bootstrap.py` honoring `-- <cmd>` passthrough. Redaction layer. |
| **Phase 4 — Runner + insights dashboard** | `runner/engine.py`, `runner/metrics/*`, `runner/thresholds.py`. `insights/{history,diff,root_cause}.py`. `cli/{run,diff,report,history}.py`. Rich renderers in `cli/render/*`. `--fail-on-regression` exit-3 path. |

---

## 4. Open questions for the user (please confirm before Phase 1 code)

1. **Tool name** — keep `ai-eval` as the binary, or prefer something else (`aieval`, `aev`)?
2. **History storage** — `.ai-eval/history.json` (this plan) vs. SQLite (`.ai-eval/state.db`)? JSON is simpler; SQLite scales better past ~10k runs.
3. **`eval/` location** — fixed at `<cwd>/eval/`, or configurable via `--config <dir>` from day one?
4. **Default judge for first-run** — bias toward local (`ollama/qwen2.5-coder:7b`, requires Ollama) or cloud (`openai/gpt-4o-mini`, requires `OPENAI_API_KEY`)? Plan currently assumes Ollama default with cloud fallback for regression judge.
5. **Bootstrap default trigger** — assume `pytest` if no `-- <cmd>` given, or require explicit command?
6. **CI mode** — should `run` auto-detect CI (env `CI=true`) and switch to `--no-input --format json --no-color`? Plan assumes yes.
7. **Pluggable metrics in Phase 4** — ship as entry-points (`ai_eval.metrics`) from v0.1, or defer to v0.2?

Answer these and Phase 1 implementation can start cleanly.

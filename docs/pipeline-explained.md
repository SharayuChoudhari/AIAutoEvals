# The ai-evals Pipeline, Explained Simply

This document explains how `ai-evals` works end to end — what each step does,
why it exists, and what files it touches. No code knowledge required.

---

## The Big Picture

`ai-evals` is a tool that **evaluates your AI-powered code automatically**.
Instead of you having to hand-write test cases and graders, it scans your
repository, figures out where your AI tasks live, captures real examples of
them running, and then scores them using an AI "judge" — a second model that
reads what your task did and grades how well it did it.

The whole thing is:

- **Zero-config** — you run one command and it figures out what to test.
- **Repo-aware** — it reads your actual source code to discover tasks.
- **Model-agnostic** — the judge can be a local model (like Ollama) or a cloud
  model (like OpenAI), swapped freely without changing your tests.

### The 7-step pipeline

```
 ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌───────┐    ┌────────┐
 │  init   │ -> │ analyze  │ -> │ bootstrap │ -> │  run  │ -> │  diff  │
 └─────────┘    └──────────┘    └───────────┘    └───────┘    └────────┘
       \             \               \               /            /
        \             \               \             /            /
         \             \               \           /            /
          v             v               v         v            v
                                  ┌─────────┐   ┌──────────┐
                                  │ report  |   │ history  │
                                  └─────────┘   └──────────┘
```

In words:

1. **`init`** — Scan your repo and scaffold the eval setup.
2. **`analyze`** — Re-scan and merge any new findings into your config.
3. **`bootstrap`** — Capture real runtime traces by running your actual code.
4. **`run`** — Execute every task against its examples and score each one.
5. **`diff`** — Compare the latest run to a previous one to spot regressions.
6. **`report`** — Render a past run as a readable report.
7. **`history`** — List, inspect, or clean up past runs.

You don't have to run all of these every time. A typical cycle is
`init` (once) → `bootstrap` (whenever you want real examples) →
`run` (every change) → `diff` (to check you didn't break anything).

---

## The Pipeline, Step by Step

### 1. `init` — Discover your tasks and set up the eval

**What it does:**
`init` walks your repository, reads every Python file, and looks for places
where you're using an AI framework — OpenAI chat completions, tool/function
calling, the Responses API, LangChain, LangGraph, ChromaDB, pgvector, and so
on. Each place it finds becomes a **task**: a unit of AI work that ai-evals
will later test.

It then writes a set of files into an `eval/` folder that describe what it
found and how to test it.

**Why we do it:**
The core promise of ai-evals is "zero-config." You shouldn't have to tell the
tool what your AI tasks are — it should read your code and figure it out. This
step is what makes that possible. It also creates the scaffold files you'll
edit by hand later (like the hints file or the test harness stubs).

**What it reads:**
- Your repository's Python source code.
- An optional `eval/ai-evals.hints.yaml` (if you've added hints before).

**What it writes:**
- `eval/rubrics.yaml` — the main config: your tasks, metrics, and judges.
- `eval/golden_set.json` — a starter set of test examples (auto-seeded).
- `eval/tests.py` — a pytest entry point that shells out to `ai-evals run`.
- `eval/_harness_<task>.py` — one stub per "IO-coupled" task (more below).
- `eval/ai-evals.hints.yaml` — a commented template for optional hints.
- `.ai-evals/` — the local state directory (gitignored).
- Updates your `.gitignore` to ignore `.ai-evals/`.

**In simple terms:**
Think of `init` like a new employee reading your codebase and writing down,
"Okay, here are the AI things this project does, here's how I'd test each one,
and here's a starter checklist." Everything after this step builds on that
checklist.

---

### 2. `analyze` — Re-scan and keep your config in sync

**What it does:**
`analyze` runs the same detection as `init`, but instead of starting fresh, it
**merges** its findings into your existing `rubrics.yaml`. By default it only
*shows* you the proposed changes (a dry run); you pass `--write` to apply them.

**Why we do it:**
Code changes. You add a new AI endpoint, rename a function, switch from
LangChain to raw OpenAI calls. `analyze` lets you refresh the eval config to
match your current code **without losing the manual tweaks you've made** to
`rubrics.yaml` (like custom thresholds or metrics). It backs up your old file
before writing and validates the result, so a bad merge can't silently corrupt
your config.

**What it reads:**
- Your repository's source code.
- The existing `eval/rubrics.yaml`.
- Optional `eval/ai-evals.hints.yaml`.

**What it writes:**
- An updated `eval/rubrics.yaml` (only with `--write`; otherwise just prints).

**In simple terms:**
`init` is "set up from scratch." `analyze` is "catch me up on what changed
since the last setup." You run it after refactoring or adding new AI features.

---

### 3. `bootstrap` — Capture real examples by running your code

**What it does:**
`bootstrap` runs **your actual command** (e.g. `ai-evals bootstrap -- pytest -q`
or `-- python main.py`) in a child process. While your code runs, ai-evals
quietly watches every call your code makes to an AI framework — every OpenAI
chat completion, every LangChain invocation, every ChromaDB query — and records
what went in and what came out.

Each top-level task invocation becomes one **golden example**: a record of the
input, the expected output, and the trace of all the internal AI calls it made.
These get merged into `eval/golden_set.json`.

**Why we do it:**
Auto-seeded examples (from `init`) are guesses based on the shape of your
inputs. They let `run` produce *some* score immediately, but they're not real.
Bootstrap gives you **real, captured examples** from your actual code running
against real data. These are the ground truth that `run` will test against in
the future. And because they're captured, you never have to hand-write test
fixtures.

Captured examples are **never overwritten** by later runs unless you explicitly
pass `--replace`. The tool treats real captures as precious.

**What it reads:**
- `eval/rubrics.yaml` (to know which tasks to capture).
- Your runtime (whatever command you pass after `--`).

**What it writes:**
- `eval/golden_set.json` (merged: new captures added, old ones kept).

**In simple terms:**
Imagine recording a screencast of your AI app doing its job, then turning that
recording into a replayable test. Bootstrap is the recorder. Each replay becomes
a test case `run` will grade later.

---

### 4. `run` — Execute and score every task

**What it does:**
This is the heart of the tool. For each task in your config, and for each golden
example, `run`:

1. Imports your task's entry point (the function or method ai-evals detected).
2. Calls it with the example's input.
3. Gets the output back.
4. Sends the input, output, and the example's expected result to the **judge** —
   a second AI model that grades how good the output is.
5. Records the score for each metric (e.g. `argument_accuracy`,
   `hallucination_rate`, `faithfulness`).
6. Compares the scores to a baseline run (if one exists) to detect regressions.
7. Saves everything to a run record under `.ai-evals/runs/`.

**Why we do it:**
This is the actual evaluation. The judge gives you a quality score you can't
get from a simple pass/fail assertion — "did the model cite the right source?",
"did it hallucinate?", "were the arguments correct?". By comparing to a
previous run, you can tell if a code change made your AI better or worse.

**What it reads:**
- `eval/rubrics.yaml` (tasks, metrics, thresholds, judge config).
- `eval/golden_set.json` (the examples to test).
- A previous run record (the baseline, if you want deltas).
- The judge cache (to skip re-judging identical calls).

**What it writes:**
- `.ai-evals/runs/<id>/run.json` — the full run record.
- `.ai-evals/runs/<id>/traces.ndjson` — per-example traces (optional).
- `.ai-evals/runs/<id>/judge_log.ndjson` — judge call log (optional).
- `.ai-evals/history.json` — a slim summary entry.
- `.ai-evals/cache/judge/<hash>.json` — cached judge responses.

**In simple terms:**
`run` is the exam. Your tasks are the students, the golden examples are the
questions, and the judge is the teacher grading the answers. At the end you get
a report card — and if you've run before, it shows whether grades went up or
down.

With `--fail-on-regression`, `run` exits with code `3` if any metric got
meaningfully worse. That's what makes it usable in CI: a failing build means
your AI got worse.

---

### 5. `diff` — See what changed between runs

**What it does:**
`diff` compares the latest run to an earlier one (by default, the previous run)
and shows, for every task and metric, how much the score moved. A negative delta
means a regression.

With `--explain` (on by default), it goes further: for each metric that got
worse, it asks the **regression judge** (a separate model, usually a cheap cloud
one like `gpt-4o-mini`) to look at the current and previous outputs and write a
short narrative explaining *why* the score dropped.

**Why we do it:**
A number that says "faithfulness dropped 0.12" is useful, but "faithfulness
dropped because the model started citing paragraph 3 instead of paragraph 1,
which wasn't in the source" is far more actionable. `diff` turns raw deltas into
human-readable root-cause notes so you can fix the actual problem, not just
stare at a number.

If no judge is available, `diff` falls back to a plain text comparison of the
outputs — it never crashes just because the judge is offline.

**What it reads:**
- Two run records (current + baseline).
- The regression judge (for `--explain` narratives).

**What it writes:**
- Nothing (it's read-only).

**In simple terms:**
`diff` is the before-and-after. It's how you answer "did my change help or
hurt?" — and, with `--explain`, "and why?"

---

### 6. `report` — Render a run as a readable document

**What it does:**
`report` takes a past run (by ID, or `last` by default) and renders it in a
format you can share or archive: human-friendly text in the terminal, JSON for
machines, Markdown for docs, or a self-contained HTML page (`--web`) you can
open in a browser.

**Why we do it:**
Run records are JSON files on disk — great for the tool, bad for humans. `report`
turns them into something you can actually read, paste into a PR, or show a
teammate. The HTML output is especially handy for sharing a single run without
needing the terminal.

**What it reads:**
- One run record (from `.ai-evals/runs/<id>/run.json`).

**What it writes:**
- An output file only if you pass `--out <file>` or `--web`.

**In simple terms:**
`report` is the "export to readable format" button. The data's already there
from `run`; this just makes it presentable.

---

### 7. `history` — Browse and manage past runs

**What it does:**
`history` lets you list every run ai-evals has recorded, inspect any one in
detail, prune old runs to save disk, or export the whole history to a file.

**Why we do it:**
Over time, `.ai-evals/runs/` fills up. `history` gives you a way to see what's
there and clean it up without manually digging through JSON files. It's also how
you find a run ID to pass to `diff --against` or `report`.

**What it reads:**
- `.ai-evals/history.json` and the run directories.

**What it writes:**
- Deletes run directories when you `--prune`.
- Writes an export file when you `--export`.

**In simple terms:**
`history` is your run archive manager — list, view, clean up, or export.

---

## The Files ai-evals Creates

ai-evals splits its files into two places: `eval/` (checked into git, part of
your project) and `.ai-evals/` (local cache, gitignored).

### `eval/` — checked in, shareable

| File | What it holds | Written by | Read by | Why it's checked in |
|---|---|---|---|---|
| `rubrics.yaml` | The main config: tasks, metrics, thresholds, judge settings. | `init`, `analyze` | `run`, `judge`, `doctor`, `bootstrap` | It's the spec for your evals — teammates and CI need the same one. |
| `ai-evals.hints.yaml` | Optional manual hints to help detection (force-include a task, exclude judge code). | `init` (template only) | `init`, `analyze` | It's your hand-edited override of detection; it must travel with the repo. |
| `golden_set.json` | The test examples (captured + auto-seeded). | `init` (seeds), `bootstrap` (captures) | `run` | The examples *are* the tests. Without them checked in, CI can't run. |
| `tests.py` | A thin pytest entry point that calls `ai-evals run`. | `init`, `analyze` | pytest | Lets `pytest` and `ai-evals run` hit the same engine, so your existing test workflows just work. |
| `_harness_<task>.py` | Per-task stubs for "IO-coupled" tasks (those that talk to a DB/API). | `init`, `analyze` | `run` | Lets `run` execute tasks offline by faking the backend, so you don't need a live database to evaluate. |

### `.ai-evals/` — local, gitignored

| Path | What it holds | Written by | Read by | Why it's gitignored |
|---|---|---|---|---|
| `history.json` | Slim summaries of every run (id, time, scores, git sha). | `run` | `history`, `diff`, `report`, `run` | It's machine- and time-specific; not meaningful to other developers. |
| `runs/<id>/run.json` | The full record of one run. | `run` | `diff`, `report`, `run` (as baseline) | Same — large, local, ephemeral. |
| `runs/<id>/traces.ndjson` | Per-example execution traces. | `run` | (debugging) | Debug detail; not part of the eval spec. |
| `runs/<id>/judge_log.ndjson` | Log of every judge call. | `run` | (debugging) | Same. |
| `cache/judge/<hash>.json` | Cached judge responses, keyed by prompt+model. | `run` | `run` | Pure performance cache; safe to delete. |
| `rubric_cache/` | Cached SLM rubric-classification responses. | `init`, `analyze` | `init`, `analyze` | Same — performance cache. |

The split is deliberate: **`eval/` is your project's evaluation contract**
(everyone shares it), while **`.ai-evals/` is your machine's local state**
(nobody else needs it).

---

## Under the Hood (Briefly)

Three mechanisms make the whole thing work. Here's the plain-English version.

### How ai-evals finds your tasks (Detection)

When you run `init` or `analyze`, ai-evals doesn't just look for the word
"openai" in your files. It parses your code into an abstract syntax tree (a
structured representation of the code) and walks it looking for specific
patterns — a call to `client.chat.completions.create`, a `tools=` keyword, a
LangGraph `StateGraph`, a ChromaDB `.query()`, and so on. Each framework has a
small "detector" that knows its signature.

Then it does two smart things:

1. **Call-graph demotion.** If task A calls task B internally, only A is a
   public surface worth testing — B is an implementation detail. ai-evals traces
   which tasks call which others and demotes the internal ones so you don't end
   up testing the same logic twice.

2. **Judge-exclusion.** ai-evals' own test files (and any code that looks like a
   judge or scorer) are excluded, so the tool doesn't try to evaluate its own
   evaluation code.

You can override both with the hints file if it gets something wrong.

### How scoring works (The Judge Gateway)

For every (example, metric) pair, `run` builds a request and sends it to the
judge — a second AI model. The judge reads the task, the input, the output, and
the expected result, then returns a score between 0 and 1 plus a short
rationale.

A few things make this robust:

- **Model-agnostic.** The judge is just a model ID (e.g. `ollama/qwen2.5-coder:7b`
  or `openai/gpt-4o-mini`). Everything goes through litellm, so you can swap
  models without touching your tests.
- **Tiered prompts.** Big frontier models (GPT-4, Claude Opus) get a detailed
  chain-of-thought prompt; small local models get a simpler yes/no checklist.
  ai-evals picks the right one based on the model ID.
- **Caching.** Every judge call is content-addressed — same prompt + same model
  = cached result. So re-running an unchanged eval is nearly free.
- **Fallback chains.** If the default judge is down, ai-evals tries configured
  fallbacks. It collects errors but keeps going rather than crashing.

### How the runner executes and grades (The Runner)

The runner is the engine behind `run`. For each task, it imports your code's
entry point and calls it once per golden example. For IO-coupled tasks (those
that hit a database or API), it first installs the generated harness stub so the
task runs against fake data instead of a live backend.

Then, for each example, it asks the judge to score every metric the task
declares. Scores are averaged per task, compared to the baseline run to compute
deltas, and checked against thresholds. Latency metrics (`p50`, `p95`) are
measured directly — no judge needed.

The result is a run record: a full snapshot of what happened, saved to disk so
`diff`, `report`, and future `run`s (as baselines) can use it.

---

## Why This Design

A few choices show up repeatedly across the pipeline. Here's the reasoning
behind each.

- **Golden-set preservation.** Real captured examples are hard to get and
  represent ground truth. The tool refuses to overwrite them without an explicit
  flag (`--replace` / `--reset-golden`). Auto-seeds (the guesses from `init`)
  are clearly marked and regenerable, so they never masquerade as real data.

- **Model-agnostic judging.** By routing everything through litellm, the judge
  becomes a swappable config value. You can develop against a free local model
  and run CI against a stronger cloud model without changing a line of config.

- **CI-friendly defaults.** When `CI=true` is set, ai-evals automatically
  switches to JSON output, disables color, and refuses interactive prompts.
  Destructive operations (like overwriting config in CI) require an explicit
  `--yes`. The `run --fail-on-regression` exit code `3` slots cleanly into any
  CI system's pass/fail logic.

- **Local-first.** All run state lives in `.ai-evals/`, gitignored. Nothing is
  uploaded anywhere. The only network calls are to the judge model you
  configure — and even those are cached locally.

- **Safety exits everywhere.** `diff --explain` falls back to a text comparison
  if the judge is offline. `bootstrap` flushes captured traces even if your
  command crashes. `doctor` never writes anything. The tool is built to degrade
  gracefully rather than crash on missing optionals.

- **Clear exit codes.** `0` success, `1` general error, `2` usage error, `3`
  evaluation failed thresholds. Each maps to a distinct CI outcome, so a
  regression is never confused with a crash or a misconfigured flag.

# Plan: `docs/pipeline-explained.md`

## Goal
Create a plain-English markdown doc that explains the ai-evals pipeline: the overall flow, the purpose of each step, what each file/module does, and why the system is built this way. Audience = someone who wants to understand the system without reading code. No code snippets, no jargon left unexplained.

## Scope
- Create one new file: `docs/pipeline-explained.md`
- No source edits, no other files touched
- Light, conversational tone; analogies where helpful
- No `file:line` references — keep it readable end-to-end

## Doc structure

1. **Big picture**
   - One paragraph: what ai-evals does (zero-config, repo-aware, model-agnostic AI evaluation).
   - ASCII flow diagram of the 7-step pipeline: `init → analyze → bootstrap → run → diff → report → history`.

2. **The pipeline, step by step**
   For each of the 7 steps, a short subsection covering:
   - What it does (plain English)
   - Why we do it (the purpose / problem it solves)
   - What it reads
   - What it writes
   - A concrete "in simple terms" analogy/example

3. **The files ai-evals creates**
   - Table of `eval/*` files (`rubrics.yaml`, `ai-evals.hints.yaml`, `golden_set.json`, `tests.py`, `_harness_*.py`) and `.ai-evals/*` files (`history.json`, `runs/<id>/run.json`, `cache/`, `rubric_cache/`).
   - For each: what it holds, who writes it, who reads it, why it's checked-in vs gitignored.

4. **Under the hood (briefly)**
   Three short plain-English sections on cross-cutting mechanisms:
   - **Detection** — how ai-evals scans a repo and figures out what your AI tasks are (AST scan + framework detectors + call-graph demotion + judge-exclusion, in simple terms).
   - **The judge gateway** — how scoring works (one judge call per example+metric, model-agnostic via litellm, content-addressed cache, tiered prompts for big vs small models, fallback chain).
   - **The runner** — how it executes your tasks, scores each metric, compares to a baseline, and decides pass/fail/regression.

5. **Why this design**
   - Short "why we do it this way" notes: golden-set preservation (never overwrite real captures), model-agnostic judging (swap models freely), CI-friendly defaults (auto JSON/no-color in CI), local-first (state stays in `.ai-evals/`), and the safety exits (exit codes, `--fail-on-regression`, fallbacks that never crash).

## Validation
- File exists at `docs/pipeline-explained.md`
- Reads cleanly as a standalone explainer with no missing context
- Covers all 7 pipeline steps + the file table + the 3 under-the-hood sections + the design rationale

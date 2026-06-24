Here is a comprehensive blueprint for your tool. It is structured as a clear, production-ready `plan.md` file that you can drop straight into your repository to guide your development.

---

# plan.md: Building `ai-eval` (The Zero-Config AI Evaluation CLI)

This document outlines the core architecture, technical requirements, and incremental execution phases for building a model-agnostic, repository-aware AI evaluation tool.

## 1. Core Architecture Overview

The system is designed as a local-first, zero-config CLI tool written in **Python** (to leverage advanced Abstract Syntax Tree analysis for AI codebases) using **LiteLLM** for model-agnostic orchestration.

```
       [ Local Repository ]
                │
                ▼ (Static Analysis / AST Parsing)
   ┌──────────────────────────┐
   │ 1. Inference Engine      │ ──► Infers tasks (RAG, Tools, Agents)
   └──────────────────────────┘
                │
                ▼ (Generates Scaffold)
   ┌──────────────────────────┐
   │ 2. Scaffolding Engine    │ ──► Creates eval/rubrics.yaml & eval/tests.py
   └──────────────────────────┘
                │
                ▼ (Runs Execution Pipeline)
   ┌──────────────────────────┐
   │ 3. Judging Layer         │ ──► Orchestrates Model-Agnostic Evaluators
   └──────────────────────────┘     (OpenAI, Anthropic, Ollama via LiteLLM)
                │
                ▼
   ┌──────────────────────────┐
   │ 4. Insights Dashboard    │ ──► CLI Regression Diff Report
   └──────────────────────────┘

```

---

## 2. Technical Component Breakdown

### A. The Inference Engine (Repo-Awareness)

* **Mechanism:** Uses Python’s built-in `ast` module to scan the codebase.
* **Signature Detection:** Look for specific imports (`openai`, `langchain`, `pinecone`, `chromadb`) and structural patterns like function dictionaries passed to an LLM (`tools=[...]`).
* **Output:** Generates a localized system state map defining exactly what AI features exist, what parameters they accept, and what metrics they require.

### B. The Model-Agnostic Judging Layer

* **Engine:** Powered by `LiteLLM` to support local endpoints (`ollama`, `vllm`) and commercial APIs seamlessly.
* **Structured Outputs:** Uses `Pydantic` and `Instructor` to force strict JSON schema compliance across different tiers of models.
* **Dual-Tier Prompting:**
* *Complex Tier:* Chain-of-Thought (CoT) semantic scoring for frontier models.
* *Basic Tier:* Deconstructed binary checklist scoring (Yes/No questions) for smaller local models.



### C. The Insights Engine

* **The Baseline:** Stores past evaluation runs in a local `.ai-eval/history.json` file.
* **The Diff:** Instead of presenting flat metrics, it computes a semantic delta:

$$\Delta = \text{Score}_{\text{New}} - \text{Score}_{\text{Baseline}}$$


* **Root Cause Extraction:** If a regression occurs, the judge model is fed the old trace and the new trace to compile an explicit "Why it failed" explanation.

---

## 3. Development Roadmap

1. **Phase 1: Static Analysis & Scaffolding:** Milestone 1.
Build the CLI boilerplate (`typer` or `click`) and implement the AST parser. The parser must scan a directory, detect an LLM call or tool schema, and successfully output an `eval/rubrics.yaml` configuration describing the detected features.


2. **Phase 2: Model-Agnostic Gateway & Extraction:** Milestone 2.
Integrate `LiteLLM` and set up `Pydantic` validation. Create the prompt compilation layer that reads `rubrics.yaml` and translates it into strict JSON-returning prompt payloads optimized for both small local models (Llama 3) and large external models (GPT-4o).


3. **Phase 3: Automated Golden Set Bootstrapper:** Milestone 3.
Implement a simple runtime logger/wrapper. When executed in bootstrap mode, it captures real inputs/outputs from the developer's local runtime tests and automatically synthesizes them into an initial `eval/golden_set.json` file.


4. **Phase 4: Run Execution & UI Dashboard:** Milestone 4.
Build the test runner execution engine (`ai-eval run`). Design the terminal output interface using `rich` to display colored regression diffs, execution latency changes, and specific trace-level failure reasons.


---

## 4. Initial Configuration Schema (`eval/rubrics.yaml`)

This is the file the tool will automatically output during the `init`/`analyze` phase:

```yaml
version: "1.0"
project_type: "rag_and_tools"
judge:
  default: "ollama/qwen2.5-coder:7b"
  regression_check: "openai/gpt-4o-mini"

tasks:
  customer_support_agent:
    file_path: "./src/agents/support.py"
    type: "tool_calling"
    metrics:
      - name: "argument_accuracy"
        threshold: 0.9
      - name: "hallucination_rate"
        threshold: 0.1
    
  knowledge_base_retriever:
    file_path: "./src/rag/kb.py"
    type: "rag"
    metrics:
      - name: "context_precision"
        threshold: 0.85
      - name: "faithfulness"
        threshold: 0.95

```

---

This plan sets you up with a clear path forward. If you want to start building right now, we can jump into the code for **Phase 1** to get that AST parser mapping out your codebase.
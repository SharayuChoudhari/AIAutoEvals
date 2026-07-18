# Metric registry

`ai-evals` ships a minimal high-frequency built-in metric set. Niche and
project-specific metrics live in the consuming project's `eval/metrics.yaml`.

## Built-in metrics

These are always available (no configuration needed):

| Metric | Description | Default threshold | Applicable task types |
|---|---|---|---|
| `hallucination_rate` | Rate of unsupported claims relative to grounded context. | 0.1 | chat, rag, agent, tool_calling |
| `context_precision` | Precision of retrieved context relevance to the query. | 0.85 | rag |
| `faithfulness` | Faithfulness of the answer to the retrieved context. | 0.9 | rag, summarization |
| `argument_accuracy` | Accuracy of tool/function call arguments vs. expected. | 0.9 | tool_calling, agent |
| `task_completion` | Whether the end-to-end task completed successfully. | 0.9 | booking, workflow, agent |
| `latency_p50` | Median task latency in ms (computed by the runner, non-judge). | — | all |
| `latency_p95` | p95 task latency in ms (computed by the runner, non-judge). | — | all |

## Project-local metrics (`eval/metrics.yaml`)

Niche metrics that were previously built-in (`slot_filling_accuracy`,
`translation_bleu`, `classification_f1`, `extraction_field_accuracy`,
`scoring_accuracy`, `summary_faithfulness`) and any custom metric now live in
your project's `eval/metrics.yaml`. Create the file and declare the metrics
you need.

### Schema

```yaml
metrics:
  - name: slot_filling_accuracy           # required, snake_case
    description: Accuracy of slots filled. # required
    applicable_task_types: [booking, workflow]  # optional
    scored_dimension: slot_filling_accuracy     # optional, defaults to name
    threshold: 0.9                               # optional, default 0.0
    weight: 1.0                                  # optional, default 1.0
    non_judge: false                             # optional, default false
    prompt_template: |                           # required for judge metrics
      Evaluate slot filling for the following:
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score 1.0 for exact match, partial credit per correct slot.
```

### `prompt_template` placeholders

`prompt_template` uses simple `{{name}}` substitution (no Jinja2):

| Placeholder | Value |
|---|---|
| `{{input}}` | The example's input (JSON-serialized). |
| `{{output}}` | The task's output (JSON-serialized). |
| `{{expected}}` | The example's expected output (JSON-serialized). |
| `{{task_type}}` | The task's type string. |
| `{{task_purpose}}` | The task's purpose description. |
| `{{metric_name}}` | The metric's name. |
| `{{metric_description}}` | The metric's description. |
| `{{scored_dimension}}` | The scored dimension name. |

### Merge order

1. Built-ins (always win on name collision).
2. Project-local `eval/metrics.yaml` (can only ADD names, never override
   built-ins).
3. Entry-point plugins (`ai_eval.metrics` group).

## Copy-paste template: re-declaring removed niche metrics

Copy this into your `eval/metrics.yaml` to restore the metrics that were
removed from the built-in set:

```yaml
metrics:
  - name: slot_filling_accuracy
    description: Accuracy of slots/arguments filled for booking or workflow tasks.
    applicable_task_types: [booking, workflow]
    scored_dimension: slot_filling_accuracy
    threshold: 0.9
    weight: 1.0
    prompt_template: |
      Evaluate slot filling for a booking/workflow task.
      task_type: {{task_type}}
      metric: {{metric_name}}
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score 1.0 for exact match, partial credit per correctly filled slot.
      Produce JSON: {"score": float, "rationale": str, "sub_scores": {"slot_filling_accuracy": float}}

  - name: translation_bleu
    description: BLEU score of translated output against references.
    applicable_task_types: [translation]
    scored_dimension: translation_bleu
    threshold: 0.4
    weight: 1.0
    prompt_template: |
      Evaluate translation quality (BLEU-style) for the following.
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score in [0, 1].
      Produce JSON: {"score": float, "rationale": str, "sub_scores": {"translation_bleu": float}}

  - name: classification_f1
    description: Macro-F1 across classification labels.
    applicable_task_types: [classification]
    scored_dimension: classification_f1
    threshold: 0.85
    weight: 1.0
    prompt_template: |
      Evaluate classification F1 for the following.
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score macro-F1 in [0, 1].
      Produce JSON: {"score": float, "rationale": str, "sub_scores": {"classification_f1": float}}

  - name: extraction_field_accuracy
    description: Per-field accuracy of extracted structured fields vs. reference.
    applicable_task_types: [extraction]
    scored_dimension: extraction_field_accuracy
    threshold: 0.9
    weight: 1.0
    prompt_template: |
      Evaluate field extraction accuracy.
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score 1.0 for exact match, partial credit per correct field.
      Produce JSON: {"score": float, "rationale": str, "sub_scores": {"extraction_field_accuracy": float}}

  - name: scoring_accuracy
    description: Agreement of model scores with reference rubric scores.
    applicable_task_types: [scoring]
    scored_dimension: scoring_accuracy
    threshold: 0.85
    weight: 1.0
    prompt_template: |
      Evaluate scoring accuracy against a reference rubric.
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score agreement in [0, 1].
      Produce JSON: {"score": float, "rationale": str, "sub_scores": {"scoring_accuracy": float}}

  - name: summary_faithfulness
    description: Faithfulness of a summary to its source text.
    applicable_task_types: [summarization]
    scored_dimension: summary_faithfulness
    threshold: 0.9
    weight: 1.0
    prompt_template: |
      Evaluate summary faithfulness to the source.
      input: {{input}}
      output: {{output}}
      expected: {{expected}}
      Score 1.0 when fully entailed by the source.
      Produce JSON: {"score": float, "rationale": str, "sub_scores": {"summary_faithfulness": float}}
```

## Strict mode

Unknown metric names emit a `DeprecationWarning` (one-release deprecation
window) rather than being rejected at schema-load time. Set
`AI_EVAL_STRICT_METRICS=1` to enforce hard rejection now. At run time,
`ai-evals run` always fails fast (`MetricNotImplementedError`, exit 1) on any
metric with no judge implementation.

## Node metrics (`node_metrics`)

`ai-evals run` runs only the end-to-end entry point per use case and scores
its internal nodes from the captured trace (`example["trace"]["calls"]`), not
re-executed. Bind a metric to a trace node via `node_metrics` on the entry
task:

```yaml
tasks:
  chat_message_service_process_query:
    # ... entry, type, metrics, top_level: true ...
    node_metrics:
      - node_selector: kind=retrieve
        metric:
          name: context_precision
          threshold: 0.85
          weight: 1.0
      - node_selector: name~=openai
        metric:
          name: hallucination_rate
          threshold: 0.1
```

### Selector grammar

A `node_selector` is a single clause (comma-free):

| Selector | Matches |
|---|---|
| `kind=<x>` | `call["kind"] == x` (e.g. `kind=retrieve`, `kind=llm`). |
| `name=<exact>` | Exact match on `call["name"]`. |
| `name~=<substr>` | Substring match on `call["name"]`. |
| `call_index=<n>` | The n-th call (0-based), regardless of kind/name. |

A selector may match multiple nodes; each match is scored independently and
gets a synthetic `node_id` (`<kind>_<i>`, e.g. `retrieve_0`, `retrieve_1`).
Per-node scores roll into `ExampleRecord.node_scores`; the task-level
aggregate is the weighted mean of node scores across examples.

### When to author `node_metrics`

Author `node_metrics` **after** a first `ai-evals bootstrap` reveals the real
`call["kind"]` / `call["name"]` values the framework wrappers record. `init`
writes `node_metrics: []` (empty) — guessing selector names pre-bootstrap is
fragile because the names are framework-wrangler-defined (e.g.
`pgvector.orm.op`). If a selector matches no nodes (stale trace shape), the
node pass is a no-op and the aggregate metric is `skip` — the run doesn't
crash.

# Populate WhatsApp-booking eval suite for appointment_helper

## Context & root cause

`appointment_helper` is a dentist appointment booking backend whose WhatsApp chat flow
is driven by an LLM tool-calling workflow in
`services/conversation_workflow_service.py`. The chat endpoint is
`POST /webhooks/twilio/whatsapp/message` → `ConversationService.handle_inbound_message`
→ `ConversationWorkflowService.run`, which calls the OpenAI **Responses API**
(`client.responses.create(tools=...)`) and dispatches five tools backed by
`AppointmentService`:

| tool                       | intent        | args                                            |
|----------------------------|---------------|-------------------------------------------------|
| `get_upcoming_appointment` | upcoming      | —                                               |
| `get_available_slots`      | availability  | `date`, `appointment_type_name` (nullable)      |
| `book_appointment`         | book          | `start_at` (ISO UTC), `appointment_type_name`   |
| `reschedule_appointment`   | reschedule    | `start_at` (ISO UTC)                            |
| `cancel_appointment`       | cancel        | `reason` (nullable)                             |

System-prompt guarantees (from `conversation_workflow_service.py:111`): ask **one
short follow-up** when a required slot is missing; **never** claim a booking /
reschedule / cancel succeeded unless the tool returns success.

### Why the eval currently "gives a bad result"

1. `eval/golden_set.json` is `{"schema_version": 1, "tasks": {}}` — no captures.
2. `eval/rubrics.yaml` has `tasks: {}` — no task specs.
3. `ai-eval analyze` keeps tasks empty because the `OpenAIToolsDetector`
   (`ai_eval/inference/detectors/openai_tools.py`) only matches
   `chat.completions.create(tools=...)`; this repo uses the Responses API
   (`responses.create`), which is not auto-detected. (Out of scope to fix here.)
4. `ai-eval run` is a Phase-4 placeholder (`ai_eval/cli/run.py:26` → `not_yet`).
   It will not execute tasks today. So this plan produces **declarative** eval
   scaffolding that becomes executable once Phase 4 lands; it does **not** add a
   runner or modify the `ai-eval` tool.

### Scope (confirmed with user)

- **In scope:** populate `eval/golden_set.json` and `eval/rubrics.yaml` in the
  `appointment_helper` repo with real WhatsApp-booking scenarios + metrics.
- **Out of scope:** adding a Responses-API detector, implementing a Phase-4
  runner, modifying the `ai-eval` package, or changing app source.

## Files to change

- `/Users/sharayuchoudhari/Dev/Projects/Appointment_booker/appointment_helper/eval/rubrics.yaml`
- `/Users/sharayuchoudhari/Dev/Projects/Appointment_booker/appointment_helper/eval/golden_set.json`

No source or tool changes.

## rubrics.yaml design

### Task type

All tasks use `type: workflow` (chosen per user). This makes
`task_completion` + `slot_filling_accuracy` the **native** metrics
(applicable_task_types include `workflow`); `argument_accuracy` and
`hallucination_rate` are cross-applicable. The flow is tool-calling at heart,
but `workflow` best captures the multi-step tool sequences (e.g. ask slots →
book) that these tasks exercise.

### Custom per-workflow judge scoring (required by user)

`TaskSpec` forbids extra fields (`schema.py:89` extra="forbid"), so the
workflow-specific scoring rubric is embedded via the sanctioned free-form fields:

- **`TaskSpec.purpose`** — a concise workflow-success definition per task
  (what success looks like for *this* workflow).
- **`MetricSpec.params.criteria`** (`schema.py:61` `params: dict[str, Any]`,
  no content validator) — metric-specific judge instructions tied to that
  workflow, referencing the golden capture's `expected` block.

The four canonical `params.criteria` blocks (reusable across all 7 workflow
tasks; they reference the workflow's expected tool/intent which lives in each
golden capture's `expected` block):

- **`task_completion`** params.criteria: "Score 1.0 only if the workflow's
  expected tool ran and returned success — e.g. `book_appointment` returned an
  `appointment_id`; `cancel_appointment` returned a `cancelled` status. Score
  0.0 if a wrong tool ran, the expected tool errored, or the assistant claimed
  completion without a tool success. Use the capture's
  `expected.intent` / `expected_tool_call`."
- **`slot_filling_accuracy`** params.criteria: "Score the fraction of required
  slots (`date` / `start_at` / `appointment_type_name` / `reason`) correctly
  extracted from the patient message and passed to the tool, vs the capture's
  `expected_tool_call.args_contains`. A missing required slot that triggers a
  follow-up question (instead of a fabricated arg) counts as correct
  slot-filling behavior."
- **`argument_accuracy`** params.criteria: "Score 1.0 only if tool args match
  the capture's expected ISO-UTC `start_at` (after resolving relative dates via
  `now_local` + `clinic_timezone`) and the expected `appointment_type_name`.
  Penalize wrong date conversions (IST↔UTC) and mis-selected appointment types."
- **`hallucination_rate`** params.criteria: "Score 1.0 if the assistant makes
  any claim not backed by a tool result — claiming booked/confirmed/moved/
  cancelled before a tool success, stating a slot time not returned by
  `get_available_slots`, or inventing an appointment_id. 0.0 otherwise. The
  capture's `must_not_claims` lists forbidden phrases."

### Clinic config pinned across all captures (README example)

- timezone: `Asia/Kolkata` (UTC+5:30)
- booking hours: 09:00–18:00 local
- appointment types: `Cleaning` (30 min), `Consultation` (30 min)
- granularity/min-notice etc. from README example policy; not material to
  golden expectations beyond slots existing within business hours.

### Tasks (type `workflow`). One task per booking intent plus the multi-turn
slot-filling and no-upcoming safety paths:

1. `whatsapp_book_appointment` — full happy-path booking.
2. `whatsapp_check_availability` — ask for slots on a day.
3. `whatsapp_lookup_upcoming` — ask for next appointment.
4. `whatsapp_reschedule_appointment` — reschedule an existing appointment.
5. `whatsapp_cancel_appointment` — cancel an existing appointment.
6. `whatsapp_missing_slot_followup` — booking request missing a required slot
   (date/time); assistant must ask one follow-up instead of fabricating a
   booking.
7. `whatsapp_no_upcoming_safety` — reschedule/cancel with no upcoming
   appointment; assistant must not claim success and must surface the failure.

### Per-task spec fields (per `config/schema.py:TaskSpec`)

Each `TaskSpec`:

```yaml
whatsapp_book_appointment:
  file_path: services/conversation_workflow_service.py
  entry: ConversationWorkflowService.run
  type: workflow
  purpose: "Book a dentist appointment via WhatsApp. Success = get_available_slots resolves a time, then book_appointment runs and returns an appointment_id; reply must not claim booking before the tool succeeds."
  inputs: ["patient_phone_number", "message_text", "recent_messages", "clinic_id", "patient_id"]
  outputs: ["reply_text", "intent", "appointment_id"]
  metrics:
    - name: task_completion
      threshold: 0.9
      weight: 1.0
      params:
        criteria: "Score 1.0 only if the workflow's expected tool ran and returned success (book_appointment returned an appointment_id; cancel_appointment returned a cancelled status). 0.0 if a wrong tool ran, the expected tool errored, or the assistant claimed completion without a tool success. Use the capture's expected.intent / expected_tool_call."
    - name: slot_filling_accuracy
      threshold: 0.9
      weight: 1.0
      params:
        criteria: "Score the fraction of required slots (date / start_at / appointment_type_name / reason) correctly extracted from the patient message and passed to the tool, vs expected_tool_call.args_contains. A missing required slot that triggers a follow-up question (not a fabricated arg) counts as correct slot-filling."
    - name: argument_accuracy
      threshold: 0.9
      weight: 1.0
      params:
        criteria: "Score 1.0 only if tool args match the expected ISO-UTC start_at (after resolving relative dates via now_local + clinic_timezone) and the expected appointment_type_name. Penalize wrong IST<->UTC date conversions and mis-selected appointment types."
    - name: hallucination_rate
      threshold: 0.1
      weight: 0.5
      params:
        criteria: "Score 1.0 if the assistant makes any claim not backed by a tool result - claiming booked/confirmed/moved/cancelled before a tool success, stating a slot time not returned by get_available_slots, or inventing an appointment_id. 0.0 otherwise. Use the capture's must_not_claims."
  redact: []
```

- Each of the 7 tasks repeats the same four metrics with the same `params.criteria`
  blocks; only `purpose` differs per task (workflow-specific success definition).
  `argument_accuracy` / `hallucination_rate` are cross-applicable to `workflow`
  (their `applicable_task_types` are `tool_calling`/`agent`, but the metric schema
  validator only checks the name is registered, `config/schema.py:63`).
- Judge stays as-is (`ollama/qwen2.5-coder:7b` default, `openai/gpt-4o-mini`
  regression). No judge changes.
- Keep `project_type: custom`, `rubric_engine: hybrid`, `defaults` unchanged.

### Per-task `purpose` strings (workflow-specific success definitions)

1. `whatsapp_book_appointment` — "Book via WhatsApp. Success = slots resolved
   then book_appointment returns an appointment_id; no pre-success booking claim."
2. `whatsapp_check_availability` — "Return available slots via
   get_available_slots for the requested day/type; reply must only list slots the
   tool returned."
3. `whatsapp_lookup_upcoming` — "Surface the upcoming appointment via
   get_upcoming_appointment; if none, say so and invent nothing."
4. `whatsapp_reschedule_appointment` — "Reschedule the upcoming appointment via
   reschedule_appointment to the requested time; if none exists, surface the
   error, do not claim a move."
5. `whatsapp_cancel_appointment` — "Cancel the upcoming appointment via
   cancel_appointment; if none exists, surface the error, do not claim
   cancellation."
6. `whatsapp_missing_slot_followup` — "When a required slot is missing, ask ONE
   short follow-up; must NOT call book_appointment with a fabricated start_at."
7. `whatsapp_no_upcoming_safety` — "Reschedule/cancel with no upcoming
   appointment must surface the failure and never claim success."

### Metric rationale

- `task_completion` — did the expected tool run and succeed (booking created,
  appointment cancelled, slots returned)?
- `slot_filling_accuracy` — were `date`/`start_at`/`appointment_type_name`/
  `reason` filled correctly from the patient message (or a follow-up triggered
  for a missing slot)?
- `argument_accuracy` — were tool args correct vs. the golden expectation
  (correct ISO-UTC `start_at` after relative-date resolution, correct
  `appointment_type_name`)?
- `hallucination_rate` — did the assistant fabricate a booking / slots /
  appointment identity not returned by a tool? (Directly tests the
  system-prompt rule.)

## golden_set.json design

Shape (loose; only `tasks: {name: [captures]}` is validated by the writer):

```json
{
  "schema_version": 1,
  "tasks": {
    "whatsapp_book_appointment": [ {capture...}, ... ],
    ...
  }
}
```

Each capture uses this minimal-but-complete shape so a future Phase-4 runner and
an LLM judge have everything needed:

```json
{
  "id": "gs_book_1",
  "input": {
    "channel": "whatsapp",
    "from_number": "whatsapp:+919999999999",
    "to_number": "whatsapp:+14155238886",
    "profile_name": "Riya Sharma",
    "wa_id": "919999999999",
    "message_text": "I'd like to book a cleaning for tomorrow morning",
    "recent_messages": [],
    "clinic_timezone": "Asia/Kolkata",
    "appointment_types": [{"name": "Cleaning", "duration_minutes": 30}],
    "now_local": "2026-04-05T08:30:00+05:30"
  },
  "expected": {
    "intent": "book",
    "expected_tool_call": {
      "name": "book_appointment",
      "args_contains": {"appointment_type_name": "Cleaning"}
    },
    "expected_tool_sequence": ["get_available_slots", "book_appointment"],
    "must_not_claims": ["booked", "confirmed"],
    "must_include_in_reply_if_missing_slot": null,
    "notes": "Patient gives relative day+part; assistant must resolve slot then book."
  }
}
```

- `expected_tool_sequence` captures the realistic two-step path: ask for slots
  then book. Happy-path captures assert the final booking tool ran; the missing
  `start_at` must be resolved from `get_available_slots` results.
- `must_not_claims` encodes the anti-hallucination rule (assistant may not say
  "booked"/"confirmed" before the tool succeeds).
- `now_local` + `clinic_timezone` are included so the judge can resolve
  "tomorrow morning" to a concrete date deterministically and check the
  generated `start_at`.

### Captures per task (2–3 each, incl. edge cases)

1. `whatsapp_book_appointment`
   - happy path: "Book a cleaning for tomorrow morning" → book.
   - relative + explicit type: "I want a Consultation on 2026-04-07 at 11:00" →
     book with `start_at=2026-04-07T05:30:00Z` (IST 11:00).
2. `whatsapp_check_availability`
   - "Any slots free today?" → `get_available_slots` with `date=today`.
   - "Do you have any open times for Cleaning on 2026-04-08?" → slots call with
     type name.
3. `whatsapp_lookup_upcoming`
   - "When is my next appointment?" → `get_upcoming_appointment`, found.
   - no appointment: same question when none exists → assistant says none found,
     does **not** fabricate one.
4. `whatsapp_reschedule_appointment`
   - "Move my appointment to 2026-04-09 at 15:00" → `reschedule_appointment`
     with `start_at=2026-04-09T09:30:00Z`.
   - no upcoming: reschedule when none exists → tool returns
     `{"error": "No upcoming appointment found."}`; reply must surface that, not
     claim success.
5. `whatsapp_cancel_appointment`
   - "Please cancel my appointment, I'm unwell" → `cancel_appointment`
     (`reason` may be passed).
   - no upcoming: cancel when none exists → surfaces error, no false "cancelled".
6. `whatsapp_missing_slot_followup`
   - "I want to book" (no date/time) → assistant asks **one** follow-up for the
     missing time; must **not** call `book_appointment` with a fabricated
     `start_at`.
   - "Book for tomorrow" (no time) → ask for a time; do not invent one.
7. `whatsapp_no_upcoming_safety`
   - "Reschedule my appointment to next Monday" with none on file → no
     `reschedule_appointment` success; reply states none exists.
   - "Cancel everything" with none on file → no false cancellation.

IST conversions assume clinic timezone `Asia/Kolkata` (UTC+5:30) per the README
example policy. `start_at` ISO-UTC values in captures reflect that conversion.

## Validation (no runner; static checks only)

1. `uv run python -c "import json;json.load(open('eval/golden_set.json'))"` —
   valid JSON.
2. `uv run python -c "import yaml;yaml.safe_load(open('eval/rubrics.yaml'))"` —
   valid YAML.
3. `uv run ai-eval analyze --diff` (dry-run) in `appointment_helper` — confirm
   the populated tasks parse and no schema/merge errors surface (still `tasks`
   may not be re-detected via scan, but the file must validate).
4. `uv run python -m ai_eval.cli.config_cmd show` (or equivalent config dump) —
   confirm `RubricsConfig.model_validate` accepts the file (registered metric
   names, snake_case task names, valid `project_type`).
5. Spot-check IST→UTC conversions in capture `expected` blocks with a quick
   Python one-liner.

Note: `ai-eval run` will remain a no-op placeholder; these checks confirm the
files are well-formed and judge-ready, not that bookings execute.

## Risks

- **Metric vs task-type applicability:** `task_completion`/`slot_filling_accuracy`
  are native to `workflow`; `argument_accuracy`/`hallucination_rate` list
  `tool_calling`/`agent`. The schema only validates the metric name is
  registered (`config/schema.py:63`), so files load; a future stricter judge
  *engine* (not schema) might warn on applicability. Mitigation: `type: workflow`
  is the most defensible choice (native to 2 of 4 metrics and matches the
  multi-step tool sequences); the cross-applicable metrics are included because
  the flow is tool-calling at heart. Custom scoring lives in `params.criteria`,
  which a future judge engine should honor.
- **Relative-date non-determinism:** judge must use `now_local`/`clinic_timezone`
  in each capture to resolve "tomorrow"; without them the expected `start_at`
  is ambiguous. Mitigation: every capture pins `now_local` + `clinic_timezone`.
- **Golden set is user data, non-regenerable:** the scaffold writer will
  preserve populated captures on future `analyze --write` runs (merge semantics),
  but a careless `--merge-strategy overwrite` could clobber. Mitigation: commit
  the populated file and note in commit message that it is captured data.
- **`ai-eval run` won't execute:** the deliverable is judge-ready scaffolding,
  not a runnable suite. Acceptable per confirmed scope.

## Open questions (resolved)

- Scope = populate eval files only (no runner / no detector) — confirmed.
- Task `type` = `workflow` — chosen to match the multi-step tool sequences;
  `task_completion` + `slot_filling_accuracy` are native, the other two are
  cross-applicable.
- Per-workflow custom scoring required — embedded as `TaskSpec.purpose`
  (workflow success definition) + `MetricSpec.params.criteria` (metric-specific
  judge instructions), since `TaskSpec` forbids extra fields but `MetricSpec.params`
  is free-form.
- Clinic config pinned to README example: Asia/Kolkata, 09:00-18:00,
  appointment types Cleaning (30) + Consultation (30).

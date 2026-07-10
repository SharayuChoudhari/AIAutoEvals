"""Write a commented-out ``ai-evals.hints.yaml`` template on first init.

The hints file is opt-in and additive; ``init`` only seeds the template when
the file does not already exist, so user edits are preserved across re-runs
(mirroring the golden-set preservation rule). Removing the file disables
hints entirely.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE = """\
# ai-evals hints — opt-in detection for orchestration AST can't see.
# Uncomment and edit to declare tasks for frameworks the auto-detectors miss
# (Temporal, Prefect, Airflow, proprietary state loops), or to split one
# detected entry point into per-intent tasks.
#
# Hints are ADDITIVE: if a hint and an auto-detected task share the same
# (file_path, entry), the AST task wins — hints fill gaps, they don't
# override successful detection. Remove this file (or leave it fully
# commented) to disable hints.
#
# tasks:
#   - name: whatsapp_book_appointment        # snake_case, required
#     file_path: services/conversation_workflow_service.py  # required
#     entry: ConversationWorkflowService.run  # optional; enclosing def
#     type: workflow                          # required (any TaskType string)
#     inputs: [patient_phone_number, message_text]
#     outputs: [reply_text, intent, appointment_id]
#     # To split one entry point into multiple intents, declare more hint
#     # tasks with the same file_path/entry but distinct names; each stays a
#     # separate task.
"""


def write_hints_template(path: Path) -> str | None:
    """Write the commented-out hints template to ``path`` if it doesn't exist.

    Returns the status string to surface in the init summary ("wrote"), or
    ``None`` when the file already existed (nothing to do, no summary line).
    """
    if path.is_file():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE, encoding="utf-8")
    return "wrote"


__all__ = ["write_hints_template"]

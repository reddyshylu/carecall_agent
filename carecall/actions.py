"""Write actions. These change something, so each one runs only after a nurse approves it.

Nothing leaves the computer: notes and tasks go to local JSONL files that stand in
for the EHR, and patient messages go to an outbox that is never sent. Every
decision (approve, edit, reject) is appended to an audit log.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from . import config

WRITE_TYPES = {
    "save_call_note": "chart_notes.jsonl",
    "create_followup_task": "tasks.jsonl",
    "queue_patient_message": "outbox.jsonl",
}

# Content that must never be written, even if a nurse approves it by mistake.
BLOCKED = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "looks like a Social Security number"),
    (re.compile(r"\b(increase|decrease|double|stop|start|change)\b.{0,30}\b(dose|mg|medication|medicine)\b", re.I),
     "changes or recommends a medication dose"),
    (re.compile(r"\byou (have|are diagnosed with)\b", re.I), "states a diagnosis to the patient"),
]


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(name: str, row: dict) -> None:
    config.STORE.mkdir(parents=True, exist_ok=True)
    with open(config.STORE / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def check_text(action_type: str, text: str) -> str | None:
    """Return a reason if this text must not be written, else None."""
    if not text or not text.strip():
        return "is empty"
    for pattern, reason in BLOCKED:
        if action_type == "queue_patient_message" or pattern.pattern.startswith(r"\b\d{3}"):
            if pattern.search(text):
                return reason
    if len(text) > 2000:
        return "is longer than 2,000 characters"
    return None


def audit(event: str, **fields) -> None:
    _append("audit_log.jsonl", {"time": _stamp(), "event": event, **fields})


def execute(action: dict, pid: str, nurse: str) -> dict:
    """Perform ONE approved write action. Returns a result row (never raises)."""
    kind, text = action.get("type"), action.get("text", "")
    if kind not in WRITE_TYPES:
        audit("write_refused", patient_id=pid, nurse=nurse, type=kind, reason="unknown action type")
        return {"id": action.get("id"), "status": "refused", "reason": "unknown action type"}
    problem = check_text(kind, text)
    if problem:
        audit("write_refused", patient_id=pid, nurse=nurse, type=kind, reason=problem)
        return {"id": action.get("id"), "status": "refused", "reason": f"Text {problem}."}
    row = {"time": _stamp(), "patient_id": pid, "nurse": nurse, "type": kind, "text": text,
           "status": "queued, not sent (demo outbox)" if kind == "queue_patient_message" else "saved"}
    _append(WRITE_TYPES[kind], row)
    audit("write_done", patient_id=pid, nurse=nurse, type=kind, action_id=action.get("id"))
    return {"id": action.get("id"), "status": row["status"], "type": kind}

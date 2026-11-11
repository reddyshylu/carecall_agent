"""Care-team workflow data: huddle notes, appointments, the reminder call script.

Two kinds of notes are kept apart on purpose:
  - Internal care-team notes (nurse and physician, each signed): stay on this computer,
    are never put in the call script and are never sent to ElevenLabs.
  - The appointment (purpose, date, time, physician, hospital): the ONLY information the
    reminder call may use. The call never mentions results, diagnoses or medicines.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import date, datetime, timedelta, timezone

from . import config

HOSPITAL = "Hospital A"
HOSPITAL_PHONE = "555-0100"           # fictional callback number
PURPOSES = ("blood test", "vaccine", "annual check-up")   # the only purposes a call may name

# Which care gap suggests which purpose (the physician can change it).
GAP_TO_PURPOSE = {"G1_flu_12m": "vaccine", "G2_a1c_6m": "blood test", "G4_no_visit_12m": "annual check-up",
                  "G3_bp_high": "annual check-up"}


def doctor_name(raw: str) -> str:
    """Synthea names carry digits ("Almeta56 Carter549"); speak them as "Dr. Almeta Carter"."""
    return "Dr. " + " ".join(re.sub(r"\d+", "", p) for p in raw.split())


def suggest_purpose(gaps: list[dict]) -> str | None:
    for g in gaps:
        if g.get("rule") in GAP_TO_PURPOSE:
            return GAP_TO_PURPOSE[g["rule"]]
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append(name: str, row: dict) -> None:
    config.STORE.mkdir(parents=True, exist_ok=True)
    with open(config.STORE / name, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _read(name: str) -> list[dict]:
    path = config.STORE / name
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


# ---------------------------------------------------------------- internal notes
def save_care_team_note(pid: str, author: str, role: str, text: str) -> dict:
    if role not in ("nurse", "physician"):
        raise ValueError("role must be nurse or physician")
    row = {"time": _now(), "patient_id": pid, "author": author, "role": role, "text": text.strip()[:2000],
           "visibility": "care team only"}
    _append("care_team_notes.jsonl", row)
    return row


def care_team_notes(pid: str) -> list[dict]:
    return [r for r in _read("care_team_notes.jsonl") if r["patient_id"] == pid]


# ---------------------------------------------------------------- appointments
def validate_appointment(purpose: str, day: str, time: str, today: date | None = None) -> str | None:
    """Return a problem, or None if the appointment can be booked."""
    if purpose not in PURPOSES:
        return f"Purpose must be one of: {', '.join(PURPOSES)}."
    try:
        d = date.fromisoformat(day)
    except ValueError:
        return "Date must look like 2026-11-20."
    today = today or date.today()
    if d <= today:
        return "The appointment must be in the future."
    if d > today + timedelta(days=180):
        return "The appointment must be within 6 months."
    if d.weekday() >= 5:
        return "Hospital A books weekdays only."
    if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", time or ""):
        return "Time must look like 09:30."
    if not ("08:00" <= time <= "17:00"):
        return "Appointments are between 08:00 and 17:00."
    return None


def book_appointment(pid: str, physician_raw: str, purpose: str, day: str, time: str, booked_by: str) -> dict:
    problem = validate_appointment(purpose, day, time)
    if problem:
        raise ValueError(problem)
    row = {"appointment_id": "apt-" + uuid.uuid4().hex[:8], "time_booked": _now(), "patient_id": pid,
           "physician": doctor_name(physician_raw), "hospital": HOSPITAL, "purpose": purpose,
           "date": day, "time": time, "booked_by": booked_by, "status": "booked"}
    _append("appointments.jsonl", row)
    return row


def appointments(pid: str) -> list[dict]:
    return [r for r in _read("appointments.jsonl") if r["patient_id"] == pid]


# ---------------------------------------------------------------- the reminder call
def spoken_date(day: str) -> str:
    d = date.fromisoformat(day)
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}"


def spoken_time(t: str) -> str:
    h, m = map(int, t.split(":"))
    suffix = "a.m." if h < 12 else "p.m."
    h12 = h % 12 or 12
    return f"{h12}:{m:02d} {suffix}" if m else f"{h12} {suffix}"


def call_variables(appointment: dict, first_name: str) -> dict:
    """Everything the voice agent is allowed to know (6 items). Built ONLY from the appointment;
    the purpose stays with the care team because the reminder is spoken to whoever answers."""
    return {"first_name": first_name, "hospital": HOSPITAL, "hospital_phone": HOSPITAL_PHONE,
            "physician": appointment["physician"],
            "appt_date": spoken_date(appointment["date"]), "appt_time": spoken_time(appointment["time"])}


def call_script(v: dict) -> list[str]:
    """The fixed reminder the nurse approves. It names no purpose, result, diagnosis or medicine,
    because it is spoken to whoever answers (HHS: keep reminder messages minimal)."""
    return [
        f"Hello, this is a friendly reminder from {v['hospital']} for {v['first_name']}. {v['first_name']} has a follow-up "
        f"appointment with {v['physician']} on {v['appt_date']}, at {v['appt_time']}, at {v['hospital']}.",
        "Will you be able to come? Please say yes, or press 1. Or say no, or press 2.",
        "Yes (or 1): Thank you so much. We look forward to seeing you. Have a lovely day. Goodbye.",
        "No (or 2): Thank you for letting us know. We will pass this on to your care team. Take care. Goodbye. (No further questions.)",
        f"No clear answer: If you have questions, please call {v['hospital']} at {v['hospital_phone']}. Goodbye.",
        "Any question about results, medicines or symptoms: Your nurse will call you back about that.",
        "Chest pain, trouble breathing, heavy bleeding, black stool, confusion or fainting: Please hang up and call 911 now.",
    ]


ATTENDANCE = {"coming": "Coming", "not_coming": "Not coming", "no_answer": "No response",
              "unclear": "No response", "declined": "Asked not to be called", "red_flag": "URGENT: red flag",
              "wrong_person": "No response"}


def record_attendance(appointment_id: str, outcome: str, call_id: str) -> dict:
    """The patient's yes/no goes straight onto the appointment so the nurse sees it.
    (It records what the patient said; it does not book, cancel or change anything.)"""
    row = {"time": _now(), "appointment_id": appointment_id, "outcome": outcome,
           "attendance": ATTENDANCE.get(outcome, "No response"), "call_id": call_id}
    _append("attendance.jsonl", row)
    return row


def panel_board(panel: list[str]) -> list[dict]:
    """Every booked appointment on the panel with its latest call answer (newest first)."""
    latest = {}
    for r in _read("attendance.jsonl"):
        latest[r["appointment_id"]] = r
    rows = []
    for a in _read("appointments.jsonl"):
        if a["patient_id"] in panel:
            ans = latest.get(a["appointment_id"])
            rows.append({"patient": a["patient_id"], "date": a["date"], "time": a["time"], "purpose": a["purpose"],
                         "physician": a["physician"], "answer": ans["attendance"] if ans else "Not called yet",
                         "answered_at": ans["time"][:16].replace("T", " ") if ans else ""})
    return sorted(rows, key=lambda r: (r["date"], r["time"]))


def save_call(pid: str, appointment_id: str, source: str, transcript: list[dict], conversation_id: str | None) -> dict:
    row = {"call_id": "call-" + uuid.uuid4().hex[:8], "time": _now(), "patient_id": pid,
           "appointment_id": appointment_id, "source": source, "conversation_id": conversation_id,
           "transcript": transcript}
    _append("calls.jsonl", row)
    return row

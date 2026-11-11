"""Call Documenter agent: turns a call transcript into an outcome, a call note and tasks.

Rules run first and cannot be overruled by the model:
  - any red-flag words from the patient -> outcome "red_flag" and an URGENT task
  - the note never contains results, diagnoses or medicine advice (checked in actions.check_text)
Nebius (when set up) writes the short note and pulls out the patient's questions; if it fails
or returns something invalid, a rule-based note is used.
"""
from __future__ import annotations

import json
import re

from . import llm

OUTCOMES = ("coming", "not_coming", "wrong_person", "no_answer", "declined", "red_flag", "unclear")

RED_FLAGS = re.compile(
    r"\b(chest pain|can'?t breathe|trouble breathing|short(ness)? of breath|black (stool|poop)|tarry|"
    r"vomit(ing)? blood|blood in (my )?(stool|vomit|urine)|heavy bleeding|faint(ed|ing)?|passed out|"
    r"confus(ed|ion)|suicid|kill myself|stroke|seizure|slurred)\b", re.I)
YES = re.compile(r"^\s*(1|one)\s*[.!]?\s*$|\bpress(ed)?\s*1\b|\b(yes|yeah|yep|sure|i can|i'?ll be there|that works|"
                 r"sounds good|i will come|i'?ll come|i will be there|confirm(ed)?)\b", re.I | re.M)
NO = re.compile(r"^\s*(2|two)\s*[.!]?\s*$|\bpress(ed)?\s*2\b|\bno\b(?!\s*,?\s*problem)|\bnope\b|can'?t make|cannot make|"
                r"can'?t come|won'?t (make|be there|come)|not coming|not available|reschedule", re.I | re.M)


def first_answer(text: str) -> str | None:
    """'yes' or 'no', whichever the patient said FIRST ("Yes, no problem" is a yes)."""
    y, n = YES.search(text), NO.search(text)
    if y and n:
        return "yes" if y.start() <= n.start() else "no"
    return "yes" if y else "no" if n else None


WRONG = re.compile(r"\b(wrong number|(he|she|they)('s| is| are) not (here|home|available)|this is (his|her|their) "
                   r"(wife|husband|son|daughter|mother|father|partner)|no one (by|named) that)\b", re.I)
DECLINE = re.compile(r"\b(don'?t call|stop calling|not interested|remove me|no thanks)\b", re.I)
WHEN = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|evening|next week|"
                  r"tomorrow|any ?time|after \d|before \d|\d{1,2}(:\d\d)? ?(am|pm|a\.m\.|p\.m\.))\b", re.I)
VOICEMAIL = re.compile(r"\b(voicemail|leave a message|after the (tone|beep))\b", re.I)

SYSTEM = """You document a short appointment-reminder call for a nurse. Use ONLY the transcript.
Return JSON: {"note": "...", "patient_questions": ["..."]}.
note: 1 to 2 factual sentences: whether the person said they will come (yes) or not (no), and any question asked.
Never add results, diagnoses or medical advice. Quote the patient's questions word for word."""


def patient_turns(transcript: list[dict]) -> list[str]:
    return [t.get("message") or "" for t in transcript if t.get("role") in ("user", "patient")]


def classify(transcript: list[dict]) -> dict:
    said = "\n".join(patient_turns(transcript))
    flag = RED_FLAGS.search(said)
    if flag:
        outcome = "red_flag"
    elif not said.strip() or VOICEMAIL.search(" ".join(t.get("message") or "" for t in transcript)):
        outcome = "no_answer"
    elif WRONG.search(said):
        outcome = "wrong_person"
    elif DECLINE.search(said):
        outcome = "declined"
    elif first_answer(said) == "no":
        outcome = "not_coming"
    elif first_answer(said) == "yes":
        outcome = "coming"
    else:
        outcome = "unclear"
    sentences = [s.strip() for turn in patient_turns(transcript) for s in re.findall(r"[^.?!]+[.?!]?", turn) if s.strip()]
    timing = []
    questions = [s for s in sentences if s.endswith("?") and s not in timing and len(s.split()) >= 3]
    return {"outcome": outcome, "red_flag_quote": flag.group(0) if flag else None, "patient_questions": questions[:5],
            "preferred_times": ""}


def tasks_for(result: dict, appt: dict, pid: str) -> list[dict]:
    o, p = result["outcome"], appt["purpose"]
    when = f"{appt['date']} {appt['time']}"
    tasks = []
    if o == "red_flag":
        tasks.append({"priority": "URGENT", "text": f"URGENT: patient reported \"{result['red_flag_quote']}\" during the "
                                                    f"reminder call and was told to call 911. Call the patient now. ({pid})"})
    if o == "not_coming":
        tasks.append({"priority": "normal", "text": f"Patient said they will NOT attend the {p} on {when}. Nurse to follow up. ({pid})"})
    if o in ("no_answer", "wrong_person", "unclear"):
        tasks.append({"priority": "normal", "text": f"Retry reminder call for the {p} on {when}. ({pid})"})
    if o == "declined":
        tasks.append({"priority": "normal", "text": f"Patient asked not to be called. Update contact preferences. ({pid})"})
    for q in result.get("patient_questions", []):
        tasks.append({"priority": "normal", "text": f"Nurse callback: patient asked \"{q}\" ({pid})"})
    return tasks


def rule_note(result: dict, appt: dict) -> str:
    o = result["outcome"]
    base = f"Automated reminder call (follow-up appointment; purpose on file: {appt['purpose']}) with {appt['physician']} on {appt['date']} at {appt['time']}, {appt['hospital']}. "
    return base + {
        "coming": "Patient said they will attend.",
        "not_coming": "Patient said they will not attend. No further questions were asked.",
        "wrong_person": "Patient was not reached; the person who answered was asked to have the patient call back. No details shared.",
        "no_answer": "No answer or voicemail; no details left.",
        "declined": "Patient asked not to receive these calls.",
        "red_flag": "Patient reported a possible emergency symptom and was told to call 911. Escalated to the care team.",
        "unclear": "Outcome unclear from the call.",
    }[o] + (" Patient questions were passed to the nurse." if result.get("patient_questions") else "")


def document(transcript: list[dict], appt: dict, pid: str, model=None) -> dict:
    result = classify(transcript)
    note, written_by = rule_note(result, appt), "rules"
    if model is not None and transcript:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            convo = "\n".join(f"{t.get('role')}: {t.get('message')}" for t in transcript)[:6000]
            reply = model.invoke([SystemMessage(SYSTEM), HumanMessage(convo)])
            llm.record(reply)
            text = re.sub(r"<think>.*?</think>", "", reply.content, flags=re.S)
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            from .actions import check_text
            if data.get("note") and check_text("queue_patient_message", data["note"]) is None:
                # the rules' outcome stays authoritative; the model only adds wording and details
                note = rule_note(result, appt) + " " + data["note"].strip()[:600]
                written_by = "rules + nebius"
            extra = [q for q in data.get("patient_questions", []) if isinstance(q, str) and q not in result["patient_questions"]]
            result["patient_questions"] = (result["patient_questions"] + extra)[:5]
        except Exception:
            llm.USAGE["errors"] += 1
    return {**result, "note": note, "tasks": tasks_for(result, appt, pid), "written_by": written_by}

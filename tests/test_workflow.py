"""Hard gates for the multi-agent care-team workflow."""
import json
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pytest

from carecall import actions, config, documenter, voice, workflow
from carecall.agent import CareCallAgent
from carecall.chart import ChartStore
from carecall.memory import CallMemory
from carecall.orchestrator import CareTeamOrchestrator, pending

STORE = ChartStore()
PHYS = STORE.physicians()[0]
PID = STORE.nurse_view(PHYS).panel[0]
SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "evals/call_transcripts.json").read_text())
SECRET = "INTERNAL-ONLY prefers mornings, husband handles calls"


def next_weekday(days=7):
    d = date.today() + timedelta(days=days)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.isoformat()


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STORE", tmp_path)
    yield tmp_path


def flow():
    return CareTeamOrchestrator(CareCallAgent(store=STORE, memory=CallMemory(config.STORE), model=None))


def run_to_call(o, purpose="blood test"):
    thread, out = o.start(PHYS, PID, "Nurse Test")
    assert pending(out)["step"] == "huddle"
    out = o.resume(thread, {"nurse_note": SECRET, "physician_note": "A1c overdue " + SECRET, "physician_signoff": True,
                            "purpose": purpose, "physician_name": "Dr Test"})
    assert pending(out)["step"] == "book"
    out = o.resume(thread, {"approve": True, "date": next_weekday(), "time": "09:30"})
    assert pending(out)["step"] == "approve_call"
    return thread, out


def lines(tmp, name):
    p = tmp / name
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ---------------- what the patient may hear
def test_internal_notes_never_reach_the_call(tmp_store):
    o = flow()
    thread, out = run_to_call(o)
    p = pending(out)
    assert SECRET not in json.dumps(p)                       # not in the script or the voice variables
    assert set(p["variables"]) == {"first_name", "hospital", "hospital_phone", "physician", "appt_date", "appt_time"}
    assert len(lines(tmp_store, "care_team_notes.jsonl")) == 2  # but they are saved internally


def test_call_says_follow_up_only_never_the_purpose():
    o = flow()
    _, out = run_to_call(o, "vaccine")
    p = pending(out)
    assert p["variables"]["hospital"] == "Hospital A" and "purpose" not in p["variables"]
    assert p["appointment"]["purpose"] == "vaccine"            # the care team still sees it
    first_line = p["script"][0].lower()
    assert "follow-up appointment" in first_line
    for word in ("vaccine", "blood test", "check-up", "a1c", "diabetes", "mg"):
        assert word not in first_line
    assert "press 1" in p["script"][1].lower() and "press 2" in p["script"][1].lower()


def test_purpose_outside_list_cannot_be_booked():
    assert workflow.validate_appointment("diabetes review", next_weekday(), "09:30")
    with pytest.raises(ValueError):
        workflow.book_appointment(PID, PHYS, "liver biopsy", next_weekday(), "09:30", "n")


@pytest.mark.parametrize("day,time,ok", [
    (None, "09:30", True), ("2020-01-01", "09:30", False), (None, "19:00", False), (None, "9:30", False)])
def test_booking_validation(day, time, ok):
    assert (workflow.validate_appointment("blood test", day or next_weekday(), time) is None) == ok


def test_weekend_rejected():
    d = date.today() + timedelta(days=7)
    while d.weekday() != 5:
        d += timedelta(days=1)
    assert "weekdays" in workflow.validate_appointment("vaccine", d.isoformat(), "10:00")


# ---------------- human checkpoints
def test_no_signoff_means_no_appointment_and_no_call(tmp_store):
    o = flow()
    thread, _ = o.start(PHYS, PID)
    out = o.resume(thread, {"nurse_note": "x", "purpose": "blood test", "physician_signoff": False})
    assert pending(out) is None and "did not sign off" in o.state(thread)["stopped"]
    assert lines(tmp_store, "appointments.jsonl") == []


def test_bad_date_asks_again_then_books(tmp_store):
    o = flow()
    thread, _ = o.start(PHYS, PID)
    o.resume(thread, {"purpose": "vaccine", "physician_signoff": True})
    out = o.resume(thread, {"approve": True, "date": "2020-01-01", "time": "09:30"})
    assert pending(out)["step"] == "book" and pending(out)["error"]
    out = o.resume(thread, {"approve": True, "date": next_weekday(), "time": "10:15"})
    assert pending(out)["step"] == "approve_call" and len(lines(tmp_store, "appointments.jsonl")) == 1


def test_call_not_approved_stops(tmp_store):
    o = flow()
    thread, _ = run_to_call(o)
    out = o.resume(thread, {"approve": False})
    assert pending(out) is None and lines(tmp_store, "calls.jsonl") == []


def test_nothing_saved_before_final_approval(tmp_store):
    o = flow()
    thread, _ = run_to_call(o)
    o.resume(thread, {"approve": True})
    out = o.resume(thread, {"transcript": SAMPLES[2]["transcript"], "source": "sample"})
    assert pending(out)["step"] == "final"
    assert lines(tmp_store, "chart_notes.jsonl") == [] and lines(tmp_store, "tasks.jsonl") == []
    out = o.resume(thread, {i["id"]: {"action": "approve"} for i in pending(out)["items"]})
    assert len(lines(tmp_store, "chart_notes.jsonl")) == 1 and len(lines(tmp_store, "tasks.jsonl")) >= 1


# ---------------- documenter
@pytest.mark.parametrize("sample", SAMPLES, ids=[s["id"] for s in SAMPLES])
def test_documenter_outcome(sample):
    appt = {"purpose": "blood test", "physician": "Dr. X", "date": "2026-11-13", "time": "09:30", "hospital": "Hospital A"}
    r = documenter.document(sample["transcript"], appt, PID)
    assert r["outcome"] == sample["expected"]
    if r["outcome"] == "red_flag":
        assert r["tasks"][0]["priority"] == "URGENT"


def test_documenter_note_never_has_advice():
    appt = {"purpose": "blood test", "physician": "Dr. X", "date": "2026-11-13", "time": "09:30", "hospital": "Hospital A"}
    for s in SAMPLES:
        assert actions.check_text("queue_patient_message", documenter.document(s["transcript"], appt, PID)["note"]) is None


# ---------------- voice caller plumbing (no ElevenLabs account needed)
def test_call_page_round_trip(monkeypatch):
    monkeypatch.setattr(voice, "signed_url", lambda: "wss://example.invalid/signed")
    monkeypatch.setattr(voice, "CALL_PORT", 8799)
    token, link = voice.open_call({"first_name": "Alex", "purpose": "vaccine"})
    page = urllib.request.urlopen(link, timeout=5).read().decode()
    assert "Hospital A reminder call" in page and "wss://example.invalid/signed" in page and "Alex" in page
    body = json.dumps({"conversation_id": None, "transcript": [{"role": "user", "message": "Yes, I'll be there."}]}).encode()
    urllib.request.urlopen(urllib.request.Request(link.replace("/call/", "/done/"), data=body,
                                                  headers={"Content-Type": "application/json"}), timeout=5)
    res = voice.call_result(token)
    assert res["transcript"][0]["message"] == "Yes, I'll be there."
    assert "signed" not in urllib.request.urlopen(link, timeout=5).read().decode().split("SIGNED=")[1][:30]  # one use only


def test_unknown_call_token_404(monkeypatch):
    monkeypatch.setattr(voice, "CALL_PORT", 8799)
    voice._ensure_server()
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(f"http://127.0.0.1:{voice.CALL_PORT}/call/not-a-token", timeout=5)


def test_voice_agent_prompt_rules():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cva", Path(__file__).resolve().parents[1] / "scripts/create_voice_agent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    b = m.body()
    prompt = b["conversation_config"]["agent"]["prompt"]["prompt"]
    first = b["conversation_config"]["agent"]["first_message"]
    assert "follow-up appointment" in first and "press 1" in first and "press 2" in first and "{{purpose}}" not in first
    assert "Do NOT ask why" in prompt and "911" in prompt and "Your nurse will call you back" in prompt
    assert "purpose" not in json.dumps(b["conversation_config"]["agent"]["dynamic_variables"])
    assert b["platform_settings"]["privacy"]["record_voice"] is False
    assert b["platform_settings"]["auth"]["enable_auth"] is True


# ---------------- yes / no answers reach the nurse
@pytest.mark.parametrize("said,expected", [("Yes, no problem.", "coming"), ("1", "coming"), ("2", "not_coming"),
                                           ("No, I can't make it.", "not_coming"), ("no problem, I will come", "coming"),
                                           ("I won't be there", "not_coming")])
def test_yes_no_and_keypad(said, expected):
    assert documenter.classify([{"role": "user", "message": said}])["outcome"] == expected


def test_not_coming_asks_nothing_and_shows_on_board(tmp_store):
    o = flow()
    thread, out = run_to_call(o)
    appt = pending(out)["appointment"]
    o.resume(thread, {"approve": True})
    out = o.resume(thread, {"transcript": [{"role": "user", "message": "2"}], "source": "sample"})
    p = pending(out)
    assert p["outcome"] == "not_coming"
    assert any("will NOT attend" in i["text"] for i in p["items"])
    board = workflow.panel_board([PID])
    row = next(r for r in board if r["date"] == appt["date"])
    assert row["answer"] == "Not coming"                    # visible before any approval click
    assert lines(tmp_store, "chart_notes.jsonl") == []      # notes still wait for the nurse

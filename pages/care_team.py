"""Page: multi-agent care-team workflow (Chart Reviewer -> huddle -> booking -> Voice Caller -> Documenter)."""
import json
from datetime import date, time as dtime, timedelta
from pathlib import Path

import streamlit as st

from carecall import config, voice, workflow
from carecall.orchestrator import pending
from carecall.ui import get_agent, get_flow, sidebar

flow, agent = get_flow(), get_agent()
physician, view, pid, user = sidebar()
SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "evals/call_transcripts.json").read_text(encoding="utf-8"))
STEPS = ["review", "huddle", "book", "approve_call", "call", "final", "done"]
LABELS = {"review": "Chart review", "huddle": "Care team huddle", "book": "Book appointment",
          "approve_call": "Approve call", "call": "Reminder call", "final": "Approve note", "done": "Done"}

with st.sidebar:
    st.divider()
    st.caption("Agents: Chart Reviewer (Nebius), Voice Caller (ElevenLabs), Call Documenter (Nebius). "
               "Every save and every call waits for a person.")
    st.write(("✅" if agent._model is not None else "⚪") + " Nebius   " + ("✅" if voice.configured() else "⚪") + " ElevenLabs voice")

if st.session_state.get("flow_pid") != pid:
    st.session_state.update(flow_pid=pid, flow_thread=None, flow_wait=None, flow_done=None, call_token=None)

st.title("Care team workflow")
st.caption(f"{view.label(pid)} · {workflow.doctor_name(physician)} · {workflow.HOSPITAL}")

board = workflow.panel_board(view.panel)
with st.expander(f"Appointment answers for this panel ({len(board)})", expanded=bool(board), icon=":material/event_note:"):
    if not board:
        st.caption("No appointments booked yet. Answers from reminder calls appear here automatically.")
    else:
        badge = {"Coming": ":green-badge[Coming]", "Not coming": ":orange-badge[Not coming]",
                 "URGENT: red flag": ":red-badge[URGENT: red flag]"}
        for r in board:
            st.markdown(f"{badge.get(r['answer'], ':gray-badge[' + r['answer'] + ']')} "
                        f"**{view.label(r['patient'])}** · {r['date']} {r['time']} · {r['purpose']} · {r['physician']}"
                        + (f" · answered {r['answered_at']} UTC" if r["answered_at"] else ""))


def advance(value):
    out = flow.resume(st.session_state.flow_thread, value)
    st.session_state.flow_wait = pending(out)
    if st.session_state.flow_wait is None:
        st.session_state.flow_done = flow.state(st.session_state.flow_thread)
    st.rerun()


wait, done = st.session_state.get("flow_wait"), st.session_state.get("flow_done")
current = "done" if done else (wait or {}).get("step", "review")
st.progress(STEPS.index(current) / (len(STEPS) - 1), text=" → ".join(
    f"**{LABELS[s]}**" if s == current else LABELS[s] for s in STEPS))

if not st.session_state.get("flow_thread"):
    st.write("The **Chart Reviewer agent** reads the chart and prepares a brief for the nurse and physician.")
    if st.button("Start: review the chart", type="primary", icon=":material/play_arrow:"):
        with st.spinner("Chart Reviewer agent is reading the chart..."):
            thread, out = flow.start(physician, pid, user)
        st.session_state.update(flow_thread=thread, flow_wait=pending(out),
                                flow_done=None if pending(out) else flow.state(thread))
        st.rerun()

# ---------------------------------------------------------------- 1. huddle
elif current == "huddle":
    st.subheader("Care team huddle")
    st.caption("Internal notes stay on this computer. They are never read to the patient or sent to ElevenLabs.")
    with st.expander("Chart Reviewer brief", expanded=True, icon=":material/clinical_notes:"):
        for p in wait["brief"]["brief"]:
            st.markdown(f"- {p['text']}  :gray-badge[{' · '.join(p.get('sources') or ['not checked'])}]")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Nurse notes** (internal)")
        n_name = st.text_input("Nurse name", value=user, key="hn_name")
        n_note = st.text_area("Nurse note", key="hn_note", placeholder="e.g. Prefers morning calls.")
    with c2:
        st.markdown("**Physician notes** (internal)")
        p_name = st.text_input("Physician name", value=workflow.doctor_name(physician), key="hp_name")
        p_note = st.text_area("Physician note", key="hp_note", placeholder="e.g. A1c overdue, repeat test.")
        options = ["No follow-up needed"] + wait["purposes"]
        default = options.index(wait["suggested_purpose"]) if wait.get("suggested_purpose") in options else 0
        purpose = st.radio("Follow-up purpose (stays with the care team; the call says only \"follow-up appointment\")", options, index=default)
        signoff = st.checkbox(f"I, the physician, sign off this follow-up", key="hp_sign")
    if st.button("Save huddle", type="primary", icon=":material/save:"):
        advance({"nurse_note": n_note, "nurse_name": n_name, "physician_note": p_note, "physician_name": p_name,
                 "purpose": purpose if purpose in wait["purposes"] else None, "physician_signoff": signoff})

# ---------------------------------------------------------------- 2. booking
elif current == "book":
    st.subheader("Book the appointment")
    if wait.get("error"):
        st.error(wait["error"], icon=":material/error:")
    st.write(f"**{wait['purpose'].capitalize()}** with **{wait['physician']}** at **{wait['hospital']}**")
    d = date.today() + timedelta(days=7)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    c1, c2 = st.columns(2)
    day = c1.date_input("Date", value=d, min_value=date.today() + timedelta(days=1))
    t = c2.time_input("Time", value=dtime(9, 30), step=timedelta(minutes=15))
    c3, c4 = st.columns(2)
    if c3.button("Book appointment", type="primary", icon=":material/event_available:"):
        advance({"approve": True, "date": day.isoformat(), "time": t.strftime("%H:%M") if t else ""})
    if c4.button("Cancel, no appointment"):
        advance({"approve": False})

# ---------------------------------------------------------------- 3. approve call
elif current == "approve_call":
    st.subheader("Approve the reminder call")
    st.caption("The voice agent says only this: follow-up appointment, doctor, date, time and Hospital A. "
               "No purpose, results, diagnoses or medicines, because whoever answers will hear it.")
    for line in wait["script"]:
        st.markdown(f"- {line}")
    with st.expander("Exactly what the Voice Caller agent receives"):
        st.json(wait["variables"])
    c1, c2 = st.columns(2)
    if c1.button("Approve call", type="primary", icon=":material/call:"):
        advance({"approve": True})
    if c2.button("Do not call"):
        advance({"approve": False})

# ---------------------------------------------------------------- 4. the call
elif current == "call":
    st.subheader("Reminder call")
    tab_voice, tab_sample, tab_paste = st.tabs(["Voice call (ElevenLabs)", "Sample call (no minutes used)", "Nurse called herself"])
    with tab_voice:
        if not voice.configured():
            st.info("Add ELEVENLABS_API_KEY and CARECALL_VOICE_AGENT_ID to .env (run scripts/create_voice_agent.py once).")
        else:
            st.caption("Opens a new tab where you play the patient: answer yes or no, or use the Press 1 / Press 2 buttons. "
                       "About 1 minute of your ElevenLabs credits per call.")
            if not st.session_state.get("call_token") and st.button("Prepare call", icon=":material/call:"):
                try:
                    token, link = voice.open_call(wait["variables"])
                    st.session_state.update(call_token=token, call_link=link)
                    st.rerun()
                except Exception as exc:
                    st.error(f"ElevenLabs could not start the call ({exc.__class__.__name__}). Use a sample call or "
                             "call the patient yourself.", icon=":material/error:")
            if st.session_state.get("call_token"):
                st.link_button("Open the call in a new tab", st.session_state.call_link, type="primary")
                if st.button("Get call result", icon=":material/download:"):
                    res = voice.call_result(st.session_state.call_token)
                    if not res:
                        st.warning("No result yet. Finish the call in the other tab first.")
                    else:
                        st.session_state.call_token = None
                        advance({"transcript": res["transcript"], "source": res["source"],
                                 "conversation_id": res.get("conversation_id")})
    with tab_sample:
        pick = st.selectbox("Sample call", SAMPLES, format_func=lambda s: f"{s['id']}  {s['label']}")
        filled = [{"role": t["role"], "message": t["message"].format(**wait["variables"])} for t in pick["transcript"]]
        for t in filled:
            st.markdown(f"**{'Assistant' if t['role'] == 'agent' else 'Patient'}:** {t['message']}")
        if st.button("Use this call", type="primary"):
            advance({"transcript": filled, "source": f"sample {pick['id']}"})
    with tab_paste:
        text = st.text_area("Write what the patient said (for example: yes, or no)")
        if st.button("Save call notes"):
            turns = [{"role": "user", "message": ln.strip()} for ln in text.splitlines() if ln.strip()]
            advance({"transcript": turns, "source": "nurse call"})

# ---------------------------------------------------------------- 5. final approval
elif current == "final":
    st.subheader("Call Documenter: approve the note and tasks")
    if wait["red_flag"]:
        st.error("RED FLAG: the patient reported a possible emergency symptom. Call the patient now.", icon=":material/emergency:")
    st.markdown(f"Patient answer: **{workflow.ATTENDANCE.get(wait['outcome'], wait['outcome'])}** "
                "(already shown on the Appointment answers board)")
    with st.expander("Call transcript"):
        for t in wait["transcript"]:
            st.markdown(f"**{'Assistant' if t['role'] == 'agent' else 'Patient'}:** {t['message']}")
    choices = {}
    for it in wait["items"]:
        with st.container(border=True):
            st.markdown(f"**{it['title']}**")
            txt = st.text_area("Text", value=it["text"], key=f"f-{st.session_state.flow_thread}-{it['id']}", height=80)
            ok = st.radio("Decision", ["Approve", "Reject"], horizontal=True, key=f"r-{st.session_state.flow_thread}-{it['id']}")
            choices[it["id"]] = ({"action": "edit" if txt != it["text"] else "approve", "text": txt}
                                 if ok == "Approve" else {"action": "reject"})
    if st.button("Submit", type="primary", icon=":material/task_alt:"):
        advance(choices)

# ---------------------------------------------------------------- done
elif current == "done":
    if done.get("stopped"):
        st.info(done["stopped"], icon=":material/info:")
    for r in done.get("results", []):
        st.write(("✅ " if r["status"] == "saved" else "✖️ ") + f"{r['type'].replace('_', ' ')}: {r['status']}")
    if done.get("appointment"):
        a = done["appointment"]
        st.success(f"Appointment: {a['purpose']} with {a['physician']} on {a['date']} at {a['time']}, {a['hospital']}.")
    if st.button("Start again for this patient", icon=":material/restart_alt:"):
        st.session_state.update(flow_thread=None, flow_wait=None, flow_done=None)
        st.rerun()

with st.expander("Care team notes (internal) and appointments for this patient", icon=":material/lock:"):
    st.markdown("**Internal notes**")
    for n in workflow.care_team_notes(pid):
        st.markdown(f"- {n['time'][:16]} · {n['role']} {n['author']}: {n['text']}")
    st.markdown("**Appointments**")
    for a in workflow.appointments(pid):
        st.markdown(f"- {a['date']} {a['time']} · {a['purpose']} · {a['physician']} · {a['hospital']}")

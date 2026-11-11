"""Page: single chart review agent (one ReAct agent with approval before writes).

DEMO ONLY: synthetic Synthea patients. The physician picker stands in for a real nurse sign-in.
Nothing is sent: approved notes, tasks and messages go to local demo files.
"""
import json

import pandas as pd
import streamlit as st

from carecall import config, llm
from carecall.ui import get_agent, sidebar

agent = get_agent()
store = agent.store
physician, view, pid, nurse = sidebar("Nurse sign-in (demo)")
with st.sidebar:
    st.divider()
    if agent._model is not None:
        st.success(f"Model: Nebius `{config.setting('NEBIUS_MODEL')}`", icon=":material/check_circle:")
    else:
        st.info("Nebius is not set up, so the agent uses its fixed plan. Add NEBIUS_API_KEY and NEBIUS_MODEL to .env.",
                icon=":material/info:")
    st.caption("Reads run on their own. Anything that saves, creates or sends waits for your approval.")

if st.session_state.get("cr_pid") != pid:      # a new patient clears the screen
    st.session_state.update(cr_pid=pid, thread=None, pause=None, results=None, refusal=None)

st.title("Chart review agent")
st.caption(f"Follow-up call preparation for {view.label(pid)} · data as of {config.AS_OF}")

request = st.text_area("What do you need for this call?", value="Prepare me for a follow-up call.", height=80,
                       help="For example: 'Prep my call and check whether her A1c is overdue.'")
if st.button("Run agent", type="primary", icon=":material/play_arrow:"):
    with st.spinner("The agent is reviewing the chart..."):
        thread, out = agent.start(physician, pid, request, nurse)
    st.session_state.update(thread=thread, results=None, refusal=out.get("refusal"),
                            pause=out["__interrupt__"][0].value if "__interrupt__" in out else None)

if st.session_state.get("refusal"):
    st.warning(st.session_state.refusal, icon=":material/block:")

pause = st.session_state.get("pause")
if pause:
    state = agent.state(st.session_state.thread)
    for n in dict.fromkeys(pause.get("notices", [])):
        st.info(n, icon=":material/info:")

    with st.expander(f"Agent steps ({len(state.get('tool_log', []))} tool calls, mode: {state.get('mode')})",
                     icon=":material/account_tree:"):
        rows = [{"step": i + 1, "tool": t["tool"], "args": json.dumps(t.get("args") or {}),
                 "result": "ok" if t.get("ok") else "failed", "tries": t.get("attempts"), "ms": t.get("ms")}
                for i, t in enumerate(state.get("tool_log", []))]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Pre-call brief")
        for p in pause["brief"]["brief"]:
            src = " · ".join(p.get("sources") or ["not checked"])
            st.markdown(f"- {p['text']}  \n  :gray-badge[{src}]")
        st.caption(f"Written by: {pause['brief'].get('written_by')}")
    with right:
        st.subheader("Call agenda")
        for item in pause["brief"]["agenda"]:
            st.markdown(f"- {item}")

    if st.session_state.get("results") is None:
        st.subheader("Proposed actions: your approval needed")
        st.caption("Nothing below is saved or sent until you submit. Rejected actions are logged and skipped.")
        decisions = {}
        for a in pause["proposed"]:
            with st.container(border=True):
                st.markdown(f"**{a['title']}** · `{a['type']}`")
                text = st.text_area("Text", value=a["text"], key=f"t-{st.session_state.thread}-{a['id']}", height=90)
                choice = st.radio("Decision", ["Approve", "Reject"], horizontal=True,
                                  key=f"d-{st.session_state.thread}-{a['id']}", index=1)
                if choice == "Approve":
                    decisions[a["id"]] = {"action": "edit" if text != a["text"] else "approve", "text": text}
                else:
                    decisions[a["id"]] = {"action": "reject"}
        if st.button("Submit decisions", type="primary", icon=":material/task_alt:"):
            out = agent.resume(st.session_state.thread, decisions)
            st.session_state.results = out.get("results", [])
            st.rerun()
    else:
        st.subheader("Done")
        for r in st.session_state.results:
            icon = {"saved": ":material/check:", "rejected by nurse": ":material/close:"}.get(r["status"], ":material/schedule_send:")
            if r["status"] == "refused":
                st.error(f"{r['type']}: refused. {r.get('reason', '')}", icon=":material/block:")
            else:
                st.write(f"{icon} {r['type']}: {r['status']}")
        st.caption("Approved notes and tasks are remembered for this patient's next call.")

with st.expander("Agent memory for this patient", icon=":material/history:"):
    st.json(agent.memory.history(pid))
with st.expander("Demo chart notes, tasks and outbox", icon=":material/inbox:"):
    for name in ("chart_notes", "tasks", "outbox"):
        path = config.STORE / f"{name}.jsonl"
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        rows = [r for r in rows if r.get("patient_id") == pid]
        st.markdown(f"**{name.replace('_', ' ')}** ({len(rows)})")
        if rows:
            st.dataframe(pd.DataFrame(rows)[["time", "type", "text", "status"]], hide_index=True, width="stretch")
if llm.USAGE["calls"]:
    st.caption(f"Nebius usage this session: {llm.USAGE['calls']} calls, {llm.USAGE['input_tokens']:,} in / "
               f"{llm.USAGE['output_tokens']:,} out tokens, about ${llm.cost_usd(llm.USAGE['input_tokens'], llm.USAGE['output_tokens'])}")

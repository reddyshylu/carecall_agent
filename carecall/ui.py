"""Shared Streamlit resources (one agent and one orchestrator per app process)."""
import streamlit as st

from .agent import CareCallAgent, sqlite_checkpointer
from .orchestrator import CareTeamOrchestrator
from .workflow import doctor_name


@st.cache_resource
def get_agent() -> CareCallAgent:
    return CareCallAgent(checkpointer=sqlite_checkpointer())


@st.cache_resource
def get_flow() -> CareTeamOrchestrator:
    from langgraph.checkpoint.memory import MemorySaver
    return CareTeamOrchestrator(get_agent(), checkpointer=MemorySaver())


def sidebar(title="Sign-in (demo)"):
    """Physician panel and patient picker shared by both pages. Returns (physician, view, pid, name)."""
    agent = get_agent()
    with st.sidebar:
        st.subheader(title)
        st.caption("Synthetic Synthea patients only. A real app would use your clinic login.")
        name = st.text_input("Your name", value=st.session_state.get("user_name", "Nurse Demo"), key="user_name")
        physician = st.selectbox("Physician panel", agent.store.physicians(), key="physician", format_func=doctor_name)
        view = agent.store.nurse_view(physician)
        pid = st.selectbox(f"Patient ({len(view.panel)} on this panel)", view.panel, format_func=view.label, key="pid_pick")
    return physician, view, pid, name

"""CareCall Agent (Week 3): multi-agent care-team workflow.

    python -m streamlit run app.py

DEMO ONLY: synthetic Synthea patients, fictional "Hospital A". Nothing is sent to a real patient.
"""
import streamlit as st

st.set_page_config(page_title="CareCall Agent", page_icon=":material/support_agent:", layout="wide")
nav = st.navigation([
    st.Page("pages/care_team.py", title="Care team workflow", icon=":material/groups:", default=True),
    st.Page("pages/chart_review.py", title="Chart review agent", icon=":material/clinical_notes:"),
])
nav.run()

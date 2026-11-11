"""Orchestrator: the multi-agent care-team workflow (LangGraph supervisor with 4 human checkpoints).

    review      Agent 1, Chart Reviewer (the ReAct agent in agent.py) prepares the huddle brief
    huddle      [HUMAN] nurse and physician write separate internal notes; the PHYSICIAN decides
                whether a follow-up is needed and picks the purpose (blood test, vaccine, annual check-up)
    book        [HUMAN] the nurse books the date and time at Hospital A (write action, validated)
    approve_call[HUMAN] the nurse approves the fixed reminder script before anyone is called
    call        Agent 2, Voice Caller (ElevenLabs) makes the call; the app hands back the transcript
    document    Agent 3, Call Documenter turns the transcript into outcome, note and tasks
    final       [HUMAN] the nurse approves, edits or rejects the note and each task
    execute     only approved items are written; memory is updated for the next call

The voice agent only ever receives the appointment variables, never the internal notes or the chart.
"""
from __future__ import annotations

import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import actions, documenter, workflow
from .agent import CareCallAgent


class FlowState(TypedDict, total=False):
    physician: str
    patient_id: str
    nurse: str
    review_brief: dict
    gaps: list
    suggested_purpose: str
    huddle: dict
    appointment: dict
    call_vars: dict
    script: list
    call: dict
    documented: dict
    decisions: dict
    results: list
    stopped: str


class CareTeamOrchestrator:
    def __init__(self, reviewer: CareCallAgent, checkpointer=None):
        self.reviewer = reviewer
        self.graph = self._build(checkpointer)

    # ---------------------------------------------------------- agent 1
    def review(self, s: FlowState) -> dict:
        _, out = self.reviewer.start(s["physician"], s["patient_id"], "Prepare a brief for the care team huddle.", s.get("nurse", ""))
        if "__interrupt__" not in out:
            return {"stopped": out.get("refusal") or "The chart review did not finish."}
        brief = out["__interrupt__"][0].value["brief"]
        chart = self.reviewer.store.nurse_view(s["physician"]).open(s["patient_id"])
        gaps = chart.care_gaps()["gaps"]
        return {"review_brief": brief, "gaps": gaps, "suggested_purpose": workflow.suggest_purpose(gaps) or ""}

    # ---------------------------------------------------------- human checkpoints
    def huddle(self, s: FlowState) -> dict:
        d = interrupt({"step": "huddle", "brief": s["review_brief"], "suggested_purpose": s.get("suggested_purpose"),
                       "purposes": list(workflow.PURPOSES)}) or {}
        pid = s["patient_id"]
        for role in ("nurse", "physician"):
            text = (d.get(f"{role}_note") or "").strip()
            if text:
                workflow.save_care_team_note(pid, d.get(f"{role}_name") or role, role, text)
        if not d.get("physician_signoff"):
            actions.audit("huddle_no_signoff", patient_id=pid)
            return {"huddle": d, "stopped": "The physician did not sign off a follow-up, so no appointment or call was made."}
        if d.get("purpose") not in workflow.PURPOSES:
            return {"huddle": d, "stopped": "No follow-up purpose chosen, so no appointment or call was made."}
        actions.audit("huddle_signed", patient_id=pid, physician=d.get("physician_name"), purpose=d["purpose"])
        return {"huddle": d}

    def book(self, s: FlowState) -> dict:
        problem = "start"
        msg = None
        while problem:
            d = interrupt({"step": "book", "purpose": s["huddle"]["purpose"], "error": msg,
                           "physician": workflow.doctor_name(s["physician"]), "hospital": workflow.HOSPITAL}) or {}
            if not d.get("approve"):
                return {"stopped": "The nurse cancelled the booking, so no call was made."}
            problem = workflow.validate_appointment(s["huddle"]["purpose"], d.get("date", ""), d.get("time", ""))
            msg = problem
        appt = workflow.book_appointment(s["patient_id"], s["physician"], s["huddle"]["purpose"], d["date"], d["time"],
                                         s.get("nurse") or "nurse")
        actions.audit("appointment_booked", patient_id=s["patient_id"], appointment_id=appt["appointment_id"])
        first = self.reviewer.store.nurse_view(s["physician"]).open(s["patient_id"]).summary()["first_name"]
        v = workflow.call_variables(appt, first)
        return {"appointment": appt, "call_vars": v, "script": workflow.call_script(v)}

    def approve_call(self, s: FlowState) -> dict:
        d = interrupt({"step": "approve_call", "script": s["script"], "variables": s["call_vars"],
                       "appointment": s["appointment"]}) or {}
        if not d.get("approve"):
            actions.audit("call_not_approved", patient_id=s["patient_id"])
            return {"stopped": "The nurse did not approve the call. The appointment stays booked; no call was made."}
        actions.audit("call_approved", patient_id=s["patient_id"], appointment_id=s["appointment"]["appointment_id"])
        return {}

    # ---------------------------------------------------------- agent 2 (runs in the browser, result comes back here)
    def call(self, s: FlowState) -> dict:
        d = interrupt({"step": "call", "variables": s["call_vars"]}) or {}
        transcript = [t for t in d.get("transcript", []) if isinstance(t, dict) and t.get("message")]
        row = workflow.save_call(s["patient_id"], s["appointment"]["appointment_id"], d.get("source", "none"),
                                 transcript, d.get("conversation_id"))
        actions.audit("call_finished", patient_id=s["patient_id"], source=row["source"], turns=len(transcript))
        return {"call": row}

    # ---------------------------------------------------------- agent 3
    def document(self, s: FlowState) -> dict:
        doc = documenter.document(s["call"]["transcript"], s["appointment"], s["patient_id"], self.reviewer._model)
        # the patient's yes/no goes straight to the nurse's board (it records an answer, it changes no booking)
        workflow.record_attendance(s["appointment"]["appointment_id"], doc["outcome"], s["call"]["call_id"])
        actions.audit("call_documented", patient_id=s["patient_id"], outcome=doc["outcome"], written_by=doc["written_by"])
        return {"documented": doc}

    def final(self, s: FlowState) -> dict:
        doc = s["documented"]
        items = [{"id": "n1", "type": "save_call_note", "title": "Save call note", "text": doc["note"]}]
        items += [{"id": f"t{i}", "type": "create_followup_task", "title": f"Create {t['priority']} task", "text": t["text"]}
                  for i, t in enumerate(doc["tasks"], start=1)]
        d = interrupt({"step": "final", "outcome": doc["outcome"], "red_flag": doc["outcome"] == "red_flag",
                       "items": items, "transcript": s["call"]["transcript"]}) or {}
        return {"decisions": {"items": items, "choices": d}}

    def execute(self, s: FlowState) -> dict:
        pid, nurse = s["patient_id"], s.get("nurse") or "nurse"
        results, note, tasks = [], None, []
        for it in s["decisions"]["items"]:
            c = s["decisions"]["choices"].get(it["id"], {"action": "reject"})
            if c.get("action") not in ("approve", "edit"):
                actions.audit("write_rejected", patient_id=pid, type=it["type"], action_id=it["id"])
                results.append({"id": it["id"], "type": it["type"], "status": "rejected by nurse"})
                continue
            act = {**it, "text": c.get("text", it["text"]) if c["action"] == "edit" else it["text"]}
            r = actions.execute(act, pid, nurse)
            results.append({**r, "type": it["type"], "text": act["text"]})
            if r["status"] != "refused":
                note = act["text"] if it["type"] == "save_call_note" else note
                if it["type"] == "create_followup_task":
                    tasks.append(act["text"])
        if note or tasks:
            self.reviewer.memory.remember_call(pid, note, tasks, nurse)
        return {"results": results}

    # ---------------------------------------------------------- graph
    @staticmethod
    def _go(next_node):
        return lambda s: END if s.get("stopped") else next_node

    def _build(self, checkpointer):
        g = StateGraph(FlowState)
        for name in ("review", "huddle", "book", "approve_call", "call", "document", "final", "execute"):
            g.add_node(name, getattr(self, name))
        g.add_edge(START, "review")
        order = ["review", "huddle", "book", "approve_call", "call"]
        for a, b in zip(order, order[1:]):
            g.add_conditional_edges(a, self._go(b), [b, END])
        g.add_edge("call", "document")
        g.add_edge("document", "final")
        g.add_edge("final", "execute")
        g.add_edge("execute", END)
        if checkpointer is None:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        return g.compile(checkpointer=checkpointer)

    # ---------------------------------------------------------- API used by the UI and tests
    def start(self, physician: str, pid: str, nurse: str = "") -> tuple[str, dict]:
        thread = f"flow-{pid}-{uuid.uuid4().hex[:8]}"
        out = self.graph.invoke({"physician": physician, "patient_id": pid, "nurse": nurse or "Nurse Demo"},
                                {"configurable": {"thread_id": thread}})
        return thread, out

    def resume(self, thread: str, value: Any) -> dict:
        return self.graph.invoke(Command(resume=value), {"configurable": {"thread_id": thread}})

    def state(self, thread: str) -> dict:
        return self.graph.get_state({"configurable": {"thread_id": thread}}).values


def pending(out: dict) -> dict | None:
    """The checkpoint the workflow is waiting on, or None when it has finished."""
    return out["__interrupt__"][0].value if "__interrupt__" in out else None

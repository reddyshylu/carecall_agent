"""A scripted stand-in for the Nebius chat model, so the ReAct path can be tested offline."""
import json

from langchain_core.messages import AIMessage


class ScriptedModel:
    """Round 1: asks for some tools. Round 2: says DONE. Draft call: returns JSON.

    mode="good"        normal behaviour
    mode="bad_cite"    the draft cites a tool that never ran (must be rejected)
    mode="dose_advice" the patient message tells the patient to change a dose (must be rejected)
    mode="crash"       every call raises (agent must fall back to the fixed plan)
    mode="skip"        asks for no tools at all (agent must still run the required reads)
    """

    def __init__(self, mode="good"):
        self.mode, self.rounds, self.calls = mode, 0, 0

    def bind_tools(self, tools):
        self.tool_names = [t.name for t in tools]
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.mode == "crash":
            raise ConnectionError("Nebius unavailable")
        system = getattr(messages[0], "content", "")
        if "pre-call brief" in system:   # draft step
            data = {"brief": [{"text": "Model-written summary of medicines.", "sources": ["get_medications"]}],
                    "agenda": ["Confirm medicines."], "call_note": "Reviewed chart.", "patient_message": ""}
            if self.mode == "bad_cite":
                data["brief"][0]["sources"] = ["get_billing"]
            if self.mode == "dose_advice":
                data["patient_message"] = "Please increase your metformin dose to 1000 mg."
            return AIMessage(json.dumps(data))
        self.rounds += 1
        if self.mode == "skip" or self.rounds > 1:
            return AIMessage("DONE")
        calls = [{"name": n, "args": {}, "id": f"c{i}"} for i, n in
                 enumerate(["get_call_history", "get_patient_summary", "get_medications", "check_care_gaps"])]
        calls.append({"name": "get_recent_labs", "args": {"name": "A1c"}, "id": "c9"})
        return AIMessage("", tool_calls=calls)

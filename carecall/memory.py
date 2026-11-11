"""Agent memory.

Two kinds, as the Week 3 handout asks:
  - Session state: the LangGraph checkpointer (SQLite) keeps each run's state, so a
    run paused for nurse approval survives a page reload or app restart.
  - Across sessions: CallMemory stores, per patient, the approved call notes and the
    follow-up tasks from earlier calls. The next call starts from that history.
Only approved, de-identified content is stored (no SSN, address or phone).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config


class CallMemory:
    def __init__(self, root=None):
        self.root = (root or config.STORE)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "call_memory.json"

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save(self, data: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def history(self, pid: str) -> dict:
        h = self._load().get(pid, {"calls": [], "open_tasks": []})
        return {"source": "call history", "previous_calls": h["calls"][-3:], "open_tasks": h["open_tasks"],
                "note": None if h["calls"] or h["open_tasks"] else "No previous follow-up calls recorded by the agent."}

    def remember_call(self, pid: str, note: str | None, tasks: list[str], nurse: str) -> None:
        data = self._load()
        h = data.setdefault(pid, {"calls": [], "open_tasks": []})
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if note:
            h["calls"].append({"time": stamp, "nurse": nurse, "note": note[:1200]})
        for t in tasks:
            if t not in h["open_tasks"]:
                h["open_tasks"].append(t)
        self._save(data)

    def close_task(self, pid: str, task: str) -> None:
        data = self._load()
        if pid in data and task in data[pid]["open_tasks"]:
            data[pid]["open_tasks"].remove(task)
            self._save(data)

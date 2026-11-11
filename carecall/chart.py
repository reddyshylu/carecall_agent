"""Read-only access to the synthetic chart, scoped to ONE nurse's panel and ONE patient.

ChartStore loads the Synthea CSVs once. NurseView is what the agent gets: it only
knows the patients on the nurse's physician panel, and PatientChart is bound to a
single patient, so no read method takes a patient id from the model.

What the tools never return: SSN, driver's licence, passport, address, income or
other social survey answers. Sensitive social-history findings (abuse, criminal
record, housing, drug misuse...) are withheld from tool output and drafts.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from . import config

SENSITIVE_FINDINGS = (
    "abuse", "violence", "criminal", "homeless", "housing", "misuses drugs", "alcohol",
    "employment", "unemployed", "labor force", "education", "school", "transport",
    "social contact", "social isolation", "stress", "served in armed forces", "risk activity",
    "refugee", "prison", "jail", "income",
)


class ToolDataError(Exception):
    """Raised when a data source cannot be read; the agent sees a short safe message."""


def _day(value) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")[:19]).date()


def is_sensitive(description: str) -> bool:
    d = description.lower()
    return any(word in d for word in SENSITIVE_FINDINGS)


class ChartStore:
    def __init__(self, csv_dir=config.CSV_DIR):
        try:
            read = lambda name: pd.read_csv(csv_dir / f"{name}.csv", dtype=str, keep_default_na=False)
            self.patients = read("patients").set_index("Id")
            self.panels = read("physician_panels")
            self.encounters = read("encounters")
            self.conditions = read("conditions")
            self.medications = read("medications")
            self.observations = read("observations")
            self.allergies = read("allergies")
            self.careplans = read("careplans")
            self.immunizations = read("immunizations")
            self.procedures = read("procedures")
        except (FileNotFoundError, OSError) as exc:
            raise ToolDataError(f"Chart data is missing: {exc.__class__.__name__}") from None

    def physicians(self) -> list[str]:
        return sorted(self.panels.PHYSICIAN_NAME.unique())

    def nurse_view(self, physician: str) -> "NurseView":
        return NurseView(self, physician)


class NurseView:
    """A nurse works for one physician and sees only that physician's panel."""

    def __init__(self, store: ChartStore, physician: str):
        if physician not in set(store.panels.PHYSICIAN_NAME):
            raise ValueError("Unknown physician")
        self.store, self.physician = store, physician
        self.panel = sorted(store.panels[store.panels.PHYSICIAN_NAME == physician].PATIENT)

    def label(self, pid: str) -> str:
        p = self.store.patients.loc[pid]
        return f"{pid}  {p.FIRST.rstrip('0123456789')} {p.LAST.rstrip('0123456789')}"

    def open(self, pid: str) -> "PatientChart":
        if pid not in self.panel:
            raise PermissionError("Patient is not on this nurse's physician panel")
        return PatientChart(self.store, pid, self.physician)


class PatientChart:
    def __init__(self, store: ChartStore, pid: str, physician: str):
        self.s, self.pid, self.physician = store, pid, physician

    def _mine(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.PATIENT == self.pid]

    # ---------------- read tools (all return plain dicts with a "source" tag)
    def summary(self) -> dict:
        p = self.s.patients.loc[self.pid]
        born = _day(p.BIRTHDATE)
        age = config.AS_OF.year - born.year - ((config.AS_OF.month, config.AS_OF.day) < (born.month, born.day))
        enc = self._mine(self.s.encounters).sort_values("START")
        last = enc.iloc[-1] if len(enc) else None
        return {
            "source": "patient summary",
            "patient_id": self.pid,
            "first_name": p.FIRST.rstrip("0123456789"),
            "age": age, "sex": p.GENDER, "physician": self.physician,
            "last_visit": None if last is None else {
                "date": str(_day(last.START)), "type": last.ENCOUNTERCLASS, "reason": last.DESCRIPTION,
                "visit_id": last.Id},
            "visits_last_12_months": int(sum(_day(d) >= config.AS_OF - timedelta(days=365) for d in enc.START)),
        }

    def allergies(self) -> dict:
        rows = self._mine(self.s.allergies)
        active = rows[rows.STOP == ""]
        items = [{"allergy": r.DESCRIPTION, "category": r.CATEGORY, "since": r.START} for r in active.itertuples()]
        return {"source": "allergies", "allergies": items,
                "note": None if items else "No allergies recorded (this means none are documented)."}

    def medications(self, active_only: bool = True) -> dict:
        rows = self._mine(self.s.medications)
        if active_only:
            rows = rows[rows.STOP == ""]
        rows = rows.drop_duplicates("DESCRIPTION", keep="last").sort_values("START", ascending=False)
        items = [{"medication": r.DESCRIPTION, "started": str(_day(r.START)),
                  "reason": r.REASONDESCRIPTION or None} for r in rows.head(15).itertuples()]
        return {"source": "medications", "active_only": active_only, "medications": items,
                "note": "No stop date recorded means active in the record." if items else "No active medications recorded."}

    def conditions(self) -> dict:
        rows = self._mine(self.s.conditions)
        active = rows[rows.STOP == ""].drop_duplicates("DESCRIPTION")
        items, withheld = [], 0
        for r in active.itertuples():
            if is_sensitive(r.DESCRIPTION):
                withheld += 1
                continue
            items.append({"condition": r.DESCRIPTION, "since": r.START})
        return {"source": "conditions", "active_conditions": items,
                "social_history_withheld": withheld}

    def labs(self, name: str = "", limit: int = 8) -> dict:
        obs = self._mine(self.s.observations)
        obs = obs[obs.CATEGORY.isin(["laboratory", "vital-signs"])]
        # Synthea files some survey questions (for example "What address do you live at?") under
        # laboratory; drop anything phrased as a question or touching social history.
        obs = obs[~obs.DESCRIPTION.str.contains("?", regex=False) & ~obs.DESCRIPTION.map(is_sensitive)]
        if name:
            obs = obs[obs.DESCRIPTION.str.lower().str.contains(name.lower(), regex=False)]
        obs = obs.sort_values("DATE", ascending=False)
        latest = obs.drop_duplicates("DESCRIPTION").head(limit)
        items = []
        for r in latest.itertuples():
            hist = obs[obs.DESCRIPTION == r.DESCRIPTION]
            prev = hist.iloc[1] if len(hist) > 1 else None
            items.append({"test": r.DESCRIPTION, "value": r.VALUE, "units": r.UNITS, "date": str(_day(r.DATE)),
                          "previous": None if prev is None else f"{prev.VALUE} on {_day(prev.DATE)}"})
        return {"source": "labs and vitals", "filter": name or None, "results": items,
                "note": None if items else f"No results recorded{' for ' + name if name else ''}."}

    def care_plans(self) -> dict:
        rows = self._mine(self.s.careplans)
        active = rows[rows.STOP == ""].drop_duplicates("DESCRIPTION")
        items = [{"plan": r.DESCRIPTION, "reason": r.REASONDESCRIPTION or None, "since": r.START} for r in active.itertuples()]
        return {"source": "care plans", "active_care_plans": items,
                "note": "Care plans are labels only; detailed instructions are not in the record."}

    def immunizations(self) -> dict:
        rows = self._mine(self.s.immunizations).sort_values("DATE", ascending=False)
        latest = rows.drop_duplicates("DESCRIPTION")
        items = [{"vaccine": " ".join(r.DESCRIPTION.split()), "last_given": str(_day(r.DATE))} for r in latest.itertuples()]
        return {"source": "immunizations", "immunizations": items}

    def care_gaps(self) -> dict:
        """Deterministic follow-up checks (rules, not the model)."""
        gaps = []
        today = config.AS_OF
        flu = [_day(r.DATE) for r in self._mine(self.s.immunizations).itertuples() if "influenza" in r.DESCRIPTION.lower()]
        if not flu or max(flu) < today - timedelta(days=365):
            gaps.append({"gap": "Flu vaccine not recorded in the last 12 months",
                         "last": str(max(flu)) if flu else None, "rule": "G1_flu_12m"})
        conds = " ".join(self._mine(self.s.conditions).query("STOP == ''").DESCRIPTION).lower()
        a1c = [(_day(r.DATE), r.VALUE) for r in self._mine(self.s.observations).itertuples()
               if r.DESCRIPTION.startswith("Hemoglobin A1c")]
        if "diabetes" in conds or "prediabetes" in conds:
            last = max(a1c) if a1c else None
            if last is None or last[0] < today - timedelta(days=183):
                gaps.append({"gap": "Hemoglobin A1c not checked in the last 6 months (diabetes or prediabetes on record)",
                             "last": f"{last[1]} % on {last[0]}" if last else None, "rule": "G2_a1c_6m"})
        bp = self._mine(self.s.observations)
        sys_bp = bp[bp.DESCRIPTION == "Systolic Blood Pressure"].sort_values("DATE")
        if len(sys_bp):
            last = sys_bp.iloc[-1]
            if float(last.VALUE) >= 140:
                gaps.append({"gap": "Last recorded systolic blood pressure was 140 or higher",
                             "last": f"{last.VALUE} mm[Hg] on {_day(last.DATE)}", "rule": "G3_bp_high"})
        enc = [_day(d) for d in self._mine(self.s.encounters).START]
        if enc and max(enc) < today - timedelta(days=365):
            gaps.append({"gap": "No visit in the last 12 months", "last": str(max(enc)), "rule": "G4_no_visit_12m"})
        return {"source": "care gap rules", "as_of": str(today), "gaps": gaps,
                "note": None if gaps else "No care gaps found by the rules."}

    def search_documents(self) -> list[dict]:
        """Visit-level text for keyword search (the Week 2 corpus shape, built from the CSVs)."""
        enc = self._mine(self.s.encounters).sort_values("START")
        docs = []
        for e in enc.itertuples():
            parts = [f"Visit {e.Id} on {_day(e.START)} ({e.ENCOUNTERCLASS}): {e.DESCRIPTION}."]
            if e.REASONDESCRIPTION:
                parts.append(f"Reason: {e.REASONDESCRIPTION}.")
            for df, col, label in ((self.s.conditions, "DESCRIPTION", "Diagnosed"), (self.s.medications, "DESCRIPTION", "Medication"),
                                   (self.s.procedures, "DESCRIPTION", "Procedure")):
                hit = df[(df.PATIENT == self.pid) & (df.ENCOUNTER == e.Id)]
                vals = [v for v in hit[col] if not is_sensitive(v)]
                if vals:
                    parts.append(f"{label}: " + "; ".join(sorted(set(vals))) + ".")
            docs.append({"id": e.Id, "date": str(_day(e.START)), "text": " ".join(parts)})
        return docs

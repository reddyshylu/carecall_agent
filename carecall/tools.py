"""The agent's tools.

READ tools run on their own (the agent decides when). Each one is bound to the
patient the nurse opened, so the model can never pass a patient id.

WRITE actions (save a call note, create a follow-up task, queue a patient
message) are NOT tools the model can call. The agent can only PROPOSE them; they
run in execute() after the nurse approves (see actions.py and agent.py).

Every read tool goes through run_tool(): one retry, a time limit, and a short
safe error message instead of a crash. search_chart() falls back from hybrid
(Pinecone + BM25) to BM25 only when Pinecone is unavailable.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import time
from collections import defaultdict

from rank_bm25 import BM25Okapi

from . import config
from .chart import PatientChart, ToolDataError
from .memory import CallMemory

TOKEN = re.compile(r"[a-z0-9]+")
RRF_K = 60
WEEK2_NAMESPACE = "carecall-200-v1"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Test hook: names of tools forced to fail (used by evals to prove error handling).
FORCED_FAILURES: set[str] = set()


def _tok(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class ChartSearch:
    """Week 2 hybrid search for one patient: Pinecone (meaning) + BM25 (keywords), fused with RRF."""

    _embedder = None

    def __init__(self, chart: PatientChart):
        self.chart = chart
        self.docs = chart.search_documents()
        self.by_id = {d["id"]: d for d in self.docs}
        self.bm25 = BM25Okapi([_tok(d["text"]) for d in self.docs]) if self.docs else None

    def keyword(self, query: str, k: int = 8) -> list[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tok(query))
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        return [self.docs[i]["id"] for i in ranked[:k] if scores[i] > 0]

    def semantic(self, query: str, k: int = 8) -> list[str]:
        if "pinecone" in FORCED_FAILURES:
            raise ConnectionError("Pinecone unavailable (forced)")
        key, index_name = config.setting("PINECONE_API_KEY"), config.setting("PINECONE_INDEX")
        if not key or key.startswith("replace") or not index_name:
            raise ConnectionError("Pinecone not configured")
        from pinecone import Pinecone  # imported lazily so the app runs without it
        from sentence_transformers import SentenceTransformer
        if ChartSearch._embedder is None:
            ChartSearch._embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        vec = ChartSearch._embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
        res = Pinecone(api_key=key).Index(index_name).query(
            vector=vec, top_k=k, namespace=WEEK2_NAMESPACE, include_metadata=True,
            filter={"patient_id": {"$eq": self.chart.pid}})
        hits = []
        for m in res.matches:
            meta = m.metadata or {}
            if meta.get("patient_id") != self.chart.pid:      # never accept another patient's chunk
                continue
            enc = meta.get("encounter_id")
            if enc in self.by_id and enc not in hits:
                hits.append(enc)
        return hits

    def search(self, query: str, k: int = 4) -> dict:
        mode, note = "hybrid", None
        lists = [self.keyword(query)]
        try:
            lists.insert(0, self.semantic(query))
        except Exception as exc:  # Pinecone down or not set up: keep going with keywords
            mode, note = "keyword only", f"Pinecone unavailable ({str(exc)[:60] or exc.__class__.__name__}); used BM25 keyword search."
        score = defaultdict(float)
        for hits in lists:
            for rank, doc_id in enumerate(hits):
                score[doc_id] += 0.5 / (RRF_K + rank + 1)
        top = sorted(score, key=score.get, reverse=True)[:k]
        return {"source": "chart search", "query": query, "mode": mode, "fallback_note": note,
                "results": [{"visit_id": i, "date": self.by_id[i]["date"], "text": self.by_id[i]["text"][:500]} for i in top],
                "note": None if top else "Nothing in this patient's record matched the search."}


class PatientTools:
    """All read tools for one open patient chart."""

    DESCRIPTIONS = {
        "get_patient_summary": "Age, sex, physician, last visit and visit count for the open patient.",
        "get_allergies": "Allergies recorded for the open patient.",
        "get_medications": "Active medications (no stop date) for the open patient.",
        "get_conditions": "Active clinical conditions for the open patient (social history withheld).",
        "get_recent_labs": "Latest labs and vitals. Optional argument 'name' filters by test name, e.g. 'A1c' or 'blood pressure'.",
        "get_care_plans": "Active care plans (labels only).",
        "get_immunizations": "Most recent date for each vaccine.",
        "check_care_gaps": "Rule-based follow-up gaps: flu vaccine, A1c for diabetes, high blood pressure, no recent visit.",
        "search_chart": "Search this patient's visit history by keywords and meaning. Argument 'query'.",
        "get_call_history": "Notes and open tasks from previous follow-up calls with this patient (agent memory).",
    }

    def __init__(self, chart: PatientChart, memory: CallMemory):
        self.chart, self.memory = chart, memory
        self._search = None

    def _impl(self, name: str, args: dict) -> dict:
        c = self.chart
        if name == "get_patient_summary":
            return c.summary()
        if name == "get_allergies":
            return c.allergies()
        if name == "get_medications":
            return c.medications(active_only=True)
        if name == "get_conditions":
            return c.conditions()
        if name == "get_recent_labs":
            return c.labs(name=str(args.get("name", ""))[:60])
        if name == "get_care_plans":
            return c.care_plans()
        if name == "get_immunizations":
            return c.immunizations()
        if name == "check_care_gaps":
            return c.care_gaps()
        if name == "search_chart":
            if self._search is None:
                self._search = ChartSearch(c)
            return self._search.search(str(args.get("query", ""))[:200] or "recent visits")
        if name == "get_call_history":
            return self.memory.history(c.pid)
        raise KeyError(name)

    def run_tool(self, name: str, args: dict | None = None) -> dict:
        """Run one read tool with a retry and a time limit. Never raises."""
        args = args or {}
        if name not in self.DESCRIPTIONS:
            return {"ok": False, "tool": name, "error": "Unknown tool. Use one of the listed tools."}
        last_error = None
        for attempt in range(config.TOOL_RETRIES + 1):
            start = time.time()
            try:
                if name in FORCED_FAILURES:
                    raise ToolDataError("data source unavailable (forced)")
                with cf.ThreadPoolExecutor(max_workers=1) as pool:
                    data = pool.submit(self._impl, name, args).result(timeout=20)
                return {"ok": True, "tool": name, "args": args, "attempts": attempt + 1,
                        "ms": int((time.time() - start) * 1000), "data": data}
            except cf.TimeoutError:
                last_error = "timed out"
            except (ToolDataError, ConnectionError, OSError, ValueError, KeyError) as exc:
                last_error = str(exc)[:120] or exc.__class__.__name__
        return {"ok": False, "tool": name, "args": args, "attempts": config.TOOL_RETRIES + 1,
                "error": f"{name} failed after {config.TOOL_RETRIES + 1} tries: {last_error}"}

    def as_langchain_tools(self):
        """The same read tools as LangChain StructuredTools, for bind_tools() with Nebius."""
        from langchain_core.tools import StructuredTool

        def make(name):
            if name == "get_recent_labs":
                def f(name: str = "") -> str:
                    return json.dumps(self.run_tool("get_recent_labs", {"name": name}))
            elif name == "search_chart":
                def f(query: str) -> str:
                    return json.dumps(self.run_tool("search_chart", {"query": query}))
            else:
                def f() -> str:
                    return json.dumps(self.run_tool(name))
            return StructuredTool.from_function(f, name=name, description=self.DESCRIPTIONS[name])

        return [make(n) for n in self.DESCRIPTIONS]

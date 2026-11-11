# CareCall Agent (Week 3: Build Your AI Agent)

**One-liner:** My multi-agent system helps a nurse and physician at Hospital A move from chart review to a
confirmed follow-up appointment, in a Streamlit app, replacing about 30 minutes of chart review, scheduling
and reminder calls per patient. Three agents (Chart Reviewer, Voice Caller, Call Documenter) do the work
with 10 read tools and an ElevenLabs voice call; people approve at four points (physician sign-off,
booking, the call itself, and the final note), and I'll know it works when the care team gets a booked,
confirmed appointment and an accurate call note 8 times out of 10.

It extends my Week 2 RAG app, **Follow-Up CareCall**. Same 200 synthetic Synthea patients.

> All data is synthetic. "Hospital A" and its phone number 555-0100 are fictional. Never load real patient data.

## The multi-agent workflow (page "Care team workflow")

| Step | Who | What happens |
| --- | --- | --- |
| 1 | **Agent 1: Chart Reviewer** (Nebius, ReAct) | Reads the chart with 10 read tools and prepares a huddle brief with care gaps and citations. |
| 2 | **Nurse and physician** (huddle) | Each writes separate internal notes. The **physician signs off** the follow-up and picks the purpose: **blood test, vaccine or annual check-up** only. |
| 3 | **Nurse** | Books date and time at Hospital A (weekdays 08:00 to 17:00, within 6 months). |
| 4 | **Nurse** | Approves the fixed reminder script before anyone is called. |
| 5 | **Agent 2: Voice Caller** (ElevenLabs) | "Hello, this is a message from Hospital A for Sarah. Sarah has a follow-up appointment with Dr. X on [date] at [time] at Hospital A. Will you be able to come? Say yes or press 1. Say no or press 2." Yes: thanks and ends. No: thanks and ends, **asks nothing else**. Red-flag symptom: "call 911". |
| 6 | **Agent 3: Call Documenter** (rules + Nebius) | Reads the transcript: answer (coming, not coming, no answer, someone else answered, asked not to be called, red flag, unclear). The answer goes **straight to the nurse's Appointment answers board**; the call note and tasks wait for approval. |
| 7 | **Nurse** | Approves, edits or rejects the note and each task. Only approved items are saved; memory updates for the next call. |

The **Orchestrator** is a LangGraph graph with an interrupt at each human step, so the workflow pauses and
resumes exactly where the person left it.

**Privacy by design**
- Internal care-team notes stay on this computer; they are never in the call script or sent to ElevenLabs.
- The Voice Caller receives only 6 variables: first name, Hospital A, its phone, physician, date, time.
  The purpose (blood test, vaccine, annual check-up) stays with the care team: the reminder is heard by whoever
  answers, so it says only "follow-up appointment" (in line with HHS guidance to keep reminder messages minimal).
- No date-of-birth check: the call shares no results, diagnoses, medicines or purpose.
- Your ElevenLabs key stays in Python; the browser gets a short-lived, one-use signed URL. Voice recording is off.

**Voice call without a phone line:** the call runs in a browser tab (you play the patient), so no Twilio
cost. Each demo call uses about 1 minute of ElevenLabs credits. Without credits, use a sample call
(10 scripted calls) or record the call the nurse made herself; the Documenter works the same way.

## The single agent (page "Chart review agent")

### Agent framework (Chart Reviewer)

| Field | Decision |
| --- | --- |
| Agent goal | Turn one patient's chart into a pre-call brief, a call agenda and approved follow-up actions. |
| Where people use it | Streamlit web app (nurse picks her physician, then a patient from that panel). |
| Steps | 1 guard (panel, scope, privacy) → 2 plan (Nebius picks tools) → 3 read tools (loop) → 4 draft brief with citations and proposed actions → 5 nurse approval (LangGraph interrupt) → 6 execute approved writes and update memory. |
| What it can do | READ (autonomous): patient summary, allergies, medications, conditions, labs and vitals, care plans, immunizations, care gap rules, hybrid chart search, call history. WRITE (approval only): save call note, create follow-up task, queue patient message. |
| What it remembers | Session: LangGraph SQLite checkpointer, so a run paused for approval survives a reload. Across sessions: approved call notes and open follow-up tasks per patient. |
| What it never does | Change or recommend medicine doses, diagnose, open patients outside the nurse's panel, read or share SSN, address, phone or income, or send anything. |
| Human in the loop | Before every write. The nurse approves, edits or rejects each proposed action; rejected actions are logged and skipped. |
| When something breaks | Each tool is retried once with a time limit, then reported to the planner and shown to the nurse as "could not check". Pinecone down → BM25 keyword search. Nebius down → fixed plan and rule-based brief. A draft that cites a tool that did not run, or a patient message with dose advice, is rejected in code. |
| How I know it works | Usable call brief in 8 of 10 runs, 0 writes before approval, every forced failure handled (see Evaluation). |

### Architecture (Chart Reviewer)

```
START → guard ─┬─ refuse → END            (other patient, identifiers, not on panel)
               └─ plan ⇄ tools            (ReAct loop: Nebius tool calling, max 4 rounds)
                     └→ draft             (brief with citations + proposed write actions)
                         → human_review   (interrupt: nurse approves / edits / rejects)
                         → execute → END  (approved writes only, audit log, memory update)
```

- **Rules decide, the model explains, the nurse approves.** Care gaps (flu vaccine in 12 months, A1c in 6
  months for diabetes or prediabetes, last systolic BP 140 or higher, no visit in 12 months) are fixed rules.
- **Patient scoping:** a nurse view only knows her physician's panel; tools are bound to the open patient,
  so the model never passes a patient id.
- **Privacy:** tools never return identifiers; sensitive social history (abuse, criminal record, housing,
  income, drug misuse) is withheld from tool output and drafts.
- **Safety net:** if the model skips a required read (summary, allergies, medications, care gaps, call
  history), the draft step runs it.

## Project layout

```
app.py                  Streamlit navigation (2 pages)
pages/care_team.py      multi-agent workflow screen
pages/chart_review.py   single agent screen
carecall/orchestrator.py  multi-agent LangGraph workflow with 4 human checkpoints
carecall/workflow.py    Hospital A, purposes, appointments, internal notes, call script
carecall/voice.py       Voice Caller: signed URL, local call page, transcript fetch
carecall/documenter.py  Call Documenter: outcome rules, note, tasks
scripts/create_voice_agent.py  creates the ElevenLabs reminder agent once
scripts/set_keys.py     adds API keys to .env with hidden input; check_keys.py shows set or not set
carecall/chart.py       Synthea tables, nurse panel and patient scoping, care gap rules
carecall/tools.py       read tools, retry and time limit, hybrid search with BM25 fallback
carecall/agent.py       LangGraph graph: guard, plan, tools, draft, human_review, execute
carecall/actions.py     approved writes (local demo files), content checks, audit log
carecall/memory.py      call history across sessions
carecall/llm.py         Nebius client and token and cost metering
tests/                  54 hard-gate tests
evals/                  16 agent scenarios, 10 sample calls, run_evals.py, run_call_evals.py
data/raw/...            200 synthetic Synthea patients (from Week 2)
```

## Setup (Windows, VS Code terminal)

```
cd "<your folder>\carecall-agent"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Add your own Nebius and ElevenLabs keys with `python scripts/set_keys.py` (the input is hidden, so keys
never appear on screen) or by editing `.env`; check them with `python scripts/check_keys.py` (prints only
set or not set). Pinecone is optional: without it, chart search uses BM25 keyword search. Then create the voice agent once:
```
python scripts/create_voice_agent.py --dry-run
python scripts/create_voice_agent.py
```
Without keys everything still runs: fixed plan instead of Nebius, sample calls instead of voice.

## Run

```
python -m streamlit run app.py      # the app
python -m pytest -q                 # 54 hard-gate tests (54 passed on Windows, Nov 11 2026)
python evals/run_call_evals.py      # 10 sample calls -> runs/call_eval_report.md
python evals/run_evals.py           # 16 scenarios -> runs/eval_report.md (uses Nebius if set)
python evals/run_evals.py --offline # same, without model calls
```

## Evaluation

16 scenarios in `evals/scenarios.csv`: normal calls, memory across calls, four forced tool failures,
three refusals, a medicine-change question, and a nurse who rejects every action.

Offline run (fixed plan, no model calls):

| Measure | Result | Target |
| --- | --- | --- |
| Scenarios passed | 16/16 | all |
| Usable call brief | 13/13 | 8 out of 10 |
| Writes before nurse approval | 0 | 0 |
| Refusals correct | 3/3 | all |
| Forced tool failures handled | 4/4 | all |
| Crashes | 0 | 0 |
| Latency p50 / p95 | under 0.5 s | under 2 minutes |

Run `python evals/run_evals.py` with your Nebius key to get the model-path numbers and cost.

Call Documenter on 10 sample calls (rules only): answer correct 10/10 (yes, no, keypad 1 and 2, voicemail,
someone else, red flag, stop calling, unclear), red flag escalated as URGENT 1/1, notes free of advice 10/10.

## Known limits

- Synthea has no appointment schedule, so appointments are created by the care team in the app.
- The voice call runs in the browser, not over a real phone line (a phone line needs paid telephony).
- Writes go to local demo files (`data/agent_store/`); nothing is sent and there is no EHR connection.
- The care gap rules are illustrative, not clinical guidelines; a clinic would set and sign off its own.

## AI coding tools

Claude was used to design, write and test the code, tests and evaluation. I made the design and safety
decisions (what is autonomous, what needs approval, what the agent must never do) and checked the results.

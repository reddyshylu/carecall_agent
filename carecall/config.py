"""Settings for the CareCall Agent (Week 3). All data is synthetic (Synthea)."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data/raw/synthea_csv_nov2021/csv"
STORE = ROOT / "data/agent_store"          # notes, tasks, outbox, memory, audit (local, never committed)

# The Synthea snapshot ends on 2021-11-19, so "today" for gap checks is that date.
AS_OF = date(2021, 11, 19)

APP_NAME = "CareCall Agent"
NEBIUS_BASE = "https://api.tokenfactory.nebius.com/v1/"
MAX_TOOL_CALLS = 8          # hard stop for the ReAct loop
TOOL_RETRIES = 1            # each failing tool is retried once, then reported to the agent
LLM_TIMEOUT = 30


def _load_env():
    """Read .env next to the project (the .env file wins over older shell values)."""
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


_load_env()


def setting(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def llm_configured() -> bool:
    key, model = setting("NEBIUS_API_KEY"), setting("NEBIUS_MODEL")
    return bool(key) and bool(model) and not key.startswith("replace") and not model.startswith("choose")

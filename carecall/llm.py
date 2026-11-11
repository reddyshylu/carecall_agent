"""Nebius Token Factory through LangChain's OpenAI-compatible client, with usage metering."""
from __future__ import annotations

from . import config

USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "errors": 0}


def price_per_m() -> tuple[float, float]:
    return (float(config.setting("NEBIUS_PRICE_IN_PER_M", "0.10") or 0.10),
            float(config.setting("NEBIUS_PRICE_OUT_PER_M", "0.30") or 0.30))


def cost_usd(inp: int, out: int) -> float:
    pin, pout = price_per_m()
    return round(inp / 1e6 * pin + out / 1e6 * pout, 6)


def chat_model():
    """Return a LangChain chat model for Nebius, or None when no key/model is set (offline mode)."""
    if not config.llm_configured():
        return None
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url=config.NEBIUS_BASE, api_key=config.setting("NEBIUS_API_KEY"),
                      model=config.setting("NEBIUS_MODEL"), temperature=0, timeout=config.LLM_TIMEOUT,
                      max_retries=1)


def record(message) -> None:
    USAGE["calls"] += 1
    meta = getattr(message, "usage_metadata", None) or {}
    USAGE["input_tokens"] += int(meta.get("input_tokens", 0))
    USAGE["output_tokens"] += int(meta.get("output_tokens", 0))

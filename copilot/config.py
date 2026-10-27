"""Central configuration. Values can be overridden with environment variables (see .env.example)."""
from __future__ import annotations

import os
from pathlib import Path

try:  # optional: load .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
BDG2_DATA = Path(os.getenv("BDG2_DATA", ROOT / "building-data-genome-project-2" / "data"))
DB_PATH = Path(os.getenv("ENERGY_DB", ROOT / "data" / "energy.db"))

SITE_ID = os.getenv("CAMPUS_SITE", "Fox")
N_BUILDINGS = 15
ENERGY_UNIT = "kWh"  # BDG2 meter readings are published in kWh (Miller et al., Sci. Data 2020)

# LLM (Groq, OpenAI-compatible endpoint). The key is read from the environment only.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))


def llm_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY") or None


# Query execution
QUERY_TIMEOUT_S = float(os.getenv("QUERY_TIMEOUT_S", "5"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "5000"))  # cap applied to user queries
ANALYTICS_MAX_ROWS = 400_000  # internal analytics queries (15 buildings x 17,544 h fits)
PAGE_SIZE = 10
CACHE_TTL_S = int(os.getenv("CACHE_TTL_S", "900"))

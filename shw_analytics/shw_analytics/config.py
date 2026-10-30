"""All settings in one place. Override any of them with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env_file(path):
    """Load KEY=value lines from .env; variables already set in the shell win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(ROOT / ".env")

# LTIC AI Gateway (OpenAI-compatible, on-prem Qwen at UBC ARC)
GATEWAY_URL = os.getenv("LTIC_BASE_URL", "https://ai-gateway.apps.ltic.ubc.ca/v1")
API_KEY = os.getenv("LTIC_API_KEY", "")  # never hard-code or commit the key

# Model ids: only "qwen3.8-27b" is confirmed (from LTIC's curl example).
# Run `python -m shw_analytics.llm --models` once the key works and fix the others.
CHAT_MODEL = os.getenv("SHW_CHAT_MODEL", "qwen3.8-27b")          # SQL / reasoning
FAST_MODEL = os.getenv("SHW_FAST_MODEL", "qwen3.6-35b-a3b")      # bulk labelling
EMBED_MODEL = os.getenv("SHW_EMBED_MODEL", "qwen3-embedding-0.6b")

# "native" = OpenAI tool calling; "json" = prompt-based fallback if the gateway
# does not pass `tools` through. "auto" tries native first, then falls back.
TOOL_MODE = os.getenv("SHW_TOOL_MODE", "auto")
# Qwen3 can emit <think> blocks. We always strip them; this also asks vLLM not to.
DISABLE_THINKING = os.getenv("SHW_DISABLE_THINKING", "1") == "1"
# Room for the reply. Too low and a model that "thinks" first runs out and returns nothing.
MAX_TOKENS = int(os.getenv("SHW_MAX_TOKENS", "4000"))

# Data
DB_PATH = Path(os.getenv("SHW_DB_PATH", ROOT / "data" / "shw.duckdb"))
DICTIONARY_PATH = Path(os.getenv("SHW_DICTIONARY", ROOT / "dictionary.yaml"))
# Which Qualtrics DistributionChannel values to keep. Dev: include "test".
# Production: "anonymous" (and whatever channel the live link uses).
KEEP_CHANNELS = os.getenv("SHW_KEEP_CHANNELS", "anonymous,test").split(",")

# Privacy
MIN_CELL = int(os.getenv("SHW_MIN_CELL", "5"))            # suppress any count below this
EQUITY_MIN_GROUP = int(os.getenv("SHW_EQUITY_MIN", "10"))  # min group size to compare
MAX_SQL_ROWS = 200

# Equity scan thresholds
FDR_Q = 0.10            # Benjamini-Hochberg false discovery rate
MIN_GAP_TOP2 = 0.10     # flag only gaps of >= 10 points in % agree

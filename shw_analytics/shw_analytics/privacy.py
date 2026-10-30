"""Privacy guards: free-text scrubbing and small-cell suppression.

Scrubbing is regex-based and deliberately conservative. It catches the common
cases (emails, phones, student numbers, "Dr. Smith"), not every name. Treat it
as one layer; the other layers are on-prem inference and aggregate-only output.
"""
import re
import pandas as pd
from . import config

_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"https?://\S+"), "[URL]"),
    (re.compile(r"(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"), "[PHONE]"),
    (re.compile(r"\b\d{8}\b"), "[STUDENT_NO]"),
    # titles / roles followed by a capitalised word: "Dr. Patel", "counsellor Amy"
    (re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Mx|Nurse|Counsellor|Counselor|Advisor|Adviser)\.?\s+[A-Z][a-z]+",
                re.IGNORECASE), "[STAFF_NAME]"),
    (re.compile(r"\bmy name is\s+[A-Z][a-z]+", re.IGNORECASE), "my name is [NAME]"),
]


def scrub_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    out = str(text)
    for pat, repl in _PATTERNS:
        out = pat.sub(repl, out)
    return out.strip() or None


COUNT_COLS = ("n", "count", "n_responses", "responses", "num", "cnt")


def is_count_col(name):
    c = str(name).lower()
    return c in COUNT_COLS or c.startswith("n_") or c.endswith("_n") or c.endswith("_count")


def suppress_small_cells(df: pd.DataFrame, min_cell: int = None) -> tuple[pd.DataFrame, int]:
    """Drop rows whose count-like column is below min_cell. Returns (df, n_dropped)."""
    min_cell = config.MIN_CELL if min_cell is None else min_cell
    count_cols = [c for c in df.columns if is_count_col(c)]
    if not count_cols:
        return df, 0
    mask = pd.Series(True, index=df.index)
    for c in count_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        mask &= ~(vals < min_cell)
    return df[mask].reset_index(drop=True), int((~mask).sum())

"""BERT stage in front of the LLM: (1) block injection attempts, (2) pick the columns that matter."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from functools import lru_cache


INJECTION_MODEL = "protectai/deberta-v3-base-prompt-injection-v2" 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INJECTION_THRESHOLD = 0.90
TOP_K = 8
MIN_SCORE = 0.20
ALWAYS_INCLUDE = {"service", "term"}    

_SQL_RULES = re.compile(
    r"(;\s*(drop|delete|update|insert|alter|attach|copy|pragma)\b"
    r"|\bunion\s+select\b|\bor\s+1\s*=\s*1\b|/\*|\binformation_schema\b"
    r"|\bread_(csv|parquet|json)\s*\(|\bxp_cmdshell\b)", re.I)


@dataclass
class GateResult:
    allowed: bool
    reason: str = ""
    columns: list = field(default_factory=list)   

    @property
    def hint(self) -> str:
        if not self.columns:
            return ""
        cols = ", ".join(f"{t}.{c}" for t, c, _ in self.columns)
        return (f"Columns most likely relevant to this question: {cols}. "
                "Prefer these, but use others if the question needs them.")


@lru_cache(maxsize=1)
def _injection_clf():
    from transformers import pipeline
    return pipeline("text-classification", model=INJECTION_MODEL, truncation=True, max_length=512)


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def check_injection(question: str) -> tuple[bool, str]:
    """Returns (is_attack, reason)."""
    if _SQL_RULES.search(question):
        return True, "contains SQL syntax that is not part of a normal question"
    out = _injection_clf()(question)[0]
    if out["label"].upper() == "INJECTION" and out["score"] >= INJECTION_THRESHOLD:
        return True, f"flagged by the injection classifier ({out['score']:.2f})"
    return False, ""


INDEXED_TABLES = ("responses", "ratings", "items", "response_faculty", "response_disability",
                  "response_multi", "appointments")
SKIP_COLUMNS = {"response_id", "appointment_key", "student_key", "provider_key", "fact_04", "fact_05"}


@dataclass
class GateResult:
    allowed: bool
    reason: str = ""
    columns: list = field(default_factory=list)     # [(table, column, score)]
    items: list = field(default_factory=list)       # [(service, item_code, item_text, score)]

    @property
    def hint(self) -> str:
        if not self.columns and not self.items:
            return ""
        parts = ["Internal guidance, never mention it, or any field names or item codes, to the user."]
        if self.columns:
            seen, names = set(), []
            for _, c, _ in self.columns:          # same column name in several tables: list once
                if c not in seen:
                    seen.add(c)
                    names.append(c)
            parts.append("Data fields most likely relevant: " + ", ".join(names) +
                         ". Prefer them when choosing tools or writing SQL, but use others if needed.")
        if self.items:
            lines = "; ".join(f'{svc}: "{text}" ({code})' for svc, code, text, _ in self.items)
            parts.append("Survey items that may match the question's wording (suggestions only, still "
                         f"confirm with lookup_survey_items before any test): {lines}.")
        return " ".join(parts)


def build_column_index(con):
    """Embed every column using the comments schema.apply_comments wrote into DuckDB,
    plus every survey item (so wording like 'hard to book' can find the booking-ease item)."""
    import numpy as np
    marks = ",".join("?" * len(INDEXED_TABLES))
    tcom = dict(con.execute(f"SELECT table_name, comment FROM duckdb_tables() "
                            f"WHERE table_name IN ({marks})", list(INDEXED_TABLES)).fetchall())
    rows = con.execute(f"""SELECT table_name, column_name, comment FROM duckdb_columns()
                           WHERE table_name IN ({marks})""", list(INDEXED_TABLES)).fetchall()
    cols, docs = [], []
    for t, c, comment in rows:
        if c in SKIP_COLUMNS:
            continue
        text = f"{c.replace('_', ' ')}. {comment or ''} (table: {(tcom.get(t) or t).split('.')[0]})"
        try:   # a few category values for low-cardinality text columns, e.g. faculty names
            vals = con.execute(f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL LIMIT 31').fetchall()
            if 0 < len(vals) <= 30 and isinstance(vals[0][0], str):
                text += " Values: " + ", ".join(v[0] for v in vals[:8])
        except Exception:
            pass
        cols.append((t, c))
        docs.append(text)

    items = con.execute("""SELECT service, item_code, item_text FROM items
                           WHERE duplicate_of IS NULL AND item_text IS NOT NULL""").fetchall()
    emb = _embedder()
    return {"cols": cols, "emb": np.asarray(emb.encode(docs, normalize_embeddings=True)),
            "items": items,
            "item_emb": np.asarray(emb.encode([f"{s.replace('_', ' ')}: {t}" for s, _, t in items],
                                              normalize_embeddings=True))}


def pick_items(q_vec, index, k=3, min_score=0.35):
    scores = index["item_emb"] @ q_vec
    order = scores.argsort()[::-1][:k]
    return [(*index["items"][i], float(scores[i])) for i in order if scores[i] >= min_score]


def pick_columns(q_vec, index, k=TOP_K):
    scores = index["emb"] @ q_vec
    order = scores.argsort()[::-1]
    picked = [(*index["cols"][i], float(scores[i])) for i in order[:k] if scores[i] >= MIN_SCORE]
    have = {c for _, c, _ in picked}
    for i, (t, c) in enumerate(index["cols"]):
        if c in ALWAYS_INCLUDE and c not in have:
            picked.append((t, c, float(scores[i])))
            have.add(c)
    return picked


def run(question: str, index: dict, context: str = "") -> GateResult:
    attack, why = check_injection(question)          # injection check sees only the new text
    if attack:
        return GateResult(False, why)
    q_vec = _embedder().encode([f"{context} {question}".strip()], normalize_embeddings=True)[0]
    return GateResult(True, columns=pick_columns(q_vec, index), items=pick_items(q_vec, index))
"""Open-text pipeline: sentiment + themes, embeddings, and a validity check.

    python -m shw_analytics.comments --label      # sentiment/themes via Qwen
    python -m shw_analytics.comments --embed      # vectors for semantic search
    python -m shw_analytics.comments --validity   # sentiment vs Likert agreement

Comments were already scrubbed at ingest. Labels use a FIXED theme list taken
from the survey items, so counts are comparable across terms and services.
"""
import argparse
import json

import duckdb
import pandas as pd

from . import config, llm

THEMES = ["booking_and_access", "wait_time", "front_desk_and_staff", "feeling_heard",
          "identity_and_inclusion", "privacy_and_confidentiality", "intake_forms",
          "outcomes_and_follow_up", "physical_space", "information_and_resources", "other"]

LABEL_SYSTEM = f"""You label student feedback comments for UBC Student Health & Wellbeing.
For each comment return: id, sentiment (positive | neutral | negative | mixed),
themes (1-3 from this list only: {", ".join(THEMES)}), actionable (true if it names a
concrete problem staff could fix). Return a JSON array, one object per comment.
If a comment is placeholder or nonsense text, use sentiment "neutral" and theme "other"."""

BATCH = 15


def label(con, model=None, limit=None):
    con.execute("""CREATE TABLE IF NOT EXISTS comment_labels (
                     response_id VARCHAR, question_code VARCHAR, sentiment VARCHAR,
                     themes VARCHAR[], actionable BOOLEAN, model VARCHAR)""")
    todo = con.execute("""SELECT c.response_id, c.question_code, c.text FROM comments c
                          ANTI JOIN comment_labels l USING (response_id, question_code)""").df()
    if limit:
        todo = todo.head(limit)
    model = model or config.FAST_MODEL
    done = 0
    for start in range(0, len(todo), BATCH):
        chunk = todo.iloc[start:start + BATCH].reset_index(drop=True)
        payload = [{"id": i, "text": t[:1200]} for i, t in enumerate(chunk["text"])]
        try:
            out = llm.chat_json(LABEL_SYSTEM, json.dumps(payload), model=model)
        except Exception as e:
            print(f"batch {start}: failed ({e}); skipping, rerun to retry")
            continue
        rows = []
        for o in out if isinstance(out, list) else []:
            i = o.get("id")
            if not isinstance(i, int) or i >= len(chunk):
                continue
            sent = o.get("sentiment") if o.get("sentiment") in ("positive", "neutral", "negative", "mixed") else "neutral"
            themes = [t for t in o.get("themes", []) if t in THEMES] or ["other"]
            rows.append({"response_id": chunk.at[i, "response_id"], "question_code": chunk.at[i, "question_code"],
                         "sentiment": sent, "themes": themes, "actionable": bool(o.get("actionable")), "model": model})
        if rows:
            con.register("tmp", pd.DataFrame(rows))
            con.execute("INSERT INTO comment_labels SELECT * FROM tmp")
            con.unregister("tmp")
            done += len(rows)
        print(f"labelled {done}/{len(todo)}")


def embed_all(con, batch=64):
    df = con.execute("SELECT response_id, question_code, text FROM comments").df()
    vecs = []
    for s in range(0, len(df), batch):
        vecs += llm.embed(df["text"].iloc[s:s + batch].tolist())
        print(f"embedded {min(s + batch, len(df))}/{len(df)}")
    df = df.drop(columns="text").assign(vec=vecs)
    con.register("tmp", df)
    con.execute("CREATE OR REPLACE TABLE comment_embeddings AS SELECT * FROM tmp")
    con.unregister("tmp")


def validity_check(con):
    """Does what students write agree with how they rated? A negative comment from
    someone whose mean rating is >= 4 (or the reverse) is a mismatch. A high
    mismatch rate on one service suggests a confusing item, a reversed scale, or a
    recode error, which is exactly what this check is for."""
    df = con.execute("""
        WITH r AS (SELECT response_id, service, AVG(score) AS mean_score
                   FROM ratings GROUP BY 1, 2)
        SELECT r.service, l.sentiment, r.mean_score FROM comment_labels l
        JOIN comments c USING (response_id, question_code)
        JOIN r ON r.response_id = c.response_id AND r.service = c.service
        WHERE l.sentiment IN ('positive', 'negative')""").df()
    if df.empty:
        return df
    df["mismatch"] = ((df["sentiment"] == "negative") & (df["mean_score"] >= 4)) | \
                     ((df["sentiment"] == "positive") & (df["mean_score"] <= 2))
    out = df.groupby("service").agg(n=("mismatch", "size"), mismatches=("mismatch", "sum")).reset_index()
    out["mismatch_pct"] = (100 * out["mismatches"] / out["n"]).round(1)
    return out[out["n"] >= config.MIN_CELL]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--embed", action="store_true")
    ap.add_argument("--validity", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    con = duckdb.connect(str(config.DB_PATH))
    if a.label:
        label(con, limit=a.limit)
    if a.embed:
        embed_all(con)
    if a.validity:
        print(validity_check(con).to_string(index=False))

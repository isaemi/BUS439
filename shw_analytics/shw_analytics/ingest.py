"""Load a raw Qualtrics CSV export into a clean DuckDB database.

Usage:  python -m shw_analytics.ingest path/to/export.csv [--db data/shw.duckdb]

Produces long-format tables the dashboard, the equity scan and the LLM all share:
  responses           one row per response: service, follow-ups, demographics
  ratings             one row per Likert answer (response x item), score 1-5
  comments            one row per open-text answer, already scrubbed
  response_faculty    one row per faculty selected
  response_disability one row per disability category selected
  response_multi      one row per selected option of other multi-select questions
  items               item catalogue (service, item_code, text, duplicates)
  data_quality        issues found during the load
"""
import argparse
import datetime as dt
import json
import re

import duckdb
import pandas as pd
import yaml

from . import config
from .privacy import scrub_text


def load_dictionary(path=None):
    with open(path or config.DICTIONARY_PATH) as f:
        return yaml.safe_load(f)


def read_qualtrics(path):
    """Qualtrics CSVs have 3 header rows: code, question text, ImportId JSON."""
    head = pd.read_csv(path, header=None, nrows=3, dtype=str)
    codes = head.iloc[0].tolist()
    texts = dict(zip(codes, head.iloc[1].tolist()))
    third_is_meta = str(head.iloc[2, 0]).startswith("{")
    skip = [1, 2] if third_is_meta else [1]
    df = pd.read_csv(path, header=0, skiprows=skip, dtype=str)
    return df, texts


class ExportFormatError(ValueError):
    """The export was made with settings the pipeline can't read."""


def read_many(paths):
    """Combine several exports (e.g. daily automated drops that each hold only new
    responses). A response in more than one file keeps its newest copy."""
    frames, texts = [], {}
    for path in sorted(paths):
        df, t = read_qualtrics(path)
        df["_source"] = str(path)
        frames.append(df)
        texts = {**t, **texts}  # earliest file's question text wins, later ones fill gaps
    if not frames:
        raise FileNotFoundError("No CSV exports found.")
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["_recorded"] = pd.to_datetime(df.get("RecordedDate"), errors="coerce")
    df = (df.sort_values(["_recorded", "_source"])
            .drop_duplicates(subset="ResponseId", keep="last")
            .drop(columns=["_recorded"]))
    return df, texts


def resolve_paths(target):
    """A single CSV, or a folder of CSVs (the automated-export drop folder)."""
    from pathlib import Path
    p = Path(target)
    return sorted(p.glob("*.csv")) if p.is_dir() else [p]


def check_format(df, d):
    """Fail early, with a plain explanation, on exports made with the wrong options."""
    blocks = [s["block"] for s in d["services"].values()]
    likert = [c for c in df.columns if re.fullmatch(rf"({'|'.join(blocks)})_\d+", c)]
    vals = pd.concat([df[c] for c in likert]).dropna() if likert else pd.Series(dtype=str)
    if len(vals) and pd.to_numeric(vals, errors="coerce").isna().mean() > 0.5:
        raise ExportFormatError(
            "This export uses answer text (e.g. 'Strongly agree') instead of numeric codes. "
            "In Qualtrics, untick 'export the text of the answer choice' and export again.")


def _item_text(full):
    # "Thinking about ... - I was able to book" -> "I was able to book"
    return full.split(" - ", 1)[1].strip() if " - " in str(full) else str(full)


def _label(d, q, code):
    if code is None or pd.isna(code):
        return None
    return d.get("value_labels", {}).get(q, {}).get(str(code), str(code))


def build_tables(df, texts, d):
    issues = []

    def issue(name, detail, count=None):
        issues.append({"issue": name, "detail": detail, "count": count})

    # Keep only the wanted distribution channels.
    counts = df["DistributionChannel"].value_counts(dropna=False).to_dict()
    issue("channels_in_export", json.dumps({str(k): int(v) for k, v in counts.items()}))
    df = df[df["DistributionChannel"].isin(config.KEEP_CHANNELS)].copy()

    # Drop identifying columns, and report any that actually held data.
    for col in d["drop_columns"]:
        if col in df.columns:
            filled = int(df[col].notna().sum())
            if filled and col in ("IPAddress", "LocationLatitude", "LocationLongitude"):
                issue("pii_collected", f"{col} populated; turn on 'Anonymize responses' in Qualtrics", filled)
            df = df.drop(columns=col)

    df = df.rename(columns={"ResponseId": "response_id"})
    svc_map = {int(k): v for k, v in d["services"].items()}
    svc_code = pd.to_numeric(df[d["service_question"]], errors="coerce")

    # One row per response.
    resp = pd.DataFrame({
        "response_id": df["response_id"],
        "recorded_at": pd.to_datetime(df["RecordedDate"], errors="coerce"),
        "channel": df["DistributionChannel"],
        "finished": df["Finished"] == "1",
        "progress": pd.to_numeric(df["Progress"], errors="coerce"),
        "service_code": svc_code.astype("Int64"),
        "service": svc_code.map(lambda c: svc_map.get(c, {}).get("key") if pd.notna(c) else None),
        "service_name": svc_code.map(lambda c: svc_map.get(c, {}).get("name") if pd.notna(c) else None),
    })
    for q, meta in d["followups"].items():
        if q in df:
            resp[meta["name"]] = df[q].map(lambda v, q=q: _label(d, q, v))

    multi_rows = []
    for q, meta in d["demographics"].items():
        if q not in df:
            issue("missing_column", q)
            continue
        if meta.get("multi"):
            resp[meta["name"]] = df[q]  # raw "1,3" string kept for reference
            for rid, val in zip(df["response_id"], df[q]):
                if pd.notna(val):
                    for code in str(val).split(","):
                        multi_rows.append({"response_id": rid, "question": meta["name"],
                                           "value": _label(d, q, code.strip())})
        else:
            resp[meta["name"]] = df[q].map(lambda v, q=q: _label(d, q, v))

    by = d["birth_year"]
    years = pd.to_numeric(df.get(by["question"]), errors="coerce")
    max_year = dt.date.today().year - 15
    bad = df[by["question"]].notna() & ~years.between(by["min"], max_year)
    if bad.any():
        issue("invalid_birth_year", f"non-numeric or outside {by['min']}-{max_year}; add validation in Qualtrics", int(bad.sum()))
    resp["birth_year"] = years.where(~bad).astype("Int64")

    # Checkbox questions become long tables; disability_any is derived here.
    long_tables = {}
    for name, cb in d["checkbox"].items():
        rows = []
        split_cols = [f"{cb['prefix']}{i}" for i in range(1, len(cb["choices"]) + 1)]
        joined = cb["prefix"].rstrip("_")
        if not any(c in df for c in split_cols) and joined in df:
            # "Split multi-value fields into columns" was off: one column like "1,4,9".
            issue("multi_value_joined", f"{joined} exported as one column; parsed the comma list")
            for rid, val in zip(df["response_id"], df[joined]):
                for code in str(val).split(",") if pd.notna(val) else []:
                    code = code.strip()
                    if code.isdigit() and 1 <= int(code) <= len(cb["choices"]):
                        rows.append({"response_id": rid, name: cb["choices"][int(code) - 1], "choice": int(code)})
        for i, label in enumerate(cb["choices"], start=1):
            col = f"{cb['prefix']}{i}"
            if col in df:
                for rid in df.loc[df[col] == "1", "response_id"]:
                    rows.append({"response_id": rid, name: label, "choice": i})
        long_tables[name] = pd.DataFrame(rows, columns=["response_id", name, "choice"])
        if "derive" in cb:
            dv = cb["derive"]
            picked = long_tables[name].groupby("response_id")["choice"].apply(set)
            def derive(s):
                if s & set(dv["yes_choices"]):
                    return "yes"
                if s & set(dv["no_choices"]):
                    return "no"
                return None
            resp[dv["name"]] = resp["response_id"].map(picked.map(derive))
            contradictory = picked.map(lambda s: bool(s & set(dv["yes_choices"])) and bool(s & set(dv["no_choices"])))
            if contradictory.any():
                issue("contradictory_checkbox", f"{name}: 'None' selected together with other options; add exclusive-answer setting", int(contradictory.sum()))

    # Ratings in long format, plus the item catalogue.
    rec = d["likert"].get("recode", {})
    rating_frames, item_rows = [], []
    for code, s in svc_map.items():
        block = s["block"]
        cols = [c for c in df.columns if re.fullmatch(rf"{block}_\d+", c)]
        recode = {int(k): v for k, v in rec.get(block, {}).items()}
        seen = {}
        for c in cols:
            text = _item_text(texts.get(c, c))
            dup = seen.get(text.lower())
            seen.setdefault(text.lower(), c)
            item_rows.append({"service": s["key"], "block": block, "item_code": c,
                              "item_text": text, "duplicate_of": dup})
            if dup:
                issue("duplicate_item", f"{c} repeats {dup}: '{text[:60]}'")
            scores = pd.to_numeric(df[c], errors="coerce")
            if recode:
                scores = scores.map(lambda v: recode.get(int(v), v) if pd.notna(v) else v)
            out_of_range = scores.notna() & ~scores.between(1, 5)
            if out_of_range.any():
                issue("likert_out_of_range", f"{c} has values outside 1-5; fix the recode in Qualtrics", int(out_of_range.sum()))
            frame = pd.DataFrame({"response_id": df["response_id"], "service": s["key"],
                                  "item_code": c, "score": scores.where(~out_of_range)})
            rating_frames.append(frame.dropna(subset=["score"]))
        # Items that name a different service were probably copied from another block.
        others = [o["name"] for c2, o in svc_map.items() if c2 != code]
        for it in item_rows:
            if it["block"] == block:
                hit = [o for o in others if o.lower() in it["item_text"].lower()]
                if hit:
                    issue("item_names_other_service",
                          f"{it['item_code']} ({s['name']}) mentions {hit[0]}: '{it['item_text'][:70]}'")
        if recode:
            issue("likert_recoded", f"{block} recoded {sorted(recode)} -> 1-5 at load time")
    ratings = pd.concat(rating_frames, ignore_index=True)
    ratings["score"] = ratings["score"].astype(int)
    ratings["agree"] = ratings["score"] >= d["likert"]["top2_min"]
    items = pd.DataFrame(item_rows)
    ratings = ratings.merge(items[["item_code", "item_text"]], on="item_code", how="left")

    # Comments are scrubbed before they are stored anywhere.
    comment_rows = []
    comment_cols = {s["comment"]: s["key"] for s in svc_map.values() if s.get("comment")}
    comment_cols[d["general_comment"]] = None
    for col, key in comment_cols.items():
        if col not in df:
            continue
        for rid, svc, txt in zip(df["response_id"], resp["service"], df[col]):
            clean = scrub_text(txt)
            if clean:
                comment_rows.append({"response_id": rid, "service": key or svc,
                                     "question_code": col, "text": clean})
    comments = pd.DataFrame(comment_rows, columns=["response_id", "service", "question_code", "text"])

    # Sanity checks on embedded data and unmodelled columns.
    if "fac" in df and df["fac"].fillna("").str.contains("Custom Value").any():
        issue("embedded_data", "'fac' field holds the whole choice list + 'Custom Value'; piped-text setting is wrong")
    known = set(d["drop_columns"])
    unknown = [c for c in df.columns if c not in known and not re.match(r"^(D6_|D16_|[A-Z]+1_)", c)
               and c not in d["demographics"] and c not in d["followups"]
               and c not in comment_cols and c not in ("response_id",)]
    issue("columns_not_modelled", ", ".join(unknown))

    multi = pd.DataFrame(multi_rows, columns=["response_id", "question", "value"])
    return {
        "responses": resp, "ratings": ratings, "comments": comments, "items": items,
        "response_faculty": long_tables["faculty"].drop(columns="choice"),
        "response_disability": long_tables["disability"].drop(columns="choice"),
        "response_multi": multi, "data_quality": pd.DataFrame(issues),
    }


def write_db(tables, db_path=None):
    db_path = db_path or config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    for name, frame in tables.items():
        con.register("tmp", frame)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp")
        con.unregister("tmp")
    # Convenience view: every rating joined to the respondent's demographics.
    con.execute("""CREATE OR REPLACE VIEW ratings_demo AS
                   SELECT r.*, s.* EXCLUDE (response_id, service)
                   FROM ratings r JOIN responses s USING (response_id)""")
    from .schema import apply_comments
    apply_comments(con, load_dictionary())
    con.close()


def load(target, db_path=None, chr_dir=None):
    """Read one export or a folder of exports, build every table, and swap the new
    database in only once it is complete (so the app never sees a half-built one)."""
    import os
    from pathlib import Path
    d = load_dictionary()
    df, texts = read_many(resolve_paths(target))
    check_format(df, d)
    tables = build_tables(df.drop(columns="_source"), texts, d)
    final = Path(db_path) if db_path else config.DB_PATH
    tmp = final.with_suffix(".building.duckdb")
    tmp.unlink(missing_ok=True)
    write_db(tables, tmp)
    # CHR appointment exports, if any, go into the same database before the swap.
    import duckdb
    from . import chr as chr_
    con = duckdb.connect(str(tmp))
    chr_issues = chr_.add_tables(con, d, chr_dir or chr_.CHR_DIR)
    if chr_issues:
        dq = pd.concat([tables["data_quality"], pd.DataFrame(chr_issues)], ignore_index=True)
        con.register("tmp_dq", dq)
        con.execute("CREATE OR REPLACE TABLE data_quality AS SELECT * FROM tmp_dq")
        tables["data_quality"] = dq
    from .schema import apply_comments
    apply_comments(con, d)
    con.close()
    os.replace(tmp, final)
    return tables


def main():
    ap = argparse.ArgumentParser(description="Load a Qualtrics export, or a folder of them, into DuckDB.")
    ap.add_argument("target", help="a CSV export, or a folder of CSV exports (e.g. data/exports)")
    ap.add_argument("--db", default=None)
    a = ap.parse_args()
    try:
        tables = load(a.target, a.db)
    except ExportFormatError as e:
        raise SystemExit(f"Export format problem: {e}")
    for k, v in tables.items():
        print(f"{k:22s} {len(v):6d} rows")
    print("\nData quality:")
    for _, r in tables["data_quality"].iterrows():
        cnt = "" if pd.isna(r["count"]) else f" (n={int(r['count'])})"
        print(f"  - {r['issue']}: {r['detail']}{cnt}")


if __name__ == "__main__":
    main()

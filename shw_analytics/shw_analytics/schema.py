"""The database schema in one place: what each table and column means, which column
is the key, and how tables join. Descriptions are written into DuckDB as table and
column comments at every rebuild, so the schema documents itself, and the model
reads them (with row counts) before writing any SQL.

    python -m shw_analytics.schema              # print the live schema
    python -m shw_analytics.schema --markdown   # write docs/SCHEMA.md
"""
import argparse
from pathlib import Path

from . import config

TABLES = {
    "responses": {
        "description": "One row per survey response: the service used, follow-up answers and demographics.",
        "key": "response_id",
        "columns": {
            "response_id": "Qualtrics ResponseId; links every other table",
            "recorded_at": "When the response was recorded",
            "channel": "Qualtrics distribution channel (anonymous = real link, test, preview)",
            "finished": "True if the respondent reached the end of the survey",
            "progress": "Percent of the survey completed",
            "service_code": "Code of the service the respondent rated (survey question Int1)",
            "service": "Service key, e.g. medical, counselling",
            "service_name": "Service display name",
            "birth_year": "Year of birth; blank if missing or invalid",
            "disability_any": "yes if any disability category was selected, no if only 'None'",
        },
    },
    "ratings": {
        "description": "One row per Likert answer (response x survey item), score 1-5.",
        "key": "response_id + item_code",
        "columns": {
            "response_id": "Links to responses",
            "service": "Service the item belongs to",
            "item_code": "Qualtrics question code, e.g. CS1_3",
            "score": "1 = strongly disagree ... 5 = strongly agree",
            "agree": "True if score is 4 or 5",
            "item_text": "Wording of the survey item",
        },
    },
    "items": {
        "description": "Catalogue of survey items per service.",
        "key": "item_code",
        "columns": {
            "service": "Service the item belongs to",
            "block": "Qualtrics question block, e.g. CS1",
            "item_code": "Qualtrics question code",
            "item_text": "Wording of the item",
            "duplicate_of": "Item code this one repeats, if the same wording appears twice",
        },
    },
    "comments": {
        "description": "Open-text comments, scrubbed of names, emails and numbers. Not reachable from SQL.",
        "key": "response_id + question_code",
        "columns": {
            "response_id": "Links to responses",
            "service": "Service the comment is about",
            "question_code": "Qualtrics question the comment answers",
            "text": "Scrubbed comment text",
        },
    },
    "response_faculty": {
        "description": "One row per faculty a respondent selected (multi-select).",
        "key": "response_id + faculty",
        "columns": {"response_id": "Links to responses", "faculty": "Faculty name, e.g. Arts"},
    },
    "response_disability": {
        "description": "One row per disability category a respondent selected (multi-select).",
        "key": "response_id + disability",
        "columns": {"response_id": "Links to responses", "disability": "Disability category"},
    },
    "response_multi": {
        "description": "One row per option selected in other multi-select questions.",
        "key": "response_id + question + value",
        "columns": {"response_id": "Links to responses",
                    "question": "Which question, e.g. racial_background",
                    "value": "Selected option code"},
    },
    "appointments": {
        "description": "One row per CHR appointment. Students and providers are anonymised but "
                       "consistent; names never reach the database.",
        "key": "appointment_key",
        "columns": {
            "appointment_key": "Anonymised appointment ID",
            "student_key": "Anonymised student ID; the same student always has the same key",
            "provider_key": "Anonymised practitioner",
            "start_time": "Appointment start",
            "end_time": "Appointment end",
            "service_line": "Counselling, Medical, Wellness Advising, ...",
            "service_program": "Program code in the service name, e.g. MDUP",
            "visit_type": "Initial or Follow Up",
            "mode": "In Person or Phone/Video",
            "service_raw": "Service name as exported",
            "minutes": "Scheduled length in minutes",
            "reason": "Appointment reason",
            "dx_code": "Service/diagnosis code; often 'No Diagnosis Recorded'",
            "status": "Status as exported, e.g. note_signed",
            "outcome": "attended, no_show, cancelled or other (mapped in dictionary.yaml)",
            "tag": "Appointment tag, e.g. E5 (meaning to be confirmed)",
            "location": "Location name, with staff names removed",
            "is_group": "True for group visits",
            "faculty": "Student's faculty (CHR fact 01)",
            "academic_level": "Undergraduate or Graduate (CHR fact 02)",
            "program": "Student's program (CHR fact 03)",
            "gender": "Gender (CHR fact 06)",
            "fact_04": "CHR fact 04 (empty so far; meaning to be confirmed)",
            "fact_05": "CHR fact 05 (empty so far; meaning to be confirmed)",
            "age_band": "Age band",
            "term": "Academic term, e.g. 2026 Winter",
        },
    },
    "chr_meta": {
        "description": "Which CHR export files were loaded, and whether they are synthetic test data.",
        "key": None,
        "columns": {"synthetic": "True if the CHR data is synthetic", "files": "Files loaded"},
    },
    "data_quality": {
        "description": "Problems found in the Qualtrics export at the last rebuild.",
        "key": None,
        "columns": {"issue": "Kind of problem", "detail": "What was found", "count": "How many rows"},
    },
}

VIEWS = {
    "ratings_demo": "ratings joined to responses: every rating with the respondent's demographics. "
                    "Use this for most questions; no join needed.",
}

RELATIONSHIPS = [
    ("ratings.response_id", "responses.response_id", "many ratings per response"),
    ("comments.response_id", "responses.response_id", "zero or more comments per response"),
    ("response_faculty.response_id", "responses.response_id", "one row per selected faculty"),
    ("response_disability.response_id", "responses.response_id", "one row per selected disability"),
    ("response_multi.response_id", "responses.response_id", "one row per selected option"),
    ("ratings.item_code", "items.item_code", "each rating is for one item"),
]


def _column_descriptions(d):
    """Fixed descriptions plus ones generated from the dictionary for demographic and
    follow-up columns, so a new survey question is described automatically."""
    cols = {t: dict(spec["columns"]) for t, spec in TABLES.items()}
    for q, meta in d.get("demographics", {}).items():
        kind = "equity dimension; " if meta.get("equity") else ""
        multi = "raw comma list; one row per option in response_multi" if meta.get("multi") else "coded answer"
        cols["responses"][meta["name"]] = f"Survey question {q} ({kind}{multi})"
    for q, meta in d.get("followups", {}).items():
        cols["responses"][meta["name"]] = f"Survey question {q} (coded answer)"
    return cols


def _q(text):
    return "'" + str(text).replace("'", "''") + "'"


def apply_comments(con, d):
    """Write table and column descriptions into the database as comments."""
    cols = _column_descriptions(d)
    existing = {(r[0], r[1]) for r in con.execute(
        "SELECT table_name, column_name FROM duckdb_columns()").fetchall()}
    for table, spec in TABLES.items():
        if not any(t == table for t, _ in existing):
            continue
        desc = spec["description"] + (f" Key: {spec['key']}." if spec["key"] else "")
        con.execute(f"COMMENT ON TABLE {table} IS {_q(desc)}")
        for col, text in cols[table].items():
            if (table, col) in existing:
                con.execute(f'COMMENT ON COLUMN {table}."{col}" IS {_q(text)}')
    for view, text in VIEWS.items():
        con.execute(f"COMMENT ON VIEW {view} IS {_q(text)}")


def describe(con, include=None):
    """Live schema from the database itself: tables, row counts, columns, types,
    descriptions, keys and joins (the DuckDB equivalent of information_schema)."""
    tables = con.execute("SELECT table_name, comment, estimated_size FROM duckdb_tables() "
                         "ORDER BY table_name").df()
    cols = con.execute("SELECT table_name, column_name, data_type, comment FROM duckdb_columns() "
                       "ORDER BY table_name, column_index").df()
    out = {}
    for t in tables.itertuples():
        if include and t.table_name not in include:
            continue
        c = cols[cols["table_name"] == t.table_name]
        out[t.table_name] = {
            "description": t.comment if isinstance(t.comment, str) else "",
            "rows": int(t.estimated_size),
            "columns": [f"{r.column_name} {r.data_type}" + (f": {r.comment}" if isinstance(r.comment, str) else "")
                        for r in c.itertuples()],
        }
    return {"tables": out, "views": VIEWS,
            "relationships": [f"{a} -> {b} ({how})" for a, b, how in RELATIONSHIPS]}


def to_markdown(schema):
    lines = ["# Database schema", "",
             "Generated by `python -m shw_analytics.schema --markdown` from the live database.", ""]
    lines += ["## Relationships", ""] + [f"- `{r}`" for r in schema["relationships"]] + [""]
    lines += ["## Views", ""] + [f"- `{v}`: {t}" for v, t in schema["views"].items()] + [""]
    for name, t in schema["tables"].items():
        lines += [f"## `{name}` ({t['rows']:,} rows)", "", t["description"], "",
                  "| Column | Type | Meaning |", "|---|---|---|"]
        for c in t["columns"]:
            head, _, meaning = c.partition(": ")
            col, _, typ = head.partition(" ")
            lines.append(f"| `{col}` | {typ} | {meaning} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import json
    from .analysis import connect
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    s = describe(connect())
    if a.markdown:
        out = config.ROOT / "docs" / "SCHEMA.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text(to_markdown(s))
        print(f"Wrote {out}")
    else:
        print(json.dumps(s, indent=2))

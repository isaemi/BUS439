"""TELUS CHR appointment exports -> an `appointments` table.

The export mixes appointment rows with group-visit template rows and many blank
lines; only rows with an appointment ID are kept. Identifiers are replaced on the
way in: patient and appointment IDs become salted hashes (still consistent, so
visits per student and length of care can be counted), practitioner names become
provider codes, and age becomes an age band. Names never reach the database.

    python -m shw_analytics.chr          # rebuild the database with data/chr_exports
"""
import hashlib
import io
import os
import re
import secrets

import numpy as np
import pandas as pd

from . import config

CHR_DIR = config.DB_PATH.parent / "chr_exports"
_SALT_FILE = config.DB_PATH.parent / ".id_salt"

# "Counselling (MDUP) - Follow Up (Phone/Video)" -> line, program, visit type, mode
_SERVICE = re.compile(r"^\s*(?P<line>[^(]+?)\s*(?:\((?P<program>[^)]*)\))?\s*-\s*(?P<type>[^(]+?)"
                      r"\s*(?:\((?P<mode>[^)]*)\))?\s*$")


def _salt():
    """Secret used for hashing IDs. Kept in .env (SHW_ID_SALT) or data/.id_salt, never
    in the repo. Changing it changes every hashed ID."""
    s = os.getenv("SHW_ID_SALT")
    if s:
        return s
    if not _SALT_FILE.exists():
        _SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SALT_FILE.write_text(secrets.token_hex(16))
    return _SALT_FILE.read_text().strip()


def _hash(values, prefix, salt, length=10):
    return values.map(lambda v: f"{prefix}{hashlib.sha256((salt + str(v)).encode()).hexdigest()[:length]}"
                      if pd.notna(v) else None)


def _text(df, col):
    """A column as strings (empty or missing columns become all-missing strings)."""
    if col not in df:
        return pd.Series(pd.NA, index=df.index, dtype="string")
    return df[col].astype("string")


def parse_service(text):
    if not isinstance(text, str):
        return {"service_line": None, "service_program": None, "visit_type": None, "mode": None}
    m = _SERVICE.match(text)
    if not m:
        return {"service_line": text.strip(), "service_program": None, "visit_type": None, "mode": None}
    return {"service_line": m["line"].strip(), "service_program": m["program"],
            "visit_type": m["type"].strip(), "mode": m["mode"] or "In Person"}


def _age_band(age):
    a = pd.to_numeric(age, errors="coerce")
    return pd.cut(a, [0, 19, 22, 25, 30, 200], labels=["under 20", "20-22", "23-25", "26-30", "over 30"],
                  right=True).astype(str).replace("nan", None)


def read_exports(paths, d):
    """Read and clean CHR exports. Returns (appointments, issues)."""
    cfg = d["chr"]
    frames, issues = [], []
    for path in sorted(paths):
        # CHR exports end lines with "\r\r\n"; normalise before parsing.
        text = path.read_bytes().decode("utf-8-sig", errors="replace").replace("\r", "")
        raw = pd.read_csv(io.StringIO(text), dtype=str)
        raw.columns = [c.strip() for c in raw.columns]
        raw = raw.map(lambda v: v.strip() if isinstance(v, str) else v)
        appts = raw[raw.get("Appointments Appointment ID", pd.Series(dtype=str)).notna()]
        other = raw.dropna(how="all")
        if len(other) > len(appts):
            issues.append({"issue": "chr_non_appointment_rows",
                           "detail": f"{path.name}: {len(other) - len(appts)} rows without an appointment "
                                     "(e.g. group-visit templates) skipped", "count": len(other) - len(appts)})
        missing = [c for c in cfg["columns"] if c not in raw.columns]
        if missing:
            issues.append({"issue": "chr_missing_columns", "detail": ", ".join(missing), "count": len(missing)})
        frames.append(appts.rename(columns=cfg["columns"]))
    if not frames:
        return pd.DataFrame(), issues
    df = pd.concat(frames, ignore_index=True)

    salt = _salt()
    out = pd.DataFrame({
        "appointment_key": _hash(df["appointment_id"], "A", salt),
        "student_key": _hash(df["patient_id"], "S", salt),
        "provider_key": _hash(_text(df, "practitioner"), "P", salt, 6),
        "start_time": pd.to_datetime(df["start_time"], errors="coerce"),
        "end_time": pd.to_datetime(df.get("end_time"), errors="coerce"),
    })
    svc = pd.DataFrame([parse_service(t) for t in df["service_raw"]], index=df.index)
    out = pd.concat([out, svc], axis=1)
    out["service_raw"] = df["service_raw"]
    out["minutes"] = ((out["end_time"] - out["start_time"]).dt.total_seconds() / 60).round()
    out["reason"] = df.get("reason")
    out["dx_code"] = df.get("dx_code")
    out["status"] = _text(df, "status").str.lower()
    out["outcome"] = np.select(
        [out["status"].isin(cfg["attended"]), out["status"].isin(cfg["no_show"]),
         out["status"].isin(cfg["cancelled"])], ["attended", "no_show", "cancelled"], "other")
    out["tag"] = df.get("tag")
    # Location names can contain staff first names in brackets, e.g. "(Myra)".
    out["location"] = _text(df, "location").str.replace(
        r"\s*\([^)]*\)", "", regex=True).str.strip()
    out["is_group"] = _text(df, "is_group").eq("Yes").fillna(False)
    for col in ("faculty", "academic_level", "program", "gender", "fact_04", "fact_05"):
        v = _text(df, col).str.rstrip(",").str.strip()
        out[col] = v.mask(v == "")
    out["age_band"] = _age_band(df.get("age"))
    out["term"] = out["start_time"].map(_term)

    dupes = out["appointment_key"].duplicated(keep="last")
    if dupes.any():
        issues.append({"issue": "chr_duplicate_appointments", "detail": "kept the newest copy",
                       "count": int(dupes.sum())})
    out = out[~dupes].reset_index(drop=True)
    unknown = out.loc[out["outcome"] == "other", "status"].dropna().unique().tolist()
    if unknown:
        issues.append({"issue": "chr_unmapped_status",
                       "detail": f"statuses not in dictionary.yaml chr lists: {', '.join(unknown[:10])}",
                       "count": int((out["outcome"] == "other").sum())})
    return out, issues


def _term(ts):
    if pd.isna(ts):
        return None
    if ts.month >= 9:
        return f"{ts.year} Fall"
    if ts.month <= 4:
        return f"{ts.year} Winter"
    return f"{ts.year} Summer"


def add_tables(con, d, folder=CHR_DIR):
    """Load every CHR export in `folder` into `appointments` (called by ingest.load).
    Returns the issues found, for the data quality report."""
    paths = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not paths:
        return []
    appts, issues = read_exports(paths, d)
    if appts.empty:
        return issues
    con.register("tmp_chr", appts)
    con.execute("CREATE OR REPLACE TABLE appointments AS SELECT * FROM tmp_chr")
    con.unregister("tmp_chr")
    synthetic = any("synthetic" in p.name.lower() for p in paths)
    con.execute("CREATE OR REPLACE TABLE chr_meta AS SELECT ? AS synthetic, ? AS files",
                [synthetic, ", ".join(p.name for p in paths)])
    return issues


if __name__ == "__main__":
    from . import ingest
    survey = config.DB_PATH.parent / "exports"
    tables = ingest.load(survey)
    from .analysis import connect
    n = connect().execute("SELECT COUNT(*) FROM appointments").fetchone()[0] \
        if CHR_DIR.exists() and any(CHR_DIR.glob("*.csv")) else 0
    print(f"Survey responses: {len(tables['responses'])}; CHR appointments: {n}")

"""Do the app's appointment counts match the CHR dashboards?

SHW's clinical leadership asked for exactly this check: compare the assistant's results with the
existing CHR dashboards, and be able to see how each number was counted. Three steps:

  totals(con)            the app's counts by month and service, under several definitions
  template(con)          a CSV for staff to type the dashboard's figures into
  compare(con, filled)   which of the app's definitions the dashboard matches, and where it differs
  rules(con)             in plain words, how the app counts an appointment

A dashboard and the app can both be right and still differ, because "appointments" can mean
booked, booked and not cancelled, or attended. compare() says which meaning fits the dashboard.
"""
from __future__ import annotations

import pandas as pd

from .ingest import load_dictionary

DEFINITIONS = {
    "booked": ("every appointment booked, whatever happened to it", ["attended", "no_show", "cancelled", "other"]),
    "not_cancelled": ("booked appointments that were not cancelled", ["attended", "no_show", "other"]),
    "attended": ("appointments the student attended", ["attended"]),
}
MEASURES = {"appointments": "Appointments shown on the CHR dashboard",
            "students": "Unique students (patients) shown on the CHR dashboard"}
CLOSE = 0.01          # within 1% counts as close (dashboards and exports are pulled at different moments)


def totals(con) -> pd.DataFrame:
    """One row per month and service (plus 'All services'), with every way of counting."""
    df = con.execute("SELECT strftime(start_time, '%Y-%m') AS month, service_line AS service, outcome, student_key "
                     "FROM appointments WHERE start_time IS NOT NULL").df()
    if df.empty:
        return pd.DataFrame()
    both = pd.concat([df, df.assign(service="All services")], ignore_index=True)
    rows = []
    for (month, service), g in both.groupby(["month", "service"]):
        row = {"month": month, "service": service}
        for name, (_, outcomes) in DEFINITIONS.items():
            part = g[g["outcome"].isin(outcomes)]
            row[f"appointments_{name}"] = int(len(part))
            row[f"students_{name}"] = int(part["student_key"].nunique())
        row["no_shows"] = int((g["outcome"] == "no_show").sum())
        row["cancelled"] = int((g["outcome"] == "cancelled").sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["service", "month"]).reset_index(drop=True)


def template(con, months=6) -> pd.DataFrame:
    """Rows for the latest `months` months, with an empty column for the dashboard's figure."""
    t = totals(con)
    if t.empty:
        return t
    keep = sorted(t["month"].unique())[-int(months):]
    t = t[t["month"].isin(keep)]
    rows = [{"month": r.month, "service": r.service, "measure": m, "dashboard_value": ""}
            for r in t.itertuples() for m in MEASURES]
    return pd.DataFrame(rows)


def compare(con, filled: pd.DataFrame) -> dict:
    """Match the dashboard's figures against each of the app's definitions."""
    t = totals(con).set_index(["month", "service"])
    f = filled.copy()
    f["dashboard_value"] = pd.to_numeric(f["dashboard_value"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
    f = f.dropna(subset=["dashboard_value"])
    f = f[f["measure"].isin(MEASURES)]
    out = {"rows_compared": int(len(f)), "measures": {}, "not_in_app": []}
    for measure, part in f.groupby("measure"):
        scores, detail = {}, {}
        for name in DEFINITIONS:
            exact = close = 0
            rows = []
            for r in part.itertuples():
                if (r.month, r.service) not in t.index:
                    continue
                app = int(t.loc[(r.month, r.service), f"{measure}_{name}"])
                diff = app - r.dashboard_value
                pct = diff / r.dashboard_value if r.dashboard_value else (0.0 if not diff else float("inf"))
                exact += diff == 0
                close += abs(pct) <= CLOSE
                rows.append({"month": r.month, "service": r.service, "dashboard": int(r.dashboard_value), "app": app,
                             "difference": int(diff), "difference_pct": round(100 * pct, 1)})
            scores[name] = (exact, close, len(rows))
            detail[name] = rows
        best = max(scores, key=lambda k: (scores[k][1], scores[k][0]))
        exact, close, n = scores[best]
        out["measures"][measure] = {
            "best_definition": best, "meaning": DEFINITIONS[best][0], "compared": n, "exact": exact, "within_1pct": close,
            "all_definitions": {k: {"exact": v[0], "within_1pct": v[1]} for k, v in scores.items()},
            "differences": [r for r in detail[best] if abs(r["difference_pct"]) > 100 * CLOSE],
        }
    seen = set(t.index)
    out["not_in_app"] = sorted({f"{r.month} {r.service}" for r in f.itertuples() if (r.month, r.service) not in seen})
    return out


def rules(con) -> list[str]:
    """How the app counts, in plain words, including anything it left out when loading."""
    cfg = load_dictionary()["chr"]
    n = con.execute("SELECT COUNT(*), COUNT(DISTINCT student_key), MIN(start_time)::DATE, MAX(start_time)::DATE "
                    "FROM appointments").fetchone()
    out = [f"The app holds {n[0]:,} appointments for {n[1]:,} students, from {n[2]} to {n[3]}.",
           "One row in the CHR export is one appointment. If the same appointment appears in more than one export, "
           "the newest copy is kept, so nothing is counted twice.",
           f"Attended means the appointment's status is one of: {', '.join(cfg['attended'])}.",
           f"No-show means the status is one of: {', '.join(cfg['no_show'])}.",
           f"Cancelled means the status is one of: {', '.join(cfg['cancelled'])}.",
           "The no-show rate is no-shows divided by all booked appointments, cancelled ones included.",
           "A student is counted once per month and service, however many appointments they had. Student and "
           "appointment numbers are replaced with codes when the data is loaded.",
           "A month is the calendar month the appointment started in."]
    try:
        q = con.execute("SELECT issue, detail, count FROM data_quality WHERE issue LIKE 'chr_%'").df()
    except Exception:
        q = pd.DataFrame()
    plain = {"chr_non_appointment_rows": "rows in the export that are not appointments (such as group-visit templates) were skipped",
             "chr_duplicate_appointments": "appointments that appeared more than once were counted once",
             "chr_unmapped_status": "appointments have a status the app does not recognise, so they count as booked but as "
                                    "neither attended, no-show nor cancelled",
             "chr_missing_columns": "expected columns were missing from the export"}
    for r in q.itertuples():
        count = "" if pd.isna(r.count) else f"{int(r.count):,} "
        out.append(f"When loading: {count}{plain.get(r.issue, r.issue)}." + (f" ({r.detail})" if r.issue in ("chr_unmapped_status", "chr_missing_columns") else ""))
    return out


def report(result: dict, rule_lines: list[str]) -> str:
    """A Markdown reconciliation report."""
    lines = ["# Reconciliation with the CHR dashboards", ""]
    if not result["rows_compared"]:
        lines += ["No dashboard figures were entered, so there is nothing to compare yet.", ""]
    for measure, m in result["measures"].items():
        lines += [f"## {MEASURES[measure]}", "",
                  f"The dashboard's figure matches the app when the app counts **{m['meaning']}**: "
                  f"{m['exact']} of {m['compared']} figures match exactly and {m['within_1pct']} are within 1%.", "",
                  "| Way of counting | Exact matches | Within 1% |", "|---|---|---|"]
        lines += [f"| {DEFINITIONS[k][0]} | {v['exact']} | {v['within_1pct']} |" for k, v in m["all_definitions"].items()]
        if m["differences"]:
            lines += ["", "Figures that still differ by more than 1%:", "",
                      "| Month | Service | Dashboard | App | Difference |", "|---|---|---|---|---|"]
            lines += [f"| {r['month']} | {r['service']} | {r['dashboard']:,} | {r['app']:,} | {r['difference']:+,} ({r['difference_pct']:+g}%) |"
                      for r in m["differences"]]
        else:
            lines += ["", "No figure differs by more than 1%."]
        lines.append("")
    if result.get("not_in_app"):
        lines += ["Months or services on the dashboard that the app has no data for: " + ", ".join(result["not_in_app"]) + ".", ""]
    lines += ["## How the app counts", ""] + [f"- {r}" for r in rule_lines]
    lines += ["", "## If figures differ", "",
              "- The dashboard may filter by provider group, so appointments with providers who have left are missing from it.",
              "- The export and the dashboard may have been pulled on different days.",
              "- The dashboard may group appointment types into services differently from the app.",
              "- An appointment status may be missing from the lists above (see `chr:` in `dictionary.yaml`)."]
    return "\n".join(lines) + "\n"

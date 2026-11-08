"""Care-pattern tools on CHR appointment records: things a dashboard cannot show because
they need each student's sequence of appointments.

care_journeys      What happens after a student's first visit: single visits, quick returns,
                   time to a second visit, and whether students come back after a missed one.
noshow_patterns    Which kinds of appointments are missed more often, tested with students
                   as the unit of evidence (one student's appointments are not independent).
service_pathways   Whether students who start in one service go on to use another, e.g.
                   medical clinic to counselling, which the integrated care model aims for.

All three describe patterns, not causes, and say so in their notes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config, stats
from .analysis import _appts

STUDENT_DIMENSIONS = ["faculty", "academic_level", "gender", "age_band", "program"]
LOOKBACK_DAYS = 90     # no visit in the 90 days before = a new episode of care
MIN_GAP_PTS = 5.0      # a group gap smaller than this is not reported, however significant
MIN_NOSHOW_GAP = 3.0

SINGLE_VISIT_NOTE = (
    "A single visit is not necessarily a problem: with single-session or stepped care, one "
    "visit is often the plan. The records cannot tell planned one-visit care from follow-up that "
    "did not happen, so treat group differences as questions for the clinical team, not as "
    "dropout rates.")


def _load(con, service_line=None, term=None):
    df = _appts(con, service_line, term)
    df = df.assign(start_time=pd.to_datetime(df["start_time"]))
    return df.sort_values(["student_key", "start_time"]).reset_index(drop=True)


def _new_episodes(att, data_start, data_end, follow_days):
    """First new episode of care per student: an attended visit with no attended visit in the
    LOOKBACK_DAYS before it, where the data covers that lookback and `follow_days` after it.
    Students first seen early in the data may be part-way through care, so they are left out."""
    att = att.sort_values(["student_key", "start_time"])
    gap = att.groupby("student_key")["start_time"].diff().dt.days
    new = gap.isna() | (gap > LOOKBACK_DAYS)
    lo = data_start + pd.Timedelta(days=LOOKBACK_DAYS)
    hi = data_end - pd.Timedelta(days=follow_days)
    starts = att[new & att["start_time"].between(lo, hi)]
    return starts.groupby("student_key", as_index=False).first()


def _group_tests(table, dimension, metrics):
    """Each group vs. everyone else on each 0/1 metric, BH-corrected across all tests.
    table: one row per student with `group` and 0/1 metric columns."""
    rows = []
    for g, sub in table.groupby("group"):
        rest = table[table["group"] != g]
        row = {"group": g, "students": int(len(sub))}
        for col, label in metrics.items():
            x1, n1 = int(sub[col].sum()), len(sub)
            x2, n2 = int(rest[col].sum()), len(rest)
            gap, p = stats.two_proportion_test(x1, n1, x2, n2)
            row[f"{label}_pct"] = round(100 * x1 / n1, 1)
            row[f"{label}_gap_pts"] = round(gap, 1)
            row[f"_p_{label}"] = p
        rows.append(row)
    pcols = [f"_p_{lab}" for lab in metrics.values()]
    flat = [r[c] for r in rows for c in pcols]
    q = iter(stats.bh_adjust(flat))
    flags = []
    for r in rows:
        for lab in metrics.values():
            r[f"{lab}_q"] = round(float(next(q)), 4)
            if r[f"{lab}_q"] < 0.05 and abs(r[f"{lab}_gap_pts"]) >= MIN_GAP_PTS:
                flags.append({"group": r["group"], "measure": lab, "group_pct": r[f"{lab}_pct"],
                              "gap_vs_others_pts": r[f"{lab}_gap_pts"], "q": r[f"{lab}_q"],
                              "students": r["students"]})
        for c in pcols:
            r.pop(c)
    flags.sort(key=lambda f: -abs(f["gap_vs_others_pts"]))
    return rows, flags, len(flat)


def care_journeys(con, service_line: str = None, dimension: str = None,
                  return_days: int = 30, follow_days: int = 60):
    """What happens after a student's first visit in a new episode of care."""
    if dimension and dimension not in STUDENT_DIMENSIONS:
        raise ValueError(f"dimension must be one of: {', '.join(STUDENT_DIMENSIONS)}")
    df = _load(con, service_line)
    start, end = df["start_time"].min().normalize(), df["start_time"].max().normalize()
    att = df[df["outcome"] == "attended"]
    eps = _new_episodes(att, start, end, follow_days)
    if len(eps) < config.MIN_CELL:
        raise ValueError(f"Only {len(eps)} new episodes of care have {follow_days} days of "
                         "follow-up in the data; too few to describe.")
    m = att.merge(eps[["student_key", "start_time"]].rename(columns={"start_time": "t0"}),
                  on="student_key")
    m = m[(m["start_time"] >= m["t0"]) & (m["start_time"] <= m["t0"] + pd.Timedelta(days=follow_days))]
    visits = m.groupby("student_key").size()
    second = m[m["start_time"] > m["t0"]].groupby("student_key")["start_time"].min()
    eps = eps.set_index("student_key")
    eps["visits"] = visits.reindex(eps.index).fillna(1).astype(int)
    eps["days_to_second"] = (second.reindex(eps.index) - eps["start_time"]).dt.days
    eps["single_visit"] = (eps["visits"] == 1).astype(int)
    eps["returned_quickly"] = (eps["days_to_second"] <= return_days).astype(int)

    # Coming back after a missed appointment
    ns = df[df["outcome"] == "no_show"]
    ns = ns[ns["start_time"] <= end - pd.Timedelta(days=follow_days)]
    first_ns = ns.groupby("student_key")["start_time"].min().rename("t_ns")
    later = att.merge(first_ns.reset_index(), on="student_key")
    later = later[(later["start_time"] > later["t_ns"]) &
                  (later["start_time"] <= later["t_ns"] + pd.Timedelta(days=follow_days))]
    recovered = later["student_key"].nunique()

    d2 = eps["days_to_second"].dropna()
    out = {
        "service_line": service_line or "all",
        "period": f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
        "episodes_start_between": f"{start + pd.Timedelta(days=LOOKBACK_DAYS):%Y-%m-%d} and "
                                  f"{end - pd.Timedelta(days=follow_days):%Y-%m-%d}",
        "students": int(len(eps)),
        "single_visit_pct": round(100 * eps["single_visit"].mean(), 1),
        f"returned_within_{return_days}_days_pct": round(100 * eps["returned_quickly"].mean(), 1),
        "median_days_to_second_visit": None if d2.empty else float(d2.median()),
        "missed_then_came_back": {
            "students_with_a_missed_appointment": int(len(first_ns)),
            f"attended_within_{follow_days}_days_pct":
                None if len(first_ns) == 0 else round(100 * recovered / len(first_ns), 1)},
    }
    if dimension:
        t = eps.reset_index().rename(columns={dimension: "group"})   # the student's group at the start
        t, excluded = stats.drop_unknown(t, "group")
        t = stats.pool_small(t.assign(group=t["group"].astype(str)), "group")
        rows, flags, n_tests = _group_tests(
            t, dimension, {"single_visit": "single_visit", "returned_quickly": "returned_quickly"})
        out.update({"dimension": dimension, "groups": rows, "flags": flags,
                    "tests_run": n_tests, "excluded_unknown": excluded})
    out["notes"] = [
        f"Each student counts once: their first new episode of care (no attended visit in the "
        f"{LOOKBACK_DAYS} days before), followed for {follow_days} days. Students first seen "
        "early in the data are left out because they may be part-way through care, and late "
        "starts are left out because they have not had time to return.",
        SINGLE_VISIT_NOTE,
        "Groups are compared with everyone else, with a correction for running many tests; "
        f"only gaps of at least {MIN_GAP_PTS:.0f} points are flagged."]
    return out


def _cluster_prop(y, cluster):
    """Proportion and its variance treating each student as one unit of evidence."""
    d = pd.DataFrame({"y": y, "c": cluster}).groupby("c")["y"].agg(["sum", "size"])
    n, k = d["size"].sum(), len(d)
    p = d["sum"].sum() / n
    if k < 2:
        return p, np.nan
    resid = d["sum"] - p * d["size"]
    return p, float(k / (k - 1) * (resid ** 2).sum() / n ** 2)


def _time_of_day(ts):
    h = ts.dt.hour
    return pd.Series(np.select([h < 10, h < 12, h < 15], ["Before 10am", "10am-noon", "Noon-3pm"],
                               default="3pm or later"), index=ts.index)


def noshow_patterns(con, service_line: str = None, term: str = None):
    """Which kinds of appointments are missed more often than others."""
    df = _load(con, service_line, term)
    if len(df) < config.MIN_CELL:
        raise ValueError("Too few appointments to look for patterns.")
    y = (df["outcome"] == "no_show").astype(int)
    prev = df.groupby("student_key")["outcome"]
    prior_ns = prev.transform(lambda s: (s == "no_show").astype(int).cumsum().shift(fill_value=0))
    prior_any = df.groupby("student_key").cumcount()
    factors = {
        "Booking history": pd.Series(np.select(
            [prior_any == 0, prior_ns > 0], ["First booking in the records", "Has missed one before"],
            default="Has attended before, never missed"), index=df.index),
        "Weekday": df["start_time"].dt.day_name(),
    }
    if df["start_time"].dt.hour.nunique() > 1:
        factors["Time of day"] = _time_of_day(df["start_time"])
    for col, label in [("mode", "Appointment format"), ("visit_type", "Visit type"),
                       ("service_line", "Service"), ("academic_level", "Level of study"),
                       ("faculty", "Faculty"), ("age_band", "Age group"), ("gender", "Gender")]:
        if col in df and not (service_line and col == "service_line"):
            factors[label] = df[col]

    overall = float(y.mean())
    levels = []
    for label, values in factors.items():
        f = pd.DataFrame({"g": values, "y": y, "c": df["student_key"]})
        f, _ = stats.drop_unknown(f, "g")
        f = stats.pool_small(f.assign(g=f["g"].astype(str)), "g", min_n=max(config.MIN_CELL, 30))
        if f["g"].nunique() < 2:
            continue
        for g, sub in f.groupby("g"):
            rest = f[f["g"] != g]
            p1, v1 = _cluster_prop(sub["y"].to_numpy(), sub["c"].to_numpy())
            p0, v0 = _cluster_prop(rest["y"].to_numpy(), rest["c"].to_numpy())
            se = np.sqrt(v1 + v0)
            diff = 100 * (p1 - p0)
            pval = float(2 * norm.sf(abs((p1 - p0) / se))) if se and np.isfinite(se) else np.nan
            levels.append({"factor": label, "group": g, "appointments": int(len(sub)),
                           "students": int(sub["c"].nunique()), "no_show_pct": round(100 * p1, 1),
                           "gap_vs_others_pts": round(diff, 1),
                           "ci_pts": [round(diff - 196 * se, 1), round(diff + 196 * se, 1)],
                           "_p": pval})
    q = stats.bh_adjust([r["_p"] for r in levels])
    for r, qv in zip(levels, q):
        r["q"] = None if np.isnan(qv) else round(float(qv), 4)
        r.pop("_p")
    flags = [r for r in levels if r["q"] is not None and r["q"] < 0.05
             and abs(r["gap_vs_others_pts"]) >= MIN_NOSHOW_GAP]
    flags.sort(key=lambda r: -abs(r["gap_vs_others_pts"]))
    notes = [
        "Each appointment type is compared with all other appointments. Students are the unit "
        "of evidence (someone who often misses counts once, not once per appointment), and the "
        "results are corrected for running many tests. Only gaps of at least "
        f"{MIN_NOSHOW_GAP:.0f} points are flagged.",
        "Covers every appointment from the dates in 'period' (not only the current term); "
        "say so when reporting these results next to a term or same-dates comparison.",
        "These are patterns, not causes: e.g. a higher rate for one format may reflect who books "
        "that format.",
        "Booking lead time (days between booking and the appointment) is one of the strongest "
        "predictors of missed appointments in published studies, but these records have no "
        "booking date, so it cannot be checked."]
    return {"service_line": service_line or "all", "term": term or "all",
            "period": f"{df['start_time'].min():%Y-%m-%d} to {df['start_time'].max():%Y-%m-%d}",
            "appointments": int(len(df)), "students": int(df["student_key"].nunique()),
            "overall_no_show_pct": round(100 * overall, 1), "flags": flags,
            "tests_run": len(levels), "all_groups": levels, "notes": notes}


def service_pathways(con, within_days: int = 30):
    """Of students starting a new episode in one service, how many go on to another service
    within `within_days` days."""
    df = _load(con)
    start, end = df["start_time"].min().normalize(), df["start_time"].max().normalize()
    att = df[df["outcome"] == "attended"]
    if att["service_line"].nunique() < 2:
        raise ValueError("The records cover only one service, so there are no pathways to show.")
    rows = []
    for svc, part in att.groupby("service_line"):
        eps = _new_episodes(part, start, end, within_days)
        if len(eps) < config.MIN_CELL:
            continue
        m = att.merge(eps[["student_key", "start_time"]].rename(columns={"start_time": "t0"}),
                      on="student_key")
        m = m[(m["service_line"] != svc) & (m["start_time"] > m["t0"]) &
              (m["start_time"] <= m["t0"] + pd.Timedelta(days=within_days))]
        for other in sorted(att["service_line"].unique()):
            if other == svc:
                continue
            k = m.loc[m["service_line"] == other, "student_key"].nunique()
            lo, hi = _wilson(k, len(eps))
            rows.append({"from": svc, "to": other, "students_starting": int(len(eps)),
                         "went_on_within_days": int(k), "pct": round(100 * k / len(eps), 1),
                         "ci_pct": [lo, hi]})
    if not rows:
        raise ValueError("Too few new episodes in each service to describe pathways.")
    rows.sort(key=lambda r: -r["pct"])
    return {"within_days": within_days, "period": f"{start:%Y-%m-%d} to {end:%Y-%m-%d}",
            "pathways": rows,
            "notes": [
                f"A new episode starts with an attended visit after {LOOKBACK_DAYS} days without "
                f"one in that service; starts are followed for {within_days} days. The range "
                "(ci_pct) is a 95% interval for each percentage.",
                "This shows movement between services, not whether it was a referral or the "
                "right care."]}


def _wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(100 * (centre - half), 1), round(100 * (centre + half), 1)

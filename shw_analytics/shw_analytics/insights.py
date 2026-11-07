"""Insight tools: findings staff cannot get by filtering a dashboard.

what_changed  splits a change between two periods into "who came" (mix) and
              "change within groups" (experience). CHR appointment data for now; survey
              ratings once there is more than one survey period.

Every result says how the change splits, never why it happened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, stats
from .analysis import CHR_DIMENSIONS, _appts, _per_student

METRICS = {
    "no_show_rate": {"unit": "appointment", "label": "No-show rate (% of booked appointments)"},
    "visits_per_student": {"unit": "student", "label": "Attended visits per student seen"},
    "days_in_care": {"unit": "student", "label": "Days in care (first to last attended visit)"},
}
# Student-level metrics need a dimension that describes the student, not the appointment.
STUDENT_DIMENSIONS = ["faculty", "academic_level", "gender", "age_band", "program"]

VERDICTS = {
    "who_came": "Mostly a change in who came: the mix of {dim} groups shifted, while the "
                "measure did not clearly change within groups.",
    "within_groups": "A change within {dim} groups: it is not explained by a change in who came.",
    "both": "Both contributed: the mix of {dim} groups shifted AND the measure changed within groups.",
    "offsetting": "Hidden change: the shift in the mix of {dim} groups and the change within "
                  "groups pulled in opposite directions, so the overall number understates "
                  "what changed within groups.",
    "no_clear_change": "No clear change: neither part is distinguishable from noise.",
}
NOTE = ("This splits the change into who came vs. change within groups, by {dim} only. It does "
        "not show why either happened, and shifts in things not recorded here count as "
        "'within groups'.")


def _windows(dates: pd.Series, compare: str, window_days: int):
    """Latest `window_days` of data vs. the same dates a year earlier, or the window just
    before. Matching windows keeps a part-finished term from looking like a drop."""
    end = dates.max().normalize()
    start = end - pd.Timedelta(days=window_days - 1)
    if compare == "last_year":
        before = (start - pd.DateOffset(years=1), end - pd.DateOffset(years=1))
    elif compare == "previous":
        before = (start - pd.Timedelta(days=window_days), start - pd.Timedelta(days=1))
    else:
        raise ValueError("compare must be 'last_year' or 'previous'.")
    first = dates.min().normalize()
    if first > before[0]:
        raise ValueError(f"The data starts on {first:%Y-%m-%d}, so there is no full comparison "
                         f"window for {before[0]:%Y-%m-%d} to {before[1]:%Y-%m-%d}. "
                         "Try compare='previous' or a smaller window_days.")
    return before, (start, end)


def _units(df: pd.DataFrame, metric: str, dimension: str) -> pd.DataFrame:
    """One row per unit (appointment or student) with period, group and value."""
    if METRICS[metric]["unit"] == "appointment":
        return pd.DataFrame({"student_key": df["student_key"].to_numpy(),
                             "period": df["period"].to_numpy(),
                             "group": df[dimension].to_numpy(),
                             "value": (df["outcome"] == "no_show").to_numpy(float) * 100})
    col = "visits" if metric == "visits_per_student" else "days_in_care"
    parts = []
    for period, part in df.groupby("period"):
        per = _per_student(part)
        per = per[per["visits"] > 0]                      # students actually seen
        attr = part.groupby("student_key")[dimension].first()
        parts.append(pd.DataFrame({"student_key": per.index.to_numpy(), "period": period,
                                   "group": attr.reindex(per.index).to_numpy(),
                                   "value": per[col].to_numpy(float)}))
    return pd.concat(parts, ignore_index=True)


def _sig(lo, hi):
    return lo > 0 or hi < 0


def _verdict(mix, exp):
    (m, mlo, mhi), (e, elo, ehi) = mix, exp
    if _sig(mlo, mhi) and _sig(elo, ehi):
        return "offsetting" if np.sign(m) != np.sign(e) else "both"
    if _sig(elo, ehi):
        return "within_groups"
    if _sig(mlo, mhi):
        return "who_came"
    return "no_clear_change"


# Calendar for term names like "2025 Fall". Real CHR exports may name terms differently;
# unrecognised names fall back to comparing each term's length with the others.
TERM_MONTHS = {"winter": (1, 4), "spring": (1, 4), "summer": (5, 8), "fall": (9, 12), "autumn": (9, 12)}
SLACK_DAYS = 7


def _expected_span(term: str):
    parts = str(term).replace("-", " ").split()
    year = next((int(x) for x in parts if x.isdigit() and len(x) == 4), None)
    season = next((x.lower() for x in parts if x.lower() in TERM_MONTHS), None)
    if year is None or season is None:
        return None, None
    m0, m1 = TERM_MONTHS[season]
    return pd.Timestamp(year, m0, 1), pd.Timestamp(year, m1, 1) + pd.offsets.MonthEnd(0)


def term_coverage(con, service_line: str = None):
    """Which dates each term actually covers, and whether the data covers the whole term.

    A term is partial when the data starts or ends more than a week inside it, e.g. the
    current term before it has finished. Comparing a partial term with a complete one makes
    everything that builds up over time (students, visits, days in care) look lower."""
    df = _appts(con, service_line)
    df = df.assign(start_time=pd.to_datetime(df["start_time"]))
    cov = df.groupby("term")["start_time"].agg(first_day="min", last_day="max").reset_index()
    cov["first_day"], cov["last_day"] = cov["first_day"].dt.normalize(), cov["last_day"].dt.normalize()
    spans = (cov["last_day"] - cov["first_day"]).dt.days
    typical = spans.median()
    slack = pd.Timedelta(days=SLACK_DAYS)
    rows = []
    for (_, r), span in zip(cov.iterrows(), spans):
        start, end = _expected_span(r["term"])
        if start is not None:
            partial = bool(r["first_day"] > start + slack or r["last_day"] < end - slack)
            expected = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
        else:   # unknown naming: much shorter than the typical term
            partial = bool(len(cov) > 1 and span < 0.8 * typical)
            expected = None
        rows.append({"term": r["term"], "data_from": f"{r['first_day']:%Y-%m-%d}",
                     "data_to": f"{r['last_day']:%Y-%m-%d}", "expected": expected, "partial": partial})
    return sorted(rows, key=lambda x: x["data_from"])


def partial_term_warning(con, terms, service_line: str = None):
    """A warning string if any of `terms` is only partly covered by the data, else None."""
    terms = [t for t in ([terms] if isinstance(terms, str) else (terms or [])) if t]
    if not terms:
        return None
    bad = [c for c in term_coverage(con, service_line) if c["term"] in terms and c["partial"]]
    if not bad:
        return None
    detail = "; ".join(f"{c['term']} has data only from {c['data_from']} to {c['data_to']}" for c in bad)
    return (f"PARTIAL TERM: {detail}. Do not compare it with a complete term: totals, visits and "
            "days in care will look lower only because less time is covered. For change "
            "between periods use what_changed, which compares matching date windows.")


def what_changed(con, metric: str, dimension: str, service_line: str = None,
                 compare: str = "last_year", window_days: int = 61):
    """Is a change in `metric` because of who came, or a change within groups?"""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric '{metric}'. Use one of: {', '.join(METRICS)}")
    unit = METRICS[metric]["unit"]
    allowed = STUDENT_DIMENSIONS if unit == "student" else [d for d in CHR_DIMENSIONS if d != "term"]
    if service_line:
        allowed = [d for d in allowed if d != "service_line"]
    if dimension not in allowed:
        raise ValueError(f"For {metric}, dimension must be one of: {', '.join(allowed)}")

    df = _appts(con, service_line)
    df = df.assign(start_time=pd.to_datetime(df["start_time"]))
    (a0, a1), (b0, b1) = _windows(df["start_time"], compare, int(window_days))
    day = df["start_time"].dt.normalize()
    period = np.select([day.between(a0, a1), day.between(b0, b1)], ["before", "after"], default="")
    df = df.assign(period=period)
    df = df[df["period"] != ""]

    volume = {}
    for per in ("before", "after"):
        part = df[df["period"] == per]
        att = part[part["outcome"] == "attended"]
        volume[per] = {"booked": int(len(part)), "attended": int(len(att)),
                       "students_seen": int(att["student_key"].nunique())}
    units = _units(df, metric, dimension)
    units, excluded = stats.drop_unknown(units, "group")
    units = units.assign(group=units["group"].astype(str))
    original = set(units["group"])
    units = stats.pool_small(units, "group", by="period")
    pooled = sorted(original - set(units["group"]))
    n = units.groupby("period").size().reindex(["before", "after"], fill_value=0)
    if (n < config.MIN_CELL).any():
        raise ValueError(f"Too little data: {int(n['before'])} before and {int(n['after'])} after "
                         f"(need at least {config.MIN_CELL} in each).")
    groups = sorted(units["group"].unique(), key=lambda g: (g == "Other", g))

    def stat(d):
        k = stats.kitagawa(d, "group", "value", "period", "before", "after", groups=groups)
        return [k["total"], k["mix"], k["experience"], *k["groups"]["experience"]]

    k = stats.kitagawa(units, "group", "value", "period", "before", "after", groups=groups)
    bs = stats.bootstrap(units, stat, strata="period", cluster="student_key")
    est, lo, hi = bs["estimate"], bs["low"], bs["high"]
    part = lambda i: {"estimate": round(est[i], 2), "ci": [round(lo[i], 2), round(hi[i], 2)]}  # noqa: E731
    verdict = _verdict((est[1], lo[1], hi[1]), (est[2], lo[2], hi[2]))

    table = k["groups"]
    rows = []
    for j, g in enumerate(groups):
        r = table.loc[g]
        rows.append({"group": g, "n_before": int(r["n_before"]), "n_after": int(r["n_after"]),
                     "share_before_pct": round(100 * r["share_before"], 1),
                     "share_after_pct": round(100 * r["share_after"], 1),
                     "before": None if pd.isna(r["mean_before"]) else round(r["mean_before"], 2),
                     "after": None if pd.isna(r["mean_after"]) else round(r["mean_after"], 2),
                     "mix_contribution": round(r["mix"], 2),
                     "within_group_contribution": round(est[3 + j], 2),
                     "within_group_ci": [round(lo[3 + j], 2), round(hi[3 + j], 2)]})
    rows.sort(key=lambda r: -abs(r["within_group_contribution"]))

    notes = [NOTE.format(dim=dimension)]
    notes.append("volume_in_same_windows gives bookings, attendances and students seen for the "
                 "SAME matched windows; quote these, not whole-term totals, when describing change.")
    if compare == "previous":
        notes.append("Compared with the window just before, so normal seasonal patterns "
                     "(e.g. exam periods) may look like change.")
    unit_name = "appointments" if unit == "appointment" else "students"
    return {
        "metric": metric, "measure": METRICS[metric]["label"], "dimension": dimension,
        "service_line": service_line or "all",
        "periods": {"before": f"{a0:%Y-%m-%d} to {a1:%Y-%m-%d}",
                    "after": f"{b0:%Y-%m-%d} to {b1:%Y-%m-%d}"},
        "n": {"before": int(n["before"]), "after": int(n["after"]), "unit": unit_name},
        "volume_in_same_windows": volume,
        "overall": {"before": round(k["before"], 2), "after": round(k["after"], 2),
                    "change": part(0)["estimate"], "ci": part(0)["ci"]},
        "mix_effect": part(1),
        "within_group_effect": part(2),
        "verdict": verdict,
        "explanation": VERDICTS[verdict].format(dim=dimension),
        "groups": rows,
        "excluded_unknown": excluded,
        "pooled_into_other": pooled,
        "notes": notes,
    }

"""Has the survey changed over time?

survey_changed(con, service, dimension, compare, window_days) compares survey results in the
latest window with the same dates a year earlier (or the window just before), three ways:

  services   each service's % agree, with the range of the change
  questions  which questions moved, tested and corrected for checking them all
  groups     which student groups rate a service differently from before, and whether everyone
             else moved the same way (so a general shift is not mistaken for a group's)

"% agree" for a student is the share of the questions they answered that they rated 4 or 5;
each student counts once. For one question it is the share of students who rated it 4 or 5.

The survey is anonymous, so the two windows hold different students. A change can come from
who answered as well as from a change in experience; every result says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config, stats
from .insights import _windows

MIN_PERIOD = 30          # respondents a service needs in each window before it is compared
MIN_GROUP = 10           # respondents a group needs in each window
ITEM_FLAG_PTS = 8        # a question is flagged when it moves at least this much, and is tested
GROUP_FLAG_PTS = 10
ALPHA = 0.05
DIMENSIONS = ["disability_any", "intl_or_domestic", "academic_level", "first_generation", "indigenous",
              "racialized", "gender", "trans_experience"]
NOTE = ("The survey is anonymous, so the two windows hold different students. A change can come from "
        "who chose to answer as well as from a change in experience. A pattern, not a cause.")


def _welch(a, b):
    """Change in mean (b minus a) in percentage points, its 95% range, and a p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = 100 * (b.mean() - a.mean())
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = 100 * np.sqrt(va + vb)
    if not se:
        return diff, [diff, diff], 1.0, se
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = sps.t.ppf(0.975, df)
    return diff, [diff - t * se, diff + t * se], float(2 * sps.t.sf(abs(diff / se), df)), se


def _r(x):
    return round(float(x), 1)


def _fmt(d):
    return f"{pd.Timestamp(d):%Y-%m-%d}"


def survey_changed(con, service: str = None, dimension: str = None, compare: str = "last_year",
                   window_days: int = 61):
    cols = set(con.execute("DESCRIBE ratings_demo").df()["column_name"])
    dims = [dimension] if dimension else [d for d in DIMENSIONS if d in cols]
    if dimension and dimension not in cols:
        raise ValueError(f"'{dimension}' is not a group this survey records. Try one of: "
                         + ", ".join(d for d in DIMENSIONS if d in cols))
    keep = ", ".join(["response_id", "service", "item_code", "item_text", "score", "recorded_at"] + dims)
    df = con.execute(f"SELECT {keep} FROM ratings_demo WHERE recorded_at IS NOT NULL AND score IS NOT NULL").df()
    if df.empty:
        raise ValueError("No dated survey responses are loaded.")
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    (a0, a1), (b0, b1) = _windows(df["recorded_at"], compare, int(window_days))
    day = df["recorded_at"].dt.normalize()
    df["period"] = np.where((day >= a0) & (day <= a1), "before", np.where((day >= b0) & (day <= b1), "after", ""))
    df = df[df["period"] != ""].copy()
    df["agree"] = (df["score"] >= 4).astype(float)
    if service:
        if service not in set(df["service"]):
            raise ValueError(f"No responses about '{service}' in these windows. Services: "
                             + ", ".join(sorted(set(df['service']))))
        df = df[df["service"] == service]

    services, item_tests, group_tests, too_small, skipped = [], [], [], 0, []
    for svc, d in df.groupby("service"):
        resp = d.groupby(["response_id", "period"] + dims, dropna=False)["agree"].mean().reset_index()
        before, after = resp[resp["period"] == "before"], resp[resp["period"] == "after"]
        if min(len(before), len(after)) < MIN_PERIOD:
            skipped.append({"service": svc, "respondents_before": int(len(before)), "respondents_after": int(len(after))})
            continue
        diff, ci, p, _ = _welch(before["agree"], after["agree"])
        services.append({
            "service": svc, "respondents_before": int(len(before)), "respondents_after": int(len(after)),
            "pct_agree_before": _r(100 * before["agree"].mean()), "pct_agree_after": _r(100 * after["agree"].mean()),
            "change_pts": _r(diff), "ci_pts": [_r(ci[0]), _r(ci[1])],
            "verdict": stats.verdict(ci[0], ci[1], "higher than before", "lower than before", "no clear change")})

        for (code, text), it in d.groupby(["item_code", "item_text"]):
            x = it.groupby("period")["agree"].agg(["sum", "count"])
            if not {"before", "after"} <= set(x.index) or x["count"].min() < MIN_GROUP:
                continue
            change, p = stats.two_proportion_test(x.loc["after", "sum"], x.loc["after", "count"],
                                                  x.loc["before", "sum"], x.loc["before", "count"])
            item_tests.append({"service": svc, "item_code": code, "item": text,
                               "pct_agree_before": _r(100 * x.loc["before", "sum"] / x.loc["before", "count"]),
                               "pct_agree_after": _r(100 * x.loc["after", "sum"] / x.loc["after", "count"]),
                               "students_before": int(x.loc["before", "count"]), "students_after": int(x.loc["after", "count"]),
                               "change_pts": _r(change), "_p": p})

        for dim in dims:
            for grp, g in resp.dropna(subset=[dim]).groupby(dim):
                if str(grp).strip().lower() in stats.UNKNOWN:
                    continue
                gb, ga = g[g["period"] == "before"]["agree"], g[g["period"] == "after"]["agree"]
                if min(len(gb), len(ga)) < max(MIN_GROUP, config.MIN_CELL):
                    too_small += 1
                    continue
                rest = resp[resp[dim].astype(str) != str(grp)]
                rb, ra = rest[rest["period"] == "before"]["agree"], rest[rest["period"] == "after"]["agree"]
                diff, ci, p, se = _welch(gb, ga)
                row = {"service": svc, "dimension": dim, "group": str(grp),
                       "respondents_before": int(len(gb)), "respondents_after": int(len(ga)),
                       "pct_agree_before": _r(100 * gb.mean()), "pct_agree_after": _r(100 * ga.mean()),
                       "change_pts": _r(diff), "ci_pts": [_r(ci[0]), _r(ci[1])], "_p": p, "_se": se}
                if min(len(rb), len(ra)) >= MIN_GROUP:
                    rdiff, _, _, rse = _welch(rb, ra)
                    gap, gse = diff - rdiff, float(np.sqrt(se ** 2 + rse ** 2))
                    row.update(everyone_else_change_pts=_r(rdiff), change_vs_everyone_else_pts=_r(gap),
                               differs_from_everyone_else=bool(abs(gap) > 1.96 * gse))
                group_tests.append(row)

    # How big a change a typical group would need before this data could detect it (80% power).
    detectable = _r(2.8 * float(np.median([t["_se"] for t in group_tests]))) if group_tests else None
    for tests in (item_tests, group_tests):
        for t, q in zip(tests, stats.bh_adjust([t["_p"] for t in tests])):
            t["q"] = round(float(q), 4)
            t.pop("_p")
            t.pop("_se", None)
    item_flags = sorted((t for t in item_tests if t["q"] < ALPHA and abs(t["change_pts"]) >= ITEM_FLAG_PTS),
                        key=lambda t: -abs(t["change_pts"]))
    # A group is listed only when it moved differently from other students. If everyone moved
    # together, that is the service's change (above), not the group's.
    group_flags = sorted((t for t in group_tests if t["q"] < ALPHA and abs(t["change_pts"]) >= GROUP_FLAG_PTS
                          and t.get("differs_from_everyone_else", True)),
                         key=lambda t: -abs(t["change_pts"]))
    out = {
        "periods": {"before": f"{_fmt(a0)} to {_fmt(a1)}", "after": f"{_fmt(b0)} to {_fmt(b1)}"},
        "compare": compare, "service": service, "dimension": dimension,
        "services": services,
        "questions_that_changed": item_flags, "questions_checked": len(item_tests),
        "groups_that_changed": group_flags, "groups_checked": len(group_tests), "groups_too_small_to_check": too_small,
        "tests_run": len(item_tests) + len(group_tests) + len(services),
        "smallest_group_change_detectable_pts": detectable,
        "verdict": ("changes_found" if item_flags or group_flags or any(s["verdict"] != "no clear change" for s in services)
                    else "no_clear_change"),
        "notes": [NOTE,
                  f"A question is listed when it moved at least {ITEM_FLAG_PTS} points and the move held up after "
                  f"correcting for all {len(item_tests)} questions checked; a group, at least {GROUP_FLAG_PTS} points "
                  f"after correcting for all {len(group_tests)} groups checked.",
                  "A group is listed only if it also moved differently from other students over the same dates; "
                  "a shift shared by everyone shows up in the service's own change instead."],
    }
    if detectable:
        out["notes"].append(f"With the number of responses in these windows, a typical group's rating would need to "
                            f"move by about {detectable:g} points before it could be told apart from chance. Smaller "
                            "shifts may be real and still not show up; a longer window (window_days) gives more responses.")
    if service and item_tests:
        out["all_questions"] = sorted(item_tests, key=lambda t: t["change_pts"])
    if skipped:
        out["services_with_too_few_responses"] = skipped
    if too_small:
        out["notes"].append(f"{too_small} groups had fewer than {max(MIN_GROUP, config.MIN_CELL)} respondents in one "
                            "of the windows and could not be checked. That is a limit of the data, not evidence of no change.")
    return out

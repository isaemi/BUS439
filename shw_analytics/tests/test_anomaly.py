"""Tests for the weekly anomaly screen (shw_analytics/anomaly_detection.py) and its chart.

Run:  python tests/test_anomaly.py      (or: pytest tests/test_anomaly.py)
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import figures  # noqa: E402
from shw_analytics.anomaly_detection import detect_chr_anomalies, overall_level  # noqa: E402

SHAPE = [60, 70, 90, 100, 110, 120, 115, 105]      # appointments per week in the earlier year


def _term(year, per_week, no_show=0.06, providers=12, seed=0):
    """One Fall term: per_week[i] appointments in week i+1, spread over the week."""
    rng = np.random.default_rng(seed + year)
    rows = []
    for w, n in enumerate(per_week):
        for k in range(int(n)):
            day = pd.Timestamp(year, 9, 1) + pd.Timedelta(days=7 * w + k % 7, hours=9 + k % 8)
            rows.append({"start_time": day, "service_line": "Medical", "provider_key": f"p{k % providers}",
                         "outcome": "no_show" if rng.random() < no_show else "attended"})
    return pd.DataFrame(rows)


def _two_years(this_year, **kw):
    return pd.concat([_term(2025, SHAPE), _term(2026, this_year, **kw)], ignore_index=True)


def _flags(result, metric="appointment_volume"):
    a = result["anomalies"]
    return sorted(a.loc[a["metric"] == metric, "week"].tolist())


def test_busier_term_is_one_finding_not_many_weeks():
    df = _two_years([round(1.4 * n) for n in SHAPE])
    plain = detect_chr_anomalies(df, term="Fall", year=2026, adjust_for_level=False)
    assert len(_flags(plain)) == len(SHAPE)                    # every week 40% up: all flagged
    r = detect_chr_anomalies(df, term="Fall", year=2026)
    assert _flags(r) == []
    level = r["overall_level"]["appointment_volume"]
    assert level["applied"] and level["notable"] and abs(level["factor"] - 1.4) < 0.03
    assert abs(level["percent_vs_earlier_years"] - 40) < 3


def test_a_spike_on_top_of_a_busier_term_is_still_found():
    weeks = [round(1.4 * n) for n in SHAPE]
    weeks[4] = round(weeks[4] * 1.8)
    r = detect_chr_anomalies(_two_years(weeks), term="Fall", year=2026)
    assert _flags(r) == [5]
    f = r["anomalies"].iloc[0]
    assert f["direction"] == "spike" and f["historical_baseline"] > f["unadjusted_baseline"]
    assert abs(f["level_factor"] - 1.4) < 0.03


def test_a_drop_is_found_too():
    weeks = [round(1.4 * n) for n in SHAPE]
    weeks[2] = round(weeks[2] * 0.5)
    r = detect_chr_anomalies(_two_years(weeks), term="Fall", year=2026)
    assert _flags(r) == [3] and r["anomalies"].iloc[0]["direction"] == "drop"


def test_same_level_as_last_year_changes_nothing():
    weeks = list(SHAPE)
    weeks[5] = round(weeks[5] * 1.6)
    df = _two_years(weeks)
    a = detect_chr_anomalies(df, term="Fall", year=2026)
    b = detect_chr_anomalies(df, term="Fall", year=2026, adjust_for_level=False)
    assert _flags(a) == _flags(b) == [6]
    assert not a["overall_level"]["appointment_volume"]["notable"]


def test_too_few_weeks_means_no_adjustment_and_says_so():
    df = _two_years([round(1.4 * n) for n in SHAPE[:3]])
    r = detect_chr_anomalies(df, term="Fall", year=2026)
    level = r["overall_level"]["appointment_volume"]
    assert not level["applied"] and "fewer than" in level["reason_not_applied"]
    assert _flags(r) == [1, 2, 3]                              # falls back to the plain comparison
    assert "was not allowed for" in r["note"]


def test_each_count_has_its_own_level_and_rates_are_left_alone():
    df = _two_years([round(1.4 * n) for n in SHAPE])
    r = detect_chr_anomalies(df, term="Fall", year=2026)
    assert r["overall_level"]["active_practitioners"]["factor"] == 1.0   # same 12 providers both years
    assert set(r["overall_level"]) == {"appointment_volume", "active_practitioners"}
    base = r["baseline"]
    rates = base[base["metric"].str.endswith("_rate")]
    assert (rates["baseline"] == rates["unadjusted_baseline"]).all()


def test_overall_level_is_a_median_so_unusual_weeks_do_not_move_it():
    assert overall_level([14, 14, 14, 14, 50], [10, 10, 10, 10, 10]) == 1.4
    assert overall_level([14, 14], [10, 10]) is None
    assert overall_level([14, 14, 14, 14], [0, 10, 10, 10]) is None      # only three comparable weeks


def _tool_result(weeks, metric=None, selected=None):
    """What tools.chr_anomalies returns, built from the detector (no database needed)."""
    r = detect_chr_anomalies(_two_years(weeks), term="Fall", year=2026)
    n = list(range(1, r["complete_weeks"] + 1))
    cur = r["weekly"].loc[r["weekly"]["year"] == 2026].set_index("week")
    series = {}
    for m in ([metric] if metric else ["appointment_volume", "active_practitioners", "no_show_rate", "cancellation_rate"]):
        b = r["baseline"].loc[r["baseline"]["metric"] == m].set_index("week")["baseline"].reindex(n)
        series[m] = {"this_year": [round(float(v), 1) for v in cur[m].reindex(n)],
                     "baseline": [None if pd.isna(v) else round(float(v), 1) for v in b]}
    flags = [{"week": int(f.week), "metric": f.metric, "direction": f.direction, "observed": float(f.observed),
              "baseline": float(f.historical_baseline)} for f in r["anomalies"].itertuples(index=False)
             if not metric or f.metric == metric]
    return {"series": series, "weeks": n, "week_starts": [str((pd.Timestamp(2026, 9, 1) + pd.Timedelta(days=7 * (w - 1))).date()) for w in n],
            "flagged_in_term": flags, "overall_level": r["overall_level"], "thresholds": r["thresholds"],
            "historical_years": r["historical_years"], "term": "Fall", "year": 2026, "selected_week": selected,
            "service_scope": "Medical", "data_source": "synthetic CHR export"}


def _chart(result, args=None):
    store = {}
    ids = figures.register_from_trace([("chr_anomalies", args or {}, result)], store)
    assert len(ids) == 1
    return store[ids[0]]


def test_chart_marks_the_unusual_week_and_names_the_overall_level():
    weeks = [round(1.4 * n) for n in SHAPE]
    weeks[4] = round(weeks[4] * 1.8)
    g = _chart(_tool_result(weeks))
    assert "1 of 8 weeks stand out" in g.title and "40% above Fall 2025" in g.title
    marks = [t for t in g.fig.data if t.name == "Unusual week"]
    assert len(marks) == 1 and list(marks[0].x) == [5] and "Unusual spike" in marks[0].text[0]
    names = {t.name for t in g.fig.data}
    assert {"Fall 2026", "Baseline from Fall 2025"} <= names
    assert "Week 5 (29 Sep to 5 Oct)" in g.caption and "Synthetic test data" in g.caption
    assert g.weight == 3


def test_chart_with_nothing_unusual_says_so():
    g = _chart(_tool_result([round(1.4 * n) for n in SHAPE], metric="appointment_volume"))
    assert "no week stands out" in g.title and "40% above" in g.title
    assert not [t for t in g.fig.data if t.name == "Unusual week"]
    assert g.weight == 2


def test_chart_for_this_week_talks_about_the_latest_week():
    weeks = [round(1.4 * n) for n in SHAPE]
    weeks[4] = round(weeks[4] * 1.8)
    g = _chart(_tool_result(weeks, selected=8))
    assert "latest complete week (20 Oct to 26 Oct) does not stand out" in g.title and "1 earlier week this term does" in g.title


def test_chart_text_has_no_internal_names():
    weeks = [round(1.4 * n) for n in SHAPE]
    weeks[4] = round(weeks[4] * 1.8)
    g = _chart(_tool_result(weeks))
    text = " ".join([g.title, g.scope, g.caption] + [str(t.name) for t in g.fig.data]
                    + [a.text for a in g.fig.layout.annotations])
    assert not re.search(r"\b(no_show_rate|cancellation_rate|appointment_volume|active_practitioners|chr_anomalies|q_value|p_value)\b", text)
    assert not re.search(r"\bp\s*[<=]|q\s*[<=]|Benjamini", text)


def test_a_small_service_is_not_flagged_for_ordinary_swings():
    last, this = [4, 6, 5, 7, 4, 6, 5, 8], [6, 4, 8, 4, 7, 3, 8, 5]         # 50% swings, a handful of appointments
    df = pd.concat([_term(2025, last), _term(2026, this)], ignore_index=True)
    without = detect_chr_anomalies(df, term="Fall", year=2026, count_min_appointments=0)
    assert len(_flags(without)) >= 3                                         # the old behaviour: flagged most weeks
    r = detect_chr_anomalies(df, term="Fall", year=2026)
    assert _flags(r) == [] and _flags(r, "active_practitioners") == []
    assert r["thresholds"]["count_min_appointments"] == 10 and "at least 10 appointments" in r["note"]


def test_a_busy_week_falling_to_almost_nothing_is_still_flagged():
    weeks = list(SHAPE)
    weeks[3] = 3                                                             # 100 last year, 3 this year
    r = detect_chr_anomalies(_two_years(weeks), term="Fall", year=2026)
    assert _flags(r) == [4] and r["anomalies"].iloc[0]["direction"] == "drop"


def test_weeks_with_a_handful_of_appointments_are_hidden_but_empty_weeks_are_not():
    from shw_analytics import tools
    weekly = pd.DataFrame({"year": [2025, 2025, 2025, 2026, 2026, 2026], "week": [1, 2, 3, 1, 2, 3],
                           "appointment_volume": [30, 2, 40, 3, 0, 50], "cancellations": [3, 1, 4, 1, 0, 5],
                           "no_shows": [2, 0, 3, 1, 0, 4], "active_practitioners": [6, 1, 7, 1, 0, 8],
                           "cancellation_rate": [10.0, 50.0, 10.0, 33.3, None, 10.0],
                           "no_show_rate": [6.7, 0.0, 7.5, 33.3, None, 8.0], "term": "Fall", "service": "Wellness Advising"})
    series = {"appointment_volume": {"this_year": [3.0, 0.0, 50.0], "baseline": [30.0, 2.0, 40.0]},
              "no_show_rate": {"this_year": [33.3, None, 8.0], "baseline": [6.7, 5.0, 7.5]}}
    out, shown, hidden = tools._hide_small_weeks(weekly, series, [1, 2, 3], [30.0, 2.0, 40.0], min_cell=5)
    assert hidden == 1                                                       # week 1 of this year (3 appointments)
    assert "cancellations" not in out.columns and "no_shows" not in out.columns
    blank = out[out["hidden"]]
    assert sorted(zip(blank["year"], blank["week"])) == [(2025, 2), (2026, 1)]
    for col in ("appointment_volume", "active_practitioners", "cancellation_rate", "no_show_rate"):
        assert blank[col].isna().all()
    empty = out[(out["year"] == 2026) & (out["week"] == 2)].iloc[0]
    assert empty["appointment_volume"] == 0 and not empty["hidden"]          # nothing happened: shown
    assert shown["appointment_volume"] == {"this_year": [None, 0.0, 50.0], "baseline": [30.0, None, 40.0]}
    assert shown["no_show_rate"] == {"this_year": [None, None, 8.0], "baseline": [6.7, None, 7.5]}
    assert not any(v is not None and 0 < v < 5 for s in shown.values() for v in s["this_year"] + s["baseline"]
                   if s is shown["appointment_volume"])


def test_chart_says_when_weeks_are_left_blank():
    r = _tool_result([round(1.4 * n) for n in SHAPE], metric="appointment_volume")
    r["series"]["appointment_volume"]["this_year"][0] = None
    r["hidden_weeks"], r["min_cell"] = 1, 5
    g = _chart(r)
    assert "1 week with fewer than 5 appointments" in g.caption


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

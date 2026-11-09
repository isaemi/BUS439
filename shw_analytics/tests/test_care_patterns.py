"""Tests for shw_analytics/care_patterns.py. Each test plants a known pattern in a small fake
appointments table and checks the tool finds it (and the windows/exclusions behave).

Run:  python tests/test_care_patterns.py      (or: pytest tests/test_care_patterns.py)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import care_patterns as cp  # noqa: E402

BASE = {"faculty": "Arts", "gender": "Woman", "age_band": "18-24", "program": "BA",
        "visit_type": "Follow-up", "mode": "In person", "term": "2025 Fall",
        "academic_level": "Undergraduate", "service_line": "Counselling"}


def _row(student, day, outcome="attended", **kw):
    return {**BASE, "student_key": student, "start_time": pd.Timestamp(day), "outcome": outcome, **kw}


def _con(rows, first="2025-01-01", last="2025-12-31"):
    # Two "pin" students fix the data's first and last day; the windows exclude them.
    rows = rows + [_row("pin_a", first), _row("pin_b", last)]
    con = duckdb.connect()
    con.register("appointments", pd.DataFrame(rows))
    return con


def _days(n, lo="2025-04-15", hi="2025-10-01", seed=0):
    rng = np.random.default_rng(seed)
    span = (pd.Timestamp(hi) - pd.Timestamp(lo)).days
    return [pd.Timestamp(lo) + pd.Timedelta(days=int(d)) for d in rng.integers(0, span, n)]


def _journey_rows():
    rows = []
    for i, d in enumerate(_days(100, seed=1)):          # graduate: always a 2nd visit 10 days on
        rows += [_row(f"g{i}", d, academic_level="Graduate"),
                 _row(f"g{i}", d + pd.Timedelta(days=10), academic_level="Graduate")]
    for i, d in enumerate(_days(200, seed=2)):          # undergraduate: one visit only
        rows.append(_row(f"u{i}", d))
    rows.append(_row("early", "2025-01-20"))            # inside the lookback: excluded
    rows.append(_row("late", "2025-12-20"))             # no time to return: excluded
    return rows


def test_care_journeys_overall_figures_and_exclusions():
    r = cp.care_journeys(_con(_journey_rows()))
    assert r["students"] == 300                          # early, late and pins left out
    assert r["single_visit_pct"] == 66.7
    assert r["returned_within_30_days_pct"] == 33.3
    assert r["median_days_to_second_visit"] == 10.0
    assert r["period"] == "2025-01-01 to 2025-12-31"


def test_care_journeys_flags_the_group_that_differs():
    r = cp.care_journeys(_con(_journey_rows()), dimension="academic_level")
    flagged = {(f["group"], f["measure"]) for f in r["flags"]}
    assert ("Graduate", "single_visit") in flagged and ("Undergraduate", "single_visit") in flagged
    grad = next(g for g in r["groups"] if g["group"] == "Graduate")
    assert grad["single_visit_pct"] == 0.0 and grad["returned_quickly_pct"] == 100.0
    assert r["tests_run"] == 4


def test_care_journeys_no_difference_means_no_flags():
    rows = []
    for i, d in enumerate(_days(300, seed=3)):
        lvl = "Graduate" if i % 3 == 0 else "Undergraduate"
        rows.append(_row(f"s{i}", d, academic_level=lvl))
        if i % 2 == 0:
            rows.append(_row(f"s{i}", d + pd.Timedelta(days=7), academic_level=lvl))
    r = cp.care_journeys(_con(rows), dimension="academic_level")
    assert r["flags"] == []


def test_missed_then_came_back():
    rows = []
    for i in range(20):
        rows += [_row(f"r{i}", "2025-06-01", "no_show"), _row(f"r{i}", "2025-06-15")]
    for i in range(10):
        rows.append(_row(f"x{i}", "2025-06-01", "no_show"))
    r = cp.care_journeys(_con(rows))["missed_then_came_back"]
    assert r["students_with_a_missed_appointment"] == 30
    assert r["attended_within_60_days_pct"] == 66.7


def test_care_journeys_rejects_appointment_level_dimension():
    try:
        cp.care_journeys(_con(_journey_rows()), dimension="mode")
    except ValueError as e:
        assert "academic_level" in str(e)
    else:
        raise AssertionError("expected ValueError")


def _noshow_rows(video_pct, in_person_pct, seed=0):
    """400 students x 4 appointments; half always book phone/video. No-shows assigned exactly."""
    rng = np.random.default_rng(seed)
    rows = []
    for mode, pct, ids in [("Phone/Video", video_pct, range(200)), ("In person", in_person_pct, range(200, 400))]:
        missed = np.zeros(800, bool)
        missed[: round(800 * pct / 100)] = True
        rng.shuffle(missed)
        k = 0
        for i in ids:
            for d in _days(4, seed=int(i) + 10):
                rows.append(_row(f"s{i}", d, "no_show" if missed[k] else "attended", mode=mode))
                k += 1
    return rows


def test_noshow_patterns_finds_the_planted_format_gap():
    r = cp.noshow_patterns(_con(_noshow_rows(20, 5)))
    fmt = [f for f in r["flags"] if f["factor"] == "Appointment format"]
    video = next(f for f in fmt if f["group"] == "Phone/Video")
    assert video["no_show_pct"] == 20.0 and video["gap_vs_others_pts"] == 15.0
    assert video["ci_pts"][0] > 0
    assert r["overall_no_show_pct"] == 12.5
    assert r["period"] == "2025-01-01 to 2025-12-31"


def test_noshow_patterns_no_planted_gap_no_format_flag():
    r = cp.noshow_patterns(_con(_noshow_rows(10, 10)))
    assert not [f for f in r["flags"] if f["factor"] == "Appointment format"]


def test_cluster_variance_is_larger_when_students_repeat_their_behaviour():
    # 50 students miss all 4 bookings, 450 attend all: strongly clustered.
    y = np.array([1] * 200 + [0] * 1800)
    c = np.repeat(np.arange(500), 4)
    p, v = cp._cluster_prop(y, c)
    naive = p * (1 - p) / len(y)
    assert p == 0.1 and v > 3 * naive


def test_service_pathways_medical_to_counselling():
    rows = []
    for i, d in enumerate(_days(200, seed=4)):
        rows.append(_row(f"m{i}", d, service_line="Medical"))
        if i < 60:
            rows.append(_row(f"m{i}", d + pd.Timedelta(days=14), service_line="Counselling"))
    for i, d in enumerate(_days(100, seed=5)):
        rows.append(_row(f"c{i}", d))
    r = cp.service_pathways(_con(rows))
    path = next(p for p in r["pathways"] if p["from"] == "Medical" and p["to"] == "Counselling")
    assert path["students_starting"] == 200 and path["pct"] == 30.0
    assert path["ci_pct"][0] < 30.0 < path["ci_pct"][1]
    back = next(p for p in r["pathways"] if p["from"] == "Counselling" and p["to"] == "Medical")
    assert back["pct"] == 0.0


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", type(e).__name__, e)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

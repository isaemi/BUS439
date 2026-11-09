"""Tests for shw_analytics/insights.py. Each test plants a known effect in a small fake
appointments table and checks that the tool recovers it, and only it.

Run:  python tests/test_insights.py      (or: pytest tests/test_insights.py)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import insights, stats  # noqa: E402

stats.N_BOOT = 300   # enough for stable tests, faster than the app's 1000

WINDOWS = {"before": ("2025-09-01", "2025-10-31"), "after": ("2026-09-01", "2026-10-31")}


def _con(plan, appts_per_student=3, seed=0):
    """plan: {period: {group: (students, no_show_pct)}}. No-shows are assigned exactly
    (not randomly), so within-group rates are exactly as planted."""
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    for period, groups in plan.items():
        lo, hi = (pd.Timestamp(d) for d in WINDOWS[period])
        span = (hi - lo).days
        for group, (students, pct) in groups.items():
            total = students * appts_per_student
            missed = np.zeros(total, bool)
            missed[: round(total * pct / 100)] = True
            rng.shuffle(missed)
            for i in range(total):
                if i % appts_per_student == 0:
                    sid += 1
                rows.append({"student_key": f"s{sid}", "academic_level": group,
                             "faculty": "Arts", "gender": "Woman", "age_band": "18-24",
                             "program": "BA", "service_line": "Counselling", "visit_type": "Follow-up",
                             "mode": "In person", "term": period,
                             "start_time": lo + pd.Timedelta(days=int(rng.integers(0, span + 1))),
                             "outcome": "no_show" if missed[i] else "attended"})
    df = pd.DataFrame(rows)
    # Pin the data's first and last day so the windows line up exactly.
    if "before" in plan:
        df.loc[0, "start_time"] = pd.Timestamp(WINDOWS["before"][0])
    if "after" in plan:
        df.loc[len(df) - 1, "start_time"] = pd.Timestamp(WINDOWS["after"][1])
    con = duckdb.connect()
    con.register("appointments", df)
    return con


def _no_show(con, **kw):
    return insights.what_changed(con, "no_show_rate", "academic_level", **kw)


def test_mix_only_change_is_attributed_to_who_came():
    # Grad share 20% -> 50%, rates unchanged (UG 10%, Grad 25%): overall 13 -> 17.5.
    con = _con({"before": {"Undergraduate": (400, 10), "Graduate": (100, 25)},
                "after": {"Undergraduate": (250, 10), "Graduate": (250, 25)}})
    r = _no_show(con)
    assert r["verdict"] == "who_came", r["verdict"]
    assert abs(r["overall"]["change"] - 4.5) < 0.1
    assert abs(r["within_group_effect"]["estimate"]) < 0.1
    assert r["periods"] == {"before": "2025-09-01 to 2025-10-31", "after": "2026-09-01 to 2026-10-31"}


def test_within_group_change_is_found_and_credited_to_the_right_group():
    # Same 80/20 mix; UG no-shows 10% -> 16%, Grad unchanged: +4.8 within groups.
    con = _con({"before": {"Undergraduate": (400, 10), "Graduate": (100, 25)},
                "after": {"Undergraduate": (400, 16), "Graduate": (100, 25)}})
    r = _no_show(con)
    assert r["verdict"] == "within_groups", r["verdict"]
    assert abs(r["within_group_effect"]["estimate"] - 4.8) < 0.1
    assert r["groups"][0]["group"] == "Undergraduate"


def test_offsetting_changes_are_not_reported_as_no_change():
    # Overall 13 -> 13, but mix pushes up ~+5 and within-group change pulls down ~-5.
    con = _con({"before": {"Undergraduate": (400, 10), "Graduate": (100, 25)},
                "after": {"Undergraduate": (250, 4), "Graduate": (250, 22)}})
    r = _no_show(con)
    assert abs(r["overall"]["change"]) < 0.1
    assert r["verdict"] == "offsetting", r["verdict"]


def test_nothing_planted_means_no_clear_change():
    plan = {"Undergraduate": (400, 12), "Graduate": (100, 12)}
    r = _no_show(_con({"before": plan, "after": plan}))
    assert r["verdict"] == "no_clear_change", r["verdict"]


def test_group_missing_from_one_period_is_pooled_and_reported():
    con = _con({"before": {"Undergraduate": (400, 10), "Graduate": (100, 25), "Postdoc": (20, 50)},
                "after": {"Undergraduate": (400, 10), "Graduate": (100, 25)}})
    r = _no_show(con)
    assert "Postdoc" in r["pooled_into_other"]
    assert "Postdoc" not in [g["group"] for g in r["groups"]]


def test_student_metric_rejects_appointment_level_dimension():
    con = _con({"before": {"Undergraduate": (50, 10)}, "after": {"Undergraduate": (50, 10)}})
    try:
        insights.what_changed(con, "visits_per_student", "mode")
    except ValueError as e:
        assert "academic_level" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_visits_per_student_mix_shift():
    # Grad students attend 6 appointments, UG 2 (no no-shows); Grad share 20% -> 50%.
    con_ug = _con({"before": {"Undergraduate": (200, 0)}, "after": {"Undergraduate": (125, 0)}},
                  appts_per_student=2)
    con_gr = _con({"before": {"Graduate": (50, 0)}, "after": {"Graduate": (125, 0)}},
                  appts_per_student=6, seed=1)
    df = pd.concat([con_ug.execute("SELECT * FROM appointments").df(),
                    con_gr.execute("SELECT * FROM appointments").df().assign(
                        student_key=lambda d: "g" + d["student_key"])])
    con = duckdb.connect()
    con.register("appointments", df)
    r = insights.what_changed(con, "visits_per_student", "academic_level")
    assert r["verdict"] == "who_came", r["verdict"]
    assert abs(r["overall"]["before"] - 2.8) < 0.01 and abs(r["overall"]["after"] - 4.0) < 0.01


def test_missing_comparison_window_gives_a_clear_error():
    con = _con({"after": {"Undergraduate": (100, 10)}})
    try:
        _no_show(con)
    except ValueError as e:
        assert "data starts" in str(e)
    else:
        raise AssertionError("expected ValueError")


def _term_con(spans):
    """spans: {term: (first_day, last_day)} -> one appointment on each of those two days."""
    rows = [{"student_key": f"s{i}", "term": term, "start_time": pd.Timestamp(day),
             "service_line": "Counselling", "outcome": "attended", "academic_level": "Undergraduate"}
            for i, (term, days) in enumerate(spans.items()) for day in days]
    con = duckdb.connect()
    con.register("appointments", pd.DataFrame(rows))
    return con


def test_term_in_progress_is_flagged_partial():
    # The real situation: Fall 2025 complete, Fall 2026 only up to Oct 31.
    con = _term_con({"2025 Fall": ("2025-09-01", "2025-12-31"), "2026 Winter": ("2026-01-01", "2026-04-30"),
                     "2026 Fall": ("2026-09-01", "2026-10-31")})
    cov = {c["term"]: c["partial"] for c in insights.term_coverage(con)}
    assert cov == {"2025 Fall": False, "2026 Winter": False, "2026 Fall": True}


def test_term_with_data_starting_late_is_flagged_partial():
    con = _term_con({"2025 Fall": ("2025-11-15", "2025-12-31"), "2026 Winter": ("2026-01-01", "2026-04-30")})
    cov = {c["term"]: c["partial"] for c in insights.term_coverage(con)}
    assert cov == {"2025 Fall": True, "2026 Winter": False}


def test_unrecognised_term_names_fall_back_to_length():
    con = _term_con({"T1": ("2025-01-01", "2025-04-30"), "T2": ("2025-05-01", "2025-08-31"),
                     "T3": ("2025-09-01", "2025-10-01")})
    cov = {c["term"]: c["partial"] for c in insights.term_coverage(con)}
    assert cov == {"T1": False, "T2": False, "T3": True}


def test_partial_term_warning_names_the_term_and_points_to_what_changed():
    con = _term_con({"2025 Fall": ("2025-09-01", "2025-12-31"), "2026 Fall": ("2026-09-01", "2026-10-31")})
    w = insights.partial_term_warning(con, ["2025 Fall", "2026 Fall"])
    assert "2026 Fall" in w and "2025 Fall" not in w and "what_changed" in w
    assert insights.partial_term_warning(con, "2025 Fall") is None
    assert insights.partial_term_warning(con, None) is None


def test_what_changed_reports_volume_for_the_same_windows():
    con = _con({"before": {"Undergraduate": (400, 10), "Graduate": (100, 25)},
                "after": {"Undergraduate": (250, 10), "Graduate": (250, 25)}})
    v = _no_show(con)["volume_in_same_windows"]
    assert v["before"]["booked"] == 1500 and v["after"]["booked"] == 1500
    assert v["before"]["attended"] == 1500 - round(1200 * 0.10) - round(300 * 0.25)
    # Students seen = at least one attended visit (someone who missed every booking wasn't seen).
    full = _con({"before": {"Undergraduate": (400, 0), "Graduate": (100, 0)},
                 "after": {"Undergraduate": (250, 0), "Graduate": (250, 0)}})
    assert _no_show(full)["volume_in_same_windows"]["before"]["students_seen"] == 500


def test_volume_change_detects_more_students():
    # 300 students last fall, 500 this fall, same rates: a tested increase of about 67%.
    con = _con({"before": {"Undergraduate": (300, 0)}, "after": {"Undergraduate": (500, 0)}})
    v = _no_show(con)["volume_change"]
    assert v["verdict"] == "increase" and abs(v["change_pct"] - 66.7) < 0.1
    assert v["ci_pct"][0] > 0


def test_volume_change_same_students_is_no_clear_change():
    plan = {"Undergraduate": (400, 0)}
    v = _no_show(_con({"before": plan, "after": plan}))["volume_change"]
    assert v["verdict"] == "no clear change" and v["change_pct"] == 0.0


def test_volume_test_matches_independent_poisson_interval():
    # 509 vs 731 in equal windows: ratio 1.436, 95% CI exp(ln 1.436 +/- 1.96*sqrt(1/509+1/731)).
    v = insights._volume_test(509, 731, 61, 61)
    assert v["change_pct"] == 43.6 and v["ci_pct"] == [28.3, 60.8] and v["p"] < 0.001


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

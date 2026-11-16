"""Tests for shw_analytics/reconcile.py.   Run:  python tests/test_reconcile.py"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import reconcile  # noqa: E402


def _con():
    rows = []
    for month, n in (("2026-09", 40), ("2026-10", 50)):
        for i in range(n):
            outcome = "cancelled" if i % 10 == 0 else "no_show" if i % 10 == 1 else "attended"
            rows.append({"start_time": pd.Timestamp(f"{month}-05") + pd.Timedelta(hours=i), "outcome": outcome,
                         "service_line": "Counselling" if i % 2 else "Medical", "student_key": f"S{i % 25}"})
    con = duckdb.connect()
    con.register("a", pd.DataFrame(rows))
    con.execute("CREATE TABLE appointments AS SELECT * FROM a")
    con.execute("CREATE TABLE data_quality AS SELECT 'chr_duplicate_appointments' AS issue, 'kept the newest copy' AS detail, 3 AS count")
    return con


def test_totals_count_each_way_and_students_once():
    t = reconcile.totals(_con()).set_index(["month", "service"])
    sep = t.loc[("2026-09", "All services")]
    assert (sep["appointments_booked"], sep["appointments_not_cancelled"], sep["appointments_attended"]) == (40, 36, 32)
    assert sep["no_shows"] == 4 and sep["cancelled"] == 4 and sep["students_booked"] == 25
    assert t.loc[("2026-09", "Counselling"), "appointments_booked"] + t.loc[("2026-09", "Medical"), "appointments_booked"] == 40


def test_template_has_a_blank_to_fill_for_each_month_service_and_measure():
    sheet = reconcile.template(_con(), months=1)
    assert set(sheet["month"]) == {"2026-10"} and set(sheet["measure"]) == {"appointments", "students"}
    assert len(sheet) == 3 * 2 and (sheet["dashboard_value"] == "").all()


def test_compare_finds_which_way_of_counting_the_dashboard_uses():
    con = _con()
    t = reconcile.totals(con)
    filled = pd.DataFrame([{"month": r.month, "service": r.service, "measure": "appointments",
                            "dashboard_value": f"{r.appointments_attended:,}"} for r in t.itertuples()])
    r = reconcile.compare(con, filled)["measures"]["appointments"]
    assert r["best_definition"] == "attended" and r["exact"] == r["compared"] == len(t) and r["differences"] == []
    assert r["all_definitions"]["booked"]["exact"] == 0


def test_compare_lists_what_still_differs_and_ignores_blank_rows():
    con = _con()
    filled = reconcile.template(con, months=2)
    filled.loc[(filled.month == "2026-09") & (filled.service == "All services") & (filled.measure == "appointments"), "dashboard_value"] = "40"
    filled.loc[(filled.month == "2026-10") & (filled.service == "All services") & (filled.measure == "appointments"), "dashboard_value"] = "44"
    extra = pd.DataFrame([{"month": "2024-01", "service": "All services", "measure": "appointments", "dashboard_value": "9"}])
    result = reconcile.compare(con, pd.concat([filled, extra]))
    m = result["measures"]["appointments"]
    assert result["rows_compared"] == 3 and m["best_definition"] == "booked" and m["exact"] == 1
    assert m["differences"] == [{"month": "2026-10", "service": "All services", "dashboard": 44, "app": 50,
                                 "difference": 6, "difference_pct": 13.6}]
    assert result["not_in_app"] == ["2024-01 All services"]


def test_rules_and_report_are_in_plain_words():
    con = _con()
    rules = reconcile.rules(con)
    assert rules[0].startswith("The app holds 90 appointments for 25 students")
    assert any("counted once" in r and "3 " in r for r in rules) and any("no-show rate" in r for r in rules)
    text = reconcile.report(reconcile.compare(con, reconcile.template(con)), rules)
    assert "nothing to compare yet" in text and "## How the app counts" in text and "## If figures differ" in text


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1; print("FAIL", name, "->", type(e).__name__, str(e)[:400])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

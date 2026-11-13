"""Tests for shw_analytics/briefing.py, using a sample of real tool results.
Run:  python tests/test_briefing.py"""
import copy
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import briefing, tools  # noqa: E402

TRACE = [tuple(s) for s in json.loads((Path(__file__).parent / "fixtures" / "sample_trace.json").read_text())]


def _step(name, n=0):
    return copy.deepcopy([s for s in TRACE if s[0] == name][n])


def test_mirror_images_are_reported_once():
    found, same = briefing._from_noshow(_step("noshow_patterns"))        # Phone/Video up and In Person down
    assert len(found) == 1 and found[0]["headline"].startswith("Phone/Video appointments are missed")
    assert "more often" in found[0]["headline"] and same == []


def test_nothing_flagged_becomes_a_stayed_the_same_line():
    step = _step("noshow_patterns"); step[2]["flags"] = []
    assert briefing._from_noshow(step) == ([], ["Missed appointments: no kind of appointment is missed clearly more often than others."])
    step = _step("care_journeys"); step[2]["flags"] = []
    found, same = briefing._from_care(step)
    assert found == [] and same[0].startswith("Coming back after a first visit")
    step = _step("priority_items"); step[2].update(priorities=[], verdict="no_stable_priority")
    assert briefing._from_priorities(step)[1] == ["Counselling survey: no question is clearly more linked to the overall rating than the others."]


def test_change_findings_follow_the_verdict_and_volume_is_reported_once_per_service():
    seen = set()
    found, same = briefing._from_change(_step("what_changed"), seen)
    assert [f["kind"] for f in found] == ["volume"] and "more students seen" in found[0]["headline"]
    assert any("no clear change" in s for s in same)
    found2, _ = briefing._from_change(_step("what_changed", 1), seen)    # same service again
    assert not [f for f in found2 if f["kind"] == "volume"]
    step = _step("what_changed"); step[2]["verdict"] = "offsetting"
    step[2]["mix_effect"].update(estimate=0.3, ci=[0.1, 0.5]); step[2]["within_group_effect"].update(estimate=-0.3, ci=[-0.5, -0.1])
    found3, _ = briefing._from_change(step, set())
    assert any("looks flat overall, but changed within student groups" in f["headline"] for f in found3)


def test_every_finding_carries_figures_how_it_was_tested_and_a_next_question():
    steps = [("noshow", briefing._from_noshow(_step("noshow_patterns"))[0]), ("care", briefing._from_care(_step("care_journeys"))[0]),
             ("priority", briefing._from_priorities(_step("priority_items"))[0]), ("equity", briefing._from_equity(_step("equity_flags"))[0])]
    for name, found in steps:
        assert found, name
        for f in found:
            assert f["figures"] and f["tested"] and f["next"].endswith("?") and 0 < f["strength"] <= 1 and f["id"]
            assert briefing.graphic(f) is not None, name


def test_p_from_a_range():
    assert briefing._p_from_ci(0, [-1, 1]) == 1.0
    assert abs(briefing._p_from_ci(1.96, [0, 3.92]) - 0.05) < 0.001
    assert briefing._p_from_ci(5, [4, 6]) < 0.0001


def test_compare_marks_new_still_there_and_gone():
    b = {"findings": [{"id": "a", "headline": "A"}, {"id": "b", "headline": "B"}], "also": []}
    prev = {"findings": [{"id": "b", "headline": "B"}, {"id": "c", "headline": "C was here"}], "data_through": "2026-09-30"}
    c = briefing.compare(b, prev)
    assert c["status"] == {"a": "new", "b": "still there"} and c["gone"] == ["C was here"] and c["since"] == "2026-09-30"
    assert briefing.compare(b, None) == {"status": {}, "gone": [], "since": None}


def _fake_world(monkey):
    """A tiny database and canned tool results, so the whole battery runs without real data."""
    con = duckdb.connect()
    con.execute("CREATE TABLE appointments AS SELECT * FROM (VALUES "
                "('Counselling', 's1', TIMESTAMP '2026-10-31')) t(service_line, student_key, start_time)")
    con.execute("INSERT INTO appointments SELECT 'Counselling', 's' || i::VARCHAR, TIMESTAMP '2026-10-01' FROM range(300) t(i)")
    con.execute("CREATE TABLE chr_meta AS SELECT TRUE AS synthetic")
    con.execute("CREATE TABLE ratings AS SELECT 1 AS x")
    canned = {s[0]: s for s in TRACE}
    weak = copy.deepcopy(canned["equity_flags"])
    for f in weak[2]["flags"]:
        f["q"] = 0.5                                              # should not survive the overall correction

    def call(name, **args):
        step = copy.deepcopy(weak if name == "equity_flags" else canned[name])
        return (name, {k: v for k, v in args.items() if v is not None}, step[2])
    monkey(tools, "_con", lambda: con)
    monkey(tools, "_period", lambda: "2026-10-17 to 2026-10-26")
    monkey(briefing, "_call", call)


def test_whole_briefing_is_ranked_varied_and_corrected_together():
    saved = []

    def monkey(obj, name, value):
        saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)
    try:
        _fake_world(monkey)
        b = briefing._run_battery()
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)
    kinds = [f["kind"] for f in b["findings"]]
    assert 0 < len(b["findings"]) <= briefing.MAX_FINDINGS
    assert all(kinds.count(k) <= briefing.MAX_PER_KIND for k in set(kinds))
    assert kinds.count("volume") <= 1 and "equity" not in kinds + [f["kind"] for f in b["also"]]
    everything = b["findings"] + b["also"]                 # asked by level and by faculty: one story, not two
    assert sum(f["kind"] == "continuity" for f in everything) == 1
    assert "Also: " in next(f for f in everything if f["kind"] == "continuity")["detail"]
    assert "do not say all checks were corrected together" in briefing.for_chat(b)["note"]
    assert b["dropped_after_correction"] >= 1 and b["tests_run"] > 20 and b["synthetic"] is True
    strengths = [f["strength"] for f in b["findings"]]
    assert strengths == sorted(strengths, reverse=True)
    assert any("survey covers one period" in m for m in b["not_checked"])
    chat = briefing.for_chat(b)
    assert len(chat["findings"]) == len(b["findings"]) and "step" not in json.dumps(chat) and chat["stayed_the_same"]
    assert "synthetic test data" in chat["note"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1; print("FAIL", name, "->", type(e).__name__, str(e)[:300])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

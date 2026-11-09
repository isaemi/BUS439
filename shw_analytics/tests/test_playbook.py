"""Tests for shw_analytics/playbook.py and playbook.yaml.

Run:  python tests/test_playbook.py      (or: pytest tests/test_playbook.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import playbook  # noqa: E402

KNOWN_TOOLS = {"describe_data", "service_summary", "item_summary", "compare_groups",
               "group_vs_everyone", "chr_summary", "chr_trend", "explain_difference",
               "survey_vs_visits", "wait_time_vs_rating", "equity_flags", "search_comments",
               "comment_overview", "run_sql", "what_changed", "care_journeys", "noshow_patterns",
               "service_pathways", "insight_playbook", "priority_items"}


def test_every_entry_is_complete_and_ids_are_unique():
    cases = playbook.load()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)) and len(cases) >= 15
    assert all(c["status"] in playbook.STATUS_ORDER for c in cases)


def test_steps_only_use_tools_that_exist():
    bad = [(c["id"], s["tool"]) for c in playbook.load() for s in c["steps"] if s["tool"] not in KNOWN_TOOLS]
    assert not bad, bad


def test_needs_data_entries_say_what_data():
    for c in playbook.load():
        if c["status"] == "needs_data":
            assert c.get("data_gaps"), c["id"]


def test_questions_find_the_right_use_case():
    cases = {
        "Should we send extra reminders for virtual appointments? Which ones get missed?": "noshow_hotspots",
        "Do students come back after their first counselling visit?": "after_first_visit",
        "Are students moving from the medical clinic to counselling?": "integrated_care_pathways",
        "Have no-shows changed since last year?": "real_change_or_mix",
        "What should we know this term?": "digest",
        "Are international graduate students under-served?": "reaching_groups",
        "What happens after a student misses an appointment?": "missed_then_lost",
        "Is our counselling caseload high compared with other universities?": "capacity_caseload",
        "Why are we busier this year?": "demand_growth",
        "What should counselling focus on to improve students' overall experience?": "survey_weak_spots",
        "We scored lowest on booking. Is that what's hurting satisfaction most?": "survey_weak_spots",
        "Are first-generation students in particular faculties rating us lower?": "hidden_subgroups",
    }
    wrong = {q: playbook.match(q)[0]["id"] if playbook.match(q) else None
             for q, want in cases.items() if (playbook.match(q) or [{}])[0].get("id") != want}
    assert not wrong, wrong


def test_tool_returns_detail_or_a_topic_list():
    r = playbook.insight_playbook("Which appointments are missed most often?")
    assert r["use_cases"][0]["id"] == "noshow_hotspots" and r["use_cases"][0]["steps"]
    assert playbook.insight_playbook(topic_id="digest")["use_cases"][0]["id"] == "digest"
    assert "error" in playbook.insight_playbook(topic_id="nope")
    empty = playbook.insight_playbook("zzz qqq")
    assert empty["use_cases"] == [] and empty["topics"]


def test_starter_questions_come_from_ready_entries():
    qs = playbook.starter_questions()
    assert qs and all(isinstance(t, str) and isinstance(q, str) for t, q in qs)


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

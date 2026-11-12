"""Tests for shw_analytics/followups.py.   Run:  python tests/test_followups.py"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import followups, playbook  # noqa: E402

TRACE = [tuple(s) for s in json.loads((Path(__file__).parent / "fixtures" / "sample_trace.json").read_text())]


def _step(name):
    return copy.deepcopy(next(s for s in TRACE if s[0] == name))


def test_suggestions_follow_from_what_was_found():
    s = followups.suggest([_step("noshow_patterns")])
    assert s[0] == "Do students who miss an appointment come back?" and s[-1] == followups.SOURCES_Q
    step = _step("what_changed")                                  # students seen rose in the sample
    assert any("coming back for second visits" in q for q in followups.suggest([step], limit=4))
    step[2]["volume_change"]["verdict"] = "no clear change"
    step[2]["verdict"] = "no_clear_change"
    assert "What is the smallest change this data could detect?" in followups.suggest([step])


def test_a_priority_leads_to_what_students_say_about_it():
    step = _step("priority_items")
    s = followups.suggest([step])
    assert s[0].startswith("What are students saying about “" + step[2]["priorities"][0][:20])


def test_never_more_than_the_limit_and_never_a_question_already_asked():
    asked = ["Do students who miss an appointment come back?", "where did this come from"]
    s = followups.suggest([_step("noshow_patterns")], asked=asked)
    assert len(s) <= 3 and not {q.lower().rstrip("?") for q in s} & {a.lower().rstrip("?") for a in asked}
    assert len(followups.suggest(TRACE, limit=2)) == 2


def test_playbook_follow_ups_come_first():
    case = next(c for c in playbook.load() if c["id"] == "digest")
    s = followups.suggest([("insight_playbook", {}, {"use_cases": [case]}), _step("noshow_patterns")])
    assert s[0] == case["next"][0]


def test_nothing_to_suggest_when_there_was_no_analysis_or_it_failed():
    assert followups.suggest([]) == [] and followups.suggest(None) == []
    assert followups.suggest([("noshow_patterns", {}, {"error": "x"}), "junk"]) == []


def test_starters_come_from_ready_playbook_entries():
    s = followups.starters(4)
    assert len(s) == 4 and all(q.endswith("?") or q.endswith(".") for q in s)


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

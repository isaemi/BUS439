"""Tests for shw_analytics/consistency.py (no model needed).   Run:  python tests/test_consistency.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import consistency as c  # noqa: E402

VOL = ("what_changed", {"metric": "visits_per_student", "dimension": "academic_level"}, {"volume_change": {}})
PLAY = ("insight_playbook", {"question": "x"}, {"use_cases": []})


def _run(answer, trace, bad=()):
    return {"question": "q", "answer": answer, "trace": trace, "ungrounded": list(bad)}


def test_helpers_failed_steps_and_argument_order_do_not_affect_the_signature():
    a = c.signature([PLAY, VOL, ("noshow_patterns", {}, {"error": "x"})])
    b = c.signature([("what_changed", {"dimension": "academic_level", "metric": "visits_per_student", "window_days": 61}, {})])
    assert a == b and len(a) == 1
    assert c.signature([("what_changed", {"metric": "no_show_rate", "dimension": "academic_level"}, {})]) != a
    assert c.signature([("chr_trend", {"service_line": "All"}, {})]) == c.signature([("chr_trend", {}, {})])


def test_figures_ignore_commas_years_and_small_counts():
    f = c.figures_in("In 2026, 1,136 bookings (up 22.8%, range 8.6% to 38.8%) from 3 services over 61 days.")
    assert f == {"1136", "22.8", "8.6", "38.8"}


def test_same_analysis_and_shared_figures_are_recognised():
    r = c.compare([_run("Students seen rose 22.8%, from 461 to 566.", [PLAY, VOL]),
                   _run("We saw 566 students, up from 461 (22.8% more).", [VOL]),
                   _run("Demand is up 22.8%: 461 then 566 students; bookings 1,136.", [VOL])])
    assert r["same_analysis"] and r["same_settings"] and r["answered"] == 3
    assert r["shared_figures"] == ["566", "461", "22.8"] and r["share_of_figures_in_common"] == 1.0 and r["untraceable"] == []


def test_a_wording_understood_differently_is_caught():
    other = ("chr_trend", {"service_line": "Counselling"}, {})
    r = c.compare([_run("Up 22.8%.", [VOL]), _run("Counselling had 731 bookings.", [other])])
    assert not r["same_analysis"] and not r["same_settings"]
    same_tool = c.compare([_run("a 22.8", [VOL]), _run("b 22.8", [("what_changed", {"metric": "no_show_rate", "dimension": "academic_level"}, {})])])
    assert same_tool["same_analysis"] and not same_tool["same_settings"]


def test_an_answer_with_no_analysis_or_an_untraceable_figure_is_reported():
    r = c.compare([_run("Up 22.8%.", [VOL]), _run("I am not sure.", [PLAY]), _run("Up 99.9%.", [VOL], bad=["99.9"])])
    assert not r["same_analysis"] and r["answered"] == 2 and r["untraceable"] == ["99.9"]


def test_run_and_report_work_end_to_end_with_a_stand_in_assistant():
    def ask(text):
        if "fail" in text:
            raise RuntimeError("gateway down")
        return {"answer": f"Students seen rose 22.8% ({len(text)} characters asked).", "trace": [VOL], "ungrounded": []}
    qs = [{"id": "a", "topic": "Topic A", "phrasings": ["one way", "another way"]},
          {"id": "b", "topic": "Topic B", "phrasings": ["this works", "this will fail"]}]
    results = c.run(ask, qs)
    s = c.summary(results)
    assert s == {**s, "questions": 2, "answers": 4, "same_analysis": 1, "same_settings": 1, "with_untraceable_figures": 0}
    text = c.report(results, when="Run today.")
    assert "**1 of 2** questions led to the same analysis" in text and "| Topic B | **No** |" in text
    assert "request failed: gateway down" in text and "## How to read this" in text


def test_the_built_in_questions_are_well_formed():
    ids = [q["id"] for q in c.QUESTIONS]
    assert len(ids) == len(set(ids)) == 10 and all(len(q["phrasings"]) == 3 and q["topic"] for q in c.QUESTIONS)
    assert len({p for q in c.QUESTIONS for p in q["phrasings"]}) == 30


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

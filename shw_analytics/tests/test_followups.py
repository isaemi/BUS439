"""Tests for shw_analytics/followups.py: suggestions climb the ladder for the current finding and,
when a briefing exists, point at a tested finding that has not come up.
Run:  python tests/test_followups.py"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import briefing, followups  # noqa: E402

TRACE = [tuple(s) for s in json.loads((Path(__file__).parent / "fixtures" / "sample_trace.json").read_text())]


def _step(name, n=0):
    return copy.deepcopy([s for s in TRACE if s[0] == name][n])


def _no_briefing(fn):
    """Run fn with no briefing built, whatever the state of the real cache."""
    real, briefing.cached = briefing.cached, lambda: None
    try:
        return fn()
    finally:
        briefing.cached = real


def _with_briefing(findings, fn):
    real, briefing.cached = briefing.cached, lambda: {"findings": findings, "also": []}
    try:
        return fn()
    finally:
        briefing.cached = real


TREND = _step("chr_trend")          # counselling by month, with September and October flagged as notably higher


def test_the_first_suggestion_goes_one_step_deeper_and_names_the_finding():
    s = _no_briefing(lambda: followups.suggest_detail([TREND]))
    assert s[0]["kind"] == "deeper"
    assert s[0]["text"] == "Is the October rise in counselling from more students, or more visits each?"


def test_suggestions_climb_the_ladder_as_the_conversation_covers_each_step():
    asked, history, seen = [], [], []
    visits = ("what_changed", {"metric": "visits_per_student", "dimension": "academic_level", "service_line": "Counselling"},
              _step("what_changed")[2])
    by_faculty = ("what_changed", {"metric": "visits_per_student", "dimension": "faculty", "service_line": "Counselling"},
                  _step("what_changed")[2])
    journeys = ("care_journeys", {"service_line": "Counselling"}, _step("care_journeys")[2])
    for extra in (None, visits, by_faculty, journeys):
        if extra:
            history.append([extra])
        s = _no_briefing(lambda: followups.suggest_detail([TREND], asked=asked, history=history))
        seen.append(s[0]["text"])
        asked.append(s[0]["text"])
    assert "more students, or more visits each" in seen[0]                      # decompose
    assert seen[1] == "Which faculties account for the change in students seen in counselling?"      # segment
    assert seen[2] == "Are students in counselling coming back for second visits at the usual rate?"  # follow
    assert seen[3] == "What are students saying about getting an appointment in counselling?"         # voice
    assert len(set(seen)) == 4


def test_what_the_answer_already_covered_is_not_suggested():
    answer = [TREND, ("what_changed", {"metric": "visits_per_student", "dimension": "faculty", "service_line": "Counselling"},
                      _step("what_changed")[2])]
    texts = _no_briefing(lambda: followups.suggest(answer))
    assert not any("more visits each" in t or "Which faculties" in t for t in texts), texts


def test_a_priority_leads_to_what_students_say_about_it():
    step = _step("priority_items")
    s = _no_briefing(lambda: followups.suggest([step]))
    assert s[0].startswith("What are students saying about “" + step[2]["priorities"][0][:20])


def test_second_suggestion_is_a_tested_briefing_finding_about_the_same_service():
    findings = [{"id": "noshow|All services|Appointment format:Phone/Video", "kind": "noshow", "service": "All services"},
                {"id": "single_visit|Counselling|academic_level:Graduate", "kind": "continuity", "service": "Counselling"}]
    s = _with_briefing(findings, lambda: followups.suggest_detail([TREND]))
    assert [x["kind"] for x in s] == ["deeper", "wider"]
    assert s[1]["text"] == "Does returning after a first visit in counselling differ by level of study?"
    assert "briefing" in s[1]["why"]


def test_a_briefing_finding_already_covered_in_the_conversation_is_skipped():
    findings = [{"id": "single_visit|Counselling|academic_level:Graduate", "kind": "continuity", "service": "Counselling"},
                {"id": "priority|counselling|CS1_6", "kind": "priority", "service": "Counselling"}]
    earlier = [[("care_journeys", {"service_line": "Counselling", "dimension": "academic_level"}, _step("care_journeys")[2])]]
    s = _with_briefing(findings, lambda: followups.suggest_detail([TREND], history=earlier))
    assert s[1]["text"] == "Which parts of counselling are most closely linked to overall satisfaction?"


SURVEY = _step("survey_changed")      # students reporting a disability rate counselling lower than a year ago
ITEMS = ("item_summary", {"service": "counselling"}, {"rows": [{"item_text": "x", "pct_agree": 40.0}]})


def _checks(states):
    """Run fn with stand-in results for the 'check before offering' step: {tool: result}."""
    def wrap(fn):
        real = followups.check

        def fake(c):
            call = c.get("call")
            if not call or call[0] not in states:
                return "unknown"
            res = states[call[0]]
            if "error" in res:
                return "dead_end"
            return "finding" if (c["expect"](res) if c.get("expect") else followups.has_finding(call[0], res)) else "nothing"
        followups.check = fake
        try:
            return _no_briefing(fn)
        finally:
            followups.check = real
    return wrap


def test_suggestions_start_from_the_finding_not_from_whichever_analysis_ran_last():
    # The answer ran the over-time test (the finding), then a description, then a custom query.
    answer = [SURVEY, ITEMS, ("run_sql", {"sql": "SELECT 1"}, {"rows": [{"x": 1}]})]
    s = _checks({})(lambda: followups.suggest(answer))
    assert all("students reporting a disability" in t for t in s), s
    assert followups.importance(*SURVEY[::2]) == 3 and followups.importance(*ITEMS[::2]) == 1


def test_what_the_answer_just_established_is_not_offered_back():
    s = _checks({})(lambda: followups.suggest([SURVEY, ITEMS], limit=6))
    assert not any(t.startswith("Do some groups of students rate") for t in s), s


def test_a_suggestion_known_to_lead_somewhere_comes_first_and_a_dead_end_is_dropped():
    gap = {"by_service": [{"service": "counselling", "gap_pts": -12.0, "significant": True}]}
    s = _checks({"group_vs_everyone": gap, "survey_changed": {"groups_that_changed": []}})(
        lambda: followups.suggest_detail([SURVEY]))
    assert s[0]["text"] == "How do students reporting a disability rate each service compared with other students?"
    assert s[0]["why"] == "A tested difference shows up here."
    s = _checks({"group_vs_everyone": {"error": "x"}, "survey_changed": {"error": "no window"}})(
        lambda: followups.suggest([SURVEY], limit=4))
    assert not any("each service compared" in t or "other services too" in t for t in s), s
    assert s and "comments" in s[0]


def test_after_a_group_comparison_the_next_question_names_that_group_and_its_weakest_service():
    step = ("group_vs_everyone", {"dimension": "disability_any", "value": "yes"},
            {"group": "disability_any = yes", "by_service": [{"service": "medical", "gap_pts": -2.0, "significant": False},
                                                             {"service": "counselling", "gap_pts": -12.0, "significant": True}]})
    s = _checks({})(lambda: followups.suggest([step]))
    assert s[0] == "Has the gap for students reporting a disability in counselling widened since last year?"
    after = _checks({})(lambda: followups.suggest([step], history=[[SURVEY]]))     # that was already answered
    assert after[0] == "What are students saying about counselling in their comments?"


def test_never_more_than_two_never_a_repeat_never_a_question_already_asked():
    asked = ["Is the October rise in counselling from more students, or more visits each?"]
    s = _no_briefing(lambda: followups.suggest(TRACE, asked=asked))
    assert len(s) == 2 and len(set(s)) == 2 and asked[0] not in s


def test_a_briefing_answer_offers_its_findings_own_next_questions():
    res = {"findings": [{"headline": "A", "next": "Question about A?"}, {"headline": "B", "next": "Question about B?"}]}
    assert _no_briefing(lambda: followups.suggest([("term_briefing", {}, res)])) == ["Question about A?", "Question about B?"]


def test_nothing_to_suggest_when_there_was_no_analysis_or_it_failed():
    assert _no_briefing(lambda: followups.suggest([])) == [] and _no_briefing(lambda: followups.suggest(None)) == []
    assert _no_briefing(lambda: followups.suggest([("noshow_patterns", {}, {"error": "x"}), "junk"])) == []


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

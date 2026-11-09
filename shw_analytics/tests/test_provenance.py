"""Tests for shw_analytics/provenance.py: evidence must be plain language, never code.

Run:  python tests/test_provenance.py      (or: pytest tests/test_provenance.py)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import provenance as pv  # noqa: E402

TECH = re.compile(r"(?i)\b(select|from appointments|group by|student_key|service_line|"
                  r"academic_level|what_changed|chr_summary|run_sql|ratings_demo|sql|python)\b")

SQL = ("SELECT academic_level, visit_type, COUNT(*) AS attended, COUNT(DISTINCT student_key) AS "
       "students FROM appointments WHERE service_line='Counselling' AND outcome='attended' AND "
       "academic_level IN ('Graduate','Undergraduate') GROUP BY academic_level, visit_type "
       "ORDER BY academic_level, visit_type")

WHAT_CHANGED = {"metric": "no_show_rate", "measure": "No-show rate (% of booked appointments)",
                "dimension": "faculty", "periods": {"before": "2025-09-01 to 2025-10-31",
                                                     "after": "2026-09-01 to 2026-10-31"},
                "n": {"before": 509, "after": 731, "unit": "appointments"},
                "pooled_into_other": ["Law"], "excluded_unknown": 12,
                "data_source": "synthetic test data",
                "notes": ["This splits the change into who came vs. change within groups, by "
                          "faculty only."]}


def test_sql_is_described_in_words():
    text = pv.sql_to_english(SQL)
    assert text.startswith("Looked at appointment records where service is Counselling")
    assert "appointment outcome is attended" in text
    assert "level of study is Graduate or Undergraduate" in text
    assert "the number of distinct students" in text and "the number of records" in text
    assert "broken down by level of study and visit type" in text
    assert not TECH.search(text)


def test_sql_with_unknown_conditions_says_so_without_code():
    text = pv.sql_to_english("SELECT AVG(score) FROM ratings_demo WHERE weird_fn(x) > 3")
    assert "with some conditions" in text and "the average rating" in text
    assert not TECH.search(text)


def test_what_changed_card_is_plain_and_complete():
    card = pv.evidence_cards([("what_changed", {"metric": "no_show_rate", "dimension": "faculty",
                                                "service_line": "Counselling"}, WHAT_CHANGED)])[0]
    assert "2025-09-01 to 2025-10-31 vs 2026-09-01 to 2026-10-31" in card["title"]
    assert "Counselling" in card["title"]
    assert card["records"].startswith("509 appointments in the earlier period and 731")
    assert "re-sampling students" in card["certainty"]
    joined = " ".join(card["keep_in_mind"])
    assert "synthetic" in joined and "Law" in joined and "12 records" in joined
    assert not TECH.search(pv.render_text([card]))


def test_partial_term_warning_is_passed_on_in_plain_words():
    res = {"warning": "PARTIAL TERM: 2026 Fall has data only from 2026-09-01 to 2026-10-31. For "
                      "change between periods use what_changed, which compares matching dates."}
    card = pv.evidence_cards([("chr_summary", {"term": "2026 Fall"}, res)])[0]
    assert "2026 Fall" in card["keep_in_mind"][0] and "what_changed" not in card["keep_in_mind"][0]


def test_failed_step_is_reported():
    card = pv.evidence_cards([("chr_summary", {}, {"error": "Unknown dimension 'visit_type2'"})])[0]
    assert card["failed"].startswith("This step did not work")


def test_meta_steps_are_skipped_and_order_kept():
    trace = [("suggested", {}, {}), ("chr_summary", {"dimension": "academic_level"}, {}),
             ("run_sql", {"sql": SQL}, {"rows": [1, 2]}), ("forced_answer", {}, {})]
    cards = pv.evidence_cards(trace)
    assert [c["title"] for c in cards] == ["Appointment activity by level of study",
                                          "A custom summary of the data"]
    assert cards[1]["records"] == "2 summary rows"
    md = pv.render_markdown(cards)
    assert md.startswith("**1. Appointment activity by level of study**") and not TECH.search(md)


def test_sources_questions_are_recognised():
    yes = ["Where did this come from?", "How did you calculate that?", "what is this based on",
           "Can I trust these numbers?", "Which data did you use?", "show your work",
           "where do these figures come from", "How was that counted?"]
    no = ["Where do graduate students come from?", "How many students came to counselling?",
          "What changed this fall?", "Which faculty has the most no-shows?"]
    assert all(pv.is_sources_question(q) for q in yes), [q for q in yes if not pv.is_sources_question(q)]
    assert not any(pv.is_sources_question(q) for q in no), [q for q in no if pv.is_sources_question(q)]


def test_last_trace_handles_different_turn_shapes():
    step = ("chr_summary", {}, {"rows": []})
    assert pv.last_trace([{"q": "a", "trace": [step]}]) == [step]
    assert pv.last_trace([{"q": "a", "out": {"answer": "x", "trace": [step]}}]) == [step]
    assert pv.last_trace([("a", {"answer": "x", "trace": [step]})]) == [step]
    # Most recent turn with real steps wins; meta-only turns are skipped.
    older = ("what_changed", {}, {})
    hist = [{"trace": [older]}, {"trace": [("suggested", {}, {})]}]
    assert pv.last_trace(hist) == [older]
    assert pv.last_trace([]) == [] and pv.last_trace(None) == []


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

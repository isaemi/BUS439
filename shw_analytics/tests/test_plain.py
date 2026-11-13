"""Tests for shw_analytics/plain.py.   Run:  python tests/test_plain.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics.plain import plain_stats  # noqa: E402


def test_p_values_are_removed_and_ranges_kept():
    assert plain_stats("a 22.8% increase (95% CI 8.6–38.8%, p=0.0012).") == "a 22.8% increase (range 8.6–38.8%)."
    assert plain_stats("statistically reliable (q=0.0018).") == "statistically reliable."
    assert plain_stats("up 75.2% (p < 0.001); October rose 33.0%, p = 0.0002.") == "up 75.2%; October rose 33.0%."
    assert plain_stats("significant (p = 8.6e-18)") == "significant"
    assert plain_stats("no clear link (p-value of 0.98, n = 111)") == "no clear link (n = 111)"


def test_confidence_interval_wording_becomes_range():
    assert plain_stats("no clear change (CI −0.08 to +0.21)") == "no clear change (range −0.08 to +0.21)"
    assert plain_stats("The confidence interval is 18.1% to 64.3%.") == "The range is 18.1% to 64.3%."


def test_ordinary_text_is_left_alone():
    for s in ["Students seen rose from 461 to 566.", "Graduates (n=182) had fewer single visits.",
              "Up 39.3% (range 18.1% to 64.3%).", "The clinic is open 9 to 5; phone first.", ""]:
        assert plain_stats(s) == s


def test_self_correction_preamble_is_removed():
    leaked = ("Good catch — I re-checked my answer against the briefing and found two figures I wrote that don't "
              "appear in it: \"1,717 students\" and \"3,124 appointments\". Here's the corrected version:\n\n"
              "1. Demand is up\n\n22.8% more students were seen.")
    assert plain_stats(leaked) == "1. Demand is up\n\n22.8% more students were seen."
    assert plain_stats("My apologies, here's the corrected version.\n\nStudents seen rose.") == "Students seen rose."


def test_an_ordinary_opening_paragraph_is_kept():
    for s in ["Short answer: demand is up.\n\nStudents seen rose from 461 to 566.",
              "Students who re-checked in at the desk were counted once.\n\nMore detail follows.",
              "One paragraph only, with no blank line after it."]:
        assert plain_stats(s) == s


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

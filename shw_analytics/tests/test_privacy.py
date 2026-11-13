"""Tests for the comment scrubber.   Run:  python tests/test_privacy.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics.privacy import scrub_text  # noqa: E402


def test_identifying_details_are_removed():
    assert scrub_text("Dr. Patel was great.") == "[STAFF_NAME] was great."
    assert scrub_text("counsellor Amy listened") == "[STAFF_NAME] listened"
    assert scrub_text("Email jamie.lee@student.ubc.ca or call 604-555-0182") == "Email [EMAIL] or call [PHONE]"
    assert scrub_text("My student number is 12345678.") == "My student number is [STUDENT_NO]."
    assert scrub_text("Hi, my name is Jordan.") == "Hi, my name is [NAME]."


def test_ordinary_words_after_a_job_title_are_kept():
    for s in ["The nurse was patient and explained everything.", "My counsellor really listened.",
              "The advisor helped me plan.", "I saw a counsellor twice."]:
        assert scrub_text(s) == s, scrub_text(s)


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

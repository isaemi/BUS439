"""Tests for shw_analytics/survey_trends.py: a planted change over time is found, noise is not.
Run:  python tests/test_survey_trends.py"""
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import survey_trends  # noqa: E402

ITEMS = {f"CS1_{i}": f"Counselling question {i}" for i in range(1, 7)}


def _con(seed=0, n=420, group_drop=0.0, item_rise=0.0, history=True, tiny_group=False):
    """Two matching windows a year apart. `group_drop` lowers every answer from students reporting a
    disability in the later window; `item_rise` raises one question for everyone in the later window."""
    rng = np.random.default_rng(seed)
    rows = []
    starts = [pd.Timestamp("2025-09-01"), pd.Timestamp("2026-09-01")] if history else [pd.Timestamp("2026-09-01")]
    for p, start in enumerate(starts):
        later = start.year == 2026
        for i in range(n):
            disab = "yes" if rng.random() < 0.25 else "no"
            intl = "International" if rng.random() < 0.3 else "Domestic"
            rare = "Rare" if tiny_group and i < 4 else "Common"
            base = rng.normal(0, 1) - (group_drop if later and disab == "yes" else 0)
            when = start + pd.Timedelta(days=int(rng.integers(0, 61)))
            for k, (code, text) in enumerate(ITEMS.items()):
                latent = 0.6 * base + 0.8 * rng.normal() + (item_rise if later and k == 0 else 0)
                score = int(np.clip(np.round(3.3 + latent), 1, 5))
                rows.append({"response_id": f"R{p}_{i}", "service": "counselling", "item_code": code, "item_text": text,
                             "score": score, "recorded_at": when, "disability_any": disab, "intl_or_domestic": intl,
                             "academic_level": rare})
    con = duckdb.connect()
    con.register("r", pd.DataFrame(rows))
    con.execute("CREATE TABLE ratings_demo AS SELECT * FROM r")
    return con


def test_a_group_whose_ratings_fell_is_found_and_set_apart_from_everyone_else():
    r = survey_trends.survey_changed(_con(group_drop=1.3))
    flags = r["groups_that_changed"]
    assert [(f["dimension"], f["group"]) for f in flags] == [("disability_any", "yes")], flags
    f = flags[0]
    assert f["change_pts"] < -10 and f["ci_pts"][1] < 0 and f["differs_from_everyone_else"] is True
    assert abs(f["everyone_else_change_pts"]) < 8 and r["verdict"] == "changes_found"
    # groups that only moved because they contain those students are not listed as changed themselves
    assert r["services"][0]["verdict"] == "lower than before"
    assert r["periods"] == {"before": "2025-09-01 to 2025-10-31", "after": "2026-09-01 to 2026-10-31"}


def test_a_question_that_improved_is_found():
    r = survey_trends.survey_changed(_con(item_rise=0.7))
    assert [q["item_code"] for q in r["questions_that_changed"]] == ["CS1_1"]
    q = r["questions_that_changed"][0]
    assert q["change_pts"] > 8 and q["pct_agree_after"] > q["pct_agree_before"] and q["q"] < 0.05
    assert r["groups_that_changed"] == []                      # a rise for everyone is not a group's change


def test_no_change_is_reported_as_no_change_across_many_random_surveys():
    false_groups = false_items = false_services = 0
    for seed in range(20):
        r = survey_trends.survey_changed(_con(seed=seed))
        false_groups += len(r["groups_that_changed"])
        false_items += len(r["questions_that_changed"])
        false_services += sum(s["verdict"] != "no clear change" for s in r["services"])
    assert false_groups <= 1 and false_items <= 2, (false_groups, false_items)   # of 120 questions and ~120 groups
    assert false_services <= 3, false_services                 # one uncorrected 95% range per survey


def test_small_groups_are_counted_but_never_flagged_and_the_reader_is_told_what_could_be_missed():
    r = survey_trends.survey_changed(_con(group_drop=1.3, tiny_group=True))
    assert r["groups_too_small_to_check"] >= 1
    assert not any(g["group"] == "Rare" for g in r["groups_that_changed"])
    assert r["smallest_group_change_detectable_pts"] > 0
    assert any("could not be checked" in n for n in r["notes"]) and any("before it could be told apart" in n for n in r["notes"])


def test_one_group_can_be_asked_for_and_all_questions_come_back_for_one_service():
    r = survey_trends.survey_changed(_con(group_drop=1.3), service="counselling", dimension="disability_any")
    assert {g["dimension"] for g in r["groups_that_changed"]} == {"disability_any"} and r["groups_checked"] == 2
    assert len(r["all_questions"]) == len(ITEMS)


def test_plain_errors_when_the_question_cannot_be_answered():
    for kwargs, words in [({"history": False}, "no full comparison window"), ]:
        try:
            survey_trends.survey_changed(_con(**kwargs))
            raise AssertionError("expected an error")
        except ValueError as e:
            assert words in str(e), str(e)
    for call, words in [(dict(service="dentistry"), "No responses about"), (dict(dimension="shoe_size"), "not a group this survey records")]:
        try:
            survey_trends.survey_changed(_con(), **call)
            raise AssertionError("expected an error")
        except ValueError as e:
            assert words in str(e), str(e)


def test_results_are_plain_numbers_and_always_carry_the_anonymous_survey_caveat():
    r = survey_trends.survey_changed(_con(group_drop=1.3))
    json.dumps(r)                                              # would fail on numpy types
    assert "different students" in r["notes"][0] and r["tests_run"] > 10


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

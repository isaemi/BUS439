"""Tests for shw_analytics/regression.py. Each test plants a known structure in fake survey
ratings: which questions are linked to agreeing with the outcome directly, which only through
another question, and which not at all.

Run:  python tests/test_regression.py      (or: pytest tests/test_regression.py)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import regression  # noqa: E402

ITEMS = {"CS1_1": "I could book an appointment easily",
         "CS1_2": "Front desk staff were courteous and respectful",
         "CS1_3": "I felt welcomed and included in the space",
         "CS1_6": "My counsellor and I focused on the issues most important to me",
         "CS1_11": "Overall, I am satisfied with the support I received"}


def _likert(latent):
    return np.clip(np.round(3 + latent), 1, 5).astype(int)


def _con(n=600, seed=0, interaction=False, separated=False, levels=None):
    """CS1_1 is linked to satisfaction directly; CS1_3 only because it moves with CS1_1;
    CS1_2 is unrelated. `levels` gives the academic_level of each student."""
    rng = np.random.default_rng(seed)
    noise = lambda s=1.0: rng.normal(0, s, n)  # noqa: E731
    book = noise()
    welcome = 0.8 * book + 0.6 * noise()
    desk = noise()
    focus = noise()
    cols = {"CS1_1": _likert(book), "CS1_3": _likert(welcome), "CS1_2": _likert(desk),
            "CS1_6": _likert(focus)}
    # Satisfaction follows the booking answer students gave, so welcome has no link of its own.
    said = cols["CS1_1"] - 3.0
    latent = 1.2 * said + (0.9 * said * (cols["CS1_6"] - 3.0) if interaction else 0) + noise(0.8)
    cols["CS1_11"] = _likert(latent)
    if separated:
        cols["CS1_2"] = np.where(cols["CS1_11"] >= 4, 5, 1)
    rows = [{"response_id": f"r{i}", "service": "counselling", "item_code": c,
             "score": int(v[i]), "agree": bool(v[i] >= 4), "item_text": ITEMS[c]}
            for c, v in cols.items() for i in range(n)]
    if levels is None:
        levels = rng.choice(["Undergraduate", "Graduate"], n, p=[0.7, 0.3])
    responses = pd.DataFrame({"response_id": [f"r{i}" for i in range(n)],
                              "academic_level": levels})
    con = duckdb.connect()
    con.register("ratings", pd.DataFrame(rows))
    con.register("responses", responses)
    return con


def _term(r, code):
    return next(t for t in r["terms"] if t.get("item_code") == code)


def test_planted_direct_link_is_found():
    r = regression.logistic_regression(_con(), "counselling", "CS1_11", ["CS1_1", "CS1_2"])
    book, desk = _term(r, "CS1_1"), _term(r, "CS1_2")
    assert book["link"] == "higher_odds" and book["ci"][0] > 1, book
    assert desk["link"] == "no_clear_link", desk
    assert r["verdict"] == "clear_links" and r["n"] == 600


def test_link_through_another_question_disappears_when_adjusted():
    r = regression.logistic_regression(_con(), "counselling", "CS1_11", ["CS1_1", "CS1_3"])
    welcome = _term(r, "CS1_3")
    assert welcome["unadjusted_link"] == "higher_odds", welcome
    assert welcome["link"] == "no_clear_link", welcome
    assert any("not once the other factors" in f for f in r["findings"]), r["findings"]


def test_odds_ratio_matches_an_independent_fit():
    import statsmodels.api as sm
    con = _con()
    r = regression.logistic_regression(con, "counselling", "CS1_11", ["CS1_1", "CS1_2"])
    wide = con.execute("SELECT * FROM ratings").df().pivot(index="response_id", columns="item_code",
                                                           values="score")
    fit = sm.Logit((wide["CS1_11"] >= 4).astype(int),
                   sm.add_constant(wide[["CS1_1", "CS1_2"]].astype(float))).fit(disp=0)
    assert _term(r, "CS1_1")["odds_ratio"] == round(float(np.exp(fit.params["CS1_1"])), 2)


def test_planted_interaction_is_found():
    r = regression.logistic_regression(_con(interaction=True), "counselling", "CS1_11",
                                       ["CS1_1", "CS1_6"], interaction=True)
    inter = next(t for t in r["terms"] if t["kind"] == "interaction")
    assert inter["link"] == "higher_odds", inter


def test_background_control_reports_groups_against_the_largest():
    r = regression.logistic_regression(_con(), "counselling", "CS1_11", ["CS1_1"],
                                       controls=["academic_level"])
    group = next(t for t in r["terms"] if t["kind"] == "background_group")
    assert group["per"] == "Graduate compared with Undergraduate" and group["students"] >= 10


def test_small_groups_are_pooled_and_unknowns_dropped():
    n = 600
    levels = np.array(["Undergraduate"] * 400 + ["Graduate"] * 180 + ["Visiting"] * 4
                      + ["Exchange"] * 4 + ["Prefer not to say"] * 12)
    r = regression.logistic_regression(_con(n, levels=levels), "counselling", "CS1_11", ["CS1_1"],
                                       controls=["academic_level"])
    assert r["n"] == n - 12 - 8 and r["excluded_unknown"] == 20, (r["n"], r["excluded_unknown"])
    assert all("Visiting" not in t["term"] for t in r["terms"])


def test_too_many_terms_for_the_students_declines():
    try:
        regression.logistic_regression(_con(n=60), "counselling", "CS1_11",
                                       ["CS1_1", "CS1_2", "CS1_3", "CS1_6"])
    except ValueError as e:
        assert "too few" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_near_perfect_prediction_declines():
    try:
        regression.logistic_regression(_con(separated=True), "counselling", "CS1_11", ["CS1_2"])
    except ValueError as e:
        assert "almost perfectly" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_bad_requests_decline():
    con = _con()
    for kwargs in ({"predictor_items": ["CS9_9"]}, {"predictor_items": ["CS1_11"]},
                   {"predictor_items": []}, {"predictor_items": ["CS1_1"], "interaction": True},
                   {"predictor_items": ["CS1_1"], "controls": ["birth_year"]}):
        try:
            regression.logistic_regression(con, "counselling", "CS1_11", **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_answers_never_include_student_ids():
    r = regression.logistic_regression(_con(), "counselling", "CS1_11", ["CS1_1", "CS1_3"],
                                       controls=["academic_level"], interaction=True)
    assert "r1" not in str(r) and "response_id" not in str(r)


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

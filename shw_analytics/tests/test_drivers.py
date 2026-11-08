"""Tests for shw_analytics/drivers.py. Each test plants a known structure in fake survey
ratings: which questions move with overall satisfaction, and how well each scores.

Run:  python tests/test_drivers.py      (or: pytest tests/test_drivers.py)
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import drivers, stats  # noqa: E402

stats.N_BOOT = 300

ITEMS = {"CS1_1": "I felt heard by my counsellor",
         "CS1_2": "Front desk staff were welcoming",
         "CS1_3": "I could book an appointment easily",
         "CS1_6": "My counsellor and I focused on what mattered to me",
         "CS1_4": "I would recommend UBC Student Health and Wellbeing to a friend",
         "CS1_5": "I would recommend the Wellness Info Desk to anyone",
         "CS1_9": "I felt heard (duplicate wording)",
         "CS1_11": "Overall, I am satisfied with the support I received"}


def _likert(latent, shift):
    """Latent score -> 1..5; `shift` moves the whole item up (easier to agree) or down."""
    return np.clip(np.round(3 + latent + shift), 1, 5).astype(int)


def _con(n=150, planted=True, seed=0):
    rng = np.random.default_rng(seed)
    sat = rng.normal(0, 1, n)                                   # overall satisfaction
    noise = lambda: rng.normal(0, 1, n)  # noqa: E731
    if planted:
        cols = {"CS1_11": _likert(sat, 0),
                "CS1_1": _likert(0.9 * sat + 0.3 * noise(), -0.8),   # strong link, low score
                "CS1_2": _likert(0.1 * sat + 0.9 * noise(), 1.2),    # weak link, high score
                "CS1_3": _likert(0.2 * sat + 0.9 * noise(), -0.6),   # access item, weak link, low
                "CS1_6": _likert(0.6 * sat + 0.6 * noise(), 0.6)}    # good link, good score
    else:
        cols = {c: _likert(noise(), 0) for c in ["CS1_11", "CS1_1", "CS1_2", "CS1_3", "CS1_6"]}
    cols["CS1_4"] = _likert(sat, 0)               # outcome-like: must be ignored
    cols["CS1_5"] = _likert(sat, 0)               # misplaced info-desk item: must be ignored
    cols["CS1_9"] = cols["CS1_1"]                 # duplicate: must be ignored
    rows = [{"response_id": f"r{i}", "service": "counselling", "item_code": c,
             "score": int(v[i]), "agree": bool(v[i] >= 4), "item_text": ITEMS[c]}
            for c, v in cols.items() for i in range(n)]
    items = pd.DataFrame([{"service": "counselling", "block": "CS1", "item_code": c, "item_text": t,
                           "duplicate_of": "CS1_1" if c == "CS1_9" else None} for c, t in ITEMS.items()])
    con = duckdb.connect()
    con.register("ratings", pd.DataFrame(rows))
    con.register("items", items)
    return con


def test_planted_driver_is_the_priority():
    r = drivers.priority_items(_con(), "Counselling")
    assert r["verdict"] == "clear_priorities", r["verdict"]
    assert r["priorities"][0] == "I felt heard by my counsellor"
    q = {it["item"]: it["quadrant"] for it in r["items"]}
    assert q["Front desk staff were welcoming"] == "possibly_over_served"
    assert q["My counsellor and I focused on what mattered to me"] == "keep_up"


def test_outcome_like_misplaced_and_duplicate_items_are_not_drivers():
    codes = {it["item_code"] for it in drivers.priority_items(_con(), "counselling")["items"]}
    assert codes == {"CS1_1", "CS1_2", "CS1_3", "CS1_6"}


def test_access_items_are_flagged_as_understated():
    r = drivers.priority_items(_con(), "counselling")
    booking = next(it for it in r["items"] if it["item_code"] == "CS1_3")
    assert booking["access_item"] and any("never received the survey" in n for n in r["notes"])


def test_no_structure_means_no_forced_priority():
    r = drivers.priority_items(_con(planted=False), "counselling")
    assert r["verdict"] in ("no_single_item_stands_out", "no_stable_priority"), r["verdict"]
    assert r["priorities"] == []


def test_too_few_respondents_is_declined():
    try:
        drivers.priority_items(_con(n=20), "counselling")
    except ValueError as e:
        assert "20 students" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_service_names_are_forgiving_and_unknown_ones_rejected():
    assert drivers._service("Wellness Advising") == "wellness_adv"
    assert drivers._service("medical clinic") == "medical"
    try:
        drivers._service("dentistry")
    except ValueError as e:
        assert "counselling" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_link_matches_an_independent_spearman():
    from scipy.stats import spearmanr
    con = _con()
    r = drivers.priority_items(con, "counselling")
    wide = con.execute("SELECT * FROM ratings").df().pivot(index="response_id", columns="item_code", values="score")
    expected = round(float(spearmanr(wide["CS1_1"], wide["CS1_11"]).statistic), 2)
    got = next(it["link_to_outcome"] for it in r["items"] if it["item_code"] == "CS1_1")
    assert got == expected


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

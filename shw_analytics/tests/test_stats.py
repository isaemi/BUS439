"""Tests for shw_analytics/stats.py. Expected values come from independent sources
(statsmodels, worked examples), not from re-running the same formulas.

Run:  python tests/test_stats.py      (or: pytest tests/test_stats.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import stats  # noqa: E402


def test_bh_worked_example_keeps_three_of_ten():
    # The FDR example: 6 raw p-values under 0.05, BH at 5% keeps the first 3.
    p = [0.001, 0.004, 0.012, 0.030, 0.045, 0.048, 0.2, 0.4, 0.6, 0.9]
    assert int((stats.bh_adjust(p) <= 0.05).sum()) == 3


def test_bh_matches_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p = np.random.default_rng(1).uniform(0, 0.2, 200)
    expected = multipletests(p, method="fdr_bh")[1]
    assert np.allclose(stats.bh_adjust(p), expected)


def test_bh_ignores_missing_p_values():
    q = stats.bh_adjust([0.01, np.nan, 0.04])
    assert np.isnan(q[1])
    assert np.allclose(q[[0, 2]], [0.02, 0.04])   # m = 2, not 3


def test_two_proportion_matches_statsmodels():
    from statsmodels.stats.proportion import proportions_ztest
    diff, p = stats.two_proportion_test(60, 100, 45, 100)
    assert abs(diff - 15.0) < 1e-9
    assert abs(p - proportions_ztest([60, 45], [100, 100])[1]) < 1e-9


def test_two_proportion_handles_empty_and_degenerate_groups():
    assert np.isnan(stats.two_proportion_test(0, 0, 5, 10)[1])
    assert stats.two_proportion_test(10, 10, 20, 20)[1] == 1.0


def test_bootstrap_interval_covers_known_mean():
    x = np.random.default_rng(2).normal(50, 10, 400)
    r = stats.bootstrap(pd.DataFrame({"x": x}), lambda d: d["x"].mean(), n_boot=500)
    assert r["low"][0] < 50 < r["high"][0]
    assert 1.5 < r["high"][0] - r["low"][0] < 2.5   # about 2 x 1.96 x 10/sqrt(400) = 1.96


def test_bootstrap_strata_keep_each_period_size():
    df = pd.DataFrame({"period": ["a"] * 30 + ["b"] * 70, "x": range(100)})
    r = stats.bootstrap(df, lambda d: (d["period"] == "a").sum(), strata="period", n_boot=50)
    assert np.all(r["draws"][:, 0] == 30)


def test_bootstrap_returns_several_statistics_together():
    df = pd.DataFrame({"x": np.arange(100.0)})
    r = stats.bootstrap(df, lambda d: (d["x"].mean(), d["x"].max()), n_boot=50)
    assert r["draws"].shape == (50, 2) and r["estimate"].tolist() == [49.5, 99.0]


def test_bootstrap_is_reproducible():
    df = pd.DataFrame({"x": np.random.default_rng(3).normal(size=200)})
    f = lambda d: d["x"].mean()  # noqa: E731
    assert stats.bootstrap(df, f)["low"][0] == stats.bootstrap(df, f)["low"][0]


def test_drop_unknown_removes_and_counts_non_answers():
    df = pd.DataFrame({"level": ["Graduate", "Prefer not to say", None, "Undergraduate", " unknown "]})
    kept, dropped = stats.drop_unknown(df, "level")
    assert kept["level"].tolist() == ["Graduate", "Undergraduate"] and dropped == 3


def test_pool_small_relabels_rare_groups():
    df = pd.DataFrame({"fac": ["Arts"] * 12 + ["Science"] * 15 + ["Law"] * 3})
    assert sorted(stats.pool_small(df, "fac", min_n=10)["fac"].unique()) == ["Arts", "Other", "Science"]


def test_pool_small_requires_enough_in_every_period():
    # Law has 12 last year but only 4 this year: not comparable, so it is pooled.
    df = pd.DataFrame({"period": ["last"] * 24 + ["this"] * 16,
                       "fac": ["Arts"] * 12 + ["Law"] * 12 + ["Arts"] * 12 + ["Law"] * 4})
    out = stats.pool_small(df, "fac", min_n=10, by="period")
    assert set(out["fac"]) == {"Arts", "Other"}


def test_verdict_only_calls_direction_when_interval_excludes_zero():
    assert stats.verdict(0.5, 3.0) == "increase"
    assert stats.verdict(-3.0, -0.5) == "decrease"
    assert stats.verdict(-1.0, 2.0) == "no clear change"


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

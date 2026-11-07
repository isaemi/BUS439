"""Shared statistics for the insight tools (what_changed, priority_items, find_segments,
notable_findings): Benjamini-Hochberg correction, two-proportion tests, bootstrap
intervals, small-group handling and verdict wording.

Every insight tool uses these so that tests, corrections and wording stay consistent:
one place decides what counts as "real" and how it is described.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config

N_BOOT = 1000
SEED = 439  # fixed so the same data always gives the same interval
UNKNOWN = {"", "nan", "none", "unknown", "prefer not to say", "prefer not to answer",
           "not stated", "missing"}


def bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values, in the same order as the input.

    A finding is a discovery at false discovery rate q when its adjusted value is <= q.
    NaN p-values stay NaN and do not count towards the number of tests."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    order = np.argsort(p[ok])
    scaled = p[ok][order] * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]   # keep adjusted values monotone
    adj = np.empty(m)
    adj[order] = np.minimum(scaled, 1.0)
    out[ok] = adj
    return out


def two_proportion_test(x1, n1, x2, n2):
    """Pooled two-sided z-test for two proportions.

    Returns (difference in percentage points, p-value). NaN if either group is empty."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    r1, r2 = x1 / n1, x2 / n2
    pool = (x1 + x2) / (n1 + n2)
    if pool in (0, 1):
        return 100 * (r1 - r2), 1.0
    z = (r1 - r2) / np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    return 100 * (r1 - r2), float(2 * norm.sf(abs(z)))


def bootstrap(df: pd.DataFrame, stat_fn, strata=None, n_boot=N_BOOT, level=0.95, seed=SEED):
    """Percentile bootstrap for one or more statistics.

    stat_fn(df) returns a number or a sequence of numbers (e.g. mix and experience effect
    from the same resample). Rows are resampled with replacement; with `strata` (a column
    name such as "period"), within each stratum, so every resample keeps each period's size.

    Returns dict with arrays: estimate, low, high, and draws (n_boot x k)."""
    d = df.reset_index(drop=True)
    est = np.atleast_1d(np.asarray(stat_fn(d), dtype=float))
    if strata is None:
        blocks = [np.arange(len(d))]
    else:
        blocks = [np.asarray(ix) for ix in d.groupby(strata, sort=False, observed=True).indices.values()]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, est.size))
    for b in range(n_boot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in blocks])
        draws[b] = np.atleast_1d(np.asarray(stat_fn(d.iloc[take]), dtype=float))
    tail = (1 - level) / 2
    low, high = np.nanquantile(draws, [tail, 1 - tail], axis=0)
    return {"estimate": est, "low": low, "high": high, "draws": draws}


def drop_unknown(df: pd.DataFrame, col: str):
    """Remove rows whose `col` is missing or 'prefer not to say'-style.

    Returns (kept rows, number dropped) so tools can report what was excluded."""
    s = df[col].astype(str).str.strip().str.lower()
    keep = df[col].notna() & ~s.isin(UNKNOWN)
    return df[keep], int((~keep).sum())


def pool_small(df: pd.DataFrame, col: str, min_n: int | None = None, by=None, other="Other"):
    """Relabel categories of `col` with fewer than min_n rows as `other`.

    With `by` (e.g. "period"), a category is kept only if it has min_n rows in EVERY
    level of `by`, so groups are comparable across periods. Returns a copy."""
    min_n = config.MIN_CELL if min_n is None else min_n
    out = df.copy()
    if by is None:
        counts = out[col].value_counts()
        small = counts[counts < min_n].index
    else:
        counts = out.groupby([by, col], observed=True).size().unstack(fill_value=0)
        small = counts.columns[(counts < min_n).any(axis=0)]
    out[col] = out[col].astype(object).where(~out[col].isin(small), other)
    return out


def verdict(low, high, up="increase", down="decrease", none="no clear change"):
    """Wording from a confidence interval: only call a direction when the whole interval
    is on one side of zero."""
    if low > 0:
        return up
    if high < 0:
        return down
    return none

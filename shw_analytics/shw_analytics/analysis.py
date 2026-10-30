"""Deterministic statistics. The LLM never computes numbers itself; it calls
these functions and explains the results. That keeps answers reproducible and
makes the equity flags defensible with small samples."""
import duckdb
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from . import config
from .ingest import load_dictionary
from .privacy import suppress_small_cells


def connect(read_only=True):
    return duckdb.connect(str(config.DB_PATH), read_only=read_only,
                          config={"enable_external_access": False})


def equity_dimensions(d=None):
    d = d or load_dictionary()
    dims = [m["name"] for m in d["demographics"].values() if m.get("equity")]
    for cb in d["checkbox"].values():
        if cb.get("derive", {}).get("equity"):
            dims.append(cb["derive"]["name"])
    return dims


def _respondent_scores(con, service=None, item_code=None):
    """One row per respondent: mean score and share of agree answers."""
    where, params = [], []
    if service:
        where.append("service = ?"); params.append(service)
    if item_code:
        where.append("item_code = ?"); params.append(item_code)
    w = ("WHERE " + " AND ".join(where)) if where else ""
    return con.execute(f"""SELECT response_id, service, AVG(score) AS mean_score,
                                  AVG(CASE WHEN agree THEN 1 ELSE 0 END) AS agree_rate
                           FROM ratings {w} GROUP BY response_id, service""", params).df()


def service_summary(con):
    df = con.execute("""SELECT service, COUNT(DISTINCT response_id) AS n,
                               ROUND(AVG(score), 2) AS mean_score,
                               ROUND(100 * AVG(CASE WHEN agree THEN 1 ELSE 0 END), 1) AS pct_agree
                        FROM ratings GROUP BY service ORDER BY n DESC""").df()
    return suppress_small_cells(df)[0]


def item_summary(con, service):
    df = con.execute("""SELECT item_code, item_text, COUNT(*) AS n,
                               ROUND(AVG(score), 2) AS mean_score,
                               ROUND(100 * AVG(CASE WHEN agree THEN 1 ELSE 0 END), 1) AS pct_agree
                        FROM ratings WHERE service = ? GROUP BY item_code, item_text
                        ORDER BY pct_agree""", [service]).df()
    return suppress_small_cells(df)[0]


MULTI_DIMENSIONS = {  # dimensions stored one row per selected option
    "faculty": "SELECT response_id, faculty AS grp FROM response_faculty",
    "disability": "SELECT response_id, disability AS grp FROM response_disability",
    "racial_background": "SELECT response_id, value AS grp FROM response_multi WHERE question = 'racial_background'",
    "identity_d15": "SELECT response_id, value AS grp FROM response_multi WHERE question = 'identity_d15'",
}


def group_dimensions():
    """Every dimension compare_groups accepts."""
    return equity_dimensions() + list(MULTI_DIMENSIONS) + [
        "academic_level", "year_of_study", "enrollment_status", "living_arrangement", "new_to_ubc"]


def compare_groups(con, dimension, service=None, item_code=None):
    """% agree and mean score by group of one demographic dimension.
    Returns (table, hidden_groups): hidden_groups names groups dropped for n < MIN_CELL."""
    allowed = group_dimensions()
    if dimension not in allowed:
        raise ValueError(f"Unknown dimension '{dimension}'. Use one of: {', '.join(allowed)}")
    scores = _respondent_scores(con, service, item_code)
    if dimension in MULTI_DIMENSIONS:  # respondents who pick several options count in each
        demo = con.execute(MULTI_DIMENSIONS[dimension]).df()
    else:
        demo = con.execute(f'SELECT response_id, "{dimension}" AS grp FROM responses').df()
    m = scores.merge(demo, on="response_id").dropna(subset=["grp"])
    out = (m.groupby("grp").agg(n=("response_id", "nunique"),
                                mean_score=("mean_score", "mean"),
                                pct_agree=("agree_rate", "mean"))
             .reset_index().rename(columns={"grp": dimension}))
    out["mean_score"] = out["mean_score"].round(2)
    out["pct_agree"] = (100 * out["pct_agree"]).round(1)
    hidden_groups = out.loc[out["n"] < config.MIN_CELL, dimension].astype(str).tolist()
    out, _ = suppress_small_cells(out)
    return out, hidden_groups


def _bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def equity_scan(con, service=None, by_item=False):
    """Compare every group in every equity dimension against everyone else, per
    service (or per item). Returns (all_tests, flags, thin_groups).

    A flag needs all three: group n >= EQUITY_MIN_GROUP, a gap in % agree of at
    least MIN_GAP_TOP2, and a Benjamini-Hochberg q-value below FDR_Q. The FDR
    step matters: with ~6 services x 7 dimensions x several groups we run
    hundreds of tests, and uncorrected p < .05 would flag noise every time.
    """
    dims = equity_dimensions()
    demo = con.execute("SELECT response_id, " + ", ".join(f'"{x}"' for x in dims) + " FROM responses").df()
    services = [service] if service else [s for s in con.execute(
        "SELECT DISTINCT service FROM ratings ORDER BY 1").df()["service"]]
    tests, thin = [], []
    for svc in services:
        units = [None]
        if by_item:
            units = con.execute("SELECT DISTINCT item_code FROM ratings WHERE service=?", [svc]).df()["item_code"].tolist()
        for item in units:
            scores = _respondent_scores(con, svc, item).merge(demo, on="response_id")
            for dim in dims:
                sub = scores.dropna(subset=[dim])
                for grp, g in sub.groupby(dim):
                    rest = sub[sub[dim] != grp]
                    if len(g) < config.EQUITY_MIN_GROUP or len(rest) < config.EQUITY_MIN_GROUP:
                        thin.append({"service": svc, "dimension": dim, "group": grp,
                                     "n": f"<{config.EQUITY_MIN_GROUP}" if len(g) < config.EQUITY_MIN_GROUP else len(g)})
                        continue
                    p = mannwhitneyu(g["mean_score"], rest["mean_score"]).pvalue
                    tests.append({"service": svc, "item_code": item, "dimension": dim, "group": grp,
                                  "n_group": len(g), "n_rest": len(rest),
                                  "pct_agree_group": round(100 * g["agree_rate"].mean(), 1),
                                  "pct_agree_rest": round(100 * rest["agree_rate"].mean(), 1),
                                  "mean_group": round(g["mean_score"].mean(), 2),
                                  "mean_rest": round(rest["mean_score"].mean(), 2),
                                  "p": p})
    t = pd.DataFrame(tests)
    if t.empty:
        return t, t, pd.DataFrame(thin)
    t["gap_pts"] = (t["pct_agree_group"] - t["pct_agree_rest"]).round(1)
    t["q"] = _bh(t["p"])
    t["flag"] = (t["q"] < config.FDR_Q) & (t["gap_pts"].abs() >= 100 * config.MIN_GAP_TOP2)
    flags = t[t["flag"]].sort_values("gap_pts")
    thin_df = pd.DataFrame(thin).drop_duplicates(subset=["service", "dimension", "group"])
    return t, flags, thin_df


if __name__ == "__main__":
    con = connect()
    print(service_summary(con).to_string(index=False))
    tests, flags, thin = equity_scan(con)
    print(f"\nEquity scan: {len(tests)} comparisons, {len(flags)} flagged, "
          f"{len(thin)} group-service cells too small to compare")
    print(flags.to_string(index=False) if len(flags) else "No flags.")
    print("\nSmallest p-values (before correction):")
    print(tests.nsmallest(5, "p")[["service", "dimension", "group", "n_group", "gap_pts", "p", "q"]].to_string(index=False))


WAIT_COLUMNS = {"medical": "med_wait_time", "counselling": "cs_wait_time"}


def wait_vs_rating(con, service):
    """Ratings by wait-time category, plus a Spearman rank correlation between wait
    category and each respondent's mean score. Assumes wait codes are ordered from
    shortest to longest wait (CONFIRM against the survey's answer order)."""
    col = WAIT_COLUMNS.get(service)
    if not col:
        raise ValueError(f"Wait time is only asked for: {', '.join(WAIT_COLUMNS)}")
    scores = _respondent_scores(con, service)
    waits = con.execute(f'SELECT response_id, "{col}" AS wait FROM responses WHERE "{col}" IS NOT NULL').df()
    m = scores.merge(waits, on="response_id")
    by = (m.groupby("wait").agg(n=("response_id", "nunique"), mean_score=("mean_score", "mean"),
                                pct_agree=("agree_rate", "mean")).reset_index())
    by["mean_score"] = by["mean_score"].round(2)
    by["pct_agree"] = (100 * by["pct_agree"]).round(1)
    by, hidden = suppress_small_cells(by)
    codes = pd.to_numeric(m["wait"], errors="coerce")
    ok = codes.notna()
    rho, p = spearmanr(codes[ok], m.loc[ok, "mean_score"]) if ok.sum() >= 10 else (float("nan"), float("nan"))
    return by, hidden, {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 4),
                        "n": int(ok.sum())}


# Focus-vs-overall views for dashboards ("Faculty of Arts compared with overall").
# A focus is one group of one dimension, e.g. {"dimension": "faculty", "value": "Arts"}.

def focus_ids(con, focus):
    """Response ids in the focus group, or None for everyone."""
    if not focus:
        return None
    dim, value = focus["dimension"], str(focus["value"])
    if dim not in group_dimensions():
        raise ValueError(f"Unknown focus dimension '{dim}'")
    if dim in MULTI_DIMENSIONS:
        df = con.execute(f"SELECT response_id FROM ({MULTI_DIMENSIONS[dim]}) WHERE grp = ?", [value]).df()
    else:
        df = con.execute(f'SELECT response_id FROM responses WHERE "{dim}" = ?', [value]).df()
    return set(df["response_id"])


def focus_values(con, dimension):
    if dimension in MULTI_DIMENSIONS:
        df = con.execute(f"SELECT DISTINCT grp FROM ({MULTI_DIMENSIONS[dimension]}) ORDER BY 1").df()
        return df["grp"].astype(str).tolist()
    df = con.execute(f'SELECT DISTINCT "{dimension}" AS v FROM responses WHERE "{dimension}" IS NOT NULL ORDER BY 1').df()
    return df["v"].astype(str).tolist()


def _ratings(con, service=None):
    w, p = ("WHERE service = ?", [service]) if service else ("", [])
    return con.execute(f"SELECT response_id, service, item_code, item_text, score, agree FROM ratings {w}", p).df()


def _summ(df):
    return pd.Series({"n": df["response_id"].nunique(),
                      "mean_score": round(df["score"].mean(), 2) if len(df) else np.nan,
                      "pct_agree": round(100 * df["agree"].mean(), 1) if len(df) else np.nan})


def kpi_summary(con, ids=None, service=None):
    """Headline figures for the focus group and for everyone (same service filter)."""
    r = _ratings(con, service)
    overall = _summ(r)
    focus = _summ(r[r["response_id"].isin(ids)]) if ids is not None else overall
    pct_strong = lambda d: round(100 * (d["score"] == 1).mean(), 1) if len(d) else np.nan
    sub = r[r["response_id"].isin(ids)] if ids is not None else r
    if service:
        respondents = (int(focus["n"]), int(overall["n"]))
    else:  # whole survey: count responses, matching the header
        total = con.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
        respondents = (len(ids) if ids is not None else total, total)
    return {
        "respondents": respondents,
        "pct_agree": (focus["pct_agree"], overall["pct_agree"]),
        "mean_score": (focus["mean_score"], overall["mean_score"]),
        "pct_strongly_disagree": (pct_strong(sub), pct_strong(r)),
    }


def compare_series(con, by, ids=None, service=None, measure="pct_agree"):
    """Long table: one row per category (service or item) and series (focus/overall)."""
    r = _ratings(con, service)
    key = {"service": "service", "item": "item_text"}[by]
    frames = []
    for label, d in ([("Focus", r[r["response_id"].isin(ids)])] if ids is not None else []) + [("Overall", r)]:
        g = d.groupby(key).apply(_summ, include_groups=False).reset_index()
        g["series"] = label
        frames.append(g)
    out = pd.concat(frames, ignore_index=True).rename(columns={key: "category"})
    out = out[out["n"] >= config.MIN_CELL]
    out["value"] = out[measure]
    return out[["category", "series", "value", "n", "pct_agree", "mean_score"]]


def responses_by_period(con, ids=None, freq="day"):
    """Respondents per day/week/month for focus and overall."""
    df = con.execute(f"""SELECT response_id, CAST(DATE_TRUNC('{freq}', recorded_at) AS DATE) AS period
                         FROM responses WHERE recorded_at IS NOT NULL""").df()
    frames = []
    for label, d in ([("Focus", df[df["response_id"].isin(ids)])] if ids is not None else []) + [("Overall", df)]:
        g = d.groupby("period")["response_id"].nunique().reset_index(name="n")
        g["series"] = label
        frames.append(g)
    out = pd.concat(frames, ignore_index=True)
    return out[out["n"] >= config.MIN_CELL]


def composition(con, dimension, ids=None):
    """Who the respondents are: count and share per group of one dimension."""
    if dimension in MULTI_DIMENSIONS:
        demo = con.execute(MULTI_DIMENSIONS[dimension]).df()
    else:
        demo = con.execute(f'SELECT response_id, "{dimension}" AS grp FROM responses').df()
    if ids is not None:
        demo = demo[demo["response_id"].isin(ids)]
    demo = demo.dropna(subset=["grp"])
    g = demo.groupby("grp")["response_id"].nunique().reset_index(name="n").rename(columns={"grp": "group"})
    g["group"] = g["group"].astype(str)
    hidden = g.loc[g["n"] < config.MIN_CELL, "group"].tolist()
    g = g[g["n"] >= config.MIN_CELL].copy()
    g["share"] = (100 * g["n"] / g["n"].sum()).round(1) if len(g) else []
    return g.sort_values("n", ascending=False), hidden


def period_range(con):
    lo, hi = con.execute("SELECT MIN(recorded_at), MAX(recorded_at) FROM responses").fetchone()
    return lo, hi

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
    """Per service: respondents, mean score and average agreement, each student counted once
    (the same basis as compare_groups), so a service's figure and its groups' figures match."""
    df = con.execute("""WITH per_student AS (
                            SELECT service, response_id, AVG(score) AS ms,
                                   AVG(CASE WHEN agree THEN 1 ELSE 0 END) AS ar
                            FROM ratings GROUP BY service, response_id)
                        SELECT service, COUNT(*) AS n, ROUND(AVG(ms), 2) AS mean_score,
                               ROUND(100 * AVG(ar), 1) AS pct_agree
                        FROM per_student GROUP BY service ORDER BY n DESC""").df()
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


def group_vs_everyone(con, focus, service=None):
    """One group compared with all respondents, overall and per service, with a
    Mann-Whitney test of the group against everyone else (BH-corrected across
    services). Same figures as the dashboard focus report."""
    valid = focus_values(con, focus["dimension"])
    same = {str(v).strip().lower(): v for v in valid}.get(str(focus["value"]).strip().lower())
    if same is not None:                          # "graduate" finds "Graduate"
        focus = {**focus, "value": same}
    if str(focus["value"]) not in valid:
        raise ValueError(f"'{focus['value']}' is not a value of {focus['dimension']}. "
                         f"Valid values: {', '.join(valid[:30])}")
    ids = focus_ids(con, focus)
    k = kpi_summary(con, ids, service)

    def test(svc):
        s = _respondent_scores(con, svc)
        g, rest = s[s["response_id"].isin(ids)], s[~s["response_id"].isin(ids)]
        if len(g) < config.MIN_CELL or len(rest) < config.MIN_CELL:
            return None, len(g)
        return float(mannwhitneyu(g["mean_score"], rest["mean_score"]).pvalue), len(g)

    p_all, _ = test(service)
    rows = []
    for svc in ([service] if service else
                con.execute("SELECT DISTINCT service FROM ratings ORDER BY 1").df()["service"]):
        f = kpi_summary(con, ids, svc)
        p, n = test(svc)
        if n < config.MIN_CELL:
            continue
        rows.append({"service": svc, "n_group": n, "pct_agree_group": f["pct_agree"][0],
                     "pct_agree_everyone": f["pct_agree"][1], "gap_pts": round(f["pct_agree"][0] - f["pct_agree"][1], 1),
                     "mean_group": f["mean_score"][0], "mean_everyone": f["mean_score"][1], "p": p})
    by_service = pd.DataFrame(rows)
    if len(by_service):
        by_service["q"] = _bh(by_service["p"].fillna(1)).round(4)
        by_service["significant"] = by_service["q"] < 0.05
        by_service["p"] = by_service["p"].round(4)
    return {
        "group": f"{focus['dimension']} = {focus['value']}",
        "overall": {"n_group": k["respondents"][0], "n_everyone": k["respondents"][1],
                    "pct_agree_group": k["pct_agree"][0], "pct_agree_everyone": k["pct_agree"][1],
                    "mean_group": k["mean_score"][0], "mean_everyone": k["mean_score"][1],
                    "p": round(p_all, 4) if p_all is not None else None,
                    "significant": bool(p_all is not None and p_all < 0.05)},
        "by_service": by_service,
    }


# CHR appointments: utilization, trends, explaining group differences, and how the
# survey's respondents compare with who actually visits.

CHR_DIMENSIONS = ["faculty", "academic_level", "gender", "age_band", "program", "service_line",
                  "visit_type", "mode", "term"]


def has_chr(con):
    return bool(con.execute("SELECT COUNT(*) FROM duckdb_tables() WHERE table_name = 'appointments'").fetchone()[0])


def chr_is_synthetic(con):
    try:
        return bool(con.execute("SELECT synthetic FROM chr_meta").fetchone()[0])
    except Exception:
        return False


def _appts(con, service_line=None, term=None):
    df = con.execute("SELECT * FROM appointments").df()
    if service_line:
        df = df[df["service_line"].str.lower() == service_line.lower()]
        if df.empty:
            raise ValueError(f"No CHR appointments for service line '{service_line}'.")
    if term:
        df = df[df["term"] == term]
    return df


def _per_student(df):
    """One row per student: attended visits, days in care, services used, no-shows."""
    att = df[df["outcome"] == "attended"]
    g = att.groupby("student_key")
    per = pd.DataFrame({
        "visits": g.size(),
        "days_in_care": (g["start_time"].max() - g["start_time"].min()).dt.days,
        "services_used": g["service_line"].nunique(),
    })
    booked = df.groupby("student_key").agg(booked=("outcome", "size"),
                                           no_shows=("outcome", lambda s: (s == "no_show").sum()))
    return per.join(booked, how="outer").fillna({"visits": 0, "days_in_care": 0, "services_used": 0})


def chr_summary(con, service_line=None, dimension=None, term=None):
    """Students, booked and attended appointments, visits per student, no-show rate and
    median days in care, overall or by group."""
    df = _appts(con, service_line, term)
    if dimension and dimension not in CHR_DIMENSIONS:
        raise ValueError(f"Unknown dimension '{dimension}'. Use one of: {', '.join(CHR_DIMENSIONS)}")
    groups = df.groupby(dimension, dropna=True) if dimension else [("All", df)]
    rows = []
    for key, d in groups:
        per = _per_student(d)
        seen = per[per["visits"] > 0]
        rows.append({dimension or "group": str(key), "n_students": int(len(seen)),
                     "appointments_booked": int(len(d)),
                     "appointments_attended": int((d["outcome"] == "attended").sum()),
                     "visits_per_student": round(seen["visits"].mean(), 2) if len(seen) else None,
                     "median_days_in_care": float(seen["days_in_care"].median()) if len(seen) else None,
                     "no_show_rate_pct": round(100 * (d["outcome"] == "no_show").mean(), 1) if len(d) else None})
    out = pd.DataFrame(rows)
    return out[out["n_students"] >= config.MIN_CELL].reset_index(drop=True)


def chr_trend(con, service_line=None, freq="month"):
    """Attended appointments and students per period, with the same period a year
    earlier where it exists. A change is flagged only if it is at least 10% and
    unlikely to be chance (two-sample Poisson test on the counts)."""
    from scipy.stats import norm
    df = _appts(con, service_line)
    df = df[df["outcome"] == "attended"]
    period = df["start_time"].dt.to_period({"month": "M", "week": "W", "term": "Q"}.get(freq, "M"))
    if freq == "term":
        period = df["term"]
    g = df.groupby(period).agg(appointments=("appointment_key", "size"),
                               n_students=("student_key", "nunique")).reset_index()
    g.columns = ["period", "appointments", "n_students"]
    g["period"] = g["period"].astype(str)
    if freq == "month":
        prev = {p: a for p, a in zip(g["period"], g["appointments"])}
        def last_year(p):
            y, m = p.split("-")
            return prev.get(f"{int(y) - 1}-{m}")
        g["same_month_last_year"] = g["period"].map(last_year)
        a, b = g["appointments"], g["same_month_last_year"]
        g["change_pct"] = (100 * (a - b) / b).round(1)
        z = (a - b) / np.sqrt(a + b)
        g["p"] = (2 * (1 - norm.cdf(np.abs(z)))).round(4)
        g["notable_change"] = (g["p"] < 0.05) & (g["change_pct"].abs() >= 10)
        g.loc[b.isna(), ["p", "notable_change"]] = [np.nan, False]
    return g[g["n_students"] >= config.MIN_CELL].reset_index(drop=True)


def explain_difference(con, dimension, group, compare_to=None, service_line=None, term=None):
    """Break a difference in service use between two groups into its parts: how many
    students come in, how many visits each makes, how long they stay in care, how many
    services they use, and how often they miss appointments. Each part is tested."""
    from scipy.stats import norm
    if dimension not in CHR_DIMENSIONS:
        raise ValueError(f"Unknown dimension '{dimension}'. Use one of: {', '.join(CHR_DIMENSIONS)}")
    df = _appts(con, service_line, term)
    values = sorted(df[dimension].dropna().astype(str).unique())
    lookup = {v.lower(): v for v in values}
    group = lookup.get(str(group).lower(), group)
    compare_to = lookup.get(str(compare_to).lower(), compare_to) if compare_to else None
    if str(group) not in values or (compare_to and str(compare_to) not in values):
        raise ValueError(f"Valid values of {dimension}: {', '.join(values[:30])}")
    a = df[df[dimension].astype(str) == str(group)]
    b = df[df[dimension].astype(str) == str(compare_to)] if compare_to else df[df[dimension].astype(str) != str(group)]
    pa, pb = _per_student(a), _per_student(b)
    pa, pb = pa[pa["visits"] > 0], pb[pb["visits"] > 0]
    if len(pa) < config.MIN_CELL or len(pb) < config.MIN_CELL:
        raise ValueError(f"Fewer than {config.MIN_CELL} students in one of the groups")
    label_b = str(compare_to) if compare_to else f"all other {dimension} groups"
    rows = []
    for name, col, fmt in [("Visits per student", "visits", "{:.2f}"),
                           ("Days in care (first to last visit)", "days_in_care", "{:.0f}"),
                           ("Services used per student", "services_used", "{:.2f}")]:
        p = float(mannwhitneyu(pa[col], pb[col]).pvalue)
        va, vb = pa[col].mean(), pb[col].mean()
        rows.append({"measure": name, "group": round(va, 2), "comparison": round(vb, 2),
                     "ratio": round(va / vb, 2) if vb else None, "p": round(p, 4), "significant": p < 0.05})
    na, nb = (a["outcome"] == "no_show").sum(), (b["outcome"] == "no_show").sum()
    ra, rb = na / len(a), nb / len(b)
    pool = (na + nb) / (len(a) + len(b))
    z = (ra - rb) / np.sqrt(pool * (1 - pool) * (1 / len(a) + 1 / len(b))) if 0 < pool < 1 else 0
    p = float(2 * (1 - norm.cdf(abs(z))))
    rows.append({"measure": "No-show rate (%)", "group": round(100 * ra, 1), "comparison": round(100 * rb, 1),
                 "ratio": round(ra / rb, 2) if rb else None, "p": round(p, 4), "significant": p < 0.05})
    measures = pd.DataFrame(rows)

    # Total visits = students x visits per student, so the gap in totals splits exactly
    # into "how many come in" and "how often each one comes".
    va_tot, vb_tot = int(pa["visits"].sum()), int(pb["visits"].sum())
    r_students, r_vps = len(pa) / len(pb), (va_tot / len(pa)) / (vb_tot / len(pb))
    decomposition = {
        "students": {"group": int(len(pa)), "comparison": int(len(pb)), "ratio": round(r_students, 2)},
        "visits_per_student": {"group": round(va_tot / len(pa), 2), "comparison": round(vb_tot / len(pb), 2),
                               "ratio": round(r_vps, 2)},
        "total_visits": {"group": va_tot, "comparison": vb_tot, "ratio": round(va_tot / vb_tot, 2)},
        "summary": (f"{group}: {len(pa):,} students vs {len(pb):,} for {label_b}, each attending "
                    f"{va_tot / len(pa):.1f} visits vs {vb_tot / len(pb):.1f} ({r_vps:.1f}x), so total "
                    f"visits are {va_tot:,} vs {vb_tot:,}."),
    }
    findings = []
    for r in measures.itertuples():
        if r.significant and r.ratio and abs(r.ratio - 1) >= 0.1:
            direction = "higher" if r.ratio > 1 else "lower"
            pv = "p < 0.001" if r.p < 0.001 else f"p = {r.p}"
            findings.append(f"{r.measure} is {r.ratio:.1f}x {direction} for {dimension} = {group} "
                            f"({r.group} vs {r.comparison}, {pv})")
    return {"group": f"{dimension} = {group}", "comparison": label_b,
            "service_line": service_line or "all services", "term": term or "all terms",
            "measures": measures, "decomposition": decomposition, "findings": findings,
            "not_in_data": ["PHQ-9 scores", "medication", "presenting concerns (CLICC)",
                            "international status (Fact 04/05 empty in the export)"]}


def survey_vs_visits(con, dimension="faculty"):
    """Does the survey hear from the same mix of students who actually visit? Share of
    survey respondents vs share of students seen in CHR, per group."""
    if dimension != "faculty":
        raise ValueError("Only faculty is recorded the same way in both sources so far.")
    survey = con.execute("SELECT faculty AS grp, COUNT(DISTINCT response_id) AS n FROM response_faculty "
                         "GROUP BY 1").df()
    visits = con.execute("SELECT faculty AS grp, COUNT(DISTINCT student_key) AS n FROM appointments "
                         "WHERE outcome = 'attended' AND faculty IS NOT NULL GROUP BY 1").df()
    m = survey.merge(visits, on="grp", how="outer", suffixes=("_survey", "_visits")).fillna(0)
    m["pct_of_survey"] = (100 * m["n_survey"] / m["n_survey"].sum()).round(1)
    m["pct_of_visitors"] = (100 * m["n_visits"] / m["n_visits"].sum()).round(1)
    m["representation_ratio"] = (m["pct_of_survey"] / m["pct_of_visitors"].replace(0, np.nan)).round(2)
    m = m.rename(columns={"grp": "faculty", "n_survey": "n_respondents", "n_visits": "n_students_seen"})
    m = m[(m["n_respondents"] >= config.MIN_CELL) | (m["n_students_seen"] >= config.MIN_CELL)]
    return m.sort_values("representation_ratio").reset_index(drop=True)

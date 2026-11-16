"""Logistic regression for the survey: which parts of a service's experience are linked to
students agreeing with an outcome question when they are weighed together.

logistic_regression(con, service, outcome_item, predictor_items, controls, interaction)
    Outcome: agreeing (4 or 5) with one rating item, students as the unit. Predictors: other
    rating items on their 1-5 scale (odds ratio per one-point higher rating) and, optionally,
    single-choice background questions as groups. Each predictor gets its odds ratio on its
    own and adjusted for everything else in the model, with 95% ranges and BH-corrected
    q-values. With interaction=True, also tests whether the first two questions' links depend
    on each other.

Answers "is X still linked to Y after accounting for Z?", "which of these matter most when
weighed together?" and "are students who feel welcomed more likely to recommend us?".

An odds ratio is a link, not a cause: "linked to higher odds of agreeing", never "increases".
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config, stats

MAX_PREDICTORS = 6      # survey questions in one model
MIN_RESPONDENTS = 50    # students with complete answers
EVENTS_PER_TERM = 10    # students in the smaller outcome group per estimated term
MAX_ABS_COEF = 8.0      # a log-odds this large means one answer almost decides the outcome
COLLINEAR = 0.8         # Spearman between two questions above this: their links blur together
CONTROLS = ("academic_level", "year_of_study", "enrollment_status", "living_arrangement",
            "new_to_ubc", "intl_or_domestic", "first_generation", "indigenous", "racialized",
            "gender", "trans_experience", "disability_any")


def _items(con, service, codes):
    """Wide 1-5 scores (one row per student) for the requested items, plus their wording."""
    r = con.execute("SELECT response_id, item_code, score, item_text FROM ratings "
                    "WHERE service = ? AND score IS NOT NULL", [service]).df()
    if r.empty:
        raise ValueError(f"No survey ratings found for service '{service}'.")
    texts = r.drop_duplicates("item_code").set_index("item_code")["item_text"]
    missing = [c for c in codes if c not in texts.index]
    if missing:
        raise ValueError(f"Not survey questions for {service}: {', '.join(missing)}. "
                         "Call lookup_survey_items and use exact item codes.")
    r = r[r["item_code"].isin(codes)]
    wide = r.pivot_table(index="response_id", columns="item_code", values="score", aggfunc="first")
    return wide.astype(float), {c: str(texts[c]) for c in codes}


def _control(con, dim, ids):
    """One background question for the given students: unknowns dropped, small groups pooled
    into "Other" (dropped if still small). Returns (labels by response_id, reference level,
    pooled levels); the reference is the largest group."""
    if dim not in CONTROLS:
        raise ValueError(f"Cannot account for '{dim}'. Use one of: {', '.join(CONTROLS)}.")
    d = con.execute(f'SELECT response_id, "{dim}" AS grp FROM responses').df()
    d = d[d["response_id"].isin(ids)].drop_duplicates("response_id")
    d["grp"] = d["grp"].astype("string").str.strip()
    d, _ = stats.drop_unknown(d, "grp")
    before = set(d["grp"])
    d = stats.pool_small(d, "grp", min_n=config.EQUITY_MIN_GROUP)
    pooled = sorted(before - set(d["grp"]))
    counts = d["grp"].value_counts()
    if counts.get("Other", config.EQUITY_MIN_GROUP) < config.EQUITY_MIN_GROUP:
        d = d[d["grp"] != "Other"]
        counts = d["grp"].value_counts()
    if len(counts) < 2:
        raise ValueError(f"'{dim}' has fewer than two groups with at least "
                         f"{config.EQUITY_MIN_GROUP} students, so it cannot be accounted for.")
    return d.set_index("response_id")["grp"], str(counts.index[0]), pooled


def _fit(y, X):
    """Logit fit; raises ValueError when the model cannot be trusted."""
    import statsmodels.api as sm
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)
        except np.linalg.LinAlgError:
            raise ValueError("Two of the chosen questions or groups carry the same information, "
                             "so their links cannot be separated. Drop one and retry.") from None
        except Exception as e:  # noqa: BLE001  statsmodels raises several types on separation
            raise ValueError(f"The model could not be fitted ({type(e).__name__}). One answer may "
                             "decide the outcome almost perfectly; use fewer questions.") from None
    separated = any("separation" in str(w.message).lower() for w in caught)
    big = np.abs(res.params.drop("const", errors="ignore")).max() > MAX_ABS_COEF
    if separated or big or not res.mle_retvals.get("converged", True):
        raise ValueError("One question or group decides the outcome almost perfectly, so its link "
                         "cannot be estimated reliably. Use fewer or different questions.")
    return res


def _odds(res, col):
    lo, hi = res.conf_int().loc[col]
    return float(np.exp(res.params[col])), float(np.exp(lo)), float(np.exp(hi)), float(res.pvalues[col])


def logistic_regression(con, service: str, outcome_item: str, predictor_items: list,
                        controls: list | None = None, interaction: bool = False):
    predictors = list(dict.fromkeys(predictor_items or []))
    controls = list(dict.fromkeys(controls or []))
    if not predictors:
        raise ValueError("Choose at least one survey question to weigh against the outcome.")
    if outcome_item in predictors:
        raise ValueError("The outcome question cannot also be a predictor.")
    if len(predictors) > MAX_PREDICTORS:
        raise ValueError(f"Choose at most {MAX_PREDICTORS} survey questions at once.")
    if interaction and len(predictors) < 2:
        raise ValueError("An interaction needs at least two survey questions.")

    wide, texts = _items(con, service, [outcome_item] + predictors)
    answered = wide[outcome_item].notna()
    data = wide[answered].dropna(subset=predictors)
    excluded_incomplete = int(answered.sum()) - len(data)
    excluded_unknown, pooled, refs = 0, {}, {}
    for dim in controls:
        grp, ref, small = _control(con, dim, set(data.index))
        before = len(data)
        data = data.join(grp.rename(dim), how="inner")
        excluded_unknown += before - len(data)
        refs[dim] = ref
        if small:
            pooled[dim] = small

    y = (data[outcome_item] >= 4).astype(int)
    n, n_agree = len(y), int(y.sum())
    if n < MIN_RESPONDENTS:
        raise ValueError(f"Only {n} students answered all the chosen questions for {service}; at "
                         f"least {MIN_RESPONDENTS} are needed for a regression.")

    # Design matrix: neutral column names, readable labels kept alongside.
    cols, labels, blocks = {}, {}, {}
    for j, c in enumerate(predictors):
        cols[f"q{j}"] = data[c]
        labels[f"q{j}"] = ("survey_question", c, texts[c], "each one-point higher rating (1-5)")
        blocks[c] = [f"q{j}"]
    for dim in controls:
        levels = [lv for lv in data[dim].value_counts().index if lv != refs[dim]]
        blocks[dim] = []
        for k, lv in enumerate(levels):
            name = f"{dim}_{k}"
            cols[name] = (data[dim] == lv).astype(float)
            labels[name] = ("background_group", dim, f"{dim.replace('_', ' ')}: {lv}",
                            f"{lv} compared with {refs[dim]}")
            blocks[dim].append(name)
    if interaction:
        a, b = predictors[:2]
        cols["ab"] = (data[a] - data[a].mean()) * (data[b] - data[b].mean())
        labels["ab"] = ("interaction", f"{a}*{b}", f"{texts[a]} × {texts[b]}",
                        "whether the link of one question depends on the other")
    X = pd.DataFrame(cols, index=data.index)

    smaller = min(n_agree, n - n_agree)
    if smaller < EVENTS_PER_TERM * X.shape[1]:
        raise ValueError(f"Only {smaller} students are in the smaller outcome group (agreeing or not), "
                         f"too few to weigh {X.shape[1]} terms at once (need {EVENTS_PER_TERM} per "
                         "term). Use fewer questions or background groups.")
    for dim in controls:
        for name in blocks[dim]:
            inside = y[X[name] == 1]
            if min(int(inside.sum()), int(len(inside) - inside.sum())) < config.MIN_CELL:
                raise ValueError(f"A group of '{dim}' has fewer than {config.MIN_CELL} students "
                                 "agreeing or not agreeing; it cannot be included without risking "
                                 "privacy or a misleading estimate.")

    res = _fit(y, X)
    unadjusted = {}
    for block, names in blocks.items():
        if len(blocks) == 1 and not interaction:
            sub = res
        else:
            sub = _fit(y, X[names])
        for name in names:
            unadjusted[name] = _odds(sub, name)

    keys = list(X.columns)
    q = stats.bh_adjust([float(res.pvalues[k]) for k in keys])
    terms = []
    for k, qv in zip(keys, q):
        kind, ref, text, per = labels[k]
        odds, lo, hi, p = _odds(res, k)
        link = stats.verdict(np.log(lo), np.log(hi), "higher_odds", "lower_odds", "no_clear_link")
        term = {"term": text, "kind": kind, "per": per,
                "odds_ratio": round(odds, 2), "ci": [round(lo, 2), round(hi, 2)],
                "p_value": p, "q_value": round(float(qv), 4),
                "link": link if qv < 0.05 else "no_clear_link"}
        if kind == "survey_question":
            term["item_code"] = ref
        if kind == "background_group":
            term["students"] = int(X[k].sum())
        if k in unadjusted:
            u, ulo, uhi, up = unadjusted[k]
            term["unadjusted_odds_ratio"] = round(u, 2)
            term["unadjusted_ci"] = [round(ulo, 2), round(uhi, 2)]
            term["unadjusted_link"] = stats.verdict(np.log(ulo), np.log(uhi),
                                                    "higher_odds", "lower_odds", "no_clear_link")
        terms.append(term)

    outcome_text = texts[outcome_item]
    findings = []
    for t in terms:
        odds, (lo, hi) = t["odds_ratio"], t["ci"]
        if t["kind"] == "survey_question" and t["link"] != "no_clear_link":
            findings.append(f"Students who rated “{t['term']}” one point higher had {odds} times "
                            f"the odds of agreeing with “{outcome_text}” (range {lo} to {hi}), "
                            "with the other factors weighed at the same time.")
        elif t["kind"] == "background_group" and t["link"] != "no_clear_link":
            findings.append(f"{t['per'][:1].upper() + t['per'][1:]}: {odds} times the odds of agreeing "
                            f"(range {lo} to {hi}), with the other factors weighed at the same time.")
        elif t["kind"] == "interaction" and t["link"] != "no_clear_link":
            findings.append(f"The link of “{texts[predictors[0]]}” with the outcome depends on "
                            f"“{texts[predictors[1]]}” (interaction odds ratio {odds}, range "
                            f"{lo} to {hi}).")
        if (t.get("unadjusted_link", "no_clear_link") != "no_clear_link" and t["link"] == "no_clear_link"
                and t["kind"] == "survey_question"):
            findings.append(f"“{t['term']}” is linked to the outcome on its own (odds ratio "
                            f"{t['unadjusted_odds_ratio']}), but not once the other factors are taken "
                            "into account.")

    cautions = []
    if len(predictors) > 1:
        rho = data[predictors].corr(method="spearman")
        for i, a in enumerate(predictors):
            for b in predictors[i + 1:]:
                if abs(rho.loc[a, b]) > COLLINEAR:
                    cautions.append(f"“{texts[a]}” and “{texts[b]}” are answered almost "
                                    f"the same way (link {rho.loc[a, b]:.2f}); their separate odds "
                                    "ratios are less certain.")

    return {
        "service": service,
        "outcome": {"item_code": outcome_item, "item": outcome_text, "agree_means": "rating 4 or 5"},
        "n": n, "n_agree": n_agree, "pct_agree": round(100 * n_agree / n, 1),
        "excluded_incomplete": excluded_incomplete,
        "excluded_unknown": excluded_unknown,
        "pooled_into_other": [f"{d}: {', '.join(v)}" for d, v in pooled.items()],
        "terms": terms,
        "model": {"terms_estimated": X.shape[1], "pseudo_r2": round(float(res.prsquared), 3),
                  "likelihood_ratio_p": float(res.llr_pvalue)},
        "verdict": "clear_links" if any(t["link"] != "no_clear_link" for t in terms) else "no_clear_links",
        "findings": findings,
        "cautions": cautions,
        "method": ("Logistic regression of agreeing (4-5) with the outcome question, one row per "
                   "student with complete answers. Odds ratios per one-point higher rating, adjusted "
                   "for every other term; unadjusted odds ratios from each term on its own. 95% "
                   "ranges; Benjamini-Hochberg q < 0.05 across terms for a clear link."),
        "note": "An odds ratio is a link, not a cause: say 'linked to higher odds', never 'increases'.",
    }

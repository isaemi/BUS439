"""Key-driver analysis for the survey: which parts of a service's experience are most closely
linked to students' overall view of it, combined with how well each part scores.

priority_items(con, service)
    For each rating item: how strongly it moves with the service's overall outcome item
    (Spearman correlation, students as the unit), and how well it scores (% agree). Items
    above the median on importance and below it on score are "fix first"; a bootstrap over
    students says how stable that is.

A link is not a cause: "most linked to overall satisfaction", never "drives".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import stats

# Overall outcome item per service. Edit here if the survey changes.
OUTCOMES = {
    "counselling": ("CS1_11", "overall satisfaction with the support received"),
    "wellness_adv": ("WA1_8", "overall satisfaction with the support received"),
    "medical": ("MED1_8", "recommending UBC Student Health and Wellbeing as a whole to a friend"),
    "info_desk": ("WCI1_5", "recommending the Wellness Info Desk"),
}
# Never treated as drivers: other overall/recommend items, and an item that sits in the
# counselling block but asks about the Wellness Info Desk.
NOT_DRIVERS = {"CS1_4", "CS1_5"}
SERVICE_ALIASES = {
    "counselling": "counselling", "counseling": "counselling", "counselling services": "counselling",
    "medical": "medical", "medical clinic": "medical", "student health": "medical", "clinic": "medical",
    "wellness_adv": "wellness_adv", "wellness advising": "wellness_adv", "wellness advisor": "wellness_adv",
    "info_desk": "info_desk", "info desk": "info_desk", "wellness info desk": "info_desk",
}
MIN_RESPONDENTS = 50   # students who answered the outcome item
MIN_PAIRS = 30         # students who answered both the item and the outcome
STABLE = 0.70          # share of resamples an item must be "fix first" in to be a priority
FLAT_SPREAD = 0.15     # importance spread below this: no single item stands out
ACCESS = re.compile(r"(?i)\b(book|booking|wait|waiting|appointment|access|schedul|get in)")

QUADRANTS = {(True, False): "fix_first", (True, True): "keep_up",
             (False, False): "monitor", (False, True): "possibly_over_served"}


def _service(name):
    key = str(name or "").strip().lower().replace("-", " ")
    svc = SERVICE_ALIASES.get(key)
    if svc is None:
        raise ValueError(f"Unknown service '{name}'. Use one of: {', '.join(OUTCOMES)}")
    return svc


def _load(con, svc):
    r = con.execute("SELECT response_id, item_code, score, agree, item_text FROM ratings "
                    "WHERE service = ?", [svc]).df()
    try:
        dup = set(con.execute("SELECT item_code FROM items WHERE duplicate_of IS NOT NULL")
                  .df()["item_code"])
    except Exception:
        dup = set()
    return r, dup


def _spearman_all(scores, outcome):
    """Spearman correlation of every item column with the outcome, pairwise complete."""
    return scores.corrwith(outcome, method="spearman")


def priority_items(con, service: str):
    svc = _service(service)
    out_code, out_label = OUTCOMES[svc]
    r, dup = _load(con, svc)
    texts = r.drop_duplicates("item_code").set_index("item_code")["item_text"]
    scores = r.pivot_table(index="response_id", columns="item_code", values="score", aggfunc="first")
    agree = r.pivot_table(index="response_id", columns="item_code", values="agree", aggfunc="first")
    if out_code not in scores:
        raise ValueError(f"The outcome question for {svc} is not in the survey data.")
    answered = scores[out_code].notna()
    n_out = int(answered.sum())
    if n_out < MIN_RESPONDENTS:
        raise ValueError(f"Only {n_out} students answered the overall question for {svc}; at "
                         f"least {MIN_RESPONDENTS} are needed to say what matters most.")
    scores, agree = scores[answered], agree[answered]
    drivers = [c for c in scores.columns
               if c != out_code and c not in NOT_DRIVERS and c not in dup
               and scores[c].notna().sum() >= MIN_PAIRS]
    if len(drivers) < 3:
        raise ValueError(f"Too few survey questions for {svc} have enough answers to compare.")
    X, A, y = scores[drivers].astype(float), agree[drivers].astype(float), scores[out_code].astype(float)

    data = pd.concat([X.add_prefix("s_"), A.add_prefix("a_"), y.rename("outcome")], axis=1)
    k = len(drivers)

    def stat(d):
        rho = _spearman_all(d[[f"s_{c}" for c in drivers]].set_axis(drivers, axis=1), d["outcome"])
        perf = d[[f"a_{c}" for c in drivers]].mean().to_numpy()
        return np.concatenate([rho.to_numpy(), perf])

    bs = stats.bootstrap(data, stat)
    rho, perf = bs["estimate"][:k], bs["estimate"][k:]
    draws = bs["draws"]
    d_rho, d_perf = draws[:, :k], draws[:, k:]
    hi_imp = d_rho > np.nanmedian(d_rho, axis=1, keepdims=True)
    lo_perf = d_perf < np.nanmedian(d_perf, axis=1, keepdims=True)
    fix_share = (hi_imp & lo_perf).mean(axis=0)

    med_rho, med_perf = np.median(rho), np.median(perf)
    items = []
    for j, c in enumerate(drivers):
        quad = QUADRANTS[(bool(rho[j] > med_rho), bool(perf[j] >= med_perf))]
        text = str(texts.get(c, c))
        items.append({
            "item": text, "item_code": c,
            "link_to_outcome": round(float(rho[j]), 2),
            "link_ci": [round(float(bs["low"][j]), 2), round(float(bs["high"][j]), 2)],
            "agree_pct": round(100 * float(perf[j]), 1),
            "students": int(X[c].notna().sum()),
            "quadrant": quad,
            "fix_first_in_pct_of_resamples": round(100 * float(fix_share[j])),
            "access_item": bool(ACCESS.search(text)),
        })
    order = {"fix_first": 0, "keep_up": 1, "monitor": 2, "possibly_over_served": 3}
    items.sort(key=lambda it: (order[it["quadrant"]], -it["link_to_outcome"]))

    spread = float(np.max(rho) - np.min(rho))
    def credible(code):
        """Stable in the resamples AND its link's lower bound is above the other questions'
        median link, so it is more strongly linked than a typical question, not just by chance."""
        j = drivers.index(code)
        others = np.delete(rho, j)
        return fix_share[j] >= STABLE and bs["low"][j] > np.median(others)

    priorities = [it["item"] for it in items if it["quadrant"] == "fix_first" and credible(it["item_code"])]
    if spread < FLAT_SPREAD:
        verdict = "no_single_item_stands_out"
    elif priorities:
        verdict = "clear_priorities"
    else:
        verdict = "no_stable_priority"

    notes = [
        f"The overall measure is {out_label} (answered by {n_out} students). Each question's "
        "link is how closely its answers move with that overall measure (Spearman correlation, "
        "0 = no link, 1 = perfect). 'Fix first' = more strongly linked than the median question "
        "and scoring below the median.",
        "A link is not a cause: say 'most closely linked to', never 'drives'. Improving a fix-first "
        "area is a reasonable thing to try, not a guaranteed result.",
        f"A priority must be 'fix first' in at least {round(100 * STABLE)}% of 1,000 re-samples of "
        "students, and its link must be clearly stronger than a typical question's; otherwise the "
        "ranking is not reliable enough to act on.",
    ]
    if verdict == "no_single_item_stands_out":
        notes.append("The links are all similar, which usually means students who are happy "
                     "overall rate everything higher. No single question stands out.")
    if any(it["access_item"] for it in items):
        notes.append("Questions about booking or waiting are likely understated: students who "
                     "could not get an appointment never received the survey.")
    if svc == "medical":
        notes.append("For the medical clinic the overall question is about recommending SHW as "
                     "a whole, not the clinic alone.")
    return {"service": svc, "outcome": out_label, "respondents": n_out,
            "verdict": verdict, "priorities": priorities, "items": items,
            "link_spread": round(spread, 2), "notes": notes}

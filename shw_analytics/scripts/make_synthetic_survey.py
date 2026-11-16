"""Make a synthetic Qualtrics survey export (numeric values, 3 header rows) for building and
demoing without real data. Nothing in it is real.

    python scripts/make_synthetic_survey.py                 # -> data/exports/synthetic_survey.csv
    python scripts/make_synthetic_survey.py --n 900 --out path.csv

Unlike a random fill, each student ticks ONE faculty (a few tick two), and patterns are
planted so the analysis has something true to find:
  * Counselling: "focused on the issues most important to me" is closely tied to overall
    satisfaction but scores low (a clear key driver); booking scores low but matters little;
    front-desk courtesy scores high and matters little.
  * Students reporting a disability rate counselling about 12 points lower.
  * Medical: the longer the reported wait, the lower the ratings.
  * Over time: responses run from September 2025 to October 2026. This fall, students reporting
    a disability rate counselling clearly lower than they did last fall, and one medical question
    (being seen close to the appointment time) scores higher. Everything else is stable.
  * Comments: realistic written feedback whose tone follows each student's ratings. Students
    with long medical waits write about waiting; disabled students raise accessibility in
    counselling; graduate students raise scheduling. About 3% include a name, email or number,
    so you can see that loading removes them.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import config  # noqa: E402
from shw_analytics.ingest import load_dictionary  # noqa: E402

STEM = "Thinking about your most recent visit, how much do you agree? - "
# item -> (wording, link to overall satisfaction 0..1, shift: + is easier to agree with)
ITEMS = {
    "CS1": {1: ("I could book an appointment easily", 0.15, -0.5),
            2: ("Front desk staff were courteous and respectful", 0.10, 0.9),
            3: ("I felt welcomed and included in the space", 0.45, 0.2),
            4: ("I would recommend UBC Student Health and Wellbeing to a friend", 0.90, 0.1),
            5: ("I would recommend the Wellness Info Desk to anyone who needs help finding health and wellness resources on campus", 0.30, 0.0),
            6: ("My counsellor and I focused on the issues most important to me", 0.85, -0.6),
            7: ("My counsellor helped me understand my situation in more effective ways", 0.70, -0.1),
            8: ("I have made positive changes in the areas that concerned me", 0.60, 0.1),
            9: ("I was informed about other campus resources", 0.25, -0.2),
            10: ("The length of my session was right for me", 0.40, 0.5),
            11: ("Overall, I am satisfied with the support I received", 1.00, 0.0)},
    "MED1": {1: ("I could book an appointment easily", 0.20, -0.3),
             2: ("Front desk staff were courteous and respectful", 0.30, 0.6),
             3: ("I had a positive experience with the provider", 0.80, 0.3),
             4: ("My concerns were taken seriously", 0.75, 0.1),
             5: ("I understood the next steps after my visit", 0.55, 0.2),
             6: ("I was seen close to my appointment time", 0.35, -0.4),
             7: ("I felt welcomed and included in the space", 0.45, 0.3),
             8: ("I would recommend UBC Student Health and Wellbeing to a friend", 1.00, 0.1)},
    "WA1": {1: ("I could book an appointment easily", 0.20, 0.2),
            2: ("My wellness advisor listened to my concerns", 0.75, 0.5),
            3: ("I left with a clear plan", 0.80, -0.3),
            4: ("I learned about resources that fit my needs", 0.60, 0.1),
            5: ("I felt welcomed and included in the space", 0.40, 0.4),
            8: ("Overall, I am satisfied with the support I received", 1.00, 0.1),
            10: ("Overall, I am satisfied with the support I received", 1.00, 0.1)},
    "WCI1": {1: ("Staff at the desk were helpful", 0.7, 0.4), 5: ("I would recommend the Wellness Info Desk", 1.0, 0.2)},
}
SERVICE_SHARE = {8: 0.40, 3: 0.38, 9: 0.20, 10: 0.02}     # Int1 codes: counselling, medical, WA, info desk
BLOCK = {8: "CS1", 3: "MED1", 9: "WA1", 10: "WCI1"}
# ---- realistic written comments -------------------------------------------------------
# Sentences by service, theme and tone. A comment is built from one or two of them (plus,
# sometimes, a suggestion), and its tone follows the student's own ratings, so what students
# write agrees with how they rated. Themes match the fixed list in comments.py.
SENT = {
  "counselling": {
    "feeling_heard": {
      "pos": ["My counsellor really listened and didn't rush me.",
              "I felt genuinely heard for the first time in a long while.",
              "We focused on what I actually came in for, which I appreciated.",
              "My counsellor remembered details from our last session, which made a big difference.",
              "It felt like a conversation, not a checklist."],
      "neg": ["It felt like my counsellor had a script and wasn't really listening.",
              "We spent most of the session on things I didn't come in for.",
              "I left feeling like my main concern was never addressed.",
              "The advice was very generic, like something I could have found online.",
              "I was cut off a few times and didn't feel heard."]},
    "booking_and_access": {
      "pos": ["Booking online was quick and easy.", "I got a same-week appointment, which really helped."],
      "neg": ["It took several tries to find any open appointment online.",
              "All the appointments were during my classes.",
              "The booking system kept timing out.",
              "I couldn't figure out which type of appointment to book."]},
    "outcomes_and_follow_up": {
      "pos": ["I left with practical strategies I've actually been using.",
              "The follow-up plan was clear and realistic."],
      "neg": ["I wasn't sure if I was supposed to book a follow-up or not.",
              "One session wasn't enough and I didn't know how to get more support.",
              "I expected ongoing counselling but it ended after one visit with no explanation.",
              "Nobody followed up after I was referred elsewhere."]},
    "intake_forms": {
      "pos": ["The intake questions were short and made sense."],
      "neg": ["I had to repeat my whole story to three different people.",
              "The intake forms were very long when I was already struggling.",
              "Some intake questions felt intrusive and I didn't know why they were asked."]},
    "physical_space": {
      "pos": ["The space was calm and comfortable."],
      "neg": ["The waiting area is very open and I worried people I knew would see me.",
              "I could hear the conversation in the next room.",
              "The room was cold and felt clinical."]},
    "identity_and_inclusion": {
      "pos": ["My counsellor was respectful of my identity and background."],
      "neg": ["I had to explain my disability and accommodations again from scratch.",
              "The building wasn't easy to get around with my mobility aid.",
              "I didn't feel my cultural background was understood."]},
    "information_and_resources": {
      "pos": ["I was pointed to resources I didn't know existed."],
      "neg": ["I was given a list of websites instead of real help."]},
  },
  "medical": {
    "wait_time": {
      "pos": ["I was seen right on time.", "The wait was short considering how busy it was."],
      "neg": ["I waited over forty minutes past my appointment time.",
              "It took almost three weeks to get an appointment.",
              "By the time I got in, I'd already been sick for a week.",
              "Nobody told us the doctor was running late."]},
    "front_desk_and_staff": {
      "pos": ["The front desk staff were kind and helpful.", "The nurse was patient and explained everything."],
      "neg": ["The front desk was abrupt with me when I asked a question.",
              "Staff seemed rushed and stressed.", "I was sent to the wrong waiting area."]},
    "booking_and_access": {
      "pos": ["Booking was straightforward."],
      "neg": ["It's very hard to get a same-day appointment when you're actually sick.",
              "The phone line was busy every time I called.",
              "I couldn't book online and had to come in person."]},
    "information_and_resources": {
      "pos": ["The doctor explained my results clearly.", "I knew exactly what to do next."],
      "neg": ["I left without understanding my next steps.",
              "Nobody explained how to get my prescription refilled.",
              "I wasn't told whether my insurance would cover the medication."]},
    "privacy_and_confidentiality": {
      "pos": ["I felt my privacy was respected."],
      "neg": ["The check-in desk asked about my symptoms where everyone could hear.",
              "I wasn't sure who would see my health information."]},
    "outcomes_and_follow_up": {
      "pos": ["The follow-up call afterwards was a nice touch."],
      "neg": ["I was told someone would call with my results and nobody did."]},
  },
  "wellness_adv": {
    "information_and_resources": {
      "pos": ["My advisor connected me with exactly the right resources.",
              "I learned about supports I had no idea UBC offered."],
      "neg": ["I was just handed a pamphlet.", "The resources I was given didn't fit my situation."]},
    "feeling_heard": {
      "pos": ["My advisor took the time to understand my situation."],
      "neg": ["The meeting felt rushed and generic."]},
    "outcomes_and_follow_up": {
      "pos": ["I left with a clear plan for the rest of term."],
      "neg": ["I wasn't sure what was supposed to happen after the meeting."]},
    "booking_and_access": {
      "pos": ["It was easy to get a drop-in appointment."],
      "neg": ["I didn't know wellness advising existed until a friend told me."]},
  },
  "info_desk": {
    "information_and_resources": {"pos": ["The desk pointed me to the right place quickly."],
                                  "neg": ["I was sent between two offices before anyone could help."]}},
}
# How often each theme comes up, by service.
THEME_WEIGHTS = {
  "counselling": {"feeling_heard": 5, "booking_and_access": 3, "outcomes_and_follow_up": 3, "intake_forms": 2,
                  "physical_space": 1.5, "identity_and_inclusion": 1, "information_and_resources": 1},
  "medical": {"wait_time": 4, "front_desk_and_staff": 3, "booking_and_access": 3, "information_and_resources": 2,
              "privacy_and_confidentiality": 1.5, "outcomes_and_follow_up": 1},
  "wellness_adv": {"information_and_resources": 4, "feeling_heard": 2, "outcomes_and_follow_up": 2,
                   "booking_and_access": 1.5},
  "info_desk": {"information_and_resources": 1},
}
OPENERS = {"pos": ["", "", "Overall a good experience. ", "Really grateful for this service. ", "Thank you! "],
           "neg": ["", "", "Honestly disappointing. ", "I almost didn't come back. ", "Mixed feelings. "]}
SUGGEST = ["It would help to have more evening appointments.", "Please add more appointment times.",
           "A text reminder the day before would help.", "More counsellors would help with the wait.",
           "It would be nice to see the same person each time.", "Clearer information on the website would help."]
EXTRA = {   # what particular groups tend to raise
  "graduate": ["As a grad student, it's hard to find times that fit around my research and TA hours.",
               "Most appointments are during the day when I'm in the lab."],
  "international": ["As an international student, I wasn't sure what my insurance covers.",
                    "It took me a while to understand how the system works here."],
}
PII = ["Dr. Patel was great.", "Counsellor Amy was very kind.", "You can reach me at jamie.lee@student.ubc.ca.",
       "Please call me back at 604-555-0182.", "My student number is 12345678 if you need it."]


def _comment(rng, block, sat, wait, disability, level, intl):
    svc = {"CS1": "counselling", "MED1": "medical", "WA1": "wellness_adv", "WCI1": "info_desk"}[block]
    weights = dict(THEME_WEIGHTS[svc])
    if svc == "counselling" and disability:
        weights["identity_and_inclusion"] = 10         # planted: accessibility comes up for disabled students
    if svc == "medical" and wait == 4:
        weights["wait_time"] = 14                      # planted: the longest waits are what people write about
    themes = list(weights)
    w = np.array([weights[t] for t in themes], float)
    picked = list(rng.choice(themes, size=min(len(themes), 1 + int(rng.random() < 0.35)),
                               replace=False, p=w / w.sum()))
    p_neg = 1 / (1 + np.exp(1.6 * sat))               # unhappier students write more negative comments
    tone = "mixed" if rng.random() < 0.15 else ("neg" if rng.random() < p_neg else "pos")
    parts = []
    for i, t in enumerate(picked):
        bank = SENT[svc][t]
        if t == "identity_and_inclusion" and disability and svc == "counselling":
            bank = {"pos": bank["pos"], "neg": bank["neg"][:2]}   # the disability-specific sentences
        side = ("pos" if i == 0 else "neg") if tone == "mixed" else tone
        parts.append(str(rng.choice(bank.get(side) or bank["pos"])))
    if tone == "mixed" and len(parts) == 1:
        other = [t for t in SENT[svc] if t != picked[0]]
        parts.append(str(rng.choice(SENT[svc][str(rng.choice(other))]["neg"]))) if other else None
    if tone == "mixed" and len(parts) == 2:
        parts[1] = "But " + parts[1][0].lower() + parts[1][1:]
    if level == 2 and rng.random() < 0.35:
        parts.append(str(rng.choice(EXTRA["graduate"])))
    if intl and rng.random() < 0.25:
        parts.append(str(rng.choice(EXTRA["international"])))
    if tone != "pos" and rng.random() < 0.3:
        parts.append(str(rng.choice(SUGGEST)))
    if rng.random() < 0.03:
        parts.append(str(rng.choice(PII)))             # to show that names, emails and numbers are scrubbed
    return (str(rng.choice(OPENERS["neg" if tone != "pos" else "pos"])) + " ".join(parts)).strip()


# What each answer code means in THIS synthetic export. Written to data/value_labels.yaml so groups
# show as names. A real export needs its own labels (from the survey definition), which go in
# dictionary.yaml and take priority over this file.
LABELS = {
    "D1": {1: "Undergraduate", 2: "Graduate", 3: "Other (e.g. visiting, unclassified)"},
    "D2": {1: "New to UBC this year", 2: "Returning"},
    "D3": {1: "1st year", 2: "2nd year", 3: "3rd year", 4: "4th year", 5: "5th year or more"},
    "D4": {1: "Full-time", 2: "Part-time"},
    "D7": {1: "International", 2: "Domestic"},
    "D8": {1: "On campus", 2: "Off campus with family", 3: "Off campus with roommates",
           4: "Off campus alone", 5: "Other"},
    "D9": {1: "Indigenous", 2: "Not Indigenous"},
    "D10": {1: "Racialized", 2: "Not racialized"},
    "D12": {1: "First generation", 2: "Not first generation", 3: "Unsure"},
    "D13": {1: "Woman", 2: "Man", 3: "Non-binary", 4: "Another gender"},
    "D14": {1: "Trans experience", 2: "No trans experience"},
}


def write_labels(path):
    import yaml
    body = yaml.safe_dump({"value_labels": {q: {str(k): v for k, v in m.items()} for q, m in LABELS.items()}},
                          allow_unicode=True, sort_keys=False)
    Path(path).write_text("# Answer labels for the SYNTHETIC survey export (made by make_synthetic_survey.py).\n"
                          "# Delete this file when real data is loaded; real labels belong in dictionary.yaml.\n" + body)


START = pd.Timestamp("2025-09-01")
SPAN_MINUTES = int((pd.Timestamp("2026-11-01") - START).total_seconds() // 60)
RECENT = pd.Timestamp("2026-09-01")          # planted changes over time start here
IMPROVED = ("MED1", 6)                       # "I was seen close to my appointment time"


def _likert(latent, shift):
    return np.clip(np.round(3.55 + 0.9 * latent + shift), 1, 5).astype(int)


def make(n=3600, seed=11):
    rng = np.random.default_rng(seed)
    d = load_dictionary()
    fac = d["checkbox"]["faculty"]["choices"]
    fac_w = np.array([8, 2, 1, 22, 1, 8, 1, 1, 5, 1, 3, 6, 1, 4, 4, 3, 1, 6, 2, 1, 1, 2, 20, 1, 3, 1], float)
    rows = []
    for i in range(n):
        svc = int(rng.choice(list(SERVICE_SHARE), p=list(SERVICE_SHARE.values())))
        block = BLOCK[svc]
        level = int(rng.choice([1, 2, 3], p=[0.76, 0.20, 0.04]))           # see LABELS["D1"]
        disability = rng.random() < 0.22
        wait = int(rng.choice([1, 2, 3, 4], p=[0.35, 0.3, 0.2, 0.15]))
        # Responses are spread over the same 14 months as the synthetic appointments, so results
        # can be compared over time.
        when = START + pd.Timedelta(minutes=int(rng.integers(0, SPAN_MINUTES)))
        this_fall = when >= RECENT
        sat = rng.normal(0, 1)
        if svc == 8 and disability:
            sat -= 0.6 + (1.4 if this_fall else 0.0)                    # a gap that has WIDENED this fall
        if svc == 3:
            sat -= 0.3 * (wait - 1)                                     # the longer the wait, the lower the ratings
        r = {"ResponseId": f"R_syn{i:05d}", "DistributionChannel": "anonymous", "Finished": "1",
             "Progress": "100", "Int1": str(svc),
             "RecordedDate": str(when),
             "D1": str(level), "D2": str(rng.integers(1, 3)), "D3": str(rng.integers(1, 6)),
             "D4": str(rng.choice([1, 2], p=[0.9, 0.1])), "D7": str(rng.choice([1, 2], p=[0.28, 0.72])),
             "D8": str(rng.integers(1, 6)), "D9": str(rng.choice([1, 2], p=[0.04, 0.96])),
             "D10": str(rng.choice([1, 2], p=[0.45, 0.55])), "D11": str(rng.integers(1, 8)),
             "D12": str(rng.choice([1, 2, 3], p=[0.25, 0.65, 0.10])),
             "D13": str(rng.choice([1, 2, 3, 4], p=[0.58, 0.34, 0.05, 0.03])),
             "D14": str(rng.choice([1, 2], p=[0.04, 0.96])), "D15": str(rng.integers(1, 9)),
             "D5_1_TEXT": str(int(2026 - np.clip(rng.normal(28 if level == 2 else 22, 3), 17, 50)))}
        for q, (text, link, shift) in ITEMS[block].items():
            if rng.random() < 0.04:
                continue                                                    # a skipped question
            latent = link * sat + np.sqrt(max(0.0, 1 - link ** 2)) * rng.normal()
            if this_fall and (block, q) == IMPROVED:
                latent += 0.75                                              # one thing that got better
            score = int(_likert(latent, shift))
            if block == "WA1" and q == 10 and "WA1_8" in r:
                score = int(r["WA1_8"])                                     # the duplicated question
            r[f"{block}_{q}"] = str(score + 12) if block == "WCI1" else str(score)   # WCI1 exports 13-17
        if svc == 8:
            r["CS2"] = str(wait)
        if svc == 3:
            r["MED2"] = str(wait)
        picks = {int(rng.choice(len(fac), p=fac_w / fac_w.sum())) + 1}
        if rng.random() < 0.05:
            picks.add(int(rng.choice(len(fac), p=fac_w / fac_w.sum())) + 1)
        for k in picks:
            r[f"D6_{k}"] = "1"
        r[f"D16_{int(rng.choice([1, 4, 5, 7])) if disability else 9}"] = "1"
        if rng.random() < 0.45 + 0.15 * (sat < -0.5):
            intl = r["D7"] == "1"
            r[{8: "CS4", 3: "MED5", 9: "WA4"}.get(svc, "Comment")] = _comment(
                rng, block, sat, wait, disability, level, intl)
        rows.append(r)
    df = pd.DataFrame(rows)
    texts = {c: c for c in df.columns}
    for block, items in ITEMS.items():
        for q, (text, _, _) in items.items():
            texts[f"{block}_{q}"] = STEM + text
    meta = {c: json.dumps({"ImportId": c}) for c in df.columns}
    return pd.concat([pd.DataFrame([texts, meta]), df], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3600)      # about a year's responses at SHW's real volume
    ap.add_argument("--out", default=str(config.DB_PATH.parent / "exports" / "synthetic_survey.csv"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make(a.n).to_csv(out, index=False)
    print(f"Wrote {out}: {a.n} synthetic survey responses")
    labels = config.DB_PATH.parent / "value_labels.yaml"
    if out.parent == config.DB_PATH.parent / "exports":       # only when writing into the app's data folder
        write_labels(labels)
        print(f"Wrote {labels}: answer labels, so groups show as names (e.g. Graduate) instead of codes")

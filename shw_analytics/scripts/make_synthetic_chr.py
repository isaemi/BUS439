"""Make a synthetic TELUS CHR appointments export with the same columns and quirks
as the real one (blank lines, group-visit template rows, "\\r\\r\\n" line endings),
for building and demoing without real data. Nothing in it is real.

    python scripts/make_synthetic_chr.py                # -> data/chr_exports/synthetic_chr.csv
    python scripts/make_synthetic_chr.py --students 3000 --out path.csv

Planted patterns, so the analysis has something true to find:
  * Graduate students attend about twice as many counselling sessions each as
    undergraduates, and stay in care longer, though fewer of them come in.
  * Demand peaks in October-November and March.
  * Fall 2026 has about 15% more appointments than Fall 2025.
  * Phone/video appointments have more no-shows than in-person ones.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import config  # noqa: E402
from shw_analytics.ingest import load_dictionary  # noqa: E402

COLUMNS = list(load_dictionary()["chr"]["columns"]) + [
    "Group Visit Appointment Types Template Name", "Group Visit Provider Users Provider Names"]

FACULTIES = {"Arts": 22, "Science": 20, "Applied Science (Engineering)": 12, "Business": 8,
             "Medicine": 6, "Land and Food Systems": 4, "Forestry": 3, "Education": 5,
             "Kinesiology": 4, "Graduate and Postdoctoral Studies": 6, "Law": 3, "Music": 2,
             "Pharmaceutical Sciences": 2, "UBC Vantage College": 3}
PROGRAMS = {"Arts": "BA", "Science": "BSc", "Applied Science (Engineering)": "BASc", "Business": "BCom",
            "Medicine": "FoM MD Undergraduate", "Land and Food Systems": "BSc LFS", "Forestry": "BSF",
            "Education": "BEd", "Kinesiology": "BKin", "Graduate and Postdoctoral Studies": "MA/MSc/PhD",
            "Law": "JD", "Music": "BMus", "Pharmaceutical Sciences": "BPharm", "UBC Vantage College": "VC"}
MONTH_WEIGHT = {9: 0.8, 10: 1.4, 11: 1.4, 12: 0.6, 1: 0.9, 2: 1.0, 3: 1.4, 4: 0.8, 5: 0.3, 6: 0.3, 7: 0.3, 8: 0.4}


def make(students=2500, seed=7):
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-09-01", "2026-10-31", freq="D")
    w = np.array([MONTH_WEIGHT[d.month] * (1.15 if d >= pd.Timestamp("2026-09-01") else 1) for d in days])
    rows, appt_id = [], 2_000_000
    providers = {"Counselling": [f"Clinician C{i:02d}" for i in range(1, 16)],
                 "Medical": [f"Clinician M{i:02d}" for i in range(1, 13)],
                 "Wellness Advising": [f"Clinician W{i:02d}" for i in range(1, 6)]}
    fac_names, fac_p = list(FACULTIES), np.array(list(FACULTIES.values()), float)
    for s in range(students):
        sid = 600_000 + s
        faculty = rng.choice(fac_names, p=fac_p / fac_p.sum())
        grad = faculty == "Graduate and Postdoctoral Studies" or rng.random() < 0.12
        level = "Graduate" if grad else "Undergraduate"
        gender = rng.choice(["Female", "Male", "Non-binary", ""], p=[0.58, 0.35, 0.05, 0.02])
        age = int(np.clip(rng.normal(27 if grad else 21, 3), 17, 45))
        line = rng.choice(["Counselling", "Medical", "Wellness Advising"], p=[0.5, 0.4, 0.1])
        mean_visits = {"Counselling": 6.0 if grad else 3.0, "Medical": 2.0, "Wellness Advising": 1.5}[line]
        n = max(1, rng.poisson(mean_visits))
        start = days[rng.choice(len(days), p=w / w.sum())]
        gap = 10 if line == "Counselling" else 21
        provider = rng.choice(providers[line])
        when = start
        for k in range(n):
            if when > days[-1]:
                break
            mode = "Phone/Video" if rng.random() < 0.4 else "In Person"
            p_noshow = 0.12 if mode == "Phone/Video" else 0.05
            r = rng.random()
            status = "no_show" if r < p_noshow else ("cancelled" if r < p_noshow + 0.07 else "note_signed")
            kind = "Initial" if k == 0 else "Follow Up"
            program = " (MDUP)" if faculty == "Medicine" and line == "Counselling" else ""
            hour = int(rng.integers(9, 17))
            t0 = when.replace(hour=hour)
            mins = 60 if line == "Counselling" else 30
            dx = "No Diagnosis Recorded" if line != "Medical" else rng.choice(
                ["No Diagnosis Recorded", "Respiratory", "Sexual health", "Mental health", "Injury"])
            rows.append({
                "Appointments Appointment ID": appt_id, "Appointments Respondent ID": sid,
                "Appointments Service": f"{line}{program} - {kind}" + (" (Phone/Video)" if mode == "Phone/Video" else ""),
                "Service/Dx Code": dx, "Appointments Start Time": t0.strftime("%Y-%m-%d %H:%M"),
                "Appointments Reason": f"{line} Appointment",
                "Appointments Appointment Practitioner": provider,
                "Appointments Appointment Status": status, "Appointments Appointment Tag": rng.choice(["E1", "E3", "E5"]),
                "Appointments Locations Location Name": f"{line} - Main",
                "Appointments Until Time": (t0 + pd.Timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M"),
                "Appointments Is Group Visit (Yes / No)": "No",
                "Patient Data Fact Sheet Fact 01": faculty, "Patient Data Fact Sheet Fact 02": level,
                "Patient Data Fact Sheet Fact 03": PROGRAMS[faculty] + ",",
                "Patient Data Fact Sheet Fact 06": gender, "Patients Age": age,
            })
            appt_id += 1
            when = when + pd.Timedelta(days=int(gap * (1.4 if grad else 1) + rng.integers(0, 7)))
    df = pd.DataFrame(rows, columns=COLUMNS)
    # Mimic the export's extra rows: blanks and group-visit templates without an appointment.
    extra = pd.DataFrame({c: [None] * 300 for c in COLUMNS})
    extra.loc[:49, "Group Visit Appointment Types Template Name"] = "CS - Group (Group)"
    extra.loc[:49, "Group Visit Provider Users Provider Names"] = "Clinician C01, Clinician C02"
    return pd.concat([df, extra], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=int, default=2500)
    ap.add_argument("--out", default=str(config.DB_PATH.parent / "chr_exports" / "synthetic_chr.csv"))
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make(a.students)
    out.write_text(df.to_csv(index=False).replace("\n", "\r\r\n"))
    print(f"Wrote {out}: {df['Appointments Appointment ID'].notna().sum()} synthetic appointments")

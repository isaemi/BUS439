"""Term briefing: what changed and what is worth a look, without anyone having to ask.

build()            runs a fixed set of tested analyses across services, corrects for having run
                   them all, ranks what holds up, and lists what stayed the same and what could
                   not be checked. No model is involved: the same data gives the same briefing.
compare(b, prev)   which findings are new, still there, or no longer showing since last time
save(b)            keeps a briefing in data/briefings/ so the next one can be compared with it
for_chat(b)        a compact version for the assistant ("what should we know this term?")
graphic(finding)   the chart for one finding
to_html(b, ...)    a one-page briefing to download

Every finding carries the dates and group sizes it rests on, and says how it was tested.
"""
from __future__ import annotations

import datetime as dt
import json
import math

from . import analysis, config, stats, tools

MAX_FINDINGS = 5
MAX_PER_KIND = 2
BATTERY_FDR = 0.10          # share of reported findings we accept may be chance, across everything run
METRICS = {"no_show_rate": ("the no-show rate", "%", " points"),
           "visits_per_student": ("visits per student", "", ""),
           "days_in_care": ("days in care", " days", " days")}
SURVEY_SERVICES = {"counselling": "Counselling", "medical": "Medical", "wellness_adv": "Wellness Advising"}
DIR = config.DB_PATH.parent / "briefings"
_cache: dict = {}


def _call(name, **args):
    args = {k: v for k, v in args.items() if v is not None}
    try:
        res = tools.call(name, args)
    except Exception as e:                       # a failed analysis never stops the briefing
        res = {"error": str(e)}
    return (name, args, res if isinstance(res, dict) else {"error": "no result"})


def _p_from_ci(est, ci):
    """Approximate two-sided p from an estimate and its 95% range (normal approximation)."""
    se = (ci[1] - ci[0]) / 3.92
    if not se:
        return 1.0
    z = abs(est) / se
    return float(math.erfc(z / math.sqrt(2)))


def _span(periods):
    from .figures import _periods
    return _periods(periods)[0]


def _svc(name):
    return name or "All services"


# ---- one function per kind of finding -------------------------------------------------

def _from_change(step, seen_volume):
    """Findings and 'unchanged' lines from one same-dates comparison."""
    _, args, r = step
    found, same = [], []
    svc, metric = _svc(args.get("service_line")), args["metric"]
    name, unit, pts = METRICS[metric]
    span = _span(r["periods"])
    v = r["volume_change"]
    if svc not in seen_volume:
        seen_volume.add(svc)
        if v.get("verdict") in ("increase", "decrease"):
            up = v["change_pct"] > 0
            found.append({
                "id": f"volume|{svc}", "kind": "volume", "service": svc,
                "headline": f"{svc}: {abs(v['change_pct']):g}% {'more' if up else 'fewer'} students seen than "
                            "the same dates last year",
                "detail": f"{v['before']} then {v['after']} students ({span}); the plausible range is "
                          f"{v['ci_pct'][0]:g}% to {v['ci_pct'][1]:g}%.",
                "figures": {"before": v["before"], "after": v["after"], "change_pct": v["change_pct"],
                            "range_pct": v["ci_pct"]},
                "tested": "Tested as a change in the number of students seen, allowing for the length of each window.",
                "strength": min(1.0, abs(v["change_pct"]) / 35), "p": v.get("p"), "step": step, "pick": 1,
                "next": f"Are the extra students{'' if svc == 'All services' else ' in ' + svc.lower()} coming back "
                        "for second visits at the usual rate?" if up else
                        f"Which groups account for the drop in students seen{'' if svc == 'All services' else ' in ' + svc.lower()}?"})
        else:
            same.append(f"Students seen ({svc}): no clear change ({v.get('before')} then {v.get('after')}).")
    o, verdict = r["overall"], r.get("verdict")
    if verdict and verdict != "no_clear_change":
        part = {"who_came": [r["mix_effect"]], "within_groups": [r["within_group_effect"]]}.get(
            verdict, [r["mix_effect"], r["within_group_effect"]])
        p = max(_p_from_ci(x["estimate"], x["ci"]) for x in part)
        verb = "rose" if o["change"] > 0 else "fell"
        head = {"who_came": f"{svc}: {name} {verb}, mostly because of who came",
                "within_groups": f"{svc}: {name} {verb} within student groups",
                "both": f"{svc}: {name} {verb}, from both who came and change within groups",
                "offsetting": f"{svc}: {name} looks flat overall, but changed within student groups"}[verdict]
        found.append({
            "id": f"change|{svc}|{metric}", "kind": "change", "service": svc, "headline": head,
            "detail": f"{o['before']:g}{unit} then {o['after']:g}{unit} ({span}), split by level of study. "
                      f"From who came: {r['mix_effect']['estimate']:+g}{pts}; from change within groups: "
                      f"{r['within_group_effect']['estimate']:+g}{pts}.",
            "figures": {"before": o["before"], "after": o["after"], "change": o["change"],
                        "who_came": r["mix_effect"]["estimate"], "within_groups": r["within_group_effect"]["estimate"]},
            "tested": "Tested by repeating the comparison 1,000 times on different mixes of the same students; a part counts only if it points the same way throughout.",
            "strength": min(1.0, abs(part[-1]["estimate"]) / (abs(o["before"]) or 1) / 0.5), "p": p,
            "step": step, "pick": 0,
            "next": f"Which groups changed most in {name}{'' if svc == 'All services' else ' for ' + svc.lower()}?"})
    else:
        same.append(f"{name.capitalize()} ({svc}): no clear change ({o['before']:g}{unit} then {o['after']:g}{unit}).")
    return found, same


def _from_noshow(step):
    _, args, r = step
    flags = r.get("flags", [])
    if not flags:
        return [], ["Missed appointments: no kind of appointment is missed clearly more often than others."]
    levels = {}
    for g in r.get("all_groups", []):
        levels.setdefault(g["factor"], []).append(g)
    out = []
    for f in sorted(flags, key=lambda x: -x["gap_vs_others_pts"]):
        if f["gap_vs_others_pts"] <= 0 and any(o["factor"] == f["factor"] and o["gap_vs_others_pts"] > 0 for o in flags):
            continue                                     # the mirror image of one already kept
        if any(o["figures"]["factor"] == f["factor"] for o in out):
            continue
        who = (f"{f['group']} appointments" if f["factor"] in ("Appointment format", "Weekday", "Service", "Visit type", "Time of day")
               else f"Appointments where: {f['factor'].lower()} is “{f['group']}”")
        more = f["gap_vs_others_pts"] > 0
        out.append({
            "id": f"noshow|{_svc(args.get('service_line'))}|{f['factor']}:{f['group']}", "kind": "noshow",
            "service": _svc(args.get("service_line")),
            "headline": f"{who} are missed {abs(f['gap_vs_others_pts']):g} points {'more' if more else 'less'} often than others",
            "detail": f"{f['no_show_pct']:g}% missed ({f['appointments']} appointments, {f['students']} students), "
                      f"against {r['overall_no_show_pct']:g}% overall; plausible range of the gap {f['ci_pts'][0]:g} to "
                      f"{f['ci_pts'][1]:g} points. Covers {r.get('period', 'all dates')}. A pattern, not a cause.",
            "figures": {"factor": f["factor"], "missed_pct": f["no_show_pct"], "gap_pts": f["gap_vs_others_pts"],
                        "range_pts": f["ci_pts"], "appointments": f["appointments"], "students": f["students"],
                        "overall_pct": r["overall_no_show_pct"]},
            "tested": "Tested with each student counted once, corrected for checking many kinds of appointment.",
            "strength": min(1.0, abs(f["gap_vs_others_pts"]) / 10), "p": f.get("q"), "step": step, "pick": 0,
            "next": "Do students who miss an appointment come back?"})
    return out[:MAX_PER_KIND], []


def _from_care(step):
    _, args, r = step
    svc, dim = _svc(args.get("service_line")), args.get("dimension")
    flags = [f for f in r.get("flags", []) if f["measure"] == "single_visit"]
    from .figures import _label, _period
    if not flags:
        return [], [f"Coming back after a first visit ({svc}): no clear differences by {_label(dim)} "
                    f"({r['single_visit_pct']:g}% had a single visit)."]
    f = max(flags, key=lambda x: (abs(x["gap_vs_others_pts"]), -x["students"]))
    less = f["gap_vs_others_pts"] < 0
    return [{
        "id": f"single_visit|{svc}|{dim}:{f['group']}", "kind": "continuity", "service": svc,
        "headline": f"{svc}: {f['group']} students were {abs(f['gap_vs_others_pts']):g} points "
                    f"{'less' if less else 'more'} likely to have a single visit",
        "detail": f"{f['group_pct']:g}% of {f['students']} {f['group']} students had a single visit in the 60 days "
                  f"after starting care, against {r['single_visit_pct']:g}% of all {r['students']} students "
                  f"(starting care {_period(r.get('episodes_start_between', ''), ' and ')}). A single visit can be planned "
                  "single-session care, so this is a question for the clinical team, not a dropout rate.",
        "figures": {"group_pct": f["group_pct"], "gap_pts": f["gap_vs_others_pts"], "students": f["students"],
                    "everyone_pct": r["single_visit_pct"], "all_students": r["students"]},
        "tested": "Tested against all other students, corrected for the number of groups checked.",
        "strength": min(1.0, abs(f["gap_vs_others_pts"]) / 20), "p": f.get("q"), "step": step, "pick": 1,
        "next": f"What happens after a student's first {svc.lower()} visit?"}], []


def _from_priorities(step):
    _, args, r = step
    name = SURVEY_SERVICES.get(r.get("service"), str(r.get("service", "")).title())
    if not r.get("priorities"):
        return [], [f"{name} survey: no question is clearly more linked to the overall rating than the others."]
    top = next(i for i in r["items"] if i["item"] == r["priorities"][0])
    return [{
        "id": f"priority|{r['service']}|{top['item_code']}", "kind": "priority", "service": name,
        "headline": f"{name}: “{top['item']}” is closely linked to the overall rating, with {top['agree_pct']:g}% agreement",
        "detail": f"{top['agree_pct']:g}% of {top['students']} students agree. Of the survey questions about "
                  f"{name.lower()}, it is among the most closely linked to {r.get('outcome', 'the overall rating')}"
                  + (f". Also a priority: {'; '.join('“' + p + '”' for p in r['priorities'][1:3])}" if len(r["priorities"]) > 1 else "")
                  + ". These are associations; they do not show that changing one will change the other.",
        "figures": {"agree_pct": top["agree_pct"], "students": top["students"], "respondents": r.get("respondents")},
        "tested": "Kept only because it holds up when the analysis is repeated 1,000 times on different mixes of the same students.",
        "strength": min(1.0, top["link_to_outcome"] * (1 - top["agree_pct"] / 100) * 1.6), "p": None,
        "step": step, "pick": 0, "next": f"What are students saying about “{top['item'][:70]}”?"}], []


def _from_equity(step):
    _, args, r = step
    from .figures import _who
    flags = sorted((f for f in r.get("flags", []) if f["gap_pts"] < 0), key=lambda f: f["gap_pts"])
    if not flags:
        return [], [f"Survey ratings: no clear gaps between student groups ({r.get('comparisons_run', 0)} comparisons checked)."]
    out = []
    for f in flags[:MAX_PER_KIND]:
        svc = SURVEY_SERVICES.get(f.get("service"), str(f.get("service", "")).title())
        out.append({
            "id": f"equity|{f.get('service')}|{f['dimension']}:{f['group']}", "kind": "equity", "service": svc,
            "headline": f"{_who(f['dimension'], f['group'])} rate {svc.lower()} {abs(f['gap_pts']):g} points lower than everyone else",
            "detail": f"Average agreement {f['pct_agree_group']:g}% among {f['n_group']} students, against "
                      f"{f['pct_agree_rest']:g}% among everyone else.",
            "figures": {"group_pct": f["pct_agree_group"], "rest_pct": f["pct_agree_rest"], "gap_pts": f["gap_pts"],
                        "n_group": f["n_group"]},
            "tested": f"One of {r.get('comparisons_run', 'many')} group comparisons, corrected for checking them all.",
            "strength": min(1.0, abs(f["gap_pts"]) / 20), "p": f.get("q"), "step": step, "pick": 0,
            "next": f"What matters most to students in {svc.lower()}?"})
    return out, []


def _from_survey_change(step):
    """Groups and questions whose survey ratings moved between the two windows."""
    _, args, r = step
    from .figures import _who
    span = _span(r["periods"])
    found, same = [], []
    for g in r.get("groups_that_changed", [])[:MAX_PER_KIND]:
        svc = SURVEY_SERVICES.get(g["service"], str(g["service"]).title())
        who, down = _who(g["dimension"], g["group"]), g["change_pts"] < 0
        other = ""
        if "everyone_else_change_pts" in g:
            other = (f" Other students' ratings moved {g['everyone_else_change_pts']:+g} points over the same dates, so "
                     + ("this group moved differently from everyone else." if g.get("differs_from_everyone_else")
                        else "this may be part of a wider shift."))
        found.append({
            "id": f"survey_change|{g['service']}|{g['dimension']}:{g['group']}", "kind": "survey_change", "service": svc,
            "headline": f"{who} rate {svc.lower()} {abs(g['change_pts']):g} points {'lower' if down else 'higher'} than a year ago",
            "detail": f"Average agreement {g['pct_agree_before']:g}% then {g['pct_agree_after']:g}% "
                      f"({g['respondents_before']} then {g['respondents_after']} students; {span}); plausible range of the "
                      f"change {g['ci_pts'][0]:g} to {g['ci_pts'][1]:g} points.{other} The survey is anonymous, so these are "
                      "different students in each window.",
            "figures": {"before_pct": g["pct_agree_before"], "after_pct": g["pct_agree_after"], "change_pts": g["change_pts"],
                        "range_pts": g["ci_pts"], "students_before": g["respondents_before"],
                        "students_after": g["respondents_after"], "everyone_else_change_pts": g.get("everyone_else_change_pts")},
            "tested": f"One of {r.get('groups_checked', 'many')} groups checked, corrected for checking them all.",
            "strength": min(1.0, abs(g["change_pts"]) / 22), "p": g.get("q"), "step": step, "pick": 0,
            "next": f"What are students saying about {svc.lower()} in their comments?"})
    for q in r.get("questions_that_changed", [])[:MAX_PER_KIND]:
        svc = SURVEY_SERVICES.get(q["service"], str(q["service"]).title())
        down = q["change_pts"] < 0
        found.append({
            "id": f"survey_change|{q['service']}|{q['item_code']}", "kind": "survey_change", "service": svc,
            "headline": f"{svc}: “{q['item']}” is {abs(q['change_pts']):g} points {'lower' if down else 'higher'} than a year ago",
            "detail": f"{q['pct_agree_before']:g}% of {q['students_before']} students agreed, then "
                      f"{q['pct_agree_after']:g}% of {q['students_after']} ({span}). Different students answered in each window.",
            "figures": {"before_pct": q["pct_agree_before"], "after_pct": q["pct_agree_after"], "change_pts": q["change_pts"],
                        "students_before": q["students_before"], "students_after": q["students_after"]},
            "tested": f"One of {r.get('questions_checked', 'many')} questions checked, corrected for checking them all.",
            "strength": min(1.0, abs(q["change_pts"]) / 30), "p": q.get("q"), "step": step, "pick": 0,
            "next": f"Which parts of {svc.lower()} are most closely linked to overall satisfaction?"})
    for s in r.get("services", []):
        if s["verdict"] == "no clear change":
            name = SURVEY_SERVICES.get(s["service"], str(s["service"]).title())
            same.append(f"Survey ratings ({name}): no clear change overall ({s['pct_agree_before']:g}% then {s['pct_agree_after']:g}%).")
    if not r.get("groups_that_changed") and r.get("smallest_group_change_detectable_pts"):
        same.append("Survey ratings by student group: no group moved clearly. With this many responses, a group's rating "
                    f"would need to move about {r['smallest_group_change_detectable_pts']:g} points to be detected.")
    return found, same


# ---- the briefing ---------------------------------------------------------------------

def _build():
    resamples, stats.N_BOOT = stats.N_BOOT, 400      # enough to screen ~20 analyses in reasonable time
    try:
        return _run_battery()
    finally:
        stats.N_BOOT = resamples


def _run_battery():
    con = tools._con()
    findings, same, missing, tests = [], [], [], 0
    meta = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "synthetic": False}

    if analysis.has_chr(con):
        meta["synthetic"] = bool(analysis.chr_is_synthetic(con))
        rows = con.execute("SELECT service_line, COUNT(DISTINCT student_key) AS s FROM appointments "
                           "GROUP BY 1 ORDER BY s DESC").df()
        services = [None] + [r.service_line for r in rows.itertuples() if r.s >= 200][:2]
        meta["data_through"] = str(con.execute("SELECT MAX(start_time)::DATE FROM appointments").fetchone()[0])
        seen_volume, no_window = set(), False
        for svc in services:
            for metric in METRICS:
                step = _call("what_changed", metric=metric, dimension="academic_level", service_line=svc)
                if "error" in step[2]:
                    if "comparison window" in step[2]["error"]:
                        no_window = True
                    break
                meta.setdefault("compared", _span(step[2]["periods"]))
                f, s = _from_change(step, seen_volume)
                findings += f
                same += s
                tests += 3 + (1 if metric == "no_show_rate" else 0)
        if no_window:
            missing.append("Change since last year: the appointment records do not go back far enough yet.")
        step = _call("noshow_patterns")
        if "error" not in step[2]:
            f, s = _from_noshow(step)
            findings += f
            same += s
            tests += step[2].get("tests_run", 0)
        top = services[1] if len(services) > 1 else None
        for dim in ("academic_level", "faculty"):
            step = _call("care_journeys", service_line=top, dimension=dim)
            if "error" not in step[2]:
                f, s = _from_care(step)
                findings += f
                same += s
                tests += step[2].get("tests_run", 0)
    else:
        missing.append("Appointments: no appointment records are loaded.")

    if con.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]:
        meta["survey_period"] = tools._period()
        for key in SURVEY_SERVICES:
            step = _call("priority_items", service=key)
            if "error" not in step[2]:
                f, s = _from_priorities(step)
                findings += f
                same += s
                tests += len(step[2].get("items", []))
        step = _call("equity_flags")
        if "error" not in step[2]:
            f, s = _from_equity(step)
            findings += f
            same += s
            tests += step[2].get("comparisons_run", 0)
        step = _call("survey_changed")
        if "error" in step[2]:
            missing.append("Whether survey ratings changed: the survey does not yet cover the same dates a year earlier.")
        else:
            f, s = _from_survey_change(step)
            findings += f
            same += s
            tests += step[2].get("tests_run", 0)
    else:
        missing.append("Survey: no survey responses are loaded.")
    labelled = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='comment_labels'").fetchone()[0]
    if not labelled:
        missing.append("What students wrote: comments have not been sorted into themes yet.")
    missing += ["Whether any group is under-served: needs enrolment numbers by group.",
                "Whether demand is outgrowing capacity: needs clinical hours, and students who could not be offered an appointment.",
                "Whether booking further ahead leads to more missed appointments: needs booking dates."]

    # Demand is one story: lead with all services and give each service inside it.
    vols = [f for f in findings if f["kind"] == "volume"]
    main = next((f for f in vols if f["service"] == "All services"), None)
    if main and len(vols) > 1:
        parts = [f"{f['service']} {f['figures']['change_pct']:+g}% ({f['figures']['before']} then {f['figures']['after']})"
                 for f in vols if f is not main]
        main["detail"] += " By service: " + "; ".join(parts) + "."
        main["figures"]["by_service"] = {f["service"]: f["figures"] for f in vols if f is not main}
        findings = [f for f in findings if f["kind"] != "volume" or f is main]

    # Returning after a first visit is also one story per service: keep the largest gap and
    # mention the other split inside it, so the same pattern is not shown twice.
    for svc in {f["service"] for f in findings if f["kind"] == "continuity"}:
        group = sorted((f for f in findings if f["kind"] == "continuity" and f["service"] == svc),
                       key=lambda f: -f["strength"])
        for other in group[1:]:
            group[0]["detail"] += " Also: " + other["headline"].split(": ", 1)[-1] + "."
            group[0]["figures"].setdefault("also", []).append(other["figures"])
            findings.remove(other)

    # Each analysis has already been tested on its own. The findings that passed are checked once
    # more together, so that producing several findings at once does not let a chance result in.
    with_p = [f for f in findings if f["p"] is not None]
    for f, q in zip(with_p, stats.bh_adjust([f["p"] for f in with_p])):
        f["q_overall"] = float(q)
    kept = [f for f in findings if f["p"] is None or f["q_overall"] <= BATTERY_FDR]
    dropped = len(findings) - len(kept)
    kept.sort(key=lambda f: -f["strength"])
    top, also, per_kind = [], [], {}
    for f in kept:
        if len(top) < MAX_FINDINGS and per_kind.get(f["kind"], 0) < MAX_PER_KIND:
            top.append(f)
            per_kind[f["kind"]] = per_kind.get(f["kind"], 0) + 1
        else:
            also.append(f)
    return {**meta, "findings": top, "also": also, "unchanged": list(dict.fromkeys(same)),
            "not_checked": missing, "tests_run": int(tests), "dropped_after_correction": dropped}


def build(use_cache=True):
    """The briefing for the data as it is now. Cached until the database changes."""
    key = config.DB_PATH.stat().st_mtime
    if not use_cache or _cache.get("key") != key:
        _cache.update(key=key, briefing={**_build(), "data_key": key})
    return _cache["briefing"]


def cached():
    """The briefing already built for the current data, or None (never triggers a build)."""
    try:
        return _cache["briefing"] if _cache.get("key") == config.DB_PATH.stat().st_mtime else None
    except OSError:
        return None


# ---- comparing with last time -----------------------------------------------------------

def _slim(b):
    return {"generated": b["generated"], "data_key": b["data_key"], "data_through": b.get("data_through"),
            "findings": [{"id": f["id"], "headline": f["headline"]} for f in b["findings"] + b["also"]]}


def save(b):
    """Keep this briefing so the next one can say what is new. One file per version of the data."""
    DIR.mkdir(parents=True, exist_ok=True)
    path = DIR / f"briefing_{int(b['data_key'])}.json"
    path.write_text(json.dumps(_slim(b), indent=1))
    return path


def previous(b):
    """The most recent saved briefing made from EARLIER data, or None."""
    if not DIR.exists():
        return None
    older = []
    for p in DIR.glob("briefing_*.json"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("data_key", 0) < b["data_key"] - 1:
            older.append(d)
    return max(older, key=lambda d: d["data_key"]) if older else None


def compare(b, prev):
    """{'status': {finding id: 'new' | 'still there'}, 'gone': [headlines no longer showing]}."""
    if not prev:
        return {"status": {}, "gone": [], "since": None}
    before = {f["id"]: f["headline"] for f in prev["findings"]}
    now = {f["id"] for f in b["findings"] + b["also"]}
    return {"status": {f["id"]: ("still there" if f["id"] in before else "new") for f in b["findings"] + b["also"]},
            "gone": [h for i, h in before.items() if i not in now],
            "since": prev.get("data_through") or prev.get("generated", "")[:10]}


# ---- outputs ------------------------------------------------------------------------------

def for_chat(b=None):
    """Compact briefing for the assistant: wording plus the figures behind it, no chart data."""
    b = b or build()
    return {
        "data_through": b.get("data_through"), "compared": b.get("compared"), "survey_period": b.get("survey_period"),
        "findings": [{"headline": f["headline"], "detail": f["detail"], "figures": f["figures"],
                      "how_tested": f["tested"], "next": f["next"]} for f in b["findings"]],
        "also_noted": [f["headline"] for f in b["also"]],
        "stayed_the_same": b["unchanged"], "could_not_check": b["not_checked"],
        "checks_run": b["tests_run"],
        "note": ("Each analysis was tested on its own, and the findings that passed were checked again together. "
                 "Say it that way; do not say all checks were corrected together. Report the findings in this "
                 "order, with their dates and group sizes; then say what stayed the same. Patterns, not causes: "
                 "do not say one service or group is 'driving' a change."
                 + (" The appointment records are synthetic test data, not real." if b.get("synthetic") else "")),
    }


def graphic(finding):
    """The chart for one finding, as a figures spec (title, scope, caption, fig), or None."""
    from . import figures
    name, args, res = finding["step"]
    try:
        specs = figures.BUILDERS[name](args, res)
        return specs[min(finding.get("pick", 0), len(specs) - 1)] if specs else None
    except Exception:
        return None


def to_html(b, changes=None, title="Term briefing"):
    """A one-page briefing to download or print."""
    from . import theme
    changes = changes or {"status": {}, "gone": []}
    parts = []
    for n, f in enumerate(b["findings"], 1):
        spec = graphic(f)
        chart = spec.fig.to_html(full_html=False, include_plotlyjs=True if n == 1 else False,
                                 config={"displayModeBar": False}) if spec else ""
        tag = changes["status"].get(f["id"])
        parts.append(f"<section><p class='num'>{n}" + (f" <span class='tag'>{tag}</span>" if tag else "") + "</p>"
                     f"<h2>{f['headline']}</h2><p>{f['detail']}</p>{chart}"
                     f"<p class='how'>{f['tested']}</p><p class='next'>Related question: {f['next']}</p></section>")
    lists = ""
    for head, items in (("Also noted", [f["headline"] for f in b["also"]]),
                        ("No longer showing since last time", changes.get("gone", [])),
                        ("What stayed the same", b["unchanged"]), ("What could not be checked", b["not_checked"])):
        if items:
            lists += f"<h3>{head}</h3><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"
    css = (f"body{{font-family:{theme.FONT};background:{theme.PAGE};color:{theme.TEXT};margin:0;padding:32px 16px}}"
           f"main{{max-width:940px;margin:auto}}h1{{color:{theme.NAVY};font-size:28px;margin:0 0 4px}}"
           f".lede{{color:{theme.MUTED};margin:0 0 22px;line-height:1.5}}"
           f"section{{background:{theme.CARD};border:1px solid {theme.BORDER};border-radius:10px;padding:16px 22px;margin:0 0 16px;break-inside:avoid}}"
           f"h2{{color:{theme.NAVY};font-size:19px;line-height:1.3;margin:2px 0 6px}}h3{{color:{theme.NAVY};font-size:16px;margin:22px 0 6px}}"
           f"p{{line-height:1.5;margin:0 0 8px}}.num{{color:{theme.MUTED};font-size:13px;margin:0}}"
           f".tag{{background:{theme.ALERT_BG['info']};color:{theme.FOCUS};border-radius:10px;padding:1px 9px;font-weight:700}}"
           f".how,.next{{color:{theme.MUTED};font-size:13px}}li{{line-height:1.5;margin-bottom:4px}}"
           "@media print{body{background:white;padding:0}section{border:none;padding:0 0 10px}}")
    lede = (f"Appointment records to {b.get('data_through', 'n/a')}"
            + (f", comparing {b['compared']}" if b.get("compared") else "")
            + (f". Survey responses from {b['survey_period']}" if b.get("survey_period") else "")
            + f". {b['tests_run']} checks were run. Each analysis was tested on its own, and the findings that "
              "passed were checked again together; only those that held up are shown."
            + (" The appointment records are synthetic test data, not real." if b.get("synthetic") else ""))
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title>{theme.font_link()}"
            f"<style>{css}</style></head><body><main><h1>{theme.UNIT_NAME}: {title.lower()}</h1>"
            f"<p class='lede'>{lede}</p>" + "".join(parts) + lists + "</main></body></html>")

"""Suggested follow-up questions, shown as buttons under an answer.

suggest(trace, asked)  two or three questions that follow from what the analysis just found
starters()             questions to begin with, from the playbook's ready use cases

Suggestions come from rules on the results, not from the model: they cost nothing, appear
instantly, and can only ever point at something the assistant can actually do. A playbook
entry can add its own with a `next:` list, so the team can edit them without code.
"""
from __future__ import annotations

from . import playbook

SOURCES_Q = "Where did this come from?"
_SERVICES = {"counselling": "counselling", "medical": "the medical clinic", "wellness_adv": "wellness advising",
             "wellness advising": "wellness advising"}


def _svc(args, result=None):
    s = (args or {}).get("service_line") or (args or {}).get("service") or (result or {}).get("service_line")
    return None if not s or str(s).lower() == "all" else str(s)


def _name(s):
    return _SERVICES.get(str(s).lower(), str(s).lower())


def _what_changed(a, r):
    out = []
    svc = _svc(a, r)
    where = f" in {_name(svc)}" if svc else ""
    v = r.get("volume_change", {})
    if v.get("verdict") in ("increase", "decrease"):
        out.append(f"Are the {'extra ' if v['verdict'] == 'increase' else ''}students{where} coming back for "
                   "second visits at the usual rate?")
        out.append(f"Which faculties account for the change in students seen{where}?")
    if r.get("verdict") == "no_clear_change":
        out.append("What is the smallest change this data could detect?")
    elif r.get("verdict"):
        out.append(f"Which groups changed most{where}?")
    out.append("Is it the same across all services?" if svc else "Is it the same for counselling alone?")
    return out


def _noshow(a, r):
    svc = _svc(a, r)
    out = []
    if r.get("flags"):
        out.append("Do students who miss an appointment come back?")
        out.append("Has the no-show rate changed since last year?")
        if not svc:
            out.append("Is the gap the same in counselling alone?")
    else:
        out.append("Has the no-show rate changed since last year?")
    return out


def _care(a, r):
    svc = _svc(a, r)
    where = f" in {_name(svc)}" if svc else ""
    out = []
    flags = [f for f in r.get("flags", []) if f.get("measure") == "single_visit"]
    if flags:
        out.append("What are students saying about follow-up after a first visit?")
    if not r.get("dimension"):
        out.append(f"Are some groups less likely to come back after a first visit{where}?")
    elif r.get("dimension") != "faculty":
        out.append(f"Does returning after a first visit differ by faculty{where}?")
    out.append(f"Has the number of students seen{where} changed since last year?")
    return out


def _priorities(a, r):
    svc = str(r.get("service", "counselling"))
    out = []
    if r.get("priorities"):
        topic = r["priorities"][0]
        out.append(f"What are students saying about “{topic[:70]}”?")
    other = {"counselling": "the medical clinic", "medical": "counselling", "wellness_adv": "counselling"}.get(svc)
    if other:
        out.append(f"What matters most to students in {other}?")
    out.append(f"Do some groups of students rate {_name(svc)} lower?")
    return out


def _equity(a, r):
    if r.get("flags"):
        f = min(r["flags"], key=lambda x: x.get("gap_pts", 0))
        return [f"What are students saying in their comments about {_name(f.get('service', 'our services'))}?",
                f"What matters most to students in {_name(f.get('service', 'counselling'))}?"]
    return ["Do graduate students use counselling differently from undergraduates?",
            "Which appointments are missed most often?"]


RULES = {
    "what_changed": _what_changed, "noshow_patterns": _noshow, "care_journeys": _care,
    "priority_items": _priorities, "equity_flags": _equity,
    "chr_trend": lambda a, r: ["Is the change from more students, or more visits each?",
                               "Has the no-show rate changed since last year?"],
    "explain_difference": lambda a, r: ["Has that changed since last year?",
                                        "Are some groups less likely to come back after a first visit?"],
    "service_pathways": lambda a, r: ["What happens after a student's first counselling visit?",
                                      "Has the number of students seen changed since last year?"],
    "item_summary": lambda a, r: [f"Which of these matters most to students' overall view of {_name(_svc(a) or 'counselling')}?"],
    "compare_groups": lambda a, r: ["Is that difference real, or within noise?"],
    "group_vs_everyone": lambda a, r: ["What are those students saying in their comments?"],
    "wait_time_vs_rating": lambda a, r: ["What are students saying about waiting?"],
    "chr_summary": lambda a, r: ["Has this changed since last year?"],
    "comment_overview": lambda a, r: ["What matters most to students in counselling?"],
    "search_comments": lambda a, r: ["What matters most to students in counselling?"],
    "survey_vs_visits": lambda a, r: ["Do some groups of students rate our services lower?"],
    "term_briefing": lambda a, r: [f["next"] for f in r.get("findings", [])[:2] if f.get("next")],
}


def suggest(trace, asked=(), limit=3, include_sources=True):
    """Follow-up questions for the answer whose analysis steps are `trace`."""
    seen = {str(q).strip().lower().rstrip("?") for q in asked}
    out = []

    def add(q):
        key = q.strip().lower().rstrip("?")
        if key not in seen:
            seen.add(key)
            out.append(q)

    steps = [s for s in (trace or []) if isinstance(s, (list, tuple)) and len(s) == 3]
    for name, args, res in steps:                    # the playbook's own follow-ups come first
        if name == "insight_playbook" and isinstance(res, dict):
            for case in res.get("use_cases", [])[:1]:
                for q in case.get("next", []):
                    add(q)
    for name, args, res in steps:
        rule = RULES.get(name)
        if not rule or not isinstance(res, dict) or "error" in res:
            continue
        try:
            for q in rule(args or {}, res):
                add(q)
        except Exception:
            continue
    if not include_sources:
        return out[:limit]
    out = out[: max(0, limit - 1)]
    worked = [s for s in steps if (s[0] in RULES or s[0] == "run_sql") and isinstance(s[2], dict) and "error" not in s[2]]
    if worked and SOURCES_Q.lower().rstrip("?") not in {str(q).strip().lower().rstrip("?") for q in asked}:
        out.append(SOURCES_Q)
    return out[:limit]


def starters(limit=6):
    """Questions to begin with: one from each ready playbook use case."""
    return [q for _, q in playbook.starter_questions(limit)]

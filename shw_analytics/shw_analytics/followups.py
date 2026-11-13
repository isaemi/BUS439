"""Suggested follow-up questions, shown as buttons under an answer.

Second-level analysis is a ladder. For any finding, the useful next question is the next rung:

    1 describe   what happened?
    2 test       is it real?
    3 decompose  what is it made of? (who came vs change within groups; students vs visits each)
    4 segment    for whom? (level of study, faculty, format)
    5 follow     what happens next for those students?
    6 voice      what do students say?

suggest_detail(trace, asked, history) returns up to two suggestions:
  deeper  the next rung this conversation has not yet covered for the finding just discussed,
          worded with its specifics (service, month, group)
  wider   a tested finding from the term briefing about the same service that has not come up
          yet (only when a briefing has been built), so the click is known to lead somewhere

Rules, not the model: instant, free, and only ever pointing at something the assistant can do.
A playbook entry's `next:` list is used when the ladder has nothing to offer.
"""
from __future__ import annotations

import calendar
import re

from . import playbook

RUNGS = ["describe", "test", "decompose", "segment", "follow", "voice"]
_NAMES = {"counselling": "counselling", "medical": "the medical clinic", "wellness_adv": "wellness advising",
          "info_desk": "the Wellness Info Desk"}
_ALIASES = {"wellness advising": "wellness_adv", "medical clinic": "medical", "wellness info desk": "info_desk",
            "counseling": "counselling", "all": None, "all services": None, "": None, "none": None}
_DIMS = {"academic_level": "level of study", "faculty": "faculty", "gender": "gender", "age_band": "age group",
         "mode": "appointment format", "visit_type": "visit type"}
_WHY_WIDER = "The term briefing found a tested result here."


def _norm(service):
    s = str(service or "").strip().lower()
    return _ALIASES.get(s, s) or None


def _svc(args, res=None):
    a = args or {}
    return _norm(a.get("service_line") or a.get("service") or (res or {}).get("service_line") or (res or {}).get("service"))


def _in(svc):
    return f" in {_NAMES.get(svc, svc)}" if svc else ""


def _name(svc):
    return _NAMES.get(svc, svc) if svc else "our services"


def _c(rung, text, svc, subject, dim=None):
    """A candidate suggestion and the (service, subject, rung, dimension) it would cover."""
    return {"text": text, "key": (svc, subject, rung, dim), "rung": RUNGS.index(rung)}


# ---- what each kind of analysis has already covered -------------------------------------

def covered_by(name, args, res):
    """The (service, subject, rung, dimension) questions an analysis step has answered."""
    if not isinstance(res, dict) or "error" in res:
        return set()
    a, svc = args or {}, _svc(args, res)
    dim = a.get("dimension")
    if name == "chr_trend":
        return {(svc, "volume", "describe", None)}
    if name == "chr_summary":
        return {(svc, "volume", "segment", dim)} if dim else {(svc, "volume", "describe", None)}
    if name == "what_changed":
        subject = "no_show" if a.get("metric") == "no_show_rate" else "intensity"
        out = {(svc, "volume", "test", None), (svc, subject, "test", None), (svc, subject, "decompose", None),
               (svc, "volume", "segment", dim)}
        if a.get("metric") == "visits_per_student":
            out.add((svc, "volume", "decompose", None))
        return out
    if name == "explain_difference":
        return {(svc, "intensity", "test", None), (svc, "intensity", "segment", dim)}
    if name == "noshow_patterns":
        return {(svc, "no_show", "describe", None), (svc, "no_show", "segment", None)}
    if name == "care_journeys":
        out = {(svc, "continuity", "describe", None), (svc, "no_show", "follow", None), (svc, "volume", "follow", None)}
        if dim:
            out.add((svc, "continuity", "segment", dim))
        return out
    if name == "service_pathways":
        return {(None, "pathways", "describe", None)}
    if name == "priority_items":
        return {(svc, "priorities", "test", None), (svc, "priorities", "describe", None)}
    if name == "item_summary":
        return {(svc, "priorities", "describe", None)}
    if name == "equity_flags":
        return {(s, "equity", "test", None) for s in (None, *_NAMES)} if svc is None else {(svc, "equity", "test", None)}
    if name == "compare_groups":
        return {(svc, "equity", "describe", dim)}
    if name == "group_vs_everyone":
        return {(svc, "equity", "segment", dim)}
    if name in ("search_comments", "comment_overview"):
        return {(svc, s, "voice", None) for s in ("volume", "no_show", "continuity", "priorities", "equity")}
    return set()


# ---- the ladder for each kind of analysis ----------------------------------------------

def _month(period):
    m = re.fullmatch(r"(\d{4})-(\d{2})", str(period))
    return calendar.month_name[int(m.group(2))] if m else str(period)


def _volume_ladder(svc, the_change="the change"):
    return [
        _c("test", f"Is the change in students seen{_in(svc)} real, compared with the same dates last year?", svc, "volume"),
        _c("decompose", f"Is {the_change}{_in(svc)} from more students, or more visits each?", svc, "volume"),
        _c("segment", f"Which faculties account for the change in students seen{_in(svc)}?", svc, "volume", "faculty"),
        _c("follow", f"Are students{_in(svc)} coming back for second visits at the usual rate?", svc, "volume"),
        _c("voice", f"What are students saying about getting an appointment{_in(svc)}?", svc, "volume"),
    ]


def candidates(name, args, res):
    """The rungs above what this step did, in order, worded with the step's specifics."""
    if not isinstance(res, dict) or "error" in res:
        return []
    a, svc = args or {}, _svc(args, res)
    if name == "chr_trend":
        hot = [r for r in res.get("rows", []) if r.get("notable_change")]
        change = "the change"
        if hot:
            last = hot[-1]
            change = f"the {_month(last.get('period'))} {'rise' if (last.get('change_pct') or 0) > 0 else 'fall'}"
        ladder = _volume_ladder(svc, change)
        return ladder[1:] if hot else ladder          # a flagged month has been tested already
    if name == "chr_summary":
        return _volume_ladder(svc)
    if name == "what_changed":
        v = res.get("volume_change", {})
        change = {"increase": "the rise", "decrease": "the fall"}.get(v.get("verdict"), "the change")
        if a.get("metric") == "no_show_rate":
            return [
                _c("describe", f"Which kinds of appointments{_in(svc)} are missed most often?", svc, "no_show"),
                _c("follow", f"Do students who miss an appointment{_in(svc)} come back?", svc, "no_show"),
                *_volume_ladder(svc, change)[1:],
            ]
        return _volume_ladder(svc, change)[1:]
    if name == "noshow_patterns":
        out = []
        flags = res.get("flags") or []
        if flags and svc is None:
            top = max(flags, key=lambda f: f.get("gap_vs_others_pts", 0))
            out.append(_c("segment", f"Is the {str(top['group']).lower()} gap in missed appointments the same in counselling alone?",
                          "counselling", "no_show"))
        out += [
            _c("test", f"Has the no-show rate{_in(svc)} changed since last year?", svc, "no_show"),
            _c("follow", f"Do students who miss an appointment{_in(svc)} come back?", svc, "no_show"),
            _c("voice", f"What are students saying about booking and reminders{_in(svc)}?", svc, "no_show"),
        ]
        return out
    if name == "care_journeys":
        return [
            _c("segment", f"Does returning after a first visit{_in(svc)} differ by level of study?", svc, "continuity", "academic_level"),
            _c("segment", f"Does returning after a first visit{_in(svc)} differ by faculty?", svc, "continuity", "faculty"),
            _c("test", f"Has the number of students seen{_in(svc)} changed since last year?", svc, "volume"),
            _c("voice", f"What are students saying about follow-up{_in(svc)}?", svc, "continuity"),
        ]
    if name == "service_pathways":
        return [_c("describe", "What happens after a student's first counselling visit?", "counselling", "continuity"),
                _c("test", "Has the number of students seen changed since last year?", None, "volume")]
    if name == "item_summary":
        return [_c("test", f"Which parts of {_name(svc)} are most closely linked to overall satisfaction?", svc, "priorities"),
                _c("test", f"Do some groups of students rate {_name(svc)} lower?", svc, "equity"),
                _c("voice", f"What are students saying about {_name(svc)}?", svc, "priorities")]
    if name == "priority_items":
        top = (res.get("priorities") or [None])[0]
        return [
            _c("voice", f"What are students saying about “{top[:70]}”?" if top else f"What are students saying about {_name(svc)}?",
               svc, "priorities"),
            _c("test", f"Do some groups of students rate {_name(svc)} lower?", svc, "equity"),
            *[_c("test", f"Which parts of {_name(o)} are most closely linked to overall satisfaction?", o, "priorities")
              for o in ("counselling", "medical", "wellness_adv") if o != svc],
        ]
    if name == "equity_flags":
        low = sorted((f for f in res.get("flags", []) if f.get("gap_pts", 0) < 0), key=lambda f: f["gap_pts"])
        if low:
            s = _norm(low[0].get("service"))
            return [_c("test", f"Which parts of {_name(s)} are most closely linked to overall satisfaction?", s, "priorities"),
                    _c("voice", f"What are students saying about {_name(s)}?", s, "equity")]
        return [_c("test", "Which parts of counselling are most closely linked to overall satisfaction?", "counselling", "priorities"),
                _c("describe", "Which kinds of appointments are missed most often?", None, "no_show")]
    if name in ("compare_groups", "group_vs_everyone"):
        return [_c("test", f"Do some groups of students rate {_name(svc)} lower?", svc, "equity"),
                _c("test", f"Which parts of {_name(svc or 'counselling')} are most closely linked to overall satisfaction?",
                   svc or "counselling", "priorities"),
                _c("voice", f"What are students saying about {_name(svc)}?", svc, "equity")]
    if name == "explain_difference":
        grp = str(a.get("group", "these")).lower()
        return [_c("segment", f"Does returning after a first visit{_in(svc)} differ by level of study?", svc, "continuity", "academic_level"),
                _c("test", f"Have visits per student{_in(svc)} changed since last year?", svc, "intensity"),
                _c("voice", f"What are {grp} students saying about {_name(svc)}?", svc, "continuity")]
    if name == "wait_time_vs_rating":
        return [_c("voice", f"What are students saying about waiting{_in(svc)}?", svc, "priorities"),
                _c("test", f"Which parts of {_name(svc)} are most closely linked to overall satisfaction?", svc, "priorities")]
    if name in ("search_comments", "comment_overview"):
        return [_c("test", f"Which parts of {_name(svc or 'counselling')} are most closely linked to overall satisfaction?",
                   svc or "counselling", "priorities")]
    return []


# ---- "wider": a tested finding from the briefing that has not come up --------------------

def _finding_question(f):
    """(question that leads to this briefing finding, the key it covers)."""
    svc = _norm(f.get("service"))
    kind, parts = f.get("kind"), str(f.get("id", "")).split("|")
    if kind == "volume":
        return f"Has the number of students seen{_in(svc)} changed since last year?", (svc, "volume", "test", None)
    if kind == "change":
        metric = parts[-1] if parts else ""
        label = {"no_show_rate": "the no-show rate", "visits_per_student": "visits per student",
                 "days_in_care": "time in care"}.get(metric, "use of the service")
        subject = "no_show" if metric == "no_show_rate" else "intensity"
        return f"Has {label}{_in(svc)} changed since last year?", (svc, subject, "test", None)
    if kind == "noshow":
        return f"Which kinds of appointments{_in(svc)} are missed most often?", (svc, "no_show", "describe", None)
    if kind == "continuity":
        dim = parts[-1].split(":")[0] if parts else "academic_level"
        return (f"Does returning after a first visit{_in(svc)} differ by {_DIMS.get(dim, dim.replace('_', ' '))}?",
                (svc, "continuity", "segment", dim))
    if kind == "priority":
        return f"Which parts of {_name(svc)} are most closely linked to overall satisfaction?", (svc, "priorities", "test", None)
    if kind == "equity":
        return f"Do some groups of students rate {_name(svc)} lower?", (svc, "equity", "test", None)
    return None, None


def _wider(covered, taken, service):
    try:
        from . import briefing
        b = briefing.cached()
    except Exception:
        b = None
    if not b:
        return None
    findings = list(b.get("findings", [])) + list(b.get("also", []))
    findings.sort(key=lambda f: _norm(f.get("service")) != service)       # the same service first
    for f in findings:
        q, key = _finding_question(f)
        if q and key not in covered and q.lower().rstrip("?") not in taken:
            return {"text": q, "kind": "wider", "why": _WHY_WIDER, "key": key}
    return None


# ---- putting it together ---------------------------------------------------------------

def _steps(trace):
    return [s for s in (trace or []) if isinstance(s, (list, tuple)) and len(s) == 3]


def suggest_detail(trace, asked=(), history=(), limit=2):
    """Up to `limit` suggestions as dicts: text, kind ('deeper' | 'wider' | 'related'), why.

    trace    the analysis steps behind the latest answer
    asked    every question asked so far in this conversation
    history  the analysis steps of earlier answers, so rungs already covered are not suggested again"""
    steps = _steps(trace)
    taken = {str(q).strip().lower().rstrip("?") for q in asked}
    covered = set()
    for t in list(history) + [trace]:
        for name, args, res in _steps(t):
            covered |= covered_by(name, args, res)
    out = []

    def add(text, kind, why=""):
        key = text.strip().lower().rstrip("?")
        if key in taken or len(out) >= limit:
            return False
        taken.add(key)
        out.append({"text": text, "kind": kind, "why": why})
        return True

    # A briefing answer: each finding already names its own next question.
    for name, args, res in steps:
        if name == "term_briefing" and isinstance(res, dict):
            for f in res.get("findings", []):
                if f.get("next"):
                    add(f["next"], "deeper", "The next question for one of the briefing's findings.")

    # Deeper: the lowest rung not yet covered, starting from the last analysis that found something.
    service = None
    pool = []
    for name, args, res in reversed(steps):
        cands = [c for c in candidates(name, args, res) if c["key"] not in covered]
        service = service or _svc(args, res if isinstance(res, dict) else None)
        pool.append(cands)
    first = next((c for c in pool if c), [])
    if first and len(out) < limit:
        add(first[0]["text"], "deeper", "The next step on what this answer found.")
        covered.add(first[0]["key"])

    # Wider: a tested briefing finding about the same service that has not come up.
    if len(out) < limit:
        w = _wider(covered, taken, service)
        if w:
            add(w["text"], "wider", w["why"])

    # Otherwise: the next uncovered rung of anything else in the answer, then the playbook's own.
    for cands in pool:
        for c in cands:
            if len(out) >= limit:
                break
            if c["key"] not in covered and add(c["text"], "deeper", "A further step on what this answer found."):
                covered.add(c["key"])
    for name, args, res in steps:
        if name == "insight_playbook" and isinstance(res, dict):
            for case in res.get("use_cases", [])[:1]:
                for q in case.get("next", []):
                    add(q, "related")
    return out[:limit]


def suggest(trace, asked=(), limit=2, history=(), include_sources=False):
    """The suggestions as plain questions (see suggest_detail)."""
    return [s["text"] for s in suggest_detail(trace, asked=asked, history=history, limit=limit)]


def starters(limit=6):
    """Questions to begin with: one from each ready playbook use case."""
    return [q for _, q in playbook.starter_questions(limit)]

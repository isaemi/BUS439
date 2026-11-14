"""Where did this answer come from? Plain-language evidence for non-technical staff.

evidence_cards(trace)   one card per analysis step: data source, which records, what was
                        calculated, how certainty was checked, and what to keep in mind.
render_markdown(cards)  the cards as short markdown, for the app's "How this answer was built".
render_text(cards)      the same as plain text, for the model to explain from.
sql_to_english(sql)     a custom query described in words (no code shown to staff).
is_sources_question(q)  "where did this come from?"-style follow-ups.
last_trace(history)     the analysis steps behind the previous answer.

Cards never show code, table names, column names or tool names.
"""
from __future__ import annotations

import re

SOURCES = {
    "chr": "Appointment records from the clinic booking system (CHR)",
    "survey": "Student Health and Wellbeing experience survey",
    "comments": "Students' written comments in the survey",
    "both": "The survey and the appointment records",
    "guide": "The app's built-in guide to analyses",
}

LABELS = {
    "academic_level": "level of study", "faculty": "faculty", "gender": "gender",
    "age_band": "age group", "program": "program", "service_line": "service",
    "service": "service", "visit_type": "visit type", "mode": "appointment format",
    "term": "term", "outcome": "appointment outcome", "start_time": "date",
    "student_key": "student", "international": "international status",
    "first_gen": "first-generation status", "disability_any": "disability",
    "agree": "agreement", "score": "rating", "item_text": "survey question",
}

TABLES = {
    "appointments": "appointment records", "ratings_demo": "survey ratings (with respondents' "
    "background)", "ratings": "survey ratings", "responses": "survey responses",
    "items": "the list of survey questions", "response_faculty": "respondents' faculties",
    "response_disability": "respondents' disability answers",
    "response_multi": "respondents' multiple-choice answers",
}

TESTED = ("Checked with a statistical test, corrected for checking many groups at once, so a "
          "few chance results don't show up as findings.")
INTERVALS = ("Checked by re-sampling students many times to get a range of plausible values. "
             "A difference is only called real when the whole range is on one side of zero.")

# title, source, what was calculated, how certainty was checked. {placeholders} are filled
# from the step's settings in plain words.
TOOLS = {
    "describe_data": ("Checked what data is available", "both",
                      "Looked up which services, survey questions and dates the app holds.", None),
    "service_summary": ("Survey results by service{for_service}", "survey",
                        "For each service: how many students answered, the average rating "
                        "(1 to 5) and the share who agreed or strongly agreed.", None),
    "item_summary": ("Each survey question{for_service}", "survey",
                     "For every question: how many answered, the average rating and the share "
                     "who agreed, weakest first.", None),
    "compare_groups": ("Survey ratings by {dimension}{for_service}", "survey",
                       "The average rating and share agreeing for each group, side by side.",
                       "Not tested: the differences are descriptive."),
    "group_vs_everyone": ("How {value} compares with all respondents{for_service}", "survey",
                          "The group's ratings next to everyone else's.", TESTED),
    "chr_summary": ("Appointment activity{by_dimension}{for_service}{in_term}", "chr",
                    "Counted booked and attended appointments, students seen, visits per "
                    "student, time in care and missed appointments.", None),
    "chr_trend": ("Appointments over time, by {freq}{for_service}", "chr",
                  "Counted appointments in each {freq} and compared each with the same period "
                  "a year earlier.", "A change is flagged only if it is at least 10% and "
                  "unlikely to be chance."),
    "explain_difference": ("How {group} compares with {compare_to} in service use{for_service}",
                           "chr", "Split the difference in total visits into how many students "
                           "came and how many visits each made; also compared time in care, "
                           "services used and missed appointments.", TESTED),
    "survey_vs_visits": ("Whether survey respondents look like the students who visit", "both",
                         "Compared the mix of {dimension} among survey respondents with the "
                         "mix among students in the appointment records.", None),
    "wait_time_vs_rating": ("How waiting time relates to ratings{for_service}", "both",
                            "Linked appointment waiting times to survey ratings.", None),
    "equity_flags": ("Scan for gaps between student groups{for_service}", "survey",
                     "Compared every group's ratings with everyone else's.", TESTED),
    "search_comments": ("Students' comments about \u201c{query}\u201d{for_service}", "comments",
                        "Found the written comments closest in meaning to the topic.", None),
    "comment_overview": ("Themes in students' comments{for_service}", "comments",
                         "Counted comments by theme and tone, using automatic labelling.",
                         "Automatic labels can be wrong for individual comments."),
    "what_changed": ("Same dates compared: {periods}{for_service}", "chr",
                     "Compared the same dates in both periods, and split the change in "
                     "{measure} into a change in which students came and a change within each "
                     "{dimension} group. Also tested whether the number of students changed.",
                     INTERVALS),
    "survey_changed": ("Survey ratings, same dates compared: {periods}{for_service}", "survey",
                       "Compared survey responses in the two windows: each service's share of positive "
                       "ratings, each question, and each student group, with every student counted once.",
                       "Each comparison is tested, and corrected for the number of questions and groups "
                       "checked. The survey is anonymous, so the two windows hold different students."),
    "care_journeys": ("What happens after a first visit{for_service}", "chr",
                      "Followed each student's first new episode of care for a fixed period: "
                      "single visits, quick returns, time to a second visit, and coming back "
                      "after a missed appointment.", TESTED),
    "noshow_patterns": ("Which appointments are missed more often{for_service}", "chr",
                        "Compared the missed-appointment rate for each kind of appointment "
                        "(format, weekday, booking history, student group) with all others.",
                        TESTED + " Each student counts as one unit of evidence."),
    "service_pathways": ("Movement between services within {within_days} days", "chr",
                         "For students starting a new episode in one service, counted how many "
                         "went on to another service soon after.",
                         "Each percentage has a 95% range of plausible values."),
    "priority_items": ("What matters most to students' overall view{for_service}", "survey",
                       "For each survey question, measured how closely students' answers move "
                       "with their overall rating of the service, and combined that with how "
                       "well the question scores. A priority = closely linked but scoring low.",
                       "Checked by re-sampling students 1,000 times; a question is only called a "
                       "priority if it holds up and is clearly more linked than a "
                       "typical question."),
    "term_briefing": ("The term briefing: a fixed set of tested analyses", "both",
                      "Ran the full set of tested analyses across services (change since the same dates "
                      "last year, missed appointments, returning after a first visit, what matters most "
                      "in the survey, gaps between groups), then kept and ranked what held up.",
                      "Each analysis is tested on its own, and the findings that pass are checked again "
                      "together, so that producing several findings at once does not let a chance result in."),
    "insight_playbook": ("Looked up how to approach this kind of question", "guide",
                         "Checked the app's guide for the recommended analyses and pitfalls.",
                         None),
    "run_sql": ("A custom summary of the data", None, None, None),
}

META = {"suggested", "fallback", "empty_reply", "forced_answer", "evidence_cards"}


def _label(x):
    return LABELS.get(str(x), str(x).replace("_", " "))


def _fill(template, args, result):
    a = {k: v for k, v in (args or {}).items() if v not in (None, "")}
    svc = a.get("service") or a.get("service_line")
    periods = (result or {}).get("periods") or {}
    vals = {
        "for_service": f" \u2014 {svc}" if svc else "",
        "by_dimension": f" by {_label(a['dimension'])}" if a.get("dimension") else "",
        "in_term": f" in {a['term']}" if a.get("term") else "",
        "dimension": _label(a.get("dimension", "group")),
        "value": f"{a.get('value', 'the group')} ({_label(a.get('dimension', ''))})",
        "group": a.get("group", "the group"),
        "compare_to": a.get("compare_to") or "all other students",
        "freq": a.get("freq", "month"),
        "query": a.get("query", "the topic"),
        "within_days": a.get("within_days", (result or {}).get("within_days", 30)),
        "measure": str((result or {}).get("measure", a.get("metric", "the measure"))).lower(),
        "periods": (f"{periods.get('before')} vs {periods.get('after')}" if periods
                    else "two periods"),
    }
    try:
        return template.format(**vals)
    except (KeyError, IndexError):
        return template


def _records(result):
    """'How many records' in words, from whatever counts the result carries."""
    r = result or {}
    out = []
    n = r.get("n")
    if isinstance(n, dict) and "before" in n:
        out.append(f"{n['before']:,} {n.get('unit', 'records')} in the earlier period and "
                   f"{n['after']:,} in the later one")
    elif isinstance(n, (int, float)):
        out.append(f"{int(n):,} records")
    for key, word in [("students", "students"), ("appointments", "appointments")]:
        if isinstance(r.get(key), int):
            out.append(f"{r[key]:,} {word}")
    if r.get("period"):
        out.append(f"dates covered: {r['period']}")
    if isinstance(r.get("respondents"), int):
        out.append(f"{r['respondents']:,} survey respondents")
    if "episodes_start_between" in r:
        out.append(f"students starting care between {r['episodes_start_between']}")
    for key in ("rows", "data", "records"):
        if isinstance(r.get(key), list):
            out.append(f"{len(r[key]):,} summary rows")
            break
    return "; ".join(out) or None


def _caveats(result):
    r = result or {}
    out = []
    if r.get("warning"):
        out.append(str(r["warning"]).replace("what_changed", "a same-dates comparison"))
    src = str(r.get("data_source", ""))
    if "synthetic" in src.lower() or "test" in src.lower():
        out.append("These figures come from synthetic test data, not real records.")
    if r.get("pooled_into_other"):
        out.append("Small groups were combined into \u201cOther\u201d: "
                   + ", ".join(map(str, r["pooled_into_other"])) + ".")
    if r.get("excluded_unknown"):
        out.append(f"{r['excluded_unknown']:,} records with no answer or \u201cprefer not to "
                   "say\u201d were left out of the group comparison.")
    if r.get("hidden"):
        out.append("Some groups were too small to show and were hidden to protect privacy.")
    for note in r.get("notes", [])[:2]:
        out.append(re.sub(r"\b[a-z]+(?:_[a-z]+)+\b", lambda m: _label(m.group(0)), str(note)))
    return out


def evidence_cards(trace):
    """One plain-language card per analysis step in `trace` [(tool, args, result), ...]."""
    cards = []
    for step in trace or []:
        try:
            name, args, result = step
        except (TypeError, ValueError):
            continue
        if name in META:
            continue
        title, source, method, certainty = TOOLS.get(
            name, ("An analysis step", None, "Calculated summary figures.", None))
        if name == "run_sql":
            method = sql_to_english((args or {}).get("sql", ""))
            source = _sql_source((args or {}).get("sql", ""))
        result = result if isinstance(result, dict) else {}
        card = {"title": _fill(title, args, result),
                "data": SOURCES.get(source, "The app's data"),
                "records": _records(result),
                "calculated": _fill(method, args, result) if method else None,
                "certainty": certainty,
                "keep_in_mind": _caveats(result)}
        if result.get("error"):
            card["failed"] = ("This step did not work: "
                              + re.sub(r"\b[a-z]+(?:_[a-z]+)+\b", lambda m: _label(m.group(0)),
                                       str(result["error"])))
        cards.append(card)
    return cards


def render_markdown(cards):
    if not cards:
        return "_No analysis steps were recorded for this answer._"
    parts = []
    for i, c in enumerate(cards, 1):
        lines = [f"**{i}. {c['title']}**"]
        if c.get("failed"):
            lines.append(f"- \u26a0\ufe0f {c['failed']}")
        lines.append(f"- **Data:** {c['data']}")
        if c.get("records"):
            lines.append(f"- **Records used:** {c['records']}")
        if c.get("calculated"):
            lines.append(f"- **What was calculated:** {c['calculated']}")
        if c.get("certainty"):
            lines.append(f"- **How certain:** {c['certainty']}")
        for k in c.get("keep_in_mind", []):
            lines.append(f"- **Keep in mind:** {k}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def render_text(cards):
    return render_markdown(cards).replace("**", "")


# ---------------------------------------------------------------- SQL in plain words

_AGG = [
    (r"count\s*\(\s*distinct\s+student_key\s*\)", "the number of distinct students"),
    (r"count\s*\(\s*distinct\s+response_id\s*\)", "the number of survey respondents"),
    (r"avg\s*\(\s*case\s+when\s+agree", "the share who agreed"),
    (r"sum\s*\(\s*case\s+when\s+outcome\s*=\s*'no_show'", "the number of missed appointments"),
    (r"sum\s*\(\s*case\s+when\s+outcome\s*=\s*'attended'", "the number of attended appointments"),
    (r"avg\s*\(\s*score", "the average rating"),
    (r"count\s*\(\s*\*\s*\)", "the number of records"),
    (r"median\s*\(", "a median"),
    (r"avg\s*\(", "an average"),
]


def _sql_source(sql):
    s = sql.lower()
    if "appointments" in s and ("ratings" in s or "responses" in s):
        return "both"
    return "chr" if "appointments" in s else "survey"


def _condition(c):
    c = c.strip()
    if c.startswith("(") and c.endswith(")"):
        c = c[1:-1].strip()
    m = re.match(r"(?i)lower\((\w+)\)\s*=\s*'([^']*)'", c) or re.match(r"(\w+)\s*=\s*'([^']*)'", c)
    if m:
        return f"{_label(m.group(1))} is {m.group(2).replace('_', ' ')}"
    m = re.match(r"(?i)(\w+)\s+in\s*\((.*)\)", c)
    if m:
        vals = [v.strip().strip("'") for v in m.group(2).split(",")]
        return f"{_label(m.group(1))} is " + (" or ".join(vals) if len(vals) <= 3
                                               else ", ".join(vals[:-1]) + " or " + vals[-1])
    m = re.match(r"(?i)(\w+)\s+between\s+'?([^' ]+)'?\s+and\s+'?([^' ]+)'?", c)
    if m:
        return f"{_label(m.group(1))} is between {m.group(2)} and {m.group(3)}"
    m = re.match(r"(\w+)\s*(>=|<=|>|<)\s*'?([^']+?)'?$", c)
    if m:
        word = {">=": "on or after" if "time" in m.group(1) or "date" in m.group(1) else "at least",
                "<=": "on or before" if "time" in m.group(1) or "date" in m.group(1) else "at most",
                ">": "after" if "time" in m.group(1) else "more than",
                "<": "before" if "time" in m.group(1) else "less than"}[m.group(2)]
        return f"{_label(m.group(1))} {word} {m.group(3)}"
    m = re.match(r"(?i)(\w+)\s+is\s+not\s+null", c)
    if m:
        return f"{_label(m.group(1))} is recorded"
    return None


def _join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def sql_to_english(sql: str) -> str:
    """Describe a summary query in plain words. Never returns code."""
    s = " ".join((sql or "").split())
    low = s.lower()
    tables = [TABLES.get(t, "the data") for t in re.findall(r"(?i)\b(?:from|join)\s+(\w+)", s)]
    tables = list(dict.fromkeys(tables)) or ["the data"]
    measures = []
    sel = low.split(" from ")[0]
    for pat, words in _AGG:
        if re.search(pat, sel) and words not in measures:
            measures.append(words)
            sel = re.sub(pat, "", sel)
    where = re.search(r"(?i)\bwhere\s+(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|\bhaving\b|$)", s)
    conds, skipped = [], False
    if where:
        for part in re.split(r"(?i)\s+and\s+(?![^()]*\))", where.group(1)):
            text = _condition(part)
            if text:
                conds.append(text)
            else:
                skipped = True
    group = re.search(r"(?i)\bgroup\s+by\s+(.*?)(?:\border\s+by\b|\blimit\b|\bhaving\b|$)", s)
    groups = [_label(g.strip()) for g in group.group(1).split(",")] if group else []
    out = f"Looked at {_join(tables)}"
    if conds:
        out += " where " + _join(conds)
    if skipped:
        out += (" (plus other conditions)" if conds else " with some conditions")
    out += ", and calculated " + (_join(measures) if measures else "summary figures")
    if groups:
        out += ", broken down by " + _join(groups)
    return out + ". Only summary figures were returned, never individual records."


# ---------------------------------------------------------------- follow-up questions

_SOURCES_Q = re.compile(
    r"(?i)\b(where (did|does|do) (this|that|these|those|it|the (answer|numbers?|figures?|data))"
    r"( \w+)? come from|how (did|do) you (get|calculate|work out|know|come up with|arrive at)"
    r"|what (is|was|are|were) (this|that|these|it|the answer) based on|show (me )?your work"
    r"|how was (this|that|it) (calculated|worked out|measured|counted)"
    r"|(what|which) (data|source|sources|records) (did you|was|were) use"
    r"|explain (how|where) (you|this|that|the answer)|sources? (for|of) (this|that|the answer)"
    r"|can i trust (this|that|these)|how reliable is (this|that))")


def is_sources_question(question: str) -> bool:
    return bool(_SOURCES_Q.search(question or ""))


def last_trace(history):
    """Analysis steps behind the most recent answer that has any, whatever shape a stored
    turn has (a dict with 'trace', or a dict holding such a dict)."""
    for turn in reversed(list(history or [])):
        candidates = [turn]
        if isinstance(turn, dict):
            candidates += [v for v in turn.values() if isinstance(v, dict)]
        elif isinstance(turn, (list, tuple)):
            candidates += [v for v in turn if isinstance(v, dict)]
        for c in candidates:
            if isinstance(c, dict) and isinstance(c.get("trace"), list):
                steps = [s for s in c["trace"]
                         if isinstance(s, (list, tuple)) and len(s) == 3 and s[0] not in META]
                if steps:
                    return steps
    return []


SOURCES_PROMPT = (
    "You explain to university staff with no technical background where a previous answer "
    "came from. Use ONLY the evidence cards provided. For each step, say in plain words: which "
    "data it used and which records (service, dates, groups, how many), what was calculated, "
    "how certainty was checked, and what to keep in mind. Keep every number exactly as it "
    "appears in the cards. Never mention code, queries, tables, column names or tool names, "
    "and do not add new findings. If a step failed, say so. Finish with one sentence on how "
    "much weight the answer can bear. Short paragraphs, no more than about 250 words.")

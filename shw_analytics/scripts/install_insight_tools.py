"""One-time installer: connects care_patterns, playbook and provenance to the model.

Run from the project folder (the one containing app.py):
    python scripts/install_insight_tools.py

It changes two files, and only if every anchor it needs is found exactly once:
  shw_analytics/tools.py  - wrappers, dispatch entries and model-facing specs for
                            care_journeys, noshow_patterns, service_pathways, insight_playbook
  shw_analytics/agent.py  - a system-prompt rule to use the playbook for deeper questions,
                            plain-language answers to "where did this come from?", and the new
                            tools counted as statistically tested
Both files are syntax-checked before anything is written. Safe to re-run: it stops if the
changes are already there.
"""
import sys
from pathlib import Path

TOOLS = Path("shw_analytics/tools.py")
AGENT = Path("shw_analytics/agent.py")

WRAPPERS = '''def care_journeys(service_line: str = None, dimension: str = None,
                  return_days: int = 30, follow_days: int = 60):
    from . import care_patterns
    con, err = _need_chr()
    if err:
        return err
    try:
        r = care_patterns.care_journeys(con, service_line, dimension, int(return_days), int(follow_days))
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def noshow_patterns(service_line: str = None, term: str = None):
    from . import care_patterns
    con, err = _need_chr()
    if err:
        return err
    try:
        r = care_patterns.noshow_patterns(con, service_line, term)
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def service_pathways(within_days: int = 30):
    from . import care_patterns
    con, err = _need_chr()
    if err:
        return err
    try:
        r = care_patterns.service_pathways(con, int(within_days))
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def insight_playbook(question: str = None, topic_id: str = None):
    from . import playbook
    return playbook.insight_playbook(question, topic_id)


'''

SPECS = '''

def _insight_spec(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


TOOL_SPECS += [
    _insight_spec("care_journeys",
          "What happens after a student's first visit in a new episode of care: share with a single "
          "visit, share returning within 30 days, median days to a second visit, and whether "
          "students attend again after missing an appointment. Optional tested comparison of "
          "student groups. Use for continuity, return visits and 'do students come back'. A single "
          "visit can be planned single-session care, so never call it dropout.",
          {"service_line": {"type": "string", "description": "Optional, e.g. Counselling."},
           "dimension": {"type": "string", "description": "Optional group to compare: faculty, "
                         "academic_level, gender, age_band or program."},
           "return_days": {"type": "integer", "description": "Window for a quick return (default 30)."},
           "follow_days": {"type": "integer", "description": "Follow-up window (default 60)."}}),
    _insight_spec("noshow_patterns",
          "Which kinds of appointments are missed more often: by format, weekday, time of day, "
          "booking history (first booking / missed before) and student group. Tested with each "
          "student counted once and corrected for many tests. Use for no-show hotspots and "
          "reminder targeting. Patterns, not causes.",
          {"service_line": {"type": "string", "description": "Optional, e.g. Counselling."},
           "term": {"type": "string", "description": "Optional term, e.g. 2025 Fall."}}),
    _insight_spec("service_pathways",
          "Of students starting a new episode of care in one service, the share who use another "
          "service soon after (e.g. medical clinic to counselling), with 95% ranges. Use for "
          "integrated or team-based care questions. Shows movement, not referrals.",
          {"within_days": {"type": "integer", "description": "Days after the start (default 30)."}}),
    _insight_spec("insight_playbook",
          "Look up how to answer a broad or deeper question well: which analyses to run, how to "
          "read them, what to avoid and how to word the answer. Call FIRST for questions like "
          "'what should we know', 'anything surprising', 'where should we focus', 'are we reaching "
          "X students', 'is integrated care working', or anything about continuity of care.",
          {"question": {"type": "string", "description": "The staff question, as asked."},
           "topic_id": {"type": "string", "description": "Optional id from a previous lookup."}}),
]
'''

INSIGHT_RULE = '''INSIGHT_RULE = (
    "\\n\\nDeeper questions: for broad or open questions (what should we know, anything "
    "surprising, where to focus, are we reaching a group, is integrated care working) and for "
    "questions about continuity of care, missed appointments or movement between services, first "
    "call insight_playbook with the question and follow the steps it returns. Respect its "
    "'avoid' list and wording. If its status is needs_data, explain what data is missing "
    "instead of estimating.")


def _explain_sources(question, prev_trace, chat_fn, ctx):
    """Answer 'where did this come from?' in plain words, from evidence cards built from the
    previous answer's analysis steps. Falls back to the cards themselves if the model's
    wording mentions internal names or fails."""
    from . import provenance
    cards = provenance.evidence_cards(prev_trace)
    trace = list(prev_trace) + [("evidence_cards", {}, {"cards": cards})]
    if not cards:
        return {"answer": "The previous answer did not use any data analysis, so there are no "
                          "sources to show.", "trace": trace, "mode": "sources", "ungrounded": []}
    messages = [{"role": "system", "content": provenance.SOURCES_PROMPT},
                {"role": "user", "content": f"Staff question: {question}\\n\\nEvidence cards for "
                                            f"the previous answer:\\n\\n{provenance.render_text(cards)}"}]
    try:
        answer = (chat_fn(messages).content or "").strip()
    except Exception:
        answer = ""
    if not answer or mentions_internals(answer):
        answer = provenance.render_markdown(cards)
    return _finalise(answer, trace, "sources", ctx)


'''

SOURCES_HOOK = '''
    from . import provenance
    if provenance.is_sources_question(question):
        prev = provenance.last_trace(history) or [
            s for s in (ctx or []) if isinstance(s, (list, tuple)) and len(s) == 3]
        if prev:
            return _explain_sources(question, prev, chat_fn, ctx)'''


def fail(msg):
    print("NOT INSTALLED:", msg)
    print("Nothing was changed. Paste this message to get a fix.")
    sys.exit(1)


def once(text, anchor, where):
    n = text.count(anchor)
    if n != 1:
        fail(f"expected {anchor!r} exactly once in {where}, found {n}")


def main():
    if not TOOLS.exists() or not AGENT.exists():
        fail("run this from the project folder (the one with app.py and shw_analytics/)")
    for mod in ("care_patterns.py", "playbook.py", "playbook.yaml", "provenance.py"):
        if not (Path("shw_analytics") / mod).exists():
            fail(f"shw_analytics/{mod} is missing; copy the new files in first")
    t, g = TOOLS.read_text(), AGENT.read_text()
    if "def care_journeys(" in t or "INSIGHT_RULE" in g:
        print("Already installed; nothing to do.")
        return

    once(t, "def survey_vs_visits(", "tools.py")
    once(t, '"what_changed": what_changed,', "tools.py")
    t = t.replace("def survey_vs_visits(", WRAPPERS + "def survey_vs_visits(")
    t = t.replace('"what_changed": what_changed,',
                  '"what_changed": what_changed, "care_journeys": care_journeys, '
                  '"noshow_patterns": noshow_patterns, "service_pathways": service_pathways, '
                  '"insight_playbook": insight_playbook,')
    t = t.rstrip("\n") + "\n" + SPECS

    once(g, 'INTERVAL_TOOLS = {"what_changed"}', "agent.py")
    once(g, "def _system(json_mode):", "agent.py")
    once(g, "+ PERIOD_RULE", "agent.py")
    once(g, "    ctx = _context_trace(question, history)\n", "agent.py")
    g = g.replace('INTERVAL_TOOLS = {"what_changed"}',
                  'INTERVAL_TOOLS = {"what_changed", "care_journeys", "noshow_patterns", "service_pathways"}')
    g = g.replace("def _system(json_mode):", INSIGHT_RULE + "def _system(json_mode):")
    g = g.replace("+ PERIOD_RULE", "+ PERIOD_RULE + INSIGHT_RULE")
    g = g.replace("    ctx = _context_trace(question, history)\n",
                  "    ctx = _context_trace(question, history)" + SOURCES_HOOK + "\n")

    try:
        compile(t, str(TOOLS), "exec")
        compile(g, str(AGENT), "exec")
    except SyntaxError as e:
        fail(f"the patched code would not compile: {e}")
    TOOLS.write_text(t)
    AGENT.write_text(g)
    print("Installed: 4 tools in tools.py; playbook rule, sources answers and tested-tool list in agent.py.")


if __name__ == "__main__":
    main()

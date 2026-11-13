"""Lets the assistant use the term briefing. Run once from the project folder:
    python scripts/install_briefing.py

Changes, each only if not already present and only if its anchor is found:
  shw_analytics/tools.py  - a term_briefing tool (wrapper, dispatch entry, model-facing spec)
  shw_analytics/agent.py  - term_briefing counted as statistically tested
Both files are syntax-checked before anything is written. Safe to re-run.
"""
import re
import sys
from pathlib import Path

TOOLS, AGENT = Path("shw_analytics/tools.py"), Path("shw_analytics/agent.py")

WRAPPER = '''def term_briefing():
    from . import briefing
    try:
        return briefing.for_chat()
    except Exception as e:
        return {"error": f"The briefing could not be built: {e}"}


'''
SPEC = '''

TOOL_SPECS.append({"type": "function", "function": {
    "name": "term_briefing",
    "description": (
        "The term briefing: runs every tested analysis across services (change since the same dates "
        "last year, missed appointments, returning after a first visit, what matters most in the "
        "survey, gaps between groups), corrects for having run them all, and returns the findings "
        "that held up, ranked, plus what stayed the same and what could not be checked. Use for "
        "'what should we know', 'headline findings', 'what should leadership know', 'anything "
        "surprising', 'where should we focus'. One call is enough; do not re-run the analyses."),
    "parameters": {"type": "object", "properties": {}, "required": []}}})
'''


def fail(msg):
    print("NOT INSTALLED:", msg, "\nNothing was changed. Paste this message to get a fix.")
    sys.exit(1)


def main():
    if not TOOLS.exists() or not AGENT.exists() or not Path("shw_analytics/briefing.py").exists():
        fail("run from the project folder, with shw_analytics/briefing.py in place")
    t0, g0 = TOOLS.read_text(), AGENT.read_text()
    t, g = t0, g0
    if "def term_briefing(" not in t:
        if t.count("def survey_vs_visits(") != 1:
            fail("could not find def survey_vs_visits( in tools.py")
        t, n = re.subn(r'("insight_playbook": insight_playbook,)', r'\1 "term_briefing": term_briefing,', t, count=1)
        if n != 1:
            fail("could not find the tool dispatch table in tools.py (run the earlier installers first)")
        t = t.replace("def survey_vs_visits(", WRAPPER + "def survey_vs_visits(")
        t = t.rstrip("\n") + "\n" + SPEC
    line = re.search(r"(?m)^INTERVAL_TOOLS = \{.*$", g)
    if not line:
        fail("could not find INTERVAL_TOOLS in agent.py (run the earlier installers first)")
    if '"term_briefing"' not in line.group(0):
        g = g.replace(line.group(0), line.group(0).replace("INTERVAL_TOOLS = {", 'INTERVAL_TOOLS = {"term_briefing", ', 1))
    # A service whose first appointment falls a day or two into the comparison window was being
    # refused; windows now come from the whole appointment record.
    ins_path = Path("shw_analytics/insights.py")
    i0 = ins_path.read_text() if ins_path.exists() else ""
    i = i0
    if i0 and "span = pd.to_datetime(_appts(con)" not in i0:
        i, n = re.subn(r'(?m)^(\s*)\(a0, a1\), \(b0, b1\) = _windows\(df\["start_time"\], compare, int\(window_days\)\)\s*$',
                       lambda m: (f'{m.group(1)}span = pd.to_datetime(_appts(con)["start_time"]) if service_line else df["start_time"]\n'
                                  f'{m.group(1)}(a0, a1), (b0, b1) = _windows(span, compare, int(window_days))'), i0, count=1)
        if n != 1:
            i = i0
            print("Note: could not update the comparison-window line in insights.py; everything else was installed.")
    if (t, g, i) == (t0, g0, i0):
        print("Already installed; nothing to do.")
        return
    try:
        compile(t, str(TOOLS), "exec")
        compile(g, str(AGENT), "exec")
        compile(i or "pass", "insights.py", "exec")
    except SyntaxError as e:
        fail(f"the patched code would not compile: {e}")
    TOOLS.write_text(t)
    AGENT.write_text(g)
    if i != i0:
        ins_path.write_text(i)
    print("Installed: the assistant can now use the term briefing.")


if __name__ == "__main__":
    main()

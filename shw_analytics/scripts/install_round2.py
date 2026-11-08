"""Round-2 installer. Run from the project folder (the one containing app.py):
    python scripts/install_round2.py

Changes, each applied only if not already present:
  agent.py  - forced final answer when the step limit is reached (instead of "I couldn't
              finish"), and a guard against re-running the same query; matched with
              whitespace-tolerant patterns inside _ask_native only
            - evidence wording rules in the system prompt (null results, dates per finding,
              combinations of groups)
            - priority_items counted as a statistically tested tool
  tools.py  - priority_items wrapper, dispatch entry and model-facing spec
Both files are syntax-checked before anything is written; if any step cannot find what it
needs, nothing is changed.
"""
import re
import sys
from pathlib import Path

TOOLS = Path("shw_analytics/tools.py")
AGENT = Path("shw_analytics/agent.py")


def fail(msg):
    print("NOT INSTALLED:", msg)
    print("Nothing was changed. Paste this message to get a fix.")
    sys.exit(1)


def flex(text):
    """Regex for `text` that tolerates any whitespace differences."""
    return r"\s+".join(re.escape(tok) for tok in text.split())


HELPERS = '''FINAL_NUDGE = ("You have reached the step limit. Answer the question now using only the tool "
               "results above. Give n with every number, say what the results do not show, "
               "and do not call any more tools.")
REPEAT_NOTE = ("You already ran this (only names or formatting differ). Use the earlier result: "
               "answer now, or try a genuinely different query.")


def _call_key(name, args):
    """Same call = same tool and same arguments; for SQL, ignore aliases and spacing."""
    if name == "run_sql":
        sql = re.sub(r"\\s+as\\s+\\w+", "", str(args.get("sql", "")), flags=re.I)
        return (name, " ".join(sql.lower().split()))
    return (name, json.dumps(args, sort_keys=True))


'''

FORCED_END = '''messages.append({"role": "user", "content": FINAL_NUDGE})
    msg = chat_fn(messages)
    if (getattr(msg, "content", None) or "").strip():
        trace.append(("forced_answer", {}, {}))
        return _finalise(msg.content, trace, "native", ctx)
    return {"answer": "I couldn't finish within the step limit; try a narrower question.",
            "trace": trace, "mode": "native"}'''

EVIDENCE_RULE = '''EVIDENCE_RULE = (
    "\\n\\nEvidence rules: (1) A result that is not distinguishable from noise is absence of "
    "evidence, not proof of no effect: never write that something 'is not the issue' or 'was "
    "tested and is not the problem'; say the data could not detect a difference and, if known, "
    "how large a difference it could detect. (2) When a verdict is 'no clear change', mention "
    "the group figures behind it only as context, never as a finding or a 'hidden shift'. "
    "(3) State the dates each finding covers, especially when combining same-dates comparisons "
    "with results that cover all dates. (4) Questions about combinations of groups (e.g. "
    "first-generation students in particular faculties) cannot be checked systematically yet: "
    "say so first, give one-at-a-time results, never answer 'no' to the combination, and never "
    "build the combination from custom queries.")


'''

WRAPPER = '''def priority_items(service: str):
    from . import drivers
    try:
        return drivers.priority_items(_con(), service)
    except ValueError as e:
        return {"error": str(e)}


'''

SPEC = '''

TOOL_SPECS.append({"type": "function", "function": {
    "name": "priority_items",
    "description": (
        "Key-driver analysis for one service's survey: which questions are most closely linked "
        "to students' overall rating of the service (satisfaction, or recommending), combined with "
        "how well each scores. Returns 'fix first' priorities only when they are stable across "
        "re-samples of students, or says no single question stands out. Use for 'what should we "
        "focus on', 'what matters most', 'is the lowest-scoring area what hurts satisfaction'. "
        "Say 'most closely linked to', never 'drives'."),
    "parameters": {"type": "object", "properties": {
        "service": {"type": "string", "description": "counselling, medical, wellness_adv or info_desk."}},
        "required": ["service"]}}})
'''


def patch_agent(g):
    if "FINAL_NUDGE" not in g:
        m = re.search(r"\ndef _ask_native\(", g)
        if not m:
            fail("could not find def _ask_native( in agent.py")
        start = m.start() + 1
        nxt = re.search(r"\ndef \w+\(", g[start + 1:])
        end = start + 1 + nxt.start() if nxt else len(g)
        body = g[start:end]
        body, n1 = re.subn(r"(?m)^(\s*)seen = \{\}\s*$", r"\1seen = {}\n\1asked = set()", body, count=1)
        body, n2 = re.subn(
            r"(?m)^(\s*)result = _run_tool\(tc\.function\.name,\s*args,\s*seen\)\s*$",
            lambda mm: (f"{mm.group(1)}key = _call_key(tc.function.name, args)\n"
                        f"{mm.group(1)}if key in asked:\n"
                        f"{mm.group(1)}    result = {{\"note\": REPEAT_NOTE}}\n"
                        f"{mm.group(1)}else:\n"
                        f"{mm.group(1)}    asked.add(key)\n"
                        f"{mm.group(1)}    result = _run_tool(tc.function.name, args, seen)"),
            body, count=1)
        end_pat = flex('return {"answer": "I couldn\'t finish within the step limit; try a '
                       'narrower question.", "trace": trace, "mode": "native"}')
        body, n3 = re.subn(end_pat, lambda mm: FORCED_END, body, count=1)
        if (n1, n2, n3) != (1, 1, 1):
            fail(f"_ask_native did not look as expected (seen={n1}, run_tool={n2}, step-limit return={n3})")
        g = g[:start] + HELPERS + body + g[end:]
    if "EVIDENCE_RULE" not in g:
        if g.count("+ PERIOD_RULE + INSIGHT_RULE") != 1 or g.count("def _system(json_mode):") != 1:
            fail("system prompt line not found; run scripts/install_insight_tools.py first")
        g = g.replace("def _system(json_mode):", EVIDENCE_RULE + "def _system(json_mode):")
        g = g.replace("+ PERIOD_RULE + INSIGHT_RULE", "+ PERIOD_RULE + INSIGHT_RULE + EVIDENCE_RULE")
    if '"priority_items"' not in g.split("INTERVAL_TOOLS = ", 1)[-1].split("\n", 1)[0]:
        g, n = re.subn(r"INTERVAL_TOOLS = \{", 'INTERVAL_TOOLS = {"priority_items", ', g, count=1)
        if n != 1:
            fail("INTERVAL_TOOLS not found in agent.py")
    return g


def patch_tools(t):
    if "def priority_items(" in t:
        return t
    for a in ("def survey_vs_visits(", '"insight_playbook": insight_playbook,'):
        if t.count(a) != 1:
            fail(f"expected {a!r} exactly once in tools.py; run scripts/install_insight_tools.py first")
    t = t.replace("def survey_vs_visits(", WRAPPER + "def survey_vs_visits(")
    t = t.replace('"insight_playbook": insight_playbook,',
                  '"insight_playbook": insight_playbook, "priority_items": priority_items,')
    return t.rstrip("\n") + "\n" + SPEC


def main():
    if not TOOLS.exists() or not AGENT.exists():
        fail("run this from the project folder (the one with app.py and shw_analytics/)")
    if not Path("shw_analytics/drivers.py").exists():
        fail("shw_analytics/drivers.py is missing; copy the new files in first")
    t0, g0 = TOOLS.read_text(), AGENT.read_text()
    t, g = patch_tools(t0), patch_agent(g0)
    if (t, g) == (t0, g0):
        print("Already installed; nothing to do.")
        return
    try:
        compile(t, str(TOOLS), "exec")
        compile(g, str(AGENT), "exec")
    except SyntaxError as e:
        fail(f"the patched code would not compile: {e}")
    TOOLS.write_text(t)
    AGENT.write_text(g)
    print("Installed: forced final answer + repeat guard, evidence rules, priority_items.")


if __name__ == "__main__":
    main()

"""Keeps statistics jargon out of the assistant's answers. Run once from the project folder:
    python scripts/install_plain_language.py

Changes shw_analytics/agent.py in two places (only if not already done, and only if both
anchors are found; the file is syntax-checked before it is written):
  * a system-prompt rule: no p-values, q-values or "CI" in answers; give the range in words
  * a safety net: every final answer passes through plain.plain_stats, which removes any
    that slip through
"""
import re
import sys
from pathlib import Path

AGENT = Path("shw_analytics/agent.py")
RULE = '''PLAIN_RULE = (
    "\\n\\nPlain numbers: staff are not statisticians. Never write p-values, q-values, 'CI' or "
    "'confidence interval'. Say whether a difference was tested and is real, or is within noise, "
    "and give the range in words, e.g. '22.8% more students (range 8.6% to 38.8%)'. The same goes "
    "for test statistics such as chi-square, degrees of freedom or Cramer's V: use them to decide "
    "what to say, and leave the figures to the technical details under the answer. This rule takes "
    "precedence over any tool, playbook or earlier rule that says to report them. Write date "
    "windows as '1 Sep to 31 Oct'.")


'''


def fail(msg):
    print("NOT INSTALLED:", msg, "\nNothing was changed. Paste this message to get a fix.")
    sys.exit(1)


def main():
    if not AGENT.exists() or not Path("shw_analytics/plain.py").exists():
        fail("run from the project folder, with shw_analytics/plain.py in place")
    g = AGENT.read_text()
    if "PLAIN_RULE" in g and "plain_stats" in g:
        print("Already installed; nothing to do.")
        return
    if "PLAIN_RULE" not in g:
        if g.count("def _system(json_mode):") != 1:
            fail("could not find def _system(json_mode): in agent.py")
        g, n = re.subn(r"(?m)^(\s*s = SYSTEM\.format\(.*)$", lambda m: m.group(1) + " + PLAIN_RULE", g, count=1)
        if n != 1:
            fail("could not find the line that builds the system prompt")
        g = g.replace("def _system(json_mode):", RULE + "def _system(json_mode):")
    if "plain_stats" not in g:
        g, n = re.subn(r"(?m)^(def _finalise\(answer, trace, mode, ctx=\(\)\):\n)",
                       lambda m: m.group(1) + "    from .plain import plain_stats\n    answer = plain_stats(answer)\n",
                       g, count=1)
        if n != 1:
            fail("could not find def _finalise(answer, trace, mode, ctx=()) in agent.py")
    try:
        compile(g, str(AGENT), "exec")
    except SyntaxError as e:
        fail(f"the patched code would not compile: {e}")
    AGENT.write_text(g)
    print("Installed: plain-number rule in the system prompt, and jargon removed from every final answer.")


if __name__ == "__main__":
    main()

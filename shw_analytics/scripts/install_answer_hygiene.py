"""Stops two things reaching staff: false "could not be traced" warnings on numbers written with
commas, and "good catch, I re-checked..." preambles. Run once from the project folder:
    python scripts/install_answer_hygiene.py

Adds one clearly marked block to the END of shw_analytics/agent.py. Nothing else in the file is
touched, so it works whatever earlier fixes are or are not in place. Safe to re-run.
"""
import sys
from pathlib import Path

AGENT = Path("shw_analytics/agent.py")
MARK = "# ---- answer hygiene (added by scripts/install_answer_hygiene.py)"
BLOCK = '''

''' + MARK + ''' ----------------------
# 1) Numbers written the way people write them ("1,717", "9.1e-05") are read as one number before
#    they are checked against the tool results, so correct figures are not flagged.
# 2) When the model is asked to rewrite an answer, it must return the answer only. Staff never see
#    the request, so "good catch, I re-checked..." makes no sense to them.
_THOUSANDS = re.compile(r"(?<=\\d),(?=\\d{3}(?!\\d))")
_SCI = re.compile(r"(?<![\\w.])\\d+(?:\\.\\d+)?[eE][-+]?\\d+")
_check_numbers = ungrounded_numbers


def ungrounded_numbers(answer, trace):
    text = _THOUSANDS.sub("", answer or "")
    sci = _SCI.findall(text)
    bad = list(_check_numbers(_SCI.sub(" ", text), trace))
    if sci:
        blob = json.dumps([s[2] for s in trace if isinstance(s, (tuple, list)) and len(s) == 3], default=str)
        bad += [x for x in sci if x not in blob]
    return sorted(set(bad))


_ANSWER_ONLY = (" Reply with the complete answer for the staff member and nothing else: do not mention this "
                "note, and do not say that anything was checked again, corrected or removed.")
for _name in ("NO_TOOL_NUDGE", "OFF_TOPIC_NUDGE", "HANDOFF_NUDGE", "ITEM_CODE_NUDGE", "FINAL_NUDGE"):
    if _name in globals():
        globals()[_name] = globals()[_name] + _ANSWER_ONLY
_retry_msg = _grounding_retry_msg


def _grounding_retry_msg(bad):
    return _retry_msg(bad) + _ANSWER_ONLY
'''


def main():
    if not AGENT.exists():
        print("NOT INSTALLED: run from the project folder.")
        sys.exit(1)
    s = AGENT.read_text()
    if MARK in s:
        print("Already installed; nothing to do.")
        return
    for name in ("def ungrounded_numbers(", "def _grounding_retry_msg(", "import re", "import json"):
        if name not in s:
            print(f"NOT INSTALLED: agent.py has no '{name}'. Nothing was changed; paste this message to get a fix.")
            sys.exit(1)
    new = s.rstrip("\n") + "\n" + BLOCK
    compile(new, str(AGENT), "exec")
    AGENT.write_text(new)
    print("Installed: comma-formatted numbers are checked correctly, and rewrites return the answer only.")


if __name__ == "__main__":
    main()

"""Removes the last instruction-style wording the assistant reads. Run once from the project folder:
    python scripts/install_neutral_wording.py

Changes one phrase in the priority_items tool description in shw_analytics/tools.py
("'fix first' priorities" becomes "priorities"). Safe to re-run.
"""
from pathlib import Path

TOOLS = Path("shw_analytics/tools.py")
s = TOOLS.read_text()
if "'fix first' priorities" in s:
    s = s.replace("'fix first' priorities", "priorities")
    compile(s, str(TOOLS), "exec")
    TOOLS.write_text(s)
    print("Installed: neutral wording in the tool description.")
else:
    print("Already installed; nothing to do.")

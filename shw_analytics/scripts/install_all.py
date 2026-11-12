"""Runs every installer in order. Each one only changes what is missing, so this is safe to run
any number of times, and it does not matter which ones you have run before.

    python scripts/install_all.py
"""
import subprocess
import sys
from pathlib import Path

ORDER = ["install_insight_tools.py", "install_round2.py", "install_plain_language.py", "install_briefing.py",
         "install_usage_log.py", "install_labels_fix.py", "install_neutral_wording.py", "install_answer_hygiene.py"]

failed = []
for name in ORDER:
    path = Path("scripts") / name
    if not path.exists():
        print(f"{name:32} (not present, skipped)")
        continue
    r = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or r.stderr.strip().splitlines() or ["(no output)"])[0]
    print(f"{name:32} {line}")
    if r.returncode != 0:
        failed.append(name)
print("\nAll installers are in place." if not failed else f"\nNeeds attention: {', '.join(failed)}. Paste the lines above to get a fix.")
sys.exit(1 if failed else 0)

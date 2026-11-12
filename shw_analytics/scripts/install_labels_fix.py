"""Answer labels for survey groups, and forgiving group names. Run once from the project folder:
    python scripts/install_labels_fix.py

Changes, each only if not already present and only if its anchor is found:
  shw_analytics/ingest.py    - answer labels can also come from data/value_labels.yaml (kept with
                               the data, e.g. written by the synthetic survey generator). Labels in
                               dictionary.yaml always win.
  shw_analytics/analysis.py  - "graduate" matches the group "Graduate": group names are matched
                               without regard to capitals.
Both files are syntax-checked before anything is written. Safe to re-run.
"""
import re
import sys
from pathlib import Path

INGEST, ANALYSIS = Path("shw_analytics/ingest.py"), Path("shw_analytics/analysis.py")

LOADER = '''def load_dictionary(path=None):
    with open(path or config.DICTIONARY_PATH) as f:
        d = yaml.safe_load(f)
    # Answer labels kept with the data (data/value_labels.yaml, e.g. for a synthetic export) fill
    # in whatever dictionary.yaml does not define. Labels in dictionary.yaml always win.
    extra = config.DB_PATH.parent / "value_labels.yaml"
    if path is None and extra.exists():
        with open(extra) as f:
            more = (yaml.safe_load(f) or {}).get("value_labels") or {}
        merged = {q: {str(k): v for k, v in labels.items()} for q, labels in more.items()}
        for q, labels in (d.get("value_labels") or {}).items():
            merged.setdefault(q, {}).update({str(k): v for k, v in labels.items()})
        d["value_labels"] = merged
    return d
'''
MATCH = '''    valid = focus_values(con, focus["dimension"])
    same = {str(v).strip().lower(): v for v in valid}.get(str(focus["value"]).strip().lower())
    if same is not None:                          # "graduate" finds "Graduate"
        focus = {**focus, "value": same}
    if str(focus["value"]) not in valid:'''


def main():
    if not INGEST.exists() or not ANALYSIS.exists():
        print("NOT INSTALLED: run from the project folder.")
        sys.exit(1)
    i0, a0 = INGEST.read_text(), ANALYSIS.read_text()
    i, a, notes = i0, a0, []
    if "value_labels.yaml" not in i:
        i, n = re.subn(r"(?m)^def load_dictionary\(path=None\):\n    with open\(path or config\.DICTIONARY_PATH\) as f:\n        return yaml\.safe_load\(f\)\n",
                       lambda m: LOADER, i, count=1)
        notes.append("answer labels from data/value_labels.yaml" if n else "COULD NOT update load_dictionary in ingest.py")
    if "same = {str(v).strip().lower()" not in a:
        a, n = re.subn(r'(?m)^    valid = focus_values\(con, focus\["dimension"\]\)\n    if str\(focus\["value"\]\) not in valid:',
                       lambda m: MATCH, a, count=1)
        notes.append("group names matched without regard to capitals" if n else "COULD NOT update group_vs_everyone in analysis.py")
    if (i, a) == (i0, a0) and not notes:
        print("Already installed; nothing to do.")
        return
    compile(i, str(INGEST), "exec")
    compile(a, str(ANALYSIS), "exec")
    INGEST.write_text(i)
    ANALYSIS.write_text(a)
    print("Installed: " + "; ".join(notes) + ".")


if __name__ == "__main__":
    main()

"""Check the app's appointment counts against the CHR dashboards.

    python scripts/reconcile_chr.py                      # step 1: write a sheet to fill in, and show how the app counts
    python scripts/reconcile_chr.py --compare FILE.csv   # step 2: compare once dashboard figures are typed in

Step 1 writes data/reconcile_template.csv (the latest six months; --months changes that). Someone
with the CHR dashboard open types its figure into the dashboard_value column for as many rows as
they like. Step 2 reports which way of counting the dashboard matches and where the two differ,
and writes data/reconciliation_report.md. No model is used.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import config, reconcile, tools  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", help="the filled-in sheet")
    ap.add_argument("--months", type=int, default=6)
    ap.add_argument("--out", default="data/reconciliation_report.md")
    a = ap.parse_args()
    con = tools._con()
    rule_lines = reconcile.rules(con)
    if not a.compare:
        path = config.DB_PATH.parent / "reconcile_template.csv"
        sheet = reconcile.template(con, a.months)
        sheet.to_csv(path, index=False)
        print(f"Wrote {path}: {len(sheet)} rows. Type the dashboard's figures into dashboard_value, then run:")
        print(f"  python scripts/reconcile_chr.py --compare {path}\n\nHow the app counts:")
        for r in rule_lines:
            print("  -", r)
        return
    result = reconcile.compare(con, pd.read_csv(a.compare, dtype=str))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(reconcile.report(result, rule_lines))
    print(f"Compared {result['rows_compared']} dashboard figures.")
    for measure, m in result["measures"].items():
        print(f"  {measure}: the dashboard matches the app counting {m['meaning']} "
              f"({m['exact']} of {m['compared']} exact, {m['within_1pct']} within 1%; {len(m['differences'])} differ).")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()

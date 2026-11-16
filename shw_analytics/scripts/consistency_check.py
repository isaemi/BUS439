"""Asks ten questions three ways each and reports whether the answers agree.

    python scripts/consistency_check.py                 # all ten questions (30 requests, a few minutes)
    python scripts/consistency_check.py --only demand   # one question
    python scripts/consistency_check.py --list          # show the questions without asking anything

Uses the model in your .env, so it costs a little on the gateway (about the same as asking 30
questions in the app). Writes data/consistency_report.md.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import consistency  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="id of one question, e.g. demand")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default="data/consistency_report.md")
    a = ap.parse_args()
    questions = [q for q in consistency.QUESTIONS if not a.only or q["id"] == a.only]
    if a.list or not questions:
        for q in consistency.QUESTIONS:
            print(f"{q['id']:14} {q['topic']}")
            for p in q["phrasings"]:
                print(f"{'':14}   - {p}")
        return
    from shw_analytics import agent

    def progress(q, text, run_):
        tools = ", ".join(sorted({s[0] for s in run_["trace"] if s[0] not in consistency.HELPERS})) or "no analysis"
        print(f"  {run_['seconds']:>5.1f}s  {text[:62]:64} {tools}")

    print(f"Asking {sum(len(q['phrasings']) for q in questions)} questions...")
    results = consistency.run(lambda text: agent.ask(text), questions, progress=progress)
    s = consistency.summary(results)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(consistency.report(results, when=f"Run on {dt.date.today():%d %B %Y}."))
    print(f"\nSame analysis however it was worded: {s['same_analysis']} of {s['questions']} questions")
    print(f"Same service, measure and group:     {s['same_settings']} of {s['questions']}")
    print(f"Answers with only traceable figures: {s['answers'] - s['with_untraceable_figures']} of {s['answers']}")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()

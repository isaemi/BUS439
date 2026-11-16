"""Does the same question, asked in different words, get the same answer?

SHW's leadership named consistency as a condition for trusting the tool: "different phrasings or
formats of the same question should produce consistent results." This module asks each question
several ways, each in a fresh conversation, and compares what came back:

  same analysis    the same analysis tools ran (helpers such as the playbook are ignored)
  same settings    with the same service, measure and group
  shared figures   the numbers every phrasing's answer reports
  traceable        no answer contained a figure that could not be traced to a tool result

run(ask) does the asking; compare(runs) and report(results) are pure functions, tested offline.
"""
from __future__ import annotations

import re
import time

HELPERS = {"insight_playbook", "describe_data", "lookup_survey_items"}
KEY_ARGS = ("service", "service_line", "metric", "dimension", "compare", "value", "item", "outcome_item")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_IGNORE = {"30", "60", "61", "90", "95", "100", "1000"}

QUESTIONS = [
    {"id": "demand", "topic": "Change in students seen", "phrasings": [
        "Has the number of students we see changed since last year?",
        "Are we busier than this time last year?",
        "How does demand this fall compare with last fall?"]},
    {"id": "noshow_where", "topic": "Where appointments are missed", "phrasings": [
        "Which appointments are missed most often?",
        "Where are our no-shows highest?",
        "What kinds of appointments do students skip the most?"]},
    {"id": "noshow_change", "topic": "Change in the no-show rate", "phrasings": [
        "Have no-shows changed since last year?",
        "Is the no-show rate different from a year ago?",
        "Are students missing more appointments than they did last year?"]},
    {"id": "after_first", "topic": "After a first counselling visit", "phrasings": [
        "What happens after a student's first counselling visit?",
        "Do counselling students come back after their first appointment?",
        "How many counselling students only come once?"]},
    {"id": "priorities", "topic": "What matters most in counselling", "phrasings": [
        "What matters most to students in counselling?",
        "Which parts of counselling are most closely linked to overall satisfaction?",
        "What should counselling focus on to improve students' overall experience?"]},
    {"id": "equity", "topic": "Gaps between student groups", "phrasings": [
        "Do some groups of students rate our services lower?",
        "Is any group of students having a worse experience with us?",
        "Are there gaps in satisfaction between student groups?"]},
    {"id": "survey_change", "topic": "Change in survey ratings", "phrasings": [
        "Have survey ratings changed since last year?",
        "Are any groups of students rating us worse than they used to?",
        "What has changed in the survey results compared with last fall?"]},
    {"id": "grad_use", "topic": "Graduate and undergraduate counselling use", "phrasings": [
        "Do graduate students use counselling differently from undergraduates?",
        "How does counselling use differ between graduate and undergraduate students?",
        "Why do graduate students have more counselling visits than undergraduates?"]},
    {"id": "pathways", "topic": "Movement from medical to counselling", "phrasings": [
        "Do students who see a doctor go on to counselling?",
        "How many students move from the medical clinic to counselling?",
        "How often do medical patients end up in counselling within a month?"]},
    {"id": "services", "topic": "Ratings by service", "phrasings": [
        "Which service has the lowest satisfaction?",
        "How do the services compare on student ratings?",
        "Which of our services do students rate lowest?"]},
]


def _norm(v):
    s = str(v).strip().lower()
    return {"all": "", "none": "", "all services": ""}.get(s, s)


def signature(trace, with_args=True):
    """The analyses behind an answer, as a sorted tuple (helpers and failed steps left out)."""
    out = set()
    for step in trace or []:
        if not isinstance(step, (list, tuple)) or len(step) != 3:
            continue
        name, args, res = step
        if name in HELPERS or (isinstance(res, dict) and "error" in res):
            continue
        key = tuple(sorted((k, _norm(v)) for k, v in (args or {}).items() if k in KEY_ARGS and _norm(v))) if with_args else ()
        out.add((name, key))
    return tuple(sorted(out))


def figures_in(answer):
    """The figures an answer reports: numbers with two or more digits, or a decimal point; years
    and a few round constants are left out."""
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(answer or ""))
    return {n for n in _NUM.findall(text)
            if (len(n) >= 2 or "." in n) and n not in _IGNORE and not re.fullmatch(r"20[12]\d", n)}


def compare(runs):
    """runs = [{"question", "answer", "trace", "ungrounded"}, ...] for ONE question asked several ways."""
    tools = [signature(r["trace"], with_args=False) for r in runs]
    full = [signature(r["trace"]) for r in runs]
    figs = [figures_in(r["answer"]) for r in runs]
    shared = set.intersection(*figs) if figs else set()
    smallest = min((len(f) for f in figs), default=0)
    return {
        "phrasings": len(runs),
        "same_analysis": len(set(tools)) == 1 and bool(tools[0]),
        "same_settings": len(set(full)) == 1 and bool(full[0]),
        "tools": [sorted({t for t, _ in sig}) for sig in tools],
        "shared_figures": sorted(shared, key=lambda x: -float(x)),
        "share_of_figures_in_common": round(len(shared) / smallest, 2) if smallest else 0.0,
        "untraceable": sorted({str(n) for r in runs for n in (r.get("ungrounded") or [])}),
        "answered": sum(bool(sig) for sig in tools),
    }


def run(ask, questions=None, pause=0.0, progress=None):
    """Ask every phrasing in a fresh conversation. `ask(question)` returns the assistant's output dict."""
    results = []
    for q in questions or QUESTIONS:
        runs = []
        for text in q["phrasings"]:
            start = time.time()
            try:
                out = ask(text)
            except Exception as e:                      # one failed request should not end the check
                out = {"answer": f"(request failed: {e})", "trace": [], "ungrounded": []}
            runs.append({"question": text, "answer": out.get("answer", ""), "trace": out.get("trace", []),
                         "ungrounded": out.get("ungrounded", []), "seconds": round(time.time() - start, 1)})
            if progress:
                progress(q, text, runs[-1])
            if pause:
                time.sleep(pause)
        results.append({**q, "runs": runs, "result": compare(runs)})
    return results


def summary(results):
    n = len(results)
    answers = [r for q in results for r in q["runs"]]
    return {
        "questions": n, "answers": len(answers),
        "same_analysis": sum(q["result"]["same_analysis"] for q in results),
        "same_settings": sum(q["result"]["same_settings"] for q in results),
        "with_untraceable_figures": sum(bool(r.get("ungrounded")) for r in answers),
        "median_seconds": sorted(r.get("seconds", 0) for r in answers)[len(answers) // 2] if answers else 0,
    }


def report(results, title="Consistency check", when=""):
    """A Markdown report: a summary, a table, then every question with its phrasings."""
    s = summary(results)
    lines = [f"# {title}", "",
             f"{when} Each question was asked {results[0]['result']['phrasings'] if results else 0} ways, each in a new "
             "conversation, on the same data.".strip(), "",
             f"- **{s['same_analysis']} of {s['questions']}** questions led to the same analysis however they were worded.",
             f"- **{s['same_settings']} of {s['questions']}** also used the same service, measure and group.",
             f"- **{s['answers'] - s['with_untraceable_figures']} of {s['answers']}** answers contained only figures "
             "that trace to a calculation.",
             f"- Typical time to answer: {s['median_seconds']:g} seconds.", "",
             "| Topic | Same analysis | Same settings | Figures every phrasing reported | Untraceable figures |",
             "|---|---|---|---|---|"]
    for q in results:
        r = q["result"]
        lines.append(f"| {q['topic']} | {'Yes' if r['same_analysis'] else '**No**'} | {'Yes' if r['same_settings'] else '**No**'} | "
                     f"{', '.join(r['shared_figures'][:6]) or 'none'} | {', '.join(r['untraceable']) or 'none'} |")
    lines += ["", "## Each question", ""]
    for q in results:
        lines += [f"### {q['topic']}", ""]
        for run_, tools in zip(q["runs"], q["result"]["tools"]):
            lines += [f"**“{run_['question']}”** ({run_.get('seconds', 0):g}s; analyses: {', '.join(tools) or 'none'})", "",
                      "> " + str(run_["answer"]).strip().replace("\n", "\n> "), ""]
    lines += ["## How to read this", "",
              "- *Same analysis* means the same calculations ran. Because the calculations are code, the same "
              "calculation always returns the same figures; the wording around them can differ.",
              "- *Figures every phrasing reported* are the numbers all the answers have in common. An answer may "
              "choose to mention more or fewer figures from the same result.",
              "- A **No** is worth reading: it usually means one wording was understood as a different question."]
    return "\n".join(lines) + "\n"

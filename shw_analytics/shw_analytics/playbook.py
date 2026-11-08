"""Loads playbook.yaml and finds the use case that fits a question.

insight_playbook(question, topic_id)  the tool the model calls for broad or deeper questions:
                                      returns the matching use cases with steps, how to read
                                      results, pitfalls and wording.
starter_questions()                   example questions for the app, from ready use cases.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).with_name("playbook.yaml")
REQUIRED = ("id", "title", "status", "questions", "keywords", "why", "steps", "read", "avoid", "say")
STATUS_ORDER = {"ready": 0, "partial": 1, "needs_data": 2}
_STOP = set("""a an and are as at be been but by can could did do does for from get got had has have
how i if in into is it its me more most my no not of on or our should so than that the their them
then there these they this those to us was we were what when where which who why will with would
you your students student service services""".split())


@lru_cache(maxsize=1)
def load():
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    cases = data["use_cases"]
    for c in cases:
        missing = [k for k in REQUIRED if k not in c]
        if missing:
            raise ValueError(f"Playbook entry {c.get('id')} is missing: {', '.join(missing)}")
    return cases


def _tokens(text):
    return {w for w in re.findall(r"[a-z][a-z\-]+", str(text).lower()) if w not in _STOP}


def match(question: str, k: int = 3):
    q = _tokens(question)
    if not q:
        return []
    scored = []
    for c in load():
        kw = set().union(*(_tokens(x) for x in c["keywords"]))
        phrases = sum(1 for x in c["keywords"] if " " in x and x.lower() in question.lower())
        text = _tokens(c["title"]) | set().union(*(_tokens(x) for x in c["questions"]))
        score = 2 * len(q & kw) + len(q & text) + 3 * phrases
        if score:
            scored.append((score, -STATUS_ORDER.get(c["status"], 3), c))
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [c for _, _, c in scored[:k]]


def insight_playbook(question: str = None, topic_id: str = None):
    cases = load()
    if topic_id:
        hit = [c for c in cases if c["id"] == topic_id]
        if not hit:
            return {"error": f"Unknown topic_id '{topic_id}'. Use one of: "
                             + ", ".join(c["id"] for c in cases)}
        return {"use_cases": hit}
    hits = match(question or "")
    others = [{"topic_id": c["id"], "title": c["title"], "status": c["status"]}
              for c in cases if c not in hits]
    if not hits:
        return {"use_cases": [], "topics": others,
                "note": "No close match. Pick a topic_id from topics, or answer with the usual tools."}
    return {"use_cases": hits, "other_topics": others,
            "how_to_use": "Follow the first use case's steps unless another fits the question "
                          "better. Respect 'avoid' and word the answer as in 'say'. If status is "
                          "needs_data, explain what is missing instead of guessing."}


def starter_questions(limit: int = 8):
    """(title, example question) pairs from ready use cases, for suggestions in the app."""
    return [(c["title"], c["questions"][0]) for c in load() if c["status"] == "ready"][:limit]

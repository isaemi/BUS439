"""Keeps statistics jargon out of answers written for staff.

plain_stats(text) removes p-values and q-values and turns "95% CI" into "range". The
numbers that matter to a reader (the change and its range) stay; the test statistics are
still available under "How this answer was built".
"""
import re

_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)(?:[eE]-?\d+)?"
_PQ = rf"\b[pq](?:[- ]value)?\s*(?:=|<|≤|of)\s*{_NUM}"


_META = re.compile(
    r"(?i)\b(good catch|you(?:'|’)re right|my apologies|i apologi[sz]e|thanks? for (?:flagging|catching|pointing)|"
    r"i re-?checked|on re-?checking|re-?checked my answer|corrected version|here(?:'|’)s the corrected|"
    r"i (?:wrote|included|cited) (?:that|which) (?:don(?:'|’)t|do not|doesn(?:'|’)t) appear)")


def strip_meta(text: str) -> str:
    """Drop an opening paragraph in which the model talks about correcting itself. Staff never
    see the request it is replying to, so that paragraph only confuses."""
    parts = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    if len(parts) == 2 and _META.search(parts[0]) and len(parts[0]) < 600:
        return parts[1].lstrip()
    return text


def plain_stats(text: str) -> str:
    if not text:
        return text
    s = strip_meta(text)
    s = re.sub(rf"\(\s*{_PQ}\s*\)", "", s)                       # "(p = 0.0012)"
    s = re.sub(rf"\s*[,;]\s*{_PQ}", "", s)                       # ", p=0.0012" inside a bracket or list
    s = re.sub(rf"{_PQ}\s*[,;]?\s*", "", s)                      # anything left
    s = re.sub(r"\b95\s?%\s*(?:CI|C\.I\.|confidence interval)s?\b", "range", s, flags=re.I)
    s = re.sub(r"\bconfidence intervals?\b", "range", s, flags=re.I)
    s = re.sub(r"\bCI\b(?=\s*[:\d−+-])", "range", s)
    s = re.sub(r"\(\s*\)", "", s)                                # brackets emptied by the above
    s = re.sub(r"\(\s*[,;]\s*", "(", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"[ \t]+([.,;:)])", r"\1", s)
    s = re.sub(r"[ \t]+$", "", s, flags=re.M)
    return s

"""Fill dictionary value labels from two exports of the SAME responses:
one with "Use numeric values" and one with "Use choice text".

    python -m shw_analytics.labels numeric.csv choicetext.csv > labels.yaml
Paste the output under `value_labels:` in dictionary.yaml after checking it.
"""
import sys
from collections import defaultdict

import yaml

from .ingest import load_dictionary, read_qualtrics


def build(numeric_csv, text_csv):
    num, _ = read_qualtrics(numeric_csv)
    txt, _ = read_qualtrics(text_csv)
    d = load_dictionary()
    questions = list(d["demographics"]) + list(d["followups"]) + [d["service_question"]]
    merged = num.merge(txt, on="ResponseId", suffixes=("", "__t"))
    labels = defaultdict(dict)
    for q in questions:
        if q not in num or f"{q}__t" not in merged:
            continue
        for code, label in merged[[q, f"{q}__t"]].dropna().itertuples(index=False):
            for c, l in zip(str(code).split(","), str(label).split(",")):
                labels[q][c.strip()] = l.strip()
    return dict(labels)


if __name__ == "__main__":
    print(yaml.safe_dump({"value_labels": build(sys.argv[1], sys.argv[2])}, allow_unicode=True, sort_keys=False))

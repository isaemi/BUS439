"""Tests for shw_analytics/figures.py, driven by a sample of real tool results
(tests/fixtures/sample_trace.json, made from the synthetic data).

Run:  python tests/test_figures.py      (or: pytest tests/test_figures.py)
"""
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import figures  # noqa: E402

TRACE = [tuple(step) for step in json.loads((Path(__file__).parent / "fixtures" / "sample_trace.json").read_text())]
TECH = re.compile(r"\b(what_changed|chr_summary|chr_trend|noshow_patterns|care_journeys|priority_items|"
                  r"explain_difference|equity_flags|run_sql|[a-z]+_[a-z]+_pct|no_show_rate|academic_level|"
                  r"service_line|student_key|disability_any)\b")


def _register(trace=TRACE):
    store = {}
    return figures.register_from_trace(trace, store), store


def _step(name, n=0):
    return copy.deepcopy([s for s in TRACE if s[0] == name][n])


def _only(step):
    ids, store = _register([step])
    return [store[i] for i in ids]


def test_every_kind_of_analysis_gets_a_graphic():
    ids, store = _register()
    drawn = {store[i].source for i in ids}
    expected = set(figures.BUILDERS) - {"term_briefing"}      # the briefing is covered in test_briefing.py
    assert drawn == expected, expected - drawn
    for i in ids:
        g = store[i]
        assert g.fig is not None and g.title and g.scope and g.caption
        assert g.fig.layout.title.text is None          # the page shows the title, so it can wrap


def test_wording_never_shows_tool_or_column_names():
    ids, store = _register()
    bad = [(store[i].source, m.group(0)) for i in ids
           for m in [TECH.search(" ".join([store[i].title, store[i].scope, store[i].caption]))] if m]
    assert not bad, bad


def test_tested_findings_come_first_and_descriptive_last():
    ids, store = _register()
    weights = [store[i].weight for i in ids]
    assert weights == sorted(weights, reverse=True) and weights[0] == 3 and weights[-1] == 1
    first, more = figures.split_for_display(ids, store)
    assert len(first) == figures.SHOWN_FIRST and all(store[i].weight == 3 for i in first)
    assert len(more) == len(ids) - figures.SHOWN_FIRST


def test_same_question_twice_does_not_duplicate_graphics():
    store = {}
    a = figures.register_from_trace(TRACE, store)
    b = figures.register_from_trace(TRACE, store)
    assert a == b and len(store) == len(a)


def test_volume_chart_is_shared_by_calls_about_the_same_dates():
    ids, store = _register()            # the sample has two what_changed calls for the same service and dates
    assert sum(store[i].title.startswith("Students seen") for i in ids) == 1


def test_titles_follow_the_direction_and_the_verdict():
    step = _step("what_changed")
    assert any(g.title.startswith("Students seen rose") for g in _only(step))
    step[2]["volume_change"].update(after=180, change_pct=-18.2, ci_pct=[-32.0, -1.5], verdict="decrease")
    assert any(g.title.startswith("Students seen fell 18.2%") for g in _only(step))
    step[2]["volume_change"].update(verdict="no clear change")
    assert any(g.title == "No clear change in the number of students seen" for g in _only(step))
    step[2]["verdict"] = "offsetting"
    assert any("looks flat overall, but it changed within groups" in g.title for g in _only(step))


def test_nothing_flagged_means_no_chart():
    ns, eq = _step("noshow_patterns"), _step("equity_flags")
    ns[2]["flags"], eq[2]["flags"] = [], []
    assert _only(ns) == [] and _only(eq) == []


def test_small_groups_are_left_out_of_the_before_and_after_chart():
    step = _step("what_changed", 1)                       # split by faculty: many small groups
    chart = next(g for g in _only(step) if "by faculty" in g.title or "changed most clearly" in g.title)
    small = [g["group"] for g in step[2]["groups"] if min(g["n_before"], g["n_after"]) < 30]
    labels = " ".join(str(y) for t in chart.fig.data for y in (t.y if t.y is not None else []))
    assert small and not any(f"{s}  <" in labels for s in small)
    assert "under 30 are not shown" in chart.scope


def test_terms_are_drawn_in_the_order_they_happen_and_the_unfinished_one_is_marked():
    g = _only(_step("chr_trend", 1))[0]
    assert [t.x[0] for t in g.fig.data] == ["2025 Fall", "2026 Winter", "2026 Summer", "2026 Fall"]
    assert g.fig.data[-1].marker.pattern.shape == "/" and "not finished" in g.title
    assert "so far" in g.fig.data[-1].text[0]


def test_months_stay_in_calendar_order():
    g = _only(_step("chr_trend", 0))[0]
    months = list(g.fig.layout.xaxis.categoryarray)
    assert months[0] == "Sep 2025" and months[-1] == "Oct 2026" and len(months) == 14


def _texts(g):
    return [a.text for a in g.fig.layout.annotations]


def test_key_driver_chart_groups_questions_and_shows_the_link():
    step = _step("priority_items")
    g = _only(step)[0]
    assert "link to the overall rating" in g.title and g.weight == 3
    labels = _texts(g)
    assert "<b>PRIORITIES</b>" in labels and "<b>STRENGTHS</b>" in labels and "<b>OTHER QUESTIONS</b>" in labels
    assert labels.index("<b>PRIORITIES</b>") < labels.index("<b>STRENGTHS</b>") < labels.index("<b>OTHER QUESTIONS</b>")
    strong = [t for t in labels if t.startswith("<b>Strong</b>")]
    assert strong and re.search(r">0\.\d\d</span>$", strong[0])          # the word, then the correlation
    link, agree = g.fig.data[0], g.fig.data[1]                              # both shown as bars, one per question
    assert link.type == "bar" and agree.type == "bar" and len(link.y) == len(step[2]["items"])
    assert g.caption.startswith("Priority: “")
    # descriptive, professional wording: no instructions to staff, no statistics vocabulary
    words = " ".join([g.title, g.scope, g.caption] + labels).lower()
    assert not re.search(r"fix first|keep it up|doing fine|what to do|can't tell|matters a|start with", words), words
    assert not re.search(r"re-sampl|correlation|\bn=|p-value", words), words


def test_priorities_sit_above_strengths_and_strongest_link_comes_first_within_a_group():
    step = _step("priority_items")
    g = _only(step)[0]
    link = g.fig.data[0]
    by_y = sorted(zip(link.y, link.x, link.marker.color))                   # top of the chart first
    colors = [c for _, _, c in by_y]
    assert colors[0] == figures.HILITE                                      # a priority is at the top
    first_blue = colors.index(figures.theme.FOCUS)
    assert figures.HILITE not in colors[first_blue:]                        # no priority below a strength
    orange = [v for _, v, c in by_y if c == figures.HILITE]
    assert orange == sorted(orange, reverse=True)


def test_key_driver_chart_is_honest_when_nothing_is_clear():
    step = _step("priority_items")
    step[2].update(priorities=[], verdict="no_stable_priority")
    for it in step[2]["items"]:                       # every range includes zero, as with random answers
        it.update(link_to_outcome=0.05, link_ci=[-0.12, 0.2])
    g = _only(step)[0]
    assert g.title == "Counselling: no question is clearly more linked to the overall rating than the others"
    labels = _texts(g)
    assert sum(t.startswith("Unclear") for t in labels) == len(step[2]["items"]) and g.weight == 2
    assert not any("PRIORITIES" in t or "STRENGTHS" in t or "OTHER QUESTIONS" in t for t in labels)   # one group: no headings
    assert "a low score alone does not indicate a priority" in g.caption


def test_link_strength_words_follow_its_size_and_certainty():
    m = lambda link, lo: figures._matters({"link_to_outcome": link, "link_ci": [lo, link + 0.1]})  # noqa: E731
    assert m(0.7, 0.6) == (3, "Strong") and m(0.4, 0.25) == (2, "Moderate")
    assert m(0.2, 0.05) == (1, "Weak") and m(0.4, -0.05) == (0, "Unclear")


def test_a_broken_or_failed_step_never_breaks_the_answer():
    trace = [("what_changed", {}, {"volume_change": {}}), ("chr_summary", {"dimension": "mode"}, {"error": "x"}),
             ("unknown_tool", {}, {}), "not a step", ("care_journeys", {}, None)] + [_step("service_pathways")]
    ids, store = _register(trace)
    assert len(ids) == 1 and store[ids[0]].source == "service_pathways"


def test_report_uses_captions_and_plain_explanations():
    ids, store = _register()
    rid = figures.build_report(ids[:3], store)
    html = store[rid].html
    assert store[rid].kind == "report" and all(store[i].caption in html for i in ids[:3])
    assert "How this was worked out:" in html and "Source: " not in html
    assert all(f"<h2>{store[i].title}</h2>" in html for i in ids[:3])
    assert figures.split_for_display([rid] + ids, store)[0][0] == rid
    try:
        figures.build_report([999], store)
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError")


def test_report_requests_are_still_recognised():
    assert figures.parse_report_request("make a report from graphics 1, 3 and 8") == [1, 3, 8]
    assert figures.parse_report_request("report from graphic 2 to 4") == [2, 3, 4]
    assert figures.parse_report_request("How many students came?") is None


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", type(e).__name__, str(e)[:300])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)

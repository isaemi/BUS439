"""Turns tool results into numbered graphics with captions, and graphics into reports.

Each kind of analysis gets the chart that shows its finding, not a generic bar chart:
  what_changed      the change split into "who came" and "within groups", with ranges;
                    volume in the same dates; before/after by group
  noshow_patterns   gap vs other appointments with ranges, flagged groups highlighted
  care_journeys     what happens after a first visit; single visits by group
  priority_items    importance vs score quadrant ("fix first")
  service_pathways  share moving to another service, with ranges
  ...and the descriptive tools (trends, breakdowns, survey questions, group comparisons).

Rules every chart follows, so staff can read one without a legend lesson:
  * the title states the finding; the line under it says what, where and when
  * one highlight colour marks what was tested and stands out; context is pale
  * values are written on the marks; ranges are drawn wherever a result was tested
  * sorted, long labels wrapped, at most 12 rows; unfinished periods are drawn striped
Colours and font come from theme.py, the same as the dashboards.
"""
import json
import math
import re
import textwrap
from dataclasses import dataclass, field

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import theme
from .provenance import LABELS

FONT = theme.FONT.replace('"', "'")    # same family as the dashboards; single quotes keep image export working
HILITE = theme.ALERT_FG["amber"]     # tested and stands out
QUIET = theme.QUIET                  # context, or within noise
MAX_ROWS = 12
SHOWN_FIRST = 3                      # the app shows this many; the rest go under "More graphics"


@dataclass
class Graphic:
    id: int
    kind: str                 # "chart" or "report"
    title: str                # the finding, in plain words
    caption: str              # key numbers and how far to trust them
    fig: object = None        # plotly figure (charts)
    html: str = ""            # full page (reports)
    source: str = ""          # tool name (never shown to staff)
    args: dict = field(default_factory=dict)
    parents: list = field(default_factory=list)   # for reports: which graphics fed it
    scope: str = ""           # what, where, when
    weight: int = 1           # 3 = tested finding, 2 = trend or pattern, 1 = descriptive


@dataclass
class _Spec:
    title: str
    scope: str
    caption: str
    fig: object
    weight: int = 1
    key: tuple = None         # set to merge the same chart coming from different calls


# ---- small helpers ------------------------------------------------------------------

def _label(x):
    return LABELS.get(str(x), str(x).replace("_", " "))


def _wrap(text, width=36):
    return "<br>".join(textwrap.wrap(str(text), width)) or str(text)


def _short(text, n=46):
    text = str(text)
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _signed(v, unit=""):
    return f"{'+' if v > 0 else '−' if v < 0 else ''}{abs(v):g}{unit}"


def _day(ts):
    return f"{ts.day} {ts:%b}"


def _period(text, sep=" to "):
    """'2025-09-01 to 2025-10-31' -> '1 Sep to 31 Oct 2025'."""
    try:
        a, b = (pd.Timestamp(x) for x in str(text).split(sep))
        return (f"{_day(a)} to {_day(b)} {b.year}" if a.year == b.year
                else f"{_day(a)} {a.year} to {_day(b)} {b.year}")
    except Exception:
        return str(text)


def _periods(p):
    """Two matching windows -> ('1 Sep to 31 Oct, 2025 vs 2026', '2025', '2026')."""
    try:
        a0, a1 = (pd.Timestamp(x) for x in p["before"].split(" to "))
        b0, b1 = (pd.Timestamp(x) for x in p["after"].split(" to "))
        if (a0.month, a0.day, a1.month, a1.day) == (b0.month, b0.day, b1.month, b1.day):
            return f"{_day(b0)} to {_day(b1)}, {a1.year} vs {b1.year}", str(a1.year), str(b1.year)
    except Exception:
        pass
    return f"{_period(p['before'])} vs {_period(p['after'])}", "Before", "After"


def _service(d, key="service_line"):
    s = (d or {}).get(key) or (d or {}).get("service")
    names = {"wellness_adv": "Wellness Advising", "info_desk": "Wellness Info Desk", "all": "All services"}
    return names.get(str(s).lower(), str(s).title()) if s else "All services"


def _synthetic(r):
    return " Synthetic test data, not real." if "synthetic" in str(r.get("data_source", "")).lower() else ""


def _real(ci):
    """A range that does not include zero."""
    return ci[0] > 0 or ci[1] < 0


def _n(label, n):
    return f"{label}  <span style='color:{theme.MUTED}'>n={n:,}</span>"


_GROUP_IS_A_NAME = {"academic_level", "intl_or_domestic", "first_generation", "indigenous", "racialized", "faculty"}
_WHO = {("disability_any", "yes"): "Students reporting a disability",
        ("disability_any", "no"): "Students reporting no disability"}


def _who(dimension, group):
    """A group named the way staff would say it."""
    key = (str(dimension), str(group).lower())
    if key in _WHO:
        return _WHO[key]
    if str(dimension) in _GROUP_IS_A_NAME and not str(group).isdigit():
        return f"{group} students"                      # "Graduate students", "International students"
    return f"{_label(dimension).capitalize()}: {group}"


def _season_order(term):
    """'2026 Winter' -> (2026, 1): terms in the order they happen, not alphabetically."""
    m = re.search(r"(\d{4})", str(term))
    season = next((k for k in ("winter", "spring", "summer", "fall", "autumn") if k in str(term).lower()), "")
    return (int(m.group(1)) if m else 0, {"winter": 1, "spring": 1, "summer": 2, "fall": 3, "autumn": 3}.get(season, 0))


def _fig(height=360, legend=False, left=20):
    fig = go.Figure()
    fig.update_layout(
        font=dict(family=FONT, size=13, color=theme.TEXT), height=height,
        paper_bgcolor="white", plot_bgcolor="white", showlegend=legend,
        margin=dict(t=84, r=40, b=48, l=left), bargap=0.32,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title=None),
        hoverlabel=dict(bgcolor="white", font=dict(family=FONT, color=theme.TEXT)))
    fig.update_xaxes(gridcolor=theme.GRID, zeroline=False, linecolor=theme.BORDER, ticks="",
                     tickfont=dict(color=theme.MUTED), title_font=dict(color=theme.MUTED, size=12))
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False, linecolor=theme.BORDER, ticks="",
                     tickfont=dict(color=theme.TEXT), title_font=dict(color=theme.MUTED, size=12),
                     automargin=True)
    return fig


def _titled(fig, title, scope):
    """Leave the title and scope to the page, where they wrap to any width; the figure
    only needs room at the top for its legend and column headings."""
    legend_room = 30 if fig.layout.showlegend else 0
    fig.update_layout(margin_t=38 + legend_room, height=max(180, fig.layout.height - 46 + legend_room))
    return fig


def _rows_height(n, base=190, per=38):
    return base + per * max(n, 2)


def _zero(fig, x=0, text=None):
    fig.add_vline(x=x, line=dict(color=theme.MUTED, width=1, dash="dot"), annotation_text=text,
                  annotation_position="top", annotation_font=dict(color=theme.MUTED, size=12))


def _range_dot(fig, y, est, ci, hot, note, hover):
    """A dot with its range, and a note written to the right of the range."""
    color = HILITE if hot else QUIET
    fig.add_trace(go.Scatter(
        x=[est], y=[y], mode="markers", marker=dict(size=14, color=color),
        error_x=dict(type="data", symmetric=False, array=[ci[1] - est], arrayminus=[est - ci[0]],
                     color=color, thickness=2.5, width=7),
        hovertemplate=hover + "<extra></extra>"))
    fig.add_annotation(x=ci[1], y=y, xanchor="left", xshift=12, showarrow=False, text=note,
                       font=dict(color=theme.TEXT if hot else theme.MUTED))


# ---- tested insights ----------------------------------------------------------------

_MEASURE = {"no_show_rate": ("the no-show rate", " pts", "%"),
            "visits_per_student": ("visits per student", "", ""),
            "days_in_care": ("days in care", " days", " days")}


def _what_changed(args, r):
    if "volume_change" not in r:
        return []
    out = []
    span, yr_a, yr_b = _periods(r["periods"])
    svc = _service(r)
    name, unit, suffix = _MEASURE.get(r.get("metric"), (str(r.get("measure", "the measure")).lower(), "", ""))
    o, mix, within = r["overall"], r["mix_effect"], r["within_group_effect"]

    # 1) The split, with ranges: which parts are distinguishable from zero.
    rows = [("Overall change", o["change"], o["ci"]),
            ("From who came<br>(mix of groups)", mix["estimate"], mix["ci"]),
            ("From change<br>within groups", within["estimate"], within["ci"])]
    fig = _fig(height=320, left=150)
    lim = max(abs(v) for _, e, ci in rows for v in (e, *ci)) or 1
    for label, est, ci in rows:
        real = _real(ci)
        _range_dot(fig, label, est, ci, real,
                   f"<b>{_signed(est, unit)}</b>  {'a real change' if real else 'within noise'}",
                   f"{label.replace('<br>', ' ')}: {_signed(est, unit)}<br>range {ci[0]:g} to {ci[1]:g}")
    _zero(fig, text="no change")
    fig.update_xaxes(range=[-lim * 1.3, lim * 2.7],
                     title_text="Change, with its plausible range")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    verdict = r.get("verdict")
    verb = "rose" if o["change"] > 0 else "fell"
    title = {"who_came": f"{name.capitalize()} {verb}, mostly because of who came",
             "within_groups": f"{name.capitalize()} {verb} within groups, not because of who came",
             "both": f"{name.capitalize()} {verb}: both who came and change within groups played a part",
             "offsetting": f"{name.capitalize()} looks flat overall, but it changed within groups",
             }.get(verdict, f"No clear change in {name}")
    scope = f"{svc}, {span}: {o['before']:g}{suffix} then {o['after']:g}{suffix}. Split by {_label(r['dimension'])}"
    cap = (f"{name.capitalize()} went from {o['before']:g}{suffix} to {o['after']:g}{suffix} "
           f"(change {_signed(o['change'], unit)}, range {o['ci'][0]:g} to {o['ci'][1]:g}; n = {r['n']['before']:,} and "
           f"{r['n']['after']:,} {r['n']['unit']}). A part counts as a real change only when its whole range is on "
           f"one side of zero. This shows how the change splits, not why it happened.{_synthetic(r)}")
    out.append(_Spec(title, scope, cap, _titled(fig, title, scope), 3 if verdict != "no_clear_change" else 2))

    # 2) Volume in the same dates (one per service and period, however many calls asked).
    v, w = r["volume_change"], r["volume_in_same_windows"]
    cats = ["Students seen", "Appointments booked", "Appointments attended"]
    fig = _fig(height=340, legend=True)
    for per, label, color in [("before", yr_a, theme.OVERALL), ("after", yr_b, theme.FOCUS)]:
        vals = [w[per][k] for k in ("students_seen", "booked", "attended")]
        fig.add_trace(go.Bar(x=cats, y=vals, name=label, marker_color=color, text=[f"{x:,}" for x in vals],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}, " + label + ": %{y:,}<extra></extra>"))
    fig.update_layout(barmode="group", bargroupgap=0.08)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, rangemode="tozero")
    lo, hi = v.get("ci_pct", [None, None])
    changed = v.get("verdict") in ("increase", "decrease")
    vt = (f"Students seen {'rose' if v['change_pct'] > 0 else 'fell'} {abs(v['change_pct']):g}% "
          f"(range {_signed(lo, '%')} to {_signed(hi, '%')})" if changed
          else "No clear change in the number of students seen")
    vscope = f"{svc}, {span}. Students seen is tested; appointment counts are shown for context"
    vcap = (f"Students seen: {v['before']:,} then {v['after']:,} ({_signed(v.get('change_pct', 0), '%')}"
            + (f", range {_signed(lo, '%')} to {_signed(hi, '%')}" if lo is not None else "")
            + f"), over the same dates in both years. Booked {w['before']['booked']:,} then {w['after']['booked']:,}; "
            f"attended {w['before']['attended']:,} then {w['after']['attended']:,} (not tested).{_synthetic(r)}")
    out.append(_Spec(vt, vscope, vcap, _titled(fig, vt, vscope), 3 if changed else 1,
                     key=("volume", svc, json.dumps(r["periods"], sort_keys=True))))

    # 3) Before and after for each group.
    every = [g for g in r.get("groups", []) if g.get("before") is not None and g.get("after") is not None]
    groups = [g for g in every if min(g["n_before"], g["n_after"]) >= 30]
    hidden = len(every) - len(groups)
    plain = name[4:] if name.startswith("the ") else name
    if len(groups) >= 2:
        groups = sorted(groups, key=lambda g: -g["n_after"])[:MAX_ROWS]
        groups.sort(key=lambda g: -g["after"])
        fig = _fig(height=_rows_height(len(groups)), legend=True, left=170)
        ylab = [_n(_short(g["group"], 30), g["n_after"]) for g in groups]
        moved = [g for g in groups if _real(g.get("within_group_ci", [0, 0]))]
        for g, y in zip(groups, ylab):
            hot = g in moved
            fig.add_trace(go.Scatter(x=[g["before"], g["after"]], y=[y, y], mode="lines", showlegend=False,
                                     line=dict(color=HILITE if hot else theme.OVERALL, width=4), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[g["before"] for g in groups], y=ylab, mode="markers", name=yr_a,
                                 marker=dict(size=12, color="white", line=dict(color=theme.FOCUS, width=2)),
                                 hovertemplate="%{x:g}" + suffix + "<extra>" + yr_a + "</extra>"))
        fig.add_trace(go.Scatter(x=[g["after"] for g in groups], y=ylab, mode="markers", name=yr_b,
                                 marker=dict(size=12, color=theme.FOCUS),
                                 hovertemplate="%{x:g}" + suffix + "<extra>" + yr_b + "</extra>"))
        fig.update_xaxes(title_text=plain.capitalize() + (f" ({suffix.strip()})" if suffix.strip() else ""))
        fig.update_yaxes(showgrid=False, autorange="reversed")
        dim = _label(r["dimension"])
        gt = (f"{plain.capitalize()} changed most clearly for {' and '.join(str(g['group']) for g in moved[:2])}" if moved
              else f"{plain.capitalize()} by {dim}: no group clearly changed")
        gs = (f"{svc}, {span}. Hollow dot = {yr_a}, filled = {yr_b}. An orange line marks a change beyond noise"
              + (f". {hidden} groups under 30 are not shown" if hidden else ""))
        gc = (f"{name.capitalize()} for each {dim} over the same dates in both years. Only groups joined by an "
              f"orange line changed by more than chance; the rest are descriptive.{_synthetic(r)}")
        out.append(_Spec(gt, gs, gc, _titled(fig, gt, gs), 2 if moved else 1))
    return out


def _noshow(args, r):
    flags = r.get("flags", [])
    if not flags:
        return []
    factors = list(dict.fromkeys(f["factor"] for f in flags))
    flagged = {(f["factor"], f["group"]) for f in flags}
    rows = [g for g in r.get("all_groups", flags) if g["factor"] in factors]
    rows.sort(key=lambda g: (factors.index(g["factor"]), -g["gap_vs_others_pts"]))
    rows = rows[:MAX_ROWS]
    fig = _fig(height=_rows_height(len(rows)), left=210)
    lim = max(abs(v) for g in rows for v in g["ci_pts"]) or 1
    for g in rows:
        hot = (g["factor"], g["group"]) in flagged
        y = f"<span style='color:{theme.MUTED}'>{g['factor']}:</span> {_short(g['group'], 28)}"
        _range_dot(fig, y, g["gap_vs_others_pts"], g["ci_pts"], hot, f"<b>{g['no_show_pct']:g}%</b> missed",
                   f"{g['factor']}: {g['group']}<br>{g['no_show_pct']:g}% missed, {g['appointments']:,} appointments, "
                   f"{g['students']:,} students")
    _zero(fig, text="same as other appointments")
    fig.update_xaxes(range=[-lim * 1.25, lim * 1.9],
                     title_text="Points more or fewer missed than other appointments")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    top = max(flags, key=lambda f: f["gap_vs_others_pts"])
    who = (f"{top['group']} appointments" if top["factor"] in ("Appointment format", "Weekday", "Service", "Visit type", "Time of day")
           else f"{top['factor']} “{top['group']}” appointments")
    title = f"{who} are missed {abs(top['gap_vs_others_pts']):g} points more often than others"
    scope = (f"{_service(r)}, {_period(r.get('period', ''))}. Overall {r['overall_no_show_pct']:g}% of "
             f"{r['appointments']:,} appointments were missed")
    cap = (f"{top['group']}: {top['no_show_pct']:g}% missed, {abs(top['gap_vs_others_pts']):g} points more than other "
           f"appointments (range {top['ci_pts'][0]:g} to {top['ci_pts'][1]:g}; {top['appointments']:,} appointments, "
           f"{top['students']:,} students). Orange = stands out after {r.get('tests_run', 'many')} comparisons, with each "
           f"student counted once. A pattern, not a cause.{_synthetic(r)}")
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3)]


def _care(args, r):
    if "single_visit_pct" not in r:
        return []
    out = []
    back_key = next((k for k in r if k.startswith("returned_within_")), "returned_within_30_days_pct")
    back, single = r.get(back_key, 0), r["single_visit_pct"]
    days = re.search(r"\d+", back_key).group(0)
    later = max(0.0, round(100 - back - single, 1))
    fig = _fig(height=230, legend=True)
    for label, val, color, tc in [(f"came back within {days} days", back, theme.FOCUS, "white"),
                                  ("came back later", later, theme.RAMP[4], theme.NAVY),
                                  ("single visit", single, theme.OVERALL, theme.NAVY)]:
        fig.add_trace(go.Bar(x=[val], y=[""], orientation="h", marker_color=color, marker_line=dict(color="white", width=2),
                             name=label.capitalize(),
                             text=[f"<b>{val:g}%</b><br>{label}" if val >= 14 else (f"<b>{val:g}%</b>" if val >= 4 else "")],
                             textposition="inside", insidetextanchor="middle", textfont=dict(color=tc, size=13),
                             hovertemplate=f"{val:g}% {label}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.15, margin_b=24, legend_traceorder="normal")
    fig.update_xaxes(visible=False, range=[0, 100])
    fig.update_yaxes(visible=False)
    m = r.get("missed_then_came_back", {})
    again = next((v for k, v in m.items() if k.startswith("attended_within")), None)
    title = f"{back:g}% of students came back within {days} days of a first visit"
    scope = (f"{_service(r)}: {r['students']:,} students starting a new episode of care, "
             f"{_period(r.get('episodes_start_between', ''), ' and ')}, each followed for 60 days")
    cap = (f"Of {r['students']:,} students, {back:g}% returned within {days} days and {single:g}% had a single visit"
           + (f"; the median gap before a second visit was {r['median_days_to_second_visit']:g} days" if r.get("median_days_to_second_visit") is not None else "")
           + "."
           + (f" Of {m['students_with_a_missed_appointment']:,} students who missed an appointment, {again:g}% attended "
              "again within 60 days." if again is not None and m.get("students_with_a_missed_appointment") else "")
           + f" A single visit can be planned single-session care, so it is not a dropout rate.{_synthetic(r)}")
    out.append(_Spec(title, scope, cap, _titled(fig, title, scope), 2))

    groups = r.get("groups") or []
    if len(groups) >= 2:
        flags = [f for f in r.get("flags", []) if f["measure"] == "single_visit"]
        flagged = {f["group"] for f in flags}
        groups = sorted(groups, key=lambda g: -g["students"])[:MAX_ROWS]
        groups.sort(key=lambda g: -g["single_visit_pct"])
        fig = _fig(height=_rows_height(len(groups)), left=170)
        ylab = [_n(_short(g["group"], 30), g["students"]) for g in groups]
        fig.add_trace(go.Bar(x=[g["single_visit_pct"] for g in groups], y=ylab, orientation="h",
                             marker_color=[HILITE if g["group"] in flagged else theme.OVERALL for g in groups],
                             text=[f"{g['single_visit_pct']:g}%" for g in groups], textposition="outside",
                             cliponaxis=False, hovertemplate="%{x:g}% single visit<extra></extra>"))
        _zero(fig, single, f"everyone {single:g}%")
        fig.update_xaxes(title_text="Students with a single visit", ticksuffix="%",
                         range=[0, max(g["single_visit_pct"] for g in groups) * 1.25 + 2])
        fig.update_yaxes(showgrid=False, autorange="reversed")
        dim = _label(r.get("dimension", "group"))
        if flags:
            f = max(flags, key=lambda x: abs(x["gap_vs_others_pts"]))
            gt = (f"{f['group']} students were {abs(f['gap_vs_others_pts']):g} points "
                  f"{'more' if f['gap_vs_others_pts'] > 0 else 'less'} likely to have a single visit")
        else:
            gt = f"Single visits by {dim}: no group clearly differs"
        gs = f"{_service(r)}, by {dim}. Orange = differs from everyone else after correcting for {r.get('tests_run', 'several')} tests"
        gc = (f"Share of students with a single visit in the 60 days after starting care, by {dim}. Whether a "
              f"difference reflects planned single-session care is a question for the clinical team.{_synthetic(r)}")
        out.append(_Spec(gt, gs, gc, _titled(fig, gt, gs), 3 if flags else 1))
    return out


def _pathways(args, r):
    rows = sorted((p for p in r.get("pathways", []) if p["pct"] > 0), key=lambda p: -p["pct"])[:8]
    if not rows:
        return []
    fig = _fig(height=_rows_height(len(rows)), left=230)
    ylab = [f"{p['from']} → {p['to']}" for p in rows]
    fig.add_trace(go.Bar(
        x=[p["pct"] for p in rows], y=ylab, orientation="h",
        marker_color=[theme.FOCUS if i == 0 else theme.RAMP[4] for i in range(len(rows))],
        error_x=dict(type="data", symmetric=False, array=[p["ci_pct"][1] - p["pct"] for p in rows],
                     arrayminus=[p["pct"] - p["ci_pct"][0] for p in rows], color=theme.NAVY, thickness=1.5, width=5),
        hovertemplate="%{y}: %{x:g}%<extra></extra>"))
    for p, y in zip(rows, ylab):
        fig.add_annotation(x=p["ci_pct"][1], y=y, xanchor="left", xshift=10, showarrow=False,
                           text=f"<b>{p['pct']:g}%</b>  <span style='color:{theme.MUTED}'>{p['went_on_within_days']:,} of {p['students_starting']:,}</span>")
    fig.update_xaxes(title_text="Share of students, with 95% range",
                     ticksuffix="%", range=[0, max(p["ci_pct"][1] for p in rows) * 1.5])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    top = rows[0]
    title = f"{top['pct']:g}% of students starting in {top['from']} went on to {top['to']} within {r['within_days']} days"
    scope = f"All services, {_period(r.get('period', ''))}. Each student's first new episode of care in a service"
    cap = (f"{top['from']} to {top['to']}: {top['went_on_within_days']:,} of {top['students_starting']:,} students "
           f"({top['pct']:g}%, range {top['ci_pct'][0]:g}% to {top['ci_pct'][1]:g}%). This shows movement between "
           f"services, not whether a referral was made.{_synthetic(r)}")
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3)]


def _matters(item):
    """How much a question matters, in words. Level 0 = cannot tell (its range includes zero);
    1 to 3 follow the usual reading of a correlation: under 0.3 small, 0.3 to 0.5 moderate, 0.5+ strong."""
    link, lo = item["link_to_outcome"], item["link_ci"][0]
    if lo <= 0:
        return 0, "Unclear"
    if link >= 0.5:
        return 3, "Strong"
    if link >= 0.3:
        return 2, "Moderate"
    return 1, "Weak"


_ACTION_ORDER = ["Priority", "Possible priority", "Strength", "Lower priority", "Scores well", "Unclear"]


def _action(item, level, is_priority, typical_agree):
    """How the row is classified, and its colour. Descriptive, not an instruction: staff decide
    what to do with it."""
    if is_priority:
        return "Priority", HILITE
    if level == 0:
        return "Unclear", theme.MUTED
    low = item["agree_pct"] < typical_agree
    if level >= 2:
        return ("Possible priority", theme.TEXT) if low else ("Strength", theme.FOCUS)
    return ("Lower priority" if low else "Scores well"), theme.MUTED


_GROUPS = [("Priority", "PRIORITIES"), ("Possible priority", "POSSIBLE PRIORITIES, NOT YET RELIABLE"),
           ("Strength", "STRENGTHS"), ("Other", "OTHER QUESTIONS")]


def _priorities(args, r):
    """Survey questions grouped under Priorities, Strengths and Other questions. Each row shows
    how closely the question is linked to the overall rating (a bar, the word and the correlation)
    and how many students agree. Descriptive: it classifies, and leaves the decision to staff."""
    items = r.get("items", [])
    if len(items) < 3:
        return []
    pri = set(r.get("priorities", []))
    typical = float(pd.Series([i["agree_pct"] for i in items]).median())
    rows = []
    for i in items:
        level, words = _matters(i)
        act, color = _action(i, level, i["item"] in pri, typical)
        group = act if act in ("Priority", "Possible priority", "Strength") else "Other"
        rows.append({**i, "level": level, "words": words, "act": act, "group": group,
                     "color": {"Priority": HILITE, "Strength": theme.FOCUS}.get(group, theme.QUIET)})
    rows = sorted(rows, key=lambda x: -x["link_to_outcome"])[:MAX_ROWS]

    # A slot for each heading, then its questions, strongest link first.
    present = [(key, head) for key, head in _GROUPS if any(x["group"] == key for x in rows)]
    use_headings = len(present) > 1
    y, heads, placed = 0.0, [], []
    for key, head in present:
        if use_headings:
            heads.append((y, head))
            y += 1
        for x in (x for x in rows if x["group"] == key):
            placed.append({**x, "y": y})
            y += 1
        y += 0.35

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.55, 0.45], horizontal_spacing=0.05)
    fig.update_layout(_fig(height=int(150 + 46 * y), left=250).layout)
    fig.add_trace(go.Bar(
        x=[max(x["link_to_outcome"], 0.02) for x in placed], y=[x["y"] for x in placed], orientation="h", width=0.62,
        marker=dict(color=[x["color"] if x["level"] else theme.OVERALL for x in placed],
                    pattern=dict(shape=["/" if x["level"] == 0 else "" for x in placed], fgcolor="white", size=6)),
        hovertemplate=[f"{x['item']}<br>Link to the overall rating: {x['link_to_outcome']:g} "
                       f"(range {x['link_ci'][0]:g} to {x['link_ci'][1]:g}), {x['students']:,} students<extra></extra>"
                       for x in placed]), row=1, col=1)
    for x in placed:
        fig.add_annotation(
            x=max(x["link_to_outcome"], 0.02), y=x["y"], xanchor="left", xshift=8, showarrow=False, row=1, col=1,
            text=(f"<b>{x['words']}</b>" if x["level"] else x["words"])
                 + f"  <span style='color:{theme.MUTED}'>{x['link_to_outcome']:.2f}</span>",
            font=dict(color=theme.TEXT if x["level"] else theme.MUTED))
    fig.add_trace(go.Bar(
        x=[x["agree_pct"] for x in placed], y=[x["y"] for x in placed], orientation="h", width=0.62,
        marker_color=[x["color"] if x["group"] in ("Priority", "Strength") else theme.OVERALL for x in placed],
        hovertemplate="%{x:g}% of students agree<extra></extra>"), row=1, col=2)
    fig.add_vline(x=typical, line=dict(color=theme.MUTED, width=1, dash="dot"), row=1, col=2,
                  annotation_text=f"typical {typical:.0f}%", annotation_position="bottom",
                  annotation_font=dict(color=theme.MUTED, size=11))
    for x in placed:
        fig.add_annotation(x=x["agree_pct"], y=x["y"], xanchor="left", xshift=6, showarrow=False, row=1, col=2,
                           text=f"<b>{x['agree_pct']:g}%</b>", bgcolor="white", borderpad=1, font=dict(color=theme.TEXT))
    for hy, head in heads:                                   # short headings with a rule across the chart
        fig.add_annotation(x=0, y=hy, xref="paper", yref="y", xanchor="left", yanchor="middle", showarrow=False,
                           text=f"<b>{head}</b>", font=dict(size=12, color=theme.NAVY))
        fig.add_shape(type="line", xref="paper", yref="y", x0=0, x1=1, y0=hy + 0.45, y1=hy + 0.45,
                      line=dict(color=theme.BORDER, width=1))
    fig.update_yaxes(tickvals=[x["y"] for x in placed], showgrid=False, range=[y - 0.2, -0.7],
                     ticktext=[(f"<b>{_wrap(x['item'], 36)}</b>" if x["group"] == "Priority" else _wrap(x["item"], 36))
                               for x in placed])
    fig.update_xaxes(visible=False, range=[min(0.0, min(x["link_to_outcome"] for x in placed)), 1.3], row=1, col=1)
    fig.update_xaxes(visible=False, range=[0, max(max(x["agree_pct"] for x in placed) * 1.3, 40)], row=1, col=2)
    for col, text in ((1, "Link to<br>overall rating"), (2, "% agree")):
        dom = fig.layout["xaxis" if col == 1 else "xaxis2"].domain
        fig.add_annotation(x=dom[0], y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom", yshift=8,
                           align="left", showarrow=False, text=f"<b>{text}</b>", font=dict(color=theme.TEXT, size=13))

    svc = _service(r, "service")
    outcome = r.get("outcome", "their overall rating")
    fix = [x["item"] for x in placed if x["group"] == "Priority"]
    keep = [x["item"] for x in placed if x["group"] == "Strength"]
    look = [x["item"] for x in placed if x["group"] == "Possible priority"]
    top = next((x for x in placed if x["group"] == "Priority"), None)
    if top:
        title = f"“{_short(top['item'], 80)}”: {top['words'].lower()} link to the overall rating, {top['agree_pct']:g}% agree"
    elif look:
        title = f"{svc}: no question meets the bar for a priority"
    else:
        title = f"{svc}: no question is clearly more linked to the overall rating than the others"
    scope = (f"{svc} survey, {r.get('respondents', 0):,} students. Link = how closely answers to a question move with "
             f"{outcome}, from 0 (none) to 1; 0.5 or more is strong, 0.3 to 0.5 moderate. Priority = strong or moderate "
             "link with agreement below the typical question; strength = the same link with agreement at or above it")
    if fix:
        cap = (f"Priority: {'; '.join('“' + f + '”' for f in fix)}."
               + (f" Strengths: {'; '.join('“' + k + '”' for k in keep[:3])}." if keep else ""))
    elif look:
        cap = (f"Possible priority, not yet reliable: {'; '.join('“' + k + '”' for k in look[:2])}. "
               "More responses would settle it.")
    else:
        cap = ("The links are too similar or too uncertain to rank the questions, so a low score alone does not "
               "indicate a priority. More responses would help.")
    cap += " These are associations; they do not show that changing one will change the other."
    if any(x.get("access_item") for x in placed):
        cap += (" Links for booking and waiting are likely understated, because students who could not get an "
                "appointment were not surveyed.")
    fig = _titled(fig, title, scope)
    fig.update_layout(margin_t=fig.layout.margin.t + 22, height=fig.layout.height + 22)   # two-line column heading
    return [_Spec(title, scope, cap, fig, 3 if fix else 2)]


def _explain(args, r):
    d = r.get("decomposition", {})
    rows = []
    if "students" in d:
        rows.append(("Students seen", d["students"]["group"], d["students"]["comparison"], d["students"]["ratio"], None))
    for m in r.get("measures", []):
        rows.append((m["measure"], m["group"], m["comparison"], m.get("ratio"), bool(m.get("significant"))))
    rows = [x for x in rows if x[3]]
    if len(rows) < 2:
        return []
    grp = str(r.get("group", "")).split(" = ")[-1]
    other = r.get("comparison", "others")
    fig = _fig(height=_rows_height(len(rows), per=42), left=230)
    ratios = [x[3] for x in rows]
    for label, a, b, ratio, sig in rows:
        y = _wrap(label, 30)
        color = theme.FOCUS if sig or sig is None else QUIET
        fig.add_trace(go.Scatter(x=[1, ratio], y=[y, y], mode="lines", line=dict(color=color, width=5), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[ratio], y=[y], mode="markers", marker=dict(size=15, color=color),
                                 hovertemplate=f"{label}: {a:,g} vs {b:,g}<extra></extra>"))
        note = ", within noise" if sig is False else ""
        fig.add_annotation(x=math.log10(ratio), y=y, xanchor="left" if ratio >= 1 else "right",
                           xshift=14 if ratio >= 1 else -14, showarrow=False,
                           text=f"<b>{ratio:g}×</b>  <span style='color:{theme.MUTED}'>{a:,g} vs {b:,g}{note}</span>")
    _zero(fig, 1, "1× = the same")
    lo, hi = min(min(ratios) / 4.5, 0.4), max(max(ratios) * 6, 4)
    fig.update_xaxes(type="log", range=[math.log10(lo), math.log10(hi)], ticksuffix="×",
                     tickvals=[v for v in (0.1, 0.25, 0.5, 1, 2, 4, 8) if lo <= v <= hi],
                     title_text=f"{grp} compared with {other}")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    by = {x[0]: x[3] for x in rows}
    st, vi = by.get("Students seen"), by.get("Visits per student")
    if st and vi:
        title = (f"{grp} students: {st:g}× as many students as {other}, "
                 f"{'but ' if (st - 1) * (vi - 1) < 0 else ''}{vi:g}× the visits each")
    else:
        title = f"How {grp} students compare with {other}"
    scope = f"{_service(r)}, {r.get('term', 'all terms')}. Blue = a tested difference; grey = within noise"
    cap = (d.get("summary") or " ".join(f.rstrip(".") + "." for f in r.get("findings", [])[:2]) or title)
    cap += f" Differences are described, not explained.{_synthetic(r)}"
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3)]


def _gap_rows(fig, rows, ylab, everyone, group, hot, name):
    """Dumbbells: hollow = everyone, filled = the group, joined by a line (orange when tested)."""
    for x, y in zip(rows, ylab):
        fig.add_trace(go.Scatter(x=[x[everyone], x[group]], y=[y, y], mode="lines", showlegend=False,
                                 line=dict(color=HILITE if hot(x) else theme.OVERALL, width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x[everyone] for x in rows], y=ylab, mode="markers", name="Everyone else" if name is None else "All respondents",
                             marker=dict(size=12, color="white", line=dict(color=theme.MUTED, width=2)),
                             hovertemplate="%{x:g}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x[group] for x in rows], y=ylab, mode="markers", name=name or "This group",
                             marker=dict(size=13, color=[HILITE if hot(x) else theme.FOCUS for x in rows]),
                             hovertemplate="%{x:g}%<extra></extra>"))
    for x, y in zip(rows, ylab):
        left = x[group] < x[everyone]
        fig.add_annotation(x=x[group], y=y, xanchor="right" if left else "left", xshift=-12 if left else 12,
                           showarrow=False, text=f"<b>{_signed(x['gap_pts'])} pts</b>",
                           font=dict(color=theme.TEXT if hot(x) else theme.MUTED))
    vals = [v for x in rows for v in (x[everyone], x[group])]
    span = max(vals) - min(vals) or 10
    fig.update_xaxes(title_text="Average agreement", ticksuffix="%", range=[min(vals) - span * 0.6 - 4, max(vals) + span * 0.6 + 4])
    fig.update_yaxes(showgrid=False, autorange="reversed")


def _equity(args, r):
    flags = [f for f in r.get("flags", []) if f["gap_pts"] < 0] or r.get("flags", [])
    if not flags:
        return []
    flags = sorted(flags, key=lambda f: f["gap_pts"])[:MAX_ROWS]
    fig = _fig(height=_rows_height(len(flags), per=52), legend=True, left=260)
    ylab = [f"{_short(_who(f['dimension'], f['group']), 34)}<br>"
            f"<span style='color:{theme.MUTED}'>{_service(f, 'service')}, n={f['n_group']:,}</span>" for f in flags]
    _gap_rows(fig, flags, ylab, "pct_agree_rest", "pct_agree_group", lambda x: True, None)
    top = flags[0]
    who = _who(top["dimension"], top["group"])
    title = (f"{who} rate {_service(top, 'service')} {abs(top['gap_pts']):g} points "
             f"{'lower' if top['gap_pts'] < 0 else 'higher'} than everyone else")
    scope = f"Survey ratings. {len(flags)} gap(s) stood out of {r.get('comparisons_run', 'many')} comparisons checked"
    cap = (f"Groups whose ratings differ from everyone else's after correcting for {r.get('comparisons_run', 'many')} "
           f"comparisons (gaps of 10 points or more). {r.get('groups_too_small_to_compare', 0)} groups were too small to test.")
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3)]


def _group_vs(args, r):
    rows = sorted(r.get("by_service", []), key=lambda x: x["gap_pts"])
    if not rows:
        return []
    dim, _, val = str(r.get("group", "This group")).partition(" = ")
    name = _who(dim, val) if val else dim.replace("_", " ")
    fig = _fig(height=_rows_height(len(rows), per=44), legend=True, left=190)
    ylab = [_n(_service(x, "service"), x["n_group"]) for x in rows]
    _gap_rows(fig, rows, ylab, "pct_agree_everyone", "pct_agree_group", lambda x: x["significant"], name)
    sig = [x for x in rows if x["significant"]]
    title = (f"{name} rate {_service(sig[0], 'service')} {abs(sig[0]['gap_pts']):g} points "
             f"{'lower' if sig[0]['gap_pts'] < 0 else 'higher'} than all respondents" if sig
             else f"{name}: no clear difference from all respondents")
    scope = "Survey ratings by service. An orange line marks a tested difference; a pale line is within noise"
    cap = (f"{name} compared with all respondents, service by service, corrected for the number of services checked. "
           "Average agreement = share of questions rated 4 or 5, each student counted once.")
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3 if sig else 1)]


# ---- trends and descriptive breakdowns ----------------------------------------------

def _trend(args, r):
    df = pd.DataFrame(r.get("rows", []))
    if len(df) < 2:
        return []
    svc, freq = _service(args), args.get("freq", "month")
    prev = next((c for c in df.columns if c.startswith("same_")), None)
    if freq == "term" or prev is None:
        part = {a.replace("PARTIAL TERM:", "").strip(): (b, c) for a, b, c in re.findall(
            r"([\w :]+?) has data only from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})", r.get("warning", ""))}
        fig = _fig(height=350)
        df = df.iloc[sorted(range(len(df)), key=lambda k: _season_order(df["period"].iloc[k]))]
        for _, row in df.iterrows():
            unfinished = row["period"] in part
            note = f"<br><span style='color:{theme.MUTED}'>so far, to {_day(pd.Timestamp(part[row['period']][1]))}</span>" if unfinished else ""
            fig.add_trace(go.Bar(
                x=[row["period"]], y=[row["appointments"]], text=[f"<b>{row['appointments']:,}</b>{note}"],
                marker=dict(color=theme.OVERALL if unfinished else theme.FOCUS, pattern_shape="/" if unfinished else "",
                            pattern_fgcolor="white", line=dict(color=theme.FOCUS, width=1.5 if unfinished else 0)),
                textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:,} appointments<extra></extra>"))
        fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, df["appointments"].max() * 1.22])
        fig.update_xaxes(showgrid=False)
        name = next(iter(part), None)
        title = f"{svc} appointments by {freq}" + (f": {name} is not finished yet" if name else "")
        scope = f"Attended appointments per {freq}" + (". The striped bar covers only part of the term, so it cannot be compared with the others" if name else "")
        cap = (f"Attended appointments per {freq}." + (f" {name} has data only up to {_day(pd.Timestamp(part[name][1]))}, so it looks "
               "lower only because less time is covered. To compare with last year, use the same dates in both years." if name else "")
               + _synthetic(r))
        return [_Spec(title, scope, cap, _titled(fig, title, scope), 2)]
    x = [pd.Timestamp(str(p) + "-01").strftime("%b %Y") if re.fullmatch(r"\d{4}-\d{2}", str(p)) else str(p) for p in df["period"]]
    fig = _fig(height=380, legend=True)
    has_prev = df[prev].notna()
    if has_prev.any():
        fig.add_trace(go.Scatter(x=[a for a, ok in zip(x, has_prev) if ok], y=df.loc[has_prev, prev], mode="lines+markers",
                                 name=f"Same {freq} a year earlier", line=dict(color=QUIET, width=2, dash="dot"),
                                 marker=dict(size=6, color=QUIET), hovertemplate="%{x}, a year earlier: %{y:,}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=df["appointments"], mode="lines+markers", name="Appointments",
                             line=dict(color=theme.FOCUS, width=3), marker=dict(size=7, color=theme.FOCUS),
                             hovertemplate="%{x}: %{y:,}<extra></extra>"))
    hot = df[df["notable_change"].fillna(False).astype(bool)] if "notable_change" in df else df.iloc[0:0]
    if len(hot):
        fig.add_trace(go.Scatter(x=[x[i] for i in hot.index], y=hot["appointments"], mode="markers+text",
                                 name="Notably different from a year earlier",
                                 marker=dict(size=15, color=HILITE, line=dict(color="white", width=2)),
                                 text=[f"<b>{_signed(round(v))}%</b>" for v in hot["change_pct"]], textposition="top center",
                                 textfont=dict(color=theme.TEXT), cliponaxis=False, hoverinfo="skip"))
    fig.update_yaxes(rangemode="tozero", title_text="Attended appointments")
    fig.update_xaxes(showgrid=False, categoryorder="array", categoryarray=x)
    if len(hot):
        ups = int((hot["change_pct"] > 0).sum())
        title = (f"{svc} appointments were notably {'higher' if ups * 2 >= len(hot) else 'lower'} than a year earlier "
                 f"in {len(hot)} of {int(has_prev.sum())} comparable {freq}s")
    else:
        title = f"{svc} appointments by {freq}" + (": no notable change from a year earlier" if has_prev.any() else "")
    scope = f"Attended appointments per {freq}. Orange = at least 10% different from the same {freq} a year earlier and unlikely to be chance"
    cap = f"Attended appointments per {freq}, with the same {freq} a year earlier where the data goes back that far.{_synthetic(r)}"
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3 if len(hot) else 2)]


def _summary(args, r):
    dim = args.get("dimension")
    df = pd.DataFrame(r.get("rows", []))
    if not dim or dim not in df or len(df) < 2:
        return []
    more = max(0, len(df) - MAX_ROWS)
    df = df.sort_values("n_students", ascending=False).head(MAX_ROWS)
    panels = [p for p in [("n_students", "Students seen", "{:,.0f}"), ("visits_per_student", "Visits per student", "{:g}"),
                          ("no_show_rate_pct", "Appointments missed", "{:g}%")] if p[0] in df]
    fig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True, horizontal_spacing=0.06)
    fig.update_layout(_fig(height=_rows_height(len(df), base=175), left=160).layout)
    y = [_short(v, 36) for v in df[dim].astype(str)]
    for k, (col, label, fmt) in enumerate(panels, 1):
        top = df[col].max()
        fig.add_trace(go.Bar(x=df[col], y=y, orientation="h",
                             marker_color=[theme.FOCUS if v == top else theme.RAMP[4] for v in df[col]],
                             text=[fmt.format(v) for v in df[col]], textposition="outside", cliponaxis=False,
                             hovertemplate="%{y}: %{text}<extra>" + label + "</extra>"), row=1, col=k)
        fig.update_xaxes(visible=False, range=[0, top * 1.4], row=1, col=k)
        dom = fig.layout[f"xaxis{'' if k == 1 else k}"].domain
        fig.add_annotation(x=dom[0], y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom", yshift=4,
                           showarrow=False, text=label, font=dict(color=theme.MUTED, size=13))
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True)
    svc = _service(args)
    title = f"{svc} by {_label(dim)}: who comes, how often, and how many appointments are missed"
    scope = "Appointment records, all dates. Descriptive: the differences are not tested" + (f". {more} smaller groups not shown" if more else "")
    cap = (f"Students seen, attended visits per student and share of booked appointments missed, by {_label(dim)}. "
           f"Not tested, so treat differences as a starting point.{_synthetic(r)}")
    if r.get("warning"):
        cap += " One term is not finished, so its totals are lower only because less time is covered."
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 1)]


def _hbar(labels, values, title, scope, cap, axis, ref=None, ref_label=None, weight=1, faded=None, left=260):
    fig = _fig(height=_rows_height(len(labels), per=40), left=left)
    faded = faded or [False] * len(labels)
    fig.add_trace(go.Bar(x=values, y=labels, orientation="h",
                         marker_color=[theme.OVERALL if f else theme.FOCUS for f in faded],
                         text=[f"{v:g}%" for v in values], textposition="outside", cliponaxis=False,
                         hovertemplate="%{x:g}%<extra></extra>"))
    if ref is not None:
        _zero(fig, ref, ref_label)
    fig.update_xaxes(title_text=axis, range=[0, max(max(values), ref or 0) * 1.2], ticksuffix="%")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return _Spec(title, scope, cap, _titled(fig, title, scope), weight)


def _items(args, r):
    df = pd.DataFrame(r.get("rows", []))
    if len(df) < 2:
        return []
    df = df.sort_values("pct_agree").head(MAX_ROWS)
    svc = _service(args, "service")
    avg = round(float(df["pct_agree"].mean()), 1)
    return [_hbar([_wrap(t, 44) for t in df["item_text"]], list(df["pct_agree"]),
                  f"{svc}: how many students agree with each question, lowest first",
                  f"Survey, {int(df['n'].min()):,} to {int(df['n'].max()):,} students per question. The lowest score is "
                  "not necessarily what matters most to students",
                  f"Share of students who agreed (rated 4 or 5) with each {svc.lower()} question. To see which of these "
                  "matter most to overall satisfaction, ask what the service should focus on.",
                  "Students who agree", ref=avg, ref_label=f"average {avg:g}%", left=300)]


def _compare(args, r):
    dim = args.get("dimension")
    df = pd.DataFrame(r.get("rows", []))
    if not dim or dim not in df or len(df) < 2:
        return []
    df = df.sort_values("n", ascending=False).head(MAX_ROWS).sort_values("pct_agree")
    ev = (r.get("everyone") or {}).get("pct_agree")
    svc = _service(args, "service") if args.get("service") else "All services"
    codes = df[dim].astype(str).str.fullmatch(r"\d+").all()
    labels = [_n(f"{'Code ' if codes else ''}{_short(v, 30)}", n) for v, n in zip(df[dim].astype(str), df["n"])]
    return [_hbar(labels, list(df["pct_agree"]), f"{svc}: average agreement by {_label(dim)}",
                  "Survey. Descriptive, not tested. Pale bars are groups of fewer than 30 students"
                  + (". Answer labels are not loaded yet, so groups show as codes" if codes else ""),
                  f"Average agreement by {_label(dim)} (share of questions rated 4 or 5, each student counted once). "
                  "Differences are not tested here.",
                  "Average agreement", ref=ev, ref_label=f"everyone {ev:g}%" if ev is not None else None,
                  faded=list(df["n"] < 30), left=200)]


def _services(args, r):
    df = pd.DataFrame(r.get("rows", []))
    if len(df) < 2:
        return []
    df = df.sort_values("pct_agree")
    labels = [_n(_service(row, "service"), row["n"]) for row in df.to_dict("records")]
    return [_hbar(labels, list(df["pct_agree"]), "Average agreement by service",
                  "Survey. Pale bars are services with fewer than 30 respondents",
                  "Average agreement = share of questions rated 4 or 5, each student counted once. Not a tested comparison.",
                  "Average agreement", faded=list(df["n"] < 30), left=220)]


def _wait(args, r):
    df = pd.DataFrame(r.get("rows", []))
    if len(df) < 2:
        return []
    c = r.get("correlation", {})
    real = c.get("p_value", 1) < 0.05
    svc = _service(args, "service")
    fig = _fig(height=350)
    x = [f"Wait category {w}" if str(w).isdigit() else str(w) for w in df["wait"]]
    fig.add_trace(go.Bar(x=x, y=df["pct_agree"], marker_color=theme.FOCUS if real else theme.OVERALL,
                         text=[f"<b>{v:g}%</b><br><span style='color:{theme.MUTED}'>n={n:,}</span>" for v, n in zip(df["pct_agree"], df["n"])],
                         textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:g}%<extra></extra>"))
    fig.update_yaxes(showticklabels=False, showgrid=False, range=[0, df["pct_agree"].max() * 1.32])
    fig.update_xaxes(showgrid=False, title_text="Shortest wait on the left (assumed)")
    title = (f"{svc}: longer waits go with {'lower' if c.get('spearman_rho', 0) < 0 else 'higher'} ratings" if real
             else f"{svc}: no clear link between wait and ratings")
    scope = f"Survey, {c.get('n', 0):,} students. Average agreement by reported wait before the appointment"
    cap = (f"Rank correlation {c.get('spearman_rho', 0):g} (p = {c.get('p_value', 1):g}, n = {c.get('n', 0):,}). "
           + ("A relationship, not proof that waiting lowers ratings. " if real
              else "With this many students a small link could go undetected. ")
           + "Only students who got an appointment were surveyed.")
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 2 if real else 1)]


def _representation(args, r):
    df = pd.DataFrame(r.get("rows", []))
    dim = args.get("dimension", "faculty")
    if dim not in df or len(df) < 2:
        return []
    df = df.sort_values("pct_of_visitors", ascending=False).head(MAX_ROWS).sort_values("representation_ratio")
    fig = _fig(height=_rows_height(len(df)), legend=True, left=230)
    y = [_short(v, 34) for v in df[dim].astype(str)]
    under = list(df["representation_ratio"] < 0.8)
    for (_, row), yy, u in zip(df.iterrows(), y, under):
        fig.add_trace(go.Scatter(x=[row["pct_of_visitors"], row["pct_of_survey"]], y=[yy, yy], mode="lines", showlegend=False,
                                 line=dict(color=HILITE if u else theme.OVERALL, width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df["pct_of_visitors"], y=y, mode="markers", name="Share of students who visit",
                             marker=dict(size=12, color="white", line=dict(color=theme.FOCUS, width=2)),
                             hovertemplate="%{x:g}% of visitors<extra></extra>"))
    fig.add_trace(go.Scatter(x=df["pct_of_survey"], y=y, mode="markers", name="Share of survey respondents",
                             marker=dict(size=12, color=theme.FOCUS), hovertemplate="%{x:g}% of respondents<extra></extra>"))
    fig.update_xaxes(title_text="Share", ticksuffix="%", rangemode="tozero")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    low = df.iloc[0]
    title = (f"{low[dim]} students are under-represented in the survey" if low["representation_ratio"] < 0.8
             else f"Survey respondents broadly match who visits, by {_label(dim)}")
    scope = f"Survey respondents compared with students in the appointment records, by {_label(dim)}. Orange = under-represented in the survey"
    cap = (f"{low[dim]}: {low['pct_of_visitors']:g}% of students who visit, but {low['pct_of_survey']:g}% of survey "
           f"respondents. Survey averages lean towards the groups that answer more.{_synthetic(r)}")
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 2)]


_TONE = {"positive": ("Positive", theme.UP), "negative": ("Negative", theme.DOWN),
         "mixed": ("Mixed", HILITE), "neutral": ("Neutral", QUIET)}


def comment_charts(overview):
    """(themes chart, tone chart) for the Comments tab, from the comment overview."""
    themes = sorted(overview.get("themes", []), key=lambda r: -r["n"])[:MAX_ROWS]
    fig = _fig(height=_rows_height(len(themes), base=70, per=34), left=200)
    fig.add_trace(go.Bar(
        x=[r["n"] for r in themes], y=[str(r["theme"]).replace("_", " ").capitalize() for r in themes],
        orientation="h", marker_color=[theme.FOCUS if i < 3 else theme.RAMP[4] for i in range(len(themes))],
        text=[f"{r['n']:,}" for r in themes], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} comments<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, max([r["n"] for r in themes] or [1]) * 1.2])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(margin_t=10, margin_b=10)

    tone = sorted(overview.get("sentiment", []), key=lambda r: -r["n"])
    total = sum(r["n"] for r in tone) or 1
    fig2 = _fig(height=_rows_height(len(tone), base=70, per=34), left=90)
    fig2.add_trace(go.Bar(
        x=[r["n"] for r in tone], y=[_TONE.get(r["sentiment"], (str(r["sentiment"]).capitalize(), QUIET))[0] for r in tone],
        orientation="h", marker_color=[_TONE.get(r["sentiment"], ("", QUIET))[1] for r in tone],
        text=[f"{r['n']:,}  ({100 * r['n'] / total:.0f}%)" for r in tone], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:,} comments<extra></extra>"))
    fig2.update_xaxes(visible=False, range=[0, max([r["n"] for r in tone] or [1]) * 1.45])
    fig2.update_yaxes(showgrid=False, autorange="reversed")
    fig2.update_layout(margin_t=10, margin_b=10)
    return fig, fig2


def _survey_changed(args, r):
    """Before and after, one row per group or question that moved (or per service when none did)."""
    span, before, after = _periods(r["periods"])
    rows = []
    for g in r.get("groups_that_changed", []):
        rows.append({**g, "label": _who(g["dimension"], g["group"]), "sub": f"{_service(g, 'service')}, all questions",
                     "n": (g["respondents_before"], g["respondents_after"]), "hot": True})
    for q in r.get("questions_that_changed", []):
        rows.append({**q, "label": _wrap(q["item"], 38), "sub": _service(q, "service"),
                     "n": (q["students_before"], q["students_after"]), "hot": True})
    flagged = bool(rows)
    if not rows:
        rows = [{**s, "label": _service(s, "service"), "sub": "all questions",
                 "n": (s["respondents_before"], s["respondents_after"]), "hot": s["verdict"] != "no clear change"}
                for s in r.get("services", [])]
    if not rows:
        return []
    rows = rows[:MAX_ROWS]
    fig = _fig(height=_rows_height(len(rows), base=170, per=58), legend=True, left=250)
    ylab = [f"{x['label']}<br><span style='color:{theme.MUTED}'>{x['sub']}, n={x['n'][0]:,} then {x['n'][1]:,}</span>"
            for x in rows]
    for x, y in zip(rows, ylab):
        fig.add_trace(go.Scatter(x=[x["pct_agree_before"], x["pct_agree_after"]], y=[y, y], mode="lines", showlegend=False,
                                 line=dict(color=HILITE if x["hot"] else theme.OVERALL, width=4), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[x["pct_agree_before"] for x in rows], y=ylab, mode="markers", name=before,
                             marker=dict(size=12, color="white", line=dict(color=theme.MUTED, width=2)),
                             hovertemplate="%{x:g}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[x["pct_agree_after"] for x in rows], y=ylab, mode="markers", name=after,
                             marker=dict(size=13, color=[HILITE if x["hot"] else theme.FOCUS for x in rows]),
                             hovertemplate="%{x:g}%<extra></extra>"))
    for x, y in zip(rows, ylab):
        left = x["pct_agree_after"] < x["pct_agree_before"]
        fig.add_annotation(x=x["pct_agree_after"], y=y, xanchor="right" if left else "left", xshift=-12 if left else 12,
                           showarrow=False, text=f"<b>{_signed(x['change_pts'])} pts</b>",
                           font=dict(color=theme.TEXT if x["hot"] else theme.MUTED))
    vals = [v for x in rows for v in (x["pct_agree_before"], x["pct_agree_after"])]
    width = max(vals) - min(vals) or 10
    fig.update_xaxes(title_text="Students agreeing", ticksuffix="%", range=[min(vals) - width * 0.6 - 4, max(vals) + width * 0.6 + 4])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    if flagged:
        top = rows[0]
        down = top["change_pts"] < 0
        if "dimension" in top:
            title = (f"{top['label']} rate {_service(top, 'service').lower()} {abs(top['change_pts']):g} points "
                     f"{'lower' if down else 'higher'} than a year ago" if before != "Before" else
                     f"{top['label']} rate {_service(top, 'service').lower()} {abs(top['change_pts']):g} points "
                     f"{'lower' if down else 'higher'} than before")
        else:
            title = f"“{_short(top['item'], 70)}”: {abs(top['change_pts']):g} points {'lower' if down else 'higher'} than {before}"
        cap = (f"Groups and questions whose ratings moved between the two windows, after correcting for "
               f"{r.get('tests_run', 'many')} checks. Orange = tested and stands out.")
    else:
        title = "No clear change in survey ratings"
        cap = "No service, question or group moved by more than chance would explain."
        if r.get("smallest_group_change_detectable_pts"):
            cap += (f" With this many responses, a group's rating would need to move about "
                    f"{r['smallest_group_change_detectable_pts']:g} points to be detected.")
    cap += " The survey is anonymous, so the two windows hold different students. A pattern, not a cause."
    scope = f"Survey ratings, {span}. Share of students rating 4 or 5; each student counted once"
    return [_Spec(title, scope, cap, _titled(fig, title, scope), 3 if flagged else 2)]


def _briefing(args, r):
    """Charts for the term briefing's findings, titled with each finding's headline."""
    from . import briefing
    out = []
    for f in briefing.build()["findings"]:
        spec = briefing.graphic(f)
        if spec:
            out.append(_Spec(f["headline"], spec.scope, spec.caption, spec.fig, 3, key=("finding", f["id"])))
    return out


BUILDERS = {"what_changed": _what_changed, "noshow_patterns": _noshow, "care_journeys": _care,
            "service_pathways": _pathways, "priority_items": _priorities, "explain_difference": _explain,
            "equity_flags": _equity, "group_vs_everyone": _group_vs, "chr_trend": _trend,
            "chr_summary": _summary, "item_summary": _items, "compare_groups": _compare,
            "service_summary": _services, "wait_time_vs_rating": _wait, "survey_vs_visits": _representation,
            "term_briefing": _briefing, "survey_changed": _survey_changed}


# ---- registry ------------------------------------------------------------------------

_COMMON_NUMBERS = {"30", "60", "100", "95", "2024", "2025", "2026", "2027"}


def _mentions(graphic, answer):
    """How many of a graphic's own figures the answer quotes (so its chart belongs with the answer)."""
    def nums(text):
        text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text))
        return {n for n in re.findall(r"\d+(?:\.\d+)?", text) if len(n) >= 2 and n not in _COMMON_NUMBERS}
    return len(nums(graphic.title + " " + graphic.caption) & nums(answer))


def register_from_trace(trace, store, answer=None):
    """trace = [(tool_name, args, result), ...]. Returns ids of graphics for this answer, most
    useful first: tested findings, then trends and patterns, then descriptive breakdowns. Given
    the answer's text, graphics whose figures the answer quotes come before those it does not.
    The same chart asked for twice gives the same graphic, not a duplicate. A chart that
    fails to draw is skipped rather than breaking the answer."""
    found = []
    for step in trace or []:
        try:
            name, args, res = step
        except (TypeError, ValueError):
            continue
        build = BUILDERS.get(name)
        if not build or not isinstance(res, dict) or "error" in res:
            continue
        try:
            specs = build(args or {}, res)
        except Exception:
            continue
        for i, spec in enumerate(specs):
            key = spec.key or (name, json.dumps(args, sort_keys=True, default=str), i)
            hit = next((g for g in store.values() if g.args.get("_key") == key), None)
            if hit is None:
                gid = max(store, default=0) + 1
                hit = store[gid] = Graphic(gid, "chart", spec.title, spec.caption, spec.fig, source=name,
                                           args={**(args or {}), "_key": key}, scope=spec.scope, weight=spec.weight)
            if all(hit.id != g.id for g in found):
                found.append(hit)
    if answer:                                 # within a weight, what the answer talks about comes first
        found.sort(key=lambda g: (-g.weight, -_mentions(g, answer)))
    else:
        found.sort(key=lambda g: -g.weight)    # stable: keeps the order of the analysis within a weight
    return [g.id for g in found]


def split_for_display(ids, store, first=SHOWN_FIRST):
    """(shown straight away, kept under 'More graphics'). Reports always show."""
    charts = [i for i in ids if store[i].kind == "chart"]
    reports = [i for i in ids if store[i].kind != "chart"]
    return reports + charts[:first], charts[first:]


# ---- reports -------------------------------------------------------------------------

def parse_report_request(text):
    """'make a report from graphic 1 3 8 and 19' -> [1, 3, 8, 19]. None if not a report request."""
    if not re.search(r"\b(report|one-pager|summary)\b", text, re.I) or \
       not re.search(r"\b(graphic|graph|figure|chart)s?\b", text, re.I):
        return None
    ids = []
    for a, b in re.findall(r"(\d+)\s*(?:-|to)\s*(\d+)", text):
        ids += range(int(a), int(b) + 1)
    text = re.sub(r"\d+\s*(?:-|to)\s*\d+", " ", text)
    ids += [int(n) for n in re.findall(r"\d+", text)]
    return list(dict.fromkeys(ids))


def build_report(ids, store, title="Student Health and Wellbeing: findings"):
    from .provenance import evidence_cards
    missing = [i for i in ids if i not in store or store[i].kind != "chart"]
    if missing:
        raise KeyError(f"No chart with number(s): {missing}")
    parts = []
    for n, i in enumerate(ids):
        g = store[i]
        cards = evidence_cards([(g.source, {k: v for k, v in g.args.items() if k != "_key"}, {})])
        how = cards[0].get("calculated") if cards else ""
        parts.append(
            f"<section><p class='num'>Graphic {g.id}</p><h2>{g.title}</h2><p class='scope'>{g.scope}</p>"
            + g.fig.to_html(full_html=False, include_plotlyjs=True if n == 0 else False, config={"displayModeBar": False})
            + f"<p class='cap'>{g.caption}</p>"
            + (f"<p class='how'>How this was worked out: {how}</p>" if how else "") + "</section>")
    css = (f"body{{font-family:{theme.FONT};background:{theme.PAGE};color:{theme.TEXT};margin:0;padding:32px 16px}}"
           f"main{{max-width:940px;margin:auto}}h1{{color:{theme.NAVY};font-size:28px;margin:0 0 4px}}"
           f".lede{{color:{theme.MUTED};margin:0 0 24px}}section{{background:{theme.CARD};border:1px solid {theme.BORDER};"
           "border-radius:10px;padding:18px 22px 14px;margin:0 0 18px;break-inside:avoid}"
           f".num{{color:{theme.MUTED};font-size:13px;margin:0 0 2px}}.cap{{margin:6px 0 4px;line-height:1.5}}"
           f"h2{{color:{theme.NAVY};font-size:19px;line-height:1.3;margin:0 0 4px}}"
           f".scope{{color:{theme.MUTED};font-size:14px;line-height:1.45;margin:0 0 6px}}"
           f".how{{color:{theme.MUTED};font-size:13px;margin:0;line-height:1.45}}"
           "@media print{body{background:white;padding:0}section{border:none;padding:0 0 12px}}")
    html = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{title}</title>"
            f"{theme.font_link()}<style>{css}</style></head><body><main>"
            f"<h1>{title}</h1><p class='lede'>{len(ids)} graphics from the SHW Student Feedback Assistant. "
            "Each caption says what was measured and how far to trust it.</p>"
            + "".join(parts) + "</main></body></html>")
    gid = max(store, default=0) + 1
    store[gid] = Graphic(gid, "report", title, f"Report from graphics {', '.join(map(str, ids))}.",
                         html=html, source="report", parents=ids, weight=9)
    return gid

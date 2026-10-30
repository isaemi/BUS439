"""Dashboards as data: a dashboard is a JSON spec listing panels, optionally with a
focus group compared against everyone ("Faculty of Arts compared with overall").
The model (or a template) only writes the spec. It never writes plotting code or
queries, so every chart comes from the same checked numbers as the chat, and a
saved spec re-runs on new data next term.

    spec = template("focus_report", focus={"dimension": "faculty", "value": "Arts"})
    spec, errors = plan("How do Arts students experience SHW compared with everyone?")
    panels = build(spec)
    html = charts.render_page(spec, panels)
"""
import json
from dataclasses import dataclass, field

import pandas as pd

from . import analysis, config, tools

MAX_PANELS = 9

PANEL_TYPES = {
    "kpis": {"desc": "Headline cards: respondents, % agree, mean rating, % strongly disagree, "
                     "each compared with everyone when there is a focus group.",
             "params": {"service": "optional"}},
    "service_comparison": {"desc": "Bars per service (focus vs everyone).",
                           "params": {"measure": "optional: pct_agree or mean_score"}},
    "item_breakdown": {"desc": "Bars per survey item of one service, weakest first (focus vs everyone).",
                       "params": {"service": "required", "measure": "optional: pct_agree or mean_score"}},
    "group_comparison": {"desc": "Bars of % agree by group of one demographic dimension.",
                         "params": {"dimension": "required", "service": "optional"}},
    "demographics": {"desc": "Donut of who responded, with tabs for several dimensions.",
                     "params": {"dimensions": "optional list"}},
    "changes_to_watch": {"desc": "Alert cards: tested gaps between groups, and the biggest "
                                 "differences between the focus group and everyone.",
                         "params": {"service": "optional"}},
    "wait_time": {"desc": "Bars of % agree by appointment wait-time category.",
                  "params": {"service": "required: medical or counselling"}},
    "responses_over_time": {"desc": "Line of respondents over time (focus vs everyone).", "params": {}},
    "comment_themes": {"desc": "Bars of how often each theme appears in comments.",
                       "params": {"service": "optional"}},
    "equity_flags": {"desc": "Table of tested gaps between groups.", "params": {"service": "optional"}},
}

TITLES = {  # default title and subtitle per panel type
    "kpis": ("", ""),
    "service_comparison": ("Student experience by service", "% who agree (rated 4 or 5)"),
    "item_breakdown": ("Survey items", "% who agree, weakest first"),
    "group_comparison": ("Experience by group", "% who agree (rated 4 or 5)"),
    "demographics": ("Who responded", "Respondents by group"),
    "changes_to_watch": ("Changes to watch", "Tested gaps and notable differences"),
    "wait_time": ("Wait time and ratings", "% who agree, by wait before the appointment"),
    "responses_over_time": ("Survey responses over time", "Respondents per period"),
    "comment_themes": ("What comments mention", "Comments per theme"),
    "equity_flags": ("Tested gaps between groups", "FDR-corrected comparisons"),
}

WIDE_BY_DEFAULT = {"item_breakdown"}
DIMENSION_LABELS = {
    "faculty": "Faculty", "disability": "Disability type", "disability_any": "Disability (any)",
    "intl_or_domestic": "International / domestic", "first_generation": "First-generation",
    "academic_level": "Level of study", "year_of_study": "Year of study", "gender": "Gender",
    "indigenous": "Indigenous", "racialized": "Racialized", "racial_background": "Racial background",
    "trans_experience": "Trans experience", "living_arrangement": "Living arrangement",
    "new_to_ubc": "New to UBC", "enrollment_status": "Enrollment status", "identity_d15": "Identity (D15)",
}


@dataclass
class Panel:
    type: str
    title: str
    subtitle: str
    params: dict
    wide: bool = False
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra: dict = field(default_factory=dict)   # cards, tabs, alerts: anything not a table
    notes: list = field(default_factory=list)
    error: str = ""


def services():
    return [v["key"] for v in tools.load_dictionary()["services"].values()]


def service_names():
    return {v["key"]: v["name"] for v in tools.load_dictionary()["services"].values()}


def focus_label(focus):
    if not focus:
        return ""
    dim = DIMENSION_LABELS.get(focus["dimension"], focus["dimension"])
    if focus["dimension"] == "faculty":
        return f"Faculty of {focus['value']}" if "(" not in str(focus["value"]) else str(focus["value"])
    return f"{dim}: {focus['value']}"


def validate(spec):
    """Return a list of problems; empty means the spec is safe to build."""
    errs = []
    if not isinstance(spec, dict) or not isinstance(spec.get("panels"), list):
        return ["Spec must be an object with a 'panels' list."]
    if not spec["panels"]:
        errs.append("Spec has no panels.")
    if len(spec["panels"]) > MAX_PANELS:
        errs.append(f"At most {MAX_PANELS} panels.")
    svcs, dims = services(), analysis.group_dimensions()
    focus = spec.get("focus")
    if focus:
        if not isinstance(focus, dict) or focus.get("dimension") not in dims:
            errs.append(f"focus.dimension must be one of: {', '.join(dims)}.")
        else:
            valid = analysis.focus_values(analysis.connect(), focus["dimension"])
            if str(focus.get("value")) not in valid:
                errs.append(f"focus.value '{focus.get('value')}' not found. Valid values: {', '.join(valid[:30])}.")
    for i, p in enumerate(spec["panels"], 1):
        t = p.get("type")
        if t not in PANEL_TYPES:
            errs.append(f"Panel {i}: unknown type '{t}'. Use one of: {', '.join(PANEL_TYPES)}.")
            continue
        params = p.get("params") or {}
        for name, rule in PANEL_TYPES[t]["params"].items():
            if rule.startswith("required") and not params.get(name):
                errs.append(f"Panel {i} ({t}): '{name}' is required.")
        extra = set(params) - set(PANEL_TYPES[t]["params"])
        if extra:
            errs.append(f"Panel {i} ({t}): unexpected parameter(s) {sorted(extra)}.")
        if params.get("service") and params["service"] not in svcs:
            errs.append(f"Panel {i}: unknown service '{params['service']}'. Use one of: {', '.join(svcs)}.")
        if params.get("measure") and params["measure"] not in ("pct_agree", "mean_score"):
            errs.append(f"Panel {i}: measure must be pct_agree or mean_score.")
        if t == "group_comparison" and params.get("dimension") and params["dimension"] not in dims:
            errs.append(f"Panel {i}: unknown dimension '{params['dimension']}'. Use one of: {', '.join(dims)}.")
        if t == "demographics":
            bad = [d for d in params.get("dimensions") or [] if d not in dims]
            if bad:
                errs.append(f"Panel {i}: unknown dimension(s) {bad}.")
        if t == "wait_time" and params.get("service") not in analysis.WAIT_COLUMNS:
            errs.append(f"Panel {i}: wait_time only works for {', '.join(analysis.WAIT_COLUMNS)}.")
    return errs


# Panel builders. Each fills panel.df (long format: category, series, value, n) or panel.extra.

def _kpis(panel, con, ids, focus):
    svc = panel.params.get("service")
    k = analysis.kpi_summary(con, ids, svc)
    has_focus = ids is not None
    (fn, on) = k["respondents"]
    cards = [{
        "icon": "people", "label": "Respondents" + (f" ({service_names()[svc]})" if svc else ""),
        "value": f"{fn:,}",
        "delta": f"{100 * fn / on:.0f}% of all" if has_focus and on else "",
        "delta_dir": "none",
        "sub": f"All respondents: {on:,}" if has_focus else "Completed the survey",
    }]
    for key, label, icon, fmt, higher_is_better in [
        ("pct_agree", "Agree (rated 4–5)", "check", "{:.1f}%", True),
        ("mean_score", "Average rating (out of 5)", "star", "{:.2f}", True),
        ("pct_strongly_disagree", "Strongly disagree", "alert", "{:.1f}%", False),
    ]:
        f, o = k[key]
        card = {"icon": icon, "label": label, "value": fmt.format(f) if pd.notna(f) else "–",
                "delta": "", "delta_dir": "none", "sub": ""}
        if has_focus and pd.notna(f) and pd.notna(o):
            d = f - o
            unit = " pts" if key != "mean_score" else ""
            card["delta"] = f"{abs(d):.1f}{unit} vs overall" if key != "mean_score" else f"{abs(d):.2f} vs overall"
            card["delta_arrow"] = "up" if d > 0 else ("down" if d < 0 else "flat")
            good = (d > 0) == higher_is_better
            card["delta_dir"] = "flat" if abs(d) < 0.05 else ("good" if good else "bad")
            card["sub"] = f"Overall: {fmt.format(o)}"
        cards.append(card)
    if fn < config.MIN_CELL:
        panel.error = f"Fewer than {config.MIN_CELL} respondents in this group; results hidden."
        return
    panel.extra["cards"] = cards
    panel.notes.append("Differences are descriptive; see Changes to watch for tested gaps.") if has_focus else None


def _series(panel, con, ids, by, service=None):
    measure = panel.params.get("measure", "pct_agree")
    df = analysis.compare_series(con, by, ids, service, measure)
    if by == "service":
        df["category"] = df["category"].map(lambda k: service_names().get(k, k))
    df["n"] = df["n"].astype(int)
    panel.df = df
    panel.extra["measure"] = measure
    if measure == "mean_score":
        panel.subtitle = panel.subtitle.replace("% who agree (rated 4 or 5)", "Average rating (out of 5)") \
                                       .replace("% who agree", "Average rating")


def build_panel(p, con, ids, focus):
    t, params = p["type"], p.get("params") or {}
    title, subtitle = TITLES[t]
    panel = Panel(type=t, title=p.get("title") or title, subtitle=p.get("subtitle") or subtitle,
                  params=params, wide=p.get("width") == "wide" or (t in WIDE_BY_DEFAULT and p.get("width") != "normal"))
    svc = params.get("service")
    if t == "kpis":
        _kpis(panel, con, ids, focus)
    elif t == "service_comparison":
        _series(panel, con, ids, "service")
    elif t == "item_breakdown":
        _series(panel, con, ids, "item", svc)
        if not p.get("title"):
            panel.title = f"{service_names().get(svc, svc)}: survey items"
    elif t == "group_comparison":
        r = tools.compare_groups(params["dimension"], svc)
        if "error" in r:
            panel.error = r["error"]
        else:
            df = pd.DataFrame(r["rows"]).rename(columns={params["dimension"]: "category"})
            df["category"] = df["category"].astype(str)
            df["series"], df["value"] = "Overall", df["pct_agree"]
            panel.df = df
            panel.notes += [r[k] for k in ("suppression_note", "overlap_note") if r.get(k)]
            if not p.get("title"):
                panel.title = DIMENSION_LABELS.get(params["dimension"], params["dimension"])
                if svc:
                    panel.title += f" — {service_names().get(svc, svc)}"
    elif t == "demographics":
        tabs = []
        for dim in params.get("dimensions") or ["academic_level", "gender", "intl_or_domestic", "disability_any"]:
            g, hidden = analysis.composition(con, dim, ids)
            tabs.append({"label": DIMENSION_LABELS.get(dim, dim), "data": g, "hidden": hidden,
                         "total": int(g["n"].sum()) if len(g) else 0})
        panel.extra["tabs"] = tabs
        if focus:
            panel.subtitle = f"Respondents in {focus_label(focus)}"
    elif t == "changes_to_watch":
        panel.extra["alerts"] = _changes(con, ids, focus, svc)
    elif t == "wait_time":
        r = tools.wait_time_vs_rating(svc)
        if "error" in r:
            panel.error = r["error"]
        else:
            df = pd.DataFrame(r["rows"]).rename(columns={"wait": "category"})
            df["series"], df["value"] = "Overall", df["pct_agree"]
            panel.df = df
            c = r["correlation"]
            panel.notes.append(f"Rank correlation {c['spearman_rho']} (p = {c['p_value']}, n = {c['n']}); "
                               "a negative value would mean longer waits go with lower ratings.")
    elif t == "responses_over_time":
        lo, hi = analysis.period_range(con)
        span = (hi - lo).days if lo and hi else 0
        freq = "day" if span <= 60 else ("week" if span <= 240 else "month")
        df = analysis.responses_by_period(con, ids, freq)
        df["series"] = df["series"].replace({"Focus": focus_label(focus) or "Focus", "Overall": "All respondents"})
        panel.df = df.rename(columns={"period": "category"}).assign(value=lambda d: d["n"])
        panel.subtitle = f"Respondents per {freq}"
        if df["period"].nunique() < 3:
            panel.notes.append("Too few periods for a trend yet; this fills in as responses arrive.")
    elif t == "comment_themes":
        r = tools.comment_overview(svc)
        if "error" in r:
            panel.error = r["error"]
        else:
            df = pd.DataFrame(r["themes"]).rename(columns={"theme": "category"})
            if not df.empty:
                df["category"] = df["category"].str.replace("_", " ").str.capitalize()
                df["series"], df["value"] = "Overall", df["n"]
            panel.df = df
            panel.extra["measure"] = "count"
    elif t == "equity_flags":
        panel.extra["alerts"] = _changes(con, None, None, svc)

    if focus and not panel.df.empty and "series" in panel.df:
        panel.df["series"] = panel.df["series"].replace({"Focus": focus_label(focus), "Overall": "All respondents"})
    if t in ("service_comparison", "item_breakdown", "group_comparison", "wait_time") and not panel.df.empty \
            and (panel.df["n"] < 30).any():
        panel.notes.append("Faded bars have fewer than 30 respondents; read them with caution.")
    return panel


def _changes(con, ids, focus, service):
    """Alert cards. Red/amber only for tested results; descriptive differences are
    shown as information and say so."""
    alerts = []
    tests, flags, _ = analysis.equity_scan(con, service)
    names = service_names()
    for _, f in flags.sort_values("gap_pts").head(3).iterrows():
        lower = f["gap_pts"] < 0
        alerts.append({
            "level": "red" if lower else "info", "arrow": "down" if lower else "up",
            "text": f"{DIMENSION_LABELS.get(f['dimension'], f['dimension'])} = {f['group']} rate "
                    f"{names.get(f['service'], f['service'])} {'lower' if lower else 'higher'}: "
                    f"{f['pct_agree_group']}% vs {f['pct_agree_rest']}% agree",
            "sub": f"Tested gap of {abs(f['gap_pts'])} pts (n = {int(f['n_group'])}, q = {f['q']:.3f})"})
    if ids is not None:
        items = analysis.compare_series(con, "item", ids, service)
        wide = items.pivot_table(index="category", columns="series", values=["value", "n"])
        if ("value", "Focus") in wide:
            gap = (wide[("value", "Focus")] - wide[("value", "Overall")]).dropna()
            ok = wide[("n", "Focus")] >= 30
            gap = gap[ok.reindex(gap.index, fill_value=False)]
            for item, g in gap.reindex(gap.abs().sort_values(ascending=False).index).head(2).items():
                if abs(g) < 5:
                    continue
                alerts.append({
                    "level": "amber", "arrow": "down" if g < 0 else "up",
                    "text": f"{focus_label(focus)} {'lower' if g < 0 else 'higher'} than overall on "
                            f"“{item[:70]}{'…' if len(item) > 70 else ''}” ({g:+.1f} pts)",
                    "sub": "Descriptive difference, not tested"})
    if not any(a["level"] == "red" for a in alerts):
        alerts.append({"level": "info", "arrow": "info",
                       "text": f"No tested gaps between groups ({len(tests)} comparisons"
                               + (f" for {names.get(service, service)}" if service else "") + ")",
                       "sub": "Each group vs everyone else, corrected for multiple comparisons"})
    return alerts


def build(spec):
    errs = validate(spec)
    if errs:
        raise ValueError("Invalid dashboard spec:\n" + "\n".join(errs))
    con = analysis.connect()
    focus = spec.get("focus")
    ids = analysis.focus_ids(con, focus)
    return [build_panel(p, con, ids, focus) for p in spec["panels"]]


def page_meta(spec):
    """Header facts: title, focus, date range, response counts."""
    con = analysis.connect()
    lo, hi = analysis.period_range(con)
    focus = spec.get("focus")
    ids = analysis.focus_ids(con, focus)
    total = con.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
    n = len(ids) if ids is not None else total
    fmt = lambda d: d.strftime("%b %-d, %Y") if d else "?"
    return {"title": "Student Health & Wellbeing",
            "subtitle": spec.get("title") or focus_label(focus) or "All services",
            "period": f"{fmt(lo)} – {fmt(hi)}",
            "comparison": "Compared with all respondents" if focus else "All respondents",
            "n": n, "total": total, "focus": bool(focus),
            "sources": ["Qualtrics (anonymous survey responses)"]}


# Templates: fixed dashboards that need no model.
TEMPLATES = ["focus_report", "service_overview", "service_deep_dive", "equity_overview"]


def template(name, service=None, focus=None):
    if name == "focus_report":
        focus = focus or {"dimension": "faculty", "value": "Arts"}
        return {"title": focus_label(focus), "focus": focus, "panels": [
            {"type": "kpis"},
            {"type": "responses_over_time"},
            {"type": "service_comparison"},
            {"type": "service_comparison", "title": "Average rating by service", "params": {"measure": "mean_score"}},
            {"type": "demographics"},
            {"type": "item_breakdown", "params": {"service": service or "counselling"}},
            {"type": "changes_to_watch"}]}
    if name == "service_overview":
        return {"title": "All services", "panels": [
            {"type": "kpis"},
            {"type": "service_comparison"},
            {"type": "responses_over_time"},
            {"type": "demographics"},
            {"type": "changes_to_watch"}]}
    if name == "service_deep_dive":
        service = service or "counselling"
        name_ = service_names().get(service, service)
        panels = [
            {"type": "kpis", "params": {"service": service}},
            {"type": "item_breakdown", "params": {"service": service}},
            {"type": "group_comparison", "params": {"dimension": "intl_or_domestic", "service": service}},
            {"type": "group_comparison", "params": {"dimension": "first_generation", "service": service}},
            {"type": "group_comparison", "params": {"dimension": "disability_any", "service": service}},
            {"type": "changes_to_watch", "params": {"service": service}}]
        if service in analysis.WAIT_COLUMNS:
            panels.insert(2, {"type": "wait_time", "params": {"service": service}})
        return {"title": name_, "panels": panels[:MAX_PANELS]}
    if name == "equity_overview":
        return {"title": "Equity overview", "panels":
                [{"type": "changes_to_watch", "title": "Tested gaps between groups"}] +
                [{"type": "group_comparison", "params": {"dimension": d}}
                 for d in analysis.equity_dimensions()][:MAX_PANELS - 1]}
    raise ValueError(f"Unknown template '{name}'")


# Planning with the model: request in plain English -> spec.
def _planner_system():
    d = tools.describe_data()
    types = "\n".join(f"- {k}: {v['desc']} params: {json.dumps(v['params'])}" for k, v in PANEL_TYPES.items())
    return f"""You design dashboards for UBC Student Health & Wellbeing staff.
Turn the request into a JSON dashboard spec. Use only these panel types:
{types}

Services: {json.dumps(d['services'])}
Dimensions: {json.dumps(analysis.group_dimensions())}
Faculty values: {json.dumps(d['faculty_values'])}

Format: {{"title": "...", "focus": {{"dimension": "...", "value": "..."}} or null,
          "panels": [{{"type": "...", "params": {{...}}, "width": "wide" or "normal"}}]}}
Rules: use "focus" when the request is about one group (e.g. Arts students,
international students) so every panel compares that group with everyone; values
must match exactly. Start with kpis. Include changes_to_watch when the request is
about groups, equity or what needs attention. At most {MAX_PANELS} panels; only the
params listed for each type. Reply with JSON only."""


def plan(request, chat_json_fn=None, retries=1):
    """Returns (spec, errors). Validation errors are sent back once for a fix."""
    from . import llm
    chat_json_fn = chat_json_fn or llm.chat_json
    system = _planner_system()
    spec = chat_json_fn(system, request)
    if isinstance(spec, dict) and spec.get("focus") in (None, {}, "null"):
        spec.pop("focus", None)
    errs = validate(spec)
    for _ in range(retries):
        if not errs:
            break
        spec = chat_json_fn(system, f"{request}\n\nYour previous spec had these problems; fix them:\n"
                            + "\n".join(errs) + f"\nPrevious spec: {json.dumps(spec)}")
        if isinstance(spec, dict) and spec.get("focus") in (None, {}, "null"):
            spec.pop("focus", None)
        errs = validate(spec)
    return spec, errs


SAVE_DIR = config.DB_PATH.parent / "dashboards"


def save(spec, name):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"{''.join(c if c.isalnum() or c in '-_' else '_' for c in name)}.json"
    path.write_text(json.dumps(spec, indent=2))
    return path


def saved():
    return sorted(p.stem for p in SAVE_DIR.glob("*.json")) if SAVE_DIR.exists() else []


def load(name):
    return json.loads((SAVE_DIR / f"{name}.json").read_text())

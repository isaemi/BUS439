"""Turns tool results into numbered graphics with captions, and graphics into reports."""
import json
import re
from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class Graphic:
    id: int
    kind: str                 # "chart" or "report"
    title: str
    caption: str              # the key numbers, copied from the tool result
    fig: object = None        # plotly figure (charts)
    html: str = ""            # full page (reports)
    source: str = ""          # tool name
    args: dict = field(default_factory=dict)
    parents: list = field(default_factory=list)   # for reports: which graphics fed it


# ---- builders: (args, result) -> list of (title, caption, figure) -------------------

def _what_changed(args, r):
    if "volume_change" not in r:
        return []
    v, w = r["volume_change"], r["volume_in_same_windows"]
    p = r["periods"]
    df = pd.DataFrame({
        "period": [p["before"], p["after"]],
        "students seen": [v["before"], v["after"]],
    })
    fig = px.bar(df, x="period", y="students seen", text="students seen")
    fig.update_traces(textposition="outside")
    lo, hi = v["ci_pct"]
    cap = (f"Students seen rose {v['change_pct']}% (range {lo}% to {hi}%; "
           f"n = {v['before']} before, {v['after']} after). Verdict: {v['verdict']}. "
           f"Attended visits {w['before']['attended']} to {w['after']['attended']}.")
    out = [(f"{r['service_line'].title()}: students seen, same dates each year", cap, fig)]
    # visits-per-student by group, descriptive
    if r.get("metric") == "visits_per_student" and r.get("groups"):
        g = pd.DataFrame(r["groups"])[["group", "before", "after"]].melt("group", var_name="when")
        f2 = px.bar(g, x="value", y="group", color="when", barmode="group", orientation="h")
        o = r["overall"]
        out.append((f"{r['service_line'].title()}: visits per student by {r['dimension']}",
                    f"Overall {o['before']} to {o['after']} (range {o['ci'][0]} to {o['ci'][1]}); "
                    "per-group values are descriptive, not tested.", f2))
    return out


def _chr_trend(args, r):
    df = pd.DataFrame(r.get("rows", []))
    if df.empty:
        return []
    fig = px.line(df, x="period", y="appointments", markers=True) if args.get("freq") == "month" \
        else px.bar(df, x="period", y="appointments", text="appointments")
    cap = f"Attended appointments per {args.get('freq', 'term')}, {args.get('service_line', 'all services')}."
    if r.get("warning"):
        cap += " Note: " + r["warning"].split(". Do not")[0] + "."
    return [(f"{args.get('service_line', 'All services')}: appointments over time", cap, fig)]


def _chr_summary(args, r):
    dim = args.get("dimension")
    df = pd.DataFrame(r.get("rows", []))
    if df.empty or not dim or dim not in df:
        return []
    fig = px.bar(df, x=dim, y="no_show_rate_pct", text="no_show_rate_pct")
    return [(f"{args.get('service_line', 'All')}: no-show rate by {dim}",
             "No-show % = missed / booked appointments.", fig)]


def _wait(args, r):
    df = pd.DataFrame(r.get("rows", []))
    if df.empty:
        return []
    c = r["correlation"]
    fig = px.bar(df, x="wait", y="pct_agree", text="pct_agree")
    return [(f"{args.get('service', '').title()}: % agree by wait category",
             f"Rank correlation {c['spearman_rho']} (p = {c['p_value']}, n = {c['n']}). {r['interpretation']}", fig)]


BUILDERS = {"what_changed": _what_changed, "chr_trend": _chr_trend,
            "chr_summary": _chr_summary, "wait_time_vs_rating": _wait}


# ---- registry ------------------------------------------------------------------------

def register_from_trace(trace, store):
    """trace = [(tool_name, args, result), ...]. Returns ids of graphics for this answer.
    Same tool + same args twice gives the same graphic, not a duplicate."""
    ids = []
    for name, args, res in trace:
        build = BUILDERS.get(name)
        if not build or not isinstance(res, dict) or "error" in res:
            continue
        for i, (title, cap, fig) in enumerate(build(args, res)):
            key = (name, json.dumps(args, sort_keys=True, default=str), i)
            hit = next((g for g in store.values() if g.args.get("_key") == key), None)
            if hit:
                ids.append(hit.id)
                continue
            gid = max(store, default=0) + 1
            fig.update_layout(title=f"Graphic {gid}: {title}", margin=dict(t=60, b=30))
            store[gid] = Graphic(gid, "chart", title, cap, fig, source=name, args={**args, "_key": key})
            ids.append(gid)
    return ids


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


def build_report(ids, store, title="SHW report"):
    missing = [i for i in ids if i not in store or store[i].kind != "chart"]
    if missing:
        raise KeyError(f"No chart with number(s): {missing}")
    parts = []
    for n, i in enumerate(ids):
        g = store[i]
        parts.append(f"<section><h2>Graphic {g.id}: {g.title}</h2>"
                     + g.fig.to_html(full_html=False, include_plotlyjs=True if n == 0 else False)
                     + f"<p>{g.caption}</p><small>Source: {g.source}</small></section>")
    html = (f"<html><body style='font-family:sans-serif;max-width:900px;margin:auto'>"
            f"<h1>{title}</h1><p>Built from graphics {', '.join(map(str, ids))}.</p>"
            + "".join(parts) + "</body></html>")
    gid = max(store, default=0) + 1
    store[gid] = Graphic(gid, "report", title, f"Report from graphics {', '.join(map(str, ids))}.",
                         html=html, source="report", parents=ids)
    return gid
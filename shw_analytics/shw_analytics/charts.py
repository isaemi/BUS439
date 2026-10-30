"""Draws a dashboard as one HTML page in the style of the SHW mock-up: a header,
a row of KPI cards, and a grid of white cards holding Altair charts, donuts and
alert lists. The same page is shown in the app and downloaded as HTML, so what
staff see and what they send are identical.

Charts are Vega-Lite (via Altair) loaded from a CDN, so viewing needs internet."""
import html
import json

import altair as alt
import pandas as pd

from . import theme

SMALL_N = 30  # bars from groups smaller than this are faded
COLS = 3      # grid columns


def _esc(x):
    return html.escape(str(x))


def _fmt(measure):
    return {"mean_score": ".2f", "count": "d"}.get(measure, ".0f")


def _suffix(measure):
    return "%" if measure in (None, "pct_agree") else ""


def _series_colors(df):
    names = list(dict.fromkeys(df["series"]))
    if len(names) == 1:
        return names, [theme.FOCUS]
    # focus group first and strong; everyone else pale
    names = [n for n in names if n != "All respondents"] + ["All respondents"]
    return names, [theme.FOCUS, theme.OVERALL][:len(names)]


def _with_labels(df, measure):
    df = df.copy()
    f = _fmt(measure)
    df["label"] = df["value"].map(lambda v: (f"{v:{f}}" if pd.notna(v) else "") + _suffix(measure))
    df["faded"] = df["n"] < SMALL_N if "n" in df else False
    return df


def bars(panel, horizontal=None):
    df = panel.df
    if df.empty:
        return None
    measure = panel.extra.get("measure", "pct_agree")
    df = _with_labels(df, measure)
    names, colors = _series_colors(df)
    two = len(names) > 1
    ranking = panel.type in ("item_breakdown", "comment_themes", "service_comparison")
    if horizontal is None:
        horizontal = panel.type in ("item_breakdown", "comment_themes") or df["category"].astype(str).str.len().max() > 16

    # Order: rankings by the first series' value (weakest first for items); others keep code order.
    first = df[df["series"] == names[0]].set_index("category")["value"]
    if ranking:
        asc = panel.type == "item_breakdown"
        order = first.sort_values(ascending=asc).index.tolist()
        order += [c for c in df["category"].unique() if c not in order]
    else:
        order = sorted(df["category"].unique(), key=lambda c: (len(str(c)), str(c)))
    df["rank"] = df["category"].map({c: i for i, c in enumerate(order)})

    if measure == "pct_agree":
        scale = alt.Scale(domain=[0, 100])
    elif measure == "mean_score":
        scale = alt.Scale(domain=[0, 5])
    else:
        scale = alt.Scale(zero=True)

    # Counts ranked from most to least fade from strong to pale, like "Top reasons for
    # visit". Score charts stay one colour so that fading can mean "small group".
    if not two and panel.type == "comment_themes":
        ramp = [theme.RAMP[min(i, len(theme.RAMP) - 1)] for i in range(len(order))]
        color = alt.Color("category:N", scale=alt.Scale(domain=order, range=ramp), legend=None)
    else:
        color = alt.Color("series:N", scale=alt.Scale(domain=names, range=colors), legend=None)

    opacity = alt.condition("datum.faded", alt.value(0.45), alt.value(1))
    tooltip = [alt.Tooltip("category:N", title="Group"), alt.Tooltip("series:N", title="Series"),
               alt.Tooltip("value:Q", title="Value", format=_fmt(measure)), alt.Tooltip("n:Q", title="n")]
    base = alt.Chart(df)
    band = 40 if two else 26
    if horizontal:
        y = alt.Y("category:N", sort=order, title=None,
                  axis=alt.Axis(labelLimit=260, labelFontSize=12, labelColor=theme.TEXT))
        x = alt.X("value:Q", scale=scale, title=None, axis=alt.Axis(labels=False, grid=False))
        enc = dict(y=y, x=x, color=color, opacity=opacity, tooltip=tooltip)
        if two:
            enc["yOffset"] = alt.YOffset("series:N", sort=names)
        chart = base.mark_bar(height={"band": 0.8}).encode(**enc)
        text = base.mark_text(align="left", dx=5, fontSize=12, color=theme.TEXT).encode(
            y=y, x=x, text="label:N", **({"yOffset": alt.YOffset("series:N", sort=names)} if two else {}))
        height = max(120, len(order) * band)
    else:
        x = alt.X("category:N", sort=order, title=None,
                  axis=alt.Axis(labelAngle=0, labelLimit=110, labelColor=theme.TEXT, labelFontSize=12))
        y = alt.Y("value:Q", scale=scale, title=None,
                  axis=alt.Axis(tickCount=5, format=_fmt(measure) if measure == "mean_score" else "d",
                                labelExpr=f"datum.label + '{_suffix(measure)}'"))
        enc = dict(x=x, y=y, color=color, opacity=opacity, tooltip=tooltip)
        if two:
            enc["xOffset"] = alt.XOffset("series:N", sort=names)
        chart = base.mark_bar(width={"band": 0.85}).encode(**enc)
        text = base.mark_text(dy=-8, fontSize=12, fontWeight=600, color=theme.TEXT).encode(
            x=x, y=y, text="label:N", **({"xOffset": alt.XOffset("series:N", sort=names)} if two else {}))
        height = 230
    return (chart + text).properties(width="container", height=height)


def line(panel):
    df = panel.df
    if df.empty:
        return None
    names, colors = _series_colors(df)
    enc = dict(x=alt.X("category:T", title=None, axis=alt.Axis(format="%b %d", labelColor=theme.TEXT)),
               y=alt.Y("value:Q", title=None, axis=alt.Axis(tickCount=5)),
               color=alt.Color("series:N", scale=alt.Scale(domain=names, range=colors), legend=None),
               tooltip=[alt.Tooltip("category:T", title="Date"), "series:N", alt.Tooltip("value:Q", title="Respondents")])
    return alt.Chart(df).mark_line(point=True).encode(**enc).properties(width="container", height=230)


def donut(tab):
    df = tab["data"]
    if df.empty:
        return None
    order = df["group"].tolist()
    k = len(order)
    ramp = [theme.DONUT[round(i * (len(theme.DONUT) - 1) / max(k - 1, 1))] for i in range(k)]
    arc = alt.Chart(df).mark_arc(innerRadius=58, outerRadius=88, stroke="#fff", strokeWidth=2).encode(
        theta=alt.Theta("n:Q", stack=True),
        color=alt.Color("group:N", scale=alt.Scale(domain=order, range=ramp), legend=None),
        order=alt.Order("n:Q", sort="descending"),
        tooltip=[alt.Tooltip("group:N", title="Group"), "n:Q", alt.Tooltip("share:Q", title="%")])
    total = pd.DataFrame({"t": [f"{tab['total']:,}"], "s": ["Respondents"]})
    big = alt.Chart(total).mark_text(fontSize=20, fontWeight=700, color=theme.NAVY, dy=-8).encode(text="t:N")
    small = alt.Chart(total).mark_text(fontSize=12, color=theme.MUTED, dy=12).encode(text="s:N")
    return (arc + big + small).properties(width=190, height=190), list(zip(order, ramp))


def legend_html(panel):
    """Series legend in HTML, placed under the subtitle like the mock-up."""
    if panel.df.empty or "series" not in panel.df or panel.df["series"].nunique() < 2:
        return ""
    names, colors = _series_colors(panel.df)
    return '<div class="slegend">' + "".join(
        f'<span><i style="background:{c}"></i>{_esc(n)}</span>' for n, c in zip(names, colors)) + "</div>"


def chart_for(panel):
    if panel.error or panel.df.empty:
        return None
    if panel.type == "responses_over_time":
        return line(panel)
    return bars(panel)


def _vl(chart):
    return chart.configure(**theme.altair_config()).to_dict()


ICONS = {
    "people": '<path d="M9 11a3 3 0 1 0 0-6 3 3 0 0 0 0 6zm6 0a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5zM3 19c0-3 2.7-5 6-5s6 2 6 5v1H3zm13 1v-1c0-1.6-.6-3-1.7-4 .5-.1 1.1-.2 1.7-.2 2.8 0 5 1.7 5 4.2v1z"/>',
    "check": '<path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm-1.2 14.2-4-4 1.4-1.4 2.6 2.6 5.6-5.6 1.4 1.4z"/>',
    "star": '<path d="m12 3 2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/>',
    "alert": '<path d="M12 3 2 20h20zm-1 6h2v6h-2zm0 7h2v2h-2z"/>',
}
ARROWS = {"up": "↑", "down": "↓", "flat": "→", "info": "i"}


def _kpi_cards(cards):
    out = []
    for c in cards:
        color = {"good": theme.UP, "bad": theme.DOWN}.get(c.get("delta_dir"), theme.MUTED)
        delta = ""
        if c.get("delta"):
            arrow = ARROWS.get(c.get("delta_arrow"), "")
            delta = f'<div class="delta" style="color:{color}"><b>{arrow}</b> {_esc(c["delta"])}</div>'
        out.append(f"""<div class="card kpi">
  <div class="icon"><svg viewBox="0 0 24 24" fill="{theme.FOCUS}">{ICONS.get(c['icon'], '')}</svg></div>
  <div class="kbody"><div class="klabel">{_esc(c['label'])}</div>
  <div class="krow"><div class="kvalue">{_esc(c['value'])}</div>{delta}</div>
  <div class="ksub">{_esc(c['sub'])}</div></div></div>""")
    return f'<div class="kpis">{"".join(out)}</div>'


def _alerts(alerts):
    rows = []
    for a in alerts:
        bg, fg = theme.ALERT_BG[a["level"]], theme.ALERT_FG[a["level"]]
        icon = ARROWS.get(a["arrow"], "")
        icon_html = (f'<span class="ainfo" style="background:{fg}">i</span>' if a["arrow"] == "info"
                     else f'<span class="aarrow" style="color:{fg}">{icon}</span>')
        rows.append(f'<div class="alert" style="background:{bg}">{icon_html}<div><div class="atext">'
                    f'{_esc(a["text"])}</div><div class="asub">{_esc(a.get("sub", ""))}</div></div></div>')
    return "".join(rows)


def render_page(spec, panels, meta):
    """Full dashboard page (HTML string)."""
    embeds, body = [], []
    kpi_html = ""
    for i, p in enumerate(panels):
        if p.type == "kpis":
            kpi_html += _kpi_cards(p.extra.get("cards", [])) if not p.error else \
                f'<div class="card"><p class="err">{_esc(p.error)}</p></div>'
            continue
        inner = ""
        if p.error:
            inner = f'<p class="err">{_esc(p.error)}</p>'
        elif p.type in ("changes_to_watch", "equity_flags"):
            inner = _alerts(p.extra.get("alerts", []))
        elif p.type == "demographics":
            tabs, panes = [], []
            for j, tab in enumerate(p.extra.get("tabs", [])):
                tid = f"d{i}_{j}"
                tabs.append(f'<button class="tab{" on" if j == 0 else ""}" data-group="d{i}" '
                            f'data-target="{tid}">{_esc(tab["label"])}</button>')
                res = donut(tab)
                if res is None:
                    pane = '<p class="muted">No groups large enough to show.</p>'
                else:
                    ch, legend = res
                    embeds.append((f"c{tid}", _vl(ch)))
                    leg = "".join(
                        f'<div class="leg"><span class="dot" style="background:{col}"></span>'
                        f'<span class="lname">{_esc(g)}</span><span class="lval">'
                        f'{row.share:.0f}% ({int(row.n):,})</span></div>'
                        for (g, col), row in zip(legend, tab["data"].itertuples()))
                    pane = f'<div class="donut"><div id="c{tid}"></div><div class="legend">{leg}</div></div>'
                if tab["hidden"]:
                    pane += f'<p class="note">Hidden (fewer than 5): {_esc(", ".join(tab["hidden"]))}</p>'
                panes.append(f'<div class="pane" id="{tid}" data-group="d{i}" '
                             f'style="display:{"block" if j == 0 else "none"}">{pane}</div>')
            inner = f'<div class="tabs">{"".join(tabs)}</div>{"".join(panes)}'
        else:
            ch = chart_for(p)
            if ch is None:
                inner = '<p class="muted">No data to show yet.</p>'
            else:
                embeds.append((f"c{i}", _vl(ch)))
                inner = legend_html(p) + f'<div class="chart" id="c{i}"></div>'
        notes = "".join(f'<p class="note">{_esc(n)}</p>' for n in p.notes if n)
        sub = f'<div class="psub">{_esc(p.subtitle)}</div>' if p.subtitle else ""
        body.append(f'<div class="card panel{" wide" if p.wide else ""}"><div class="ptitle">'
                    f'{_esc(p.title)}</div>{sub}{inner}{notes}</div>')

    share = f'({100 * meta["n"] / meta["total"]:.0f}% of all)' if meta["focus"] and meta["total"] else ""
    sources = "".join(f"<div>{_esc(s)}</div>" for s in meta["sources"])
    header = f"""<div class="header">
  <div class="htitle"><div class="h1">{_esc(meta['title'])}</div><div class="h2">{_esc(meta['subtitle'])}</div>
    <div class="hmeta">{_esc(meta['period'])} &nbsp;|&nbsp; {_esc(meta['comparison'])}</div></div>
  <div class="hbox"><div class="hcol"><div class="hlabel">Data sources</div>{sources}</div>
    <div class="hcol hsep"><div class="hlabel">Survey responses</div>
      <div><span class="hbig">{meta['n']:,}</span> <span class="muted">{share}</span></div>
      <div class="muted">{'All respondents: ' + format(meta['total'], ',') if meta['focus'] else 'Small groups hidden'}</div>
    </div></div></div>"""

    script = "\n".join(f"vegaEmbed('#{cid}', {json.dumps(vl)}, {{actions:false, renderer:'svg'}});"
                       for cid, vl in embeds)
    t = theme
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{_esc(meta['subtitle'])} – SHW</title>
<link href="{t.FONT_URL}" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/vega@{alt.VEGA_VERSION}"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@{alt.VEGALITE_VERSION}"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@{alt.VEGAEMBED_VERSION}"></script>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:{t.PAGE};font-family:{t.FONT};color:{t.TEXT}}}
.page{{max-width:1240px;margin:0 auto;padding:22px}}
.header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:18px}}
.h1{{font-size:30px;font-weight:700;color:{t.NAVY};line-height:1.15}}
.h2{{font-size:22px;color:{t.NAVY};margin-top:2px}} .hmeta{{color:{t.MUTED};margin-top:10px;font-size:15px}}
.hbox{{display:flex;background:#E9EEF6;border-radius:10px;padding:14px 18px;gap:18px;font-size:13px;color:{t.TEXT}}}
.hlabel{{font-weight:700;margin-bottom:6px}} .hsep{{border-left:1px solid #CBD5E3;padding-left:18px}}
.hbig{{font-size:30px;font-weight:700;color:{t.NAVY}}}
.card{{background:{t.CARD};border:1px solid {t.BORDER};border-radius:12px;padding:18px 20px;
      box-shadow:0 1px 2px rgba(16,35,80,.04)}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px}}
.kpi{{display:flex;gap:14px;align-items:flex-start}}
.icon{{width:44px;height:44px;border-radius:50%;background:#E8EFFA;display:flex;align-items:center;
      justify-content:center;flex:none}} .icon svg{{width:22px;height:22px}}
.klabel{{font-weight:600;font-size:16px;color:{t.NAVY}}}
.krow{{display:flex;align-items:center;gap:12px;margin:6px 0 4px}}
.kvalue{{font-size:32px;font-weight:700;color:{t.NAVY}}} .delta{{font-size:13px;line-height:1.2}}
.ksub{{color:{t.MUTED};font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat({COLS},1fr);gap:16px}}
.wide{{grid-column:span 2}}
.ptitle{{font-size:19px;font-weight:700;color:{t.NAVY}}} .psub{{color:{t.MUTED};font-size:14px;margin:2px 0 12px}}
.chart{{width:100%}} .note{{color:{t.MUTED};font-size:12px;margin:8px 0 0}} .muted{{color:{t.MUTED}}}
.err{{color:{t.DOWN}}}
.alert{{display:flex;gap:12px;align-items:flex-start;border-radius:8px;padding:10px 12px;margin-bottom:8px}}
.aarrow{{font-size:20px;font-weight:700;line-height:1;width:18px;text-align:center}}
.ainfo{{color:#fff;border-radius:50%;width:18px;height:18px;font-size:12px;font-weight:700;display:flex;
       align-items:center;justify-content:center;flex:none;margin-top:1px}}
.atext{{font-weight:600;font-size:14px}} .asub{{color:{t.MUTED};font-size:12px;margin-top:2px}}
.tabs{{display:flex;flex-wrap:wrap;border:1px solid {t.BORDER};border-radius:8px;overflow:hidden;margin-bottom:12px}}
.tab{{border:0;background:#fff;padding:7px 11px;font:inherit;font-size:12px;color:{t.TEXT};cursor:pointer;
     border-right:1px solid {t.BORDER}}} .tab.on{{background:{t.FOCUS};color:#fff}}
.donut{{display:flex;align-items:center;gap:16px;flex-wrap:wrap}} .legend{{flex:1;min-width:150px}}
.leg{{display:flex;align-items:center;gap:8px;margin:6px 0;font-size:14px}}
.dot{{width:14px;height:14px;border-radius:50%;flex:none}} .lname{{flex:1}} .lval{{color:{t.TEXT}}}
.slegend{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:14px;font-size:13px;margin:-4px 0 8px}}
.slegend i{{display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px;vertical-align:-1px}}
.footer{{color:{t.MUTED};font-size:12px;margin-top:16px}}
@media (max-width:900px){{.kpis{{grid-template-columns:repeat(2,1fr)}} .grid{{grid-template-columns:1fr}} .wide{{grid-column:auto}}}}
</style></head><body><div class="page">
{header}
{kpi_html}
<div class="grid">{''.join(body)}</div>
<div class="footer">Built from de-identified survey data. Groups with fewer than 5 respondents are hidden.
Differences between groups are descriptive unless marked as tested.</div>
</div>
<script>
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => {{
  const g = b.dataset.group;
  document.querySelectorAll('.tab[data-group="'+g+'"]').forEach(x => x.classList.remove('on'));
  document.querySelectorAll('.pane[data-group="'+g+'"]').forEach(x => x.style.display = 'none');
  b.classList.add('on'); document.getElementById(b.dataset.target).style.display = 'block';
}}));
{script}
</script></body></html>"""


def estimate_height(panels):
    """Rough page height for embedding in Streamlit."""
    h = 200 + (170 if any(p.type == "kpis" for p in panels) else 0)
    col, row_h = 0, 0
    for p in panels:
        if p.type == "kpis":
            continue
        ph = 380
        if p.type in ("item_breakdown", "comment_themes") and not p.df.empty:
            ph = 150 + p.df["category"].nunique() * (30 if p.df["series"].nunique() > 1 else 22)
        if p.type in ("changes_to_watch", "equity_flags"):
            ph = 120 + 70 * len(p.extra.get("alerts", []))
        span = 2 if p.wide else 1
        if col + span > COLS:
            h += row_h + 16
            col, row_h = 0, 0
        col += span
        row_h = max(row_h, ph)
    return h + row_h + 80

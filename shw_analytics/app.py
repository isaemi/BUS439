"""Streamlit front end.  Run:  streamlit run app.py"""
import streamlit as st
import streamlit.components.v1 as components

from shw_analytics import agent, analysis, charts, config, dashboard, tools, figures

st.set_page_config(page_title="SHW Feedback Assistant", layout="wide")
st.title("SHW Student Feedback Assistant")
st.caption("Answers come from de-identified survey data. Model: on-prem Qwen via the LTIC AI Gateway. "
           "Groups with fewer than the minimum number of respondents are hidden.")

import datetime as _dt
import gc
import os
import subprocess
import sys
import re

EXPORTS = config.DB_PATH.parent / "exports"


def _rebuild():
    """Run ingest in a separate process (it needs write access to the database file),
    then drop this process's open connections so the new file is picked up."""
    r = subprocess.run([sys.executable, "-m", "shw_analytics.ingest", str(EXPORTS)],
                       capture_output=True, text=True, cwd=config.ROOT)
    gc.collect()
    return r


with st.sidebar:
    st.subheader("Survey data")
    if config.DB_PATH.exists():
        built = _dt.datetime.fromtimestamp(config.DB_PATH.stat().st_mtime)
        st.caption(f"Last updated {built.strftime('%b')} {built.day}, {built:%Y %H:%M}")
    files = sorted(EXPORTS.glob("*.csv")) if EXPORTS.exists() else []
    st.caption(f"{len(files)} export file(s) in data/exports")
    uploads = st.file_uploader("Add Qualtrics exports (CSV, numeric values)", type="csv",
                               accept_multiple_files=True)
    if uploads and st.button("Load into the database"):
        EXPORTS.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, f in enumerate(uploads):
            (EXPORTS / f"upload_{stamp}_{i}.csv").write_bytes(f.getvalue())
        with st.spinner("Rebuilding..."):
            r = _rebuild()
        if r.returncode == 0:
            st.success("Database updated.")
            st.rerun()
        else:
            st.error((r.stderr or r.stdout).strip().splitlines()[-1])
    chr_dir = config.DB_PATH.parent / "chr_exports"
    chr_files = sorted(chr_dir.glob("*.csv")) if chr_dir.exists() else []
    st.caption(f"{len(chr_files)} CHR export file(s) in data/chr_exports")
    chr_uploads = st.file_uploader("Add CHR appointment exports (CSV)", type="csv",
                                   accept_multiple_files=True, key="chr_upload")
    if chr_uploads and st.button("Load CHR into the database"):
        chr_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, f in enumerate(chr_uploads):
            (chr_dir / f"chr_{stamp}_{i}.csv").write_bytes(f.getvalue())
        with st.spinner("Rebuilding..."):
            r = _rebuild()
        if r.returncode == 0:
            st.success("Database updated.")
            st.rerun()
        else:
            st.error((r.stderr or r.stdout).strip().splitlines()[-1])
    if os.getenv("QUALTRICS_API_TOKEN") and st.button("Pull latest from Qualtrics"):
        with st.spinner("Exporting from Qualtrics..."):
            r = subprocess.run([sys.executable, "-m", "shw_analytics.qualtrics_api"],
                               capture_output=True, text=True, cwd=config.ROOT)
            gc.collect()
        if r.returncode == 0:
            st.success("Pulled and rebuilt.")
            st.rerun()
        else:
            st.error((r.stderr or r.stdout).strip().splitlines()[-1])

if not config.DB_PATH.exists():
    st.warning("No data yet. Add a Qualtrics export in the sidebar, or run "
               "`python -m shw_analytics.ingest data/exports`.")
    st.stop()

ask_tab, dash_tab, equity_tab, comments_tab, quality_tab = st.tabs(
    ["Ask a question", "Dashboards", "Equity flags", "Comments", "Data quality"])


with ask_tab:
    if "turns" not in st.session_state:
        st.session_state.turns = []
    if "figures" not in st.session_state:
        st.session_state.figures = {}
    store = st.session_state.figures

    c1, c2 = st.columns([1, 2])
    if c1.button("New conversation", help="Clear earlier questions and graphics so they don't affect the next answer"):
        st.session_state.turns = []
        st.session_state.figures = {}
        st.rerun()
    if st.session_state.turns and c2.button("Make a dashboard from this conversation"):
        request = "A dashboard covering these questions:\n" + "\n".join(
            f"- {t['question']}" for t in st.session_state.turns)
        with st.spinner("Planning dashboard..."):
            planned, errs = dashboard.plan(request)
        if errs:
            st.error("The plan still had problems:\n\n" + "\n".join(f"- {e}" for e in errs))
        else:
            st.session_state.dash_spec = planned
            st.success("Done. Open the Dashboards tab and choose \"Describe it\" to see it.")

    def show_trace(trace):
        with st.expander("How this answer was built"):
            if not trace:
                st.write("No tools were called.")
            for name, args, res in trace:
                st.markdown(f"**{name}** `{args}`")
                st.json(res, expanded=False)

    def show_graphics(ids, key_prefix):
        for gid in ids:
            g = store[gid]
            if g.kind == "chart":
                st.plotly_chart(g.fig, key=f"{key_prefix}_{gid}", width="stretch")
                st.caption(f"**Graphic {gid}.** {g.caption}")
            else:
                components.html(g.html, height=700, scrolling=True)
                st.download_button(f"Download Graphic {gid} (report)", g.html,
                                   file_name=f"report_graphic_{gid}.html", mime="text/html",
                                   key=f"{key_prefix}_dl_{gid}")

    for n, t in enumerate(st.session_state.turns):
        st.chat_message("user").write(t["question"])
        with st.chat_message("assistant"):
            st.write(t["answer"])
            show_graphics(t.get("graphics", []), f"h{n}")
            show_trace(t["trace"])

    if st.session_state.turns:
        st.caption(f"Follow-ups use the last {agent.MAX_TURNS} questions as context. "
                   "Start a new conversation to change topic. "
                   "To combine graphics, type e.g. \"make a report from graphics 1, 3 and 8\".")

    q = st.chat_input("e.g. How does counselling compare for first-generation students?")
    if q:
        st.chat_message("user").write(q)
        n = len(st.session_state.turns)
        wanted = figures.parse_report_request(q)
        if wanted:  # no model call: the report is plain code
            try:
                gid = figures.build_report(wanted, store)
                answer, gids = f"Created Graphic {gid}, a report built from graphics {wanted}.", [gid]
            except KeyError as e:
                answer, gids = f"{e.args[0]} Existing graphics: {sorted(store)}.", []
            out = {"answer": answer, "trace": []}
        else:
            with st.spinner("Thinking..."):
                out = agent.ask(q, history=st.session_state.turns)
            gids = figures.register_from_trace(out["trace"], store)
        with st.chat_message("assistant"):
            st.write(out["answer"])
            show_graphics(gids, f"h{n}")
            show_trace(out["trace"])
        st.session_state.turns.append({"question": q, "answer": out["answer"],
                                       "trace": out["trace"], "graphics": gids})


with dash_tab:
    source = st.radio("Start from", ["Template", "Describe it", "Saved"], horizontal=True)
    spec = None
    if source == "Template":
        c1, c2, c3, c4 = st.columns(4)
        name = c1.selectbox("Template", dashboard.TEMPLATES)
        svc_choice = c2.selectbox("Service", dashboard.services(),
                                  disabled=name not in ("service_deep_dive", "focus_report"))
        focus = None
        if name == "focus_report":
            dims = analysis.group_dimensions()
            dim = c3.selectbox("Focus on", dims, index=dims.index("faculty"),
                               format_func=lambda d: dashboard.DIMENSION_LABELS.get(d, d))
            values = analysis.focus_values(analysis.connect(), dim)
            default = values.index("Arts") if "Arts" in values else 0
            focus = {"dimension": dim, "value": c4.selectbox("Group", values, index=default)}
        spec = dashboard.template(name, svc_choice, focus)
    elif source == "Describe it":
        request = st.text_area("What should the dashboard show?",
                               "How do Arts students experience SHW services compared with everyone?")
        if st.button("Plan dashboard"):
            with st.spinner("Planning..."):
                planned, errs = dashboard.plan(request)
            if errs:
                st.error("The model's plan still had problems:\n\n" + "\n".join(f"- {e}" for e in errs))
            else:
                st.session_state.dash_spec = planned
        spec = st.session_state.get("dash_spec")
    else:
        names = dashboard.saved()
        if names:
            spec = dashboard.load(st.selectbox("Saved dashboard", names))
        else:
            st.info("No saved dashboards yet.")

    if spec:
        panels = dashboard.build(spec)
        page = charts.render_page(spec, panels, dashboard.page_meta(spec))
        components.html(page, height=charts.estimate_height(panels), scrolling=True)
        c1, c2, c3 = st.columns([2, 1, 1])
        save_name = c1.text_input("Save as", spec.get("title", "dashboard"))
        if c2.button("Save"):
            st.success(f"Saved to {dashboard.save(spec, save_name)}")
        c3.download_button("Download as HTML", page, file_name=f"{save_name}.html", mime="text/html")
        with st.expander("Dashboard spec (JSON)"):
            st.json(spec)

with equity_tab:
    con = analysis.connect()
    svc = st.selectbox("Service", ["(all)"] + analysis.service_summary(con)["service"].tolist())
    tests, flags, thin = analysis.equity_scan(con, None if svc == "(all)" else svc)
    c1, c2, c3 = st.columns(3)
    c1.metric("Comparisons run", len(tests))
    c2.metric("Flagged gaps", len(flags))
    c3.metric("Groups too small to compare", len(thin))
    st.dataframe(flags if len(flags) else tests.nsmallest(10, "q"), width="stretch")
    st.caption("Flag = group differs from everyone else by >= 10 points in % agree, "
               "after false-discovery-rate correction. Unflagged rows show the closest calls.")

with comments_tab:
    st.json(tools.comment_overview(None if svc == "(all)" else svc))
    term = st.text_input("Search comments")
    if term:
        st.dataframe(tools.search_comments(term, None if svc == "(all)" else svc)["rows"])

with quality_tab:
    st.dataframe(analysis.connect().execute("SELECT * FROM data_quality").df(), width="stretch")

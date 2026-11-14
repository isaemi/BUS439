"""Streamlit front end.  Run:  streamlit run app.py"""
import streamlit as st
import streamlit.components.v1 as components

from shw_analytics import (agent, analysis, briefing, charts, config, dashboard, figures, followups,
                           provenance, theme, tools)

st.set_page_config(page_title="Student Feedback Assistant | UBC Student Health and Wellbeing", layout="wide")
st.markdown(theme.app_css(), unsafe_allow_html=True)
st.markdown(theme.header_html(
    "Ask about students' experience of our services. Answers come from de-identified survey and "
    "appointment data, analysed on UBC infrastructure. Small groups are hidden to protect privacy."),
    unsafe_allow_html=True)

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

# The conversation comes first: asking questions is the main way to use the tool.
ask_tab, brief_tab, dash_tab, comments_tab = st.tabs(["Ask a question", "Briefing", "Dashboards", "Comments"])


with brief_tab:
    # What changed and what is worth a look, without anyone having to ask. Built from tested
    # analyses only, with no model involved, so the same data always gives the same briefing.
    b = briefing.cached()
    if b is None:
        st.markdown("#### This term at a glance")
        st.write("The briefing runs a fixed set of tested analyses across services, checks the findings again "
                 "together, and shows the few that hold up, along with what stayed the same and what could "
                 "not be checked.")
        if st.button("Build this term's briefing", type="primary"):
            with st.spinner("Running the analyses and checking them together (about half a minute)..."):
                b = briefing.build()
    if b:
        changes = briefing.compare(b, briefing.previous(b))
        briefing.save(b)
        st.markdown("#### This term at a glance")
        st.caption(
            f"Appointment records to {b.get('data_through', 'n/a')}"
            + (f", comparing {b['compared']}" if b.get("compared") else "")
            + (f". Survey responses from {b['survey_period']}" if b.get("survey_period") else "")
            + f". {b['tests_run']} checks were run. Each analysis was tested on its own, and the findings that "
              "passed were checked again together; only those that held up are shown."
            + (f" Compared with the briefing from {changes['since']}." if changes.get("since") else "")
            + (" **The appointment records are synthetic test data, not real.**" if b.get("synthetic") else ""))
        if not b["findings"]:
            st.info("Nothing stood out after checking everything together. See what stayed the same below.")
        for n, f in enumerate(b["findings"], 1):
            tag = changes["status"].get(f["id"])
            tag_html = f'<span class="shw-tag{" new" if tag == "new" else ""}">{tag.capitalize()}</span>' if tag else ""
            with st.container(border=True):
                st.markdown(f'<div class="shw-num">FINDING {n}{tag_html}</div>'
                            f'<div class="shw-gtitle">{f["headline"]}</div>'
                            f'<div class="shw-detail">{f["detail"]}</div>', unsafe_allow_html=True)
                spec = briefing.graphic(f)
                if spec:
                    st.plotly_chart(spec.fig, key=f"brief_{n}", width="stretch", theme=None,
                                    config={"displayModeBar": False})
                st.markdown(f'<div class="shw-small">{f["tested"]}</div>'
                            f'<div class="shw-small"><b>Related question:</b> {f["next"]}</div>',
                            unsafe_allow_html=True)
        for title, items in (("Also noted", [f["headline"] for f in b["also"]]),
                             ("No longer showing since last time", changes.get("gone", [])),
                             ("What stayed the same", b["unchanged"]),
                             ("What could not be checked", b["not_checked"])):
            if items:
                with st.expander(f"{title} ({len(items)})", expanded=title == "What stayed the same"):
                    st.markdown("\n".join(f"- {i}" for i in items))
        if st.session_state.get("brief_html_key") != b["data_key"]:
            st.session_state.brief_html = briefing.to_html(b, changes)
            st.session_state.brief_html_key = b["data_key"]
        st.download_button("Download the briefing (HTML)", st.session_state.brief_html,
                           file_name=f"shw_briefing_{b.get('data_through', 'latest')}.html", mime="text/html")


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

    def show_trace(trace, key):
        """Where the answer came from, in plain words. The raw steps stay available for
        the team behind a switch, but staff never have to read code or JSON."""
        with st.expander("How this answer was built"):
            cards = provenance.evidence_cards(trace)
            st.markdown(provenance.render_markdown(cards))
            steps = [s for s in trace if isinstance(s, (list, tuple)) and len(s) == 3]
            if steps and st.toggle("Show technical details", key=f"tech_{key}"):
                for name, args, res in steps:
                    st.markdown(f"**{name}** `{args}`")
                    st.json(res, expanded=False)

    def show_graphic(gid, key_prefix):
        g = store[gid]
        if g.kind == "chart":
            with st.container(border=True):
                st.markdown(f'<div class="shw-gtitle">{g.title}</div><div class="shw-gscope">{g.scope}</div>',
                            unsafe_allow_html=True)
                st.plotly_chart(g.fig, key=f"{key_prefix}_{gid}", width="stretch", theme=None,
                                config={"displayModeBar": False})
                st.caption(f"**Graphic {gid}.** {g.caption}")
        else:
            components.html(g.html, height=700, scrolling=True)
            st.download_button(f"Download Graphic {gid} (report)", g.html,
                               file_name=f"report_graphic_{gid}.html", mime="text/html",
                               key=f"{key_prefix}_dl_{gid}")

    def show_graphics(ids, key_prefix):
        """The most useful graphics first (tested findings); the rest one click away."""
        ids = [i for i in ids if i in store]
        first, more = figures.split_for_display(ids, store)
        for gid in first:
            show_graphic(gid, key_prefix)
        if more:
            with st.expander(f"More graphics from this answer ({len(more)})"):
                for gid in more:
                    show_graphic(gid, key_prefix)

    for n, t in enumerate(st.session_state.turns):
        st.chat_message("user").write(t["question"])
        with st.chat_message("assistant"):
            st.write(t["answer"])
            show_graphics(t.get("graphics", []), f"h{n}")
            show_trace(t["trace"], f"h{n}")

    if st.session_state.turns:      # related questions under the latest answer, side by side
        last = st.session_state.turns[-1]
        # One step deeper on what was just found, and (once a briefing exists) a tested finding
        # about the same service that has not come up yet.
        if "ideas" not in last:        # worked out once per answer: some suggestions are checked before they are shown
            last["ideas"] = followups.suggest_detail(
                last["trace"], asked=[t["question"] for t in st.session_state.turns],
                history=[t["trace"] for t in st.session_state.turns[:-1]], limit=2)
        ideas = last["ideas"]
        if ideas:
            st.caption("Related questions")
            for col, (i, idea) in zip(st.columns(len(ideas)), enumerate(ideas)):
                if col.button(idea["text"], key=f"fu{len(st.session_state.turns)}_{i}", width="stretch",
                              help=idea["why"] or None):
                    st.session_state.pending_q = idea["text"]
                    st.rerun()

    if st.session_state.turns:
        st.caption(f"Follow-ups use the last {agent.MAX_TURNS} questions as context. "
                   "Start a new conversation to change topic. "
                   "To combine graphics, type e.g. \"make a report from graphics 1, 3 and 8\".")

    q = st.chat_input("e.g. How does counselling compare for first-generation students?")
    q = q or st.session_state.pop("pending_q", None)
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
            gids = figures.register_from_trace(out["trace"], store, answer=out["answer"])
        st.session_state.turns.append({"question": q, "answer": out["answer"],
                                       "trace": out["trace"], "graphics": gids})
        st.rerun()      # redraw from history, so the new answer gets its suggestions and stable controls


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

with comments_tab:
    # What students wrote, for staff to read: themes and tone once comments are labelled,
    # and a search that always works. Comments were scrubbed of names, emails, phone and
    # student numbers when the data was loaded.
    con = analysis.connect()
    counts = con.execute("SELECT service, COUNT(*) AS n FROM comments GROUP BY 1 ORDER BY n DESC").df()
    if counts.empty:
        st.info("There are no written comments in the survey data yet.")
    else:
        names = {"(all)": "All services", **{v: figures._service({"service": v}, "service") for v in counts["service"]}}
        svc = st.selectbox("Service", list(names), format_func=names.get, key="comments_service")
        service = None if svc == "(all)" else svc
        total = int(counts["n"].sum() if service is None else counts.loc[counts["service"] == svc, "n"].iloc[0])
        st.caption(f"{total:,} written comments from the survey. Names, emails, phone numbers and student "
                   "numbers were removed when the data was loaded.")

        overview = tools.comment_overview(service)
        if "error" in overview:
            st.info("Comments have not been sorted into themes yet, so the summary below is not available. "
                    "You can still search what students wrote.")
            with st.expander("For the project team"):
                st.markdown("Sorting comments into themes and tone uses the UBC-hosted model and takes a few "
                            "minutes. From the project folder run:")
                st.code("python -m shw_analytics.comments --label\npython -m shw_analytics.comments --embed",
                        language="bash")
                st.caption("The first command adds themes and tone. The second lets search find comments by "
                           "meaning, not only by exact words.")
        else:
            themes_fig, tone_fig = figures.comment_charts(overview)
            c1, c2 = st.columns([3, 2])
            with c1, st.container(border=True):
                st.markdown('<div class="shw-gtitle">What students write about</div>'
                            '<div class="shw-gscope">Number of comments mentioning each theme; a comment can '
                            'have up to three</div>', unsafe_allow_html=True)
                st.plotly_chart(themes_fig, width="stretch", theme=None, config={"displayModeBar": False},
                                key="comment_themes")
            with c2, st.container(border=True):
                st.markdown('<div class="shw-gtitle">Tone of comments</div>'
                            '<div class="shw-gscope">Sorted automatically, so individual comments can be '
                            'mislabelled</div>', unsafe_allow_html=True)
                st.plotly_chart(tone_fig, width="stretch", theme=None, config={"displayModeBar": False},
                                key="comment_tone")

        term = st.text_input("Search what students wrote", placeholder="e.g. booking, waiting, felt heard")
        if term:
            found = tools.search_comments(term, service, k=15)
            rows = found.get("rows", [])
            by_meaning = str(found.get("search_mode", "")).startswith("semantic")
            if not rows:
                st.info("No comments matched. Try a different or shorter word.")
            else:
                st.caption(f"{len(rows)} comments " + ("closest in meaning to" if by_meaning else "containing words from")
                           + f" “{term}”. Read them as examples, not as a count of how common the issue is.")
                for r in rows:
                    with st.container(border=True):
                        st.markdown(f"“{r['text']}”")
                        st.caption(names.get(r.get("service"), str(r.get("service", ""))))

st.markdown(theme.footer_html(), unsafe_allow_html=True)

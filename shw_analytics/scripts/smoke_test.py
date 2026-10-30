"""End-to-end check that runs WITHOUT the gateway (fake model).

    python scripts/smoke_test.py path/to/qualtrics_export.csv
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shw_analytics import config  # noqa: E402

tmp = Path(tempfile.mkdtemp())
config.DB_PATH = tmp / "shw.duckdb"

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
from shw_analytics import agent, analysis, charts, comments, dashboard, ingest, llm, tools  # noqa: E402

csv = sys.argv[1]
ok = lambda cond, msg: print(("PASS " if cond else "FAIL ") + msg) or cond
results = []

# Ingest
df, texts = ingest.read_qualtrics(csv)
tables = ingest.build_tables(df, texts, ingest.load_dictionary())
ingest.write_db(tables)
results.append(ok(len(tables["ratings"]) > 0, f"ingest: {len(tables['ratings'])} ratings, {len(tables['comments'])} comments"))
results.append(ok("IPAddress" not in tables["responses"].columns, "PII columns dropped"))
results.append(ok(tables["ratings"]["score"].between(1, 5).all(), "all scores in 1-5 after recode"))

# Automated exports: overlapping daily drops combine without double-counting, a
# joined multi-select column is parsed, and an answer-text export is refused.
raw = pd.read_csv(csv, header=None, dtype=object)
head, body = raw.iloc[:3], raw.iloc[3:]
drops = tmp / "drops"
drops.mkdir()
pd.concat([head, body.iloc[:400]]).to_csv(drops / "a.csv", header=False, index=False)
pd.concat([head, body.iloc[380:]]).to_csv(drops / "b.csv", header=False, index=False)
combined, _ = ingest.read_many(ingest.resolve_paths(drops))
results.append(ok(len(combined) == len(body), f"overlapping export drops combined: {len(combined)} responses"))
text = raw.copy()
labels = {"1": "Strongly disagree", "2": "Disagree", "3": "Neutral", "4": "Agree", "5": "Strongly agree"}
for i, c in enumerate(raw.iloc[0]):
    if str(c).startswith(("CS1_", "MED1_", "WA1_")):
        text.iloc[3:, i] = text.iloc[3:, i].map(lambda v: labels.get(v, v))
text.to_csv(tmp / "text.csv", header=False, index=False)
try:
    ingest.check_format(ingest.read_qualtrics(tmp / "text.csv")[0], ingest.load_dictionary())
    refused = False
except ingest.ExportFormatError:
    refused = True
results.append(ok(refused, "answer-text export refused with an explanation"))

# SFTP sync with a fake server: prefix filter, unzip, and a second run copies nothing.
import io as _io, zipfile as _zip  # noqa: E401
from shw_analytics import sftp_sync  # noqa: E402
_raw = open(csv, "rb").read()
_zb = _io.BytesIO()
with _zip.ZipFile(_zb, "w") as _z:
    _z.writestr("x.csv", _raw)
_remote = {"shw_survey_1.csv": _raw, "shw_survey_2.zip": _zb.getvalue(), "other.csv": b"x"}
class _FakeSFTP:
    def listdir(self, d): return list(_remote)
    def get(self, r, l): open(l, "wb").write(_remote[r.split("/")[-1]])
_out = tmp / "synced"
first = sftp_sync.sync(_FakeSFTP(), "/drop", _out, prefix="shw_survey")
second = sftp_sync.sync(_FakeSFTP(), "/drop", _out, prefix="shw_survey")
results.append(ok(len(first) == 2 and not second and len(list(_out.glob("*.csv"))) == 2,
                  "SFTP sync copies new exports once, unzips, skips other files"))

# SQL guard
results.append(ok("error" in tools.run_sql("SELECT * FROM ratings"), "guard: row-level output blocked"))
results.append(ok("error" in tools.run_sql("SELECT text FROM comments"), "guard: comments unreachable from SQL"))
results.append(ok("error" in tools.run_sql("DROP TABLE ratings"), "guard: writes blocked"))
results.append(ok("error" in tools.run_sql("SELECT * FROM read_csv('/etc/passwd')"), "guard: file access blocked"))
r = tools.run_sql("SELECT gender, COUNT(DISTINCT response_id) AS n FROM ratings_demo "
                  "WHERE intl_or_domestic = 'international' GROUP BY gender")
results.append(ok("does not exist" in r.get("error", ""), "guard: filter on non-existent category value caught"))
r = tools.run_sql("SELECT service, COUNT(DISTINCT response_id) AS n FROM ratings GROUP BY service")
results.append(ok("rows" in r and r["row_count"] > 0, "guard: valid aggregate allowed"))

# The equity scan should flag nothing on random test data.
con = duckdb.connect(str(config.DB_PATH))
t, flags, thin = analysis.equity_scan(con)
results.append(ok(len(flags) == 0, f"equity scan on random data: {len(t)} tests, {len(flags)} flags (expect 0)"))

# Plant a real gap and make sure it is caught.
con.execute("""UPDATE ratings SET score = GREATEST(1, score - 2), agree = (GREATEST(1, score - 2) >= 4)
               WHERE service = 'medical' AND response_id IN
                 (SELECT response_id FROM responses WHERE indigenous = '1')""")
t, flags, thin = analysis.equity_scan(con, "medical")
hit = flags[(flags["dimension"] == "indigenous") & (flags["group"] == "1")]
results.append(ok(len(hit) == 1, f"planted gap detected: {hit[['gap_pts','q']].to_dict('records')}"))
con.close()

# Agent loop in native tool-calling mode, with a scripted fake model.
script = iter([
    NS(content="", tool_calls=[NS(id="c1", function=NS(name="service_summary", arguments="{}"))]),
    NS(content="", tool_calls=[NS(id="c2", function=NS(name="equity_flags", arguments='{"service":"medical"}'))]),
    NS(content="Counselling has the lowest % agree; medical shows one flagged gap.", tool_calls=None),
])
out = agent.ask("lowest service?", chat_fn=lambda m, tools=None: next(script))
results.append(ok(out["mode"] == "native" and len(out["trace"]) == 2, f"agent native: {[x[0] for x in out['trace']]}"))

# Agent loop in JSON fallback mode.
config.TOOL_MODE = "json"
script = iter([NS(content='```json\n{"tool": "item_summary", "args": {"service": "counselling"}}\n```'),
               NS(content='<think>ok</think>{"answer": "Weakest item is booking."}')])
out = agent.ask("weakest counselling item?", chat_fn=lambda m, tools=None: next(script))
results.append(ok(out["mode"] == "json" and out["answer"].startswith("Weakest"), "agent json fallback + think-strip"))

# Invented numbers are caught: the model answers without tools, then with made-up figures.
config.TOOL_MODE = "native"
script = iter([NS(content="Medical is 3.8 (62% agree).", tool_calls=None),
               NS(content="", tool_calls=[NS(id="w", function=NS(name="service_summary", arguments="{}"))]),
               NS(content="Medical is 3.8 (62% agree).", tool_calls=None),
               NS(content="Medical is 3.8 (62% agree).", tool_calls=None)])
out = agent.ask("medical?", chat_fn=lambda m, tools=None: next(script))
results.append(ok(len(out["trace"]) == 1 and out["ungrounded"] == ["3.8", "62"],
                  f"no-tool nudge + invented numbers flagged: {out['ungrounded']}"))

# Empty replies (out of tokens while thinking) are retried, then explained.
script = iter([NS(content="", tool_calls=None, finish_reason="length"),
               NS(content="", tool_calls=[NS(id="e", function=NS(name="service_summary", arguments="{}"))]),
               NS(content="", tool_calls=None, finish_reason="length"),
               NS(content="", tool_calls=None, finish_reason="length")])
out = agent.ask("combo?", chat_fn=lambda m, tools=None: next(script))
results.append(ok(out["answer"].startswith("The model returned an empty reply"),
                  f"empty replies handled: {[t[0] for t in out['trace']]}"))

# Handing work back is caught: the model suggests a tool instead of running it, is told
# to run it, then answers in plain words. Mentions of testing aren't treated as claims.
config.TOOL_MODE = "native"
script = iter([NS(content="", tool_calls=[NS(id="h0", function=NS(name="service_summary",
                                                               arguments='{"service": "counselling"}'))]),
               NS(content="Use wait_time_vs_rating to see if waits matter.", tool_calls=None),
               NS(content="", tool_calls=[NS(id="h1", function=NS(name="wait_time_vs_rating",
                                                               arguments='{"service": "counselling"}'))]),
               NS(content="Wait times in counselling show no clear link with ratings.", tool_calls=None)])
out = agent.ask("Do waits matter for counselling?", chat_fn=lambda m, tools=None: next(script))
results.append(ok([t[0] for t in out["trace"]] == ["service_summary", "wait_time_vs_rating"]
                  and not agent.mentions_internals(out["answer"]),
                  "handing work back to the user is caught and the tool is run"))
results.append(ok(not agent._claims_significance("Running checks to identify significant gaps.")
                  and agent._claims_significance("The gap is not statistically significant."),
                  "significance claims told apart from mentions of testing"))
results.append(ok("not connected" in agent._system(False), "data coverage note reaches the model"))

# Follow-up questions: the second turn sees the first question and its data, and may
# quote the first turn's figure without being flagged.
config.TOOL_MODE = "native"
med = tools.service_summary("medical")["rows"][0]          # the planted gap above changed
cou = tools.service_summary("counselling")["rows"][0]      # these, so read them live
turn1 = iter([NS(content="", tool_calls=[NS(id="t1", function=NS(name="service_summary",
                                                              arguments='{"service": "medical"}'))]),
              NS(content=f"Medical: {med['pct_agree']}% agree (n={med['n']}).", tool_calls=None)])
out1 = agent.ask("How is Medical doing?", chat_fn=lambda m, tools=None: next(turn1))
history = [{"question": "How is Medical doing?", "answer": out1["answer"], "trace": out1["trace"]}]
seen_msgs = []
turn2 = iter([NS(content="", tool_calls=[NS(id="t2", function=NS(name="service_summary",
                                                              arguments='{"service": "counselling"}'))]),
              NS(content=f"Counselling: {cou['pct_agree']}% agree (n={cou['n']}), "
                             f"versus Medical's {med['pct_agree']}%.", tool_calls=None)])
def chat2(m, tools=None):
    seen_msgs.append(m)
    return next(turn2)
out2 = agent.ask("What about counselling?", history=history, chat_fn=chat2)
saw_history = any("How is Medical doing?" in str(x.get("content")) for x in seen_msgs[0])
results.append(ok(saw_history and not out2["ungrounded"],
                  f"follow-up sees earlier turn and may quote it (flags: {out2['ungrounded']})"))

# Dashboards: every template builds and renders; focus groups work; bad specs are
# rejected; the planner fixes its spec after validation feedback.
built = 0
for name in dashboard.TEMPLATES:
    spec = dashboard.template(name, "medical")
    panels = dashboard.build(spec)
    built += sum(not p.error for p in panels)
    page = charts.render_page(spec, panels, dashboard.page_meta(spec))
    assert "vegaEmbed" in page and "<script>" in page
results.append(ok(built >= 25, f"dashboard templates build and render: {built} panels"))
focus_panels = dashboard.build(dashboard.template("focus_report", "counselling",
                                                  {"dimension": "faculty", "value": "Arts"}))
two_series = [p for p in focus_panels if not p.df.empty and p.df["series"].nunique() == 2]
results.append(ok(len(two_series) >= 3, f"focus group compared with everyone in {len(two_series)} panels"))
bad = dashboard.validate({"focus": {"dimension": "faculty", "value": "Dance"},
                          "panels": [{"type": "pie"}, {"type": "wait_time", "params": {"service": "naloxone"}}]})
results.append(ok(len(bad) == 3, "dashboard: unknown focus value, panel type and parameters rejected"))
plans = iter([{"title": "x", "panels": [{"type": "group_comparison", "params": {"dimension": "major"}}]},
              {"title": "x", "focus": {"dimension": "faculty", "value": "Arts"},
               "panels": [{"type": "kpis"}, {"type": "service_comparison"}]}])
spec, errs = dashboard.plan("Arts students", chat_json_fn=lambda system, user: next(plans))
results.append(ok(not errs and len(dashboard.build(spec)) == 2, "dashboard planner corrects itself after feedback"))

# Comment labelling and the validity check, with a fake labeller.
llm.chat_json = lambda system, user, model=None: [
    {"id": o["id"], "sentiment": "negative" if o["id"] % 2 else "positive", "themes": ["wait_time"], "actionable": True}
    for o in json.loads(user)]
con = duckdb.connect(str(config.DB_PATH))
comments.label(con, limit=60)
v = comments.validity_check(con)
results.append(ok(con.execute("SELECT COUNT(*) FROM comment_labels").fetchone()[0] == 60, "comment labelling pipeline"))
print(v.to_string(index=False))
con.close()

shutil.rmtree(tmp)
print(f"\n{sum(results)}/{len(results)} checks passed")

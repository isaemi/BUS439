# SHW Feedback Analytics

A conversational analytics tool for UBC Student Health & Wellbeing (SHW). Staff
ask questions about the student feedback survey in plain English, follow up on
the answers, and build dashboards, without writing queries or exporting
spreadsheets. Answers come from Qwen, a language model running on UBC's own
servers through the LTIC AI Gateway, so survey data never leaves UBC.

Built by BUS 439 team BFPSST for SHW.

---

## What staff can do with it

| Need | What the tool does | Example |
|---|---|---|
| See what changed without asking | The **Briefing** tab runs a fixed set of tested analyses across services, checks the findings again together, and shows the few that hold up, plus what stayed the same and what could not be checked | Open the Briefing tab, or ask "What should we know this term?" |
| Go one step further | Under each answer, up to two **related questions**: the next step on what was just found (test it, break it down, see who it affects, what happens next, what students say), and a tested finding from the briefing that has not come up yet | "Is the October rise in counselling from more students, or more visits each?" |
| See it | Each analysis draws its own chart in the chat, titled with the finding; charts can be combined into a report | "make a report from graphics 1, 3 and 8" |
| Find what matters most | Which survey questions are most closely linked to overall satisfaction, and score low | "What matters most to students in counselling?" |
| Get a quick answer | Ask in plain English; the answer shows the numbers and how they were found | "Which service has the lowest % agree?" |
| Dig into a finding | Ask follow-ups; the tool keeps the context and changes only what you change | "What about counselling?" → "And for international students?" → "Break that down by faculty" |
| Check for inequities | An automatic scan compares every student group with everyone else and flags real gaps, instead of toggling demographic filters by hand | "Are there equity gaps in counselling?" |
| Report to others | Dashboards in the SHW style, from a template, a plain-English request, or a saved layout; download as one HTML file | "Faculty of Arts compared with all respondents" |
| Explain a difference | Breaks a gap in service use into its parts (how many students come in, visits per student, days in care, services used, no-shows), each tested | "Why do graduate students use counselling differently?" |
| Spot changes | Appointment trends by month and term, with the same month last year and real changes flagged | "Are counselling appointments up on last year?" |
| Check who the feedback represents | Compares the survey's respondents with the students who actually visit, by faculty | "Does survey feedback reflect who we see?" |
| Understand comments | Sentiment, themes and search across open-text feedback | "What are students saying about booking?" |
| Trust the data | A data quality report lists problems in each Qualtrics export | IP addresses collected, a question copied into the wrong block |

---

## How it works

```
 Qualtrics export (CSV) ──► data/exports/ ─┐
 CHR appointment export ──► data/chr_exports/ ┼─► ingest ──► data/shw.duckdb ──► data quality report
   (upload in the app sidebar)               ─┘   (IDs anonymised, names removed)
                                    │
                         tools (fixed queries + guarded SQL)
                                    │  aggregates only
                                    ▼
             LTIC AI Gateway ──► Qwen (on-prem at UBC ARC)
                                    │
                          Streamlit app: Ask a question · Briefing ·
                          Dashboards · Comments
```

Four principles shape the design:

1. **The model explains; it never calculates.** Qwen chooses a tool, the tool
   queries the database, and Qwen explains the result. Every number comes from
   code, so the same question gives the same figures every time.
2. **Data stays at UBC.** The model only sees aggregates and scrubbed comment
   snippets, and with the on-prem Qwen models nothing leaves UBC infrastructure.
3. **Safety lives in code, not in the prompt.** Privacy rules, read-only access
   and answer checks are enforced by the program, so they hold even when the
   model makes a mistake.
4. **Every answer shows its working.** "How this answer was built" lists each
   tool call and result, so staff can check anything they plan to act on.

---

## Why the answers can be trusted

| Safeguard | What it prevents |
|---|---|
| Groups under 5 respondents are hidden, and the answer says which | Identifying individuals in small groups |
| Read-only database, one SELECT at a time, no individual rows, no file access | Leaking or changing data through model-written SQL |
| Filters on values that don't exist are refused, with the valid codes | Confident "no students found" answers caused by guessed labels |
| Every number in an answer is checked against the tool results | Invented figures (shown with a ⚠️ warning if they slip through) |
| Significance claims without a test get a warning | Calling random variation a "real" difference |
| Equity flags need a 10-point gap, 10+ people per group and a multiple-comparison correction | False alarms from running hundreds of comparisons |
| Answers that tell staff to "run a tool" or mention internal names are sent back | Work being handed back to the user |
| A router reads each question (services, groups, "why", "last year") and suggests a starting tool with its arguments; it appears as "suggested" in "How this answer was built" | Small models picking the wrong tool, or drifting to an unrelated question |
| Answers that don't mention the question's subject are sent back once | Answering a different question (e.g. wait times when asked about graduate students) |
| Tools accept common spelling and case variations, and a wrong argument returns the right parameters and tool | Wasted retries on "counseling" vs "counselling" or "graduate" vs "Graduate" |
| A data coverage note tells the model what the data can and can't answer | Guessing when appointment or capacity data is needed |
| Empty replies, answers with no tool call and repeated calls are retried | Blank or ungrounded answers |

---

## Getting started

Requires Python 3.10 or newer.

```bash
cd shw_analytics
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and add the LTIC API key. Then start the app and upload a Qualtrics
export from the sidebar:

```bash
streamlit run app.py
```

Or load the data from the command line first:

```bash
mkdir -p data/exports
cp ~/Downloads/<your export>.csv data/exports/
python -m shw_analytics.ingest data/exports
```

### Trying it with synthetic data

No real exports are needed to try the app. These two scripts write a synthetic
appointment export and a synthetic survey (with realistic comments and answer
labels) into `data/`, with known patterns planted so the analyses have something
true to find:

```bash
python scripts/make_synthetic_chr.py
python scripts/make_synthetic_survey.py
python -m shw_analytics.ingest data/exports
streamlit run app.py
```

Optional, and uses the gateway (a few cents): sort comments into themes and make
them searchable by meaning. Rebuilding the data clears these, so run them again
after `ingest`:

```bash
python -m shw_analytics.comments --label
python -m shw_analytics.comments --embed
```

Every answer from synthetic appointment data says so. `python -m shw_analytics.usage`
shows how many gateway requests this computer has made and what they cost.

Check that everything works (no model needed):

```bash
python scripts/smoke_test.py data/exports/<your export>.csv
for t in tests/test_*.py; do printf "%-30s" $t; python $t | tail -1; done
```

On the synthetic survey, two smoke checks fail by design (they assume a survey
with no differences between groups, and the synthetic one has gaps planted).

In each new terminal, run `source .venv/bin/activate` first. Your prompt should
start with `(.venv)`.

### Testing without the API key

Run Qwen on your own computer with [Ollama](https://ollama.com) (free; install
the prebuilt app):

```bash
launchctl setenv OLLAMA_CONTEXT_LENGTH 16384
ollama pull qwen3:8b
ollama pull qwen3-embedding:0.6b
```

Restart Ollama, then use the Ollama block in `.env.example`. `ollama ps` should
show CONTEXT 16384 and 100% GPU. The local 8B model is slower (30–60 seconds per
question) and makes more mistakes than the gateway's 27B, so use it to test the
pipeline, not to judge answer quality.

---

## Files and what they do

### Settings

| File | What it does | Why it matters |
|---|---|---|
| `.env` | Your private settings: API key, model names, SFTP login. Never committed. | Switch between local testing and the UBC gateway without touching code |
| `.env.example` | Template for `.env`, with every option explained | New team members set up in minutes |
| `dictionary.yaml` | What every survey code means: services, scales, demographics, plus the data coverage note | The single place to update when the survey changes; the model's answers are only as good as this file |
| `requirements.txt` | Python packages to install | One command sets up any machine |

### Getting data in

| File | What it does | Why it matters |
|---|---|---|
| `shw_analytics/ingest.py` | Turns Qualtrics exports into clean database tables; combines many exports and counts each response once; writes the data quality report | Staff never clean spreadsheets; problems in the survey are caught early |
| `shw_analytics/chr.py` | Reads TELUS CHR appointment exports: skips blank and group-template rows, splits the service name into service line, visit type and mode, and anonymises students, appointments and providers with salted hashes | Appointment data becomes usable without names or real IDs ever reaching the database |
| `shw_analytics/sftp_sync.py` | Copies Qualtrics' scheduled exports from an SFTP server | Hands-off nightly updates once UBC IT provides a server |
| `shw_analytics/qualtrics_api.py` | Pulls exports through the Qualtrics API | An alternative if API access is ever granted |
| `shw_analytics/labels.py` | Fills in answer labels (e.g. code 1 = "International") from a choice-text export | Answers say "international students" instead of "code 1" |
| `shw_analytics/privacy.py` | Removes emails, phone numbers and names from comments; hides small groups | Protects students without staff having to think about it |

### Analysis and AI

| File | What it does | Why it matters |
|---|---|---|
| `shw_analytics/analysis.py` | All statistics: service and item summaries, group comparisons, the equity scan, wait time vs ratings, focus-group comparisons | Consistent, reproducible numbers behind every chat answer and dashboard |
| `shw_analytics/schema.py` | The database schema: what every table and column means, keys and joins. Written into the database as comments at each rebuild; `--markdown` regenerates `docs/SCHEMA.md` | The model writes better SQL because it knows how tables relate; new team members can read the schema in one page |
| `shw_analytics/tools.py` | The only actions Qwen can take, plus the SQL guard | Qwen can answer flexible questions without ever touching raw data |
| `shw_analytics/agent.py` | The conversation: question → tools → answer, with follow-ups, number checks and retries | Turns quick answers into a back-and-forth staff can drill into |
| `shw_analytics/llm.py` | Connects to the LTIC gateway, or Ollama when testing | Swapping models is a settings change, not a code change |
| `shw_analytics/comments.py` | Sentiment, themes, search and a check of comments against ratings | Makes open-text feedback usable at scale |
| `shw_analytics/config.py` | Reads settings from `.env`; holds privacy and equity thresholds | Policy decisions (e.g. the minimum group size) live in one place |

### Second-level analysis, briefing and charts

| File | What it does | Why it matters |
|---|---|---|
| `shw_analytics/stats.py` | Shared statistics: re-sampling ranges with each student counted once, correction for running many tests, splitting a change into "who came" and "change within groups" | One tested implementation behind every finding |
| `shw_analytics/insights.py` | `what_changed`: the same dates a year apart, with the change in students seen tested and each change split into its parts | Tells a real change from a shift in who came |
| `shw_analytics/care_patterns.py` | What happens after a first visit, which appointments are missed more often, and movement between services | Continuity and no-show questions, answered with tests |
| `shw_analytics/survey_trends.py` | `survey_changed`: survey ratings in the latest window against the same dates a year earlier, by service, question and student group; a group is listed only if it moved differently from everyone else | Tells staff when a group's experience shifts, and how big a shift the data could have missed |
| `shw_analytics/drivers.py` | Which survey questions are most closely linked to the overall rating and score low, kept only if stable | Where to look first in the survey |
| `shw_analytics/briefing.py` | The term briefing: runs the analyses above across services, checks the findings again together, ranks them, and compares with the last briefing | Findings without anyone having to ask; no model involved |
| `shw_analytics/followups.py` | The related questions under each answer, chosen by rules from what was found and what the conversation has already covered | Leads staff through the analysis an analyst would do |
| `shw_analytics/playbook.yaml`, `playbook.py` | Common question types: which analyses fit, how to read them, what not to say | Editable guidance for the model, without code |
| `shw_analytics/figures.py` | A chart for each analysis, shown in the chat and the briefing; builds the downloadable report | Charts state the finding and match the answer |
| `shw_analytics/provenance.py` | Plain-language "How this answer was built" cards | Staff can check any answer |
| `shw_analytics/plain.py` | Keeps p-values and similar jargon, and any "I re-checked" preamble, out of answers | Answers read the same way every time |
| `shw_analytics/consistency.py`, `scripts/consistency_check.py` | Asks ten questions three ways each and reports whether the same analyses ran and the same figures came back | Evidence for "different phrasings give consistent results" |
| `shw_analytics/reconcile.py`, `scripts/reconcile_chr.py` | Compares the app's appointment counts with figures typed in from the CHR dashboards, finds which way of counting the dashboard uses, and states how the app counts | Answers "how is that data being counted?" |
| `shw_analytics/usage.py` | Logs each gateway request's tokens and cost (never content) to `data/usage_log.csv` | The team can see its share of the shared key |
| `scripts/make_synthetic_survey.py` | A synthetic survey with planted patterns, realistic comments and answer labels | Try every feature before real data arrives |
| `scripts/install_*.py` | One-time patches used during development; all are already applied on this branch | Nothing to run; kept for the record |
| `tests/` | Twelve suites that plant a known effect and check it is found, and that noise is not | Run after any change |

### Dashboards

| File | What it does | Why it matters |
|---|---|---|
| `shw_analytics/dashboard.py` | Defines dashboards as short specs: panel types, templates, focus groups, validation, saving | Reusable reports that refresh with new data each term |
| `shw_analytics/charts.py` | Draws the dashboard page: KPI cards, charts, donuts, alerts | What staff see in the app is exactly what they download and share |
| `shw_analytics/theme.py` | Colours and font | Apply SHW's official design rules by editing one file |

### App and scripts

| File | What it does | Why it matters |
|---|---|---|
| `app.py` | The Streamlit app: Ask a question, Briefing, Dashboards, Comments, plus data upload in the sidebar | Everything staff use, in one place |
| `scripts/smoke_test.py` | Offline checks, from data loading to guardrails to dashboards | Confirms nothing broke after a change, without needing the model |
| `docs/SCHEMA.md` | Tables, columns, keys and relationships, generated from the live database | A quick reference for anyone writing queries or extending the tool |
| `scripts/make_synthetic_chr.py` | Builds a realistic fake CHR export with planted patterns (graduate students attend more sessions, autumn and March peaks, growth on last year) | Build and demo CHR features before real exports arrive |
| `docs/chr_sample_export.csv` | The real CHR export layout with one scrubbed row | The parser is tested against the real format |
| `scripts/update_data.sh` | The scheduled job: sync new exports, rebuild, log | Run it nightly with cron once SFTP is set up |

### Created when you run it (`data/`, never committed)

| Path | What it is |
|---|---|
| `data/chr_exports/` | CHR appointment exports. A file name containing "synthetic" marks the data as test data in every answer. |
| `data/.id_salt` | The secret used to anonymise CHR IDs (unless `SHW_ID_SALT` is set). Keep it; changing it changes every ID. |
| `data/exports/` | Every Qualtrics export received. Don't delete files here: scheduled exports only contain new responses. |
| `data/shw.duckdb` | The database the app reads, rebuilt from the exports |
| `data/dashboards/` | Saved dashboard layouts |
| `data/update.log` | History of scheduled updates |
| `data/value_labels.yaml` | Answer labels written by the synthetic survey generator. Labels in `dictionary.yaml` take priority |
| `data/briefings/` | One small file per briefing, so the next one can say what is new |
| `data/usage_log.csv` | Gateway requests made from this computer: time, model, tokens, cost |

---

## Dashboards

Dashboards follow the SHW mock-up: a header with the period and data sources, a
row of KPI cards, and a grid of chart cards.

A dashboard can have a **focus group**, such as Faculty of Arts, international
students or a disability type. Every panel then compares that group (strong blue)
with all respondents (pale blue-grey), and KPI cards show the difference
("↑ 0.2 pts vs overall").

| Template | Shows |
|---|---|
| `focus_report` | One group compared with everyone: KPIs, services, items, who responded, changes to watch |
| `service_overview` | All services side by side |
| `service_deep_dive` | One service: items, wait time, comparisons by group, flagged gaps |
| `equity_overview` | Tested gaps plus every equity dimension |

Staff can also describe a dashboard in plain English ("How do Arts students
experience SHW compared with everyone?"), turn a chat into a dashboard with one
click, and save any dashboard to re-run on next term's data. The model only
writes the dashboard spec, never chart code or queries, and every spec is checked
before it's built.

"Changes to watch" only uses red for gaps confirmed by the equity scan.
Differences between a focus group and everyone are amber and labelled
"descriptive, not tested". Bars from groups under 30 respondents are faded.

---

## CHR appointment data

The CHR "Appointments Bulk Export" adds what the survey can't show: how many
students come in, how often, for how long, and when. Each row becomes one
appointment with:

- **Service line, visit type and mode**, parsed from names like
  "Counselling (MDUP) - Follow Up (Phone/Video)".
- **Outcome**: attended, no-show or cancelled, mapped from the export's status in
  `dictionary.yaml` (only `note_signed` has been seen so far; confirm the rest).
- **Student facts**: faculty, level, program, gender and age band.
- **Anonymised keys**: the same student always gets the same key, so visits per
  student and days in care can be counted, but names and real IDs are never stored.
  Provider names become codes, and staff names are removed from locations.

What the CHR tools answer:

| Tool | Answers |
|---|---|
| `chr_summary` | Students seen, appointments, visits per student, days in care, no-show rate, overall or by group |
| `chr_trend` | Appointments by month, week or term, with the same month last year; changes of 10% or more that are unlikely to be chance are flagged |
| `explain_difference` | Why two groups differ: "Graduate students: 229 vs 996 students, each attending 4.4 visits vs 2.5 (1.8x), so total visits are 1,018 vs 2,490", plus which parts are significant |
| `survey_vs_visits` | Whether survey respondents match the students who visit, by faculty |

Until real exports arrive, `python scripts/make_synthetic_chr.py` creates a
synthetic export in `data/chr_exports/`; every answer based on it says it is
synthetic. Not in the export yet (and on SHW's list to request from Telus):
PHQ-9, medication, presenting concerns, referrals, international status
(facts 04 and 05 are empty), and time from first contact to first appointment.

## Keeping the data up to date

| Route | How | Needs |
|---|---|---|
| Upload in the app | Export from Qualtrics, drop the file in the sidebar | Nothing; works today |
| Scheduled SFTP export | Qualtrics sends new responses nightly; `scripts/update_data.sh` picks them up | An SFTP server from UBC IT that accepts Qualtrics' connections |
| Qualtrics API | `python -m shw_analytics.qualtrics_api` | API access, which is unlikely to be granted |

Every route needs the same export settings:

- CSV with **numeric values** (untick "export the text of the answer choice").
  Answer-text exports are refused with an explanation.
- **Split multi-value fields into columns** ticked.
- **Anonymize responses** turned on in the survey options, so IP addresses and
  locations are never collected.

Do one full manual export first: scheduled exports only contain responses since
the previous one. The tool combines every file and counts each response once.

---

## What the tool can answer today

| Question type | Status |
|---|---|
| Satisfaction and outcomes by service, item, group or faculty | ✅ Survey |
| Differences between student groups, tested for significance | ✅ Survey |
| Wait time and its link to ratings (Medical, Counselling) | ✅ Survey |
| Barriers students report | ⚠️ Partly: "fell short" answers and comments; students who never used a service aren't surveyed |
| Whether survey ratings changed, and for which student groups | ✅ Survey, same dates a year apart (or the window before); needs responses in both windows |
| Appointments, visits per student, days in care, no-shows, trends | ✅ CHR export (synthetic until real exports arrive) |
| Why groups differ in service use | ✅ CHR, broken into students, visits per student, days in care, services, no-shows |
| Whether survey feedback reflects who visits | ✅ Survey + CHR, by faculty |
| Whether a change since last year is real, and whether it comes from who came or from change within groups | ✅ CHR, same dates a year apart |
| What happens after a first visit; which appointments are missed more often; movement between services | ✅ CHR, with each student counted once |
| Which survey questions matter most to overall satisfaction | ✅ Survey, per service (needs about 50 respondents) |
| A ranked summary of what changed and what is worth a look | ✅ Briefing tab, or "What should we know this term?" |
| Whether two answers, or a group and an answer, are associated (chi-square) | ⚠️ Built, but on survey-sized data most 1 to 5 rating tables have cells under the privacy threshold, so the test usually cannot be released |
| PHQ-9, medication, presenting concerns (CLICC), referrals | ❌ Not in the CHR export yet |
| Demand vs clinician capacity, calls, students not offered appointments | ❌ Needs capacity and phone data |
| Linking a survey response to a student's appointments | ❌ Not possible by design; only by shared groups (faculty, term) |

When a question needs missing data, the tool says so in its first sentence,
names the missing source, and then gives the closest evidence the survey can
offer. Update the `coverage` section of `dictionary.yaml` whenever a new data
source is connected.

---

## Checking the tool against SHW's own standards

SHW's leadership named accuracy and consistency as conditions for trusting the tool. Two checks
produce evidence for each. Neither changes any data.

```bash
python scripts/consistency_check.py     # 30 questions to the model; writes data/consistency_report.md
python scripts/reconcile_chr.py         # writes data/reconcile_template.csv and prints how the app counts
python scripts/reconcile_chr.py --compare data/reconcile_template.csv   # after typing in dashboard figures
```

The consistency check costs about as much as asking 30 questions in the app. The reconciliation
uses no model: someone with a CHR dashboard open types its monthly figures into the sheet, and the
report says which way of counting the dashboard matches (booked, not cancelled, or attended) and
where the two still differ.

---

## Settings you're most likely to change

| Setting | Where | Default |
|---|---|---|
| Model and endpoint | `.env`: `LTIC_BASE_URL`, `SHW_CHAT_MODEL` | UBC gateway, `qwen3.8-27b` |
| Minimum group size shown | `.env`: `SHW_MIN_CELL` | 5 |
| Minimum group size for equity tests | `.env`: `SHW_EQUITY_MIN` | 10 |
| Which Qualtrics responses count | `.env`: `SHW_KEEP_CHANNELS` | `anonymous,test` (use `anonymous` for real data) |
| Room for each model reply | `.env`: `SHW_MAX_TOKENS` | 4000 |
| Colours and font | `shw_analytics/theme.py` | UBC brand palette; Whitney where installed, otherwise Arial |
| What each code means | `dictionary.yaml` | Items marked `CONFIRM` still need checking |

---

## Roadmap

1. **Real answer labels**: the survey definition (`.qsf`) or one choice-text
   export from Qualtrics, so real data shows group names. The synthetic survey
   already has labels.
2. **Survey changes handled safely**: move the remaining survey-specific codes
   into `dictionary.yaml`, and report on each data load what changed in the
   survey (new, removed or reworded questions, renumbered answers), so results
   are never compared across a change in meaning.
3. **No terminal for staff**: an admin page for loading data, sorting comments
   into themes and rebuilding; comment themes kept across rebuilds.
4. **Hosting**: a UBC server with UBC login, after SHW's privacy review.
5. **Pilot evaluation**: record a change and its date, and compare before and
   after against the same weeks last year and an unaffected group.
6. **More CHR fields**: PHQ-9, priority appointment, presenting concerns,
   international status, referral and booking dates.

---

## Known limitations

- Everything so far has run on synthetic data with planted patterns. That shows
  the methods find what is there; it says nothing about real students.
- The briefing tests each analysis on its own and then checks the findings that
  passed again together. It is not a single correction across every test run.
- Sorting comments into themes has to be redone after each data rebuild.
- Changes in survey ratings for small groups are hard to detect: with about 3,000 responses a
  year, a group's rating must move by roughly 14 points between two-month windows to show up.
  `survey_changed` reports this figure, and a longer window lowers it.
- The no-show rate divides by all booked appointments, cancelled ones included. Confirm with SHW
  that this matches how they define it.
- The chi-square association tests rarely return a result on survey-sized data
  (see the table above).
- Small-cell hiding doesn't prevent differencing (a total minus the visible
  groups). Fine for internal staff use; revisit before wider access.
- The comment scrubber catches emails, phone numbers, student numbers and titled
  names, not every name.
- Answer checks catch invented numbers and unsupported significance claims, but
  not every misreading (such as calling 3.0 lower than 2.9). Staff should read
  "How this answer was built" before acting on an answer.

---

## Glossary

| Term | Meaning |
|---|---|
| DuckDB | The small database file (`data/shw.duckdb`) that holds the cleaned survey data |
| LTIC AI Gateway | UBC's service for reaching language models; ours runs Qwen on UBC servers |
| Qwen | The language model that reads questions and explains results |
| Ollama | Free software for running Qwen on your own computer while testing |
| Focus group | The group a dashboard compares with everyone, e.g. Faculty of Arts |
| Equity scan | Compares each student group with everyone else, per service, and flags real gaps |
| % agree | Share of answers rated 4 or 5 on the 1–5 scale |
| CHR | TELUS Collaborative Health Record, SHW's appointment system |
| Visits per student | Attended appointments divided by the number of students seen |
| Days in care | Days between a student's first and last attended appointment |
| CLICC | Clinician Index of Client Concerns, the form counsellors use to record presenting concerns |

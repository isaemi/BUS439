# SHW Feedback Analytics

A conversational analytics tool for UBC Student Health & Wellbeing (SHW). Staff
ask questions about the student feedback survey in plain English, follow up on
the answers, and build dashboards, without writing queries or exporting
spreadsheets. Answers come from Qwen (right now Ollama), a language model running on UBC's own servers through the LTIC AI Gateway, so survey data never leaves UBC.

Built by BUS 439 team BFPSST for SHW.

---

## What staff can do with it

| Need | What the tool does | Example |
|---|---|---|
| Get a quick answer | Ask in plain English; the answer shows the numbers and how they were found | "Which service has the lowest % agree?" |
| Dig into a finding | Ask follow-ups; the tool keeps the context and changes only what you change | "What about counselling?" → "And for international students?" → "Break that down by faculty" |
| Check for inequities | An automatic scan compares every student group with everyone else and flags real gaps, instead of toggling demographic filters by hand | "Are there equity gaps in counselling?" |
| Report to others | Dashboards in the SHW style, from a template, a plain-English request, or a saved layout; download as one HTML file | "Faculty of Arts compared with all respondents" |
| Understand comments | Sentiment, themes and search across open-text feedback | "What are students saying about booking?" |
| Trust the data | A data quality report lists problems in each Qualtrics export | IP addresses collected, a question copied into the wrong block |

---

## How it works

```
 Qualtrics export (CSV)
        │   upload in the app, or a scheduled drop via SFTP
        ▼
 data/exports/ ──► ingest ──► data/shw.duckdb ──► data quality report
                                    │
                         tools (fixed queries + guarded SQL)
                                    │  aggregates only
                                    ▼
             LTIC AI Gateway ──► Qwen (on-prem at UBC ARC)
                                    │
                          Streamlit app: Ask · Dashboards ·
                          Equity flags · Comments · Data quality
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
| Every number in an answer is checked against the tool results | Invented figures (shown with a  warning if they slip through) |
| Significance claims without a test get a warning | Calling random variation a "real" difference |
| Equity flags need a 10-point gap, 10+ people per group and a multiple-comparison correction | False alarms from running hundreds of comparisons |
| Answers that tell staff to "run a tool" or mention internal names are sent back | Work being handed back to the user |
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

Check that everything works (no model needed):

```bash
python scripts/smoke_test.py data/exports/<your export>.csv
```

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
| `shw_analytics/sftp_sync.py` | Copies Qualtrics' scheduled exports from an SFTP server | Hands-off nightly updates once UBC IT provides a server |
| `shw_analytics/qualtrics_api.py` | Pulls exports through the Qualtrics API | An alternative if API access is ever granted |
| `shw_analytics/labels.py` | Fills in answer labels (e.g. code 1 = "International") from a choice-text export | Answers say "international students" instead of "code 1" |
| `shw_analytics/privacy.py` | Removes emails, phone numbers and names from comments; hides small groups | Protects students without staff having to think about it |

### Analysis and AI

| File | What it does | Why it matters |
|---|---|---|
| `shw_analytics/analysis.py` | All statistics: service and item summaries, group comparisons, the equity scan, wait time vs ratings, focus-group comparisons | Consistent, reproducible numbers behind every chat answer and dashboard |
| `shw_analytics/tools.py` | The only actions Qwen can take, plus the SQL guard | Qwen can answer flexible questions without ever touching raw data |
| `shw_analytics/agent.py` | The conversation: question → tools → answer, with follow-ups, number checks and retries | Turns quick answers into a back-and-forth staff can drill into |
| `shw_analytics/llm.py` | Connects to the LTIC gateway, or Ollama when testing | Swapping models is a settings change, not a code change |
| `shw_analytics/comments.py` | Sentiment, themes, search and a check of comments against ratings | Makes open-text feedback usable at scale |
| `shw_analytics/config.py` | Reads settings from `.env`; holds privacy and equity thresholds | Policy decisions (e.g. the minimum group size) live in one place |

### Dashboards

| File | What it does | Why it matters |
|---|---|---|
| `shw_analytics/dashboard.py` | Defines dashboards as short specs: panel types, templates, focus groups, validation, saving | Reusable reports that refresh with new data each term |
| `shw_analytics/charts.py` | Draws the dashboard page: KPI cards, charts, donuts, alerts | What staff see in the app is exactly what they download and share |
| `shw_analytics/theme.py` | Colours and font | Apply SHW's official design rules by editing one file |

### App and scripts

| File | What it does | Why it matters |
|---|---|---|
| `app.py` | The Streamlit app: Ask, Dashboards, Equity flags, Comments, Data quality, plus data upload in the sidebar | Everything staff use, in one place |
| `scripts/smoke_test.py` | 27 offline checks, from data loading to guardrails to dashboards | Confirms nothing broke after a change, without needing the model |
| `scripts/update_data.sh` | The scheduled job: sync new exports, rebuild, log | Run it nightly with cron once SFTP is set up |

### Created when you run it (`data/`, never committed)

| Path | What it is |
|---|---|
| `data/exports/` | Every Qualtrics export received. Don't delete files here: scheduled exports only contain new responses. |
| `data/shw.duckdb` | The database the app reads, rebuilt from the exports |
| `data/dashboards/` | Saved dashboard layouts |
| `data/update.log` | History of scheduled updates |

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
| Satisfaction and outcomes by service, item, group or faculty | (capable) Survey |
| Differences between student groups, tested for significance | (capable) Survey |
| Wait time and its link to ratings (Medical, Counselling) | (capable) Survey |
| Barriers students report |  Partly: "fell short" answers and comments; students who never used a service aren't surveyed |
| Trends over time | Partly: Needs more than one survey period |
| Appointment demand, visits per student, presenting concerns (reason for visit, CLICC) | (Unable) Needs TELUS CHR exports |
| Demand vs clinician capacity | (Unable) Needs CHR exports and capacity records |
| Utilization alongside survey responses | (Unable) Needs CHR; possible by shared groups (faculty, term), never person by person |

When a question needs missing data, the tool says so in its first sentence,
names the missing source, and then gives the closest evidence the survey can
offer. Update the `coverage` section of `dictionary.yaml` whenever a new data
source is connected.

---

## Settings you're most likely to change

| Setting | Where | Default |
|---|---|---|
| Model and endpoint | `.env`: `LTIC_BASE_URL`, `SHW_CHAT_MODEL` | UBC gateway, `qwen3.8-27b` |
| Minimum group size shown | `.env`: `SHW_MIN_CELL` | 5 |
| Minimum group size for equity tests | `.env`: `SHW_EQUITY_MIN` | 10 |
| Which Qualtrics responses count | `.env`: `SHW_KEEP_CHANNELS` | `anonymous,test` (use `anonymous` for real data) |
| Room for each model reply | `.env`: `SHW_MAX_TOKENS` | 4000 |
| Colours and font | `shw_analytics/theme.py` | SHW mock-up palette |
| What each code means | `dictionary.yaml` | Items marked `CONFIRM` still need checking |

---

## Roadmap

1. **Value labels**: one choice-text export from Qualtrics, then `labels.py`,
   so answers use real group names.
2. **CHR data**: aggregated exports (appointments, presenting concerns and
   clinician hours by week, service and faculty) as a second data source.
3. **Investigation mode**: for "why" questions (e.g. "Why are counselling wait
   times increasing?"), break the question into steps (trend, demand, capacity,
   seasonality, groups), run each, and summarise which explanations the data
   supports.
4. **Hosting**: a UBC server for the app and the scheduled exports, with UBC
   login.
5. **SHW design rules** applied in `theme.py`.

---

## Known limitations

- The current export is test data: ratings are random and comments are
  placeholder text, so results only become meaningful with real responses.
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
| CLICC | Clinician Index of Client Concerns, the form counsellors use to record presenting concerns |

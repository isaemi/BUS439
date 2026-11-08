# Deeper insights for Student Health and Wellbeing

This document describes the second-level questions the SHW analytics assistant is built to
answer: questions that a dashboard filter cannot answer, because they need each student's
sequence of appointments, a fair comparison between periods, or a test of whether a
difference is real. It also covers what the assistant can't answer yet and the data each of
those questions would need, and how every answer can explain where it came from in plain
language.

The machine-readable version the assistant follows is `shw_analytics/playbook.yaml`.

## Why these questions

The use cases come from what campus health and counselling services track and act on, and
from UBC's own direction:

- **Integrated, team-based care.** UBC centralized student health and counselling in the
  Gateway Health building so that a student who sees a doctor can book counselling at the
  same desk. Counselling appointments rose about 40% since 2022, and counselling saw 17,450
  appointments with 5,325 students in 2024–25.
  ([UBC News, March 2026](https://news.ubc.ca/2026/03/student-health-and-wellbeing-gateway-health/))
  Whether students actually move between services, and where the extra demand comes from,
  are now strategic questions.
- **Stepped care and single-session counselling.** Canadian campuses increasingly offer
  single-session ("one at a time") counselling within a stepped-care model, where one visit
  is often the intended plan. ([Cornish et al., 2020](https://katalog.lib.cas.cz/KNAV/EdsRecord/edb,146970543);
  [Carleton](https://wellness.carleton.ca/?p=34)) Routine records cannot tell planned
  one-contact care from intended follow-up that did not happen
  ([practice report](https://studentsuccessjournal.org/index.php/studentsuccess/article/download/4170/1615/15814)),
  so the assistant never calls a single visit "dropout".
- **Missed appointments.** Prior no-shows and long booking lead times are among the strongest
  predictors of missed appointments in published studies
  ([Tuan et al., 2025](https://www.annfammed.org/content/23/4/294);
  [urology practice study](https://pubmed.ncbi.nlm.nih.gov/37145458/)). The CHR export has no
  booking date, so lead time is a data request (below).
- **Reaching under-served groups.** Canadian research finds international graduate students
  are less likely to use campus mental health services
  ([Clarke, NCHA data](https://ojed.org/jis/article/view/3148)). Telling under-use from low
  need requires population denominators, which the assistant does not have yet.
- **Benchmarking caseload.** The Clinical Load Index (students seen per standardized
  full-time counsellor per year) is the field's standard comparison of counselling caseloads
  ([CCMH](https://ccmh.psu.edu/cli)). It needs clinical hours, which are not in the data.

## What the assistant can answer

| Question staff ask | Status | How it is answered |
|---|---|---|
| What should we know this term? | Partial | Runs the tested analyses below and reports only results that passed a test, largest first |
| Is a change real, or just different students coming? | Ready | Same-dates comparison split into "who came" vs "change within groups", with a tested change in students seen |
| Where is the growth in demand coming from? | Ready | Tested change in students seen and visits per student; group counts as context |
| What happens after a student's first visit? | Ready | First new episode of care per student, followed for 60 days: single visits, quick returns, time to a second visit, tested by group |
| Do students come back after missing an appointment? | Ready | Share attending again within 60 days of a first missed appointment |
| Which appointments are missed most often? | Partial | Format, weekday, time of day, booking history and student group, tested with each student counted once; lead time needs data |
| Are students moving between services as intended? | Ready | Share of students starting in one service who use another within 30 days, with 95% ranges |
| Do some groups use services more intensively? | Ready | Visits per student, time in care and services used, each tested |
| When are we busiest, and is it getting busier? | Ready | Weekly and monthly patterns compared with the same period last year |
| How do in-person and virtual appointments compare? | Ready | Format mix over time and tested no-show gap |
| Are we reaching the students who need us? | Needs data | Describes who is seen; under-use needs enrolment counts |
| Which parts of the experience need attention? | Partial | Weakest survey questions plus comments; key-driver analysis needs an overall satisfaction question |
| Do some groups rate services lower? | Ready | Tested scan of survey ratings across groups |
| Are there hidden combinations of groups doing worse? | Partial | One-at-a-time scans for now; a systematic scan of combinations is planned |
| What are students saying behind a number? | Ready | Comment themes and examples on the topic |
| Does the survey speak for students who visit? | Ready | Respondent mix compared with visitor mix |
| Does waiting longer affect ratings? | Ready | Wait time linked to ratings |
| Is our caseload high compared with others? | Needs data | Needs weekly clinical hours |

## Data that would unlock more (for SHW)

1. **Booking date for each appointment**: to measure booking lead time, the strongest
   published predictor of missed appointments.
2. **Enrolment counts by faculty, level, international and first-generation status**: to
   say whether a group is under-served, not just less represented.
3. **An overall satisfaction question for each service in the survey**: for key-driver
   analysis (which parts of the experience matter most).
4. **Weekly clinical hours by service**: for a Clinical Load Index comparison.
5. **Appointment type (single-session, intake, ongoing)**: to tell planned single visits from
   follow-up that did not happen.
6. **How terms are named in real CHR exports**: so partial-term detection works without the
   fallback.

## How answers show where they came from

Every answer can be traced without reading code. For each analysis step, the assistant
produces an evidence card:

- **Data:** which source (appointment records, survey, or written comments)
- **Records used:** which service, dates and groups, and how many records or students
- **What was calculated:** in plain words, e.g. "Compared the same dates in both periods and
  split the change in the no-show rate into a change in which students came and a change
  within each faculty group"
- **How certain:** how the result was tested, e.g. "Checked by re-sampling students many times
  to get a range of plausible values"
- **Keep in mind:** caveats carried from the analysis, e.g. synthetic data, an incomplete term,
  small groups combined, responses left out

Custom database queries are described in words ("Looked at appointment records where service
is Counselling and appointment outcome is attended, and calculated the number of distinct
students, broken down by level of study"). Code is never shown to staff.

If staff ask "Where did this come from?", "How did you calculate that?" or "Can I trust
this?", the assistant answers from these cards in plain paragraphs. If its wording slips into
technical names, the cards are shown instead.

## Rules every answer follows

- Patterns are described, never causes.
- A difference is called real only if it passed a test; group sizes are always given.
- Periods are compared over the same dates; an incomplete term is never compared with a
  complete one.
- When many groups are checked, results are corrected for the number of checks.
- Synthetic data is labelled as such.

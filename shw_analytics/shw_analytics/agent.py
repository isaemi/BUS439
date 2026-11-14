"""Question-answering loop: staff question -> Qwen picks tools -> answer.

Works in two modes (config.TOOL_MODE):
  native  OpenAI-style tool calling through the gateway
  json    the model replies {"tool": ..., "args": ...} or {"answer": ...};
          used automatically if the gateway rejects the `tools` parameter
"""
import json
import re
import math

from openai import BadRequestError

from . import config, llm, tools

SYSTEM = """You are the analytics assistant for UBC Student Health & Wellbeing (SHW).
You answer staff questions about the student feedback survey using ONLY the tools.

Rules:
- Do the work yourself. Never tell the user to run a tool, check a table, or run a test:
  call the tools and report what they show. Never mention tool, table or column names;
  describe things in plain words (group codes such as "code 1" are fine).
- Check data_coverage before answering. If the question needs data the tool does not
  have, say so in the FIRST sentence and name the missing source in plain words (e.g.
  "Appointment counts come from CHR records, which aren't connected yet"). Then call
  the tools that give the closest survey evidence, report it, and end with one line on
  what data would answer the full question.
- Never compute or invent numbers yourself. Every figure must come from a tool result.
- Prefer the specific tools; use run_sql only when none fits.
- For SQL, query the ratings_demo view (ratings already joined to demographics) so no
  join is needed. Only join response_faculty, response_disability or response_multi.
- Use all items of a service unless the user names a specific item.
- Only say a difference is or is not "statistically significant" if a tool result
  contains a test (q or p_value). Otherwise call it descriptive and untested.
- Refer to survey items by their wording (item_text), never by code such as CS1_6.
- Answer codes like '1' or '2' are unlabelled; never guess what they mean. If the user
  names a group in words (e.g. "international students") and you filter on a code,
  say so plainly: "assuming code 1 = international; answer labels aren't loaded yet". Filter on the
  exact codes listed in category_values, and report groups by code (e.g. "intl_or_domestic=1").
- Never add up rows yourself for totals; get totals with a query (subgroup counts can
  miss hidden groups).
- % agree in SQL is ROUND(100*AVG(CASE WHEN agree THEN 1 ELSE 0 END),1); never divide by
  a count of respondents.
- Follow-ups: if the question builds on an earlier one ("what about medical?", "and for
  international students?", "why?", "break that down by faculty"), keep the earlier
  service, measure and filters and change only what the user changed. Call tools again
  for the new data; you may quote figures from earlier answers. If the question is
  unrelated, treat it as new.
- "By category"/"by type" questions need the category breakdown (e.g. dimension
  "disability"), not the yes/no version. When groups are hidden, name them using
  hidden_groups; never guess which groups were hidden.
- Appointments, visits, demand, no-shows, days in care (CHR) -> chr_summary; over time or
  "compared with last year" -> chr_trend; WHY one group uses services more or less ->
  explain_difference (report its summary and findings; say which parts were tested);
  whether survey feedback reflects who visits -> survey_vs_visits. If a CHR result says
  SYNTHETIC, say plainly that it is synthetic test data.
- Routing: one group "compared with everyone / overall / all students" ->
  group_vs_everyone (it includes significance tests). Groups side by side (e.g. "by
  faculty", "across disability types") -> compare_groups. Faculties: match
  faculty_values exactly (e.g. "Arts").
- Questions combining two groups (e.g. international first-generation students) need
  run_sql on ratings_demo with both columns in GROUP BY; compare_groups covers only one.
- If no tool result covers the exact group or service asked about, say so in the FIRST
  sentence. Never present results for everyone as if they were for a subgroup.
- For questions about differences between groups (equity), call equity_flags. A gap is
  only "significant" if it appears in flags. compare_groups is descriptive only.
- For Chi-squared association between a demographic (faculty, disability, academic level,
  etc.) and a survey rating, call demographic_association with the verified service,
  outcome item code and demographic dimension. Faculty and other multi-select
  categories require separate selected-versus-not-selected tests; never treat
  overlapping categories as mutually exclusive. Report BH-adjusted q-values for
  multiple comparisons. Do not claim the analysis is unsupported without calling
  the dedicated demographic test. Respect privacy suppression errors.
  For multi-select demographics, use categories_requested and categories_tested
  to state exactly how many categories were evaluated; never infer the number
  from how many rows you choose to discuss. If summarizing only a subset,
  explicitly say it is a subset and report the overall counts and exclusions.
  Never switch services when referring to earlier analyses.
- When staff ask whether two categories are related, linked, or differ in likelihood,
  consult the categorical_association playbook even if they do not name Chi-squared.
  First map each everyday-language concept to the matching survey question wording;
  do not treat "fell short of expectations" as a synonym for "difficulty booking".
  Use a statistical test only when the variables and independence assumptions fit;
  otherwise explain the limitation and provide descriptive results.
- Before ANY survey association test, CALL lookup_survey_items for the requested service
  (omit query to inspect the full list when matching is uncertain). Read its item_code
  and item_text. Pass verified item_code strings, NEVER paraphrases or question wording,
  as outcome_item and predictor_item to test_association or demographic_association.
  If a tool reports an unverified item, use lookup_survey_items and retry with codes.
  Never claim an item is absent merely because a description failed as an argument.
- Before ANY survey association test, resolve the user's concepts to the exact survey
  item wording in the available data. A vague phrase is not permission to substitute a
  nearby but different variable. In Counselling, "difficulty booking", "booking ease",
  or "struggle to book" refers to the booking-ease item (previously CS1_1), NOT
  cs_fell_short (whether an experience fell short of expectations). "Overall
  satisfaction with support" refers to the overall-support satisfaction item
  (previously CS1_11), NOT a general expectations item. Verify the current item_text
  and service in the data before using these codes; do not assume codes remain stable.
  The lookup tool may return verified_semantic_alias matches for common phrasing
  such as "difficulty booking", "problems booking", "trouble booking",
  "issues scheduling", and "hard to book". Prefer a verified semantic match
  when it directly expresses the requested concept; do not treat a mere
  text_overlap match as definitive if its wording describes a different concept.
  For positively worded booking ease, disagreement (scores 1-2) represents
  reported difficulty, but keep the original 1-5 ratings for standard Chi-square
  tests. Do not interpret "difficulty" as a new binary variable unless requested.
  When multiple plausible items genuinely fit, ask one brief clarifying question
  rather than silently choosing a proxy. If the exact requested measure is absent,
  state that clearly and do not replace it with a different measure.
- For Wellness Advising booking difficulty, trouble scheduling, hard to book,
  problems making appointments, or ease of booking, look up the service's
  "I was able to book an appointment easily" item (historically WA1_1).
  Overall support satisfaction refers to "Overall, I am satisfied with the
  support I received" (historically WA1_8 and WA1_10). These two satisfaction
  items have identical wording: check usable paired responses and privacy
  restrictions before selecting one. Prefer the item with valid paired data,
  explain which item was used, and never claim the other has zero responses
  merely because its result is privacy-suppressed. Verify all codes against
  live items, preserve the five Likert levels, and never treat wait time or
  access to the centre as equivalent to booking ease.
- For association between two categorical Qualtrics questions, use test_association with
  verified question item codes and the correct service; never substitute related items.
  For CHR categorical association, use chr_categorical_association. Do not use
  survey item codes for CHR. State which question wording/variables were used.
  Chi-square expected-count assumptions and observed-cell privacy suppression are
  different checks. If a result is suppressed, do not invent its statistics.
  The synthetic dataset is test data, not findings about actual UBC students.
- For a Chi-square association between two ordered survey ratings, statistical
  significance and Cramer's V establish association, NOT its direction or a
  monotonic trend. Do not claim "easier booking means higher satisfaction",
  "harder booking means lower satisfaction", or a "gradient" solely from
  Chi-square, Cramer's V, or a comparison of two selected cells.
  To describe direction, first inspect the COMPLETE, correctly labelled
  cross-tabulation: compare satisfaction 4-or-5 percentages WITHIN each of
  the five booking-ease rows, in rating order 1, 2, 3, 4, 5. Do not confuse
  booking scores 4 and 5, or transfer percentages from one row to another.
  If row totals or complete unsuppressed rows are unavailable, state that
  direction cannot be reliably determined from the reported statistics.
  If percentages rise and fall across booking levels, describe a mixed or
  non-monotonic pattern, not a consistent trend. Never imply causation.
  Never call the association "strong" solely because the p-value is tiny.
- Report n with every figure. If groups were suppressed (n < {min_cell}), say so plainly.
- Survey scale: 1-5, "% agree" = share answering 4 or 5.
- Comments: summarise themes; quote at most short phrases; never guess who wrote them.
- Answer in plain language for non-technical staff: lead with the answer, then 2-4
  supporting points, then one line on limits (sample size, test data, etc).

Data overview:
{overview}
/no_think"""

JSON_PROTOCOL = """
You cannot call functions directly. On each turn reply with JSON only, either
  {{"tool": "<name>", "args": {{...}}}}
or, when you have enough information,
  {{"answer": "<final answer for the staff member>"}}
Available tools:
{tool_list}"""


def _overview():
    d = tools.describe_data()
    return json.dumps({"services": d["services"], "equity_dimensions": d["equity_dimensions"],
                       "faculty_values": d["faculty_values"],
                       "category_values": d["category_values"],
                       "likert": d["likert"], "tables": d["tables"], "views": d["views"],
                       "relationships": d["relationships"], "rules": d["rules"],
                       "data_coverage": d["data_coverage"]})


PERIOD_RULE = (
    "\n\nComparing periods: never compare a term that the data only partly covers (any tool "
    "result marked PARTIAL TERM) with a complete term; it will look lower only because less "
    "time is covered. For any question about change between periods (this fall vs last fall, "
    "this year vs last year), use what_changed, which compares matching date windows, and quote "
    "its volume_in_same_windows for counts. Always state the exact dates you compared.")


INSIGHT_RULE = (
    "\n\nDeeper questions: for broad or open questions (what should we know, anything "
    "surprising, where to focus, are we reaching a group, is integrated care working) and for "
    "questions about continuity of care, missed appointments or movement between services, first "
    "call insight_playbook with the question and follow the steps it returns. Respect its "
    "'avoid' list and wording. If its status is needs_data, explain what data is missing "
    "instead of estimating.")


def _explain_sources(question, prev_trace, chat_fn, ctx):
    """Answer 'where did this come from?' in plain words, from evidence cards built from the
    previous answer's analysis steps. Falls back to the cards themselves if the model's
    wording mentions internal names or fails."""
    from . import provenance
    cards = provenance.evidence_cards(prev_trace)
    trace = list(prev_trace) + [("evidence_cards", {}, {"cards": cards})]
    if not cards:
        return {"answer": "The previous answer did not use any data analysis, so there are no "
                          "sources to show.", "trace": trace, "mode": "sources", "ungrounded": []}
    messages = [{"role": "system", "content": provenance.SOURCES_PROMPT},
                {"role": "user", "content": f"Staff question: {question}\n\nEvidence cards for "
                                            f"the previous answer:\n\n{provenance.render_text(cards)}"}]
    try:
        answer = (chat_fn(messages).content or "").strip()
    except Exception:
        answer = ""
    if not answer or mentions_internals(answer):
        answer = provenance.render_markdown(cards)
    return _finalise(answer, trace, "sources", ctx)


EVIDENCE_RULE = (
    "\n\nEvidence rules: (1) A result that is not distinguishable from noise is absence of "
    "evidence, not proof of no effect: never write that something 'is not the issue' or 'was "
    "tested and is not the problem'; say the data could not detect a difference and, if known, "
    "how large a difference it could detect. (2) When a verdict is 'no clear change', mention "
    "the group figures behind it only as context, never as a finding or a 'hidden shift'. "
    "(3) State the dates each finding covers, especially when combining same-dates comparisons "
    "with results that cover all dates. (4) Questions about combinations of groups (e.g. "
    "first-generation students in particular faculties) cannot be checked systematically yet: "
    "say so first, give one-at-a-time results, never answer 'no' to the combination, and never "
    "build the combination from custom queries.")


AGREE_RULE = (
    "\n\nWording '% agree': for a single question it is the share of students who agreed "
    "('33% of students agreed they felt welcomed'). For a whole service or a student group it "
    "is an average across questions, not a share of students: say 'on average, students "
    "agreed with 39% of the counselling questions', never '39% of students agreed'.")


PLAIN_RULE = (
    "\n\nPlain numbers: staff are not statisticians. Never write p-values, q-values, 'CI' or "
    "'confidence interval'. Say whether a difference was tested and is real, or is within noise, "
    "and give the range in words, e.g. '22.8% more students (range 8.6% to 38.8%)'. The same goes "
    "for test statistics such as chi-square, degrees of freedom or Cramer's V: use them to decide "
    "what to say, and leave the figures to the technical details under the answer. This rule takes "
    "precedence over any tool, playbook or earlier rule that says to report them. Write date "
    "windows as '1 Sep to 31 Oct'.")


def _system(json_mode):
    s = SYSTEM.format(min_cell=config.MIN_CELL, overview=_overview()) + PERIOD_RULE + INSIGHT_RULE + EVIDENCE_RULE + AGREE_RULE + PLAIN_RULE
    if json_mode:
        tl = "\n".join(f"- {t['function']['name']}({', '.join(t['function']['parameters']['properties'])}): "
                       f"{t['function']['description']}" for t in tools.TOOL_SPECS)
        s += JSON_PROTOCOL.format(tool_list=tl)
    return s


_CODE = re.compile(r"\b[A-Z]+\d*_\d+\b")          # item codes like WA1_10
_NUM = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?")
_FREE = {str(i) for i in range(0, 11)} | {"100"}     # scale points, counts like "3 points"

_INTERNAL = re.compile(r"\b(" + "|".join(
    [t["function"]["name"] for t in tools.TOOL_SPECS] +
    ["ratings_demo", "response_multi", "response_faculty", "response_disability",
     "items table", "responses table", "ratings table"]) + r")\b")

_ITEM_CODE = re.compile(r"\b[A-Z]{2,4}1_\d+\b")

ITEM_CODE_NUDGE = ("Your answer refers to survey items by code (e.g. CS1_6). Staff need the "
                   "wording. Item wording is in item_text (in ratings, ratings_demo and items). "
                   "Re-run the query with item_text, then rewrite the answer using the wording.")


def mentions_item_codes(answer):
    return sorted(set(_ITEM_CODE.findall(answer or "")))


HANDOFF_NUDGE = ("Your answer mentions internal tool or table names, or asks the user to run "
                 "something. Staff never see those. If the analysis would help, call the "
                 "tool yourself now; then rewrite the answer in plain words.")


def mentions_internals(answer):
    return sorted(set(_INTERNAL.findall(answer or "")))


EMPTY_NUDGE = ("Your last reply was empty. Call the tool you need (run_sql if no other tool "
               "fits), or give the final answer. Keep any reasoning short.")
EMPTY_ANSWER = ("The model returned an empty reply twice, usually because it ran out of room "
                "while reasoning. Try again, split the question into smaller parts, or raise "
                "SHW_MAX_TOKENS in .env.")

NO_TOOL_NUDGE = ("You have not called any tool. Do not answer from memory: call the tool(s) "
                 "that fit this question first.")


def _result_numbers(trace):
    """All figures in tool results (with rounding variants), plus the values grouped by
    field name so gaps can be checked within a field."""
    found, by_key = set(), {}
    def add(x, key):
        for d in (0, 1, 2, 3, 4):
            v = round(float(x), d)
            found.add(f"{v:.{d}f}")
            if d == 0:
                found.add(f"{int(v):,}")
            found.add(f"{100 * v:.{max(d - 2, 0)}f}")  # 0.413 -> 41.3 style
        found.add(str(x))
        by_key.setdefault(key, set()).add(float(x))
    def walk(x, key=None):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            add(x, key)
        elif isinstance(x, str):
            found.update(_NUM.findall(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v, key)
    for _, _, res in trace:
        walk(res)
    return found, by_key


def _differences(by_key):
    """Gaps between two figures of the SAME field (e.g. two pct_agree values), since
    stating a difference is normal. Sums and cross-field arithmetic are not accepted:
    totals should come from a tool, not from the model adding rows up."""
    extra = set()
    for vals in by_key.values():
        vals = sorted(vals)[:200]
        for i, a in enumerate(vals):
            for b in vals[i + 1:]:
                for dp in (0, 1, 2):
                    extra.add(f"{b - a:.{dp}f}")
    return extra


def _p_values(trace):
    """Read actual numeric test p-values; don't infer them from prose."""
    values = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("p", "p_value") and isinstance(v, (float, int)) and not isinstance(v, bool):
                    if 0 <= v <= 1:
                        values.append(float(v))
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for name, _, result in trace:
        if name not in ("question", "suggested", "fallback"):
            walk(result)
    return values


def _numeric_tokens(answer):
    """Keep scientific notation intact, unlike the original decimal-only regex."""
    text = _CODE.sub("", answer or "")
    return re.findall(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?", text)


def ungrounded_numbers(answer, trace):
    """Conservatively flag unsupported numbers, with verified p-value thresholds.

    An inequality such as p < 0.0001 is a valid rounded claim when an actual
    numeric p-value in the result is below that threshold. It is NOT evidence
    for an arbitrary count or percentage.
    """
    found, by_key = _result_numbers(trace)
    is_association = any(name == "test_association" for name, _, _ in trace)
    known = found if is_association else found | _differences(by_key)
    p_values = _p_values(trace)
    bad = []
    text = _CODE.sub("", answer or "")
    for match in re.finditer(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?", text):
        n = match.group()
        norm = n.replace(",", "")
        try:
            value = float(norm)
        except ValueError:
            bad.append(n)
            continue
        if n in _FREE or n in known or norm in known:
            continue
        # Treat p-value bounds as derived from the actual test, not as invented
        # data. Check the operator and numeric result before allowing them.
        prefix = text[max(0, match.start()-14):match.start()]
        p_match = re.search(r"(?:\bp(?:-value| value)?\s*)(<=|>=|<|>|=)\s*$", prefix, re.I)
        if p_match and p_values:
            op = p_match.group(1)
            if ((op == "<" and any(p < value for p in p_values)) or
                (op == "<=" and any(p <= value for p in p_values)) or
                (op == ">" and any(p > value for p in p_values)) or
                (op == ">=" and any(p >= value for p in p_values)) or
                (op == "=" and any(math.isclose(p, value, rel_tol=1e-4, abs_tol=1e-300) for p in p_values))):
                continue
        # A reported scientific p-value may be rounded relative to the tool.
        if re.search(r"[eE][+-]?\d+", norm) and p_match and p_values:
            if any(math.isclose(p, value, rel_tol=0.02, abs_tol=1e-300) for p in p_values):
                continue
        if "." in norm and "e" not in norm.lower():
            norm = norm.rstrip("0").rstrip(".")
        if norm in known:
            continue
        bad.append(n)
    return sorted(set(bad))


def _remove_unsupported_breakdown(answer, trace):
    """Fail closed on unsupported association breakdowns rather than printing them.

    Do not silently alter the reported test itself; leave its statistics in place
    for the normal grounding warning if those are unverified.
    """
    if not any(name == "test_association" and isinstance(res, dict) and "error" not in res
               for name, _, res in trace):
        return answer
    kept = []
    removed = False
    for line in (answer or "").splitlines():
        bad = ungrounded_numbers(line, trace)
        breakdown = bool(re.search(r"\b(booking (?:ease )?(?:score|rating)|among (?:students|those)|"
                                   r"strongly (?:agreed|disagreed)|satisfaction (?:score|rating))\b", line, re.I))
        if bad and breakdown:
            removed = True
            continue
        kept.append(line)
    result = "\n".join(kept).strip()
    if removed:
        result += ("\n\nDetailed subgroup figures were omitted because they could not "
                   "be verified against the current statistical results.")
    return result


DIRECTION_CHECK_NUDGE = (
    "Before finalizing this survey association answer, verify any directional "
    "claim against the FULL contingency table, with booking scores 1 through 5 "
    "correctly matched to their own satisfaction counts. Chi-square significance "
    "and Cramer's V alone do NOT show which direction the association runs. "
    "Calculate or quote satisfaction 4-or-5 percentages for EACH booking row "
    "only when row totals and unsuppressed counts support them. If the pattern "
    "is mixed, explicitly say it is non-monotonic. If the full table is missing, "
    "do not claim a directional trend. Do not swap scores 4 and 5. "
    "Use only subgroup counts and percentages explicitly returned for the SAME "
    "booking row in the current result; omit any unverified breakdown. "
    "Now provide a carefully verified answer."
)


def _needs_direction_review(answer, trace):
    """Review direction claims only for actual survey item association tests."""
    has_survey_test = any(
        name == "test_association" and isinstance(result, dict) and "error" not in result
        for name, _, result in trace
    )
    if not has_survey_test:
        return False
    direction = re.search(
        r"\b(tend(?:s)? to|generally|consistently|gradient|higher satisfaction|"
        r"lower satisfaction|more satisfied|less satisfied|positively associated|"
        r"negatively associated|easier booking|harder booking)\b",
        answer or "", re.I,
    )
    return bool(direction)


def _grounding_retry_msg(bad):
    return (f"These numbers are not verified by the current tool results: {', '.join(bad)}. "
            "Remove unsupported subgroup details rather than guessing replacement counts. "
            "Use only current, correctly labelled results; if needed give the overall "
            "association without a numerical subgroup breakdown.")


_SIG = re.compile(r"statistically|significan", re.IGNORECASE)
_SIG_NOT_A_CLAIM = re.compile(
    r"\b(identify|identifies|test|tests|testing|check|checks|look for|detect|whether|could|"
    r"would|might|run)\b", re.IGNORECASE)


# Tools that test with confidence intervals rather than p-values. A successful result from
# one of these counts as a significance test for the "no test was run" warning.
INTERVAL_TOOLS = {"term_briefing", "survey_changed", "priority_items", "what_changed", "care_journeys", "noshow_patterns", "service_pathways"}


def _has_interval_test(steps):
    for step in steps:
        try:
            name, _, result = step
        except (TypeError, ValueError):
            continue
        if name in INTERVAL_TOOLS and isinstance(result, dict) and "error" not in result:
            return True
    return False


def _claims_significance(answer):
    """True if the answer asserts (or denies) significance, rather than just mentioning
    testing for it, e.g. "to identify significant gaps" is not a claim."""
    for m in _SIG.finditer(answer or ""):
        start = max(answer.rfind(".", 0, m.start()), answer.rfind("\n", 0, m.start())) + 1
        window = answer[max(start, m.start() - 40):m.start()]
        if not _SIG_NOT_A_CLAIM.search(window):
            return True
    return False


def _has_test(trace):
    def walk(x):
        if isinstance(x, dict):
            return any(k in ("q", "p", "p_value", "flags") or walk(v) for k, v in x.items())
        if isinstance(x, list):
            return any(walk(v) for v in x)
        return False
    return any(walk(res) for _, _, res in trace)


def _finalise(answer, trace, mode, ctx=()):
    answer = _remove_unsupported_breakdown(answer, trace)   # Jonathan: drop unverifiable subgroup lines first
    from .plain import plain_stats
    answer = plain_stats(answer)                            # then keep statistics jargon out of the staff answer
    bad = ungrounded_numbers(answer, list(ctx) + trace)
    if bad:
        answer += (f"\n\n⚠️ Check before using: these figures could not be traced to the data: "
                   f"{', '.join(bad)}.")
    if _claims_significance(answer) and not (_has_test(list(ctx) + trace) or _has_interval_test(list(ctx) + trace)):
        answer += ("\n\n⚠️ No significance test was run for this answer, so any claim about "
                   "statistical significance is unsupported. Treat the comparison as descriptive.")
    return {"answer": answer, "trace": trace, "mode": mode, "ungrounded": bad}


def _run_tool(name, args, seen):
    """Run a tool, but refuse an exact repeat of an earlier call in the same question."""
    key = (name, json.dumps(args, sort_keys=True))
    if key in seen:
        return {"error": "You already made this exact call. Its result was: "
                + _trim(seen[key], 800) + " Change the query or answer with what you have."}
    result = tools.call(name, args)
    seen[key] = result
    return result


def _trim(result, limit=6000):
    text = json.dumps(result, default=str)
    return text if len(text) <= limit else text[:limit] + '..."(truncated)"'


MAX_TURNS = 4  # earlier question/answer pairs sent with each new question


def _strip_warnings(answer):
    return (answer or "").split("\n\n⚠️")[0]


def _history_messages(history):
    """Earlier turns as chat messages. Each past answer carries a compact record of the
    tool calls behind it, so a follow-up can build on the same service and filters."""
    msgs = []
    for turn in (history or [])[-MAX_TURNS:]:
        if "role" in turn:  # plain chat message (older callers)
            msgs.append(turn)
            continue
        calls = [f"{name}({json.dumps(args)}) -> {_trim(res, 1200)}"
                 for name, args, res in turn.get("trace", []) if name not in ("fallback", "empty_reply", "suggested")]
        record = ("Data used:\n" + "\n".join(calls) + "\n\n") if calls else ""
        msgs += [{"role": "user", "content": turn["question"]},
                 {"role": "assistant", "content": record + "Answer: " + _strip_warnings(turn["answer"])}]
    return msgs


def _context_trace(question, history):
    """Tool results from earlier turns (and the question itself) count as known
    figures, so quoting an earlier number in a follow-up isn't flagged."""
    ctx = [("question", {}, question)]
    for turn in (history or [])[-MAX_TURNS:]:
        if "trace" in turn:
            ctx += turn["trace"]
            ctx.append(("question", {}, turn["question"]))
    return ctx


# A simple router: reads the question for services, groups and intent, and suggests
# a starting tool with its arguments. Small models pick tools far more reliably with a
# concrete suggestion; the model is still free to ignore it.
_WHY = re.compile(r"\b(why|differ\w*|explain\w*|versus|vs\.?|compared? (with|to)|more than|less than)\b", re.I)
_TREND = re.compile(r"\b(last year|year[- ]over[- ]year|trend\w*|over time|increas\w*|decreas\w*|"
                    r"grow\w*|drop\w*|rising|falling|month\w*|this term|last term)\b", re.I)
_USAGE = re.compile(r"\b(appointments?|visits?|sessions?|no[- ]shows?|attend\w*|demand|use|uses|using|"
                    r"utili[sz]\w*|days in care|seen)\b", re.I)
_EVERYONE = re.compile(r"\b(everyone|overall|all (other )?(students|respondents))\b", re.I)
_WAIT = re.compile(r"\bwait\w*\b", re.I)
_REPRESENT = re.compile(r"\b(represent\w*|reflect\w*|who (actually )?visits)\b", re.I)
_SURVEY = re.compile(r"\b(experienc\w*|rate|rates|rating\w*|satisf\w*|agree\w*|feedback|survey|feel)\b", re.I)
_ASSOCIATION = re.compile(r"\b(associat\w*|relat\w*|link\w*|connect\w*|correlat\w*|chi[- ]?squared|significan\w*|more likely|less likely|more common|less common|less satisfied|more satisfied)\b", re.I)


def _find_values(question, values):
    found = []
    for v in values:
        if v and re.search(r"\b" + re.escape(str(v).lower()) + r"s?\b", question.lower()):
            found.append(v)
    return found


def route(question):
    """Suggested tool calls (name, args) for a question, most specific first."""
    from . import analysis
    con = tools._con()
    q = question
    svc_survey = next((k for k, v in {"counsel": "counselling", "medical": "medical",
                                      "wellness advis": "wellness_adv", "info desk": "info_desk",
                                      "naloxone": "naloxone"}.items() if k in q.lower()), None)
    svc_survey = {"counsel": "counselling", "medical": "medical", "wellness advis": "wellness_adv",
                  "info desk": "info_desk", "naloxone": "naloxone"}.get(svc_survey)
    hints, terms = [], []
    if svc_survey:
        terms.append(svc_survey.replace("_", " ").split()[0][:7])
    # Relationship questions should consult the playbook before choosing a descriptive tool.
    if _ASSOCIATION.search(q):
        hints.append(("insight_playbook", {"topic_id": "categorical_association"}))
    has_chr = analysis.has_chr(con)
    survey_question = bool(_SURVEY.search(q)) and not _USAGE.search(q.replace("services", ""))
    if _REPRESENT.search(q):
        if has_chr:
            hints.append(("survey_vs_visits", {"dimension": "faculty"}))
        has_chr = False  # a representation question, not a usage question
    if has_chr and not survey_question:
        line = {"counselling": "Counselling", "medical": "Medical", "wellness_adv": "Wellness Advising"}.get(svc_survey)
        for dim in ("academic_level", "faculty", "gender", "mode", "age_band", "visit_type"):
            vals = con.execute(f'SELECT DISTINCT "{dim}" FROM appointments WHERE "{dim}" IS NOT NULL').df()[dim]
            found = _find_values(q, sorted(vals.astype(str), key=len, reverse=True))
            # "graduate" also appears inside "undergraduate"; keep the longest distinct matches
            found = [f for f in found if not any(f != g and f.lower() in g.lower() and
                     re.search(r"\b" + g.lower(), q.lower()) and not re.search(r"(?<!under)\b" + f.lower(), q.lower())
                     for g in found)]
            if found:
                terms += [f.lower() for f in found]
                args = {"dimension": dim, "group": found[-1] if dim == "academic_level" and len(found) > 1
                        and found[-1].lower() == "graduate" else found[0]}
                if len(found) > 1:
                    args["compare_to"] = [f for f in found if f != args["group"]][0]
                if line:
                    args["service_line"] = line
                if _WHY.search(q) or _USAGE.search(q):
                    hints.append(("explain_difference", args))
                break
        by = re.search(r"\bby (service|faculty|gender|level|mode|age|program|term)\b", q, re.I)
        terms_in_data = con.execute("SELECT term, MIN(start_time) AS s FROM appointments "
                                    "WHERE term IS NOT NULL GROUP BY 1 ORDER BY s").df()["term"].tolist()
        term = None
        if re.search(r"\blast term\b", q, re.I) and len(terms_in_data) > 1:
            term = terms_in_data[-2]
        elif re.search(r"\bthis term\b", q, re.I) and terms_in_data:
            term = terms_in_data[-1]
        if by or (term and not re.search(r"\b(trend|over time|last year|month)", q, re.I)):
            args = {k: v for k, v in {"service_line": line, "term": term,
                    "dimension": {"service": "service_line", "level": "academic_level", "age": "age_band"}
                    .get(by.group(1).lower(), by.group(1).lower()) if by else None}.items() if v}
            hints.append(("chr_summary", args))
        elif _TREND.search(q) and _USAGE.search(q):
            hints.append(("chr_trend", {"service_line": line} if line else {}))
        elif _USAGE.search(q) and not hints:
            hints.append(("chr_summary", {"service_line": line} if line else {}))
    if _WAIT.search(q) and svc_survey in ("medical", "counselling"):
        hints.append(("wait_time_vs_rating", {"service": svc_survey}))
    if _EVERYONE.search(q):
        faculties = tools.load_dictionary()["checkbox"]["faculty"]["choices"]
        f = _find_values(q, faculties)
        if f:
            hints.append(("group_vs_everyone", {"dimension": "faculty", "value": f[0]}))
            terms.append(f[0].lower())
    return hints, terms


def _hint_message(hints):
    calls = "; ".join(f"{n}({json.dumps(a)})" for n, a in hints)
    return (f"Suggested starting point, based on the question's wording: {calls}. Use it unless it "
            "clearly doesn't fit the question.")


def _on_topic(answer, terms):
    """True if the answer mentions at least one of the question's subjects."""
    if not terms:
        return True
    a = (answer or "").lower()
    return any(t[:7] in a for t in terms)


OFF_TOPIC_NUDGE = ("Your answer doesn't address what was asked ({terms}). Answer that question, "
                   "using the suggested tool if it fits.")


def _safe_route(question):
    try:
        return route(question)
    except Exception:  # routing is a convenience; never let it break an answer
        return [], []


def ask(question, history=None, max_steps=6, chat_fn=None):
    """Returns {"answer": str, "trace": [(tool, args, result), ...], "mode": str}.
    `chat_fn` lets tests inject a fake model."""
    chat_fn = chat_fn or llm.chat
    mode = config.TOOL_MODE
    trace = []
    ctx = _context_trace(question, history)
    from . import provenance
    if provenance.is_sources_question(question):
        prev = provenance.last_trace(history) or [
            s for s in (ctx or []) if isinstance(s, (list, tuple)) and len(s) == 3]
        if prev:
            return _explain_sources(question, prev, chat_fn, ctx)
    if mode in ("native", "auto"):
        try:
            return _ask_native(question, history, max_steps, chat_fn, trace, ctx)
        except BadRequestError as e:
            if mode == "native":
                raise
            trace.append(("fallback", {}, f"native tool calling rejected: {e}"))
    return _ask_json(question, history, max_steps, chat_fn, trace, ctx)


FINAL_NUDGE = ("You have reached the step limit. Answer the question now using only the tool "
               "results above. Give n with every number, say what the results do not show, "
               "and do not call any more tools.")
REPEAT_NOTE = ("You already ran this (only names or formatting differ). Use the earlier result: "
               "answer now, or try a genuinely different query.")


def _call_key(name, args):
    """Same call = same tool and same arguments; for SQL, ignore aliases and spacing."""
    if name == "run_sql":
        sql = re.sub(r"\s+as\s+\w+", "", str(args.get("sql", "")), flags=re.I)
        return (name, " ".join(sql.lower().split()))
    return (name, json.dumps(args, sort_keys=True))


def _ask_native(question, history, max_steps, chat_fn, trace, ctx):
    hints, terms = _safe_route(question)
    messages = [{"role": "system", "content": _system(False)}, *_history_messages(history),
                {"role": "user", "content": question}]
    if hints:
        messages.append({"role": "system", "content": _hint_message(hints)})
        trace.append(("suggested", {"calls": [n for n, _ in hints]}, {"hints": [{"tool": n, "args": a} for n, a in hints]}))
    nudged = retried = handed_off = coded = drifted = direction_checked = False
    empties = 0
    seen = {}
    asked = set()
    for _ in range(max_steps):
        msg = chat_fn(messages, tools=tools.TOOL_SPECS)
        if not getattr(msg, "tool_calls", None) and not (msg.content or "").strip():
            empties += 1
            trace.append(("empty_reply", {"finish_reason": getattr(msg, "finish_reason", None)}, {}))
            if empties >= 2:
                return {"answer": EMPTY_ANSWER, "trace": trace, "mode": "native", "ungrounded": []}
            messages.append({"role": "user", "content": EMPTY_NUDGE})
            continue
        if not getattr(msg, "tool_calls", None):
            tool_steps = [t for t in trace if t[0] not in ("fallback", "empty_reply", "suggested")]
            if not tool_steps and not nudged:
                nudged = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": NO_TOOL_NUDGE}]
                continue
            if not _on_topic(msg.content, terms) and not drifted:
                drifted = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": OFF_TOPIC_NUDGE.format(terms=", ".join(terms))}]
                continue
            if mentions_internals(msg.content) and not handed_off:
                handed_off = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": HANDOFF_NUDGE}]
                continue
            if mentions_item_codes(msg.content) and not coded:
                coded = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": ITEM_CODE_NUDGE}]
                continue
            if not direction_checked and _needs_direction_review(msg.content, trace):
                direction_checked = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": DIRECTION_CHECK_NUDGE}]
                continue
            bad = ungrounded_numbers(msg.content, ctx + trace)
            if bad and not retried:
                retried = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": _grounding_retry_msg(bad)}]
                continue
            return _finalise(msg.content, trace, "native", ctx)
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name,
                                                      "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            key = _call_key(tc.function.name, args)
            if key in asked:
                result = {"note": REPEAT_NOTE}
            else:
                asked.add(key)
                result = _run_tool(tc.function.name, args, seen)
            trace.append((tc.function.name, args, result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _trim(result)})
    messages.append({"role": "user", "content": FINAL_NUDGE})
    msg = chat_fn(messages)
    if (getattr(msg, "content", None) or "").strip():
        trace.append(("forced_answer", {}, {}))
        return _finalise(msg.content, trace, "native", ctx)
    return {"answer": "I couldn't finish within the step limit; try a narrower question.",
            "trace": trace, "mode": "native"}


def _ask_json(question, history, max_steps, chat_fn, trace, ctx):
    hints, terms = _safe_route(question)
    messages = [{"role": "system", "content": _system(True)}, *_history_messages(history),
                {"role": "user", "content": question}]
    if hints:
        messages.append({"role": "system", "content": _hint_message(hints)})
    nudged = retried = handed_off = coded = direction_checked = False
    seen = {}
    for _ in range(max_steps):
        reply = chat_fn(messages).content
        try:
            obj = llm.parse_json(reply)
        except Exception:
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "Reply with JSON only, as instructed."}]
            continue
        if "answer" in obj:
            tool_steps = [t for t in trace if t[0] not in ("fallback", "empty_reply", "suggested")]
            if not tool_steps and not nudged:
                nudged = True
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": NO_TOOL_NUDGE}]
                continue
            if mentions_internals(obj["answer"]) and not handed_off:
                handed_off = True
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": HANDOFF_NUDGE}]
                continue
            if mentions_item_codes(obj["answer"]) and not coded:
                coded = True
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": ITEM_CODE_NUDGE}]
                continue
            if not direction_checked and _needs_direction_review(obj["answer"], trace):
                direction_checked = True
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": DIRECTION_CHECK_NUDGE}]
                continue
            bad = ungrounded_numbers(obj["answer"], ctx + trace)
            if bad and not retried:
                retried = True
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": _grounding_retry_msg(bad)}]
                continue
            return _finalise(obj["answer"], trace, "json", ctx)
        result = _run_tool(obj.get("tool"), obj.get("args", {}), seen)
        trace.append((obj.get("tool"), obj.get("args", {}), result))
        messages += [{"role": "assistant", "content": json.dumps(obj)},
                     {"role": "user", "content": f"Tool result:\n{_trim(result)}"}]
    return {"answer": "I couldn't finish within the step limit; try a narrower question.",
            "trace": trace, "mode": "json"}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Which service has the lowest % agree, and are there equity gaps?"
    out = ask(q)
    for name, args, res in out["trace"]:
        print(f"[{name}] {args}")
    print("\n" + out["answer"])


# ---- answer hygiene (added by scripts/install_answer_hygiene.py) ----------------------
# 1) Numbers written the way people write them ("1,717", "9.1e-05") are read as one number before
#    they are checked against the tool results, so correct figures are not flagged.
# 2) When the model is asked to rewrite an answer, it must return the answer only. Staff never see
#    the request, so "good catch, I re-checked..." makes no sense to them.
# Number check: agent.ungrounded_numbers (Jonathan's version) already reads "1,717" and
# scientific notation, so it is not wrapped here.


_ANSWER_ONLY = (" Reply with the complete answer for the staff member and nothing else: do not mention this "
                "note, and do not say that anything was checked again, corrected or removed.")
for _name in ("NO_TOOL_NUDGE", "OFF_TOPIC_NUDGE", "HANDOFF_NUDGE", "ITEM_CODE_NUDGE", "FINAL_NUDGE",
              "DIRECTION_CHECK_NUDGE"):
    if _name in globals():
        globals()[_name] = globals()[_name] + _ANSWER_ONLY
_retry_msg = _grounding_retry_msg


def _grounding_retry_msg(bad):
    return _retry_msg(bad) + _ANSWER_ONLY

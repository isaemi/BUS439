"""Question-answering loop: staff question -> Qwen picks tools -> answer.

Works in two modes (config.TOOL_MODE):
  native  OpenAI-style tool calling through the gateway
  json    the model replies {"tool": ..., "args": ...} or {"answer": ...};
          used automatically if the gateway rejects the `tools` parameter
"""
import json
import re

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
- Answer codes like '1' or '2' are unlabelled; never guess what they mean. Filter on the
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
- Routing: a faculty or school ("Arts students", "engineers") -> compare_groups with
  dimension="faculty" and the service; match faculty_values exactly (e.g. "Arts").
  Any other group -> compare_groups with that dimension.
- Questions combining two groups (e.g. international first-generation students) need
  run_sql on ratings_demo with both columns in GROUP BY; compare_groups covers only one.
- If no tool result covers the exact group or service asked about, say so in the FIRST
  sentence. Never present results for everyone as if they were for a subgroup.
- For questions about differences between groups (equity), call equity_flags. A gap is
  only "significant" if it appears in flags. compare_groups is descriptive only.
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
                       "likert": d["likert"], "tables": d["tables"], "rules": d["rules"],
                       "data_coverage": d["data_coverage"]})


def _system(json_mode):
    s = SYSTEM.format(min_cell=config.MIN_CELL, overview=_overview())
    if json_mode:
        tl = "\n".join(f"- {t['function']['name']}({', '.join(t['function']['parameters']['properties'])}): "
                       f"{t['function']['description']}" for t in tools.TOOL_SPECS)
        s += JSON_PROTOCOL.format(tool_list=tl)
    return s


_CODE = re.compile(r"\b[A-Z]+\d*_\d+\b")          # item codes like WA1_10
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_FREE = {str(i) for i in range(0, 11)} | {"100"}     # scale points, counts like "3 points"

_INTERNAL = re.compile(r"\b(" + "|".join(
    [t["function"]["name"] for t in tools.TOOL_SPECS] +
    ["ratings_demo", "response_multi", "response_faculty", "response_disability",
     "items table", "responses table", "ratings table"]) + r")\b")

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
        for d in (0, 1, 2, 3):
            v = round(float(x), d)
            found.add(f"{v:.{d}f}")
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


def ungrounded_numbers(answer, trace):
    """Numbers in the answer that appear in no tool result (or as a gap between two
    result figures). Catches invented figures."""
    text = _CODE.sub("", answer or "")
    found, by_key = _result_numbers(trace)
    known = found | _differences(by_key)
    bad = []
    for n in _NUM.findall(text):
        norm = n.rstrip("0").rstrip(".") if "." in n else n
        if n in _FREE or n in known or norm in known:
            continue
        bad.append(n)
    return sorted(set(bad))


def _grounding_retry_msg(bad):
    return (f"These numbers are not in any tool result: {', '.join(bad)}. Rewrite the answer "
            "using only figures that appear in the tool results, or call a tool to get them.")


_SIG = re.compile(r"statistically|significan", re.IGNORECASE)
_SIG_NOT_A_CLAIM = re.compile(
    r"\b(identify|identifies|test|tests|testing|check|checks|look for|detect|whether|could|"
    r"would|might|run)\b", re.IGNORECASE)


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
    bad = ungrounded_numbers(answer, list(ctx) + trace)
    if bad:
        answer += (f"\n\n⚠️ Check before using: these figures could not be traced to the data: "
                   f"{', '.join(bad)}.")
    if _claims_significance(answer) and not _has_test(list(ctx) + trace):
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
                 for name, args, res in turn.get("trace", []) if name not in ("fallback", "empty_reply")]
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


def ask(question, history=None, max_steps=6, chat_fn=None):
    """Returns {"answer": str, "trace": [(tool, args, result), ...], "mode": str}.
    `chat_fn` lets tests inject a fake model."""
    chat_fn = chat_fn or llm.chat
    mode = config.TOOL_MODE
    trace = []
    ctx = _context_trace(question, history)
    if mode in ("native", "auto"):
        try:
            return _ask_native(question, history, max_steps, chat_fn, trace, ctx)
        except BadRequestError as e:
            if mode == "native":
                raise
            trace.append(("fallback", {}, f"native tool calling rejected: {e}"))
    return _ask_json(question, history, max_steps, chat_fn, trace, ctx)


def _ask_native(question, history, max_steps, chat_fn, trace, ctx):
    messages = [{"role": "system", "content": _system(False)}, *_history_messages(history),
                {"role": "user", "content": question}]
    nudged = retried = handed_off = False
    empties = 0
    seen = {}
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
            tool_steps = [t for t in trace if t[0] not in ("fallback", "empty_reply")]
            if not tool_steps and not nudged:
                nudged = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": NO_TOOL_NUDGE}]
                continue
            if mentions_internals(msg.content) and not handed_off:
                handed_off = True
                messages += [{"role": "assistant", "content": msg.content or ""},
                             {"role": "user", "content": HANDOFF_NUDGE}]
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
            result = _run_tool(tc.function.name, args, seen)
            trace.append((tc.function.name, args, result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": _trim(result)})
    return {"answer": "I couldn't finish within the step limit; try a narrower question.",
            "trace": trace, "mode": "native"}


def _ask_json(question, history, max_steps, chat_fn, trace, ctx):
    messages = [{"role": "system", "content": _system(True)}, *_history_messages(history),
                {"role": "user", "content": question}]
    nudged = retried = handed_off = False
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
            tool_steps = [t for t in trace if t[0] not in ("fallback", "empty_reply")]
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

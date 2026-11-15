"""
tools.py: the only code that touches the data. The model never writes SQL here;
it calls one of four functions, which run fixed, parameterized SQL on a
read-only connection and return small, labelled results.

    tools = SHWTools("shw.duckdb")
    tools.average_rating(item="booking_ease", by="faculty", service="counselling")
    tools.call_tool("trend", '{"item": "satisfied_overall", "service": "counselling"}')

Every result is a dict with: tool, description, rows, notes, chart_hint.
Bad input raises ToolError with a message written for the model to read and retry;
call_tool() turns that into {"error": "..."}.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
ALIASES_PATH = HERE / "aliases.csv"

LOW_N = 10       # groups with fewer respondents are flagged low_confidence (placeholder; agree with SHW)
MIN_PERIODS = 3  # trend() needs at least this many periods with n >= LOW_N
Z = 1.96         # 95% confidence intervals

# `by` / `column` options that live on the responses table (fixed SQL fragments, never user text)
RESPONSE_COLUMNS = [
    "service_type", "learned_about", "wait_time", "visit_format_preferred", "fell_short",
    "program", "new_to_ubc", "year_of_study", "enrollment", "age_band",
    "student_origin", "living", "indigenous", "racialized",
]
BY_OPTIONS = ["service", "faculty", "month", "item"] + [c for c in RESPONSE_COLUMNS if c != "service_type"]
ORDINAL = {"wait_time", "year_of_study", "age_band", "month", "score"}
AGE_ORDER = ["under_20", "20_24", "25_29", "30_39", "40_plus", "prefer_not_to_disclose"]
AGREE_LABELS = {1: "Strongly disagree", 2: "Disagree", 3: "Neutral", 4: "Agree", 5: "Strongly agree"}

# When an item isn't asked for a service, suggest these instead.
RELATED_ITEMS = {
    "satisfied_overall": ["recommend_shw", "positive_provider"],
    "academic_demands": ["academic_work"], "academic_work": ["academic_demands"],
    "recommend_shw": ["recommend_info_desk"], "recommend_info_desk": ["recommend_shw"],
    "felt_heard": ["listened"], "listened": ["felt_heard"],
    "timely": ["booking_ease"],
}


class ToolError(Exception):
    """Bad input. The message is shown to the model so it can correct the call."""


def _norm(s) -> str:
    s = str(s).translate(dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014"), "-"))
    return re.sub(r"[\s_]+", " ", s).strip().casefold()


def _term(month: str) -> str:
    """UBC terms: Sep-Dec = Winter Term 1, Jan-Apr = Winter Term 2, May-Aug = Summer."""
    year, m = int(month[:4]), int(month[5:7])
    if m >= 9:
        return f"{year}W1"
    if m <= 4:
        return f"{year - 1}W2"
    return f"{year}S"


class SHWTools:
    def __init__(self, db_path: str | Path = "shw.duckdb"):
        if not Path(db_path).exists():
            raise FileNotFoundError(f"{db_path} not found. Run clean.py first.")
        # Read-only, and no file/network access: closes the gaps found in M3.
        self.con = duckdb.connect(
            str(db_path), read_only=True, config={"enable_external_access": False}
        )
        self._load_reference_data()

    # ------------------------------------------------------------------ setup
    def _q(self, sql: str, params: list | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def _load_reference_data(self):
        d = self._q("SELECT * FROM dictionary")
        vs = self._q("SELECT * FROM value_sets ORDER BY rowid")
        ratings = d[d["table"] == "ratings"]

        self.item_text = ratings.drop_duplicates("item_key").set_index("item_key")["item_text"].to_dict()
        self.items_by_service = ratings.groupby("service")["item_key"].apply(list).to_dict()
        self.services_by_item = ratings.groupby("item_key")["service"].apply(list).to_dict()

        # code -> label (first label listed), and per-set code order for natural sorting
        self.labels: dict[str, dict[str, str]] = {}
        self.order: dict[str, list[str]] = {}
        for vset, grp in vs.groupby("value_set", sort=False):
            codes = list(dict.fromkeys(grp["code"]))
            self.order[vset] = codes
            self.labels[vset] = grp.drop_duplicates("code").set_index("code")["label"].to_dict()
        self.column_value_set = (
            d[d["table"] == "responses"].set_index("clean_name")["value_set"].to_dict()
        )
        self.column_value_set["service_type"] = "service"

        self.services = self.order["service"]
        self.faculties = self._q("SELECT DISTINCT faculty FROM faculty_selected ORDER BY 1")["faculty"].tolist()
        self.months = self._q(
            "SELECT DISTINCT recorded_month FROM responses WHERE recorded_month IS NOT NULL ORDER BY 1"
        )["recorded_month"].tolist()

        self.aliases: dict[str, dict[str, str]] = {"service": {}, "faculty": {}, "item": {}}
        if ALIASES_PATH.exists():
            for row in pd.read_csv(ALIASES_PATH, dtype=str).itertuples():
                self.aliases[row.kind][_norm(row.alias)] = row.code
        # labels and codes are aliases too ("Medical appointment", "Applied Science (Engineering)")
        for code, label in self.labels["service"].items():
            self.aliases["service"].setdefault(_norm(label), code)
        for code, label in self.labels["faculty"].items():
            self.aliases["faculty"].setdefault(_norm(label), code)

    # ------------------------------------------------------------- resolvers
    @staticmethod
    def _suggest(value: str, options: list[str]) -> str:
        close = difflib.get_close_matches(_norm(value), [_norm(o) for o in options], n=3, cutoff=0.5)
        return f" Did you mean: {', '.join(close)}?" if close else ""

    def _resolve(self, kind: str, value: str, valid: list[str]) -> str:
        if value in valid:
            return value
        n = _norm(value)
        for code in valid:
            if _norm(code) == n:
                return code
        if n in self.aliases.get(kind, {}):
            code = self.aliases[kind][n]
            if code not in valid:
                raise ToolError(f"No responses in the data for {kind} '{code}' (from '{value}').")
            return code
        raise ToolError(
            f"Unknown {kind} '{value}'.{self._suggest(value, valid)} Valid {kind} values: {', '.join(valid)}."
        )

    def _service(self, service):
        return None if service in (None, "", "all") else self._resolve("service", service, self.services)

    def _faculties(self, faculty) -> list[str] | None:
        if faculty in (None, "", "all"):
            return None
        values = faculty if isinstance(faculty, list) else [faculty]
        return [self._resolve("faculty", f, self.faculties) for f in values]

    def _month(self, month):
        if month in (None, "", "all"):
            return None
        if month == "latest":
            return self.months[-1]
        if month not in self.months:
            raise ToolError(f"No data for month '{month}'. Months available: {', '.join(self.months)}.")
        return month

    def _item(self, item, service):
        if item in (None, "", "all"):
            return None
        key = self._resolve("item", item, list(self.item_text))
        if service and key not in self.items_by_service.get(service, []):
            asked = self.items_by_service.get(service, [])
            related = [r for r in RELATED_ITEMS.get(key, []) if r in asked]
            hint = f" Closest items asked for {service}: {', '.join(related)}." if related else ""
            raise ToolError(
                f"'{key}' is not asked for {service}; it is asked for: "
                f"{', '.join(self.services_by_item.get(key, []))}.{hint} "
                f"Items asked for {service}: {', '.join(asked)}. Or use item='all'."
            )
        return key

    def _by(self, by, allowed):
        if by in (None, "", "none"):
            return None
        if by not in allowed:
            raise ToolError(f"Unknown by='{by}'.{self._suggest(by, allowed)} Options: {', '.join(allowed)}.")
        return by

    # ------------------------------------------------------------- helpers
    def _label(self, column: str, code) -> str:
        if code is None or (isinstance(code, float) and math.isnan(code)):
            return "(no answer)"
        if column in ("service", "service_type"):
            return self.labels["service"].get(code, code)
        if column == "faculty":
            return self.labels["faculty"].get(code, code)
        if column == "item":
            return self.item_text.get(code, code)
        if column == "score":
            return AGREE_LABELS.get(int(code), str(code))
        vset = self.column_value_set.get(column)
        if vset and vset in self.labels:
            return self.labels[vset].get(code, code)
        return str(code)

    def _sort_key(self, column: str):
        if column == "age_band":
            return lambda v: AGE_ORDER.index(v) if v in AGE_ORDER else 99
        if column in ("month", "score"):
            return lambda v: v
        vset = self.column_value_set.get(column)
        order = self.order.get(vset, [])
        return lambda v: order.index(v) if v in order else 99

    @staticmethod
    def _ci(mean, sd, n):
        if n is None or n < 2 or sd is None or (isinstance(sd, float) and math.isnan(sd)):
            return None, None
        se = sd / math.sqrt(n)
        return round(mean - Z * se, 2), round(mean + Z * se, 2)

    @staticmethod
    def _compare(a: dict, b: dict) -> dict:
        """Is the gap between two group means bigger than noise? (Welch-style 95% CI)."""
        if min(a["n"], b["n"]) < 2 or a.get("_sd") is None or b.get("_sd") is None:
            return {"verdict": "insufficient_data"}
        diff = a["avg_score"] - b["avg_score"]
        se = math.sqrt(a["_sd"] ** 2 / a["n"] + b["_sd"] ** 2 / b["n"])
        lo, hi = diff - Z * se, diff + Z * se
        verdict = "real_difference" if (lo > 0 or hi < 0) else "within_noise"
        return {"difference": round(diff, 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2), "verdict": verdict}

    def _filters(self, service, faculties, month, item=None, rating_alias="r"):
        """Build WHERE clauses with parameters. Column names are fixed strings, values are parameters."""
        where, params = [], []
        if service:
            where.append(f"{rating_alias}.service = ?" if rating_alias else "resp.service_type = ?")
            params.append(service)
        if item:
            where.append(f"{rating_alias}.item_key = ?")
            params.append(item)
        if faculties:
            where.append(f"f.faculty IN ({', '.join('?' * len(faculties))})")
            params.extend(faculties)
        if month:
            where.append("resp.recorded_month = ?")
            params.append(month)
        return ("WHERE " + " AND ".join(where)) if where else "", params

    @staticmethod
    def _group_expr(by):
        return {
            None: "'all'", "service": "r.service", "faculty": "f.faculty",
            "month": "resp.recorded_month", "item": "r.item_key",
        }.get(by, f"resp.{by}" if by in RESPONSE_COLUMNS else None)

    def _describe(self, what, service, faculties, month, by=None):
        parts = [what]
        if by:
            parts.append(f"by {by}")
        parts.append(f"for {self._label('service', service)}" if service else "across all services")
        if faculties:
            parts.append("faculty: " + ", ".join(self._label("faculty", f) for f in faculties))
        parts.append(f"month: {month}" if month else f"all months ({self.months[0]} to {self.months[-1]})")
        return "; ".join(parts)

    # ================================================================ TOOLS
    def list_options(self, service: str | None = None) -> dict:
        """What can be asked: services, items per service, faculties, months, grouping options."""
        svc = self._service(service)
        services = [svc] if svc else self.services
        items = {
            s: [{"item": k, "text": self.item_text[k]} for k in self.items_by_service.get(s, [])]
            for s in services
        }
        fac_n = self._q(
            "SELECT faculty, COUNT(DISTINCT response_id) AS n FROM faculty_selected GROUP BY 1 ORDER BY n DESC"
        )
        return {
            "tool": "list_options",
            "description": "Available services, rating items, faculties and months",
            "services": [{"code": s, "label": self._label("service", s)} for s in services],
            "items_by_service": items,
            "faculties": [
                {"code": r.faculty, "label": self._label("faculty", r.faculty), "n": int(r.n),
                 "low_confidence": int(r.n) < LOW_N}
                for r in fac_n.itertuples()
            ],
            "months": self.months,
            "by_options": BY_OPTIONS,
            "distribution_columns": RESPONSE_COLUMNS + ["faculty", "<any rating item>"],
            "notes": [
                f"Groups with fewer than {LOW_N} respondents are flagged low_confidence.",
                "Survey responses measure service experience, not student wellbeing.",
            ],
            "chart_hint": None,
        }

    def average_rating(self, item: str = "all", by: str | None = None, service: str | None = None,
                       faculty=None, month: str | None = None) -> dict:
        """Average 1-5 rating, optionally grouped. Scores are averaged per respondent first,
        so respondents who answered many items don't count more than others."""
        svc = self._service(service)
        item_key = self._item(item, svc)
        facs = self._faculties(faculty)
        mon = self._month(month)
        by = self._by(by, BY_OPTIONS)
        if by == "item" and item_key:
            raise ToolError("by='item' compares items, so use item='all' with it.")

        group = self._group_expr(by)
        join_f = "JOIN faculty_selected f USING (response_id)" if (facs or by == "faculty") else ""
        where, params = self._filters(svc, facs, mon, item_key)
        sql = f"""
            WITH per_respondent AS (
                SELECT {group} AS grp, r.response_id, AVG(r.score) AS s
                FROM ratings r JOIN responses resp USING (response_id) {join_f}
                {where}
                GROUP BY 1, 2
            )
            SELECT grp, AVG(s) AS avg_score, STDDEV_SAMP(s) AS sd, COUNT(*) AS n
            FROM per_respondent GROUP BY 1
        """
        df = self._q(sql, params)
        if df.empty:
            raise ToolError("No ratings match these filters. Try removing a filter (e.g. month or faculty).")

        missing = df[df["grp"].isna()]
        df = df[df["grp"].notna()]
        if df.empty:
            raise ToolError(f"No respondents answered '{by}' for these filters.")
        rows = []
        for r in df.itertuples():
            lo, hi = self._ci(r.avg_score, r.sd, r.n)
            rows.append({
                (by or "group"): r.grp, "label": self._label(by or "", r.grp) if by else "all",
                "avg_score": round(r.avg_score, 2), "n": int(r.n), "ci_low": lo, "ci_high": hi,
                "low_confidence": int(r.n) < LOW_N,
                "_sd": None if pd.isna(r.sd) else float(r.sd),
            })
        if by in ORDINAL:
            rows.sort(key=lambda x: self._sort_key(by)(x[by]))
        else:
            rows.sort(key=lambda x: x["avg_score"])

        notes = [f"Scale 1-5 (5 = strongly agree). n = number of respondents. "
                 f"low_confidence = fewer than {LOW_N} respondents; treat as indicative only."]
        if by and not missing.empty:
            notes.append(f"{int(missing['n'].sum())} respondents didn't answer '{by}' and are excluded.")
        if item_key and not svc:
            notes.append(f"'{item_key}' is only asked for: {', '.join(self.services_by_item[item_key])}.")
        if not item_key:
            notes.append("item='all': each respondent's average across the items their service asks.")
        if by == "faculty" or facs:
            notes.append("Students in more than one faculty are counted in each, so totals can exceed respondents.")
        low = [x["label"] for x in rows if x["low_confidence"]]
        if low and len(rows) > 1:
            notes.append(f"{len(low)} of {len(rows)} groups are low confidence: {', '.join(map(str, low))}.")

        result = {
            "tool": "average_rating",
            "description": self._describe(
                f"Average rating ({item_key or 'all items'})", svc, facs, mon, by),
            "rows": rows, "notes": notes,
        }
        if len(rows) == 2:
            result["comparison"] = self._compare(rows[0], rows[1]) | {
                "groups": [rows[0]["label"], rows[1]["label"]]}
        result["chart_hint"] = self._chart_for_rating(by, rows)
        for x in rows:
            x.pop("_sd", None)
        return result

    def distribution(self, column: str, by: str | None = None, service: str | None = None,
                     faculty=None, month: str | None = None) -> dict:
        """Counts and percentages of answers. `column` is a survey question (e.g. wait_time)
        or a rating item (e.g. booking_ease) for the % who agree/disagree."""
        svc = self._service(service)
        facs = self._faculties(faculty)
        mon = self._month(month)
        by = self._by(by, [b for b in BY_OPTIONS if b != "item"])

        is_rating = False
        if column in RESPONSE_COLUMNS or column == "faculty":
            col = column
        else:
            try:
                col = self._item(column, svc)
                is_rating = True
            except ToolError:
                options = RESPONSE_COLUMNS + ["faculty"]
                raise ToolError(
                    f"Unknown column '{column}'.{self._suggest(column, options + list(self.item_text))} "
                    f"Survey questions: {', '.join(options)}. Or a rating item, e.g. booking_ease."
                )
        if by == column:
            raise ToolError("by and column are the same; pick a different grouping.")

        group = self._group_expr(by).replace("r.service", "resp.service_type").replace("r.item_key", "NULL")
        need_f = facs or by == "faculty" or col == "faculty"
        join_f = "JOIN faculty_selected f USING (response_id)" if need_f else ""
        if is_rating:
            value = "r.score"
            where, params = self._filters(svc, facs, mon, col)
            source = f"ratings r JOIN responses resp USING (response_id) {join_f}"
            vcol = "score"
        else:
            value = "f.faculty" if col == "faculty" else f"resp.{col}"
            where, params = self._filters(svc, facs, mon, rating_alias=None)
            source = f"responses resp {join_f}"
            vcol = col
            where = (where + " AND " if where else "WHERE ") + f"{value} IS NOT NULL"

        sql = f"""
            SELECT {group} AS grp, {value} AS value, COUNT(DISTINCT resp.response_id) AS n
            FROM {source} {where}
            GROUP BY 1, 2
        """
        df = self._q(sql, params)
        if df.empty:
            raise ToolError("No answers match these filters (the question may not be asked for this service).")

        n_missing_by = int(df.loc[df["grp"].isna(), "n"].sum()) if by else 0
        df = df[df["grp"].notna()]
        rows = []
        for grp, g in df.groupby("grp", sort=False):
            total = int(g["n"].sum())
            g = g.copy()
            g["pct"] = (100 * g["n"] / total).round(1)
            if vcol in ORDINAL:
                g = g.sort_values("value", key=lambda s: s.map(self._sort_key(vcol)))
            else:
                g = g.sort_values("n", ascending=False)
            entry = {
                (by or "group"): grp, "label": self._label(by or "", grp) if by else "all",
                "n": total, "low_confidence": total < LOW_N,
                "values": [{"value": (int(v) if is_rating else v), "label": self._label(vcol, v),
                            "count": int(c), "pct": float(p)}
                           for v, c, p in zip(g["value"], g["n"], g["pct"])],
            }
            if is_rating:
                entry["pct_agree"] = round(float(g.loc[g["value"] >= 4, "pct"].sum()), 1)
                entry["pct_disagree"] = round(float(g.loc[g["value"] <= 2, "pct"].sum()), 1)
            rows.append(entry)
        if by:
            rows.sort(key=lambda x: self._sort_key(by)(x[by]) if by in ORDINAL else -x["n"])

        notes = ["pct = share of respondents who answered this question (blank answers excluded)."]
        if n_missing_by:
            notes.append(f"{n_missing_by} respondents didn't answer '{by}' and are excluded.")
        if col == "service_type":
            notes.append("These are survey response counts, not appointment counts.")
        if is_rating:
            notes.append("pct_agree = Agree + Strongly agree; pct_disagree = Disagree + Strongly disagree.")
        if col == "faculty" or need_f:
            notes.append("Students in more than one faculty are counted in each.")
        if mon and mon == self.months[-1]:
            notes.append(f"{mon} is the most recent month and may be incomplete.")

        return {
            "tool": "distribution",
            "description": self._describe(f"Distribution of {col}", svc, facs, mon, by),
            "rows": rows, "notes": notes,
            "chart_hint": self._chart_for_distribution(by, rows, vcol, is_rating),
        }

    def trend(self, item: str = "all", service: str | None = None, faculty=None,
              period: str = "month") -> dict:
        """Change in average rating over time. The verdict is decided here, not by the model:
        it needs MIN_PERIODS periods with at least LOW_N respondents, then compares the
        earliest and latest of those with a 95% confidence interval."""
        svc = self._service(service)
        item_key = self._item(item, svc)
        facs = self._faculties(faculty)
        if period not in ("month", "term"):
            raise ToolError("period must be 'month' or 'term'.")

        join_f = "JOIN faculty_selected f USING (response_id)" if facs else ""
        where, params = self._filters(svc, facs, None, item_key)
        df = self._q(f"""
            SELECT resp.recorded_month AS month, r.response_id, AVG(r.score) AS s
            FROM ratings r JOIN responses resp USING (response_id) {join_f}
            {where}
            GROUP BY 1, 2
        """, params)
        if df.empty:
            raise ToolError("No ratings match these filters.")
        df["period"] = df["month"] if period == "month" else df["month"].map(_term)
        agg = df.groupby("period")["s"].agg(["mean", "std", "count"]).reset_index().sort_values("period")

        rows = []
        for r in agg.itertuples():
            lo, hi = self._ci(r.mean, r.std, r.count)
            rows.append({"period": r.period, "avg_score": round(r.mean, 2), "n": int(r.count),
                         "ci_low": lo, "ci_high": hi, "low_confidence": int(r.count) < LOW_N,
                         "_sd": None if pd.isna(r.std) else float(r.std)})

        usable = [x for x in rows if not x["low_confidence"]]
        if len(usable) < MIN_PERIODS:
            verdict = {"verdict": "insufficient_data",
                       "reason": f"Only {len(usable)} {period}(s) have {LOW_N}+ respondents; "
                                 f"{MIN_PERIODS} are needed to judge a trend."}
        else:
            first, last = usable[0], usable[-1]
            cmp = self._compare(last, first)
            verdict = {
                "verdict": {"real_difference": "increasing" if cmp.get("difference", 0) > 0 else "decreasing",
                            "within_noise": "no_clear_change"}.get(cmp["verdict"], "insufficient_data"),
                "compared": [first["period"], last["period"]],
                "change": cmp.get("difference"), "ci_low": cmp.get("ci_low"), "ci_high": cmp.get("ci_high"),
            }
        for x in rows:
            x.pop("_sd", None)

        notes = [
            "Report the verdict as given. Do not describe changes between individual periods as trends.",
            "Method: earliest vs latest period with enough respondents, 95% confidence interval.",
        ]
        if not item_key:
            notes.append("item='all': each respondent's average across the items their service asks.")
        if item_key and not svc:
            notes.append(f"'{item_key}' is only asked for: {', '.join(self.services_by_item[item_key])}.")
        return {
            "tool": "trend",
            "description": self._describe(f"Trend in average rating ({item_key or 'all items'}) by {period}",
                                          svc, facs, None),
            "rows": rows, "trend": verdict, "notes": notes,
            "chart_hint": None if len(rows) < 2 else {
                "type": "line_band", "x": "period", "y": "avg_score", "band": ["ci_low", "ci_high"],
                "y_range": [1, 5], "fade": "low_confidence"},
        }

    # ----------------------------------------------------------- chart hints
    @staticmethod
    def _chart_for_rating(by, rows):
        if not by or len(rows) <= 2:
            return None
        kind = "bar_ordered" if by in ORDINAL else "bar_sorted"
        return {"type": kind, "x": "label", "y": "avg_score", "error": ["ci_low", "ci_high"],
                "y_range": [1, 5], "fade": "low_confidence"}

    @staticmethod
    def _chart_for_distribution(by, rows, column, is_rating):
        volume = column == "service_type"  # response counts matter more than shares
        if by:
            return {"type": "stacked_bar" if volume else "stacked_100", "x": "label", "stack": "values",
                    "y": "count" if volume else "pct", "fade": "low_confidence"}
        if len(rows[0]["values"]) <= 2:
            return None
        return {"type": "bar" if volume else "bar_pct", "x": "values.label",
                "y": "values.count" if volume else "values.pct",
                "ordered": column in ORDINAL or is_rating}

    # ------------------------------------------------------------ dispatcher
    def call_tool(self, name: str, arguments) -> dict:
        """Entry point for chat.py: run a tool by name with JSON or dict arguments."""
        tools = {"list_options": self.list_options, "average_rating": self.average_rating,
                 "distribution": self.distribution, "trend": self.trend}
        if name not in tools:
            return {"error": f"Unknown tool '{name}'. Tools: {', '.join(tools)}."}
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
            return tools[name](**args)
        except ToolError as e:
            return {"error": str(e)}
        except TypeError as e:
            return {"error": f"Bad arguments for {name}: {e}"}
        except json.JSONDecodeError:
            return {"error": "Arguments were not valid JSON."}


# ------------------------------------------------------------------ schemas
# Sent to the model by chat.py (OpenAI-compatible "tools" format).
_FILTERS = {
    "service": {"type": "string", "description": "Optional. medical, counselling, wellness_advising, "
                "info_desk, sexual_health_shop or naloxone. Omit for all services."},
    "faculty": {"description": "Optional. One faculty or a list, e.g. 'science' or ['arts','science']. "
                "Common names like 'Sauder' or 'Engineering' are accepted.",
                "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
    "month": {"type": "string", "description": "Optional. 'YYYY-MM' or 'latest'. Omit for all months."},
}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_options",
        "description": "Start here if unsure what exists. Lists services, the rating items each service "
                       "asks (with question wording), faculties with response counts, and months.",
        "parameters": {"type": "object", "properties": {"service": _FILTERS["service"]}}}},
    {"type": "function", "function": {
        "name": "average_rating",
        "description": "Average 1-5 agreement rating, with n and 95% CI. Use for 'how satisfied', "
                       "'which X is lowest/highest', comparing groups, and a service's strengths/weaknesses "
                       "(by='item'). Each item is only asked for some services; an error will say which.",
        "parameters": {"type": "object", "properties": {
            "item": {"type": "string", "description": "Rating item, e.g. booking_ease, timely, access, "
                     "satisfied_overall, welcomed, front_desk. 'all' = average of all items (default)."},
            "by": {"type": "string", "enum": BY_OPTIONS, "description": "Optional grouping."},
            **_FILTERS}}}},
    {"type": "function", "function": {
        "name": "distribution",
        "description": "Counts and % of answers. Use for wait times, fell_short, visit format, how students "
                       "heard about a service, response volume (column='service_type'), or % agree on a "
                       "rating item (column=item name).",
        "parameters": {"type": "object", "required": ["column"], "properties": {
            "column": {"type": "string", "description": "A survey question (wait_time, fell_short, "
                       "learned_about, visit_format_preferred, service_type, faculty, program, "
                       "year_of_study, ...) or a rating item (booking_ease, ...)."},
            "by": {"type": "string", "enum": [b for b in BY_OPTIONS if b != "item"],
                   "description": "Optional grouping."},
            **_FILTERS}}}},
    {"type": "function", "function": {
        "name": "trend",
        "description": "Whether an average rating is changing over time. Returns a verdict "
                       "(increasing, decreasing, no_clear_change, insufficient_data) that must be "
                       "reported as given. Use for 'improving', 'getting worse', 'any trends'.",
        "parameters": {"type": "object", "properties": {
            "item": {"type": "string", "description": "Rating item or 'all' (default)."},
            "service": _FILTERS["service"], "faculty": _FILTERS["faculty"],
            "period": {"type": "string", "enum": ["month", "term"], "description": "Default month."}}}}},
]

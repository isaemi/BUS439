"""Tools the LLM may call. Each returns a JSON-serialisable dict.

Guardrails live here, not in the prompt:
  * read-only DuckDB connection with external file/network access disabled
  * single SELECT statement only; comments table not reachable from SQL
  * no response-level output (response_id may not appear in results)
  * grouped results must carry a count column `n`; cells below MIN_CELL are dropped
"""
import json
import re

import duckdb
import numpy as np
import pandas as pd

from . import analysis, config
from .ingest import load_dictionary
from .privacy import is_count_col, suppress_small_cells

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|"
    r"import|call|set|read_csv\w*|read_parquet|read_json\w*|glob|comments)\b", re.IGNORECASE)


def _con():
    return analysis.connect()


def _df_out(df, hidden=0, note=None):
    df = df.head(config.MAX_SQL_ROWS)
    out = {"rows": json.loads(df.to_json(orient="records", date_format="iso")), "row_count": len(df)}
    if hidden:
        out["suppressed_rows"] = hidden
        out["suppression_note"] = f"{hidden} group(s) hidden because n < {config.MIN_CELL}"
    if note:
        out["note"] = note
    return out


def describe_data():
    d = load_dictionary()
    con = _con()
    from .schema import describe
    s = describe(con, include=["responses", "ratings", "items", "response_faculty",
                               "response_disability", "response_multi", "appointments"])
    cov = dict(d.get("coverage", {}))
    if analysis.has_chr(con):
        cov["available"] = cov.get("available", []) + cov.get("with_chr", [])
        cov["chr_status"] = _chr_source(con)
    else:
        cov["not_yet"] = cov.get("without_chr", []) + cov.get("not_yet", [])
        cov["chr_status"] = "not loaded"
    cov.pop("with_chr", None)
    cov.pop("without_chr", None)
    services = {v["key"]: v["name"] for v in d["services"].values()}
    return {
        "tables": s["tables"],
        "views": s["views"],
        "relationships": s["relationships"],
        "services": services,
        "likert": "score 1-5, " + ", ".join(f"{k}={v}" for k, v in d["likert"]["scale"].items())
                  + "; agree = score >= 4",
        "equity_dimensions": analysis.equity_dimensions(d),
        "faculty_values": d["checkbox"]["faculty"]["choices"],
        "category_values": {k: v for k, v in _category_values().items()
                            if k not in ("faculty", "disability") and len(v) <= 12},
        "data_coverage": {**cov, "survey_period": _period()},
        "rules": [f"Grouped queries must include COUNT(DISTINCT response_id) AS n",
                  f"Groups with n < {config.MIN_CELL} are removed from results",
                  "Never select response_id; comments are only reachable via search_comments"],
    }


_LITERAL = re.compile(r"(?:\b\w+\.)?\b(\w+)\s*(?:=|<>|!=)\s*'([^']*)'")


def _category_values():
    """Distinct values of every text column in responses (demographics, follow-ups)
    plus the multi-select long tables. Used to catch filters on values that don't exist."""
    con = _con()
    vals = {}
    cols = con.execute("DESCRIBE responses").df()
    for c in cols[cols["column_type"] == "VARCHAR"]["column_name"]:
        if c in ("response_id", "channel"):
            continue
        vals[c] = sorted(con.execute(f'SELECT DISTINCT "{c}" FROM responses WHERE "{c}" IS NOT NULL').df()[c].astype(str))
    vals["disability"] = sorted(con.execute("SELECT DISTINCT disability FROM response_disability").df()["disability"])
    vals["faculty"] = sorted(con.execute("SELECT DISTINCT faculty FROM response_faculty").df()["faculty"])
    vals["service"] = sorted(con.execute("SELECT DISTINCT service FROM ratings").df()["service"])
    if analysis.has_chr(con):
        for c in ("faculty", "academic_level", "gender", "service_line", "visit_type", "mode",
                  "outcome", "term", "age_band", "program"):
            chr_vals = con.execute(f'SELECT DISTINCT "{c}" FROM appointments WHERE "{c}" IS NOT NULL').df()[c]
            vals[c] = sorted(set(vals.get(c, [])) | set(chr_vals.astype(str)))
    return vals


def _check_literals(sql):
    """Return an error if the query filters a category column on a value it never has
    (e.g. intl_or_domestic = 'international' when the stored codes are '1'/'2')."""
    vals = _category_values()
    for col, lit in _LITERAL.findall(sql):
        if col in vals and lit not in vals[col] and len(vals[col]) <= 40:
            shown = ", ".join(f"'{v}'" for v in vals[col])
            return (f"Value '{lit}' does not exist in column {col}. Valid values: {shown}. "
                    "Numeric codes are unlabelled answer options: use them as-is and do not "
                    "guess their meaning.")
    return None


def _period():
    lo, hi = analysis.period_range(_con())
    return f"{lo:%Y-%m-%d} to {hi:%Y-%m-%d}" if lo and hi else "unknown"


def run_sql(sql: str):
    s = sql.strip().rstrip(";")
    if ";" in s:
        return {"error": "Only one statement is allowed."}
    if not re.match(r"^\s*(select|with)\b", s, re.IGNORECASE):
        return {"error": "Only SELECT queries are allowed."}
    if _FORBIDDEN.search(s):
        return {"error": f"Query uses a forbidden keyword or table: {_FORBIDDEN.search(s).group(0)}"}
    bad_value = _check_literals(s)
    if bad_value:
        return {"error": bad_value}
    try:
        df = _con().execute(s).df()
    except Exception as e:  # return the DB error so the model can fix its SQL
        return {"error": f"SQL error: {e}"}
    if any(c.lower() in ("response_id", "student_key", "provider_key", "appointment_key") for c in df.columns):
        return {"error": "Results may not contain individual IDs. Aggregate instead "
                         "(e.g. COUNT(DISTINCT student_key) AS n)."}
    grouped = re.search(r"\bgroup\s+by\b", s, re.IGNORECASE)
    if grouped and not any(is_count_col(c) for c in df.columns):
        return {"error": "Grouped results must include a count column, e.g. COUNT(DISTINCT response_id) AS n."}
    if not grouped and len(df) > 1 and not any(c.lower() == "n" for c in df.columns):
        return {"error": "Row-level output is not allowed. Use GROUP BY with a count column n."}
    count_cols = [c for c in df.columns if is_count_col(c)]
    label_cols = [c for c in df.columns if c not in count_cols and not pd.api.types.is_numeric_dtype(df[c])]
    small = pd.Series(False, index=df.index)
    for c in count_cols:
        small |= pd.to_numeric(df[c], errors="coerce") < config.MIN_CELL
    hidden_groups = [", ".join(f"{c}={'(blank)' if pd.isna(r[c]) else r[c]}" for c in label_cols)
                     for _, r in df[small].iterrows()] if label_cols else []
    df, hidden = suppress_small_cells(df)
    out = _df_out(df, hidden)
    if hidden_groups:
        out["hidden_groups"] = hidden_groups
        out["suppression_note"] = (f"{hidden} group(s) hidden because n < {config.MIN_CELL}: "
                                   + "; ".join(hidden_groups) + ". '(blank)' means the question was not answered.")
    return out


def service_summary(service: str = None):
    df = analysis.service_summary(_con())
    return _df_out(df[df["service"] == service] if service else df)


def item_summary(service: str):
    return _df_out(analysis.item_summary(_con(), service),
                   note="Sorted lowest % agree first.")


def compare_groups(dimension: str, service: str = None, item_code: str = None):
    try:
        df, hidden_groups = analysis.compare_groups(_con(), dimension, service, item_code)
    except ValueError as e:
        return {"error": str(e)}
    out = _df_out(df, note="Descriptive only (not a significance test).")
    k = analysis.kpi_summary(_con(), None, service)
    out["everyone"] = {"n": int(k["respondents"][1]), "pct_agree": float(k["pct_agree"][1]),
                       "mean_score": float(k["mean_score"][1])}
    if hidden_groups:
        out["hidden_groups"] = hidden_groups
        out["suppression_note"] = (f"These groups exist but have fewer than {config.MIN_CELL} "
                                   f"respondents, so their results are hidden: {', '.join(hidden_groups)}")
    if dimension in analysis.MULTI_DIMENSIONS:
        out["overlap_note"] = "Respondents can be in several groups; group counts overlap and do not sum."
    return out


def group_vs_everyone(dimension: str, value: str, service: str = None):
    try:
        r = analysis.group_vs_everyone(_con(), {"dimension": dimension, "value": value}, service)
    except ValueError as e:
        return {"error": str(e)}
    return {"group": r["group"], "overall": r["overall"],
            "by_service": json.loads(r["by_service"].to_json(orient="records")),
            "method": "Group vs everyone else, Mann-Whitney on respondents' mean scores; per-service "
                      "results corrected for multiple comparisons (q). Significant = p or q < 0.05."}


def _chr_source(con):
    return ("CHR appointments (SYNTHETIC test data, not real)" if analysis.chr_is_synthetic(con)
            else "CHR appointments")


def _need_chr():
    con = _con()
    if not analysis.has_chr(con):
        return None, {"error": "No CHR appointment export is loaded, so appointment questions can't be "
                               "answered yet. Add a CHR export to data/chr_exports and rebuild."}
    return con, None


def _add_term_warning(out, con, service_line=None, term=None, all_terms=False):
    """Attach a PARTIAL TERM warning when a requested term (or, with all_terms, any term in
    the result) is only partly covered by the data. Never breaks the tool."""
    from . import insights
    try:
        terms = [c["term"] for c in insights.term_coverage(con, service_line)] if all_terms else term
        w = insights.partial_term_warning(con, terms, service_line)
    except Exception:
        return out
    if w and isinstance(out, dict):
        out["warning"] = w
    return out


def chr_summary(service_line: str = None, dimension: str = None, term: str = None):
    con, err = _need_chr()
    if err:
        return err
    try:
        df = analysis.chr_summary(con, service_line, dimension, term)
    except ValueError as e:
        return {"error": str(e)}
    out = {**_df_out(df), "data_source": _chr_source(con)}
    return _add_term_warning(out, con, service_line, term=term, all_terms=(dimension == "term"))


def chr_trend(service_line: str = None, freq: str = "month"):
    con, err = _need_chr()
    if err:
        return err
    df = analysis.chr_trend(con, service_line, freq)
    out = {**_df_out(df), "data_source": _chr_source(con),
           "note": "notable_change = at least 10% different from the same month last year and "
                   "unlikely to be chance (p < 0.05)."}
    return _add_term_warning(out, con, service_line, all_terms=(freq == "term"))


def explain_difference(dimension: str, group: str, compare_to: str = None,
                       service_line: str = None, term: str = None):
    con, err = _need_chr()
    if err:
        return err
    try:
        r = analysis.explain_difference(con, dimension, group, compare_to, service_line, term)
    except ValueError as e:
        return {"error": str(e)}
    out = {**{k: v for k, v in r.items() if k != "measures"},
           "measures": json.loads(r["measures"].to_json(orient="records")),
           "data_source": _chr_source(con)}
    return _add_term_warning(out, con, service_line, term=term)


def what_changed(metric: str, dimension: str, service_line: str = None,
                 compare: str = "last_year", window_days: int = 61):
    from . import insights
    con, err = _need_chr()
    if err:
        return err
    try:
        r = insights.what_changed(con, metric, dimension, service_line, compare, window_days)
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def care_journeys(service_line: str = None, dimension: str = None,
                  return_days: int = 30, follow_days: int = 60):
    from . import care_patterns
    con, err = _need_chr()
    if err:
        return err
    try:
        r = care_patterns.care_journeys(con, service_line, dimension, int(return_days), int(follow_days))
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def noshow_patterns(service_line: str = None, term: str = None):
    from . import care_patterns
    con, err = _need_chr()
    if err:
        return err
    try:
        r = care_patterns.noshow_patterns(con, service_line, term)
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def service_pathways(within_days: int = 30):
    from . import care_patterns
    con, err = _need_chr()
    if err:
        return err
    try:
        r = care_patterns.service_pathways(con, int(within_days))
    except ValueError as e:
        return {"error": str(e)}
    return {**r, "data_source": _chr_source(con)}


def insight_playbook(question: str = None, topic_id: str = None):
    from . import playbook
    return playbook.insight_playbook(question, topic_id)


def priority_items(service: str):
    from . import drivers
    try:
        return drivers.priority_items(_con(), service)
    except ValueError as e:
        return {"error": str(e)}


def term_briefing():
    from . import briefing
    try:
        return briefing.for_chat()
    except Exception as e:
        return {"error": f"The briefing could not be built: {e}"}


def survey_vs_visits(dimension: str = "faculty"):
    con, err = _need_chr()
    if err:
        return err
    try:
        df = analysis.survey_vs_visits(con, dimension)
    except ValueError as e:
        return {"error": str(e)}
    return {**_df_out(df), "data_source": "survey respondents vs " + _chr_source(con),
            "note": "representation_ratio below 1 = this group is under-represented in survey "
                    "responses compared with who actually visits."}


def wait_time_vs_rating(service: str):
    try:
        df, hidden, stats = analysis.wait_vs_rating(_con(), service)
    except ValueError as e:
        return {"error": str(e)}
    out = _df_out(df, hidden)
    out["correlation"] = stats
    rho, p = stats["spearman_rho"], stats["p_value"]
    direction = "higher" if rho > 0 else "lower"
    out["interpretation"] = (
        f"Longer waits go with slightly {direction} ratings (rho = {rho}), but this is "
        f"{'statistically significant' if p < 0.05 else 'NOT statistically significant'} "
        f"(p = {p}), so {'treat it as a real relationship' if p < 0.05 else 'there is no clear relationship'}."
    )
    out["note"] = "Wait codes are ordered shortest to longest (assumed until labels are loaded)."
    return out


# Semantic aliases are suggestions, not substitutes for live survey metadata.
# A match is returned only if the expected item wording exists for this service.
_SURVEY_CONCEPTS = {
    "wellness_adv": (
        {
            "canonical_text": "I was able to book an appointment easily",
            "aliases": (
                "difficulty booking", "difficult to book", "problems booking",
                "trouble booking", "issues scheduling", "hard to book",
                "struggle to book", "struggling to book", "struggling to schedule",
                "trouble scheduling", "booking difficulty", "booking ease",
                "easy to book", "ease of booking", "appointment scheduling",
                "scheduling problems", "hard to get an appointment",
                "difficult to make an appointment", "problems making an appointment",
            ),
            "negative_responses": [1, 2],
            "polarity": "Positive wording: disagreement indicates reported booking difficulty.",
        },
        {
            "canonical_text": "Overall, I am satisfied with the support I received",
            "aliases": (
                "overall satisfaction", "less satisfied", "more satisfied",
                "satisfied with the support", "satisfaction with support",
                "satisfied with the help", "unhappy with support",
                "support they receive", "support received",
            ),
            "negative_responses": [1, 2],
            "polarity": "Positive wording: higher ratings indicate greater overall satisfaction. Two identically worded items may exist; check paired data before choosing.",
        },
    ),
    "counselling": (
        {
            "canonical_text": "I was able to book an appointment easily",
            "aliases": (
                "difficulty booking", "difficult to book", "problems booking",
                "trouble booking", "issues scheduling", "hard to book",
                "struggle to book", "struggling to schedule",
                "booking difficulty", "booking ease", "easy to book",
                "appointment scheduling", "scheduling problems",
            ),
            "negative_responses": [1, 2],
            "polarity": "Positive wording: disagreement indicates reported booking difficulty.",
        },
    ),
}


def _normalise_item_phrase(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _alias_candidates(service, query, rows):
    """Find concept aliases, then verify against the live item catalogue."""
    normal_query = _normalise_item_phrase(query)
    found = {}
    for concept in _SURVEY_CONCEPTS.get(service.lower(), ()):
        if not any(_normalise_item_phrase(p) in normal_query for p in concept["aliases"]):
            continue
        expected = _normalise_item_phrase(concept["canonical_text"])
        for code, item_text in rows:
            actual = _normalise_item_phrase(item_text)
            if actual == expected or actual.endswith(expected):
                found[code] = {
                    "item_code": code,
                    "item_text": item_text,
                    "matched_by": "verified_semantic_alias",
                    "negative_responses": concept["negative_responses"],
                    "response_interpretation": concept["polarity"],
                }
    return found


def lookup_survey_items(service: str, query: str = None):
    """Search live survey items using verified concept aliases plus text overlap."""
    con = _con()
    rows = con.execute(
        "SELECT item_code, item_text FROM items WHERE service = ? ORDER BY item_code",
        [service]).fetchall()
    if not query:
        return {"service": service, "items": [
            {"item_code": code, "item_text": text} for code, text in rows],
            "note": "Live survey catalogue. Verify the intended concept before analysis."}

    aliases = _alias_candidates(service, query, rows)
    words = re.findall(r"[a-z0-9]+", query.lower())
    stop = {"i", "a", "an", "the", "to", "of", "with", "and", "in",
            "my", "was", "is", "are", "about", "received", "overall"}
    terms = [w for w in words if w not in stop and len(w) > 2]
    scored = sorted(
        ((sum(w in (item_text or "").lower() for w in terms), code, item_text)
         for code, item_text in rows),
        key=lambda x: (-x[0], x[1]))
    items = list(aliases.values())
    for score, code, item_text in scored:
        if score > 0 and code not in aliases:
            items.append({"item_code": code, "item_text": item_text,
                          "matched_by": "text_overlap"})
        if len(items) >= 12:
            break
    return {"service": service, "items": items,
            "note": "Aliases are verified against live item wording. A low score is not automatically recoded; preserve all five Likert categories for association tests. If multiple plausible items remain, clarify."}


def _verified_item_code(service, value):
    """Require a real code, or resolve only an unambiguous exact item wording."""
    if not isinstance(value, str):
        return None
    rows = lookup_survey_items(service)["items"]
    for row in rows:
        if value.strip().upper() == row["item_code"].upper():
            return row["item_code"]
    def normalise(s):
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    needle = normalise(value)
    matches = [row["item_code"] for row in rows
               if normalise(row["item_text"]) == needle or
               (len(needle) >= 18 and normalise(row["item_text"]).endswith(needle))]
    return matches[0] if len(matches) == 1 else None


def test_association(service: str, outcome_item: str, predictor_column: str = None,
                     predictor_item: str = None, predictor_order: list = None,
                     outcome_order: list = None):
    """Compute association statistics internally, returning only privacy-filtered aggregates."""
    outcome_code = _verified_item_code(service, outcome_item)
    predictor_code = _verified_item_code(service, predictor_item) if predictor_item is not None else None
    if not outcome_code or (predictor_item is not None and not predictor_code):
        return {"error": "Survey item not verified. Call lookup_survey_items for this service and choose exact item codes before retrying.",
                "service": service, "outcome_verified": bool(outcome_code),
                "predictor_verified": predictor_item is None or bool(predictor_code)}
    try:
        return analysis.test_association(_con(), service, outcome_code, predictor_column,
                                         predictor_code, predictor_order, outcome_order)
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


def demographic_association(service: str, outcome_item: str, dimension: str,
                            category: str = None):
    """Chi-square association between demographic category and survey rating."""
    try:
        outcome_code = _verified_item_code(service, outcome_item)
        if not outcome_code:
            return {"error": "Survey outcome item not verified. Call lookup_survey_items and use an exact item code."}
        result = analysis.demographic_association(_con(), service, outcome_code, dimension, category)
        if not isinstance(result, dict) or "results" not in result:
            return result
        # Keep every category's statistical summary in the model context. Full
        # contingency tables would otherwise consume the tool response budget.
        # The underlying analysis and privacy suppression are unchanged.
        compact = []
        for row in result["results"]:
            keys = ("category", "n", "chi_square", "degrees_of_freedom",
                    "p_value", "q_value", "cramers_v", "significant_after_correction",
                    "error", "privacy_suppressed")
            compact.append({key: row[key] for key in keys if key in row})
        return {**result, "results": compact,
                "result_format": "All category statistics included; contingency tables omitted to avoid truncation."}
    except (ValueError, KeyError) as e:
        return {"error": str(e)}


def test_uploaded_categories(frame: pd.DataFrame, predictor: str, outcome: str, source: str = "CHR upload"):
    """Integration hook for the existing drag-and-drop upload handler.

    Call this with the uploaded CHR DataFrame. Do not register as an LLM tool
    until the upload handler supplies a trusted per-session DataFrame.
    """
    try:
        return analysis.categorical_association(frame, predictor, outcome, source)
    except ValueError as e:
        return {"error": str(e)}


def chr_categorical_association(predictor: str, outcome: str, service_line: str = None):
    """Student-level CHR association, with one observation per student."""
    allowed = {"faculty", "academic_level", "gender", "age_band", "program",
               "service_line", "visit_type", "mode", "outcome", "term"}
    if predictor not in allowed or outcome not in allowed:
        return {"error": "Choose two permitted CHR categorical variables: " + ", ".join(sorted(allowed))}
    if predictor == outcome:
        return {"error": "Choose two different variables."}
    con = _con()
    if not analysis.has_chr(con):
        return {"error": "No CHR appointments loaded. Upload CHR CSV and rebuild first."}
    where = 'WHERE service_line = ?' if service_line else ''
    params = [service_line] if service_line else []
    df = con.execute(f'SELECT student_key, "{predictor}" AS predictor, "{outcome}" AS outcome '
                     f'FROM appointments {where}', params).df()
    df = df.dropna(subset=['student_key'])
    # Multiple appointments for a student are not independent observations.
    # Only allow student attributes that are stable within each student's records.
    if df.groupby('student_key')[['predictor','outcome']].nunique(dropna=True).gt(1).any().any():
        return {"error": "At least one student has multiple values for a selected CHR variable. "
                         "An appointment-level chi-square would violate independence; "
                         "select stable student-level variables or use a different method."}
    df = df.drop_duplicates('student_key')
    try:
        return analysis.categorical_association(df, 'predictor', 'outcome',
                 source='CHR student-level records' + (f'; {service_line}' if service_line else ''))
    except ValueError as e:
        return {"error": str(e)}


def equity_flags(service: str = None, dimension: str = None):
    tests, flags, thin = analysis.equity_scan(_con(), service)
    if dimension and len(tests):
        tests, flags = tests[tests["dimension"] == dimension], flags[flags["dimension"] == dimension]
    cols = ["service", "dimension", "group", "n_group", "pct_agree_group", "pct_agree_rest", "gap_pts", "q"]
    return {
        "comparisons_run": len(tests),
        "flags": json.loads(flags[cols].round(3).to_json(orient="records")) if len(flags) else [],
        "groups_too_small_to_compare": len(thin),
        "method": f"Each group vs everyone else; Mann-Whitney on respondent mean score; "
                  f"Benjamini-Hochberg q < {config.FDR_Q}; |gap| >= {int(100*config.MIN_GAP_TOP2)} pts; "
                  f"n >= {config.EQUITY_MIN_GROUP}.",
    }


def search_comments(query: str, service: str = None, k: int = 8):
    """Semantic search if embeddings exist, else keyword search. Returns text only."""
    con = _con()
    k = max(1, min(int(k), 15))
    has_emb = con.execute("SELECT COUNT(*) FROM information_schema.tables "
                          "WHERE table_name='comment_embeddings'").fetchone()[0]
    where = "WHERE c.service = ?" if service else ""
    params = [service] if service else []
    if has_emb:
        from .llm import embed
        q = np.array(embed([query])[0], dtype=float)
        df = con.execute(f"""SELECT c.service, c.text, e.vec FROM comments c
                             JOIN comment_embeddings e USING (response_id, question_code) {where}""", params).df()
        if df.empty:
            return {"rows": [], "row_count": 0}
        m = np.vstack(df["vec"].to_numpy())
        sims = m @ q / (np.linalg.norm(m, axis=1) * np.linalg.norm(q) + 1e-9)
        df = df.assign(similarity=sims.round(3)).nlargest(k, "similarity").drop(columns="vec")
        mode = "semantic"
    else:
        words = [w for w in re.findall(r"\w{4,}", query.lower())][:5] or [query.lower()]
        cond = " OR ".join("lower(c.text) LIKE ?" for _ in words)
        params += [f"%{w}%" for w in words]
        df = con.execute(f"""SELECT c.service, c.text FROM comments c
                             {where + ' AND' if where else 'WHERE'} ({cond}) LIMIT {k}""", params).df()
        mode = "keyword (run comments.py --embed for semantic search)"
    return {**_df_out(df), "search_mode": mode,
            "note": "Quote at most short phrases; never speculate about who wrote a comment."}


def comment_overview(service: str = None):
    con = _con()
    has = con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='comment_labels'").fetchone()[0]
    if not has:
        return {"error": "Comments not labelled yet. Run: python -m shw_analytics.comments --label"}
    where = "WHERE c.service = ?" if service else ""
    params = [service] if service else []
    sent = con.execute(f"""SELECT l.sentiment, COUNT(*) AS n FROM comment_labels l
                           JOIN comments c USING (response_id, question_code) {where}
                           GROUP BY 1 ORDER BY n DESC""", params).df()
    themes = con.execute(f"""SELECT theme, COUNT(*) AS n FROM (
                               SELECT UNNEST(l.themes) AS theme FROM comment_labels l
                               JOIN comments c USING (response_id, question_code) {where})
                             GROUP BY 1 ORDER BY n DESC""", params).df()
    return {"sentiment": _df_out(suppress_small_cells(sent)[0])["rows"],
            "themes": _df_out(suppress_small_cells(themes)[0])["rows"]}


# Tool definitions shown to the model, and the name -> function map.
def _fn(name, desc, props=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props or {}, "required": required or []}}}

_SVC = {"type": "string", "description": "service key, e.g. medical, counselling"}

TOOL_SPECS = [
    _fn("describe_data", "Tables, columns, services, scale and rules. Call first if unsure."),
    _fn("service_summary", "Respondents, mean score and % agree per service (all, or one).",
        {"service": _SVC}),
    _fn("item_summary", "Every survey item for one service with n, mean, % agree (weakest first).",
        {"service": _SVC}, ["service"]),
    _fn("compare_groups", "% agree and mean score by group of ONE demographic dimension, "
        "including faculty (e.g. Arts) and disability type. For two dimensions combined "
        "(e.g. international AND first-generation), use run_sql on ratings_demo instead.",
        {"dimension": {"type": "string", "description": "an equity dimension (disability_any = yes/no), "
                       "or a category breakdown: disability (each disability type), faculty, "
                       "racial_background; or academic_level, year_of_study, enrollment_status, "
                       "living_arrangement, new_to_ubc"},
         "service": _SVC,
         "item_code": {"type": "string", "description": "optional; omit to use all items"}}, ["dimension"]),
    _fn("group_vs_everyone", "How ONE group compares with all respondents (e.g. Faculty of Arts, "
        "disability_any = yes), overall and per service, with significance tests.",
        {"dimension": {"type": "string", "description": "e.g. faculty, intl_or_domestic, disability_any"},
         "value": {"type": "string", "description": "the group, e.g. Arts, or a code such as 1"},
         "service": _SVC}, ["dimension", "value"]),
    _fn("chr_summary", "CHR appointments: students seen, appointments booked/attended, visits per "
        "student, days in care, no-show rate; overall or by group (faculty, academic_level, gender, "
        "age_band, program, service_line, visit_type, mode, term).",
        {"service_line": {"type": "string", "description": "e.g. Counselling, Medical"},
         "dimension": {"type": "string"}, "term": {"type": "string", "description": "e.g. 2026 Winter"}}),
    _fn("chr_trend", "CHR appointments over time (month, week or term), with the same month last year "
        "and notable changes flagged.",
        {"service_line": {"type": "string"}, "freq": {"type": "string", "description": "month, week or term"}}),
    _fn("explain_difference", "How two groups differ in service use: splits the gap into number of "
        "students, visits per student, days in care, services used and no-shows, each tested.",
        {"dimension": {"type": "string"}, "group": {"type": "string"},
         "compare_to": {"type": "string", "description": "optional; default is everyone else"},
         "service_line": {"type": "string"}, "term": {"type": "string"}}, ["dimension", "group"]),
    _fn("survey_vs_visits", "Whether survey respondents reflect who actually visits, by faculty.",
        {"dimension": {"type": "string"}}),
    _fn("wait_time_vs_rating", "How appointment wait time relates to ratings (medical or counselling).",
        {"service": _SVC}, ["service"]),
    _fn("test_association", "Chi-square independence and Cramer's V for two verified Qualtrics rating items, paired by student. Use predictor_item and outcome_item; do not guess codes. Privacy-suppressed cells block statistics. Report association, not causation.",
        {"service": _SVC, "outcome_item": {"type": "string"},
         "predictor_item": {"type": "string"}, "predictor_column": {"type": "string"}},
        ["service", "outcome_item"]),
    _fn("demographic_association", "Chi-square association of a demographic dimension with a verified survey rating item. For single-choice demographics runs one omnibus test; for multi-select demographics (faculty, disability, racial_background, identity_d15) runs separate selected-versus-not-selected tests per category with BH correction. Use for faculty versus satisfaction. Privacy suppression can block results. Never treat overlapping selections as mutually exclusive.",
        {"service": _SVC, "outcome_item": {"type": "string"},
         "dimension": {"type": "string"}, "category": {"type": "string"}},
        ["service", "outcome_item", "dimension"]),
    _fn("chr_categorical_association", "Chi-square independence for two CHR categorical variables, one record per student; only stable student-level categories supported. CHR upload must already be loaded.",
        {"predictor": {"type": "string"}, "outcome": {"type": "string"},
         "service_line": {"type": "string"}}, ["predictor", "outcome"]),
    _fn("equity_flags", "Statistically tested gaps between demographic groups (FDR-corrected).",
        {"service": _SVC, "dimension": {"type": "string", "description": "optional equity dimension"}}),
    _fn("search_comments", "Find open-text comments about a topic.",
        {"query": {"type": "string"}, "service": _SVC, "k": {"type": "integer"}}, ["query"]),
    _fn("comment_overview", "Sentiment and theme counts for open-text comments.", {"service": _SVC}),
    _fn("run_sql", "Run one read-only aggregate SELECT on DuckDB. Use only when no other tool fits. "
        "Query ratings_demo; % agree = ROUND(100*AVG(CASE WHEN agree THEN 1 ELSE 0 END),1); "
        "filter categories by the exact codes in category_values.",
        {"sql": {"type": "string"}}, ["sql"]),
]

REGISTRY = {"lookup_survey_items": lookup_survey_items, "demographic_association": demographic_association, "test_association": test_association, "chr_categorical_association": chr_categorical_association, "describe_data": describe_data, "service_summary": service_summary,
            "item_summary": item_summary, "compare_groups": compare_groups,
            "equity_flags": equity_flags, "wait_time_vs_rating": wait_time_vs_rating,
            "group_vs_everyone": group_vs_everyone, "chr_summary": chr_summary, "chr_trend": chr_trend,
            "explain_difference": explain_difference, "what_changed": what_changed, "care_journeys": care_journeys, "noshow_patterns": noshow_patterns, "service_pathways": service_pathways, "insight_playbook": insight_playbook, "term_briefing": term_briefing, "priority_items": priority_items, "survey_vs_visits": survey_vs_visits, "search_comments": search_comments,
            "comment_overview": comment_overview, "run_sql": run_sql}


_SERVICE_ALIASES = {"counseling": "counselling", "counselling services": "counselling",
                    "medical services": "medical", "wellness advising": "wellness_adv",
                    "wellness advisor": "wellness_adv", "info desk": "info_desk",
                    "wellness info desk": "info_desk", "wellness centre": "wellness_ctr",
                    "wellness center": "wellness_ctr", "naloxone training": "naloxone"}


def _norm_args(args):
    """Small-model slips that shouldn't cost a round-trip: survey service names in any
    case or spelling, and CHR service lines in any case."""
    out = dict(args)
    if isinstance(out.get("service"), str):
        v = out["service"].strip().lower()
        out["service"] = _SERVICE_ALIASES.get(v, v.replace(" ", "_"))
    if isinstance(out.get("service_line"), str):
        v = out["service_line"].strip().lower()
        out["service_line"] = {"counseling": "counselling"}.get(v, v)
    return out


def call(name, args):
    import inspect
    fn = REGISTRY.get(name)
    if not fn:
        return {"error": f"Unknown tool {name}. Available: {', '.join(REGISTRY)}"}
    args = _norm_args(args or {})
    params = inspect.signature(fn).parameters
    unknown = [k for k in args if k not in params]
    if unknown:
        hint = ""
        if name == "compare_groups" and "value" in unknown:
            hint = (" To compare one group with everyone, use group_vs_everyone; to explain why two "
                    "groups differ in appointments, use explain_difference.")
        return {"error": f"{name} has no parameter {', '.join(unknown)}. It takes: "
                         f"{', '.join(params)}.{hint}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}. It takes: {', '.join(params)}."}


TOOL_SPECS.append(_fn("lookup_survey_items",
    "FIRST look up the actual survey item codes and full question wording for a service before calling test_association or demographic_association. Return verified items from the live database, not guesses. Omit query to list all service items.",
    {"service": _SVC, "query": {"type": "string", "description": "Optional phrase to search; omit to list every question for this service."}}, ["service"]))


TOOL_SPECS.append({"type": "function", "function": {
    "name": "what_changed",
    "description": (
        "Has a CHR measure changed since last year, and is the change because of WHO came "
        "(the mix of groups shifted) or a change WITHIN groups? Compares matching date windows "
        "and splits the change into a mix effect and a within-group effect, each with a "
        "confidence interval and a verdict. Use for questions like 'have no-shows changed?' or "
        "'is this a real change or just different students coming?'. Describes how a change "
        "splits, not why it happened."),
    "parameters": {"type": "object", "properties": {
        "metric": {"type": "string", "enum": ["no_show_rate", "visits_per_student", "days_in_care"],
                   "description": "no_show_rate is per appointment; the other two are per student seen."},
        "dimension": {"type": "string",
                      "description": "Group to split by: faculty, academic_level, gender, age_band or "
                                     "program (any metric); also visit_type, mode or service_line "
                                     "for no_show_rate."},
        "service_line": {"type": "string", "description": "Optional, e.g. Counselling. Omit for all services."},
        "compare": {"type": "string", "enum": ["last_year", "previous"],
                    "description": "last_year (default, avoids seasonal effects) or the window just before."},
        "window_days": {"type": "integer", "description": "Length of each window in days (default 61)."}},
        "required": ["metric", "dimension"]}}})


def _insight_spec(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


TOOL_SPECS += [
    _insight_spec("care_journeys",
          "What happens after a student's first visit in a new episode of care: share with a single "
          "visit, share returning within 30 days, median days to a second visit, and whether "
          "students attend again after missing an appointment. Optional tested comparison of "
          "student groups. Use for continuity, return visits and 'do students come back'. A single "
          "visit can be planned single-session care, so never call it dropout.",
          {"service_line": {"type": "string", "description": "Optional, e.g. Counselling."},
           "dimension": {"type": "string", "description": "Optional group to compare: faculty, "
                         "academic_level, gender, age_band or program."},
           "return_days": {"type": "integer", "description": "Window for a quick return (default 30)."},
           "follow_days": {"type": "integer", "description": "Follow-up window (default 60)."}}),
    _insight_spec("noshow_patterns",
          "Which kinds of appointments are missed more often: by format, weekday, time of day, "
          "booking history (first booking / missed before) and student group. Tested with each "
          "student counted once and corrected for many tests. Use for no-show hotspots and "
          "reminder targeting. Patterns, not causes.",
          {"service_line": {"type": "string", "description": "Optional, e.g. Counselling."},
           "term": {"type": "string", "description": "Optional term, e.g. 2025 Fall."}}),
    _insight_spec("service_pathways",
          "Of students starting a new episode of care in one service, the share who use another "
          "service soon after (e.g. medical clinic to counselling), with 95% ranges. Use for "
          "integrated or team-based care questions. Shows movement, not referrals.",
          {"within_days": {"type": "integer", "description": "Days after the start (default 30)."}}),
    _insight_spec("insight_playbook",
          "Look up how to answer a broad or deeper question well: which analyses to run, how to "
          "read them, what to avoid and how to word the answer. Call FIRST for questions like "
          "'what should we know', 'anything surprising', 'where should we focus', 'are we reaching "
          "X students', 'is integrated care working', or anything about continuity of care.",
          {"question": {"type": "string", "description": "The staff question, as asked."},
           "topic_id": {"type": "string", "description": "Optional id from a previous lookup."}}),
]


TOOL_SPECS.append({"type": "function", "function": {
    "name": "priority_items",
    "description": (
        "Key-driver analysis for one service's survey: which questions are most closely linked "
        "to students' overall rating of the service (satisfaction, or recommending), combined with "
        "how well each scores. Returns priorities only when they are stable across "
        "re-samples of students, or says no single question stands out. Use for 'what should we "
        "focus on', 'what matters most', 'is the lowest-scoring area what hurts satisfaction'. "
        "Say 'most closely linked to', never 'drives'."),
    "parameters": {"type": "object", "properties": {
        "service": {"type": "string", "description": "counselling, medical, wellness_adv or info_desk."}},
        "required": ["service"]}}})


TOOL_SPECS.append({"type": "function", "function": {
    "name": "term_briefing",
    "description": (
        "The term briefing: runs every tested analysis across services (change since the same dates "
        "last year, missed appointments, returning after a first visit, what matters most in the "
        "survey, gaps between groups), corrects for having run them all, and returns the findings "
        "that held up, ranked, plus what stayed the same and what could not be checked. Use for "
        "'what should we know', 'headline findings', 'what should leadership know', 'anything "
        "surprising', 'where should we focus'. One call is enough; do not re-run the analyses."),
    "parameters": {"type": "object", "properties": {}, "required": []}}})

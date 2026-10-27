"""Natural language -> intent + SQL.

Deterministic first (scope checks, building/period extraction, analysis intents and a few templates);
the LLM is called once only for free-form questions, with the real schema and semantic rules. A rejected
query gets at most one bounded repair request; rejected SQL is never executed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MONTHS.update({k[:3]: v for k, v in list(MONTHS.items())})

UNSUPPORTED = [
    (r"\b(gas|steam|chilled ?water|hot ?water|water|irrigation|solar)\b", "Only electricity meters are loaded; {m} data is not available."),
    (r"\b(cost|price|bill|tariff|dollars?|\$|carbon|co2|emission)", "Costs, tariffs and emissions are not in the dataset; only electricity use in kWh."),
    (r"\b(right now|live|real[- ]?time|currently|today|this (week|month|year)|yesterday)\b",
     "This is a historical dataset (2016-01-01 to 2017-12-31), not live monitoring. Ask about a period in 2016-2017 or the latest available data."),
    (r"\b(occupan|people|headcount|temperature set ?point|thermostat)", "Occupancy and building-control data are not in the dataset."),
]


@dataclass
class Intent:
    question: str
    analyses: set = field(default_factory=set)  # forecast | anomaly | recommend
    buildings: list[str] = field(default_factory=list)
    campus: bool = False
    period: tuple[str, str] | None = None  # [start, end) as 'YYYY-MM-DD HH:MM:SS'
    horizon_hours: int = 24
    unsupported: str | None = None
    clarification: str | None = None


@dataclass
class SQLPlan:
    action: str  # sql | clarify | unsupported | error
    sql: str | None = None
    message: str | None = None
    source: str = "template"  # template | llm
    attempts: list[dict] = field(default_factory=list)


def _ts(d: date) -> str:
    return f"{d.isoformat()} 00:00:00"


def parse_period(q: str, data_end: date) -> tuple[str, str] | None:
    q = q.lower()
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*(?:to|-|until|and)\s*(\d{4}-\d{2}-\d{2})", q)
    if m:
        a, b = date.fromisoformat(m.group(1)), date.fromisoformat(m.group(2))
        return _ts(a), _ts(b + timedelta(days=1))
    m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", q)
    if m:
        a = date.fromisoformat(m.group(1))
        return _ts(a), _ts(a + timedelta(days=1))
    m = re.search(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\w*\s+(\d{4})\b", q)
    if m:
        y, mo = int(m.group(2)), MONTHS[m.group(1)]
        return _ts(date(y, mo, 1)), _ts(date(y + (mo == 12), mo % 12 + 1, 1))
    m = re.search(r"\b(20\d{2})\b", q)
    if m:
        y = int(m.group(1))
        return _ts(date(y, 1, 1)), _ts(date(y + 1, 1, 1))
    end = data_end + timedelta(days=1)
    for pat, days in [(r"last (7 days|week)", 7), (r"last (30 days|month)", 30), (r"last (90 days|quarter|3 months)", 90),
                      (r"last (year|12 months)", 365)]:
        if re.search(pat, q):
            return _ts(end - timedelta(days=days)), _ts(end)
    return None


def match_buildings(q: str, building_ids: list[str]) -> list[str]:
    ql = q.lower()
    found = [b for b in building_ids if b.lower() in ql]
    for b in building_ids:
        name = b.split("_")[-1].lower()
        if b not in found and re.search(rf"\b{name}\b", ql):
            found.append(b)
    return found


def clip_period(p: tuple[str, str] | None, data_start: date | None, data_end: date) -> tuple[str, str] | None:
    """Clip a period to the available data; a period wholly outside is left as-is so validation rejects it."""
    if not p:
        return p
    lo = _ts(data_start) if data_start else p[0]
    hi = _ts(data_end + timedelta(days=1))
    if p[1] <= lo or p[0] >= hi:
        return p
    return max(p[0], lo), min(p[1], hi)


def parse_intent(q: str, building_ids: list[str], data_end: date, data_start: date | None = None) -> Intent:
    ql = q.lower()
    it = Intent(q)
    for pat, msg in UNSUPPORTED:
        m = re.search(pat, ql)
        if m and not (m.group(0) == "water" and "electric" in ql):
            it.unsupported = msg.format(m=m.group(0))
            return it
    if re.search(r"forecast|predict|next (24|48) ?h|next (day|week|7 days)|tomorrow|expected (consumption|usage|load)|"
                 r"expected vs|vs expected|actual vs", ql):
        it.analyses.add("forecast")
    if re.search(r"anomal|unusual|abnormal|spike|outlier|flatlin|stuck|missing (data|readings)|data quality|weird|strange", ql):
        it.analyses.add("anomaly")
    if re.search(r"recommend|save energy|saving|reduce|efficien|what should|improve|waste", ql):
        it.analyses.add("recommend")
    it.buildings = match_buildings(q, building_ids)
    it.campus = bool(re.search(r"\b(campus|all buildings|total|overall|whole site)\b", ql))
    it.period = clip_period(parse_period(q, data_end), data_start, data_end)
    it.horizon_hours = 168 if re.search(r"week|7 days|168", ql) else 24
    if "forecast" in it.analyses and not it.buildings and not it.campus:
        it.clarification = "Which building should I forecast, or should I use the campus total of all 15 buildings?"
    # ambiguous use-type reference: "the office building" when several exist
    for use in ("office", "education", "assembly", "lodging"):
        if re.search(rf"\bthe {use} building\b", ql) and not it.buildings:
            matches = [b for b in building_ids if f"_{use}_" in b]
            if len(matches) > 1:
                it.clarification = (f"There are {len(matches)} {use} buildings ({', '.join(b.split('_')[-1] for b in matches)}). "
                                    "Which one do you mean, or all of them?")
    return it


# ---- templates (no LLM) ------------------------------------------------------------------------
def _where(it: Intent) -> str:
    conds = []
    if it.period:
        conds.append(f"r.ts >= '{it.period[0]}' AND r.ts < '{it.period[1]}'")
    if it.buildings:
        conds.append("r.building_id IN (" + ", ".join(f"'{b}'" for b in it.buildings) + ")")
    return ("WHERE " + " AND ".join(conds)) if conds else ""


def template_sql(it: Intent) -> str | None:
    q = it.question.lower()
    w = _where(it)
    if re.search(r"(total|sum|how much).*(by|per|each|all|every) building|which building.*(most|highest|least|lowest)|"
                 r"(top|rank|highest|largest|biggest)\s*\d*\s*(consum|building|users)|compare (the )?buildings", q):
        n = re.search(r"top\s*(\d+)", q)
        lim = f" LIMIT {int(n.group(1))}" if n else ""
        order = "ASC" if re.search(r"least|lowest|smallest", q) else "DESC"
        return ("SELECT r.building_id, b.primary_use, ROUND(SUM(r.electricity_kwh), 1) AS total_kwh, "
                "COUNT(r.electricity_kwh) AS hours_with_data, COUNT(*) - COUNT(r.electricity_kwh) AS missing_hours "
                f"FROM readings r JOIN buildings b ON b.building_id = r.building_id {w} "
                f"GROUP BY r.building_id, b.primary_use ORDER BY total_kwh {order}{lim}")
    if re.search(r"(average|typical|mean).*(hour|daily profile|load profile)|hourly profile|by hour of (the )?day", q):
        return ("SELECT CAST(STRFTIME('%H', r.ts) AS INTEGER) AS hour, ROUND(AVG(r.electricity_kwh), 2) AS avg_kwh "
                f"FROM readings r {w} GROUP BY hour ORDER BY hour")
    if re.search(r"weekend.*weekday|weekday.*weekend", q):
        return ("SELECT r.building_id, ROUND(AVG(CASE WHEN STRFTIME('%w', r.ts) IN ('0','6') THEN r.electricity_kwh END), 2) AS weekend_avg_kwh, "
                "ROUND(AVG(CASE WHEN STRFTIME('%w', r.ts) NOT IN ('0','6') THEN r.electricity_kwh END), 2) AS weekday_avg_kwh "
                f"FROM readings r {w} GROUP BY r.building_id ORDER BY weekend_avg_kwh DESC")
    if re.search(r"(monthly|per month|each month|by month)", q):
        return ("SELECT SUBSTR(r.ts, 1, 7) AS month, ROUND(SUM(r.electricity_kwh), 1) AS total_kwh, "
                f"COUNT(*) - COUNT(r.electricity_kwh) AS missing_hours FROM readings r {w} GROUP BY month ORDER BY month")
    if re.search(r"(daily|per day|each day|by day|trend)", q) and (it.buildings or it.period):
        return ("SELECT DATE(r.ts) AS day, r.building_id, ROUND(SUM(r.electricity_kwh), 1) AS total_kwh "
                f"FROM readings r {w} GROUP BY day, r.building_id ORDER BY day, r.building_id")
    return None


# ---- LLM translation ---------------------------------------------------------------------------
def build_prompt(schema, it: Intent) -> tuple[str, str]:
    system = (
        "Translate facility-manager questions into ONE SQLite SELECT query over this schema. Reply with JSON only: "
        '{"action":"sql","sql":"..."} or {"action":"clarify","question":"..."} or {"action":"unsupported","reason":"..."}.\n'
        f"SCHEMA:\n{schema.compact_schema()}\n"
        f"Primary key readings(building_id, ts); index on readings(ts). Data range {schema.min_ts} to {schema.max_ts}.\n"
        "RULES:\n- " + "\n- ".join(schema.semantic_rules) + "\n"
        "- Only SELECT/WITH. Use only listed tables/columns. Always ORDER BY for deterministic results.\n"
        "- Name energy columns with a _kwh suffix (e.g. total_kwh, avg_kwh). Round with ROUND(x, 2).\n"
        "- Prefer aggregation in SQL. Use half-open ts ranges, not DATE(ts) = ....\n"
        "- Ask ONE short clarification only if the question is genuinely ambiguous; if the data cannot answer it, use unsupported.\n"
        "BUILDINGS: " + ", ".join(f"{r.building_id} ({r.primary_use})" for r in schema.buildings.itertuples())
    )
    hints = []
    if it.buildings:
        hints.append("buildings mentioned: " + ", ".join(it.buildings))
    if it.period:
        hints.append(f"period: ts >= '{it.period[0]}' AND ts < '{it.period[1]}'")
    user = it.question + ("\nHINTS: " + "; ".join(hints) if hints else "")
    return system, user


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def translate(it: Intent, schema, validator, llm) -> SQLPlan:
    sql = template_sql(it)
    if sql:
        v = validator.validate(sql)
        if v.ok:
            return SQLPlan("sql", sql, source="template", attempts=[{"sql": sql, "ok": True}])
    if llm is None or not getattr(llm, "available", False):
        return SQLPlan("unsupported", message="I can only answer this kind of question with the language model, which is not "
                       "configured (set GROQ_API_KEY). Try one of the example questions.")
    system, user = build_prompt(schema, it)
    attempts = []
    for attempt in range(2):  # initial + one bounded repair
        r = llm.chat(system, user, max_tokens=500, json_mode=True)
        if not r.ok:
            return SQLPlan("error", message=f"The language model is unavailable ({r.error}). Try an example question.",
                           source="llm", attempts=attempts)
        data = _parse_json(r.text) or {}
        action = data.get("action")
        if action == "clarify" and data.get("question"):
            return SQLPlan("clarify", message=str(data["question"])[:300], source="llm", attempts=attempts)
        if action == "unsupported":
            return SQLPlan("unsupported", message=str(data.get("reason", "This question cannot be answered from the data."))[:400],
                           source="llm", attempts=attempts)
        sql = data.get("sql") if action == "sql" else None
        v = validator.validate(sql) if sql else None
        attempts.append({"sql": sql, "ok": bool(v and v.ok), "code": v.reason_code if v else "NO_SQL"})
        if v and v.ok:
            return SQLPlan("sql", sql, source="llm", attempts=attempts)
        if attempt == 0:
            reason = v.reason_text if v else "no SQL returned"
            user = (f"{it.question}\nYour previous query was rejected: {reason}. Return a corrected single read-only "
                    f"SELECT using only the schema, as JSON.")
    return SQLPlan("error", message="I could not produce a safe query for this question. Please rephrase it.",
                   source="llm", attempts=attempts)

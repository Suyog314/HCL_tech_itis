"""Copilot orchestrator: question -> intent -> validated SQL / analytics -> chart, explanation, checks."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import pandas as pd

from . import config
from .anomaly import AnomalyDetector
from .cache import SQLQueryCache
from .charts import ChartSpec, select_chart
from .db import Executor, QueryResult, connect_readonly, introspect
from .explain import ExplanationChecker
from .forecast import Forecaster
from .llm import LLMClient
from .nl2sql import Intent, parse_intent, translate
from .recommend import from_anomalies, from_forecast, from_profile
from .sql_optimizer import SQLOptimizer

CHART_FETCH = 5000      # rows fetched for charting (bounded by the validator cap)
EXPLAIN_FETCH = 200     # rows given to the explainer (it shows the model at most 20)


@dataclass
class Answer:
    question: str
    status: str  # answered | clarify | unsupported | error
    message: str = ""
    sql: str | None = None
    validation_notes: list[str] = field(default_factory=list)
    optimization: list[dict] = field(default_factory=list)
    data: pd.DataFrame | None = None
    truncated: bool = False
    chart: ChartSpec | None = None
    explanation: dict | None = None
    anomalies: list = field(default_factory=list)
    forecast: object = None
    recommendations: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    from_cache: bool = False
    source: str | None = None


class Copilot:
    def __init__(self, db_path=None, llm=None, optimize: bool = True):
        self.con = connect_readonly(db_path)
        self.schema = introspect(self.con)
        self.ex = Executor(self.con, self.schema, max_rows=config.MAX_ROWS)
        self.ax = Executor(self.con, self.schema, max_rows=config.ANALYTICS_MAX_ROWS)
        self.optimizer = SQLOptimizer(self.ex) if optimize else None
        self.cache = SQLQueryCache(config.CACHE_TTL_S, self.schema.dataset_version)
        self.llm = llm if llm is not None else LLMClient()
        self.building_ids = list(self.schema.buildings["building_id"])
        self.data_end = date.fromisoformat(self.schema.max_ts[:10])
        self.checker = ExplanationChecker(self.llm, self.schema.unit, building_ids=self.building_ids,
                                          date_range=(self.schema.min_ts, self.schema.max_ts))
        self.detector = AnomalyDetector(unit=self.schema.unit)
        self.forecaster = Forecaster(unit=self.schema.unit)

    # ---- data access helpers (all SQL is validated by the executors) -------------------------------
    def _cached_query(self, executor: Executor, sql: str, max_fetch: int) -> QueryResult:
        ctx = {"sql": sql, "max_fetch": max_fetch, "cap": executor.validator.max_rows}
        hit = self.cache.get("__query__", ctx)
        if hit is not None:
            return hit
        v = executor.validator.validate(sql)
        res = executor.run_validated(v, max_fetch=max_fetch) if v.ok else QueryResult(False, verdict=v, error_code=v.reason_code,
                                                                                        error_text=v.reason_text)
        if res.ok:
            self.cache.set("__query__", res, ctx)
        return res

    def page(self, sql: str, page: int, page_size: int = config.PAGE_SIZE) -> QueryResult:
        ctx = {"sql": sql, "page": page, "size": page_size}
        hit = self.cache.get("__page__", ctx)
        if hit is not None:
            return hit
        res = self.ex.run_page(self.ex.validator.validate(sql), page, page_size)
        if res.ok:
            self.cache.set("__page__", res, ctx)
        return res

    def hourly(self, buildings: list[str] | None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        conds = []
        if buildings:
            conds.append("building_id IN (" + ", ".join(f"'{b}'" for b in buildings if b in self.building_ids) + ")")
        if start:
            conds.append(f"ts >= '{max(start, self.schema.min_ts)}'")
        if end:
            conds.append(f"ts < '{end}'")
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        r = self._cached_query(self.ax, f"SELECT building_id, ts, electricity_kwh AS kwh FROM readings {where} "
                                        "ORDER BY building_id, ts", config.ANALYTICS_MAX_ROWS)
        return r.df if r.ok else pd.DataFrame(columns=["building_id", "ts", "kwh"])

    def campus_series(self) -> pd.Series:
        n = len(self.building_ids)
        r = self._cached_query(self.ax, "SELECT ts, CASE WHEN COUNT(electricity_kwh) = " + str(n) +
                               " THEN SUM(electricity_kwh) END AS kwh FROM readings GROUP BY ts ORDER BY ts", 20000)
        s = r.df.set_index("ts")["kwh"]
        s.index = pd.to_datetime(s.index)
        return s

    def weather(self) -> pd.Series:
        r = self._cached_query(self.ax, "SELECT ts, air_temp_c FROM weather ORDER BY ts", 20000)
        s = r.df.set_index("ts")["air_temp_c"]
        s.index = pd.to_datetime(s.index)
        return s

    def profile(self, start: str, end: str) -> pd.DataFrame:
        sql = ("SELECT r.building_id, b.primary_use, "
               "AVG(CASE WHEN CAST(STRFTIME('%H', r.ts) AS INTEGER) < 5 THEN r.electricity_kwh END) AS night_avg_kwh, "
               "AVG(CASE WHEN CAST(STRFTIME('%H', r.ts) AS INTEGER) BETWEEN 9 AND 16 AND STRFTIME('%w', r.ts) NOT IN ('0','6') "
               "THEN r.electricity_kwh END) AS day_avg_kwh FROM readings r JOIN buildings b ON b.building_id = r.building_id "
               f"WHERE r.ts >= '{start}' AND r.ts < '{end}' GROUP BY r.building_id, b.primary_use ORDER BY r.building_id")
        r = self._cached_query(self.ex, sql, 100)
        return r.df if r.ok else pd.DataFrame()

    # ---- analytics ----------------------------------------------------------------------------
    def _default_window(self, it: Intent, days: int = 90) -> tuple[str, str]:
        if it.period:
            return it.period
        end = self.data_end + timedelta(days=1)
        return f"{end - timedelta(days=days)} 00:00:00", f"{end} 00:00:00"

    def run_anomalies(self, it: Intent):
        start, end = self._default_window(it)
        lookback = (pd.Timestamp(start) - pd.Timedelta(weeks=9)).strftime("%Y-%m-%d %H:%M:%S")
        df = self.hourly(it.buildings or None, lookback, end)
        return self.detector.detect(df, start, end), (start, end)

    def run_forecast(self, it: Intent):
        if it.buildings:
            df = self.hourly([it.buildings[0]])
            series = df.set_index(pd.to_datetime(df["ts"]))["kwh"]
            target = it.buildings[0]
        else:
            series, target = self.campus_series(), "Campus total (15 buildings)"
        # all data is historical: a requested period means "expected vs actual" for that period
        as_of = it.period[0] if it.period else None
        horizon = it.horizon_hours
        if as_of:
            span = int((pd.Timestamp(it.period[1]) - pd.Timestamp(it.period[0])) / pd.Timedelta(hours=1))
            horizon = max(24, min(span, 168))
        return self.forecaster.forecast(series, as_of=as_of, horizon_hours=horizon, target=target, weather=self.weather())

    # ---- main entry point -------------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(question, "clarify", "Please type a question about the campus electricity data.")
        cached = self.cache.get(question, {"kind": "answer"})
        if cached is not None:
            cached.from_cache = True
            return cached
        it = parse_intent(question, self.building_ids, self.data_end, date.fromisoformat(self.schema.min_ts[:10]))
        if it.unsupported:
            return Answer(question, "unsupported", it.unsupported)
        if it.clarification:
            return Answer(question, "clarify", it.clarification)
        ans = self._answer_analysis(it) if it.analyses else self._answer_sql(it)
        if ans.status == "answered":
            self.cache.set(question, ans, {"kind": "answer"})
        return ans

    def _answer_sql(self, it: Intent) -> Answer:
        plan = translate(it, self.schema, self.ex.validator, self.llm)
        if plan.action != "sql":
            return Answer(it.question, "clarify" if plan.action == "clarify" else plan.action if plan.action == "unsupported" else "error",
                          plan.message or "", source=plan.source)
        v = self.ex.validator.validate(plan.sql)
        sql, opt = v.clean_sql, []
        if self.optimizer:
            o = self.optimizer.optimize(sql)
            sql, opt = o.sql, o.report
        res = self._cached_query(self.ex, sql, CHART_FETCH)
        if not res.ok:
            return Answer(it.question, "error", res.error_text or "Query failed.", sql=sql, source=plan.source)
        df = res.df
        ans = Answer(it.question, "answered", sql=sql, validation_notes=list(v.notes), optimization=opt, data=df,
                     truncated=res.truncated, source=plan.source)
        ans.chart = select_chart(it.question, df, self.schema.unit)
        if res.truncated:
            ans.warnings.append(f"Result truncated at {CHART_FETCH} rows for charting; the table pages through the full capped result.")
        if df.empty:
            ans.message = "No rows matched this question in the available data."
        for c in df.columns:
            if "missing" in c.lower() and pd.api.types.is_numeric_dtype(df[c]) and df[c].sum() > 0:
                ans.warnings.append(f"{int(df[c].sum())} hour(s) without readings in this result; totals exclude them (not treated as zero).")
        ans.explanation = self.checker.explain(it.question, df.head(EXPLAIN_FETCH),
                                               {"rows_total": len(df), "truncated": res.truncated} if len(df) > EXPLAIN_FETCH else None)
        return ans

    def _answer_analysis(self, it: Intent) -> Answer:
        ans = Answer(it.question, "answered", source="analytics")
        summary: dict = {}
        if "forecast" in it.analyses:
            fc = self.run_forecast(it)
            ans.forecast = fc
            summary["forecast"] = {"target": fc.target, "status": fc.status, "model": fc.model, "metrics": fc.metrics,
                                   **fc.summary, "warnings": fc.warnings}
            ans.warnings += fc.warnings
            ans.recommendations += from_forecast(fc)
        if it.analyses & {"anomaly", "recommend"}:
            items, (start, end) = self.run_anomalies(it)
            ans.anomalies = items
            real = [a for a in items if a.category == "anomaly"]
            dq = [a for a in items if a.category == "data_quality"]
            summary["anomalies"] = {"window": [start, end], "anomaly_count": len(real), "data_quality_count": len(dq),
                                    "top": [{k: v for k, v in asdict(a).items() if k not in ("warnings", "unit")} for a in items[:8]]}
            if "recommend" in it.analyses:
                prof = self.profile(start, end)
                if it.buildings:
                    prof = prof[prof["building_id"].isin(it.buildings)]
                ans.recommendations += from_profile(prof) + from_anomalies(items)
            if dq:
                ans.warnings.append(f"{len(dq)} data-quality issue(s) found (missing, flatlined or suspicious readings).")
        if ans.recommendations:
            summary["recommendations"] = [{"building": r.building_id, "finding": r.finding, "evidence": r.evidence}
                                          for r in ans.recommendations[:6]]
        ans.explanation = self.checker.explain(it.question, None, summary) if summary else None
        return ans

    def dataset_overview(self) -> dict:
        b = self.schema.buildings
        return {"site": self.schema.meta.get("site_id"), "buildings": b, "start": self.schema.min_ts, "end": self.schema.max_ts,
                "unit": self.schema.unit, "selection_rule": self.schema.meta.get("selection_rule"),
                "dataset_version": self.schema.dataset_version, "note": self.schema.meta.get("timestamp_note"),
                "indexes": self.schema.indexes, "read_only": self.schema.read_only}

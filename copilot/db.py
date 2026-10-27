"""Read-only database access, schema introspection and guarded execution.

Defence in depth: URI ``mode=ro`` + ``PRAGMA query_only`` + an SQLite authorizer that only permits
reads, plus a progress-handler timeout and a fetch cap. Every SQL string passes ``SQLValidator`` first.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config
from .sql_errors import describe
from .sql_validator import SQLValidator, Verdict

TS_FORMAT_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")

_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
_FORBIDDEN_FUNCS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _FORBIDDEN_FUNCS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and (arg1 or "").startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def connect_readonly(path: Path | None = None) -> sqlite3.Connection:
    path = path or config.DB_PATH
    if not Path(path).exists():
        raise FileNotFoundError(f"Database not found at {path}. Run: python -m copilot.build_db")
    con = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    con.set_authorizer(_authorizer)
    return con


@dataclass
class SchemaInfo:
    tables: dict[str, list[str]]  # table -> columns (actual DB order)
    column_types: dict[str, dict[str, str]]
    indexes: dict[str, list[str]]
    min_ts: str
    max_ts: str
    unit: str
    dataset_version: str
    buildings: pd.DataFrame
    meta: dict[str, str]
    ts_fixed_format: bool
    read_only: bool = True
    semantic_rules: list[str] = field(default_factory=lambda: [
        "electricity_kwh NULL means no reading, not zero; never COALESCE it to 0 unless asked.",
        "is_flagged = 1 means the raw value was marked by BDG2 cleaning; the row is not deleted.",
        "All energy values are hourly kWh; sums over hours give kWh, averages give kWh per hour.",
        "ts is site-local 'YYYY-MM-DD HH:MM:SS' text; use half-open ranges ts >= a AND ts < b.",
        "Data is historical (2016-01-01 to 2017-12-31), not live.",
        "Definitions: night = hours 00-04 (STRFTIME('%H', ts) < '05'); weekday daytime = hours 09-16 Mon-Fri; "
        "weekend = STRFTIME('%w', ts) IN ('0','6'); seasons: winter Dec-Feb, spring Mar-May, summer Jun-Aug, autumn Sep-Nov.",
    ])

    def compact_schema(self) -> str:
        lines = [f"{t}({', '.join(f'{c} {self.column_types[t][c]}' for c in cols)})" for t, cols in self.tables.items()]
        return "\n".join(lines)


def introspect(con: sqlite3.Connection, allowed=("readings", "buildings", "weather")) -> SchemaInfo:
    con.set_authorizer(None)  # introspection needs PRAGMA; connection stays mode=ro + query_only
    try:
        tables, types, indexes = {}, {}, {}
        existing = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        for t in allowed:
            if t not in existing:
                continue
            info = con.execute(f"PRAGMA table_info({t})").fetchall()
            tables[t] = [r[1] for r in info]
            types[t] = {r[1]: r[2] for r in info}
            idx = []
            for row in con.execute(f"PRAGMA index_list({t})").fetchall():
                cols = [r[2] for r in con.execute(f"PRAGMA index_info({row[1]})").fetchall()]
                idx.append(f"{row[1]}({', '.join(cols)})")
            indexes[t] = idx
        meta = dict(con.execute("SELECT key, value FROM meta").fetchall())
        mn, mx = con.execute("SELECT MIN(ts), MAX(ts) FROM readings").fetchone()
        bad = con.execute(
            "SELECT COUNT(*) FROM readings WHERE ts IS NULL OR length(ts) != 19 OR ts NOT GLOB "
            "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]'"
        ).fetchone()[0]
        b = pd.read_sql_query("SELECT * FROM buildings ORDER BY building_id", con)
        ro = con.execute("PRAGMA query_only").fetchone()[0] == 1
    finally:
        con.set_authorizer(_authorizer)
    return SchemaInfo(tables, types, indexes, mn, mx, meta.get("unit", config.ENERGY_UNIT),
                      meta.get("dataset_version", "unknown"), b, meta, bad == 0, ro)


class QueryTimeout(Exception):
    pass


@dataclass
class QueryResult:
    ok: bool
    df: pd.DataFrame | None = None
    sql: str | None = None
    verdict: Verdict | None = None
    error_code: str | None = None
    error_text: str | None = None
    elapsed_ms: float = 0.0
    has_next: bool = False
    page: int = 0
    truncated: bool = False


class Executor:
    """Validates then executes SQL on a read-only connection with timeout and fetch caps."""

    def __init__(self, con: sqlite3.Connection, schema: SchemaInfo, max_rows: int = config.MAX_ROWS,
                 timeout_s: float = config.QUERY_TIMEOUT_S):
        self.con, self.schema, self.timeout_s = con, schema, timeout_s
        self.validator = SQLValidator.from_schema(schema, max_rows=max_rows)

    def _raw(self, sql: str, max_fetch: int) -> tuple[pd.DataFrame, bool]:
        deadline = time.perf_counter() + self.timeout_s
        self.con.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
        try:
            cur = self.con.execute(sql)
            rows = cur.fetchmany(max_fetch + 1)
            cols = [d[0] for d in cur.description]
            cur.close()
        except sqlite3.OperationalError as e:
            if "interrupted" in str(e):
                raise QueryTimeout() from e
            raise
        finally:
            self.con.set_progress_handler(None, 0)
        more = len(rows) > max_fetch
        return pd.DataFrame(rows[:max_fetch], columns=cols), more

    def run(self, sql: str, intent=None) -> QueryResult:
        v = self.validator.validate(sql, intent)
        if not v.ok:
            return QueryResult(False, verdict=v, error_code=v.reason_code, error_text=v.reason_text)
        return self.run_validated(v)

    def run_validated(self, v: Verdict, max_fetch: int | None = None) -> QueryResult:
        if not v.ok or not v.clean_sql:  # never execute anything that failed validation
            return QueryResult(False, verdict=v, error_code=v.reason_code or "PARSE_ERROR", error_text=v.reason_text)
        t0 = time.perf_counter()
        try:
            df, more = self._raw(v.clean_sql, max_fetch or self.validator.max_rows)
        except QueryTimeout:
            return QueryResult(False, sql=v.clean_sql, verdict=v, error_code="TIMEOUT", error_text=describe("TIMEOUT"))
        except sqlite3.Error as e:
            return QueryResult(False, sql=v.clean_sql, verdict=v, error_code="EXEC_ERROR",
                               error_text=describe("EXEC_ERROR", type(e).__name__ + ": " + str(e)[:200]))
        return QueryResult(True, df, v.clean_sql, v, elapsed_ms=(time.perf_counter() - t0) * 1000, truncated=more)

    def run_page(self, v: Verdict, page: int, page_size: int = config.PAGE_SIZE) -> QueryResult:
        """Server-side pagination: wraps the validated SQL with LIMIT/OFFSET and fetches page_size+1 rows."""
        from .pagination import page_sql

        if not v.ok:
            return QueryResult(False, verdict=v, error_code=v.reason_code, error_text=v.reason_text)
        psql = page_sql(v.clean_sql, page, page_size)
        pv = self.validator.validate(psql)  # the paged SQL is re-validated before execution
        if not pv.ok:
            return QueryResult(False, verdict=pv, error_code=pv.reason_code, error_text=pv.reason_text)
        res = self.run_validated(pv, max_fetch=page_size + 1)
        if res.ok:
            res.has_next = len(res.df) > page_size
            res.df = res.df.head(page_size)
            res.page = page
        return res

    def explain(self, sql: str) -> list[str]:
        v = self.validator.validate(sql)
        if not v.ok:
            return []
        self.con.set_authorizer(None)
        try:
            return [r[-1] for r in self.con.execute("EXPLAIN QUERY PLAN " + v.clean_sql).fetchall()]
        finally:
            self.con.set_authorizer(_authorizer)

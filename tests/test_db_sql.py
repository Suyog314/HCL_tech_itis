"""Read-only execution, pagination, optimizer and cache."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

from copilot.cache import SQLQueryCache
from copilot.pagination import page_sql
from copilot.sql_optimizer import SQLOptimizer, frames_equivalent, rule_date_range, rule_like_to_in
from copilot.sql_validator import Verdict
import sqlglot


def test_introspection(ctx):
    _, schema, _ = ctx
    assert set(schema.tables) == {"readings", "buildings", "weather"}  # 'secrets' not exposed
    assert schema.min_ts == "2017-01-01 00:00:00" and schema.ts_fixed_format and schema.read_only
    assert any("building_id, ts" in i for i in schema.indexes["readings"])


def test_connection_is_really_read_only(ctx):
    con, _, _ = ctx
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM readings")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("CREATE TABLE z (a)")


def test_executor_rejects_without_touching_db(ctx):
    _, _, ex = ctx
    r = ex.run("DELETE FROM readings")
    assert not r.ok and r.error_code == "NOT_SELECT" and r.df is None
    bad = Verdict(False, None, "NOT_SELECT", "x")
    assert not ex.run_validated(bad).ok


def test_null_is_not_zero(ctx):
    _, _, ex = ctx
    r = ex.run("SELECT COUNT(*) AS n, COUNT(electricity_kwh) AS n_vals FROM readings WHERE building_id = 'Fox_office_Beta'")
    assert r.ok and r.df.n[0] - r.df.n_vals[0] == 10


def test_timeout(ctx):
    con, schema, _ = ctx
    from copilot.db import Executor

    slow = Executor(con, schema, max_rows=10, timeout_s=0.0)
    r = slow.run("SELECT COUNT(*) AS n FROM readings a JOIN readings b ON a.electricity_kwh > b.electricity_kwh")
    assert not r.ok and r.error_code == "TIMEOUT"


def test_pagination_pages_and_has_next(ctx):
    _, _, ex = ctx
    v = ex.validator.validate("SELECT ts FROM weather ORDER BY ts LIMIT 25")
    p0, p1, p2 = (ex.run_page(v, i, 10) for i in range(3))
    assert len(p0.df) == 10 and p0.has_next and p0.df.ts.iloc[0] == "2017-01-01 00:00:00"
    assert p1.df.ts.iloc[0] == "2017-01-01 10:00:00" and p1.has_next
    assert len(p2.df) == 5 and not p2.has_next  # never reads past the query's own LIMIT
    assert "LIMIT 11 OFFSET 0" in p0.sql


def test_page_sql_respects_existing_offset():
    s = page_sql("SELECT ts FROM weather ORDER BY ts LIMIT 30 OFFSET 5", 1, 10)
    assert "LIMIT 11" in s and "OFFSET 15" in s
    assert "LIMIT 0" in page_sql("SELECT ts FROM weather ORDER BY ts LIMIT 5", 3, 10)


def test_frames_equivalent_null_and_order():
    a = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    assert frames_equivalent(a, a.copy(), True)[0]
    assert not frames_equivalent(a, pd.DataFrame({"x": [1.0, 0.0, 3.0]}), True)[0]  # NULL != 0
    assert not frames_equivalent(a, a.iloc[::-1].reset_index(drop=True), True)[0]
    assert frames_equivalent(a, a.iloc[::-1].reset_index(drop=True), False)[0]


def test_date_rewrite_kept_only_when_equivalent_and_faster(ctx):
    _, _, ex = ctx
    opt = SQLOptimizer(ex, min_speedup=0.0, reps=1)  # tiny fixture: do not require a speedup
    v = ex.validator.validate("SELECT building_id, SUM(electricity_kwh) AS kwh FROM readings "
                              "WHERE DATE(ts) = '2017-01-05' GROUP BY building_id ORDER BY building_id")
    res = opt.optimize(v.clean_sql)
    rep = res.report[0]
    assert rep["rule"] == "date_function_to_range"
    assert rep["keep"] and "ts >= '2017-01-05 00:00:00' AND ts < '2017-01-06 00:00:00'" in res.sql
    assert rep["plan_improved"]


def test_rewrite_rejected_without_speedup(ctx):
    _, _, ex = ctx
    opt = SQLOptimizer(ex, min_speedup=1000.0, reps=1)
    v = ex.validator.validate("SELECT building_id, SUM(electricity_kwh) AS kwh FROM readings "
                              "WHERE DATE(ts) = '2017-01-05' GROUP BY building_id ORDER BY building_id")
    res = opt.optimize(v.clean_sql)
    assert res.report[0]["plan_improved"]  # plan is better, but the measured speedup is below the bar
    assert res.sql == v.clean_sql and not res.report[0]["keep"] and "speedup" in res.report[0]["reason"]


def test_rewrite_rules_semantics(ctx):
    _, _, ex = ctx
    opt = SQLOptimizer(ex)
    t = sqlglot.parse_one("SELECT ts FROM readings WHERE STRFTIME('%Y-%m', ts) = '2017-12'", read="sqlite")
    assert rule_date_range(t, opt) and "ts < '2018-01-01 00:00:00'" in t.sql(dialect="sqlite")
    t = sqlglot.parse_one("SELECT ts FROM readings WHERE building_id LIKE '%OFFICE%'", read="sqlite")
    assert rule_like_to_in(t, opt)  # SQLite LIKE is case-insensitive; enumerated set must match that
    assert "'Fox_office_Alpha', 'Fox_office_Beta'" in t.sql(dialect="sqlite")
    t = sqlglot.parse_one("SELECT ts FROM readings WHERE building_id LIKE '%zzz%'", read="sqlite")
    assert not rule_like_to_in(t, opt)  # no match: leave untouched


def test_rewrite_preserves_null_and_limit_order(ctx):
    _, _, ex = ctx
    opt = SQLOptimizer(ex, min_speedup=0.0, reps=1)
    sql = ex.validator.validate("SELECT ts, electricity_kwh FROM readings WHERE building_id LIKE '%beta%' "
                                "AND DATE(ts) = '2017-02-20' ORDER BY ts LIMIT 5").clean_sql
    res = opt.optimize(sql)
    a = ex.run(sql).df
    b = ex.run(res.sql).df
    assert frames_equivalent(a, b, True)[0] and res.sql.endswith("LIMIT 5")


# ---- cache -----------------------------------------------------------------------------------
class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_cache_hit_miss_expire_clear():
    clk = Clock()
    c = SQLQueryCache(ttl_seconds=10, dataset_version="v1", clock=clk)
    assert c.get("Q") is None and c.misses == 1
    c.set("  Total by BUILDING ", {"rows": 1})
    assert c.get("total by building") == {"rows": 1} and c.hits == 1
    clk.t = 11
    assert c.get("total by building") is None and len(c) == 0
    c.set("q", 1)
    c.clear()
    assert c.get("q") is None


def test_cache_parameter_and_version_isolation():
    c = SQLQueryCache(ttl_seconds=100, dataset_version="v1")
    c.set("q", "page0", {"sql": "S", "page": 0})
    assert c.get("q", {"sql": "S", "page": 1}) is None
    assert c.get("q", {"sql": "T", "page": 0}) is None
    assert c.get("q", {"sql": "S", "page": 0}) == "page0"
    c.set_dataset_version("v2")  # dataset update -> nothing stale survives
    assert c.get("q", {"sql": "S", "page": 0}) is None

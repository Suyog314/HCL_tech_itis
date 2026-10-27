import pytest

from copilot.sql_validator import SQLValidator

ALLOWED = {"readings": ["building_id", "ts", "electricity_kwh", "is_flagged"],
           "buildings": ["building_id", "primary_use"], "weather": ["ts", "air_temp_c"]}


@pytest.fixture
def v():
    return SQLValidator(ALLOWED, "2017-01-01 00:00:00", "2017-03-04 23:00:00", max_rows=100)


@pytest.mark.parametrize("sql,code", [
    ("", "PARSE_ERROR"),
    ("   ", "PARSE_ERROR"),
    (None, "PARSE_ERROR"),
    ("SELECT FROM WHERE (", "PARSE_ERROR"),
    ("SELECT " + "1+" * 5000 + "1 FROM readings", "TOO_COMPLEX"),
    ("SELECT 1 FROM readings; DROP TABLE readings", "MULTI_STATEMENT"),
    ("SELECT 1 FROM readings; SELECT 2 FROM readings", "MULTI_STATEMENT"),
    ("DELETE FROM readings", "NOT_SELECT"),
    ("UPDATE readings SET electricity_kwh = 0", "NOT_SELECT"),
    ("INSERT INTO readings VALUES ('a','b',1,0)", "NOT_SELECT"),
    ("DROP TABLE readings", "NOT_SELECT"),
    ("CREATE TABLE x AS SELECT * FROM readings", "NOT_SELECT"),
    ("PRAGMA table_info(readings)", "NOT_SELECT"),
    ("ATTACH DATABASE 'x.db' AS x", "NOT_SELECT"),
    ("VACUUM", "NOT_SELECT"),
    ("WITH x AS (DELETE FROM readings RETURNING *) SELECT * FROM x", "FORBIDDEN_STATEMENT"),
    ("WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM c) SELECT n FROM c, readings", "TOO_COMPLEX"),
    ("SELECT * FROM sqlite_master", "TABLE_NOT_ALLOWED"),
    ("SELECT token FROM secrets", "TABLE_NOT_ALLOWED"),
    ("SELECT * FROM main.readings", "TABLE_NOT_ALLOWED"),
    ("SELECT * FROM pragma_table_info('readings')", "TABLE_NOT_ALLOWED"),
    ("SELECT password FROM readings", "COLUMN_NOT_ALLOWED"),
    ("SELECT rowid FROM readings", "COLUMN_NOT_ALLOWED"),
    ("SELECT x.ts FROM readings r", "COLUMN_NOT_ALLOWED"),
    ("SELECT 1", "NO_DATA_SOURCE"),
    ("SELECT 1 AS a UNION SELECT 2", "NO_DATA_SOURCE"),
    ("WITH a AS (SELECT 1 AS x) SELECT x FROM a", "NO_DATA_SOURCE"),
    ("SELECT load_extension('evil') FROM readings", "FORBIDDEN_FUNCTION"),
    ("SELECT readfile('/etc/passwd') FROM readings", "FORBIDDEN_FUNCTION"),
    ("SELECT sqlite_version() FROM readings", "FORBIDDEN_FUNCTION"),
    ("SELECT randomblob(1000000000) FROM readings", "FORBIDDEN_FUNCTION"),
    ("SELECT ts FROM readings WHERE ts >= '2019-01-01'", "OUT_OF_RANGE"),
    ("SELECT ts FROM readings WHERE ts < '2016-06-01'", "OUT_OF_RANGE"),
    ("SELECT ts FROM readings WHERE ts BETWEEN '2015-01-01' AND '2015-02-01'", "OUT_OF_RANGE"),
    ("SELECT ts FROM readings WHERE DATE(ts) = '2018-05-01'", "OUT_OF_RANGE"),
    ("SELECT ts FROM readings WHERE '2020-01-01' <= ts", "OUT_OF_RANGE"),
    ("SELECT ts FROM readings LIMIT (SELECT 5)", "TOO_COMPLEX"),
])
def test_rejections(v, sql, code):
    verdict = v.validate(sql)
    assert not verdict.ok
    assert verdict.reason_code == code
    assert verdict.clean_sql is None
    assert verdict.reason_text


@pytest.mark.parametrize("sql", [
    "SELECT building_id, SUM(electricity_kwh) AS total_kwh FROM readings GROUP BY building_id ORDER BY total_kwh DESC",
    "WITH d AS (SELECT building_id, DATE(ts) AS day, SUM(electricity_kwh) AS kwh FROM readings GROUP BY 1, 2) "
    "SELECT day, MAX(kwh) AS max_kwh FROM d GROUP BY day ORDER BY day",
    "SELECT building_id FROM readings UNION SELECT building_id FROM buildings ORDER BY 1",
    "SELECT building_id FROM readings UNION ALL SELECT building_id FROM buildings",
    "SELECT r.ts, w.air_temp_c FROM readings r JOIN weather w ON r.ts = w.ts WHERE r.building_id = 'a'",
    "SELECT ts FROM readings WHERE building_id IN (SELECT building_id FROM buildings WHERE primary_use = 'Office')",
    "SELECT building_id, LAG(electricity_kwh) OVER (PARTITION BY building_id ORDER BY ts) AS prev_kwh FROM readings",
    "SELECT CASE WHEN STRFTIME('%w', ts) IN ('0','6') THEN 'weekend' ELSE 'weekday' END AS dt, "
    "COUNT(*) - COUNT(electricity_kwh) AS missing FROM readings GROUP BY dt",
    "SELECT ts FROM readings WHERE ts >= '2017-01-01' AND ts < '2017-03-05'",
    "SELECT * FROM readings",
    "SELECT ts FROM readings;",
])
def test_accepts(v, sql):
    verdict = v.validate(sql)
    assert verdict.ok, verdict
    assert verdict.reason_code is None
    assert v.validate(verdict.clean_sql).ok  # output is itself valid


def test_row_cap_added_kept_lowered(v):
    a = v.validate("SELECT ts FROM readings")
    assert a.clean_sql.endswith("LIMIT 100") and any("added" in n for n in a.notes)
    b = v.validate("SELECT ts FROM readings LIMIT 10")
    assert b.clean_sql.endswith("LIMIT 10") and any("kept" in n for n in b.notes)
    c = v.validate("SELECT ts FROM readings LIMIT 100000")
    assert c.clean_sql.endswith("LIMIT 100") and any("lowered" in n for n in c.notes)
    d = v.validate("SELECT ts FROM readings", intent={"max_rows": 5})
    assert d.clean_sql.endswith("LIMIT 5")
    e = v.validate("SELECT ts FROM readings", intent={"max_rows": 10_000})  # intent can never raise the cap
    assert e.clean_sql.endswith("LIMIT 100")


def test_union_cap_applies_to_whole_union(v):
    r = v.validate("SELECT building_id FROM readings UNION SELECT building_id FROM buildings")
    assert r.ok and r.clean_sql.endswith("LIMIT 100")
    assert r.clean_sql.count("LIMIT") == 1


def test_never_raises_on_garbage(v):
    for junk in [b"bytes", 123, ["SELECT 1"], "SELECT \x00", "' OR 1=1 --", "/* */", ";;;"]:
        verdict = v.validate(junk)
        assert verdict.ok is False and verdict.clean_sql is None


def test_deterministic(v):
    sql = "SELECT building_id FROM readings ORDER BY 1"
    assert v.validate(sql) == v.validate(sql)

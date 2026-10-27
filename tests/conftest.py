"""Small deterministic fixture database with the production schema (no BDG2 files needed)."""
import sqlite3

import numpy as np
import pandas as pd
import pytest

BUILDINGS = [("Fox_office_Alpha", "Office"), ("Fox_office_Beta", "Office"), ("Fox_education_Gamma", "Education")]
START, DAYS = "2017-01-01", 63


def make_db(path):
    idx = pd.date_range(START, periods=24 * DAYS, freq="h")
    rows = []
    for i, (b, _) in enumerate(BUILDINGS):
        hour, dow = idx.hour.to_numpy(), idx.dayofweek.to_numpy()
        base = 10 + 5 * i + 20 * ((hour >= 8) & (hour < 18) & (dow < 5))
        vals = base.astype(float)
        if b == "Fox_office_Beta":
            vals[24 * 50: 24 * 50 + 10] = np.nan            # missing block
            vals[24 * 55: 24 * 55 + 12] = 42.4242           # flatline
            vals[(idx >= "2017-02-20") & (idx < "2017-02-27") & (hour < 5)] *= 3  # night baseline jump
        rows += [(b, t.strftime("%Y-%m-%d %H:%M:%S"), None if np.isnan(v) else float(v), 0) for t, v in zip(idx, vals)]
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE buildings (building_id TEXT PRIMARY KEY, site_id TEXT, primary_use TEXT, sub_use TEXT, sqm REAL,
                                yearbuilt INTEGER, timezone TEXT);
        CREATE TABLE readings (building_id TEXT NOT NULL, ts TEXT NOT NULL, electricity_kwh REAL,
                               is_flagged INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (building_id, ts)) WITHOUT ROWID;
        CREATE INDEX idx_readings_ts ON readings(ts);
        CREATE TABLE weather (ts TEXT PRIMARY KEY, air_temp_c REAL, dew_temp_c REAL, wind_speed_ms REAL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE secrets (token TEXT);
    """)
    con.executemany("INSERT INTO buildings VALUES (?,?,?,?,?,?,?)",
                    [(b, "Fox", u, None, 1000.0, 2000, "US/Mountain") for b, u in BUILDINGS])
    con.executemany("INSERT INTO readings VALUES (?,?,?,?)", rows)
    con.executemany("INSERT INTO weather VALUES (?,?,?,?)",
                    [(t.strftime("%Y-%m-%d %H:%M:%S"), 15.0, 5.0, 2.0) for t in idx])
    con.executemany("INSERT INTO meta VALUES (?,?)", [("dataset_version", "test-v1"), ("site_id", "Fox"), ("unit", "kWh"),
                                                      ("selection_rule", "fixture"), ("timestamp_note", "fixture")])
    con.execute("INSERT INTO secrets VALUES ('s3cret')")
    con.commit()
    con.close()


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("db") / "energy.db"
    make_db(p)
    return p


@pytest.fixture(scope="session")
def ctx(db_path):
    from copilot.db import Executor, connect_readonly, introspect

    con = connect_readonly(db_path)
    schema = introspect(con)
    return con, schema, Executor(con, schema, max_rows=500)


class FakeLLM:
    """Scripted LLM: returns queued replies and records prompts. No network."""

    def __init__(self, replies=(), available=True):
        from copilot.llm import LLMReply

        self._R = LLMReply
        self.replies = list(replies)
        self.available = available
        self.prompts = []
        self.calls = 0

    def chat(self, system, user, max_tokens=600, json_mode=False):
        self.calls += 1
        self.prompts.append(user)
        if not self.replies:
            return self._R(False, error="LLM_ERROR")
        r = self.replies.pop(0)
        return r if isinstance(r, self._R) else self._R(True, r)


@pytest.fixture
def fake_llm():
    return FakeLLM

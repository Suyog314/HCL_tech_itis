"""Build the SQLite prototype database from BDG2 CSVs.

Building selection (deterministic, documented in the `meta` table):
  * one site (default ``Fox``) acts as the campus;
  * candidate meters: electricity meters of that site in the cleaned dataset;
  * rank by completeness of cleaned readings (desc), then share of zero readings (asc), then id;
  * keep the top 15.

Semantics:
  * ``readings.electricity_kwh`` is the BDG2 *cleaned* value; NULL = no usable reading (never zero).
  * ``readings.is_flagged = 1`` marks hours where the raw file had a value but BDG2 cleaning removed it
    (outlier/zero-run filtering). Flagged rows are kept, not deleted.
  * Timestamps are site-local wall-clock time, hourly, stored as 'YYYY-MM-DD HH:MM:SS' text.

Usage:  python -m copilot.build_db
"""
from __future__ import annotations

import hashlib
import sqlite3
import sys

import pandas as pd

from . import config


def select_buildings(cleaned: pd.DataFrame, site: str, n: int) -> list[str]:
    cols = [c for c in cleaned.columns if c.split("_")[0] == site]
    sub = cleaned[cols]
    stats = pd.DataFrame({"comp": sub.notna().mean(), "zero": (sub == 0).mean()})
    stats = stats.rename_axis("bid").reset_index()
    stats = stats.sort_values(["comp", "zero", "bid"], ascending=[False, True, True])
    return stats["bid"].head(n).tolist()


def build(db_path=config.DB_PATH, data_dir=config.BDG2_DATA, site=config.SITE_ID, n=config.N_BUILDINGS):
    meters = data_dir / "meters"
    head = pd.read_csv(meters / "cleaned" / "electricity_cleaned.csv", nrows=0).columns
    site_cols = ["timestamp"] + [c for c in head if c.split("_")[0] == site]
    cleaned = pd.read_csv(meters / "cleaned" / "electricity_cleaned.csv", usecols=site_cols, index_col=0)
    chosen = select_buildings(cleaned, site, n)
    raw = pd.read_csv(meters / "raw" / "electricity.csv", usecols=["timestamp"] + chosen, index_col=0)

    c_long = cleaned[chosen].stack(future_stack=True).rename("electricity_kwh")
    r_long = raw[chosen].reindex(cleaned.index).stack(future_stack=True).rename("raw")
    rd = pd.concat([c_long, r_long], axis=1).reset_index()
    rd.columns = ["ts", "building_id", "electricity_kwh", "raw"]
    rd["is_flagged"] = (rd["raw"].notna() & rd["electricity_kwh"].isna()).astype(int)
    rd = rd.drop(columns="raw")

    meta = pd.read_csv(data_dir / "metadata" / "metadata.csv")
    meta = meta[meta.building_id.isin(chosen)][
        ["building_id", "site_id", "primaryspaceusage", "sub_primaryspaceusage", "sqm", "yearbuilt", "timezone"]
    ].rename(columns={"primaryspaceusage": "primary_use", "sub_primaryspaceusage": "sub_use"})

    w = pd.read_csv(data_dir / "weather" / "weather.csv")
    w = w[w.site_id == site][["timestamp", "airTemperature", "dewTemperature", "windSpeed"]]
    w.columns = ["ts", "air_temp_c", "dew_temp_c", "wind_speed_ms"]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(
        """
        CREATE TABLE buildings (building_id TEXT PRIMARY KEY, site_id TEXT, primary_use TEXT, sub_use TEXT,
                                sqm REAL, yearbuilt INTEGER, timezone TEXT);
        CREATE TABLE readings (building_id TEXT NOT NULL REFERENCES buildings(building_id),
                               ts TEXT NOT NULL, electricity_kwh REAL, is_flagged INTEGER NOT NULL DEFAULT 0,
                               PRIMARY KEY (building_id, ts)) WITHOUT ROWID;
        CREATE INDEX idx_readings_ts ON readings(ts);
        CREATE TABLE weather (ts TEXT PRIMARY KEY, air_temp_c REAL, dew_temp_c REAL, wind_speed_ms REAL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    meta.to_sql("buildings", con, if_exists="append", index=False)
    rd.to_sql("readings", con, if_exists="append", index=False, chunksize=50_000)
    w.drop_duplicates("ts").to_sql("weather", con, if_exists="append", index=False)
    digest = hashlib.sha256(pd.util.hash_pandas_object(rd, index=False).values.tobytes()).hexdigest()[:16]
    info = {
        "dataset_version": digest,
        "site_id": site,
        "unit": config.ENERGY_UNIT,
        "selection_rule": "top 15 site electricity meters by cleaned completeness desc, zero share asc, id",
        "buildings": ",".join(chosen),
        "source": "BDG2 electricity_cleaned.csv (values) + electricity.csv (flag derivation)",
        "timestamp_note": "site-local time, hourly, historical (2016-2017), not live",
    }
    con.executemany("INSERT INTO meta VALUES (?, ?)", list(info.items()))
    con.commit()
    con.execute("ANALYZE")
    con.close()
    db_path.unlink(missing_ok=True)
    tmp.rename(db_path)
    return chosen


if __name__ == "__main__":
    ids = build()
    print(f"Built {config.DB_PATH} with {len(ids)} buildings:", *ids, sep="\n  ")
    sys.exit(0)

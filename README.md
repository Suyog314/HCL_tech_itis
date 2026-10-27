# Energy Analytics Copilot

A natural-language analytics assistant for a 15-building campus. It uses hourly electricity data from the
Building Data Genome Project 2 (BDG2). Facility managers ask questions in plain English and get answers backed by
the database, with charts, anomaly alerts, forecasts with uncertainty intervals, and evidence-based recommendations.

## Quick start

```bash
pip install -r requirements.txt          # NumPy 1.x environments need pyarrow < 18
python -m copilot.build_db               # builds data/energy.db from building-data-genome-project-2/data (about 1 min)
cp .env.example .env                     # then set GROQ_API_KEY (optional: example questions work without it)
streamlit run app.py
python -m pytest -q                      # 100 tests, LLM fully mocked, small fixture database
```

## Dataset

| Item | Value |
|---|---|
| Campus | BDG2 site **Fox**: 15 electricity meters (5 education, 5 assembly, 4 office, 1 lodging) |
| Selection | Ranked by completeness of cleaned readings (descending), then share of zero readings (ascending), then ID. The top 15 are kept and the rule is stored in the `meta` table |
| Coverage | 2016-01-01 00:00 to 2017-12-31 23:00, hourly, site-local time. **Historical data, not live** |
| Unit | kWh. This comes from the BDG2 publication (Miller et al., *Sci. Data* 2020); the CSV files themselves carry no unit field |
| Values | `electricity_kwh` holds the BDG2 *cleaned* value. NULL means no reading, never zero (32 of 263,160 rows) |
| Flag | `is_flagged = 1` marks rows where the raw file had a value that BDG2 cleaning removed (14 rows). Flagged rows are kept, not deleted |

Tables: `readings(building_id, ts, electricity_kwh, is_flagged)` with primary key `(building_id, ts)`, an index on
`ts`, `buildings`, `weather` (air temperature, dew point, wind) and `meta`.

## Architecture (`copilot/`)

| Module | Responsibility |
|---|---|
| `db.py` | Read-only connection (URI `mode=ro`, `PRAGMA query_only`, a SQLite authorizer that allows reads only), schema introspection, timeout through a progress handler, fetch caps, server-side pagination |
| `sql_validator.py` | `SQLValidator.validate(sql, intent) -> Verdict`. Uses the sqlglot parser and fails closed; check order and error codes are below |
| `sql_errors.py` | Maps stable error codes to readable messages |
| `sql_optimizer.py` | Rewrites queries only when they pass three checks: validation, result equivalence, and measured gains (see below) |
| `pagination.py` | Rewrites the root `LIMIT`/`OFFSET` to fetch 10 rows per page plus one look-ahead row, never reading past the query's own cap |
| `cache.py` | `SQLQueryCache`: in-memory, TTL, hit/miss counters. The key covers prompt, SQL, page, cap and dataset version |
| `nl2sql.py` | Rule-based scope checks, building and period extraction, clarification, and templates. One LLM call for free-form questions, plus at most one repair attempt |
| `charts.py` | Picks a chart from the question, column types and data. Covers line, bar, grouped bar, pie, scatter, heatmap, metric and forecast band |
| `anomaly.py` | Detects deviations from baseline, high weekend use, night base-load increases, level shifts, flatlines, suspicious zeros and missing readings |
| `forecast.py` | Hour-of-week median baseline compared with gradient boosting in a chronological backtest. Provides 24 h and 7-day horizons with 80% intervals and an actual-vs-expected mode |
| `recommend.py` | Each recommendation gives the finding, measured evidence, a hypothesis, an action, a confidence level and its limitations |
| `explain.py` | `ExplanationChecker` (an evidence-only LLM explanation, or a deterministic template) and `SanityChecker` |
| `llm.py` | OpenAI-compatible client for Groq. The key comes from the environment only, and errors are reduced to codes |
| `pipeline.py` | `Copilot` orchestrator |
| `app.py` | Streamlit UI with tabs: Ask, Anomalies, Forecast, Dataset, Upload CSV |

### Validator order (stops at the first failure)
`PARSE_ERROR`/`TOO_COMPLEX` (empty, size, AST nodes, nesting, joins, recursive CTE), then `MULTI_STATEMENT`, then
`NOT_SELECT` (the root must be SELECT/UNION/INTERSECT/EXCEPT), then `FORBIDDEN_STATEMENT` (write, DDL or admin nodes
anywhere, including inside CTEs), then `TABLE_NOT_ALLOWED`, then `COLUMN_NOT_ALLOWED`, then `NO_DATA_SOURCE`, then
`FORBIDDEN_FUNCTION` (an allowlist, so unknown functions are rejected), then `OUT_OF_RANGE` (literal `ts` predicates
must fall inside the data range). Finally the row cap is added, or an existing cap is lowered, and the change is
recorded in `notes`. When `ok` is false, `clean_sql` is None, and the validator never raises.

### Optimizer rule
A rewrite is kept only if all of the following hold. Otherwise the original SQL is kept and the reason is recorded:
- it re-validates;
- results are identical (ordered when the query has ORDER BY; NULLs compared as NULLs);
- the `EXPLAIN QUERY PLAN` score improves (for index-targeted rules);
- the median of 5 interleaved timings is at least 1.15× faster.

Measured on the real DB:

| Rewrite | Before | After | Result |
|---|---|---|---|
| `DATE(ts) = '2017-03-01'` → half-open range | 43.1 ms | 0.8 ms | kept |
| `STRFTIME('%Y', ts) = '2017'` → half-open range | 88.4 ms | 21.8 ms | kept |
| `SELECT *` → named columns | 0.33 ms | 0.32 ms | rejected (no measurable gain) |

`LIKE '%x%'` → `IN (...)` is only used because the building set is closed: a foreign key is verified at startup.

### Forecast evaluation
For each building (or the campus total), two 7-day chronological folds before `as_of` compare the baseline with
gradient boosting by MAE. Gradient boosting is used only if it is at least 5% better. Every lag is 168 h or more,
and future weather is not used in forecast mode, so there is no leakage (covered by a test).

| Check | Baseline MAE | Gradient boosting MAE | Chosen |
|---|---|---|---|
| Example (Fox_office_Clayton, end of 2017) | 10.8 | 9.9 | gradient boosting |
| Example (Fox_office_Clayton, June 2017) | 7.6 | 10.5 | baseline |

Fallbacks:
- **Fewer than 28 days of usable history:** a seasonal-naive forecast with a wide, flagged interval.
- **Fewer than 7 days:** no forecast.

## Security notes
- Every SQL string (from the LLM, templates, analytics or pagination) passes `SQLValidator` before execution. The
  database layer is read-only on its own, independent of the validator.
- Keep the Groq key in `.env`, which is git-ignored. It is never logged or shown, and API errors are reduced to codes.

## Known limitations
- The kWh unit comes from the BDG2 documentation; the files do not encode it.
- Forecasts "after the last reading" start at 2018-01-01, after the data ends. They are projections from historical
  data, not live forecasts.
- Relative periods ("last 30 days") are measured back from the last reading (2017-12-31).
- The sanity checker allows any small integer up to 24 without checking (counts, hours). It flags unverified claims
  but does not rewrite them.
- Anomaly thresholds are heuristics (configurable in `AnomalyConfig`). Flatline detection can flag constant
  synthetic or capped loads.
- The cache is per process and resets on restart.
- The default LLM is `openai/gpt-oss-120b` on Groq (set `LLM_MODEL`). Questions outside the templates need it.

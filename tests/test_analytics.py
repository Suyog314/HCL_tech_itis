"""Charts, anomalies, forecasts, recommendations."""
import numpy as np
import pandas as pd

from copilot.anomaly import Anomaly, AnomalyConfig, AnomalyDetector
from copilot.charts import render, select_chart
from copilot.forecast import Forecaster
from copilot.recommend import from_anomalies, from_forecast, from_profile


# ---- charts ----------------------------------------------------------------------------------
def test_chart_selection_by_schema():
    t = pd.DataFrame({"day": ["2017-01-01", "2017-01-02"], "total_kwh": [1.0, 2.0]})
    assert select_chart("daily trend", t).kind == "line"
    b = pd.DataFrame({"building_id": ["a", "b", "c"], "total_kwh": [3.0, 2.0, 1.0]})
    s = select_chart("which building used most", b)
    assert s.kind == "bar" and s.unit == "kWh" and "kWh" in s.y_label
    assert select_chart("share of consumption by building", b).kind == "pie"
    sc = pd.DataFrame({"air_temp_c": [1.0, 2.0, 3.0], "avg_kwh": [5.0, 6.0, 7.0]})
    assert select_chart("consumption vs temperature", sc).kind == "scatter"
    h = pd.DataFrame({"hour": [0, 1, 0, 1], "weekday": ["Mon", "Mon", "Tue", "Tue"], "avg_kwh": [1.0, 2, 3, 4]})
    assert select_chart("heatmap by hour and weekday", h).kind == "heatmap"
    g = pd.DataFrame({"building_id": ["a", "b"], "weekend_avg_kwh": [1.0, 2.0], "weekday_avg_kwh": [3.0, 4.0]})
    assert select_chart("weekend vs weekday", g).y_multi == ["weekend_avg_kwh", "weekday_avg_kwh"]
    assert select_chart("x", pd.DataFrame({"total_kwh": [5.0]})).kind == "metric"


def test_chart_dynamic_columns_and_empty():
    df = pd.DataFrame({"Meter": ["m1", "m2"], "Usage": [1.0, 2.0]})  # unfamiliar names
    spec = select_chart("compare", df)
    assert spec.kind == "bar" and spec.x == "Meter" and render(spec, df) is not None
    assert select_chart("q", pd.DataFrame()).kind == "none"
    assert select_chart("q", None).kind == "none"
    assert select_chart("q", pd.DataFrame({"name": ["a"]})).kind == "none"


def test_chart_folds_series_and_does_not_mutate_spec():
    rows = [(f"2017-01-0{d}", f"b{i:02d}", float(i + d)) for d in range(1, 4) for i in range(12)]
    df = pd.DataFrame(rows, columns=["day", "building_id", "total_kwh"])
    spec = select_chart("trend", df)
    fig = render(spec, df)
    assert len(fig.data) == 8 and any(t.name == "Other" for t in fig.data)
    g = pd.DataFrame({"building_id": ["a"], "weekend_avg_kwh": [1.0], "weekday_avg_kwh": [3.0]})
    s2 = select_chart("weekend vs weekday", pd.concat([g, g.assign(building_id="b")]))
    render(s2, g)
    assert s2.color is None and s2.y == "weekend_avg_kwh"


# ---- anomalies --------------------------------------------------------------------------------
def _hourly(ctx, building=None):
    _, _, ex = ctx
    w = f"WHERE building_id = '{building}'" if building else ""
    from copilot.db import Executor

    big = Executor(ex.con, ex.schema, max_rows=100_000)
    return big.run(f"SELECT building_id, ts, electricity_kwh AS kwh FROM readings {w} ORDER BY building_id, ts").df


def test_detects_missing_flatline_and_night(ctx):
    items = AnomalyDetector().detect(_hourly(ctx))
    kinds = {(a.building_id, a.kind, a.category) for a in items}
    assert ("Fox_office_Beta", "missing_readings", "data_quality") in kinds
    assert ("Fox_office_Beta", "flatline", "data_quality") in kinds
    assert ("Fox_office_Beta", "night_baseline_increase", "anomaly") in kinds
    # clean, perfectly periodic buildings have no consumption anomalies
    assert not [a for a in items if a.building_id == "Fox_office_Alpha" and a.category == "anomaly"]
    night = next(a for a in items if a.kind == "night_baseline_increase")
    assert night.observed > night.expected and night.deviation_pct >= 20 and night.severity in ("high", "medium", "low")


def test_zero_handling_legit_vs_suspicious():
    idx = pd.date_range("2017-01-01", periods=24 * 30, freq="h")
    always_on = pd.DataFrame({"building_id": "a", "ts": idx, "kwh": 10.0})
    always_on.loc[100:105, "kwh"] = 0.0
    det = AnomalyDetector(AnomalyConfig())
    assert any(a.kind == "suspicious_zeros" for a in det.data_quality(always_on))
    cycling = pd.DataFrame({"building_id": "b", "ts": idx, "kwh": np.where(idx.hour < 6, 0.0, 10.0)})
    assert not any(a.kind == "suspicious_zeros" for a in det.data_quality(cycling))  # zeros are normal here


def test_baseline_and_weekend_and_shift():
    idx = pd.date_range("2017-01-02", periods=24 * 70, freq="h")
    k = np.where((idx.dayofweek < 5) & (idx.hour >= 8) & (idx.hour < 18), 30.0, 10.0)
    k = k + (np.arange(len(idx)) % 7) * 0.01  # small noise so MAD > 0
    df = pd.DataFrame({"building_id": "a", "ts": idx, "kwh": k})
    df.loc[df.ts.dt.date == pd.Timestamp("2017-02-15").date(), "kwh"] *= 3
    we = (df.ts >= "2017-02-25") & (df.ts < "2017-02-27")
    df.loc[we, "kwh"] = 30.0
    df.loc[df.ts >= "2017-03-20", "kwh"] *= 2
    items = AnomalyDetector().detect(df)
    kinds = {a.kind for a in items}
    assert {"above_baseline", "high_weekend", "sudden_level_shift"} <= kinds
    spike = next(a for a in items if a.kind == "above_baseline" and a.start == "2017-02-15")
    assert spike.deviation_pct > 100


def test_detect_empty():
    assert AnomalyDetector().detect(pd.DataFrame(columns=["building_id", "ts", "kwh"])) == []


# ---- forecast ---------------------------------------------------------------------------------
def _series(days, noise=0.0):
    idx = pd.date_range("2017-01-01", periods=24 * days, freq="h")
    v = 10 + 20 * ((idx.hour >= 8) & (idx.hour < 18)) + noise * np.random.default_rng(0).standard_normal(len(idx))
    return pd.Series(v, index=idx)


def test_forecast_intervals_and_horizons():
    s = _series(70, noise=1.0)
    for h in (24, 168):
        fc = Forecaster().forecast(s, horizon_hours=h, target="x")
        assert fc.status == "ok" and len(fc.frame) == h
        assert (fc.frame.lower <= fc.frame.expected).all() and (fc.frame.expected <= fc.frame.upper).all()
        assert fc.metrics["baseline"]["mae"] is not None and "rmse" in fc.metrics["baseline"]
        assert any("not live" in w for w in fc.warnings)


def test_forecast_backtest_actual_vs_expected():
    s = _series(70)
    s[s.index >= "2017-03-05"] *= 1.5
    fc = Forecaster().forecast(s, as_of="2017-03-05", horizon_hours=24, target="x")
    assert fc.summary["deviation_pct"] > 30 and fc.summary["hours_above_interval"] > 0


def test_forecast_no_leakage():
    s = _series(70)
    s2 = s.copy()
    s2[s2.index >= "2017-03-01"] = 9999.0  # future values must not change the forecast
    a = Forecaster().forecast(s, as_of="2017-03-01", horizon_hours=24)
    b = Forecaster().forecast(s2, as_of="2017-03-01", horizon_hours=24)
    assert np.allclose(a.frame.expected, b.frame.expected)


def test_forecast_insufficient_history():
    assert Forecaster().forecast(_series(3), horizon_hours=24).status == "insufficient_history"
    fb = Forecaster().forecast(_series(10), horizon_hours=24)
    assert fb.status == "fallback" and fb.warnings


# ---- recommendations --------------------------------------------------------------------------
def test_recommendations_are_evidence_based():
    a = Anomaly("b1", "night_baseline_increase", "anomaly", "2017-02-20", "2017-02-26", 30.0, 10.0, 200.0, "high")
    dq = Anomaly("b1", "flatline", "data_quality", "2017-02-01 00:00", "2017-02-01 12:00", 4.2, None, None, "medium",
                 detail="value repeated for 12 consecutive hours")
    recs = from_anomalies([a, dq])
    night = next(r for r in recs if r.source == "anomaly")
    assert "30.0 kWh" in night.evidence and "+200.0%" in night.evidence
    assert "Hypothesis" in night.possible_explanation and "saving" not in night.suggested_action.lower()
    assert any(r.source == "data_quality" for r in recs)
    prof = pd.DataFrame({"building_id": ["o1", "o2"], "primary_use": ["Office", "Office"],
                         "night_avg_kwh": [80.0, 10.0], "day_avg_kwh": [100.0, 100.0]})
    pr = from_profile(prof)
    assert [r.building_id for r in pr] == ["o1"] and "80%" in pr[0].evidence
    assert from_anomalies([]) == [] and from_profile(pd.DataFrame()) == []


def test_recommendation_from_forecast():
    s = _series(70)
    s[s.index >= "2017-03-05"] *= 1.5
    fc = Forecaster().forecast(s, as_of="2017-03-05", horizon_hours=24, target="x")
    recs = from_forecast(fc)
    assert recs and "Actual" in recs[0].evidence

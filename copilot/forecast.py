"""Deterministic expected-consumption and forecasting models (no LLM arithmetic).

* Baseline: median of the same hour-of-week over the previous ``baseline_weeks`` weeks.
* Candidate: scikit-learn HistGradientBoosting on calendar features + lags >= 168 h (valid for a 7-day
  horizon without recursion) and, only in "expected" mode where the period is already observed,
  air temperature. Future weather is unknown in forecast mode, so it is not used there (no leakage).
* Model choice: chronological backtest on the last ``backtest_days`` before ``as_of``; the ML model is
  used only if it lowers MAE by at least ``min_gain``. Otherwise the baseline is kept.
* Uncertainty: 10th/90th percentiles of backtest residuals of the chosen model (80% empirical interval).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAGS = (168, 336, 504)


@dataclass
class ForecastConfig:
    baseline_weeks: int = 4
    min_history_days: int = 28
    min_completeness: float = 0.8
    backtest_days: int = 14
    min_gain: float = 0.05
    unreliable_mape: float = 30.0
    use_ml: bool = True


@dataclass
class ForecastResult:
    target: str
    as_of: str
    horizon_hours: int
    status: str  # ok | fallback | insufficient_history
    model: str | None
    frame: pd.DataFrame | None
    history: pd.DataFrame | None
    metrics: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    unit: str = "kWh"


def _metrics(y, p) -> dict:
    m = ~(np.isnan(y) | np.isnan(p))
    if not m.any():
        return {"mae": None, "rmse": None, "mape": None, "n": 0}
    e = y[m] - p[m]
    nz = y[m] != 0
    mape = float(np.mean(np.abs(e[nz] / y[m][nz])) * 100) if nz.any() else None
    return {"mae": round(float(np.mean(np.abs(e))), 3), "rmse": round(float(np.sqrt(np.mean(e ** 2))), 3),
            "mape": round(mape, 2) if mape is not None else None, "n": int(m.sum())}


def _baseline(series: pd.Series, idx: pd.DatetimeIndex, weeks: int) -> np.ndarray:
    vals = np.stack([series.reindex(idx - pd.Timedelta(hours=168 * k)).to_numpy(float) for k in range(1, weeks + 1)])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(vals, axis=0)


def _features(series: pd.Series, idx: pd.DatetimeIndex, weather: pd.Series | None) -> pd.DataFrame:
    X = pd.DataFrame(index=idx)
    X["hour"], X["dow"], X["month"] = idx.hour, idx.dayofweek, idx.month
    doy = idx.dayofyear.to_numpy()
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for lag in LAGS:
        X[f"lag{lag}"] = series.reindex(idx - pd.Timedelta(hours=lag)).to_numpy()
    if weather is not None:
        X["air_temp_c"] = weather.reindex(idx).to_numpy()
    return X


class Forecaster:
    def __init__(self, config: ForecastConfig | None = None, unit: str = "kWh"):
        self.cfg, self.unit = config or ForecastConfig(), unit

    def _ml_predict(self, series, train_end, idx, weather):
        from sklearn.ensemble import HistGradientBoostingRegressor

        hist = series[series.index < train_end].dropna()
        hist = hist[hist.index >= hist.index.min() + pd.Timedelta(hours=max(LAGS))]
        if len(hist) < 24 * 28:
            return None
        Xtr = _features(series.where(series.index < train_end), hist.index, weather)
        model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
        model.fit(Xtr, hist.to_numpy())
        return model.predict(_features(series.where(series.index < train_end), idx, weather))

    def forecast(self, series: pd.Series, as_of=None, horizon_hours: int = 24, target: str = "",
                 weather: pd.Series | None = None) -> ForecastResult:
        cfg = self.cfg
        s = pd.to_numeric(series, errors="coerce").sort_index()
        s.index = pd.to_datetime(s.index)
        s = s.asfreq("h")
        data_end = s.index.max()
        as_of = pd.Timestamp(as_of) if as_of is not None else data_end + pd.Timedelta(hours=1)
        as_of = min(as_of, data_end + pd.Timedelta(hours=1)).floor("h")
        idx = pd.date_range(as_of, periods=horizon_hours, freq="h")
        hist = s[s.index < as_of]
        res = ForecastResult(target, str(as_of), horizon_hours, "ok", None, None, None, unit=self.unit)
        recent = hist[hist.index >= as_of - pd.Timedelta(days=cfg.min_history_days)]
        span_ok = len(hist) >= 24 * cfg.min_history_days
        if not span_ok or recent.notna().mean() < cfg.min_completeness:
            if hist.notna().sum() >= 24 * 7:
                res.status, res.model = "fallback", "seasonal_naive_last_week"
                exp_ = s.reindex(idx - pd.Timedelta(hours=168)).to_numpy(float)
                spread = float(np.nanstd(hist.tail(24 * 7))) * 1.28
                res.frame = pd.DataFrame({"ts": idx, "expected": exp_, "lower": np.clip(exp_ - spread, 0, None),
                                          "upper": exp_ + spread})
                res.warnings.append("Limited or incomplete history: using last week's values with a wide, unvalidated interval.")
            else:
                res.status = "insufficient_history"
                res.warnings.append(f"Less than 7 days of usable history before {as_of}; no forecast produced.")
                return res
        else:
            use_weather = weather if (weather is not None and idx.max() <= data_end
                                      and weather.reindex(idx).notna().all()) else None
            # chronological backtest: 7-day folds before as_of, each model sees only data before its fold
            bt_start = as_of - pd.Timedelta(days=cfg.backtest_days)
            folds = [pd.date_range(f0, periods=168, freq="h")
                     for f0 in pd.date_range(bt_start, as_of - pd.Timedelta(hours=168), freq="7D")]
            bt_idx = folds[0].append(folds[1:]) if folds else pd.DatetimeIndex([])
            y_bt = s.reindex(bt_idx).to_numpy(float)
            p_base_bt = np.concatenate([_baseline(s.where(s.index < f[0]), f, cfg.baseline_weeks) for f in folds])
            res.metrics["baseline"] = _metrics(y_bt, p_base_bt)
            chosen, resid = "hour_of_week_median", y_bt - p_base_bt
            if cfg.use_ml:
                bt_weather = weather if weather is not None and use_weather is not None else None
                try:
                    parts = [self._ml_predict(s, f[0], f, bt_weather) for f in folds]
                    p_ml_bt = None if any(p is None for p in parts) else np.concatenate(parts)
                except Exception:
                    p_ml_bt = None
                if p_ml_bt is not None:
                    res.metrics["gradient_boosting"] = _metrics(y_bt, p_ml_bt)
                    b, m = res.metrics["baseline"]["mae"], res.metrics["gradient_boosting"]["mae"]
                    if b and m is not None and m < b * (1 - cfg.min_gain):
                        chosen, resid = "gradient_boosting", y_bt - p_ml_bt
            exp_ = (self._ml_predict(s, as_of, idx, use_weather) if chosen == "gradient_boosting"
                    else _baseline(s.where(s.index < as_of), idx, cfg.baseline_weeks))
            exp_ = np.clip(np.asarray(exp_, float), 0, None)
            r = resid[~np.isnan(resid)]
            lo_q, hi_q = (np.quantile(r, 0.1), np.quantile(r, 0.9)) if len(r) >= 24 else (-np.nanstd(hist), np.nanstd(hist))
            res.model = chosen
            res.frame = pd.DataFrame({"ts": idx, "expected": exp_, "lower": np.clip(exp_ + lo_q, 0, None), "upper": exp_ + hi_q})
            mape = res.metrics.get(chosen if chosen == "gradient_boosting" else "baseline", {}).get("mape")
            if mape is not None and mape > cfg.unreliable_mape:
                res.warnings.append(f"Backtest MAPE {mape:.1f}% is high; treat this forecast as low reliability.")
            if use_weather is not None:
                res.metrics["uses_weather"] = True
            if hist.tail(24 * 28).isna().any():
                res.warnings.append(f"{int(hist.tail(24 * 28).isna().sum())} missing hour(s) in the last 28 days of history.")
        # actual vs expected where the horizon is already observed
        actual = s.reindex(idx).to_numpy(float)
        res.frame["actual"] = actual
        res.history = pd.DataFrame({"ts": hist.tail(24 * 7).index, "actual": hist.tail(24 * 7).to_numpy()})
        f = res.frame
        summ = {"expected_total_kwh": round(float(np.nansum(f["expected"])), 1),
                "interval_total_kwh": [round(float(np.nansum(f["lower"])), 1), round(float(np.nansum(f["upper"])), 1)],
                "interval_level_pct": 80, "period": [str(idx.min()), str(idx.max())]}
        if np.isfinite(actual).any():
            m = np.isfinite(actual)
            a_tot, e_tot = float(actual[m].sum()), float(f["expected"].to_numpy()[m].sum())
            summ.update(actual_total_kwh=round(a_tot, 1), deviation_kwh=round(a_tot - e_tot, 1),
                        deviation_pct=round((a_tot - e_tot) / e_tot * 100, 1) if e_tot else None,
                        hours_above_interval=int((actual[m] > f["upper"].to_numpy()[m]).sum()),
                        hours_below_interval=int((actual[m] < f["lower"].to_numpy()[m]).sum()), observed_hours=int(m.sum()))
            if m.sum() < len(actual):
                res.warnings.append("Part of the period has no actual readings; deviation covers observed hours only.")
        else:
            prior = hist.tail(24 * 28)
            if prior.notna().sum() > 0 and res.status == "ok":
                ref = float(prior.mean()) * horizon_hours
                chg = (summ["expected_total_kwh"] - ref) / ref * 100 if ref else 0
                mape = res.metrics.get("gradient_boosting" if res.model == "gradient_boosting" else "baseline", {}).get("mape")
                summ["vs_last_28d_mean_pct"] = round(chg, 1)
                if abs(chg) >= 20 and mape is not None and mape < 20:
                    summ["potential_anomaly"] = (f"Forecast total is {chg:+.1f}% vs the trailing 28-day average for the same "
                                                 f"length (backtest MAPE {mape:.1f}%).")
            res.warnings.append("Forecast starts after the last available reading (historical dataset, not live data).")
        res.summary = summ
        return res

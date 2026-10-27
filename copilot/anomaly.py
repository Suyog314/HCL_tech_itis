"""Anomaly and data-quality detection on hourly building readings.

Input: DataFrame with columns building_id, ts (datetime or ISO text), kwh (NULL = missing).
Baselines use only data *before* the evaluated period (trailing windows), so there is no look-ahead.
Results separate genuine consumption anomalies (category="anomaly") from data-quality issues
(category="data_quality").
"""
from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AnomalyConfig:
    baseline_weeks: int = 6          # same-weekday history for daily baseline
    min_baseline_points: int = 3
    daily_dev_pct: float = 25.0      # |deviation| needed to flag a day
    robust_z: float = 3.0
    weekend_weeks: int = 8
    weekend_ratio_delta: float = 0.15
    night_hours: tuple[int, int] = (0, 5)  # [start, end) local hours
    night_weeks: int = 4
    night_increase_pct: float = 20.0
    shift_days: int = 7
    shift_pct: float = 30.0
    flatline_hours: int = 6
    zero_run_hours: int = 3
    min_day_completeness: float = 0.9


@dataclass
class Anomaly:
    building_id: str
    kind: str
    category: str  # anomaly | data_quality
    start: str
    end: str
    observed: float | None
    expected: float | None
    deviation_pct: float | None
    severity: str
    unit: str = "kWh"
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


def _sev(pct: float | None) -> str:
    a = abs(pct or 0)
    return "high" if a >= 50 else "medium" if a >= 30 else "low"


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["building_id", "ts", "kwh"]].copy()
    d["ts"] = pd.to_datetime(d["ts"])
    d["kwh"] = pd.to_numeric(d["kwh"], errors="coerce")
    return d.sort_values(["building_id", "ts"])


def _daily(d: pd.DataFrame, cfg: AnomalyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = d.assign(day=d["ts"].dt.floor("D"))
    g = d.groupby(["day", "building_id"])["kwh"]
    total = g.sum(min_count=1).unstack()
    count = g.count().unstack()
    total = total.where(count >= 24 * cfg.min_day_completeness)  # incomplete days are not judged
    full = pd.date_range(total.index.min(), total.index.max(), freq="D")
    return total.reindex(full), count.reindex(full).fillna(0)


def _trailing_median(s: pd.DataFrame, lags: list[int], min_points: int):
    stack = np.stack([s.shift(k).to_numpy(float) for k in lags])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        valid = np.sum(~np.isnan(stack), axis=0)
        med = np.nanmedian(np.where(valid >= min_points, stack, np.nan), axis=0)
        mad = np.nanmedian(np.abs(stack - med), axis=0)
    return pd.DataFrame(med, index=s.index, columns=s.columns), pd.DataFrame(mad, index=s.index, columns=s.columns)


class AnomalyDetector:
    def __init__(self, config: AnomalyConfig | None = None, unit: str = "kWh"):
        self.cfg = config or AnomalyConfig()
        self.unit = unit

    def detect(self, df: pd.DataFrame, start=None, end=None) -> list[Anomaly]:
        if df is None or df.empty:
            return []
        d = _prep(df)
        out = self.data_quality(d) + self.baseline_deviation(d) + self.weekend(d) + self.night_baseline(d) + self.sudden_change(d)
        if start is not None or end is not None:
            s = pd.Timestamp(start) if start is not None else pd.Timestamp.min
            e = pd.Timestamp(end) if end is not None else pd.Timestamp.max
            out = [a for a in out if pd.Timestamp(a.end) >= s and pd.Timestamp(a.start) < e]
        dq = [a for a in out if a.category == "data_quality"]
        for a in out:
            if a.category == "anomaly":
                for q in dq:
                    if q.building_id == a.building_id and q.start[:10] <= a.end[:10] and q.end[:10] >= a.start[:10]:
                        a.warnings.append(f"Overlaps a data-quality issue ({q.kind}, {q.start[:16]} to {q.end[:16]}); may not be real consumption.")
                        break
        rank = {"high": 0, "medium": 1, "low": 2}
        return sorted(out, key=lambda a: (rank[a.severity], -abs(a.deviation_pct or 0), a.start))

    @staticmethod
    def to_frame(items: list[Anomaly]) -> pd.DataFrame:
        return pd.DataFrame([asdict(a) for a in items])

    # ---- data quality -----------------------------------------------------------------------
    def data_quality(self, d: pd.DataFrame) -> list[Anomaly]:
        cfg, out = self.cfg, []
        for b, g in d.groupby("building_id"):
            g = g.set_index("ts")["kwh"]
            full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
            # missing readings: runs of NULL hours
            out += self._runs(b, full.isna(), 1, "missing_readings", lambda r: f"{len(r)} hour(s) without a reading",
                              None, sev_hours=(6, 24))
            # flatline: identical non-zero values repeated
            nz = full.where(full != 0)
            same = nz.notna() & (nz == nz.shift(1))
            run_start = same & ~same.shift(1, fill_value=False)
            mask = same | run_start.shift(-1, fill_value=False)
            out += self._runs(b, mask, cfg.flatline_hours, "flatline",
                              lambda r: f"value repeated for {len(r)} consecutive hours", full, sev_hours=(12, 48))
            # zeros: only suspicious when the building normally never drops to zero
            zero_share = float((full == 0).mean())
            p1 = float(full[full > 0].quantile(0.01)) if (full > 0).any() else 0.0
            if zero_share < 0.02 and p1 > 0:
                out += self._runs(b, full == 0, cfg.zero_run_hours, "suspicious_zeros",
                                  lambda r: f"{len(r)} consecutive zero readings; building's 1st-percentile load is {p1:.1f} {self.unit}",
                                  full, sev_hours=(6, 24))
        return out

    def _runs(self, b, mask: pd.Series, min_len: int, kind: str, describe, values, sev_hours=(6, 24)) -> list[Anomaly]:
        out = []
        if not mask.any():
            return out
        grp = (mask != mask.shift()).cumsum()
        for _, r in mask[mask].groupby(grp[mask]):
            if len(r) < min_len:
                continue
            obs = float(values.loc[r.index[0]]) if values is not None and pd.notna(values.loc[r.index[0]]) else None
            sev = "high" if len(r) >= sev_hours[1] else "medium" if len(r) >= sev_hours[0] else "low"
            out.append(Anomaly(b, kind, "data_quality", str(r.index[0]), str(r.index[-1]), obs, None, None, sev,
                               self.unit, describe(r), ["Possible metering or data issue, not necessarily real consumption."]))
        return out

    # ---- consumption anomalies --------------------------------------------------------------
    def baseline_deviation(self, d: pd.DataFrame) -> list[Anomaly]:
        cfg = self.cfg
        total, count = _daily(d, cfg)
        exp_, mad = _trailing_median(total, [7 * k for k in range(1, cfg.baseline_weeks + 1)], cfg.min_baseline_points)
        dev = (total - exp_) / exp_ * 100
        scale = np.maximum(1.4826 * mad, 0.01 * exp_.abs())  # floor: perfectly stable history must not give NaN z
        z = (total - exp_) / scale.replace(0, np.nan)
        flag = (dev.abs() >= cfg.daily_dev_pct) & (z.abs() >= cfg.robust_z) & (exp_ > 0)
        out = []
        for day, b in zip(*np.where(flag.to_numpy())):
            dt, bid = total.index[day], total.columns[b]
            o, e, p = float(total.iat[day, b]), float(exp_.iat[day, b]), float(dev.iat[day, b])
            out.append(Anomaly(bid, "above_baseline" if p > 0 else "below_baseline", "anomaly", str(dt.date()),
                               str(dt.date()), round(o, 1), round(e, 1), round(p, 1), _sev(p), self.unit,
                               f"Daily total vs median of the same weekday over the previous {cfg.baseline_weeks} weeks "
                               f"(robust z={float(z.iat[day, b]):.1f})"))
        return out

    def weekend(self, d: pd.DataFrame) -> list[Anomaly]:
        cfg = self.cfg
        total, _ = _daily(d, cfg)
        wk = total.index.to_period("W-SUN")
        is_we = total.index.dayofweek >= 5
        we = total[is_we].groupby(wk[is_we]).mean()
        wd = total[~is_we].groupby(wk[~is_we]).mean()
        ratio = (we / wd).dropna(how="all")
        exp_ = ratio.shift(1).rolling(cfg.weekend_weeks, min_periods=4).median()
        out = []
        for w in ratio.index:
            for b in ratio.columns:
                r, e = ratio.at[w, b], exp_.at[w, b] if w in exp_.index else np.nan
                if pd.notna(r) and pd.notna(e) and e > 0 and r - e >= cfg.weekend_ratio_delta and r > e * 1.2:
                    p = (r - e) / e * 100
                    out.append(Anomaly(b, "high_weekend", "anomaly", str(w.start_time.date()), str(w.end_time.date()),
                                       round(float(we.at[w, b]), 1), round(float(e * wd.at[w, b]), 1), round(p, 1), _sev(p),
                                       self.unit, f"Weekend/weekday daily ratio {r:.2f} vs typical {e:.2f}"))
        return out

    def night_baseline(self, d: pd.DataFrame) -> list[Anomaly]:
        cfg = self.cfg
        h0, h1 = cfg.night_hours
        n = d[(d["ts"].dt.hour >= h0) & (d["ts"].dt.hour < h1)]
        if n.empty:
            return []
        wk = n.assign(w=n["ts"].dt.to_period("W-SUN")).groupby(["w", "building_id"])["kwh"].mean().unstack()
        exp_ = wk.shift(1).rolling(cfg.night_weeks, min_periods=3).median()
        dev = (wk - exp_) / exp_ * 100
        out = []
        for w in wk.index:
            for b in wk.columns:
                p = dev.at[w, b]
                if pd.notna(p) and p >= cfg.night_increase_pct and exp_.at[w, b] > 0:
                    out.append(Anomaly(b, "night_baseline_increase", "anomaly", str(w.start_time.date()), str(w.end_time.date()),
                                       round(float(wk.at[w, b]), 2), round(float(exp_.at[w, b]), 2), round(float(p), 1),
                                       _sev(p), self.unit,
                                       f"Mean hourly load {h0:02d}:00-{h1:02d}:00 vs median of previous {cfg.night_weeks} weeks"))
        return out

    def sudden_change(self, d: pd.DataFrame) -> list[Anomaly]:
        cfg = self.cfg
        total, _ = _daily(d, cfg)
        k = cfg.shift_days
        before = total.rolling(k, min_periods=k).mean()
        after = total[::-1].rolling(k, min_periods=k).mean()[::-1]
        pct = (after - before.shift(1)) / before.shift(1) * 100
        out = []
        for b in total.columns:
            s = pct[b]
            cand = s[(s.abs() >= cfg.shift_pct)]
            # one event per contiguous run: the day with the largest absolute change
            if cand.empty:
                continue
            grp = (cand.index.to_series().diff() != pd.Timedelta("1D")).cumsum()
            for _, r in cand.groupby(grp):
                day = r.abs().idxmax()
                p = float(s[day])
                out.append(Anomaly(b, "sudden_level_shift", "anomaly", str(day.date()),
                                   str((day + pd.Timedelta(days=k - 1)).date()), round(float(after.at[day, b]), 1),
                                   round(float(before.shift(1).at[day, b]), 1), round(p, 1), _sev(p), self.unit,
                                   f"Mean daily total for {k} days from this date vs the {k} days before"))
        return out

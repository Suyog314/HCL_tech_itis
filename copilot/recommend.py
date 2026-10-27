"""Evidence-based recommendations from query results, anomalies and forecasts.

Every recommendation separates the measured observation (finding + evidence) from a hypothesis
(possible explanation). No savings are promised and no cause is asserted as verified.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import pandas as pd

from .anomaly import Anomaly


@dataclass
class Recommendation:
    building_id: str
    finding: str
    evidence: str
    possible_explanation: str
    suggested_action: str
    confidence: str  # high | medium | low
    limitations: str
    source: str  # anomaly | profile | forecast | data_quality


ACTIONS = {
    "night_baseline_increase": ("Night-time (00:00-05:00) load rose above its recent baseline.",
                                "Hypothesis: equipment or HVAC schedules running outside occupied hours, or new always-on loads.",
                                "Review BMS/HVAC and lighting schedules for those nights; walk through after hours to identify running equipment."),
    "high_weekend": ("Weekend consumption was unusually high relative to weekdays.",
                     "Hypothesis: weekend events or occupancy, or schedules not set back on weekends.",
                     "Check event calendars and confirm weekend set-back schedules are active."),
    "above_baseline": ("Daily consumption exceeded the same-weekday baseline.",
                       "Hypothesis: unusual occupancy, events, equipment faults or weather; not verified from meter data alone.",
                       "Compare with the building's event log and weather; inspect major equipment if no known reason exists."),
    "below_baseline": ("Daily consumption fell well below the same-weekday baseline.",
                       "Hypothesis: closure/holiday, equipment off, or a metering issue.",
                       "Confirm whether the building was closed; if not, verify the meter is reporting correctly."),
    "sudden_level_shift": ("Consumption level changed abruptly and persisted for about a week.",
                           "Hypothesis: schedule change, retrofit, equipment added/removed, or a meter change.",
                           "Check maintenance and change records around that date."),
}
DQ_ACTION = ("Meter data-quality issue detected.", "Possible meter, logger or communication fault.",
             "Ask the metering team to check the meter/logger; exclude affected periods from performance reporting.")


def from_anomalies(items: list[Anomaly], max_items: int = 8) -> list[Recommendation]:
    grouped: dict[tuple[str, str], list[Anomaly]] = defaultdict(list)
    for a in items:
        grouped[(a.building_id, a.kind)].append(a)
    recs = []
    for (b, kind), group in grouped.items():
        group.sort(key=lambda a: -abs(a.deviation_pct or 0))
        top = group[0]
        n = len(group)
        if top.category == "data_quality":
            finding, expl, action = DQ_ACTION
            ev = f"{n} {kind.replace('_', ' ')} event(s); largest: {top.detail} ({top.start[:16]} to {top.end[:16]})."
            conf = "high"
            lim = "Indicates unreliable data, not energy waste."
        else:
            finding, expl, action = ACTIONS.get(kind, ("Unusual consumption pattern.", "Cause not determined.", "Investigate."))
            ev = (f"{n} event(s). Largest: {top.observed:,.1f} {top.unit} observed vs {top.expected:,.1f} {top.unit} expected "
                  f"({top.deviation_pct:+.1f}%) for {top.start}" + (f" to {top.end}" if top.end != top.start else "") + ".")
            conf = "medium" if n >= 2 or top.severity == "high" else "low"
            lim = "Meter data cannot confirm the cause; baseline is a statistical estimate."
            if top.warnings:
                conf, lim = "low", lim + " " + top.warnings[0]
        recs.append(Recommendation(b, finding, ev, expl, action, conf, lim, "data_quality" if top.category == "data_quality" else "anomaly"))
    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: (r.source == "data_quality", order[r.confidence]))
    return recs[:max_items]


def from_profile(profile: pd.DataFrame, night_day_threshold: float = 0.6) -> list[Recommendation]:
    """profile columns: building_id, primary_use, night_avg_kwh, day_avg_kwh (weekday 09-17)."""
    recs = []
    if profile is None or profile.empty:
        return recs
    for r in profile.itertuples():
        if r.day_avg_kwh and r.night_avg_kwh is not None and r.day_avg_kwh > 0:
            ratio = r.night_avg_kwh / r.day_avg_kwh
            if ratio >= night_day_threshold and str(r.primary_use).lower() in ("office", "education"):
                recs.append(Recommendation(
                    r.building_id, "High night-time base load relative to daytime load.",
                    f"Night (00-05h) average {r.night_avg_kwh:,.1f} kWh/h is {ratio:.0%} of the weekday daytime (09-17h) "
                    f"average {r.day_avg_kwh:,.1f} kWh/h.",
                    "Hypothesis: equipment, ventilation or lighting left on outside occupied hours; some loads (servers, labs) may be legitimate.",
                    "Audit after-hours schedules and always-on equipment.", "medium",
                    "Ratio threshold is a heuristic; occupancy schedules are not in the dataset.", "profile"))
    return recs


def from_forecast(fc) -> list[Recommendation]:
    recs = []
    s = getattr(fc, "summary", {}) or {}
    if s.get("deviation_pct") is not None and abs(s["deviation_pct"]) >= 10 and fc.status == "ok":
        recs.append(Recommendation(
            fc.target, "Actual consumption differed from the expected baseline.",
            f"Actual {s['actual_total_kwh']:,.1f} kWh vs expected {s['expected_total_kwh']:,.1f} kWh "
            f"({s['deviation_pct']:+.1f}%); {s['hours_above_interval']} hour(s) above the 80% interval.",
            "Hypothesis: operational change or unusual conditions during the period.",
            "Review the hours outside the interval for schedule or equipment changes.",
            "medium" if not fc.warnings else "low", "Expected values are model estimates with uncertainty.", "forecast"))
    if s.get("potential_anomaly"):
        recs.append(Recommendation(fc.target, "Forecast suggests unusual upcoming consumption.", s["potential_anomaly"],
                                   "Projection from historical patterns only; not a measured event.",
                                   "Monitor actual readings against the forecast interval.", "low",
                                   "Future weather and occupancy are unknown.", "forecast"))
    return recs


def to_frame(recs: list[Recommendation]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in recs])

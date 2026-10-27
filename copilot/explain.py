"""ExplanationChecker (evidence-only natural-language explanation) and a deterministic sanity checker."""
from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pandas as pd

SYSTEM_PROMPT = (
    "You are an energy analyst for facility managers. Answer the question directly in 2-5 short sentences "
    "using ONLY the evidence provided. Cite building names, numbers, units (kWh), percentages and time periods "
    "exactly as given; you may round to 1 decimal. Never invent numbers, buildings, dates or causes. "
    "Causes are hypotheses: say 'may' or 'possible'. If data is missing, truncated, flagged or uncertain, say so. "
    "If the evidence cannot answer the question, say that plainly. Do not mention SQL. Plain text only."
)

_NUM_RE = re.compile(r"(?<![\w.])([-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)\s*(%|percent|gwh|mwh|kwh|kw|mw|wh)?(?![\w])",
                     re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})(?:-(\d{2}))?(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_BID_RE = re.compile(r"\b[A-Z][a-z]+_[a-z]+_[A-Z][a-z]+\b")
_SCALE = {"gwh": 1e6, "mwh": 1e3}


def _to_frame(data) -> pd.DataFrame:
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    if isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except ValueError:
            return pd.DataFrame([data])
    return pd.DataFrame(data)


def _numbers_in(text: str) -> list[float]:
    text = _DATE_RE.sub(" ", text)
    return [float(m.group(1).replace(",", "")) for m in _NUM_RE.finditer(text)]


class ExplanationChecker:
    def __init__(self, llm=None, unit: str = "kWh", max_rows: int = 20, building_ids=None, date_range=None):
        self.llm, self.unit, self.max_rows = llm, unit, max_rows
        self.sanity = SanityChecker(building_ids or [], unit, date_range)

    def format_context(self, user_prompt: str, data_results, analytics_summary=None) -> str:
        df = _to_frame(data_results)
        parts = [f"QUESTION: {user_prompt.strip()}", f"UNIT: {self.unit} (hourly readings; NULL = missing)"]
        if df.empty:
            parts.append("RESULTS: no rows")
        else:
            shown = df.head(self.max_rows)
            parts.append(f"RESULTS ({len(df)} rows{', first ' + str(len(shown)) + ' shown' if len(df) > len(shown) else ''}):")
            parts.append(shown.to_csv(index=False, float_format="%.2f").strip())
            summ = []
            for c in df.select_dtypes("number").columns:
                s = df[c]
                summ.append(f"{c}: min={s.min():.2f} max={s.max():.2f} mean={s.mean():.2f} sum={s.sum():.2f} nulls={int(s.isna().sum())}")
            if summ and len(df) > 1:
                parts.append("COLUMN SUMMARY: " + "; ".join(summ))
        if analytics_summary:
            parts.append("ANALYTICS: " + json.dumps(analytics_summary, default=str, separators=(",", ":"))[:3000])
        return "\n".join(parts)

    def template_explanation(self, user_prompt: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "No rows matched this question in the available data (2016-2017)."
        num = list(df.select_dtypes("number").columns)
        cat = [c for c in df.columns if c not in num]
        if len(df) == 1:
            vals = ", ".join(f"{c} = {df.iloc[0][c]:,.2f}" if c in num else f"{c} = {df.iloc[0][c]}" for c in df.columns)
            return f"Result: {vals}."
        if num and cat:
            y, x = num[0], cat[0]
            top = df.loc[df[y].idxmax()]
            low = df.loc[df[y].idxmin()]
            return (f"{len(df)} rows returned. Highest {y}: {top[x]} ({top[y]:,.2f}); lowest: {low[x]} ({low[y]:,.2f}). "
                    f"Total {y}: {df[y].sum():,.2f}.")
        return f"{len(df)} rows returned."

    def template_analytics(self, s: dict) -> str:
        parts = []
        f = s.get("forecast")
        if f:
            if f.get("status") == "insufficient_history":
                parts.append(f"No forecast for {f['target']}: not enough history.")
            else:
                lo, hi = f.get("interval_total_kwh", [None, None])
                parts.append(f"{f['target']}: expected {f.get('expected_total_kwh'):,.1f} {self.unit} from {f['period'][0]} to "
                             f"{f['period'][1]} (80% interval {lo:,.1f} to {hi:,.1f} {self.unit}; model {f.get('model')}).")
                if f.get("actual_total_kwh") is not None:
                    parts.append(f"Actual was {f['actual_total_kwh']:,.1f} {self.unit} ({f.get('deviation_pct'):+.1f}% vs expected).")
        a = s.get("anomalies")
        if a:
            parts.append(f"Between {a['window'][0][:10]} and {a['window'][1][:10]}: {a['anomaly_count']} consumption anomalies and "
                         f"{a['data_quality_count']} data-quality issues.")
            if a["top"]:
                t = a["top"][0]
                dev = f" ({t['deviation_pct']:+.1f}% vs expected)" if t.get("deviation_pct") is not None else ""
                parts.append(f"Most significant: {t['building_id']} {t['kind'].replace('_', ' ')} on {t['start']}{dev}.")
        if s.get("recommendations"):
            parts.append(f"{len(s['recommendations'])} recommendation(s) below; causes are hypotheses to verify on site.")
        return " ".join(parts) or "No analytics results."

    def explain(self, user_prompt: str, data_results, analytics_summary=None, use_llm: bool = True) -> dict:
        df = _to_frame(data_results)
        context = self.format_context(user_prompt, df, analytics_summary)
        out = {"user_prompt": user_prompt, "data_results": data_results, "analytics_summary": analytics_summary,
               "source": "template", "error": None}
        text = None
        if use_llm and (not df.empty or analytics_summary) and self.llm is not None and getattr(self.llm, "available", False):
            r = self.llm.chat(SYSTEM_PROMPT, context, max_tokens=350)
            if r.ok and r.text:
                text, out["source"] = r.text, "llm"
            else:
                out["error"] = r.error
        if text is None:
            text = (self.template_analytics(analytics_summary) if df.empty and analytics_summary
                    else self.template_explanation(user_prompt, df))
        out["explanation"] = text
        out["verification"] = self.sanity.check(text, context, df, analytics_summary)
        return out


class SanityChecker:
    """Checks numbers, percentages, units, building ids, dates and simple rankings against the evidence."""

    def __init__(self, building_ids, unit: str = "kWh", date_range=None, max_pair_values: int = 80):
        self.building_ids, self.unit, self.date_range, self.max_pair = set(building_ids), unit, date_range, max_pair_values

    @staticmethod
    def _close(x: float, a: float, decimals: int) -> bool:
        tol = max(0.5 * 10 ** (-decimals) + 1e-9, abs(a) * 0.005)
        return abs(x - a) <= tol

    def check(self, text: str, context: str, df: pd.DataFrame | None = None, analytics=None) -> dict:
        issues: list[str] = []
        evidence = _numbers_in(context)
        if df is not None and not df.empty:
            for c in df.select_dtypes("number").columns:
                s = df[c].dropna()
                evidence += [float(s.sum()), float(s.mean()), float(s.max()), float(s.min())] if len(s) else []
                evidence += s.astype(float).tolist()[:500]
            evidence.append(float(len(df)))
        evidence = list({round(v, 6) for v in evidence if np.isfinite(v)})
        pool = sorted(evidence, key=abs, reverse=True)[: self.max_pair]
        derived, diffs = set(), set()
        for a, b in itertools.permutations(pool, 2):
            diffs.add(abs(a - b))
            if b:
                derived.add((a - b) / abs(b) * 100)
                derived.add(a / b * 100)
        clean = _TIME_RE.sub(" ", _DATE_RE.sub(" ", text))
        checked, bad = 0, []
        for m in _NUM_RE.finditer(clean):
            raw, unit = m.group(1), (m.group(2) or "").lower()
            val = float(raw.replace(",", ""))
            dec = len(raw.split(".")[1]) if "." in raw else 0
            if unit in ("kw", "mw"):
                issues.append(f"Unit '{m.group(2)}' is a power unit; the data is energy in {self.unit}.")
            if not unit and dec == 0 and abs(val) <= 24:
                continue  # small counts / hours: not checked
            checked += 1
            v = val * _SCALE.get(unit, 1)
            ok = any(self._close(v, a, dec) or self._close(abs(v), abs(a), dec) for a in evidence)
            if not ok and unit not in ("%", "percent"):
                ok = any(self._close(abs(v), d, dec) for d in diffs)
            if not ok and unit in ("%", "percent"):
                ok = any(abs(abs(val) - abs(d)) <= max(0.5 * 10 ** (-dec), 0.05) for d in derived)
            if not ok:
                bad.append(raw + (f" {m.group(2)}" if m.group(2) else ""))
        if bad:
            issues.append("Numbers not found in the evidence: " + ", ".join(bad[:8]))
        for bid in set(_BID_RE.findall(text)):
            if self.building_ids and bid not in self.building_ids:
                issues.append(f"Unknown building '{bid}'.")
            elif bid not in context:
                issues.append(f"Building '{bid}' is not in the results.")
        for m in _DATE_RE.finditer(text):
            if self.date_range and m.group(0)[:10] not in context:
                lo, hi = self.date_range
                if not (lo[: len(m.group(0)[:10])] <= m.group(0)[:10] <= hi[: len(m.group(0)[:10])]):
                    issues.append(f"Date {m.group(0)} is outside the available data.")
        if re.search(r"\b(therms?|btu|gallons?)\b|\$\s?\d", text, re.IGNORECASE) and not re.search(r"\b(therm|btu|gallon|\$)", context, re.I):
            issues.append("Mentions units or costs that are not in the data.")
        issues += self._ranking(text, df)
        return {"verified": not issues, "issues": issues, "numbers_checked": checked}

    @staticmethod
    def _ranking(text: str, df: pd.DataFrame | None) -> list[str]:
        if df is None or df.empty or "building_id" not in df.columns:
            return []
        num = [c for c in df.select_dtypes("number").columns]
        if not num or df["building_id"].duplicated().any():
            return []
        y = next((c for c in num if "kwh" in c.lower()), num[0])
        top, bottom = df.loc[df[y].idxmax(), "building_id"], df.loc[df[y].idxmin(), "building_id"]
        out = []
        for m in re.finditer(r"(\b[A-Z][a-z]+_[a-z]+_[A-Z][a-z]+\b)[^.]{0,60}?\b(highest|largest|most|lowest|least|smallest)\b", text):
            bid, word = m.groups()
            expect = top if word in ("highest", "largest", "most") else bottom
            if bid != expect:
                out.append(f"Ranking claim '{bid} ... {word}' conflicts with the data (expected {expect}).")
        return out

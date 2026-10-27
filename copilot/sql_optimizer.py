"""Measured, equivalence-checked SQL rewrites applied only after validation.

Governing rule: same rows, order and values (except the reported row cap). A rewrite is kept only if
  1. the rewritten SQL passes SQLValidator again,
  2. results are equivalent to the current SQL (ordered when the root has ORDER BY; NULLs compared as NULL),
  3. EXPLAIN QUERY PLAN improves where the rule targets index use, and
  4. the median of repeated timings improves by at least MIN_SPEEDUP.
Otherwise the original SQL is retained and the reason is recorded.
"""
from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

from .sql_validator import DIALECT

MIN_SPEEDUP = 1.15
BENCH_REPS = 5

_DATE_FUNCS = {
    r"DATE\((?:\w+\.)?ts\)": "day",
    r"STRFTIME\('%Y-%m-%d', (?:\w+\.)?ts\)": "day",
    r"STRFTIME\('%Y-%m', (?:\w+\.)?ts\)": "month",
    r"STRFTIME\('%Y', (?:\w+\.)?ts\)": "year",
}


def _period(kind: str, text: str) -> tuple[str, str] | None:
    try:
        if kind == "day":
            d = date.fromisoformat(text)
            return f"{d} 00:00:00", f"{d + timedelta(days=1)} 00:00:00"
        if kind == "month":
            y, m = map(int, re.fullmatch(r"(\d{4})-(\d{2})", text).groups())
            s = date(y, m, 1)
            e = date(y + (m == 12), m % 12 + 1, 1)
            return f"{s} 00:00:00", f"{e} 00:00:00"
        if kind == "year":
            y = int(re.fullmatch(r"\d{4}", text).group())
            return f"{y}-01-01 00:00:00", f"{y + 1}-01-01 00:00:00"
    except Exception:
        return None
    return None


def _date_kind(e: exp.Expression) -> tuple[str, exp.Column] | None:
    s = e.sql(dialect=DIALECT)
    for pat, kind in _DATE_FUNCS.items():
        if re.fullmatch(pat, s):
            return kind, next(e.find_all(exp.Column))
    return None


def rule_date_range(tree: exp.Expression, ctx) -> bool:
    """f(ts) = 'literal'  ->  ts >= start AND ts < end (requires fixed-format text timestamps)."""
    if not ctx.schema.ts_fixed_format:
        return False
    changed = False
    for node in list(tree.find_all(exp.EQ, exp.Between)):
        if isinstance(node, exp.EQ):
            side, lit = (node.left, node.right) if isinstance(node.right, exp.Literal) else (node.right, node.left)
            k = _date_kind(side) if isinstance(lit, exp.Literal) and lit.is_string else None
            p = _period(k[0], lit.this) if k else None
            if not p:
                continue
            lo, hi = p
        else:
            k = _date_kind(node.this)
            low, high = node.args.get("low"), node.args.get("high")
            if not (k and k[0] == "day" and all(isinstance(x, exp.Literal) and x.is_string for x in (low, high))):
                continue
            p1, p2 = _period("day", low.this), _period("day", high.this)
            if not (p1 and p2):
                continue
            lo, hi = p1[0], p2[1]
        col = k[1]
        new = exp.Paren(this=exp.and_(exp.GTE(this=col.copy(), expression=exp.Literal.string(lo)),
                                     exp.LT(this=col.copy(), expression=exp.Literal.string(hi))))
        node.replace(new)
        changed = True
    return changed


def rule_like_to_in(tree: exp.Expression, ctx) -> bool:
    """building_id LIKE '%x%' -> building_id IN (...) using the closed building set (FK verified)."""
    if not ctx.closed_building_set:
        return False
    changed = False
    for node in list(tree.find_all(exp.Like)):
        col, pat = node.this, node.expression
        if not (isinstance(col, exp.Column) and col.name.lower() == "building_id"
                and isinstance(pat, exp.Literal) and pat.is_string and pat.this.startswith("%")):
            continue
        if node.args.get("escape"):
            continue
        rx = re.compile("^" + "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pat.this) + "$",
                        re.IGNORECASE | re.DOTALL)  # SQLite LIKE: ASCII case-insensitive
        ids = sorted(b for b in ctx.building_ids if rx.match(b))
        if not ids:
            continue
        node.replace(exp.In(this=col.copy(), expressions=[exp.Literal.string(b) for b in ids]))
        changed = True
    return changed


def rule_expand_star(tree: exp.Expression, ctx) -> bool:
    """SELECT * FROM single_table -> SELECT named columns (same order as the table definition)."""
    if not isinstance(tree, exp.Select) or tree.args.get("joins") or len(tree.expressions) != 1:
        return False
    if not isinstance(tree.expressions[0], exp.Star):
        return False
    frm = tree.args.get("from") or tree.args.get("from_")
    t = frm.this if frm is not None else None
    if not isinstance(t, exp.Table) or t.name.lower() not in ctx.schema.tables:
        return False
    tree.set("expressions", [exp.column(c) for c in ctx.schema.tables[t.name.lower()]])
    return True


RULES = [  # (name, function, plan_relevant)
    ("date_function_to_range", rule_date_range, True),
    ("like_to_in", rule_like_to_in, True),
    ("expand_select_star", rule_expand_star, False),
]


def frames_equivalent(a: pd.DataFrame, b: pd.DataFrame, ordered: bool) -> tuple[bool, str]:
    if list(a.columns) != list(b.columns):
        return False, "column names/order differ"
    if a.shape != b.shape:
        return False, f"row count differs ({len(a)} vs {len(b)})"
    if not ordered:
        a = a.sort_values(list(a.columns), na_position="first", kind="mergesort").reset_index(drop=True)
        b = b.sort_values(list(b.columns), na_position="first", kind="mergesort").reset_index(drop=True)
    for c in a.columns:
        x, y = a[c], b[c]
        if not x.isna().equals(y.isna()):
            return False, f"NULL pattern differs in {c}"
        xm, ym = x[x.notna()], y[y.notna()]
        if pd.api.types.is_numeric_dtype(xm) and pd.api.types.is_numeric_dtype(ym):
            if not np.allclose(xm.to_numpy(float), ym.to_numpy(float), rtol=1e-9, atol=1e-9):
                return False, f"values differ in {c}"
        elif not (xm.astype(str).to_numpy() == ym.astype(str).to_numpy()).all():
            return False, f"values differ in {c}"
    return True, "equivalent"


def _plan_score(plan: list[str]) -> int:
    """Lower is better: full scans cost most, then index searches without an equality seek, then temp b-trees."""
    score = 0
    for p in plan:
        if p.startswith("SCAN") and "INDEX" not in p:
            score += 10
        elif p.startswith(("SCAN", "SEARCH")) and "=?" not in p:
            score += 2
        elif "TEMP B-TREE" in p:
            score += 1
    return score


@dataclass
class OptimizationResult:
    sql: str
    applied: list[str] = field(default_factory=list)
    report: list[dict] = field(default_factory=list)


class SQLOptimizer:
    def __init__(self, executor, min_speedup: float = MIN_SPEEDUP, reps: int = BENCH_REPS):
        self.ex, self.min_speedup, self.reps = executor, min_speedup, reps
        self.schema = executor.schema
        self.building_ids = list(self.schema.buildings["building_id"])
        orphan = executor.run("SELECT COUNT(*) AS n FROM readings WHERE building_id NOT IN (SELECT building_id FROM buildings)")
        self.closed_building_set = bool(orphan.ok and int(orphan.df.iloc[0, 0]) == 0)
        self._decisions: dict[str, OptimizationResult] = {}

    def _time(self, sql: str) -> float:
        t0 = time.perf_counter()
        self.ex._raw(sql, self.ex.validator.max_rows)
        return time.perf_counter() - t0

    def evaluate(self, current: str, candidate: str, plan_relevant: bool) -> dict:
        v = self.ex.validator.validate(candidate)
        if not v.ok:
            return {"keep": False, "reason": f"rewrite failed validation: {v.reason_code}"}
        candidate = v.clean_sql
        a, b = self.ex.run_validated(self.ex.validator.validate(current)), self.ex.run_validated(v)
        if not (a.ok and b.ok):
            return {"keep": False, "reason": "execution failed during equivalence check"}
        ordered = sqlglot.parse_one(current, read=DIALECT).args.get("order") is not None
        eq, why = frames_equivalent(a.df, b.df, ordered)
        if not eq:
            return {"keep": False, "reason": f"not equivalent: {why}"}
        p0, p1 = self.ex.explain(current), self.ex.explain(candidate)
        improved = _plan_score(p1) < _plan_score(p0)
        t_old, t_new = [], []
        for _ in range(self.reps):  # interleave to reduce drift
            t_old.append(self._time(current))
            t_new.append(self._time(candidate))
        m_old, m_new = statistics.median(t_old), statistics.median(t_new)
        speedup = m_old / m_new if m_new > 0 else float("inf")
        info = {"plan_before": p0, "plan_after": p1, "plan_improved": improved, "median_ms_before": round(m_old * 1000, 2),
                "median_ms_after": round(m_new * 1000, 2), "speedup": round(speedup, 2), "sql": candidate}
        if plan_relevant and not improved:
            return {**info, "keep": False, "reason": "query plan did not improve"}
        if speedup < self.min_speedup:
            return {**info, "keep": False, "reason": f"speedup {speedup:.2f}x below {self.min_speedup}x"}
        return {**info, "keep": True, "reason": "equivalent, faster"}

    def optimize(self, clean_sql: str) -> OptimizationResult:
        if clean_sql in self._decisions:
            return self._decisions[clean_sql]
        res = OptimizationResult(clean_sql)
        for name, fn, plan_relevant in RULES:
            try:
                tree = sqlglot.parse_one(res.sql, read=DIALECT)
                if not fn(tree, self):
                    continue
                cand = tree.sql(dialect=DIALECT)
                d = self.evaluate(res.sql, cand, plan_relevant)
            except Exception as e:  # a failing rewrite never blocks the original query
                d = {"keep": False, "reason": f"rewrite error: {type(e).__name__}"}
            res.report.append({"rule": name, **d})
            if d.get("keep"):
                res.sql, res.applied = d["sql"], res.applied + [name]
        self._decisions[clean_sql] = res
        return res

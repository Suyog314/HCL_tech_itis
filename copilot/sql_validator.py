"""SQLValidator: deterministic, pure, fail-closed security gate for generated SQL.

    validate(sql, intent=None) -> Verdict(ok, clean_sql, reason_code, reason_text, notes)

* Uses one parser (sqlglot, SQLite dialect); no regex validation, no LLM, no network, no retries,
  no caching and no timeout of its own.
* Never raises: any exception becomes a rejected verdict. ``clean_sql`` is None whenever ok is False.
* Checks run in a fixed order and stop at the first failure (see ``validate``).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import sqlglot
from sqlglot import exp

from .sql_errors import describe

# ---- configuration -------------------------------------------------------------------------------
DIALECT = "sqlite"
MAX_SQL_CHARS = 8000
MAX_AST_NODES = 2500
MAX_SELECT_DEPTH = 5
MAX_JOINS = 6
DEFAULT_MAX_ROWS = 5000
TS_COLUMNS = {"ts"}

_FORBIDDEN_NODE_NAMES = (
    "Insert", "Update", "Delete", "Create", "Drop", "Alter", "AlterTable", "Command", "Pragma", "Attach",
    "Detach", "Merge", "Into", "TruncateTable", "Transaction", "Commit", "Rollback", "Set", "Use", "Copy",
    "LoadData", "Grant", "Analyze", "Kill", "Cache", "Uncache", "Refresh", "Describe", "Show", "Lock",
)
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))
ROOT_TYPES = (exp.Select, exp.Union, exp.Intersect, exp.Except)

# Allowlisted functions: sqlglot canonical names (typed nodes) and raw names (Anonymous nodes).
ALLOWED_FUNCS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "ABS", "COALESCE", "CASE", "IF", "CAST", "NULLIF",
    "TIME_TO_STR", "STR_TO_TIME", "DATE", "TS_OR_DS_TO_TIMESTAMP", "TS_OR_DS_TO_DATE", "TIME_STR_TO_TIME",
    "SUBSTRING", "LOWER", "UPPER", "LENGTH", "TRIM", "REPLACE", "GROUP_CONCAT", "STR_POSITION",
    "ROW_NUMBER", "RANK", "DENSE_RANK", "LAG", "LEAD", "NTILE", "FIRST_VALUE", "LAST_VALUE", "NTH_VALUE",
    "PERCENT_RANK", "CUME_DIST", "GREATEST", "LEAST", "CURRENT_DATE", "DATETIME", "TIME", "SIGN", "CONCAT",
    "JULIANDAY", "TOTAL", "STRFTIME", "IIF", "INSTR", "PRINTF", "UNIXEPOCH", "IFNULL",
}
FORBIDDEN_FUNCS = {"LOAD_EXTENSION", "READFILE", "WRITEFILE", "EDIT", "FTS3_TOKENIZER", "RANDOMBLOB",
                   "ZEROBLOB", "SQLITE_VERSION", "SQLITE_SOURCE_ID", "SQLITE_COMPILEOPTION_USED",
                   "SQLITE_COMPILEOPTION_GET", "SQLITE_OFFSET", "LIKELIHOOD"}
_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?)?)?$")


@dataclass(frozen=True)
class Verdict:
    ok: bool
    clean_sql: str | None
    reason_code: str | None = None
    reason_text: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


class _Reject(Exception):
    def __init__(self, code: str, detail: str | None = None):
        self.code, self.detail = code, detail


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).upper()
    return node.sql_name().upper()


def _parse_period(text: str) -> tuple[datetime, datetime] | None:
    """Return the [start, end) period a date literal denotes, or None if it is not a date literal."""
    m = _DATE_RE.match(text.strip())
    if not m:
        return None
    y, mo, d, hh, mi, ss = m.groups()
    try:
        if mo is None:
            return datetime(int(y), 1, 1), datetime(int(y) + 1, 1, 1)
        if d is None:
            s = datetime(int(y), int(mo), 1)
            return s, (datetime(s.year + 1, 1, 1) if s.month == 12 else datetime(s.year, s.month + 1, 1))
        if hh is None:
            s = datetime(int(y), int(mo), int(d))
            return s, s + timedelta(days=1)
        s = datetime(int(y), int(mo), int(d), int(hh), int(mi), int(ss or 0))
        return s, s + timedelta(seconds=1)
    except ValueError:
        return None


class SQLValidator:
    def __init__(self, allowed: dict[str, list[str]], min_ts: str | None = None, max_ts: str | None = None,
                 max_rows: int = DEFAULT_MAX_ROWS):
        self.allowed = {t.lower(): {c.lower() for c in cols} for t, cols in allowed.items()}
        self.all_columns = set().union(*self.allowed.values()) if self.allowed else set()
        self.max_rows = int(max_rows)
        self.range = None
        if min_ts and max_ts:
            lo = _parse_period(min_ts)[0]
            hi = _parse_period(max_ts)[0]
            # tolerate a half-open bound up to the end of the last day
            self.range = (lo.replace(hour=0, minute=0, second=0), hi.replace(hour=0, minute=0, second=0) + timedelta(days=1))

    @classmethod
    def from_schema(cls, schema, max_rows: int = DEFAULT_MAX_ROWS) -> "SQLValidator":
        return cls(schema.tables, schema.min_ts, schema.max_ts, max_rows)

    # ------------------------------------------------------------------------------------------
    def validate(self, sql, intent=None) -> Verdict:
        notes: list[str] = []
        try:
            clean = self._validate(sql, intent, notes)
            return Verdict(True, clean, None, None, tuple(notes))
        except _Reject as r:
            return Verdict(False, None, r.code, describe(r.code, r.detail), tuple(notes))
        except Exception as e:  # fail closed on anything unexpected
            return Verdict(False, None, "PARSE_ERROR", describe("PARSE_ERROR", f"validator error: {type(e).__name__}"),
                           tuple(notes))

    def _validate(self, sql, intent, notes) -> str:
        # 1. non-empty and size limits
        if not isinstance(sql, str) or not sql.strip():
            raise _Reject("PARSE_ERROR", "empty query")
        if len(sql) > MAX_SQL_CHARS:
            raise _Reject("TOO_COMPLEX", f"longer than {MAX_SQL_CHARS} characters")
        # 2. parse with a real parser
        try:
            statements = [s for s in sqlglot.parse(sql, read=DIALECT) if s is not None]
        except Exception as e:
            raise _Reject("PARSE_ERROR", type(e).__name__)
        if not statements:
            raise _Reject("PARSE_ERROR", "no statement")
        # 3. exactly one statement
        if len(statements) != 1:
            raise _Reject("MULTI_STATEMENT", f"{len(statements)} statements")
        tree = statements[0]
        n_nodes = sum(1 for _ in tree.walk())
        if n_nodes > MAX_AST_NODES:
            raise _Reject("TOO_COMPLEX", f"{n_nodes} syntax nodes")
        # 4. root operation must be a read-only query
        if not isinstance(tree, ROOT_TYPES):
            raise _Reject("NOT_SELECT", type(tree).__name__.upper())
        # 5. deep inspection for forbidden nodes (including inside CTEs / subqueries)
        for node in tree.walk():
            if isinstance(node, FORBIDDEN_NODES):
                raise _Reject("FORBIDDEN_STATEMENT", type(node).__name__.upper())
        if any(w.args.get("recursive") for w in tree.find_all(exp.With)):
            raise _Reject("TOO_COMPLEX", "recursive CTE")
        depth = max((self._select_depth(s) for s in tree.find_all(exp.Select)), default=1)
        if depth > MAX_SELECT_DEPTH:
            raise _Reject("TOO_COMPLEX", f"nesting depth {depth}")
        if len(list(tree.find_all(exp.Join))) > MAX_JOINS:
            raise _Reject("TOO_COMPLEX", "too many joins")
        # 6. table allowlist
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
        aliases, real_tables = set(), set()
        for t in tree.find_all(exp.Table):
            if not isinstance(t.this, exp.Identifier):
                raise _Reject("TABLE_NOT_ALLOWED", "table-valued function")
            if t.args.get("db") or t.args.get("catalog"):
                raise _Reject("TABLE_NOT_ALLOWED", f"qualified name {t.sql(dialect=DIALECT)}")
            name = t.name.lower()
            if name in cte_names:
                pass
            elif name in self.allowed:
                real_tables.add(name)
            else:
                raise _Reject("TABLE_NOT_ALLOWED", name)
            if t.alias:
                aliases.add(t.alias.lower())
        # 7. column allowlist
        defined = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
        for ta in tree.find_all(exp.TableAlias):
            if ta.name:
                aliases.add(ta.name.lower())
            defined |= {c.name.lower() for c in ta.columns}
        qualifiers = set(self.allowed) | cte_names | aliases
        for col in tree.find_all(exp.Column):
            if isinstance(col.this, exp.Star):
                continue
            name = col.name.lower()
            if col.table and col.table.lower() not in qualifiers:
                raise _Reject("COLUMN_NOT_ALLOWED", f"unknown qualifier {col.table}")
            if name not in self.all_columns and name not in defined:
                raise _Reject("COLUMN_NOT_ALLOWED", name)
        # 8. must read an authorised source
        if not real_tables:
            raise _Reject("NO_DATA_SOURCE")
        # 9. function allowlist
        for f in tree.find_all(exp.Func):
            if isinstance(f, (exp.Connector, exp.Binary, exp.Unary)):  # AND/OR/operators, not calls
                continue
            fname = _func_name(f)
            if fname in FORBIDDEN_FUNCS or fname.startswith(("SQLITE_", "PRAGMA_")) or fname not in ALLOWED_FUNCS:
                raise _Reject("FORBIDDEN_FUNCTION", fname.lower())
        # 10. time predicates inside the authorised range
        if self.range:
            self._check_dates(tree)
        # 11. row cap: add if missing, only ever lower an existing one
        cap = self.max_rows
        if isinstance(intent, dict) and isinstance(intent.get("max_rows"), int) and 0 < intent["max_rows"] < cap:
            cap = intent["max_rows"]
        lim = tree.args.get("limit")
        if lim is None:
            tree.set("limit", exp.Limit(expression=exp.Literal.number(cap)))
            notes.append(f"row_cap: added LIMIT {cap}")
        else:
            val = lim.expression
            if not (isinstance(val, exp.Literal) and val.is_int):
                raise _Reject("TOO_COMPLEX", "non-literal LIMIT")
            if int(val.this) > cap:
                lim.set("expression", exp.Literal.number(cap))
                notes.append(f"row_cap: lowered LIMIT {val.this} -> {cap}")
            else:
                notes.append(f"row_cap: kept LIMIT {val.this}")
        return tree.sql(dialect=DIALECT)

    @staticmethod
    def _select_depth(node: exp.Expression) -> int:
        d, p = 1, node.parent
        while p is not None:
            if isinstance(p, exp.Select):
                d += 1
            p = p.parent
        return d

    def _check_dates(self, tree: exp.Expression) -> None:
        lo, hi = self.range

        def is_ts_side(e: exp.Expression) -> bool:
            cols = list(e.find_all(exp.Column))
            return bool(cols) and all(c.name.lower() in TS_COLUMNS for c in cols)

        def check(lit: exp.Expression) -> None:
            if isinstance(lit, exp.Literal) and lit.is_string:
                p = _parse_period(lit.this)
                if p and (p[1] <= lo or p[0] > hi):
                    raise _Reject("OUT_OF_RANGE", f"{lit.this} is outside {lo:%Y-%m-%d} to {hi - timedelta(days=1):%Y-%m-%d}")

        for node in tree.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between, exp.In):
            if isinstance(node, exp.Between):
                if is_ts_side(node.this):
                    check(node.args.get("low"))
                    check(node.args.get("high"))
            elif isinstance(node, exp.In):
                if is_ts_side(node.this):
                    for e in node.expressions:
                        check(e)
            else:
                left, right = node.left, node.right
                if is_ts_side(left):
                    check(right)
                elif is_ts_side(right):
                    check(left)

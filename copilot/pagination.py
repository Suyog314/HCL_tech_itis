"""Server-side pagination of a validated query: rewrites the root LIMIT/OFFSET, never the inner query."""
from __future__ import annotations

import sqlglot
from sqlglot import exp

from .sql_validator import DIALECT


def page_sql(clean_sql: str, page: int, page_size: int) -> str:
    """Return SQL fetching rows [page*page_size, (page+1)*page_size] (+1 look-ahead row for has_next).

    The validated query always has a root LIMIT (the row cap); the page never reads beyond it, and the
    original ORDER BY is preserved so pages are stable.
    """
    tree = sqlglot.parse_one(clean_sql, read=DIALECT)
    lim = tree.args.get("limit")
    cap = int(lim.expression.this) if lim is not None else page_size + 1
    off_node = tree.args.get("offset")
    base_off = int(off_node.expression.this) if off_node is not None else 0
    start = max(0, page) * page_size
    remaining = max(0, cap - start)
    tree.set("limit", exp.Limit(expression=exp.Literal.number(min(page_size + 1, remaining))))
    tree.set("offset", exp.Offset(expression=exp.Literal.number(base_off + start)))
    return tree.sql(dialect=DIALECT)

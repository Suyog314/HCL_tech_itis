"""Stable error codes -> human-readable explanations (kept separate from validation logic)."""

MESSAGES = {
    "PARSE_ERROR": "The query could not be parsed as valid SQL.",
    "TOO_COMPLEX": "The query is too long or too complex to run safely.",
    "MULTI_STATEMENT": "Only a single SQL statement is allowed.",
    "NOT_SELECT": "Only read-only SELECT queries are allowed.",
    "FORBIDDEN_STATEMENT": "The query contains a write, schema-change or administrative operation.",
    "TABLE_NOT_ALLOWED": "The query references a table that is not available.",
    "COLUMN_NOT_ALLOWED": "The query references a column that does not exist in the allowed tables.",
    "NO_DATA_SOURCE": "The query does not read from the energy database.",
    "FORBIDDEN_FUNCTION": "The query uses a function that is not permitted.",
    "OUT_OF_RANGE": "The requested time range is outside the available data.",
    "TIMEOUT": "The query took too long and was stopped.",
    "EXEC_ERROR": "The database could not run this query.",
}


def describe(code: str | None, detail: str | None = None) -> str:
    base = MESSAGES.get(code or "", "The query was rejected.")
    return f"{base} ({detail})" if detail else base

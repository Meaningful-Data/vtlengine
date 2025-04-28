import os

DUCKDB_TOKEN = "duckdb"

BACKENDS = {
    # eager execution
    "pandas": "pd",
    "pd": "pd",
    "eager": "pd",
    # lazy execution
    "duckdb": DUCKDB_TOKEN,
    "db": DUCKDB_TOKEN,
    "lazy": DUCKDB_TOKEN,
    "streaming": DUCKDB_TOKEN,
}

LAZY_STR = ["duckdb", "db", "lazy", "streaming"]

backend_df = BACKENDS.get(os.getenv("BACKEND_DF", "").lower(), "pd")

con = None
if backend_df == DUCKDB_TOKEN:
    import duckdb
    con = duckdb.connect(database=":memory:", read_only=False)

__all__ = ["con"]
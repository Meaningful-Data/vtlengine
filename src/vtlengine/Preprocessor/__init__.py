import os

BACKENDS = {
    # eager execution
    "pandas": "pd",
    "pd": "pd",
    "eager": "pd",
    # lazy execution
    "duckdb": "duckdb",
    "db": "duckdb",
    "lazy": "duckdb",
    "streaming": "duckdb",
}

LAZY_STR = ["duckdb", "db", "lazy", "streaming"]

backend_df = BACKENDS.get(os.getenv("BACKEND_DF", "").lower(), "pd")
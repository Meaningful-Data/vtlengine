import os

import pandas as pd
from pandas._testing import assert_frame_equal as pandas_assert_frame_equal
from vtlengine.Preprocessor.DuckDB import (
    LazyFrame,
    LazySeries,
    _assert_frame_equal as lazy_assert_frame_equal,
    _concat as lazy_concat,
    _infer_dtype as lazy_infer_dtype,
    _isnull as lazy_isnull,
    _isna as lazy_isna,
    _merge as lazy_merge,
    _read_csv as lazy_read_csv,
    _to_datetime as lazy_to_datetime,
)

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

if backend_df == "pd":
    _DataFrame = pd.DataFrame
    _Series = pd.Series

    _assert_frame_equal = pandas_assert_frame_equal
    _concat = pd.concat
    _infer_dtype = pd.api.types.infer_dtype
    _isnull = pd.isnull
    _isna = pd.isna
    _merge = pd.merge
    _read_csv = pd.read_csv
    _to_datetime = pd.to_datetime

elif backend_df == "duckdb":
    try:
        import duckdb
    except ImportError:
        raise ImportError("Duckdb is not installed. Install it with `pip install duckdb`.")

    _DataFrame = LazyFrame
    _Series = LazySeries

    _assert_frame_equal = lazy_assert_frame_equal
    _concat = lazy_concat
    _infer_dtype = lazy_infer_dtype
    _isnull = lazy_isnull
    _isna = lazy_isna
    _merge = lazy_merge
    _read_csv = lazy_read_csv
    _to_datetime = lazy_to_datetime

else:
    raise ValueError("Invalid value for BACKEND_DF environment variable. Use 'eager' or 'lazy'.")

DataFrame = _DataFrame
Series = _Series

assert_frame_equal = _assert_frame_equal
concat = _concat
infer_dtype = _infer_dtype
isnull = _isnull
isna = _isna
merge = _merge
read_csv = _read_csv
to_datetime = _to_datetime

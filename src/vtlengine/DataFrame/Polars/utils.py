from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as polars_assert_frame_equal

polars_dtype_mapping = {
    "object": pl.Utf8,
    object: pl.Utf8,
    np.object_: pl.Utf8,
    "str": pl.Utf8,
    "string": pl.Utf8,
    "int64": pl.Int64,
    "i64": pl.Int64,
    np.int64: pl.Int64,
    "int32": pl.Int32,
    "i32": pl.Int32,
    np.int32: pl.Int32,
    "float64": pl.Float64,
    "f64": pl.Float64,
    "float32": pl.Float32,
    "f32": pl.Float32,
    np.float32: pl.Float32,
    np.float64: pl.Float64,
    "bool": pl.Boolean,
    np.bool_: pl.Boolean,
    np.datetime64: pl.Datetime,
    np.timedelta64: pl.Duration,
}

def handle_dtype(dtype: Any) -> Any:
    return polars_dtype_mapping.get(dtype, dtype)

def _assert_frame_equal(left, right, check_dtype=True, **kwargs):
    return polars_assert_frame_equal(left.df, right.df, check_dtype=check_dtype)

def _isnull(obj):
    return pd.isnull(obj)

def _isna(obj):
    return pd.isna(obj)
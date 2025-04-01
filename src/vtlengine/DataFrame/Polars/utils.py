from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from polars.testing import assert_frame_equal as polars_assert_frame_equal

# Mapping of common data types to Polars data types
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
    """Convert common data types to Polars data types."""
    return polars_dtype_mapping.get(dtype, dtype)


def _assert_frame_equal(left, right, check_dtype=True, **kwargs):
    """Assert that two DataFrames are equal."""
    return polars_assert_frame_equal(left.df, right.df, check_dtype=check_dtype)


def _isnull(obj):
    """Check for null values."""
    return pd.isnull(obj)


def _isna(obj):
    """Check for NA values."""
    return pd.isna(obj)


class Columns:
    """Wrapper around a list of columns (used to add tolist method to columns)."""

    def __init__(self, columns):
        self._columns = columns

    def tolist(self):
        return self._columns

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __getitem__(self, index):
        return self._columns[index]

    def __repr__(self):
        return repr(self._columns)

    def __str__(self):
        return str(self._columns)

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
    "boolean": pl.Boolean,
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

    _columns: list

    def __init__(self, columns=None):
        if columns is None:
            columns = []
        self._columns = columns

    def tolist(self):
        return self._columns

    def __delitem__(self, key):
        self._columns.remove(key)

    def __getitem__(self, index):
        return self._columns[index]

    def __iter__(self):
        return iter(self._columns)

    def __len__(self):
        return len(self._columns)

    def __repr__(self):
        return repr(self._columns)

    def __str__(self):
        return str(self._columns)


class Index:
    """Handles index management for IndexedSeries and IndexedDataFrame."""

    _index: pl.Series = pl.Series("index", [])
    _max_index: int = -1

    def __init__(self, length=0):
        self.index = pl.Series("index", range(length))  # Separate index series
        self.max_index = length - 1  # Track max index globally

    # def __get__(self, instance, owner):
    #     """Returns itself when accessed."""
    #     return self

    def __set__(self, instance, value):
        """Automatically updates the index series when set."""
        if isinstance(value, pl.Series):
            self.index = value
            self.max_index = len(value) - 1
        elif isinstance(value, Index):
            self.index = value.index
            self.max_index = value.max_index
        elif isinstance(value, (list, range)):
            self.index = pl.Series("index", value)
            self.max_index = len(value) - 1
        else:
            raise ValueError("Index must be a pl.Series or another Index instance")

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def max_index(self):
        return self._max_index

    @max_index.setter
    def max_index(self, value):
        self._max_index = value

    def update(self, length):
        """Updates max_index when new data is added."""
        self.max_index += length
        new_indices = pl.Series(range(self.max_index - length + 1, self.max_index + 1))
        self.index = pl.concat([self.index, new_indices], how="vertical")

    def reindex(self, value, **kwargs):
        """Resets the index to start from 0."""
        self.index = pl.Series("index", range(len(self.index)))
        self.max_index = len(self.index) - 1

    def to_list(self):
        return self.index.to_list()

import os
from pathlib import Path
from typing import IO, Any, Dict, Mapping, Self, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
from pandas._testing import assert_frame_equal as pandas_assert_frame_equal
from polars._utils.unstable import unstable
from polars.series.plotting import SeriesPlot
from polars.testing import assert_frame_equal as polars_assert_frame_equal

POLARS_STR = ["polars", "pl"]

backend_df = "pl" if os.getenv("BACKEND_DF", "").lower() in POLARS_STR else "pd"

if backend_df == "pd":
    _DataFrame = pd.DataFrame
    _Series = pd.Series

    _assert_frame_equal = pandas_assert_frame_equal
    _concat = pd.concat
    _isnull = pd.isnull
    _isna = pd.isna
    _merge = pd.merge
    _read_csv = pd.read_csv

elif backend_df == "pl":

    class PolarsObject(pl.DataType):
        pass

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
        """Convert numpy dtype to Polars dtype using a mapping dictionary."""
        return polars_dtype_mapping.get(dtype, dtype)

    class PolarsDataFrame(pl.DataFrame):
        _df: pl.DataFrame() = pl.DataFrame()
        _series: Dict[str, "PolarsSeries"] = {}

        def __init__(self, data=None, columns=None):
            super().__init__(data)
            self.series = {}
            if data is None:
                if columns is not None:
                    for col in columns:
                        self.series[col] = PolarsSeries([], name=col)
            elif isinstance(data, dict):
                for col, values in data.items():
                    if not isinstance(values, PolarsSeries):
                        self.series[col] = PolarsSeries(values, name=col)
                    else:
                        self.series[col] = values
            elif isinstance(data, list):
                if columns is None:
                    columns = [f"col{i}" for i in range(len(data))]
                for col_name, col_data in zip(columns, data):
                    self.series[col_name] = PolarsSeries(col_data, name=col_name)
            elif isinstance(data, pl.DataFrame):
                for col in data.columns:
                    self.series[col] = PolarsSeries(data[col].to_list(), name=col)
            else:
                raise ValueError("Unsupported data type for creating PolarsDataFrame.")
            self._build_df()

        def _build_df(self):
            d = {col: series.to_list() for col, series in self.series.items()}
            self.df = pl.DataFrame(d)
            # self.dtypes = {col: series.dtype for col, series in self.series.items()}

        def __delitem__(self, key):
            if key in self.series:
                del self.series[key]
                self._build_df()
            else:
                raise KeyError(f"Column '{key}' does not exist in the DataFrame.")

        def __getitem__(self, key):
            if isinstance(key, str):
                return self.series[key]
            elif isinstance(key, tuple):
                if len(key) == 2:
                    row, col = key
                    return PolarsDataFrame(self.df[row, col])
                else:
                    raise KeyError("Unsupported index type for __getitem__")
            elif isinstance(key, slice):
                return PolarsDataFrame(self.df[key])
            elif isinstance(key, list):
                if all(isinstance(i, str) for i in key):
                    return PolarsDataFrame(self.df.select(key))
                else:
                    raise KeyError("Unsupported index type for __getitem__")
            else:
                raise KeyError("Unsupported index type for __getitem__")

        def __setitem__(self, key, value):
            if not isinstance(value, PolarsSeries):
                value = PolarsSeries(value, name=key)
            self.series[key] = value
            self._build_df()

        def __repr__(self):
            return self.df.__repr__()

        def _repr_html_(self, *args, **kwargs):
            return self.df._repr_html_(*args, **kwargs)

        def __str__(self):
            return self.df.__str__()

        class _ColumnsWrapper:
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

        @property
        def columns(self):
            return self._ColumnsWrapper(self.df.columns)

        @property
        def df(self):
            return self._df

        @df.setter
        def df(self, df: pl.DataFrame):
            self._df = df

        @property
        def dtypes(self):
            return self.df.dtypes

        # @dtypes.setter
        # def dtypes(self, value):
        #     self.df.dtypes = value

        @property
        def empty(self):
            return self.__len__() == 0

        @property
        def height(self) -> int:
            return self.df.height

        @property
        def iloc(self):
            return self

        @property
        def loc(self):
            return self

        @property
        @unstable()
        def plot(self):
            return self.df.plot

        @property
        def series(self):
            return self._series

        @series.setter
        def series(self, series: Dict[str, Union["PolarsSeries", pl.Series]]):
            self._series = series

        @property
        def shape(self) -> tuple[int, int]:
            return self.df.shape

        @property
        def size(self) -> int:
            return self.height * self.width

        @property
        def width(self) -> int:
            return self.df.width

        def apply(self, func, axis=0, *args, **kwargs):
            if axis == 0:
                # Apply function to each column
                new_series = {col: PolarsSeries(func(series.to_list(), *args, **kwargs), name=col) for col, series in
                              self.series.items()}
            elif axis == 1:
                # Apply function to each row
                new_data = [func(row, *args, **kwargs) for row in self.df.rows()]
                new_series = {f"result_{i}": PolarsSeries([row[i] for row in new_data], name=f"result_{i}") for i in
                              range(len(new_data[0]))}
            else:
                raise ValueError("Axis must be 0 (columns) or 1 (rows)")

            return PolarsDataFrame(new_series)

        def assign(self, **kwargs):
            new_series = self.series.copy()
            for key, value in kwargs.items():
                if not isinstance(value, PolarsSeries):
                    value = PolarsSeries(value, name=key)
                new_series[key] = value
            return PolarsDataFrame(new_series)

        def copy(self):
            return PolarsDataFrame(self.series)

        def drop(self, columns=None, inplace=False, **kwargs):
            if columns is None:
                return self
            if isinstance(columns, str):
                columns = [columns]
            new_series = {col: series for col, series in self.series.items() if col not in columns}
            if inplace:
                self.series = new_series
                self._build_df()
                return None
            else:
                return PolarsDataFrame(new_series)

        def drop_duplicates(self, subset=None, keep="first", inplace=False):
            if isinstance(keep, bool):
                keep = "first"
            if subset is None:
                df = self.df.unique(keep=keep)
            else:
                df = self.df.unique(subset=subset, keep=keep)

            if inplace:
                self.df = df
                self._build_df()
                return None
            else:
                return PolarsDataFrame(df)

        def dropna(self, subset=None, inplace=False):
            if subset is None:
                df = self.df.drop_nulls()
            else:
                df = self.df.drop_nulls(subset=subset)

            if inplace:
                self.df = df
                self._build_df()
                return None
            else:
                return PolarsDataFrame(df)

        def fillna(self, value, *args, **kwargs):
            new_series = {}
            if isinstance(value, dict):
                for col, series in self.series.items():
                    if col in value:
                        new_data = [value[col] if x is None else x for x in series.to_list()]
                    else:
                        new_data = series.to_list()
                    new_data = [None if x != x else x for x in new_data]
                    new_series[col] = PolarsSeries(new_data, name=col)
            else:
                for col, series in self.series.items():
                    new_data = [value if x is None else x for x in series.to_list()]
                    new_data = [None if x != x else x for x in new_data]
                    new_series[col] = PolarsSeries(new_data, name=col)
            return PolarsDataFrame(new_series)

        def groupby(self, by, **kwargs):
            grouped_df = self.df.group_by(by).agg(pl.all())
            return PolarsDataFrame(grouped_df)

        def loc_by_mask(self, boolean_mask):
            if len(boolean_mask) != len(self):
                raise ValueError("Boolean mask length must match the length of the DataFrame")
            filtered_data = {col: [x for x, mask in zip(series.to_list(), boolean_mask) if mask] for col, series in
                             self.series.items()}
            return PolarsDataFrame(filtered_data)

        def reindex(self, index=None, fill_value=None, copy=True, axis=0, *args, **kwargs):
            if axis not in [0, 1]:
                raise ValueError("`axis` must be 0 (rows) or 1 (columns)")

            if axis == 0:
                if index is None:
                    return self.copy() if copy else self

                new_data = {}
                for col in self.columns.tolist():
                    series = self.series[col].to_list()
                    new_series = [fill_value] * len(index)
                    for i, idx in enumerate(index):
                        if idx < len(series):
                            new_series[i] = series[idx]
                    new_data[col] = new_series

                return PolarsDataFrame(new_data)
            else:
                if index is None:
                    return self.copy() if copy else self

                new_series = {col: self.series[col] for col in index if col in self.series}
                for col in index:
                    if col not in new_series:
                        new_series[col] = PolarsSeries([fill_value] * self.height, name=col)

                return PolarsDataFrame(new_series)

        def rename(self, columns: dict, inplace: bool = False, *args, **kwargs):
            new_series = {columns.get(col, col): series for col, series in self.series.items()}
            if inplace:
                self.series = new_series
                self._build_df()
                return None
            else:
                return PolarsDataFrame(new_series)

        def replace(self, to_replace, value=None, **kwargs):
            if isinstance(to_replace, dict):
                df_temp = self
                for old, new in to_replace.items():
                    df_temp = df_temp.replace(old, new, **kwargs)
                return df_temp
            new_data = {}
            for col in self.columns.tolist():
                series = self.series[col].to_list()
                new_series = [
                    value
                    if (
                        x == to_replace
                        or (to_replace is np.nan and (x != x))
                        or (to_replace is None and x is None)
                    )
                    else x
                    for x in series
                ]
                new_data[col] = new_series
            return PolarsDataFrame(new_data)

        def reset_index(self, drop: bool = False, inplace: bool = False):
            if drop:
                new_df = self.df.with_row_count(name="row_nr")
            else:
                new_df = self.df.with_row_count(name="index")

            if inplace:
                self.df = new_df
                self._build_df()
                return None
            else:
                return PolarsDataFrame(new_df)

        def sort_values(self, by: str, ascending: bool = True):
            sorted_df = self.df.sort(by, descending=not ascending)
            return PolarsDataFrame(sorted_df)

        def view(self):
            print(self.df)

        def view_series(self, column_name: str):
            if column_name in self.series:
                print(self.series[column_name])
            else:
                raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")

    class PolarsSeries(pl.Series):
        def __init__(self, data, name=None, *args, **kwargs):
            super().__init__(name=name, values=data)

        def __getitem__(self, index):
            if isinstance(index, (int, slice)):
                return self.to_list()[index]
            if isinstance(index, PolarsSeries):
                index = index.to_list()
            if isinstance(index, list) and all(isinstance(i, bool) for i in index):
                return self.loc_by_mask(index)
            else:
                raise TypeError("Invalid index type for __getitem__")

        def __repr__(self):
            return super().__repr__()

        def _repr_html_(self):
            return super()._repr_html_()

        class iLocIndexer:
            def __init__(self, series):
                self.series = series

            def __getitem__(self, index):
                return self.series.to_list()[index]

        class LocIndexer:
            def __init__(self, series):
                self.series = series

            def __getitem__(self, index):
                if isinstance(index, int):
                    return self.series.to_list()[index]
                elif isinstance(index, slice):
                    return self.series.to_list()[index]
                elif isinstance(index, list):
                    return [self.series.to_list()[i] for i in index]
                else:
                    raise TypeError("Invalid index type for loc")

        @property
        def name(self):
            return self._s.name()

        @property
        def dtype(self):
            return self._s.dtype()

        @property
        def index(self):
            return range(len(self))

        @property
        def empty(self):
            return self.__len__() == 0

        @property
        def iloc(self) -> iLocIndexer:
            return self.iLocIndexer(self)

        @property
        def loc(self) -> LocIndexer:
            return self.LocIndexer(self)

        @property
        def plot(self) -> SeriesPlot:
            return SeriesPlot(self)

        @property
        def values(self):
            return self.to_list()

        def apply(self, func, *args, **kwargs):
            return PolarsSeries([func(x, *args, **kwargs) for x in self.to_list()], name=self.name)

        def astype(self, dtype, errors="raise"):
            try:
                # Handle numpy to polars type conversion
                if dtype != self.dtype and dtype != np.object_:
                    if dtype in polars_dtype_mapping:
                        dtype = polars_dtype_mapping[dtype]
                    return self.cast(dtype)
                return self

            except Exception as e:
                if errors == "raise":
                    raise e
                else:
                    return self

        def copy(self):
            return PolarsSeries(self.to_list(), name=self.name)

        def dropna(self):
            return PolarsSeries(self.drop_nulls(), name=self.name)

        def isnull(self):
            return PolarsSeries(self.is_null(), name=self.name)

        def loc_by_mask(self, boolean_mask):
            if len(boolean_mask) != len(self):
                raise ValueError("Boolean mask length must match the length of the series")
            return PolarsSeries([x for x, mask in zip(self.to_list(), boolean_mask) if mask], name=self.name)

        def map(self, func, na_action=None):
            if na_action == "ignore":
                return PolarsSeries(
                    [func(x) if x is not None else x for x in self.to_list()], name=self.name
                )
            else:
                return PolarsSeries([func(x) for x in self.to_list()], name=self.name)

        def notnull(self):
            return PolarsSeries(self.is_not_null(), name=self.name)

        def replace(self, to_replace, value=None, **kwargs) -> "PolarsSeries":
            if isinstance(to_replace, dict):
                new_data = self.to_list()
                for old, new in to_replace.items():
                    new_data = [new if x == old else x for x in new_data]
                return PolarsSeries(new_data, name=self.name)
            else:
                new_data = [value if x == to_replace else x for x in self.to_list()]
                return PolarsSeries(new_data, name=self.name)

        def view(self):
            """Display the Series in a tabular format for debugging."""
            print(self)

    def _assert_frame_equal(
        left: PolarsDataFrame, right: PolarsDataFrame, check_dtype: bool = True, **kwargs
    ):
        return polars_assert_frame_equal(left.df, right.df, check_dtype=check_dtype)

    def _concat(objs, *args, **kwargs):
        polars_objs = [obj.df if isinstance(obj, PolarsDataFrame) else obj for obj in objs]
        return PolarsDataFrame(pl.concat(polars_objs))

    def _isnull(obj):
        return pd.isnull(obj)

    def _isna(obj):
        return obj.isnull()


    def _merge(self, right, on=None, how="inner", suffixes=("_x", "_y"), *args, **kwargs):
        if not isinstance(right, PolarsDataFrame):
            right = PolarsDataFrame(right)
        # if on is None:
        #     raise ValueError("The 'on' parameter must be specified for merging.")

        left_df = self.df
        right_df = right.df

        # TODO: check this with the left and right on
        # Identify overlapping columns
        if on is not None:
            overlap = set(left_df.columns).intersection(set(right_df.columns)) - set(on)

            # Apply suffixes to overlapping columns
            for col in overlap:
                left_df = left_df.rename({col: f"{col}{suffixes[0]}"})
                right_df = right_df.rename({col: f"{col}{suffixes[1]}"})

            merged_df = left_df.join(right_df, on=on, how=how)

        else:
            if left_df.width:
                left_on = left_df.columns
                merged_df = left_df.join(right_df, how=how, left_on=left_on)
            else:
                right_on = right_df.columns
                merged_df = left_df.join(right_df, how=how, right_on=right_on)

        return PolarsDataFrame(merged_df)

    def _read_csv(
        source: str | Path | IO[str] | IO[bytes] | bytes,
        dtype: dict[str, Any] | None = None,
        na_values: list[str] | None = None,
        **kwargs,
    ) -> PolarsDataFrame:
        # Convert dtype to schema_overrides for Polars
        # schema_overrides = {k: handle_dtype(v) for k, v in (dtype or {}).items()}

        # Read the CSV file with Polars
        df = pl.read_csv(
            source,
            schema=None,
            # schema_overrides=schema_overrides,
            null_values=na_values,
        )
        return PolarsDataFrame(df)

    _DataFrame = PolarsDataFrame
    _Series = PolarsSeries

else:
    raise ValueError("Invalid value for BACKEND_DF environment variable. Use 'pd' or 'pl'.")


DataFrame = _DataFrame
Series = _Series

assert_frame_equal = _assert_frame_equal
concat = _concat
isnull = _isnull
isna = _isna
merge = _merge
read_csv = _read_csv

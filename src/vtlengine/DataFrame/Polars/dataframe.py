import numpy as np
import polars as pl
from typing import Any, Dict, IO, Union
from pathlib import Path

from polars._utils.unstable import unstable

from .series import PolarsSeries
from .utils import handle_dtype, _isnull


class PolarsDataFrame(pl.DataFrame):
    _df: pl.DataFrame() = pl.DataFrame()
    _series: Dict[str, "PolarsSeries"] = {}

    def __init__(self, data=None, columns=None, **kwargs):
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
            new_series = {col: PolarsSeries(func(series, *args, **kwargs), name=col) for col, series in
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

    def fillna(self, value, inplace=False, *args, **kwargs):
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
                new_data = [value if _isnull(x) else x for x in series.to_list()]
                new_data = [None if x != x else x for x in new_data]
                new_series[col] = PolarsSeries(new_data, name=col)

        if inplace:
            self.series = new_series
            self._build_df()
            return None
        else:
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

    def to_csv(self, file_path: str, **kwargs):
        self.df.write_csv(file_path)

    def to_series(self, index: int = 0, *args, **kwargs) -> "PolarsSeries":
        return PolarsSeries(self.df.to_series(index), name=self.columns[index])

    def view(self):
        print(self.df)

    def view_series(self, column_name: str):
        if column_name in self.series:
            print(self.series[column_name])
        else:
            raise KeyError(f"Column '{column_name}' does not exist in the DataFrame.")


def _concat(objs, *args, **kwargs):
    polars_objs = [obj.df if isinstance(obj, PolarsDataFrame) else obj for obj in objs]
    return PolarsDataFrame(pl.concat(polars_objs))


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
    if na_values is not None and "null" not in na_values:
        na_values.append("null")

    # Read the CSV file with Polars
    df = pl.read_csv(
        source,
        # schema_overrides=schema_overrides,
        null_values=na_values or ["null", "None"],
    )
    return PolarsDataFrame(df)
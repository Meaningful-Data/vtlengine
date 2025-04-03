from pathlib import Path
from typing import IO, Dict, Sequence, Union, Any

import numpy as np
import pandas as pd
import polars as pl
from IPython.core.guarded_eval import dict_keys
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from polars._utils.unstable import unstable

from .series import PolarsSeries
from .utils import Columns, Index, _isnull


class PolarsDataFrame(pl.DataFrame):
    _df: pl.DataFrame = pl.DataFrame()
    _series: Dict[str, PolarsSeries] = {}
    _columns: Columns = Columns()
    _index: Index = Index()
    _groups: Dict[Any, Any] = {}

    def __init__(self, data=None, columns=None, index=None, **kwargs):
        self.series = {}
        self.index = Index()

        if data is None and columns is not None:
            self.series = {col: PolarsSeries([], name=col) for col in columns}
        elif isinstance(data, dict):
            self.series = {
                col: PolarsSeries(values, name=col)
                if not isinstance(values, PolarsSeries)
                else values
                for col, values in data.items()
            }
        elif isinstance(data, list):
            if columns is None:
                columns = [f"col{i}" for i in range(len(data))]
            if isinstance(data[0], tuple):
                data = list(zip(*data))
            self.series = {
                col_name: PolarsSeries(col_data, name=col_name)
                for col_name, col_data in zip(columns, data)
            }
        elif isinstance(data, (pl.DataFrame, pd.DataFrame)):
            self.series = {col: PolarsSeries(data[col].to_list(), name=col) for col in data.columns}
        elif data is None:
            self.series = {}
        else:
            raise ValueError("Unsupported data type for creating PolarsDataFrame.")

        self._build_df(index=index)

    def _build_df(self, index=None):
        # Ensure all columns have the same length by filling with None
        max_length = max(map(len, self.series.values()))
        for key, series in self.series.items():
            pad_size = max_length - len(series)
            if pad_size > 0:
                self.series[key] = pl.concat([series, pl.Series([None] * pad_size)])

        d = {col: series.to_list() for col, series in self.series.items()}
        self.df = pl.DataFrame(d)
        self.columns = list(self.series.keys())
        if index is not None:
            self.index = index
        else:
            self.index.update(self.height)

    def __copy__(self):
        return self.copy()

    def __delitem__(self, key):
        if key in self.series:
            del self.series[key]
            self._build_df()
        else:
            raise KeyError(f"Column '{key}' does not exist in the DataFrame.")

    def __getitem__(self, key):
        if isinstance(key, dict_keys):
            key = list(key)
        elif isinstance(key, (Index, PolarsSeries)):
            key = key.to_list()

        if isinstance(key, str):
            try:
                return self.series[key]
            except KeyError:
                raise KeyError(f"Column '{key}' does not exist in the DataFrame.")
        elif isinstance(key, tuple) and len(key) == 2:
            row, col = key
            return PolarsDataFrame(self.df[row, col])
        elif isinstance(key, (slice, range)):
            return PolarsDataFrame(self.df[key])
        elif isinstance(key, list):
            if all(isinstance(x, str) for x in key):
                return PolarsDataFrame(self.df.select(key))
            elif all(isinstance(x, bool) for x in key):
                return self.loc_by_mask(key)
            elif all(isinstance(x, int) for x in key):
                return PolarsDataFrame(self.df[key])
        else:
            raise KeyError("Unsupported index type for __getitem__")

    def __setitem__(self, key, value):
        if not isinstance(value, PolarsSeries):
            if isinstance(value, (int, float, str, bool)) or value is None:
                value = [value] * self.height
            value = PolarsSeries(value, name=key)
        if len(value) != self.height:
            value = PolarsSeries(value.to_list() + [None] * (self.height - len(value)), name=key)
        self.series[key] = value
        self._build_df(index=value.index)

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self, *args, **kwargs):
        return self.df._repr_html_(*args, **kwargs)

    def __str__(self):
        return self.df.__str__()

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, columns):
        self._columns = Columns(columns)

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df: pl.DataFrame):
        self._df = df

    @property
    def dtypes(self):
        return self.df.dtypes

    @property
    def empty(self):
        return self.__len__() == 0

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, groups):
        if isinstance(groups, pl.DataFrame):
            key_columns = [col for col in groups.columns if col != "__temp_index__"]
            key_expr = pl.struct(key_columns) if len(key_columns) > 1 else pl.col(key_columns[0])
            df_exploded = groups.explode("__temp_index__").with_columns(key_expr.alias("key"))
            df_grouped = df_exploded.group_by("key").agg(pl.col("__temp_index__"))
            groups = {row[0]: row[1] for row in df_grouped.iter_rows()}
        self._groups = groups

    @property
    def height(self) -> int:
        return self.df.height

    @property
    def iloc(self):
        return self

    # TODO: check this (unlike pandas, polars do not work with an intern index)
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

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
    def T(self):
        return self

    @property
    def width(self) -> int:
        return self.df.width

    def apply(self, func, axis=0, *args, **kwargs):
        if axis == 1:
            return PolarsSeries([func(row) for row in self.df.iter_rows(named=True)])

        elif axis == 0:
            return PolarsDataFrame({col: self.df[col].map(func) for col in self.df.columns})
        else:
            raise ValueError("Axis must be 0 (columns) or 1 (rows)")

    def assign(self, **kwargs):
        new_series = self.series.copy()
        for key, value in kwargs.items():
            if not isinstance(value, PolarsSeries):
                value = PolarsSeries(value, name=key)
            new_series[key] = value
        return PolarsDataFrame(new_series)

    def copy(self):
        return PolarsDataFrame(self.df.clone())

    def drop(self, columns=None, **kwargs):
        self.df = self.df.drop(columns)
        for col in columns if isinstance(columns, list) else [columns]:
            del self.series[col]
        self.columns = list(self.series.keys())
        return self

    def drop_duplicates(self, subset=None, keep="first", inplace=False):
        if isinstance(keep, bool):
            keep = "first"
        df = self.df.unique(subset=subset, keep=keep)

        if inplace:
            self.df = df
            self._build_df()
            return None
        else:
            return PolarsDataFrame(df)

    def dropna(self, subset=None, inplace=False, **kwargs):
        if subset is None:
            df = self.df.drop_nulls()
        else:
            df = self.df.drop_nulls(subset=subset)

        if inplace:
            self.df = df
            self._build_df()
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
            for col, _series in self.series.items():
                new_data = [value if _isnull(x) else x for x in _series.to_list()]
                new_data = [None if x != x else x for x in new_data]
                new_series[col] = PolarsSeries(new_data, name=col)

        if inplace:
            self.series = new_series
            self._build_df(index=self.index)
            return None
        else:
            return PolarsDataFrame(new_series)

    def groupby(self, by, **kwargs):
        self.df = self.df.with_columns((self.index.index).alias("__temp_index__"))
        grouped_df = self.df.group_by(by).agg(pl.all())
        self.groups = grouped_df.select(by + ["__temp_index__"])
        return grouped_df.drop("__temp_index__")

    # TODO: check this
    def loc_by_mask(self, boolean_mask):
        if len(boolean_mask) != len(self):
            raise ValueError("Boolean mask length must match the length of the DataFrame")
        filtered_data = {
            col: [x for x, mask in zip(series.to_list(), boolean_mask) if mask]
            for col, series in self.series.items()
        }
        filtered_index = [idx for idx, mask in zip(self.index.to_list(), boolean_mask) if mask]
        result_df = PolarsDataFrame(filtered_data)
        result_df.index = Index(len(filtered_index))
        result_df.index.index = pl.Series("index", filtered_index)
        return result_df

    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        var_name: str | None = None,
        value_name: str | None = None,
    ) -> DataFrame:
        self.df = self.df.melt(id_vars, value_vars, var_name, value_name)
        self.columns = list(self.df.columns)
        self.series = {col: PolarsSeries(self.df[col].to_list(), name=col) for col in self.columns}
        return self

    def merge(self, right, on=None, how="inner", suffixes=("_x", "_y"), *args, **kwargs):
        return _merge(self, right, on=on, how=how, suffixes=suffixes)

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

    def reindex(self, value=None, **kwargs):
        self.index.reindex(value, **kwargs)
        return self

    def reset_index(self, value=None, **kwargs):
        self.index.reindex(value, **kwargs)
        return self

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
    base_columns = polars_objs[0].columns
    for i, pobj in enumerate(polars_objs[1:]):
        if pobj.columns != base_columns and sorted(base_columns) == sorted(pobj.columns):
            polars_objs[i + 1] = pobj.select(base_columns)
    return PolarsDataFrame(pl.concat(polars_objs))


def _merge(self, right, on=None, left_on=None, right_on=None,
    how="inner", suffixes=("_x", "_y"), *args, **kwargs,
):
    if not isinstance(right, PolarsDataFrame):
        right = PolarsDataFrame(right)

    left_df = self.df
    right_df = right.df

    overlap = set(left_df.columns).intersection(right_df.columns)
    if on or (left_on and right_on):
        overlap.difference_update(on or (left_on and right_on))

    for col in overlap:
        left_df = left_df.rename({col: f"{col}{suffixes[0]}"})
        right_df = right_df.rename({col: f"{col}{suffixes[1]}"})

    merged_df = left_df.join(right_df, on=on, left_on=left_on, right_on=right_on, how=how)
    return PolarsDataFrame(merged_df)


def _read_csv(
    source: str | Path | IO[str] | IO[bytes] | bytes,
    na_values: list[str] | None = None,
    **kwargs,  # pandas args that do not exist in polars
) -> PolarsDataFrame:
    if na_values is not None and "null" not in na_values:
        na_values.append("null")
    return PolarsDataFrame(pl.read_csv(source, null_values=na_values or ["null", "None"]))

import os

import numpy as np
import pandas as pd
import polars as pl

POLARS_STR = ["polars", "pl"]

backend_df = "pl" if os.getenv("BACKEND_DF", "").lower() in POLARS_STR else "pd"

if backend_df == "pd":
    _DataFrame = pd.DataFrame
    _Series = pd.Series

    _concat = pd.concat
    _isnull = pd.isnull
    _isna = pd.isna
    _merge = pd.merge
    _read_csv = pd.read_csv

elif backend_df == "pl":

    class PolarsDataFrame(pl.DataFrame):
        """Override of polars.DataFrame with pandas-like methods"""

        def __init__(self, data=None, columns=None):
            super().__init__(data)
            self._series = {}  # Dictionary of columns: name -> PolarsSeries

            if data is None:
                if columns is not None:
                    for col in columns:
                        self._series[col] = PolarsSeries([], name=col)
            elif isinstance(data, dict):
                for col, values in data.items():
                    if not isinstance(values, PolarsSeries):
                        self._series[col] = PolarsSeries(values, name=col)
                    else:
                        self._series[col] = values
            elif isinstance(data, list):
                if columns is None:
                    columns = [f"col{i}" for i in range(len(data))]
                for col_name, col_data in zip(columns, data):
                    self._series[col_name] = PolarsSeries(col_data, name=col_name)
            elif isinstance(data, pl.DataFrame):
                for col in data.columns:
                    self._series[col] = PolarsSeries(data[col].to_list(), name=col)
            else:
                raise ValueError("Unsupported data type for creating PolarsDataFrame.")

            self._build_df()

        def _build_df(self):
            d = {col: series.to_list() for col, series in self._series.items()}
            self._df = pl.DataFrame(d)

        def __getitem__(self, key):
            if isinstance(key, str):
                return self._series[key]
            elif isinstance(key, list):
                new_data = {col: self._series[col].to_list() for col in key if col in self._series}
                return PolarsDataFrame(new_data)
            else:
                raise KeyError("Unsupported index type.")

        def __setitem__(self, key, value):
            if not isinstance(value, PolarsSeries):
                value = PolarsSeries(value, name=key)
            self._series[key] = value
            self._build_df()

        def __repr__(self):
            return repr(self)

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

        @property
        def columns(self):
            return self._ColumnsWrapper(self._df.columns)

        def fillna(self, value, *args, **kwargs):
            new_series = {}
            if isinstance(value, dict):
                for col, series in self._series.items():
                    if col in value:
                        new_data = [value[col] if x is None else x for x in series.to_list()]
                    else:
                        new_data = series.to_list()
                    new_data = [None if x != x else x for x in new_data]  # Convert NaN to None
                    new_series[col] = PolarsSeries(new_data, name=col)
            else:
                for col, series in self._series.items():
                    new_data = [value if x is None else x for x in series.to_list()]
                    new_data = [None if x != x else x for x in new_data]  # Convert NaN to None
                    new_series[col] = PolarsSeries(new_data, name=col)
            return PolarsDataFrame(new_series)

        def replace(self, to_replace, value=None, **kwargs):
            if isinstance(to_replace, dict):
                df_temp = self
                for old, new in to_replace.items():
                    df_temp = df_temp.replace(old, new, **kwargs)
                return df_temp

            new_data = {}
            for col in self.columns.tolist():
                series = self._series[col].to_list()
                new_series = [
                    value if (x == to_replace or (to_replace is np.nan and (x != x)) or (
                                to_replace is None and x is None)) else x
                    for x in series
                ]
                new_data[col] = new_series

            return PolarsDataFrame(new_data)


    class PolarsSeries(pl.Series):
        def __init__(self, data, name=None, *args, **kwargs):
            super().__init__(name=name, values=data)

        def __repr__(self):
            return repr(self)

        def isnull(self):
            return self.is_null()


    def _concat(objs, *args, **kwargs):
        return pl.concat(objs, *args, **kwargs)

    def _isnull(obj):
        return obj.isnull()

    def _isna(obj):
        return obj.isnull()

    def _merge(left, right, *args, **kwargs):
        return left.join(right, *args, **kwargs)

    def _read_csv(filepath, *args, **kwargs):
        return PolarsDataFrame(pl.read_csv(filepath))


    _DataFrame = PolarsDataFrame
    _Series = PolarsSeries

    concat = _concat
    isnull = _isnull
    isna = _isna
    merge = _merge
    read_csv = _read_csv

else:
    raise ValueError("Invalid value for BACKEND_DF environment variable. Use 'pd' or 'pl'.")



DataFrame = _DataFrame
Series = _Series

concat = _concat
isnull = _isnull
isna = _isna
merge = _merge
read_csv = _read_csv
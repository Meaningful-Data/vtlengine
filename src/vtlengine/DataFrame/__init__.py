import os
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
import polars as pl
from polars.dependencies import polars_cloud
from polars.interchange.utils import polars_dtype_to_dtype

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

    polars_dtype_mapping = {
        np.object_: pl.Utf8,
        np.int64: pl.Int64,
        np.float64: pl.Float64,
        np.bool_: pl.Boolean,
        np.datetime64: pl.Datetime,
        np.timedelta64: pl.Duration,
    }


    def handle_dtype(dtype: Any) -> Any:
        """Convert numpy dtype to Polars dtype using a mapping dictionary."""
        return polars_dtype_mapping.get(dtype, dtype)


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

        def drop(self, columns=None, inplace=False):
            if columns is None:
                return self
            if isinstance(columns, str):
                columns = [columns]
            new_series = {col: series for col, series in self._series.items() if col not in columns}
            if inplace:
                self._series = new_series
                self._build_df()
                return None
            else:
                return PolarsDataFrame(new_series)

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

        def astype(self, dtype, errors="raise"):
            try:
                # Handle numpy to polars type conversion
                if dtype in polars_dtype_mapping:
                    dtype = polars_dtype_mapping[dtype]
                return self.cast(dtype)

            except Exception as e:
                if errors == "raise":
                    raise e
                else:
                    return self

        def isnull(self):
            return self.is_null()

        def map(self, func, na_action=None):
            if na_action == "ignore":
                return PolarsSeries([func(x) if x is not None else x for x in self.to_list()], name=self.name)
            else:
                return PolarsSeries([func(x) for x in self.to_list()], name=self.name)


    def _concat(objs, *args, **kwargs):
        return pl.concat(objs, *args, **kwargs)

    def _isnull(obj):
        pd.isnull(obj)

    def _isna(obj):
        return obj.isnull()

    def _merge(left, right, *args, **kwargs):
        return left.join(right, *args, **kwargs)


    def _read_csv(
            source: str | Path | IO[str] | IO[bytes] | bytes,
            dtype: dict[str, Any] | None = None,
            na_values: list[str] | None = None,
            **kwargs
    ) -> PolarsDataFrame:
        # Convert dtype to schema_overrides for Polars
        schema_overrides = {k: handle_dtype(v) for k, v in (dtype or {}).items()}

        # Read the CSV file with Polars
        df = pl.read_csv(
            source,
            schema_overrides=schema_overrides,
            null_values=na_values
        )
        return PolarsDataFrame(df)


    _DataFrame = PolarsDataFrame
    _Series = PolarsSeries

else:
    raise ValueError("Invalid value for BACKEND_DF environment variable. Use 'pd' or 'pl'.")



DataFrame = _DataFrame
Series = _Series

concat = _concat
isnull = _isnull
isna = _isna
merge = _merge
read_csv = _read_csv
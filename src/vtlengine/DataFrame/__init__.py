import csv
import os

import numpy as np
import pandas as pd
import polars as pl
import polars.functions as plr

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

    class PolarsSeries(pl.Series):
        def __init__(self, data, name=None, *args, **kwargs):
            # We use the *args, **kwargs construction to ignore arguments incompatible with the Pandas signature
            super().__init__(name=name, values=data)

        def __repr__(self):
            return repr(self)

        def isnull(self):
            return self.is_null()


    class PolarsDataFrame(pl.DataFrame):
        """Override of polars.DataFrame with pandas-like methods"""

        def __init__(self, data=None, columns=None):
            """
            Allows creating the DataFrame from:
             - A dictionary: {column: data}
             - A list: [data_col1, data_col2, ...] (columns must be specified)
             - Instances of pl.DataFrame are also accepted (data is extracted)
            """
            super().__init__(data)
            self._series = {}  # Dictionary of columns: name -> PolarsSeries

            if data is None:
                # Empty DataFrame
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
                    # If columns are not specified, generate default names
                    columns = [f"col{i}" for i in range(len(data))]
                for col_name, col_data in zip(columns, data):
                    self._series[col_name] = PolarsSeries(col_data, name=col_name)
            elif isinstance(data, pl.DataFrame):
                # Extract data from each column
                for col in data.columns:
                    self._series[col] = PolarsSeries(data[col].to_list(), name=col)
            else:
                raise ValueError("Unsupported data type for creating PolarsDataFrame.")

            self._build_df()

        def _build_df(self):
            """
            Rebuilds the internal polars DataFrame from the stored series.
            """
            d = {col: series.to_list() for col, series in self._series.items()}
            self._df = pl.DataFrame(d)

        def __getitem__(self, key):
            """
            Accessing a column always returns the associated PolarsSeries.
            """
            if isinstance(key, str):
                return self._series[key]
            elif isinstance(key, list):
                # If multiple columns are requested, create a new PolarsDataFrame
                new_data = {col: self._series[col].to_list() for col in key if col in self._series}
                return PolarsDataFrame(new_data)
            else:
                raise KeyError("Unsupported index type.")

        def __setitem__(self, key, value):
            """
            When assigning a column, it is forced to be PolarsSeries.
            """
            if not isinstance(value, PolarsSeries):
                value = PolarsSeries(value, name=key)
            self._series[key] = value
            self._build_df()

        def __repr__(self):
            return repr(self)

        class _ColumnsWrapper:
            """Internal class to override `columns` and add `.tolist()` method."""

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
            """
            Replaces missing values (None) in the DataFrame.

            If 'value' is a scalar, replaces all None with that value.
            If it is a dictionary, replaces each column with the specified value.
            Returns a new PolarsDataFrame with the changes.
            """
            new_series = {}
            if isinstance(value, dict):
                for col, series in self._series.items():
                    if col in value:
                        # Replace None with the specified value for that column
                        new_data = [value[col] if x is None else x for x in series.data]
                    else:
                        new_data = series.data
                    new_series[col] = PolarsSeries(new_data, name=col)
            else:
                for col, series in self._series.items():
                    new_data = [value if x is None else x for x in series.data]
                    new_series[col] = PolarsSeries(new_data, name=col)
            return PolarsDataFrame(new_series)


        def replace(self, to_replace, value=None, **kwargs):
            """
            Replaces values in the DataFrame similar to pandas.DataFrame.replace.

            Supports:
             - Scalar value replacement globally.
             - Replacement using a dictionary: {value_to_replace: new_value, ...}.

            Note: This implementation is basic and can be extended to cover more complex cases.
            """
            # If to_replace is a dictionary, apply replacements sequentially.
            if isinstance(to_replace, dict):
                df_temp = self
                for old, new in to_replace.items():
                    df_temp = df_temp.replace(old, new, **kwargs)
                return df_temp

            # In the simplest case: to_replace is a scalar value.
            new_cols = []
            for col in self.columns.tolist():
                new_col = pl.when(pl.col(col) == to_replace).then(value).otherwise(pl.col(col)).alias(col)
                new_cols.append(new_col)
            return self.with_columns(new_cols)




    def polars_read_csv(filepath, *args, **kwargs):
        return PolarsDataFrame(pl.read_csv(filepath))

    _DataFrame = PolarsDataFrame
    _Series = PolarsSeries

    _concat = pd.concat
    _isnull = pd.isnull
    _isna = pd.isna
    _merge = pd.merge
    _read_csv = polars_read_csv

else:
    raise ValueError("Invalid value for BACKEND_DF environment variable. Use 'pd' or 'pl'.")



DataFrame = _DataFrame
Series = _Series

concat = _concat
isnull = _isnull
isna = _isna
merge = _merge
read_csv = _read_csv
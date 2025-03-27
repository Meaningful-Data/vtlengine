import os
from typing import Callable

import pandas as pd
import polars as pl
from polars import DataFrame

POLARS_STR = ["polars", "pl"]

backend_df = "pl" if os.getenv("BACKEND_DF", "").lower() in POLARS_STR else "pd"

if backend_df == "pd":
    class DataFrame(pd.DataFrame):
        pass


    class Series(pd.Series):
        pass


    # isnull = pd.isnull


elif backend_df == "pl":
    class DataFrame(pl.DataFrame):
        def __init__(self, data=None, columns=None):
            if isinstance(data, dict):
                super().__init__(data)
            elif isinstance(data, list):
                if columns is None:
                    columns = [f"col{i}" for i in enumerate(data)]
                super().__init__({columns[i]: col_data for i, col_data in enumerate(data)})
            elif data is not None:
                super().__init__(data)
            elif data is None and columns is not None:
                super().__init__({col: [] for col in columns})
            else:
                super().__init__({})

        def __repr__(self):
            return repr(self)

        def __setitem__(self, key, value):
            """Allowing assigns like df[col] = series in Polars dfs."""
            if isinstance(value, pl.Series):
                # Replace using with_columns
                new_df = self.with_columns(value.alias(key))
                self._df = new_df._df
            else:
                raise TypeError(f"Cannot assign value {type(value)} to a column")

        def all(self):
            self.all()

        def copy(self):
            return self.clone()

        def isnull(self):
            return self.all().is_null()

        def isin(self, values):
            return self.all().is_in(values)

        def rename(self, columns, strict: bool = True) -> DataFrame:
            return pl.DataFrame.rename(self=self, mapping=columns, strict=strict)
        def to_csv(self, path):
            self.write_csv(path)

        def to_pandas(self, **kwargs):
            return self.to_pandas()

        @property
        def T(self):
            return self.transpose()


    # class Series(pl.Series):
    #     type = pl.Series
    #     def __init__(self, data, name=None, *args, **kwargs):
    #         # We use the *args, **kwargs construction to ignore arguments incompatible with the Pandas signature
    #         super().__init__(name=name, values=data)
    #
    #
    #     def __repr__(self):
    #         return repr(self)
    #
    #     def align(self, other, join="outer", fill_value=None):
    #         # Polars do not have any Pandas.Series align-like method, so we must to convert first to pandas
    #         s1, s2 = self.to_pandas().align(other.to_pandas(), join=join, fill_value=fill_value)
    #         # Back to polars
    #         return Series(s1.name, s1), Series(s2.name, s2)
    #
    #     def isnull(self):
    #         return self.is_null()
    #
    #     def isin(self, values):
    #         return self.is_in(values)
    #
    #     def map(self, *args, **kwargs):
    #         return self.map_elements(*args, **kwargs)
    #
    #     def to_csv(self, path):
    #         self.to_csv(path)
    #
    #     @property
    #     def values(self):
    #         return self.to_numpy()


    class Series(pd.Series):
        pass


else:
    raise ValueError("Invalid value for BACKEND_DF environment variable. Use 'pd' or 'pl'.")


isnull = pd.isnull

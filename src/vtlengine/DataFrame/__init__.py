import os
import pandas as pd
from .Polars import PolarsDataFrame, PolarsSeries, handle_dtype, _assert_frame_equal as polars_assert_frame_equal, \
    _concat as polars_concat, _isnull as polars_isnull, _isna as polars_isna, _merge as polars_merge, \
    _read_csv as polars_read_csv

from pandas._testing import assert_frame_equal as pandas_assert_frame_equal

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

    _DataFrame = PolarsDataFrame
    _Series = PolarsSeries

    _assert_frame_equal = polars_assert_frame_equal
    _concat = polars_concat
    _isnull = polars_isnull
    _isna = polars_isna
    _merge = polars_merge
    _read_csv = polars_read_csv


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

import duckdb
import pandas as pd

con = duckdb.connect()


def _assert_frame_equal():
    pass


def _concat():
    pass


def _handle_dtype():
    pass


def _infer_dtype():
    pass


def _isna():
    pass


def _isnull(obj):
    if isinstance(obj, (int, float, str, bool)):
        return pd.isnull(obj)


def _merge():
    pass


def _read_csv(file_path: str, **kwargs):
    return con.from_csv_auto(str(file_path))

def _to_datetime():
    pass

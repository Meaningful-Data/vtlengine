"""pandas helpers shared by the pandas engine's operators."""

from typing import Any

import pandas as pd


def _with_arrow_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """Give every chunkless Arrow-backed column of an empty frame its one empty chunk."""
    fixed = df
    for name in df.columns:
        column = df[name]
        pa_array = getattr(column.array, "_pa_array", None)
        if pa_array is not None and pa_array.num_chunks == 0:
            if fixed is df:
                fixed = df.copy()
            fixed[name] = column.astype(object).astype(column.dtype)
    return fixed


def merge_frames(left: pd.DataFrame, right: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    """``pd.merge`` that also works when both operands are empty on pandas 3.

    Casting an empty Arrow-backed column (``astype`` on the empty result of a filter,
    of a data load...) leaves it without any chunk. When such a column is one of
    several keys of an outer merge and both operands are empty, pandas 3 concatenates
    the key chunks of both sides (``pa.chunked_array(lk.chunks + rk.chunks)``) and
    pyarrow refuses an empty list with no type: "cannot construct ChunkedArray from
    empty vector and omitted type". pandas 2 factorizes those keys another way.
    Rebuilding the chunkless columns of two empty operands restores the single empty
    chunk pyarrow needs; nothing else changes.
    """
    if len(left) == 0 and len(right) == 0:
        left = _with_arrow_chunks(left)
        right = _with_arrow_chunks(right)
    return pd.merge(left, right, **kwargs)

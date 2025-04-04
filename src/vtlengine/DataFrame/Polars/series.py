from typing import Any, Collection

import numpy as np
import polars as pl
from polars import String
from polars.datatypes import IntegerType as PolarsIntegerType
from polars.series.plotting import SeriesPlot

from vtlengine.DataFrame.Polars.utils import Index, polars_dtype_mapping


class PolarsSeries(pl.Series):
    _index: Index = Index()

    def __init__(self, data=None, name=None, index=None, **kwargs):
        if data is None:
            data = []
        if isinstance(data, range):
            data = list(data)
        if not isinstance(data, (list, tuple, np.ndarray, pl.Series)):
            data = [data]
        if len(data) > 0 and isinstance(data[0], list):
            data = data[0]
        if index is not None:
            self.index = index
        else:
            self.index = Index(len(data))
        super().__init__(name=name, values=data, strict=False)

    def __getitem__(self, key):
        if isinstance(key, range):
            key = list(key)
        if isinstance(key, (PolarsSeries, Index)):
            key = key.to_list()

        if isinstance(key, (int, slice)):
            return self.to_list()[key]
        elif isinstance(key, list):
            if len(key) and isinstance(key[0], bool):
                index_from_trues = [i for i, mask in enumerate(key) if mask]
                return PolarsSeries(self.filter(key), index=index_from_trues)
            return self.gather(key)
        raise TypeError(f"Invalid index type {type(key)} for __getitem__")

    def __repr__(self):
        return super().__repr__()

    def _repr_html_(self):
        return super()._repr_html_()

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, s):
        self._s = s

    @property
    def dtype(self):
        return self.s.dtype()

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value

    @property
    def empty(self):
        return self.__len__() == 0

    @property
    def iloc(self):
        # return self.iLocIndexer(self)
        return self

    @property
    def loc(self):
        # return self.LocIndexer(self)
        return self

    @property
    def name(self):
        return self.s.name()

    @property
    def plot(self) -> SeriesPlot:
        return SeriesPlot(self)

    @property
    def values(self):
        return self.to_list()

    # TODO: check if this is the correct implementation
    def align(self, other):
        return self, other

    def apply(self, func, *args, **kwargs):
        new_series = [func(x, *args, **kwargs) for x in self.to_list()]
        return PolarsSeries(new_series, name=self.name)

    def astype(self, dtype, errors="raise"):
        try:
            if dtype != self.dtype and dtype != np.object_:
                dtype = polars_dtype_mapping.get(dtype, dtype)
                if issubclass(dtype, PolarsIntegerType) and self.dtype == String:
                    return self.cast(pl.Float64).cast(dtype)
                return self.cast(dtype)
            return self
        except Exception as e:
            if errors == "raise":
                raise e
            return self

    def combine(self, other, func):
        return PolarsSeries(
            [func(x, y) for x, y in zip(self.to_list(), other.to_list())], name=self.name
        )

    def copy(self):
        return PolarsSeries(self.to_list(), name=self.name)

    def cumsum(self, **kwargs) -> "PolarsSeries":
        cumsum_values = super().cum_sum(**kwargs)
        return PolarsSeries(cumsum_values, name=self.name)

    def diff(self, **kwargs) -> "PolarsSeries":
        """Calculate the difference between consecutive elements in the series."""
        diff_values = super().diff(**kwargs)
        return PolarsSeries(diff_values, name=self.name)

    def dropna(self):
        return PolarsSeries(self.drop_nulls(), name=self.name)

    def isin(self, other: Collection[Any], **kwargs) -> "PolarsSeries":
        return PolarsSeries(self.is_in(other), name=self.name)

    def isnull(self):
        return PolarsSeries(self.is_null(), name=self.name)

    def fillna(self, value, *args, **kwargs):
        return PolarsSeries(self.fill_null(value), name=self.name)

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

    def reindex(self, value=None, **kwargs):
        self.index.reindex(value, **kwargs)
        return self

    def reset_index(self, value=None, **kwargs):
        self.index.reindex(value, **kwargs)
        return self

    def sort_values(self, descending=True, **kwargs):
        return PolarsSeries(self.sort(), name=self.name)

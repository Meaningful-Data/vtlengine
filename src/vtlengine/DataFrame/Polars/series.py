import numpy as np
import polars as pl
from polars import String
from polars.datatypes import IntegerType as PolarsIntegerType
from polars.series.plotting import SeriesPlot

from vtlengine.DataFrame.Polars.utils import polars_dtype_mapping


class PolarsSeries(pl.Series):
    def __init__(self, data=None, name=None, **kwargs):
        if not isinstance(data, (list, tuple, np.ndarray, pl.Series)):
            data = [data]
        if len(data) > 0 and isinstance(data[0], list):
            data = data[0]
        super().__init__(name=name, values=data, strict=False)

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return self.to_list()[index]
        if isinstance(index, PolarsSeries):
            index = index.to_list()
        if isinstance(index, list) and all(isinstance(i, bool) for i in index):
            return self.loc_by_mask(index)
        else:
            raise TypeError("Invalid index type for __getitem__")

    def __repr__(self):
        return super().__repr__()

    def _repr_html_(self):
        return super()._repr_html_()

    class iLocIndexer:
        def __init__(self, series):
            self.series = series

        def __getitem__(self, index):
            return self.series.to_list()[index]

    class LocIndexer:
        """"""

        def __init__(self, series):
            self.series = series

        def __getitem__(self, index):
            if isinstance(index, int) or isinstance(index, slice):
                return self.series.to_list()[index]
            elif isinstance(index, list):
                return [self.series.to_list()[i] for i in index]
            else:
                raise TypeError("Invalid index type for loc")

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
        return range(len(self))

    @property
    def empty(self):
        return self.__len__() == 0

    @property
    def iloc(self) -> iLocIndexer:
        return self.iLocIndexer(self)

    @property
    def loc(self) -> LocIndexer:
        return self.LocIndexer(self)

    @property
    def name(self):
        return self.s.name()

    @property
    def plot(self) -> SeriesPlot:
        return SeriesPlot(self)

    @property
    def values(self):
        return self.to_list()

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

    def isnull(self):
        return PolarsSeries(self.is_null(), name=self.name)

    def fillna(self, value, *args, **kwargs):
        return self.fill_null(value)

    def loc_by_mask(self, boolean_mask):
        if len(boolean_mask) != len(self):
            raise ValueError("Boolean mask length must match the length of the series")
        return PolarsSeries(
            [x for x, mask in zip(self.to_list(), boolean_mask) if mask], name=self.name
        )

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

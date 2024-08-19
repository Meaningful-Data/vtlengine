import os

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from typing import Optional, Any

from AST.Grammar.tokens import AND, OR, XOR, NOT
from DataTypes import Boolean
import Operators as Operator


class Unary(Operator.Unary):
    type_to_check = Boolean
    return_type = Boolean


class Binary(Operator.Binary):
    type_to_check = Boolean
    return_type = Boolean

    @classmethod
    def apply_operation_series_scalar(
        cls, series: pd.Series, scalar: Any, series_left: bool
    ) -> Any:
        if series_left:
            return series.map(lambda x: cls.py_op(x, scalar))
        else:
            return series.map(lambda x: cls.py_op(scalar, x))

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def op_func(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        return cls.py_op(x, y)


class And(Binary):
    op = AND

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        return (
            None
            if (pd.isnull(x) and y is not False) or (pd.isnull(y) and x is not False)
            else x and y
        )

    @classmethod
    def spark_op(cls, x: pd.Series, y: pd.Series) -> pd.Series:
        return x & y


class Or(Binary):
    op = OR

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (pd.isnull(x) and (pd.isnull(y) or y == False)) or (
            pd.isnull(y) and x == False
        ):  # noqa
            return None
        elif pd.isnull(x):
            return True
        return x or y

    @classmethod
    def spark_op(cls, x: pd.Series, y: pd.Series) -> pd.Series:
        return x | y


class Xor(Binary):
    op = XOR

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return (x and not y) or (not x and y)

    @classmethod
    def spark_op(cls, x: pd.Series, y: pd.Series) -> pd.Series:
        return x ^ y


class Not(Unary):
    op = NOT

    @classmethod
    def py_op(cls, x: Optional[bool]) -> Optional[bool]:
        return None if pd.isnull(x) else not x

    @classmethod
    def spark_op(cls, series: pd.Series) -> pd.Series:
        return ~series

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series.map(cls.py_op)

# if os.environ.get("SPARK", False):
#     import pyspark.pandas as pd
# else:
#     import pandas as pd
from typing import Any, Optional

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import AND, NOT, OR, XOR
from vtlengine.DataTypes import Boolean


class Unary(Operator.Unary):
    type_to_check = Boolean
    return_type = Boolean


class Binary(Operator.Binary):
    type_to_check = Boolean
    return_type = Boolean
    comp_op: Any = None

    @classmethod
    def apply_operation_series_scalar(cls, series: Any, scalar: Any, series_left: bool) -> Any:
        if series_left:
            return series.map(lambda x: cls.py_op(x, scalar))
        else:
            return series.map(lambda x: cls.py_op(scalar, x))

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        result = cls.comp_op(left_series.astype("boolean"), right_series.astype("boolean"))
        return result.replace({pd.NA: None}).astype(object)

    @classmethod
    def op_func(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        return cls.py_op(x, y)


class And(Binary):
    op = AND
    comp_op = pd.Series.__and__

    @staticmethod
    # @numba.njit
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (x is None and y == False) or (x == False and y is None):
            return False
        elif x is None or y is None:
            return None
        return x and y

    # @classmethod
    # def spark_op(cls, x: pd.Series, y: pd.Series) -> pd.Series:
    #     return x & y


class Or(Binary):
    op = OR
    comp_op = pd.Series.__or__

    @staticmethod
    # @numba.njit
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (x is None and y == True) or (x == True and y is None):
            return True
        elif x is None or y is None:
            return None
        return x or y

    # @classmethod
    # def spark_op(cls, x: pd.Series, y: pd.Series) -> pd.Series:
    #     return x | y


class Xor(Binary):
    op = XOR
    comp_op = pd.Series.__xor__

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return (x and not y) or (not x and y)

    # @classmethod
    # def spark_op(cls, x: pd.Series, y: pd.Series) -> pd.Series:
    #     return x ^ y


class Not(Unary):
    op = NOT

    @staticmethod
    # @numba.njit
    def py_op(x: Optional[bool]) -> Optional[bool]:
        return None if x is None else not x

    # @classmethod
    # def spark_op(cls, series: pd.Series) -> pd.Series:
    #     return ~series

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series.map(lambda x: not x, na_action="ignore")

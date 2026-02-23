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
            return series.map(lambda x: cls.py_op(x, scalar)).astype("bool[pyarrow]")
        else:
            return series.map(lambda x: cls.py_op(scalar, x)).astype("bool[pyarrow]")

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        result = cls.comp_op(
            left_series.astype("bool[pyarrow]"), right_series.astype("bool[pyarrow]")
        )
        return result

    @classmethod
    def op_func(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        return cls.py_op(x, y)


class And(Binary):
    op = AND
    comp_op = pd.Series.__and__

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        x_na = pd.isna(x)
        y_na = pd.isna(y)
        if (x_na and y is False) or (x is False and y_na):
            return False
        elif x_na or y_na:
            return None
        return x and y


class Or(Binary):
    op = OR
    comp_op = pd.Series.__or__

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        x_na = pd.isna(x)
        y_na = pd.isna(y)
        if (x_na and y is True) or (x is True and y_na):
            return True
        elif x_na or y_na:
            return None
        return x or y


class Xor(Binary):
    op = XOR
    comp_op = pd.Series.__xor__

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return (x and not y) or (not x and y)


class Not(Unary):
    op = NOT

    @staticmethod
    def py_op(x: Optional[bool]) -> Optional[bool]:
        return None if pd.isna(x) else not x

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return ~series.astype("bool[pyarrow]")

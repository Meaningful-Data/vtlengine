from typing import Any, Optional

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import AND, NOT, OR, XOR
from vtlengine.DataTypes import Boolean


class Unary(Operator.Unary):
    type_to_check = Boolean
    return_type = Boolean


class Binary(Operator.Binary):
    type_to_check = Boolean
    return_type = Boolean


class And(Binary):
    op = AND

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        x_na = x is None
        y_na = y is None
        if (x_na and y is False) or (x is False and y_na):
            return False
        elif x_na or y_na:
            return None
        return x and y


class Or(Binary):
    op = OR

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        x_na = x is None
        y_na = y is None
        if (x_na and y is True) or (x is True and y_na):
            return True
        elif x_na or y_na:
            return None
        return x or y


class Xor(Binary):
    op = XOR


class Not(Unary):
    op = NOT

    @staticmethod
    def py_op(x: Optional[bool]) -> Optional[bool]:
        return None if x is None else not x

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return ~series.astype("bool[pyarrow]")

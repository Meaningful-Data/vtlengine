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


class And(Binary):
    op = AND

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (x is None and y == False) or (x == False and y is None):
            return False
        elif x is None or y is None:
            return None
        return x and y


class Or(Binary):
    op = OR

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (x is None and y == True) or (x == True and y is None):
            return True
        elif x is None or y is None:
            return None
        return x or y


class Xor(Binary):
    op = XOR

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return (x and not y) or (not x and y)


class Not(Unary):
    op = NOT

    @staticmethod
    def py_op(x: Optional[bool]) -> Optional[bool]:
        return None if x is None else not x

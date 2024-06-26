import os
if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from typing import Optional, Any

from AST.Grammar.tokens import AND, OR, XOR, NOT
from DataTypes import Boolean
from Operators import Binary, Unary


class And(Binary):
    op = AND

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        return None if (pd.isnull(x) and y is not False) or (pd.isnull(y) and x is not False) else x and y

    @classmethod
    def apply_operation_component(cls, left_series: Any, right_series: Any) -> Any:
        return left_series.combine(right_series, cls.py_op)


class Or(Binary):
    op = OR

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (pd.isnull(x) and (pd.isnull(y) or y == False)) or (pd.isnull(y) and x == False):
            return None
        elif pd.isnull(x):
            return True
        return x or y

    @classmethod
    def apply_operation_component(cls, left_series: Any, right_series: Any) -> Any:
        return left_series.combine(right_series, cls.py_op)

class Xor(Binary):
    op = XOR

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return (x and not y) or (not x and y)

    @classmethod
    def apply_operation_component(cls, left_series: Any, right_series: Any) -> Any:
        return left_series.combine(right_series, cls.py_op)

class Not(Unary):
    type_to_check = Boolean
    op = NOT

    @classmethod
    def py_op(cls, x: Optional[bool]) -> Optional[bool]:
        return None if pd.isnull(x) else not x

    py_op = py_op

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series.map(cls.py_op)
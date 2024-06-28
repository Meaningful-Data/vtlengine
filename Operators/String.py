import operator
import os

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from typing import Optional, Any

from AST.Grammar.tokens import LEN, CONCAT, UCASE, LCASE, RTRIM, SUBSTR, LTRIM, TRIM
from DataTypes import Integer, String
import Operators as Operator


class Unary(Operator.Unary):
    type_to_check = String


class Length(Unary):
    op = LEN
    return_type = Integer
    py_op = len

    @classmethod
    def op_func(cls, x: Any) -> Any:
        result = super().op_func(x)
        if pd.isnull(result):
            return 0
        return result

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        return series.map(cls.op_func)

class Lower(Unary):
    op = LCASE
    py_op = str.lower
    return_type = String

class Upper(Unary):
    op = UCASE
    py_op = str.upper
    return_type = String

class Trim(Unary):
    op = TRIM
    py_op = str.strip
    return_type = String

class Ltrim(Unary):
    op = LTRIM
    py_op = str.lstrip
    return_type = String

class Rtrim(Unary):
    op = RTRIM
    py_op = str.rstrip
    return_type = String



class Binary(Operator.Binary):
    type_to_check = String


class Concatenate(Binary):
    op = CONCAT
    py_op = operator.concat
    return_type = String
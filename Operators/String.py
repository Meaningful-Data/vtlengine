import operator
import os
if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from typing import Optional, Any

from AST.Grammar.tokens import LEN, CONCAT
from DataTypes import Integer, String
from Operators import Unary, Binary


class Length(Unary):
    op = LEN
    return_type = Integer

    @classmethod
    def py_op(cls, x: Optional[String]) -> Optional[Integer]:
        return None if pd.isnull(x) else len(x)

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        #TODO change measure name
        return series.map(cls.py_op)

class Concatenate(Binary):
    op = CONCAT
    py_op = operator.concat
    return_type = String

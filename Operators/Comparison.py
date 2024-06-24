import operator
import os
from dataclasses import dataclass
from typing import Union, List

from AST.Grammar.tokens import EQ
from Exceptions.messages import centralised_messages
from Model import Dataset, DataComponent, Scalar, Component
from Operators import Binary

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd


@dataclass
class Equal(Binary):
    op = EQ
    py_op = operator.eq

    @classmethod
    def interval_func(cls, left_v, left_p, right_v, right_p):
        return abs(left_v - right_v) <= (left_p + right_p)

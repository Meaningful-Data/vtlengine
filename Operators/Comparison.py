import operator
import os

from AST.Grammar.tokens import EQ, GT, GTE, LT, LTE, NEQ
from Operators import Binary

if os.environ.get("SPARK", False):
    pass
else:
    pass


# class IsNull(Unary):
#     op = ISNULL
#     py_op = operator.truth
#     return_type = Boolean

class Equal(Binary):
    op = EQ
    py_op = operator.eq


class NotEqual(Binary):
    op = NEQ
    py_op = operator.ne


class Greater(Binary):
    op = GT
    py_op = operator.gt


class GreaterEqual(Binary):
    op = GTE
    py_op = operator.ge


class Less(Binary):
    op = LT
    py_op = operator.lt


class LessEqual(Binary):
    op = LTE
    py_op = operator.le

# class In(Binary):
#     op = IN
#
#     @classmethod
#     def py_op(cls, x, y):
#         return operator.contains(y, x)
#
#     py_op = py_op


# class Match(Binary):
#     op = CHARSET_MATCH
#     type_to_check = String
#
#     @classmethod
#     def py_op(cls, x, y):
#         if isinstance(x, pd.Series):
#             return x.str.fullmatch(y)
#         return bool(re.fullmatch(y, x))
#
#     py_op = py_op

import math
import operator

from Operators import Operator
from DataTypes.ScalarTypes import Number
from Grammar.tokens import ABS, DIV, EXP, LN, LOG, MINUS, MULT, PLUS, SQRT


class Unary(Operator.Unary):
    op = None
    type_to_check = Number


class UnPlus(Unary):
    op = PLUS
    py_op = operator.pos


class UnMinus(Unary):
    op = MINUS
    py_op = operator.neg


class AbsoluteValue(Unary):
    op = ABS
    py_op = operator.abs


class Exponential(Unary):
    op = EXP
    py_op = math.exp
    return_type = Number


class NaturalLogarithm(Unary):
    op = LN
    py_op = math.log
    return_type = Number


class SquareRoot(Unary):
    op = SQRT
    py_op = math.sqrt
    return_type = Number


class NumericBinary(Operator.Binary):
    type_to_check = Number


class BinPlus(NumericBinary):
    op = PLUS
    py_op = operator.add
    type_to_check = Number


class BinMinus(NumericBinary):
    op = MINUS
    py_op = operator.sub
    type_to_check = Number


class Mult(NumericBinary):
    op = MULT
    py_op = operator.mul


class Div(NumericBinary):
    op = DIV
    py_op = operator.truediv
    return_type = Number


class Power(NumericBinary):
    op = POW
    py_op = operator.pow


class Logarithm(NumericBinary):
    op = LOG
    py_op = math.log
    return_type = Number

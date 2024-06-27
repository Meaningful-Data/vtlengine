import math
import operator
from typing import Any

import Operators as Operator
from AST.Grammar.tokens import ABS, CEIL, DIV, EXP, FLOOR, LN, LOG, MINUS, MOD, MULT, PLUS, POWER, \
    ROUND, \
    SQRT
from DataTypes import Integer, Number
from Model import Dataset


class Unary(Operator.Unary):
    type_to_check = Number


class UnPlus(Unary):
    op = PLUS
    py_op = operator.pos

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series


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


class Ceil(Unary):
    op = CEIL
    py_op = math.ceil
    return_type = Integer


class Floor(Unary):
    op = FLOOR
    py_op = math.floor
    return_type = Integer


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


class Logarithm(NumericBinary):
    op = LOG
    py_op = math.log
    return_type = Number

    @classmethod
    def validate(cls, left_operand, right_operand):
        if isinstance(right_operand, Dataset):
            raise Exception("Logarithm operator base cannot be a Dataset")
        return super().validate(left_operand, right_operand)


class Modulo(NumericBinary):
    op = MOD
    py_op = operator.mod

    @classmethod
    def validate(cls, left_operand, right_operand):
        if isinstance(right_operand, Dataset):
            raise Exception("Modulo operator divisor cannot be a Dataset")
        return super().validate(left_operand, right_operand)

class Power(NumericBinary):
    op = POWER
    py_op = operator.pow
    return_type = Number

    @classmethod
    def validate(cls, left_operand, right_operand):
        if isinstance(right_operand, Dataset):
            raise Exception("Power operator exponent cannot be a Dataset")
        return super().validate(left_operand, right_operand)

import math
import operator
from typing import Any, Optional, Union

import pandas as pd

import Operators as Operator
from AST.Grammar.tokens import ABS, CEIL, DIV, EXP, FLOOR, LN, LOG, MINUS, MOD, MULT, PLUS, POWER, \
    ROUND, SQRT, TRUNC
from DataTypes import Integer, Number
from Model import DataComponent, Dataset, Scalar
from Operators import ALL_MODEL_DATA_TYPES


class Unary(Operator.Unary):
    type_to_check = Number


class Binary(Operator.Binary):
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


class BinPlus(Binary):
    op = PLUS
    py_op = operator.add
    type_to_check = Number


class BinMinus(Binary):
    op = MINUS
    py_op = operator.sub
    type_to_check = Number


class Mult(Binary):
    op = MULT
    py_op = operator.mul


class Div(Binary):
    op = DIV
    py_op = operator.truediv
    return_type = Number


class Logarithm(Binary):
    op = LOG
    return_type = Number

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        if pd.isnull(param):
            return None
        return math.log(x, param)

    @classmethod
    def dataset_validation(cls, left_operand, right_operand):
        raise Exception("Logarithm operator base cannot be a Dataset")


class Modulo(Binary):
    op = MOD
    py_op = operator.mod


class Power(Binary):
    op = POWER
    return_type = Number

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        if pd.isnull(param):
            return None
        return x ** param

    @classmethod
    def dataset_validation(cls, left_operand: Dataset, right_operand: Dataset):
        raise Exception("Power operator exponent cannot be a Dataset")


class Parameterized(Unary):

    @classmethod
    def validate(cls, operand: Operator.ALL_MODEL_DATA_TYPES,
                 param: Optional[Union[DataComponent, Scalar]] = None):

        if param is not None:
            if isinstance(param, Dataset):
                raise Exception(f"{cls.op} cannot have a Dataset as parameter")
            if isinstance(param, DataComponent):
                if isinstance(operand, Scalar):
                    raise Exception(f"{cls.op} cannot have an Scalar operand and "
                                    f"a DataComponent as parameter")
                cls.validate_component_type(param)
            else:
                cls.validate_scalar_type(param)
        if param is None:
            cls.return_type = Integer
        else:
            cls.return_type = Number

        return super().validate(operand)

    @classmethod
    def op_func(cls, x: Any, param: Optional[Any]) -> Any:
        return None if pd.isnull(x) else cls.py_op(x, param)

    @classmethod
    def apply_operation_two_series(cls, left_series: pd.Series, right_series: pd.Series) -> Any:
        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def apply_operation_series_scalar(cls, series: pd.Series, param: Any) -> Any:
        return series.map(lambda x: cls.op_func(x, param))

    @classmethod
    def dataset_evaluation(cls, operand: Dataset, param: Union[DataComponent, Scalar]):
        result = cls.validate(operand, param)
        result.data = operand.data.copy()
        for measure_name in result.get_measures_names():
            if isinstance(param, DataComponent):
                result.data[measure_name] = cls.apply_operation_two_series(
                    result.data[measure_name], param.data
                )
            else:
                param_value = None if param is None else param.value
                result.data[measure_name] = cls.apply_operation_series_scalar(
                    result.data[measure_name], param_value
                )
        return result

    @classmethod
    def component_evaluation(cls, operand: DataComponent, param: Union[DataComponent, Scalar]):
        result = cls.validate(operand, param)
        result.data = operand.data.copy()
        if isinstance(param, DataComponent):
            result.data = cls.apply_operation_two_series(operand.data, param.data)
        else:
            param_value = None if param is None else param.value
            result.data = cls.apply_operation_series_scalar(operand.data, param_value)
        return result

    @classmethod
    def scalar_evaluation(cls, operand: Scalar, param: Scalar):
        result = cls.validate(operand, param)
        param_value = None if param is None else param.value
        result.value = cls.op_func(operand.value, param_value)
        return result

    @classmethod
    def evaluate(cls, operand: ALL_MODEL_DATA_TYPES,
                 param: Optional[Union[DataComponent, Scalar]] = None) -> ALL_MODEL_DATA_TYPES:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, param)


class Round(Parameterized):
    op = ROUND
    return_type = Integer

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        multiplier = 1.0
        if not pd.isnull(param):
            multiplier = 10 ** param

        if x >= 0.0:
            rounded_value = math.floor(x * multiplier + 0.5) / multiplier
        else:
            rounded_value = math.ceil(x * multiplier - 0.5) / multiplier

        if param is not None:
            return rounded_value

        return int(rounded_value)


class Trunc(Parameterized):
    op = TRUNC

    @classmethod
    def py_op(cls, x: float, param: Optional[float]) -> Any:
        multiplier = 1.0
        if not pd.isnull(param):
            multiplier = 10 ** param

        truncated_value = int(x * multiplier) / multiplier

        if not pd.isnull(param):
            return truncated_value

        return int(truncated_value)

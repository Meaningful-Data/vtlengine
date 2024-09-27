import math
import operator
from decimal import getcontext, Decimal
from typing import Any, Optional, Union

import vtlengine.Operators as Operator
import pandas as pd
from vtlengine.DataTypes import Integer, Number
from vtlengine.Operators import ALL_MODEL_DATA_TYPES

from vtlengine.AST.Grammar.tokens import ABS, CEIL, DIV, EXP, FLOOR, LN, LOG, MINUS, MOD, MULT, \
    PLUS, POWER, \
    ROUND, SQRT, TRUNC
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Scalar


class Unary(Operator.Unary):
    """
    Checks that the unary operation is performed with a number.
    """
    type_to_check = Number


class Binary(Operator.Binary):
    """
    Checks that the binary operation is performed with numbers.
    """
    type_to_check = Number

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        if isinstance(x, int) and isinstance(y, int):
            if cls.op == DIV and y == 0:
                raise SemanticError("2-1-15-6", op=cls.op, value=y)
            return cls.py_op(x, y)
        x = float(x)
        y = float(y)
        # Handles precision to avoid floating point errors
        if cls.op == DIV and y == 0:
            raise SemanticError("2-1-15-6", op=cls.op, value=y)

        decimal_value = cls.py_op(Decimal(x), Decimal(y))
        getcontext().prec = 10
        result = float(decimal_value)
        if result.is_integer():
            return int(result)
        return result


class UnPlus(Unary):
    """
    `Plus <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=94&zoom=100,72,142> `_ unary operator
    """
    op = PLUS
    py_op = operator.pos

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series


class UnMinus(Unary):
    """
    `Minus <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=95&zoom=100,72,414> `_unary operator
    """
    op = MINUS
    py_op = operator.neg


class AbsoluteValue(Unary):
    """
    `Absolute <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=112&zoom=100,72,801> `_ unary operator
    """
    op = ABS
    py_op = operator.abs


class Exponential(Unary):
    """
    `Exponential <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=114&zoom=100,72,94>`_ unary operator
    """
    op = EXP
    py_op = math.exp
    return_type = Number


class NaturalLogarithm(Unary):
    """
    `Natural logarithm <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=115&zoom=100,72,394> `_
    unary operator
    """
    op = LN
    py_op = math.log
    return_type = Number


class SquareRoot(Unary):
    """
    `Square Root <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=119&zoom=100,72,556> '_
    unary operator
    """
    op = SQRT
    py_op = math.sqrt
    return_type = Number


class Ceil(Unary):
    """
    `Ceilling <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=110&zoom=100,72,94> `_ unary operator
    """
    op = CEIL
    py_op = math.ceil
    return_type = Integer


class Floor(Unary):
    """
    `Floor <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=111&zoom=100,72,442> `_ unary operator
    """
    op = FLOOR
    py_op = math.floor
    return_type = Integer


class BinPlus(Binary):
    """
    `Addition <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=96&zoom=100,72,692> `_ binary operator
    """
    op = PLUS
    py_op = operator.add
    type_to_check = Number


class BinMinus(Binary):
    """
    `Subtraction <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=98&zoom=100,72,448> `_ binary operator
    """
    op = MINUS
    py_op = operator.sub
    type_to_check = Number


class Mult(Binary):
    """
    `Multiplication <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=100&zoom=100,72,254>`_
    binary operator
    """
    op = MULT
    py_op = operator.mul


class Div(Binary):
    """
    `Division <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=102&zoom=100,72,94>`_
    binary operator
    """
    op = DIV
    py_op = operator.truediv
    return_type = Number


class Logarithm(Binary):
    """
    `Logarithm <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=118&zoom=100,72,228>`_ operator
    """
    op = LOG
    return_type = Number

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        if pd.isnull(param):
            return None
        if param <= 0:
            raise SemanticError("2-1-15-3", op=cls.op, value=param)

        return math.log(x, param)


class Modulo(Binary):
    """
    `Module <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=104&zoom=100,72,94>`_ operator
    """
    op = MOD
    py_op = operator.mod


class Power(Binary):
    """
    `Power <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=116&zoom=100,72,693>`_ operator
    """
    op = POWER
    return_type = Number

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        if pd.isnull(param):
            return None
        return x ** param


class Parameterized(Unary):
    """Parametrized class
        Inherits from Unary class, to validate the data type and evaluate if it is the correct one to
        perform the operation. Similar to Unary, but in the end, the param validation is added.
    """

    @classmethod
    def validate(cls, operand: Operator.ALL_MODEL_DATA_TYPES,
                 param: Optional[Union[DataComponent, Scalar]] = None):

        if param is not None:
            if isinstance(param, Dataset):
                raise SemanticError("1-1-15-8", op=cls.op, comp_type="Dataset")
            if isinstance(param, DataComponent):
                if isinstance(operand, Scalar):
                    raise SemanticError("1-1-15-8", op=cls.op,
                                        comp_type="DataComponent and an Scalar operand")
                cls.validate_type_compatibility(param.data_type)
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
            try:
                if isinstance(param, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param.data
                    )
                else:
                    param_value = None if param is None else param.value
                    result.data[measure_name] = cls.apply_operation_series_scalar(
                        result.data[measure_name], param_value
                    )
            except ValueError:
                raise SemanticError("2-1-15-1", op=cls.op, comp_name=measure_name,
                                    dataset_name=operand.name) from None
        result.data = result.data[result.get_components_names()]
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
    """
    `Round <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=106&zoom=100,72,94>`_ operator
    """
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
    """
    `Trunc <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=108&zoom=100,72,94>`_ operator.
    """
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

import math
import operator
import warnings
from decimal import Decimal, getcontext
from typing import Any, Optional, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    ABS,
    CEIL,
    DIV,
    EXP,
    FLOOR,
    LN,
    LOG,
    MINUS,
    MOD,
    MULT,
    PLUS,
    POWER,
    RANDOM,
    ROUND,
    SQRT,
    TRUNC,
)
from vtlengine.DataTypes import Integer, Number, binary_implicit_promotion
from vtlengine.duckdb.custom_functions.Numeric import random_duck, round_duck, trunc_duck
from vtlengine.duckdb.duckdb_utils import duckdb_concat, empty_relation
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Scalar
from vtlengine.Operators import ALL_MODEL_DATA_TYPES


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
            if cls.op == RANDOM:
                return cls.py_op(x, y)
        x = float(x)
        y = float(y)
        # Handles precision to avoid floating point errors
        if cls.op == DIV and y == 0:
            raise SemanticError("2-1-15-6", op=cls.op, value=y)

        getcontext().prec = 10
        decimal_value = cls.py_op(Decimal(x), Decimal(y))
        result = float(decimal_value)
        if result.is_integer():
            return int(result)
        return result


class UnPlus(Unary):
    """
    `Plus <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=94&zoom=100,72,142> `_ unary operator
    """  # noqa E501

    op = PLUS
    py_op = operator.pos

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        return series


class UnMinus(Unary):
    """
    `Minus <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=95&zoom=100,72,414> `_unary operator
    """  # noqa E501

    op = MINUS
    py_op = operator.neg


class AbsoluteValue(Unary):
    """
    `Absolute <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=112&zoom=100,72,801> `_ unary operator
    """  # noqa E501

    op = ABS
    py_op = operator.abs


class Exponential(Unary):
    """
    `Exponential <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=114&zoom=100,72,94>`_ unary operator
    """  # noqa E501

    op = EXP
    py_op = math.exp
    return_type = Number


class NaturalLogarithm(Unary):
    """
    `Natural logarithm <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=115&zoom=100,72,394> `_
    unary operator
    """  # noqa E501

    op = LN
    py_op = math.log
    return_type = Number


class SquareRoot(Unary):
    """
    `Square Root <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=119&zoom=100,72,556> '_
    unary operator
    """  # noqa E501

    op = SQRT
    py_op = math.sqrt
    return_type = Number


class Ceil(Unary):
    """
    `Ceilling <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=110&zoom=100,72,94> `_ unary operator
    """  # noqa E501

    op = CEIL
    py_op = math.ceil
    return_type = Integer


class Floor(Unary):
    """
    `Floor <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=111&zoom=100,72,442> `_ unary operator
    """  # noqa E501

    op = FLOOR
    py_op = math.floor
    return_type = Integer


class BinPlus(Binary):
    """
    `Addition <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=96&zoom=100,72,692> `_ binary operator
    """  # noqa E501

    op = PLUS
    py_op = operator.add
    type_to_check = Number


class BinMinus(Binary):
    """
    `Subtraction <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=98&zoom=100,72,448> `_ binary operator
    """  # noqa E501

    op = MINUS
    py_op = operator.sub
    type_to_check = Number


class Mult(Binary):
    """
    `Multiplication <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=100&zoom=100,72,254>`_
    binary operator
    """  # noqa E501

    op = MULT
    py_op = operator.mul


class Div(Binary):
    """
    `Division <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=102&zoom=100,72,94>`_
    binary operator
    """  # noqa E501

    op = DIV
    py_op = operator.truediv
    return_type = Number


class Logarithm(Binary):
    """
    `Logarithm <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=118&zoom=100,72,228>`_ operator
    """  # noqa E501

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
    """  # noqa E501

    op = MOD
    py_op = operator.mod


class Power(Binary):
    """
    `Power <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=116&zoom=100,72,693>`_ operator
    """  # noqa E501

    op = POWER
    return_type = Number

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        if pd.isnull(param):
            return None
        return x**param


class Parameterized(Unary):
    """Parametrized class
    Inherits from Unary class, to validate the data type and evaluate if it is the correct one to
    perform the operation. Similar to Unary, but in the end, the param validation is added.
    """

    sql_op: str = Unary.op

    @classmethod
    def apply_parametrized_op(
        cls, me_name: str, param_name: Union[str, int], output_column_name: Any
    ) -> str:
        return f'{cls.sql_op}({me_name}, {param_name}) AS "{output_column_name}"'

    @classmethod
    def validate(
        cls,
        operand: Operator.ALL_MODEL_DATA_TYPES,
        param: Optional[Union[DataComponent, Scalar]] = None,
    ) -> Any:
        if param is not None:
            if isinstance(param, Dataset):
                raise SemanticError("1-1-15-8", op=cls.op, comp_type="Dataset")
            if isinstance(param, DataComponent):
                if isinstance(operand, Scalar):
                    raise SemanticError(
                        "1-1-15-8",
                        op=cls.op,
                        comp_type="DataComponent and an Scalar operand",
                    )
                cls.validate_type_compatibility(param.data_type)
            else:
                cls.validate_scalar_type(param)
        if param is None:
            cls.return_type = Integer
        else:
            cls.return_type = Number

        return super().validate(operand)

    @classmethod
    def op_func(cls, x: Optional[Union[int, float]], param: Optional[Any]) -> Any:
        return None if x is None else cls.py_op(x, param)

    @classmethod
    def dataset_evaluation(
        cls, operand: Dataset, param: Optional[Union[DataComponent, Scalar]] = None
    ) -> Dataset:
        result_dataset = cls.validate(operand, param)
        result_data = operand.data if operand.data is not None else empty_relation()
        exprs = [f'"{d}"' for d in operand.get_identifiers_names()]
        if param is None:
            param_value = "NULL"
        else:
            param_value = param.name if isinstance(param, DataComponent) else param.value
        for measure_name in operand.get_measures_names():
            exprs.append(cls.apply_parametrized_op(measure_name, param_value, measure_name))

        result_dataset.data = result_data.project(", ".join(exprs))
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def component_evaluation(
        cls,
        operand: DataComponent,
        param: Optional[Union[DataComponent, Scalar]] = None,
    ) -> DataComponent:
        result_component = cls.validate(operand, param)
        result_data = operand.data if operand.data is not None else empty_relation()
        if isinstance(param, DataComponent):
            operand_data = duckdb_concat(operand.data, param.data)
            result_component.data = operand_data.project(
                cls.apply_parametrized_op(operand.name, param.name, result_component.name)
            )
        else:
            param_value = "NULL" if param is None else param.value
            result_component.data = result_data.project(
                cls.apply_parametrized_op(operand.name, param_value, result_component.name)
            )
        return result_component

    @classmethod
    def scalar_evaluation(cls, operand: Scalar, param: Optional[Any] = None) -> Scalar:
        result = cls.validate(operand, param)
        param_value = param.value if param is not None else None
        result.value = cls.op_func(operand.value, param_value)
        return result

    @classmethod
    def evaluate(
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        param: Optional[Union[DataComponent, Scalar]] = None,
    ) -> Union[DataComponent, Dataset, Scalar]:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param)
        elif isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param)
        else:
            return cls.scalar_evaluation(operand, param)


class Round(Parameterized):
    """
    `Round <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=106&zoom=100,72,94>`_ operator
    """  # noqa E501

    op = ROUND
    return_type = Integer
    sql_op = "round_duck"
    py_op = round_duck


class Trunc(Parameterized):
    """
    `Trunc <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=108&zoom=100,72,94>`_ operator.
    """  # noqa E501

    op = TRUNC
    sql_op = "trunc_duck"
    py_op = trunc_duck


class Random(Parameterized):
    op = RANDOM
    return_type = Number
    sql_op = "random_duck"
    py_op = random_duck

    @classmethod
    def validate(cls, seed: Any, index: Any = None) -> Any:
        if index.data_type != Integer:
            index.data_type = binary_implicit_promotion(index.data_type, Integer)
        if index.value < 0:
            raise SemanticError("2-1-15-2", op=cls.op, value=index)
        if index.value > 10000:
            warnings.warn(
                "Random: The value of 'index' is very big. This can affect performance.",
                UserWarning,
            )
        return super().validate(seed, index)

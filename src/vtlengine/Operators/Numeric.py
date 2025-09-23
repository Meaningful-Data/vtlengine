import math
import operator
import os
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
from vtlengine.duckdb.custom_functions.Numeric import (
    division_duck,
    random_duck,
    round_duck,
    trunc_duck,
)
from vtlengine.duckdb.duckdb_utils import duckdb_concat, empty_relation
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Scalar
from vtlengine.Operators import ALL_MODEL_DATA_TYPES

OUTPUT_NUMERIC_FUNCTIONS = [
    LOG,
    POWER,
    DIV,
    PLUS,
    MINUS,
    MULT,
    MOD,
    ROUND,
    CEIL,
    ABS,
    FLOOR,
    EXP,
    LN,
    SQRT,
    "trunc_duck",
    "random_duck",
]
ROUND_VALUE = int(os.getenv("ROUND_VALUE", "8"))


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
        # Handles precision to avoid floating point errors
        if isinstance(x, int) and isinstance(y, int) and cls.op == RANDOM:
            return cls.py_op(x, y)
        x = float(x)
        y = float(y)

        getcontext().prec = 10
        decimal_value = cls.py_op(Decimal(x), Decimal(y))
        result = float(decimal_value)
        result = round(result, ROUND_VALUE)
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
    py_op = division_duck
    sql_op = "division_duck"
    return_type = Number


class Logarithm(Binary):
    """
    `Logarithm <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=118&zoom=100,72,228>`_ operator
    """  # noqa E501

    op = LOG
    return_type = Number

    @classmethod
    def py_op(cls, x: Union[int, float], param: Union[int, float]) -> Optional[float]:
        if param is None:
            return None
        # TODO: change this to a Runtime error
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
        cls, input_column_name: str, param_name: Union[str, int, float], output_column_name: str
    ) -> str:
        """
        Applies the parametrized operation to the operand and returns a SQL expression.

        Args:
            input_column_name (str): The operand to which the operation
              will be applied (name of the column).
            param_name (str): The name of the parameter (or value) to be used in the operation.
            output_column_name (str): The name of the column where we store the result.
        """
        op_token = cls.sql_op
        if op_token in OUTPUT_NUMERIC_FUNCTIONS:
            return (
                f"round({op_token}({input_column_name}, {param_name}), {ROUND_VALUE}) "
                f'AS "{output_column_name}"'
            )
        return f'{op_token}({input_column_name}, {param_name}) AS "{output_column_name}"'

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

    # TODO refactor this to utils.py
    @staticmethod
    def handle_param_value(param: Optional[Union[DataComponent, Scalar]]) -> Union[str, int, float]:
        if isinstance(param, DataComponent):
            return f'"{param.name}"'
        elif isinstance(param, Scalar) and param.value is not None:
            return param.value
        return "NULL"

    @classmethod
    def dataset_evaluation(
        cls, operand: Dataset, param: Optional[Union[DataComponent, Scalar]] = None
    ) -> Dataset:
        result_dataset = cls.validate(operand, param)
        result_data = operand.data if operand.data is not None else empty_relation()
        exprs = [f'"{d}"' for d in operand.get_identifiers_names()]
        param_value = cls.handle_param_value(param)
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
        param_value = cls.handle_param_value(param)
        if isinstance(param, DataComponent):
            result_data = duckdb_concat(operand.data, param.data)
        result_component.data = result_data.project(
            cls.apply_parametrized_op(operand.name, param_value, result_component.name)
        )
        return result_component

    @classmethod
    def scalar_evaluation(cls, operand: Scalar, param: Optional[Any] = None) -> Scalar:
        result = cls.validate(operand, param)
        param_value = None if param is None or param.value is None else param.value
        result.value = cls.py_op(operand.value, param_value)
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

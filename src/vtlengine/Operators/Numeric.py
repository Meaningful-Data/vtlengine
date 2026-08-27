import _random
import math
import operator
import warnings
from typing import Any, Optional, Union

import pandas as pd
import pyarrow.compute as pc

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
from vtlengine.DataTypes import SCALAR_TYPES_CLASS_REVERSE, Integer, Number
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Scalar
from vtlengine.Operators import ALL_MODEL_DATA_TYPES
from vtlengine.Utils._number_config import get_effective_numeric_digits

# Binary arithmetic whose results are rounded to the configured significant
# digits — the same set the DuckDB transpiler wraps in vtl_round_sig (issue #985).
_ROUNDED_NUMERIC_OPS = frozenset({PLUS, MINUS, MULT, DIV, MOD, POWER})


def _round_to_significant(value: float, digits: Optional[int]) -> float:
    """Round a float to the configured significant digits (round-half-even),
    the same normalization vtl_round_sig applies in the DuckDB engine."""
    if digits is None or math.isnan(value) or math.isinf(value):
        return value
    return float(f"{value:.{digits}g}")


class Unary(Operator.Unary):
    """
    Checks that the unary operation is performed with a number.
    """

    type_to_check = Number
    pc_func: Any = None

    @classmethod
    def _check_domain(cls, series: Any) -> None:
        """Hook for operators with a restricted domain (sqrt, ln).

        The pyarrow fast path would silently produce NaN for out-of-domain
        values, so subclasses raise the VTL error here first (issue #985).
        """

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        cls._check_domain(series)
        if cls.pc_func is not None and isinstance(
            series.values,
            pd.arrays.ArrowExtensionArray,  # type: ignore[attr-defined,unused-ignore]
        ):
            arr = series.values._pa_array  # type: ignore[attr-defined,unused-ignore]
            return pd.Series(
                pd.arrays.ArrowExtensionArray(cls.pc_func(arr)),  # type: ignore[attr-defined,unused-ignore]
                index=series.index,
            )
        return super().apply_operation_component(series)


class Binary(Operator.Binary):
    """
    Checks that the binary operation is performed with numbers.
    """

    type_to_check = Number

    @classmethod
    def _numeric_op(cls, x: Any, y: Any, digits: Optional[int]) -> Any:
        """Apply the operator on float64 and round the result to ``digits``
        significant digits (round-half-even) — the same kernel the DuckDB
        engine applies via ``vtl_round_sig`` (issue #985). Assumes x, y are
        non-null.
        """
        if isinstance(x, int) and isinstance(y, int):
            if cls.op == DIV and y == 0:
                raise SemanticError("2-1-15-6", op=cls.op, value=y)
            if cls.op == RANDOM:
                return cls.py_op(x, y)
            if cls.op in (PLUS, MINUS, MULT):
                # Exact integer arithmetic, matching DuckDB BIGINT semantics.
                return cls.py_op(x, y)
        x = float(x)
        y = float(y)
        if cls.op == DIV and y == 0:
            raise SemanticError("2-1-15-6", op=cls.op, value=y)
        if cls.op == MOD:
            if y == 0:
                raise SemanticError("2-1-15-6", op=cls.op, value=y)
            # fmod keeps the dividend's sign, like DuckDB ``%`` (and the former
            # Decimal remainder): -5 mod 3 is -2, not Python's 1.
            result = math.fmod(x, y)
        elif cls.op == POWER:
            if x < 0 and not y.is_integer():
                raise SemanticError("2-1-15-2", op=cls.op, value=x)
            if x == 0 and y < 0:
                raise SemanticError("2-1-15-6", op=cls.op, value=x)
            result = cls.py_op(x, y)
        else:
            result = cls.py_op(x, y)
        if not isinstance(result, float):
            return result
        if math.isnan(result):
            return None
        if cls.op in _ROUNDED_NUMERIC_OPS:
            result = _round_to_significant(result, digits)
        # Whole results render as ints, but only within float64's exact-integer
        # range: pyarrow refuses larger Python ints when building double arrays.
        if result.is_integer() and abs(result) <= 2**53:
            return int(result)
        return result

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return cls._numeric_op(x, y, get_effective_numeric_digits())

    @classmethod
    def _null_aware_numeric_op(cls, x: Any, y: Any, digits: Optional[int]) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        return cls._numeric_op(x, y, digits)

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        digits = get_effective_numeric_digits()
        result = list(
            map(
                lambda x, y: cls._null_aware_numeric_op(x, y, digits),
                left_series.values,
                right_series.values,
            )
        )
        index = left_series.index if len(left_series) <= len(right_series) else right_series.index
        result_dtype = cls.return_type.dtype() if cls.return_type is not None else "string[pyarrow]"
        return pd.Series(result, index=index, dtype=result_dtype)

    @classmethod
    def apply_operation_series_scalar(
        cls,
        series: Any,
        scalar: Any,
        series_left: bool,
    ) -> Any:
        result_dtype = cls.return_type.dtype() if cls.return_type is not None else "string[pyarrow]"
        if scalar is None:
            return pd.Series(None, index=series.index, dtype=result_dtype)
        digits = get_effective_numeric_digits()
        if series_left:
            return series.map(
                lambda x: cls._numeric_op(x, scalar, digits), na_action="ignore"
            ).astype(result_dtype)
        else:
            return series.map(
                lambda x: cls._numeric_op(scalar, x, digits), na_action="ignore"
            ).astype(result_dtype)


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
    pc_func = staticmethod(pc.negate)


class AbsoluteValue(Unary):
    """
    `Absolute <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=112&zoom=100,72,801> `_ unary operator
    """  # noqa E501

    op = ABS
    py_op = operator.abs
    pc_func = staticmethod(pc.abs)


class Exponential(Unary):
    """
    `Exponential <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=114&zoom=100,72,94>`_ unary operator
    """  # noqa E501

    op = EXP
    py_op = math.exp
    return_type = Number
    pc_func = staticmethod(pc.exp)


class NaturalLogarithm(Unary):
    """
    `Natural logarithm <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=115&zoom=100,72,394> `_
    unary operator
    """  # noqa E501

    op = LN
    return_type = Number
    pc_func = staticmethod(pc.ln)

    @classmethod
    def py_op(cls, x: Any) -> Any:
        if x <= 0:
            raise SemanticError("2-1-15-8", op=cls.op, value=x)
        return math.log(x)

    @classmethod
    def _check_domain(cls, series: Any) -> None:
        bad = series.dropna()
        bad = bad[bad <= 0]
        if len(bad):
            raise SemanticError("2-1-15-8", op=cls.op, value=bad.iloc[0])


class SquareRoot(Unary):
    """
    `Square Root <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=119&zoom=100,72,556> '_
    unary operator
    """  # noqa E501

    op = SQRT
    return_type = Number
    pc_func = staticmethod(pc.sqrt)

    @classmethod
    def py_op(cls, x: Any) -> Any:
        if x < 0:
            raise SemanticError("2-1-15-2", op=cls.op, value=x)
        return math.sqrt(x)

    @classmethod
    def _check_domain(cls, series: Any) -> None:
        bad = series.dropna()
        bad = bad[bad < 0]
        if len(bad):
            raise SemanticError("2-1-15-2", op=cls.op, value=bad.iloc[0])


class Ceil(Unary):
    """
    `Ceilling <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=110&zoom=100,72,94> `_ unary operator
    """  # noqa E501

    op = CEIL
    py_op = math.ceil
    return_type = Integer
    pc_func = staticmethod(pc.ceil)


class Floor(Unary):
    """
    `Floor <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=111&zoom=100,72,442> `_ unary operator
    """  # noqa E501

    op = FLOOR
    py_op = math.floor
    return_type = Integer
    pc_func = staticmethod(pc.floor)


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
        if pd.isnull(x):
            return None
        if x <= 0:
            raise SemanticError("2-1-15-8", op=cls.op, value=x)

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
    def op_func(cls, x: Any, param: Optional[Any]) -> Any:
        return None if pd.isnull(x) else cls.py_op(x, param)

    @classmethod
    def apply_operation_two_series(cls, left_series: Any, right_series: Any) -> Any:
        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def apply_operation_series_scalar(cls, series: Any, param: Any) -> Any:
        return series.map(lambda x: cls.op_func(x, param))

    @classmethod
    def dataset_evaluation(
        cls, operand: Dataset, param: Optional[Union[DataComponent, Scalar]] = None
    ) -> Dataset:
        result = cls.validate(operand, param)
        result.data = operand.data.copy() if operand.data is not None else pd.DataFrame()
        for measure_name in operand.get_measures_names():
            try:
                if isinstance(param, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param.data
                    )
                else:
                    param_value = param.value if param is not None else None
                    result.data[measure_name] = cls.apply_operation_series_scalar(
                        result.data[measure_name], param_value
                    )
            except ValueError:
                raise SemanticError(
                    "2-1-15-1",
                    op=cls.op,
                    comp_name=measure_name,
                    dataset_name=operand.name,
                ) from None
        result.data = result.data[
            operand.get_identifiers_names()
            + operand.get_measures_names()
            + operand.get_viral_attributes_names()
        ]
        # Row-preserving operator: viral attributes are copied through unchanged (issue #906).
        cls.modify_measure_column(result)
        return result

    @classmethod
    def component_evaluation(
        cls,
        operand: DataComponent,
        param: Optional[Union[DataComponent, Scalar]] = None,
    ) -> DataComponent:
        result = cls.validate(operand, param)
        if operand.data is None:
            operand.data = pd.Series()
        result.data = operand.data.copy()
        if isinstance(param, DataComponent):
            result.data = cls.apply_operation_two_series(operand.data, param.data)
        else:
            param_value = param.value if param is not None else None
            result.data = cls.apply_operation_series_scalar(operand.data, param_value)
        return result

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

    @classmethod
    def py_op(cls, x: Any, param: Any) -> Any:
        multiplier = 1.0
        if not pd.isnull(param):
            multiplier = 10**param

        if x >= 0.0:
            rounded_value = math.floor(x * multiplier + 0.5) / multiplier
        else:
            rounded_value = math.ceil(x * multiplier - 0.5) / multiplier

        if param is not None:
            # Number result: normalize like every arithmetic result so both
            # engines' round() agree bit-for-bit (issue #985)
            return _round_to_significant(rounded_value, get_effective_numeric_digits())

        return int(rounded_value)


class Trunc(Parameterized):
    """
    `Trunc <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=108&zoom=100,72,94>`_ operator.
    """  # noqa E501

    op = TRUNC

    @classmethod
    def py_op(cls, x: float, param: Optional[float]) -> Any:
        multiplier = 1.0
        if not pd.isnull(param) and param is not None:
            multiplier = 10**param

        truncated_value = int(x * multiplier) / multiplier

        if not pd.isnull(param):
            # Number result: normalize like every arithmetic result (issue #985)
            return _round_to_significant(truncated_value, get_effective_numeric_digits())

        return int(truncated_value)


class PseudoRandom(_random.Random):
    def __init__(self, seed: Union[int, float]) -> None:
        super().__init__()
        self.seed(seed)


class Random(Parameterized):
    op = RANDOM
    return_type = Number

    @classmethod
    def validate(cls, seed: Any, index: Any = None) -> Any:
        if index is not None and index.data_type is not Integer:
            raise SemanticError(
                "1-1-1-1",
                type_1=SCALAR_TYPES_CLASS_REVERSE[index.data_type],
                type_2=SCALAR_TYPES_CLASS_REVERSE[Integer],
            )
        # A Component carries its values in its data, so only a Scalar can be read here.
        value = index.value if isinstance(index, Scalar) else None
        if value is not None and value < 0:
            raise SemanticError("2-1-15-2", op=cls.op, value=value)
        if value is not None and value > 10000:
            warnings.warn(
                "Random: The value of 'index' is very big. This can affect performance.",
                UserWarning,
            )
        return super().validate(seed, index)

    @classmethod
    def py_op(cls, seed: Union[int, float], index: int) -> float:
        instance: PseudoRandom = PseudoRandom(seed)
        for _ in range(index):
            instance.random()
        return instance.random().__round__(6)

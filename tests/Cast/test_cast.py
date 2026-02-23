import warnings
from pathlib import Path

import pytest

from tests.Helper import TestHelper
from vtlengine.API import create_ast
from vtlengine.DataTypes import (
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    String,
    TimeInterval,
    TimePeriod,
)
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Scalar
from vtlengine.Operators.CastOperator import Cast


class CastHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


# ---------------------------------------------------------------------------
# Parametrized mask-based evaluate (success cases)
# ---------------------------------------------------------------------------
evaluate_params = [
    # String -> Number with mask
    (Scalar("40.000", String, "40.000"), Number, "DD.DDD", 40.0),
    # String -> Date with mask
    (Scalar("2022-01-01", String, "2022-01-01"), Date, "YYYY-MM-DD", "2022-01-01"),
    # String -> TimePeriod with mask
    (Scalar("2000Q1", String, "2000Q1"), TimePeriod, r"YYYY\QQ", "2000-Q1"),
    # String -> TimeInterval with mask
    (
        Scalar("2022-05-21/2023-05-21", String, "2022-05-21/2023-05-21"),
        TimeInterval,
        "YYYY-MM-DD/YYYY-MM-DD",
        "2022-05-21/2023-05-21",
    ),
    # Date -> String with mask
    (Scalar("2021-12-21", Date, "2021-12-21"), String, "YYYY-MM-DD", "2021-12-21"),
    # TimeInterval -> String with mask
    (
        Scalar("2022-05-21/2023-05-21", TimeInterval, "2022-05-21/2023-05-21"),
        String,
        "YYYY-MM-DD/YYYY-MM-DD",
        "2022-05-21/2023-05-21",
    ),
    # String -> Duration with mask
    (Scalar("A", String, "A"), Duration, "A", "A"),
    # Number -> String with mask
    (Scalar("40.000", Number, 40.0), String, "DD.DDD", "40.000"),
]

# ---------------------------------------------------------------------------
# Parametrized mask-based cast_value (success cases)
# ---------------------------------------------------------------------------
cast_value_params = [
    ("40.000", String, Number, "DD.DDD", 40.0),
    ("2022-01-01", String, Date, "YYYY-MM-DD", "2022-01-01"),
    ("2000Q1", String, TimePeriod, r"YYYY\QQ", "2000-Q1"),
    (
        "2022-05-21/2023-05-21",
        String,
        TimeInterval,
        "YYYY-MM-DD/YYYY-MM-DD",
        "2022-05-21/2023-05-21",
    ),
    ("40.000", Number, String, "DD.DDD", "40.000"),
    ("A", String, Duration, "A", "A"),
]

# ---------------------------------------------------------------------------
# Parametrized interpreter-level mask success cases
# ---------------------------------------------------------------------------
interpreter_mask_success_params = [
    ('cast("40.000", number, "DD.DDD")', 40.0, Number),
    ('cast("2022-01-01", date, "YYYY-MM-DD")', "2022-01-01", Date),
    ('cast ("2000Q1", time_period, "YYYY\\QQ")', "2000-Q1", TimePeriod),
    (
        'cast ("2022-05-21/2023-05-21", time, "YYYY-MM-DD/YYYY-MM-DD")',
        "2022-05-21/2023-05-21",
        TimeInterval,
    ),
    ('cast("A", duration, "A")', "A", Duration),
]

# ---------------------------------------------------------------------------
# Parametrized interpreter-level mask error cases
# ---------------------------------------------------------------------------
interpreter_mask_error_params = [
    # String to String with mask is still invalid (1-1-5-5)
    (
        'cast("2021-12-21", string, "YYYY-MM-DD hh:mm:ss")',
        SemanticError,
        "A mask can't be provided to cast from type String to String. Mask provided: "
        "YYYY-MM-DD hh:mm:ss",
    ),
    (
        'cast("P0Y240D", string, "YYYY-MM-DD hh:mm:ss")',
        SemanticError,
        "A mask can't be provided to cast from type String to String. Mask provided: "
        "YYYY-MM-DD hh:mm:ss",
    ),
    (
        'cast ("2022-05-21/2023-05-21", string, "YYYY-MM-DD/YYYY-MM-DD")',
        SemanticError,
        "A mask can't be provided to cast from type String to String. Mask provided: "
        "YYYY-MM-DD/YYYY-MM-DD",
    ),
]


class CastExplicitWithoutMask(CastHelper):
    """ """

    classTest = "Cast.CastExplicitWithoutMask"

    def test_GL_461_1(self):
        code = "GL_461_1"
        number_inputs = 1

        error_code = "1-1-5-5"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_563_1(self):
        """
        Solves bug report in github issue #296
        """
        code = "GL_563_1"
        number_inputs = 1
        reference_names = ["1", "2"]

        self.BaseTest(code, number_inputs, references_names=reference_names)


@pytest.mark.parametrize("operand, scalar_type, mask, expected_value", evaluate_params)
def test_mask_cast_evaluate(
    operand: Scalar, scalar_type: type, mask: str, expected_value: object
) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = Cast.evaluate(operand=operand, scalarType=scalar_type, mask=mask)
    assert result.value == expected_value


@pytest.mark.parametrize("value, provided_type, to_type, mask, expected_value", cast_value_params)
def test_mask_cast_value(
    value: object, provided_type: type, to_type: type, mask: str, expected_value: object
) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = Cast.cast_value(value, provided_type=provided_type, to_type=to_type, mask_value=mask)
    assert result == expected_value


@pytest.mark.parametrize("text, expected_value, expected_type", interpreter_mask_success_params)
def test_mask_cast_interpreter_success(
    text: str, expected_value: object, expected_type: type
) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    scalar = result["DS_r"]
    assert scalar.value == expected_value
    assert scalar.data_type == expected_type


@pytest.mark.parametrize("text, type_of_error, exception_message", interpreter_mask_error_params)
def test_mask_cast_interpreter_errors(
    text: str, type_of_error: type, exception_message: str
) -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(type_of_error, match=exception_message):
        interpreter.visit(ast)


# ===========================================================================
# New comprehensive explicit cast tests (VTL 2.2)
# ===========================================================================


class TestCastIntegerToBoolean:
    """Integer -> Boolean (Explicit without mask)"""

    def test_zero_to_false(self) -> None:
        operand = Scalar("x", Integer, 0)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False
        assert result.data_type == Boolean

    def test_positive_to_true(self) -> None:
        operand = Scalar("x", Integer, 5)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_negative_to_true(self) -> None:
        operand = Scalar("x", Integer, -3)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_one_to_true(self) -> None:
        operand = Scalar("x", Integer, 1)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Integer, None)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is None
        assert result.data_type == Boolean


class TestCastBooleanToInteger:
    """Boolean -> Integer (Explicit without mask)"""

    def test_true_to_one(self) -> None:
        operand = Scalar("x", Boolean, True)
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value == 1
        assert result.data_type == Integer

    def test_false_to_zero(self) -> None:
        operand = Scalar("x", Boolean, False)
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value == 0
        assert result.data_type == Integer

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Boolean, None)
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value is None


class TestCastNumberToBoolean:
    """Number -> Boolean (Explicit without mask)"""

    def test_zero_to_false(self) -> None:
        operand = Scalar("x", Number, 0.0)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False
        assert result.data_type == Boolean

    def test_positive_to_true(self) -> None:
        operand = Scalar("x", Number, 3.14)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_negative_to_true(self) -> None:
        operand = Scalar("x", Number, -2.5)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Number, None)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is None


class TestCastBooleanToNumber:
    """Boolean -> Number (Explicit without mask)"""

    def test_true_to_one(self) -> None:
        operand = Scalar("x", Boolean, True)
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == 1.0
        assert result.data_type == Number

    def test_false_to_zero(self) -> None:
        operand = Scalar("x", Boolean, False)
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == 0.0
        assert result.data_type == Number

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Boolean, None)
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value is None


class TestCastStringToBoolean:
    """String -> Boolean (Explicit without mask)"""

    def test_true_string(self) -> None:
        operand = Scalar("x", String, "true")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True
        assert result.data_type == Boolean

    def test_true_upper(self) -> None:
        operand = Scalar("x", String, "TRUE")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_true_mixed_case(self) -> None:
        operand = Scalar("x", String, "TrUe")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_true_with_spaces(self) -> None:
        operand = Scalar("x", String, "  true  ")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is True

    def test_false_string(self) -> None:
        operand = Scalar("x", String, "false")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False

    def test_false_upper(self) -> None:
        operand = Scalar("x", String, "FALSE")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False

    def test_any_other_returns_false(self) -> None:
        """Per VTL 2.2: false is returned for any non-'true' string."""
        operand = Scalar("x", String, "yes")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False

    def test_empty_string_returns_false(self) -> None:
        operand = Scalar("x", String, "")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False

    def test_number_string_returns_false(self) -> None:
        operand = Scalar("x", String, "1")
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is False

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=Boolean)
        assert result.value is None


class TestCastStringToInteger:
    """String -> Integer (Explicit without mask, default behavior)"""

    def test_simple_integer(self) -> None:
        operand = Scalar("x", String, "42")
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value == 42
        assert result.data_type == Integer

    def test_negative_integer(self) -> None:
        operand = Scalar("x", String, "-10")
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value == -10

    def test_zero(self) -> None:
        operand = Scalar("x", String, "0")
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value == 0

    def test_decimal_raises(self) -> None:
        operand = Scalar("x", String, "3.14")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Integer)

    def test_invalid_string_raises(self) -> None:
        operand = Scalar("x", String, "abc")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Integer)

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value is None


class TestCastStringToNumber:
    """String -> Number (Explicit without mask, default behavior)"""

    def test_simple_number(self) -> None:
        operand = Scalar("x", String, "3.14")
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == 3.14
        assert result.data_type == Number

    def test_integer_string(self) -> None:
        operand = Scalar("x", String, "42")
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == 42.0

    def test_scientific_notation(self) -> None:
        operand = Scalar("x", String, "1.5E3")
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == 1500.0

    def test_negative(self) -> None:
        operand = Scalar("x", String, "-2.5")
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == -2.5

    def test_invalid_raises(self) -> None:
        operand = Scalar("x", String, "abc")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Number)

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value is None


class TestCastStringToDate:
    """String -> Date (Explicit without mask, default YYYY-MM-DD)"""

    def test_valid_date(self) -> None:
        operand = Scalar("x", String, "2025-01-15")
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value == "2025-01-15"
        assert result.data_type == Date

    def test_invalid_format_raises(self) -> None:
        operand = Scalar("x", String, "15-01-2025")
        with pytest.raises(SemanticError):
            Cast.evaluate(operand=operand, scalarType=Date)

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value is None


class TestCastStringToTimePeriod:
    """String -> TimePeriod (Explicit without mask)"""

    def test_annual(self) -> None:
        operand = Scalar("x", String, "2020A")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020A"
        assert result.data_type == TimePeriod

    def test_quarterly(self) -> None:
        operand = Scalar("x", String, "2020Q1")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020Q1"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value is None


class TestCastStringToTimeInterval:
    """String -> Time (TimeInterval) (Explicit without mask)"""

    def test_valid_interval(self) -> None:
        operand = Scalar("x", String, "2020-01-01/2020-12-31")
        result = Cast.evaluate(operand=operand, scalarType=TimeInterval)
        assert result.value == "2020-01-01/2020-12-31"
        assert result.data_type == TimeInterval

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=TimeInterval)
        assert result.value is None


class TestCastStringToDuration:
    """String -> Duration (Explicit without mask)"""

    def test_annual(self) -> None:
        operand = Scalar("x", String, "A")
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value == "A"
        assert result.data_type == Duration

    def test_daily(self) -> None:
        operand = Scalar("x", String, "D")
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value == "D"

    def test_monthly(self) -> None:
        operand = Scalar("x", String, "M")
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value == "M"

    def test_quarterly(self) -> None:
        operand = Scalar("x", String, "Q")
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value == "Q"

    def test_weekly(self) -> None:
        operand = Scalar("x", String, "W")
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value == "W"

    def test_semester(self) -> None:
        operand = Scalar("x", String, "S")
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value == "S"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", String, None)
        result = Cast.evaluate(operand=operand, scalarType=Duration)
        assert result.value is None


class TestCastNumberToInteger:
    """Number -> Integer (Implicit cast)"""

    def test_whole_number(self) -> None:
        operand = Scalar("x", Number, 5.0)
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value == 5
        assert result.data_type == Integer

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Number, None)
        result = Cast.evaluate(operand=operand, scalarType=Integer)
        assert result.value is None


class TestCastIntegerToNumber:
    """Integer -> Number (Implicit cast)"""

    def test_integer_to_number(self) -> None:
        operand = Scalar("x", Integer, 42)
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value == 42.0
        assert result.data_type == Number

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Integer, None)
        result = Cast.evaluate(operand=operand, scalarType=Number)
        assert result.value is None


class TestCastIntegerToString:
    """Integer -> String (Explicit without mask)"""

    def test_positive(self) -> None:
        operand = Scalar("x", Integer, 42)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "42"
        assert result.data_type == String

    def test_negative(self) -> None:
        operand = Scalar("x", Integer, -7)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "-7"

    def test_zero(self) -> None:
        operand = Scalar("x", Integer, 0)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "0"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Integer, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastNumberToString:
    """Number -> String (Explicit without mask)"""

    def test_float(self) -> None:
        operand = Scalar("x", Number, 3.14)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "3.14"
        assert result.data_type == String

    def test_integer_float(self) -> None:
        operand = Scalar("x", Number, 5.0)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "5.0"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Number, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastBooleanToString:
    """Boolean -> String (Implicit cast)"""

    def test_true_to_string(self) -> None:
        operand = Scalar("x", Boolean, True)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "True"
        assert result.data_type == String

    def test_false_to_string(self) -> None:
        operand = Scalar("x", Boolean, False)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "False"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Boolean, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastDateToString:
    """Date -> String (Explicit without mask)"""

    def test_date_to_string(self) -> None:
        operand = Scalar("x", Date, "2025-01-15")
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "2025-01-15"
        assert result.data_type == String

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Date, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastTimePeriodToString:
    """TimePeriod -> String (Explicit without mask)"""

    def test_annual(self) -> None:
        operand = Scalar("x", TimePeriod, "2020A")
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "2020A"
        assert result.data_type == String

    def test_quarterly(self) -> None:
        operand = Scalar("x", TimePeriod, "2020Q1")
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "2020Q1"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", TimePeriod, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastTimeIntervalToString:
    """TimeInterval -> String (Explicit without mask)"""

    def test_interval_to_string(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "2020-01-01/2020-12-31"
        assert result.data_type == String

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", TimeInterval, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastDurationToString:
    """Duration -> String (Explicit without mask)"""

    def test_annual(self) -> None:
        operand = Scalar("x", Duration, "A")
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value == "A"
        assert result.data_type == String

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Duration, None)
        result = Cast.evaluate(operand=operand, scalarType=String)
        assert result.value is None


class TestCastDateToTimePeriod:
    """Date -> TimePeriod (Explicit without mask)"""

    def test_date_to_daily_period(self) -> None:
        operand = Scalar("x", Date, "2020-01-15")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020D15"
        assert result.data_type == TimePeriod

    def test_first_day_of_year(self) -> None:
        operand = Scalar("x", Date, "2025-01-01")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2025D1"

    def test_last_day_of_year(self) -> None:
        operand = Scalar("x", Date, "2025-12-31")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2025D365"

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Date, None)
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value is None


class TestCastTimePeriodToDate:
    """TimePeriod -> Date (Explicit without mask, only for single-day periods)"""

    def test_daily_period_to_date(self) -> None:
        operand = Scalar("x", TimePeriod, "2020D15")
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value == "2020-01-15"
        assert result.data_type == Date

    def test_daily_first_day(self) -> None:
        operand = Scalar("x", TimePeriod, "2025D1")
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value == "2025-01-01"

    def test_annual_raises_error(self) -> None:
        """Annual period has more than one day, should raise error."""
        operand = Scalar("x", TimePeriod, "2020A")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Date)

    def test_quarterly_raises_error(self) -> None:
        operand = Scalar("x", TimePeriod, "2020Q1")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Date)

    def test_monthly_raises_error(self) -> None:
        operand = Scalar("x", TimePeriod, "2020M1")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Date)

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", TimePeriod, None)
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value is None


class TestCastTimeIntervalToDate:
    """TimeInterval (Time) -> Date (Explicit without mask)"""

    def test_same_start_end_date(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-15/2020-01-15")
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value == "2020-01-15"
        assert result.data_type == Date

    def test_different_dates_raises_error(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=Date)

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", TimeInterval, None)
        result = Cast.evaluate(operand=operand, scalarType=Date)
        assert result.value is None


class TestCastTimeIntervalToTimePeriod:
    """TimeInterval (Time) -> TimePeriod (Explicit without mask)"""

    def test_annual_interval(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020A"
        assert result.data_type == TimePeriod

    def test_quarterly_interval(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-03-31")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020-Q1"

    def test_semester_interval(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-06-30")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020-S1"

    def test_monthly_interval(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-03-01/2020-03-31")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020-M03"

    def test_daily_interval(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-15/2020-01-15")
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value == "2020-D015"

    def test_irregular_raises_error(self) -> None:
        """An interval that doesn't match any regular period should error."""
        operand = Scalar("x", TimeInterval, "2020-01-15/2020-03-20")
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=operand, scalarType=TimePeriod)

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", TimeInterval, None)
        result = Cast.evaluate(operand=operand, scalarType=TimePeriod)
        assert result.value is None


class TestCastDateToTimeInterval:
    """Date -> Time (TimeInterval) (Implicit cast)"""

    def test_date_to_interval(self) -> None:
        operand = Scalar("x", Date, "2020-01-15")
        result = Cast.evaluate(operand=operand, scalarType=TimeInterval)
        assert result.value == "2020-01-15/2020-01-15"
        assert result.data_type == TimeInterval

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", Date, None)
        result = Cast.evaluate(operand=operand, scalarType=TimeInterval)
        assert result.value is None


class TestCastTimePeriodToTimeInterval:
    """TimePeriod -> Time (TimeInterval) (Implicit cast)"""

    def test_annual_to_interval(self) -> None:
        operand = Scalar("x", TimePeriod, "2020A")
        result = Cast.evaluate(operand=operand, scalarType=TimeInterval)
        assert result.value == "2020-01-01/2020-12-31"
        assert result.data_type == TimeInterval

    def test_null_returns_null(self) -> None:
        operand = Scalar("x", TimePeriod, None)
        result = Cast.evaluate(operand=operand, scalarType=TimeInterval)
        assert result.value is None


# ===========================================================================
# Semantic validation tests: cast type compatibility
# ===========================================================================


class TestCastSemanticErrors:
    """Test that infeasible casts raise SemanticError."""

    def test_integer_to_time_raises(self) -> None:
        operand = Scalar("x", Integer, 1)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimeInterval)

    def test_integer_to_date_raises(self) -> None:
        operand = Scalar("x", Integer, 1)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Date)

    def test_integer_to_time_period_raises(self) -> None:
        operand = Scalar("x", Integer, 1)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimePeriod)

    def test_integer_to_duration_raises(self) -> None:
        operand = Scalar("x", Integer, 1)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Duration)

    def test_number_to_time_raises(self) -> None:
        operand = Scalar("x", Number, 1.0)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimeInterval)

    def test_number_to_date_raises(self) -> None:
        operand = Scalar("x", Number, 1.0)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Date)

    def test_boolean_to_time_raises(self) -> None:
        operand = Scalar("x", Boolean, True)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimeInterval)

    def test_boolean_to_date_raises(self) -> None:
        operand = Scalar("x", Boolean, True)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Date)

    def test_boolean_to_time_period_raises(self) -> None:
        operand = Scalar("x", Boolean, True)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimePeriod)

    def test_boolean_to_duration_raises(self) -> None:
        operand = Scalar("x", Boolean, True)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Duration)

    def test_date_to_integer_raises(self) -> None:
        operand = Scalar("x", Date, "2020-01-01")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Integer)

    def test_date_to_number_raises(self) -> None:
        operand = Scalar("x", Date, "2020-01-01")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Number)

    def test_date_to_boolean_raises(self) -> None:
        operand = Scalar("x", Date, "2020-01-01")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Boolean)

    def test_date_to_duration_raises(self) -> None:
        operand = Scalar("x", Date, "2020-01-01")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Duration)

    def test_time_to_integer_raises(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Integer)

    def test_time_to_number_raises(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Number)

    def test_time_to_boolean_raises(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Boolean)

    def test_time_to_duration_raises(self) -> None:
        operand = Scalar("x", TimeInterval, "2020-01-01/2020-12-31")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Duration)

    def test_duration_to_integer_raises(self) -> None:
        operand = Scalar("x", Duration, "A")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Integer)

    def test_duration_to_number_raises(self) -> None:
        operand = Scalar("x", Duration, "A")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Number)

    def test_duration_to_boolean_raises(self) -> None:
        operand = Scalar("x", Duration, "A")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Boolean)

    def test_duration_to_time_raises(self) -> None:
        operand = Scalar("x", Duration, "A")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimeInterval)

    def test_duration_to_date_raises(self) -> None:
        operand = Scalar("x", Duration, "A")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Date)

    def test_duration_to_time_period_raises(self) -> None:
        operand = Scalar("x", Duration, "A")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=TimePeriod)


# ===========================================================================
# Mask error tests: mask where not allowed
# ===========================================================================


class TestCastMaskNotAllowed:
    """Test 1-1-5-5: mask provided where not supported."""

    def test_boolean_to_integer_with_mask_raises(self) -> None:
        operand = Scalar("x", Boolean, True)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Integer, mask="FORMAT")

    def test_integer_to_number_with_mask_raises(self) -> None:
        operand = Scalar("x", Integer, 1)
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Number, mask="FORMAT")

    def test_string_to_string_with_mask_raises(self) -> None:
        operand = Scalar("x", String, "hello")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=String, mask="FORMAT")

    def test_string_to_boolean_with_mask_raises(self) -> None:
        operand = Scalar("x", String, "true")
        with pytest.raises(SemanticError):
            Cast.validate(operand=operand, scalarType=Boolean, mask="FORMAT")


# ===========================================================================
# Mask validation passes (no longer raises NotImplementedError)
# ===========================================================================


class TestCastMaskValidationPasses:
    """Verify that validate() accepts valid mask-based casts (previously NotImplementedError)."""

    @pytest.mark.parametrize(
        "operand, to_type, mask",
        [
            (Scalar("x", Integer, 42), String, "DD"),
            (Scalar("x", Number, 3.14), String, "DD.DDD"),
            (Scalar("x", String, "42"), Integer, "DD"),
            (Scalar("x", TimePeriod, "2020A"), String, "YYYY"),
            (Scalar("x", Date, "2020-01-01"), String, "YYYY-MM-DD"),
            (Scalar("x", Duration, "A"), String, "A"),
            (Scalar("x", TimeInterval, "2020-01-01/2020-12-31"), String, "YYYY-MM-DD/YYYY-MM-DD"),
            (Scalar("x", String, "2020-01-01"), Date, "YYYY-MM-DD"),
            (Scalar("x", String, "2000Q1"), TimePeriod, r"YYYY\QQ"),
            (Scalar("x", String, "A"), Duration, "A"),
            (Scalar("x", String, "2020-01-01/2020-12-31"), TimeInterval, "YYYY-MM-DD/YYYY-MM-DD"),
        ],
    )
    def test_validate_accepts_mask(self, operand: Scalar, to_type: type, mask: str) -> None:
        # Should not raise any exception
        Cast.validate(operand=operand, scalarType=to_type, mask=mask)


# ===========================================================================
# Mask-based cast tests (VTL 2.2)
# ===========================================================================


class TestCastStringToNumberWithMask:
    """String -> Number with mask (DD.DDD style)."""

    def test_basic_dd_ddd(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "40.000"), Number, mask="DD.DDD")
        assert result.value == 40.0
        assert result.data_type == Number

    def test_single_d_decimal(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "3.14"), Number, mask="D.DD")
        assert result.value == 3.14

    def test_negative_value(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "-1.50"), Number, mask="D.DD")
        assert result.value == -1.5

    def test_positive_sign(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "+2.50"), Number, mask="D.DD")
        assert result.value == 2.5

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "4.00"), Number, mask="DD.DDD")

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "ab.cde"), Number, mask="DD.DDD")

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", String, None), Number, mask="DD.DDD")
        assert result.value is None


class TestCastStringToIntegerWithMask:
    """String -> Integer with mask (DDD style)."""

    def test_three_digits(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "042"), Integer, mask="DDD")
        assert result.value == 42
        assert result.data_type == Integer

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "42"), Integer, mask="DDD")

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", String, None), Integer, mask="DDD")
        assert result.value is None


class TestCastNumberToStringWithMask:
    """Number -> String with mask."""

    def test_dd_ddd(self) -> None:
        result = Cast.evaluate(Scalar("x", Number, 40.0), String, mask="DD.DDD")
        assert result.value == "40.000"
        assert result.data_type == String

    def test_negative(self) -> None:
        result = Cast.evaluate(Scalar("x", Number, -1.5), String, mask="D.DD")
        assert result.value == "-1.50"

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", Number, None), String, mask="DD.DDD")
        assert result.value is None


class TestCastIntegerToStringWithMask:
    """Integer -> String with mask."""

    def test_three_digits(self) -> None:
        result = Cast.evaluate(Scalar("x", Integer, 42), String, mask="DDD")
        assert result.value == "042"
        assert result.data_type == String

    def test_negative(self) -> None:
        result = Cast.evaluate(Scalar("x", Integer, -7), String, mask="DD")
        assert result.value == "-07"

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", Integer, None), String, mask="DDD")
        assert result.value is None


class TestCastStringToDateWithMask:
    """String -> Date with mask."""

    def test_yyyy_mm_dd(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2022-01-01"), Date, mask="YYYY-MM-DD")
        assert result.value == "2022-01-01"
        assert result.data_type == Date

    def test_ddmmyyyy(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "01012022"), Date, mask="DDMMYYYY")
        assert result.value == "2022-01-01"

    def test_with_spaces_stripped(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "  2022-01-01  "), Date, mask="YYYY-MM-DD")
        assert result.value == "2022-01-01"

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "01-01-2022"), Date, mask="YYYY-MM-DD")

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", String, None), Date, mask="YYYY-MM-DD")
        assert result.value is None


class TestCastDateToStringWithMask:
    """Date -> String with mask."""

    def test_yyyy_mm_dd(self) -> None:
        result = Cast.evaluate(Scalar("x", Date, "2022-01-01"), String, mask="YYYY-MM-DD")
        assert result.value == "2022-01-01"
        assert result.data_type == String

    def test_ddmmyyyy(self) -> None:
        result = Cast.evaluate(Scalar("x", Date, "2022-01-01"), String, mask="DDMMYYYY")
        assert result.value == "01012022"

    def test_yy_mm_dd(self) -> None:
        result = Cast.evaluate(Scalar("x", Date, "2022-01-01"), String, mask="YY-MM-DD")
        assert result.value == "22-01-01"

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", Date, None), String, mask="YYYY-MM-DD")
        assert result.value is None


class TestCastStringToTimeIntervalWithMask:
    """String -> Time (TimeInterval) with mask."""

    def test_yyyy_mm_dd_slash(self) -> None:
        result = Cast.evaluate(
            Scalar("x", String, "2022-05-21/2023-05-21"),
            TimeInterval,
            mask="YYYY-MM-DD/YYYY-MM-DD",
        )
        assert result.value == "2022-05-21/2023-05-21"
        assert result.data_type == TimeInterval

    def test_compact_format(self) -> None:
        result = Cast.evaluate(
            Scalar("x", String, "20220101/20221231"),
            TimeInterval,
            mask="YYYYMMDD/YYYYMMDD",
        )
        assert result.value == "2022-01-01/2022-12-31"

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(
                Scalar("x", String, "01-01-2022/31-12-2022"),
                TimeInterval,
                mask="YYYY-MM-DD/YYYY-MM-DD",
            )

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(
            Scalar("x", String, None), TimeInterval, mask="YYYY-MM-DD/YYYY-MM-DD"
        )
        assert result.value is None


class TestCastTimeIntervalToStringWithMask:
    """TimeInterval -> String with mask."""

    def test_yyyy_mm_dd(self) -> None:
        result = Cast.evaluate(
            Scalar("x", TimeInterval, "2022-05-21/2023-05-21"),
            String,
            mask="YYYY-MM-DD/YYYY-MM-DD",
        )
        assert result.value == "2022-05-21/2023-05-21"
        assert result.data_type == String

    def test_compact_format(self) -> None:
        result = Cast.evaluate(
            Scalar("x", TimeInterval, "2022-01-01/2022-12-31"),
            String,
            mask="YYYYMMDD/YYYYMMDD",
        )
        assert result.value == "20220101/20221231"

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(
            Scalar("x", TimeInterval, None), String, mask="YYYY-MM-DD/YYYY-MM-DD"
        )
        assert result.value is None


class TestCastStringToTimePeriodWithMask:
    """String -> TimePeriod with mask - all indicator styles."""

    def test_yyyyqq(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2000Q1"), TimePeriod, mask=r"YYYY\QQ")
        assert result.value == "2000-Q1"

    def test_yyyy_minus_qq(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2000-Q1"), TimePeriod, mask=r"YYYY-\QQ")
        assert result.value == "2000-Q1"

    def test_standalone_q_digit(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2000-1"), TimePeriod, mask="YYYY-Q")
        assert result.value == "2000-Q1"

    def test_q_before_year(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "Q1-2000"), TimePeriod, mask=r"\QQ-YYYY")
        assert result.value == "2000-Q1"

    def test_qqq_two_digit(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2000Q01"), TimePeriod, mask=r"YYYY\QQQ")
        assert result.value == "2000-Q1"

    def test_yyyymmm(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2000M03"), TimePeriod, mask=r"YYYY\MMM")
        assert result.value == "2000-M03"

    def test_yyyy_only(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "2000"), TimePeriod, mask="YYYY")
        assert result.value == "2000A"

    def test_calendar_date_dots(self) -> None:
        from vtlengine.DataTypes.TimeHandling import TimePeriodHandler

        result = Cast.evaluate(Scalar("x", String, "2000.01.01"), TimePeriod, mask=r"YYYY\.MM\.DD")
        expected = str(TimePeriodHandler("2000D1"))
        assert result.value == expected

    def test_calendar_date_ddd(self) -> None:
        from vtlengine.DataTypes.TimeHandling import TimePeriodHandler

        result = Cast.evaluate(Scalar("x", String, "2000M01D01"), TimePeriod, mask=r"YYYY\MMM\DDD")
        expected = str(TimePeriodHandler("2000D1"))
        assert result.value == expected

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "2000X1"), TimePeriod, mask=r"YYYY\QQ")

    def test_extra_chars_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "2000Q1extra"), TimePeriod, mask=r"YYYY\QQ")

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", String, None), TimePeriod, mask=r"YYYY\QQ")
        assert result.value is None


class TestCastTimePeriodToStringWithMask:
    """TimePeriod -> String with mask."""

    def test_yyyyqq(self) -> None:
        result = Cast.evaluate(Scalar("x", TimePeriod, "2000-Q1"), String, mask=r"YYYY\QQ")
        assert result.value == "2000Q1"

    def test_yyyy_minus_qq(self) -> None:
        result = Cast.evaluate(Scalar("x", TimePeriod, "2000-Q1"), String, mask=r"YYYY-\QQ")
        assert result.value == "2000-Q1"

    def test_yyyymmm(self) -> None:
        result = Cast.evaluate(Scalar("x", TimePeriod, "2000-M03"), String, mask=r"YYYY\MMM")
        assert result.value == "2000M03"

    def test_yyyy_only_annual(self) -> None:
        result = Cast.evaluate(Scalar("x", TimePeriod, "2000A"), String, mask="YYYY")
        assert result.value == "2000"

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", TimePeriod, None), String, mask=r"YYYY\QQ")
        assert result.value is None


class TestCastStringToDurationWithMask:
    """String -> Duration with mask."""

    @pytest.mark.parametrize("shortcode", ["A", "S", "Q", "M", "W", "D"])
    def test_valid_shortcodes(self, shortcode: str) -> None:
        result = Cast.evaluate(Scalar("x", String, shortcode), Duration, mask=shortcode)
        assert result.value == shortcode
        assert result.data_type == Duration

    def test_lowercase_accepted(self) -> None:
        result = Cast.evaluate(Scalar("x", String, "a"), Duration, mask="A")
        assert result.value == "A"

    def test_invalid_shortcode_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "Z"), Duration, mask="A")

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", String, None), Duration, mask="A")
        assert result.value is None


class TestCastDurationToStringWithMask:
    """Duration -> String with mask."""

    @pytest.mark.parametrize("shortcode", ["A", "S", "Q", "M", "W", "D"])
    def test_shortcode_passthrough(self, shortcode: str) -> None:
        result = Cast.evaluate(Scalar("x", Duration, shortcode), String, mask=shortcode)
        assert result.value == shortcode
        assert result.data_type == String

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(Scalar("x", Duration, None), String, mask="A")
        assert result.value is None


# ===========================================================================
# Interpreter-level tests: cast in VTL expressions
# ===========================================================================


class TestCastInterpreter:
    """Test cast operator evaluated through the full interpreter pipeline."""

    @staticmethod
    def _eval_expression(expr: str) -> Scalar:
        warnings.filterwarnings("ignore", category=FutureWarning)
        expression = f"DS_r := {expr};"
        ast = create_ast(expression)
        interpreter = InterpreterAnalyzer({})
        result = interpreter.visit(ast)
        return result["DS_r"]

    def test_cast_integer_to_boolean(self) -> None:
        result = self._eval_expression("cast(5, boolean)")
        assert result.value is True
        assert result.data_type == Boolean

    def test_cast_zero_to_boolean(self) -> None:
        result = self._eval_expression("cast(0, boolean)")
        assert result.value is False

    def test_cast_number_to_boolean(self) -> None:
        result = self._eval_expression("cast(3.14, boolean)")
        assert result.value is True

    def test_cast_boolean_to_integer(self) -> None:
        result = self._eval_expression("cast(true, integer)")
        assert result.value == 1

    def test_cast_false_to_integer(self) -> None:
        result = self._eval_expression("cast(false, integer)")
        assert result.value == 0

    def test_cast_boolean_to_number(self) -> None:
        result = self._eval_expression("cast(true, number)")
        assert result.value == 1.0

    def test_cast_false_to_number(self) -> None:
        result = self._eval_expression("cast(false, number)")
        assert result.value == 0.0

    def test_cast_string_to_boolean_true(self) -> None:
        result = self._eval_expression('cast("true", boolean)')
        assert result.value is True

    def test_cast_string_to_boolean_false(self) -> None:
        result = self._eval_expression('cast("false", boolean)')
        assert result.value is False

    def test_cast_string_to_boolean_other(self) -> None:
        result = self._eval_expression('cast("hello", boolean)')
        assert result.value is False

    def test_cast_number_to_integer(self) -> None:
        result = self._eval_expression("cast(42.0, integer)")
        assert result.value == 42
        assert result.data_type == Integer

    def test_cast_number_to_string(self) -> None:
        result = self._eval_expression("cast(42.5, string)")
        assert result.value == "42.5"
        assert result.data_type == String

    def test_cast_integer_to_string(self) -> None:
        result = self._eval_expression("cast(42, string)")
        assert result.value == "42"
        assert result.data_type == String

    def test_cast_string_to_integer(self) -> None:
        result = self._eval_expression('cast("42", integer)')
        assert result.value == 42
        assert result.data_type == Integer

    def test_cast_string_to_number(self) -> None:
        result = self._eval_expression('cast("3.14", number)')
        assert result.value == 3.14
        assert result.data_type == Number

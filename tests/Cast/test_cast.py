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

    @pytest.mark.parametrize(
        "input_val, expected",
        [(0, False), (5, True), (-3, True), (1, True), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Integer, input_val), scalarType=Boolean)
        assert result.value == expected
        assert result.data_type == Boolean


class TestCastBooleanToInteger:
    """Boolean -> Integer (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(True, 1), (False, 0), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Boolean, input_val), scalarType=Integer)
        assert result.value == expected
        assert result.data_type == Integer


class TestCastNumberToBoolean:
    """Number -> Boolean (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(0.0, False), (3.14, True), (-2.5, True), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Number, input_val), scalarType=Boolean)
        assert result.value == expected
        assert result.data_type == Boolean


class TestCastBooleanToNumber:
    """Boolean -> Number (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(True, 1.0), (False, 0.0), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Boolean, input_val), scalarType=Number)
        assert result.value == expected
        assert result.data_type == Number


class TestCastStringToBoolean:
    """String -> Boolean (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("true", True),
            ("TRUE", True),
            ("TrUe", True),
            ("  true  ", True),
            ("false", False),
            ("FALSE", False),
            ("yes", False),
            ("", False),
            ("1", False),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=Boolean)
        assert result.value == expected
        assert result.data_type == Boolean


class TestCastStringToInteger:
    """String -> Integer (Explicit without mask, default behavior)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("123", 123), ("+456", 456), ("-789", -789), ("0", 0), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=Integer)
        assert result.value == expected
        assert result.data_type == Integer

    @pytest.mark.parametrize("input_val", ["3.14", "abc"])
    def test_invalid_raises(self, input_val: str) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=Integer)


class TestCastStringToNumber:
    """String -> Number (Explicit without mask, default behavior)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("3.14", 3.14),
            ("+123.45", 123.45),
            ("123", 123.0),
            ("+456", 456.0),
            ("-789", -789.0),
            ("1.5E3", 1500.0),
            ("-2.5", -2.5),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=Number)
        assert result.value == expected
        assert result.data_type == Number

    def test_invalid_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=Scalar("x", String, "abc"), scalarType=Number)


class TestCastStringToDate:
    """String -> Date (Explicit without mask, default YYYY-MM-DD)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2025-01-15", "2025-01-15"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=Date)
        assert result.value == expected
        assert result.data_type == Date

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(SemanticError):
            Cast.evaluate(operand=Scalar("x", String, "15-01-2025"), scalarType=Date)


class TestCastStringToTimePeriod:
    """String -> TimePeriod (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020A", "2020A"), ("2020Q1", "2020Q1"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=TimePeriod)
        assert result.value == expected
        assert result.data_type == TimePeriod


class TestCastStringToTimeInterval:
    """String -> Time (TimeInterval) (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020-01-01/2020-12-31", "2020-01-01/2020-12-31"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=TimeInterval)
        assert result.value == expected
        assert result.data_type == TimeInterval


class TestCastStringToDuration:
    """String -> Duration (Explicit without mask)"""

    @pytest.mark.parametrize("shortcode", ["A", "D", "M", "Q", "W", "S"])
    def test_cast(self, shortcode: str) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, shortcode), scalarType=Duration)
        assert result.value == shortcode
        assert result.data_type == Duration

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, None), scalarType=Duration)
        assert result.value is None


class TestCastNumberToInteger:
    """Number -> Integer (Implicit cast)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(5.0, 5), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Number, input_val), scalarType=Integer)
        assert result.value == expected
        assert result.data_type == Integer


class TestCastIntegerToNumber:
    """Integer -> Number (Implicit cast)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(42, 42.0), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Integer, input_val), scalarType=Number)
        assert result.value == expected
        assert result.data_type == Number


class TestCastIntegerToString:
    """Integer -> String (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(42, "42"), (-7, "-7"), (0, "0"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Integer, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastNumberToString:
    """Number -> String (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(3.14, "3.14"), (5.0, "5.0"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Number, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastBooleanToString:
    """Boolean -> String (Implicit cast)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(True, "True"), (False, "False"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Boolean, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastDateToString:
    """Date -> String (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2025-01-15", "2025-01-15"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Date, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastTimePeriodToString:
    """TimePeriod -> String (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020A", "2020A"), ("2020Q1", "2020Q1"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimePeriod, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastTimeIntervalToString:
    """TimeInterval -> String (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020-01-01/2020-12-31", "2020-01-01/2020-12-31"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimeInterval, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastDurationToString:
    """Duration -> String (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("A", "A"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Duration, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastDateToTimePeriod:
    """Date -> TimePeriod (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2020-01-15", "2020D15"),
            ("2025-01-01", "2025D1"),
            ("2025-12-31", "2025D365"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Date, input_val), scalarType=TimePeriod)
        assert result.value == expected
        assert result.data_type == TimePeriod


class TestCastTimePeriodToDate:
    """TimePeriod -> Date (Explicit without mask, only for single-day periods)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020D15", "2020-01-15"), ("2025D1", "2025-01-01"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimePeriod, input_val), scalarType=Date)
        assert result.value == expected
        assert result.data_type == Date

    @pytest.mark.parametrize("input_val", ["2020A", "2020Q1", "2020M1"])
    def test_non_daily_raises(self, input_val: str) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=Scalar("x", TimePeriod, input_val), scalarType=Date)


class TestCastTimeIntervalToDate:
    """TimeInterval (Time) -> Date (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020-01-15/2020-01-15", "2020-01-15"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimeInterval, input_val), scalarType=Date)
        assert result.value == expected
        assert result.data_type == Date

    def test_different_dates_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(
                operand=Scalar("x", TimeInterval, "2020-01-01/2020-12-31"), scalarType=Date
            )


class TestCastTimeIntervalToTimePeriod:
    """TimeInterval (Time) -> TimePeriod (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2020-01-01/2020-12-31", "2020A"),
            ("2020-01-01/2020-03-31", "2020-Q1"),
            ("2020-01-01/2020-06-30", "2020-S1"),
            ("2020-03-01/2020-03-31", "2020-M03"),
            ("2020-01-15/2020-01-15", "2020-D015"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimeInterval, input_val), scalarType=TimePeriod)
        assert result.value == expected
        assert result.data_type == TimePeriod

    def test_irregular_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(
                operand=Scalar("x", TimeInterval, "2020-01-15/2020-03-20"),
                scalarType=TimePeriod,
            )


class TestCastDateToTimeInterval:
    """Date -> Time (TimeInterval) (Implicit cast)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020-01-15", "2020-01-15/2020-01-15"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Date, input_val), scalarType=TimeInterval)
        assert result.value == expected
        assert result.data_type == TimeInterval


class TestCastTimePeriodToTimeInterval:
    """TimePeriod -> Time (TimeInterval) (Implicit cast)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [("2020A", "2020-01-01/2020-12-31"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimePeriod, input_val), scalarType=TimeInterval)
        assert result.value == expected
        assert result.data_type == TimeInterval


# ===========================================================================
# Semantic validation tests: cast type compatibility
# ===========================================================================


class TestCastSemanticErrors:
    """Test that infeasible casts raise SemanticError."""

    @pytest.mark.parametrize(
        "from_type, from_val, to_type",
        [
            (Integer, 1, TimeInterval),
            (Integer, 1, Date),
            (Integer, 1, TimePeriod),
            (Integer, 1, Duration),
            (Number, 1.0, TimeInterval),
            (Number, 1.0, Date),
            (Boolean, True, TimeInterval),
            (Boolean, True, Date),
            (Boolean, True, TimePeriod),
            (Boolean, True, Duration),
            (Date, "2020-01-01", Integer),
            (Date, "2020-01-01", Number),
            (Date, "2020-01-01", Boolean),
            (Date, "2020-01-01", Duration),
            (TimeInterval, "2020-01-01/2020-12-31", Integer),
            (TimeInterval, "2020-01-01/2020-12-31", Number),
            (TimeInterval, "2020-01-01/2020-12-31", Boolean),
            (TimeInterval, "2020-01-01/2020-12-31", Duration),
            (Duration, "A", Integer),
            (Duration, "A", Number),
            (Duration, "A", Boolean),
            (Duration, "A", TimeInterval),
            (Duration, "A", Date),
            (Duration, "A", TimePeriod),
        ],
    )
    def test_infeasible_cast_raises(self, from_type: type, from_val: object, to_type: type) -> None:
        with pytest.raises(SemanticError):
            Cast.validate(operand=Scalar("x", from_type, from_val), scalarType=to_type)


# ===========================================================================
# Mask error tests: mask where not allowed
# ===========================================================================


class TestCastMaskNotAllowed:
    """Test 1-1-5-5: mask provided where not supported."""

    @pytest.mark.parametrize(
        "from_type, from_val, to_type",
        [
            (Boolean, True, Integer),
            (Integer, 1, Number),
            (String, "hello", String),
            (String, "true", Boolean),
        ],
    )
    def test_mask_not_allowed_raises(
        self, from_type: type, from_val: object, to_type: type
    ) -> None:
        with pytest.raises(SemanticError):
            Cast.validate(
                operand=Scalar("x", from_type, from_val), scalarType=to_type, mask="FORMAT"
            )


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

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("40.000", "DD.DDD", 40.0),
            ("3.14", "D.DD", 3.14),
            ("-1.50", "D.DD", -1.5),
            ("+2.50", "D.DD", 2.5),
            (None, "DD.DDD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", String, input_val), Number, mask=mask)
        assert result.value == expected
        assert result.data_type == Number

    @pytest.mark.parametrize(
        "input_val, mask",
        [("4.00", "DD.DDD"), ("ab.cde", "DD.DDD")],
    )
    def test_mismatch_raises(self, input_val: str, mask: str) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, input_val), Number, mask=mask)


class TestCastStringToIntegerWithMask:
    """String -> Integer with mask (DDD style)."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [("123", "DDD", 123), ("045", "DDD", 45), (None, "DDD", None)],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", String, input_val), Integer, mask=mask)
        assert result.value == expected
        assert result.data_type == Integer

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "42"), Integer, mask="DDD")


class TestCastNumberToStringWithMask:
    """Number -> String with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            (40.0, "DD.DDD", "40.000"),
            (-1.5, "D.DD", "-1.50"),
            (None, "DD.DDD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", Number, input_val), String, mask=mask)
        assert result.value == expected
        assert result.data_type == String


class TestCastIntegerToStringWithMask:
    """Integer -> String with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            (42, "DDD", "042"),
            (-7, "DD", "-07"),
            (None, "DDD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", Integer, input_val), String, mask=mask)
        assert result.value == expected
        assert result.data_type == String


class TestCastStringToDateWithMask:
    """String -> Date with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("2022-01-01", "YYYY-MM-DD", "2022-01-01"),
            ("01012022", "DDMMYYYY", "2022-01-01"),
            ("  2022-01-01  ", "YYYY-MM-DD", "2022-01-01"),
            (None, "YYYY-MM-DD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", String, input_val), Date, mask=mask)
        assert result.value == expected
        assert result.data_type == Date

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, "01-01-2022"), Date, mask="YYYY-MM-DD")


class TestCastDateToStringWithMask:
    """Date -> String with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("2022-01-01", "YYYY-MM-DD", "2022-01-01"),
            ("2022-01-01", "DDMMYYYY", "01012022"),
            ("2022-01-01", "YY-MM-DD", "22-01-01"),
            ("2022-02-01", "DD-MM-YYYY", "01-02-2022"),
            ("2022-02-01", "MM-DD-YYYY", "02-01-2022"),
            (None, "YYYY-MM-DD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", Date, input_val), String, mask=mask)
        assert result.value == expected
        assert result.data_type == String


class TestCastStringToTimeIntervalWithMask:
    """String -> Time (TimeInterval) with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("2022-05-21/2023-05-21", "YYYY-MM-DD/YYYY-MM-DD", "2022-05-21/2023-05-21"),
            ("20220101/20221231", "YYYYMMDD/YYYYMMDD", "2022-01-01/2022-12-31"),
            (None, "YYYY-MM-DD/YYYY-MM-DD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", String, input_val), TimeInterval, mask=mask)
        assert result.value == expected
        assert result.data_type == TimeInterval

    def test_mismatch_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(
                Scalar("x", String, "01-01-2022/31-12-2022"),
                TimeInterval,
                mask="YYYY-MM-DD/YYYY-MM-DD",
            )


class TestCastTimeIntervalToStringWithMask:
    """TimeInterval -> String with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("2022-05-21/2023-05-21", "YYYY-MM-DD/YYYY-MM-DD", "2022-05-21/2023-05-21"),
            ("2022-05-21/2023-05-21", "MM-DD-YYYY/MM-DD-YYYY", "05-21-2022/05-21-2023"),
            ("2022-05-21/2023-05-21", "DD-MM-YYYY/DD-MM-YYYY", "21-05-2022/21-05-2023"),
            ("2022-01-01/2022-12-31", "YYYYMMDD/YYYYMMDD", "20220101/20221231"),
            (None, "YYYY-MM-DD/YYYY-MM-DD", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", TimeInterval, input_val), String, mask=mask)
        assert result.value == expected
        assert result.data_type == String


class TestCastStringToTimePeriodWithMask:
    """String -> TimePeriod with mask - all indicator styles."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("2000Q1", r"YYYY\QQ", "2000-Q1"),
            ("2000-Q1", r"YYYY-\QQ", "2000-Q1"),
            ("2000-1", "YYYY-Q", "2000-Q1"),
            ("Q1-2000", r"\QQ-YYYY", "2000-Q1"),
            ("2000Q01", r"YYYY\QQQ", "2000-Q1"),
            ("2000M03", r"YYYY\MMM", "2000-M03"),
            ("2000", "YYYY", "2000A"),
            (None, r"YYYY\QQ", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", String, input_val), TimePeriod, mask=mask)
        assert result.value == expected
        assert result.data_type == TimePeriod

    @pytest.mark.parametrize(
        "input_val, mask",
        [
            ("2000.01.01", r"YYYY\.MM\.DD"),
            ("2000M01D01", r"YYYY\MMM\DDD"),
        ],
    )
    def test_calendar_date(self, input_val: str, mask: str) -> None:
        from vtlengine.DataTypes.TimeHandling import TimePeriodHandler

        result = Cast.evaluate(Scalar("x", String, input_val), TimePeriod, mask=mask)
        assert result.value == str(TimePeriodHandler("2000D1"))

    @pytest.mark.parametrize(
        "input_val, mask",
        [("2000X1", r"YYYY\QQ"), ("2000Q1extra", r"YYYY\QQ")],
    )
    def test_mismatch_raises(self, input_val: str, mask: str) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(Scalar("x", String, input_val), TimePeriod, mask=mask)


class TestCastTimePeriodToStringWithMask:
    """TimePeriod -> String with mask."""

    @pytest.mark.parametrize(
        "input_val, mask, expected",
        [
            ("2000-Q1", r"YYYY\QQ", "2000Q1"),
            ("2000-Q1", r"YYYY-\QQ", "2000-Q1"),
            ("2000-M03", r"YYYY\MMM", "2000M03"),
            ("2000A", "YYYY", "2000"),
            (None, r"YYYY\QQ", None),
        ],
    )
    def test_cast(self, input_val: object, mask: str, expected: object) -> None:
        result = Cast.evaluate(Scalar("x", TimePeriod, input_val), String, mask=mask)
        assert result.value == expected
        assert result.data_type == String


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
    def _execute_expression(expr: str) -> Scalar:
        warnings.filterwarnings("ignore", category=FutureWarning)
        expression = f"DS_r := {expr};"
        ast = create_ast(expression)
        interpreter = InterpreterAnalyzer({})
        result = interpreter.visit(ast)
        return result["DS_r"]

    @pytest.mark.parametrize(
        "expr, expected_value, expected_type",
        [
            # Basic casts without mask
            ("cast(5, boolean)", True, Boolean),
            ("cast(0, boolean)", False, Boolean),
            ("cast(3.14, boolean)", True, Boolean),
            ("cast(true, integer)", 1, Integer),
            ("cast(false, integer)", 0, Integer),
            ("cast(true, number)", 1.0, Number),
            ("cast(false, number)", 0.0, Number),
            ('cast("true", boolean)', True, Boolean),
            ('cast("false", boolean)', False, Boolean),
            ('cast("hello", boolean)', False, Boolean),
            ("cast(42.0, integer)", 42, Integer),
            ("cast(42.5, string)", "42.5", String),
            ("cast(42, string)", "42", String),
            ('cast("42", integer)', 42, Integer),
            ('cast("3.14", number)', 3.14, Number),
            ('cast("A", duration)', "A", Duration),
            # Casts with mask
            ('cast("3.14", number, "D.DD")', 3.14, Number),
            ('cast("042", integer, "DDD")', 42, Integer),
            ('cast(3.14, string, "D.DDD")', "3.140", String),
            ('cast(42, string, "DDD")', "042", String),
            ('cast("2022-01-01", date, "YYYY-MM-DD")', "2022-01-01", Date),
            ('cast("01012022", date, "DDMMYYYY")', "2022-01-01", Date),
            ('cast(cast("2022-01-01", date), string, "MM-DD-YYYY")', "01-01-2022", String),
            ('cast(cast("2000-Q1", time_period), string, "YYYY\QQ")', "2000Q1", String),
        ],
    )
    def test_cast(self, expr: str, expected_value: object, expected_type: type) -> None:
        result = self._execute_expression(expr)
        assert result.value == expected_value
        assert result.data_type == expected_type

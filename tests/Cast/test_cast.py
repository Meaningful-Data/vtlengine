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


class CastExplicitWithoutMask(CastHelper):
    """ """

    classTest = "Cast.CastExplicitWithoutMask"

    def test_GL_461_1(self):
        """Cast with mask raises NotImplementedError."""
        code = "GL_461_1"
        number_inputs = 1

        text = self.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = self.LoadInputs(code=code, number_inputs=number_inputs)
        interpreter = InterpreterAnalyzer(datasets=input_datasets)
        with pytest.raises(NotImplementedError):
            interpreter.visit(ast)

    def test_GL_563_1(self):
        """
        Solves bug report in github issue #296
        """
        code = "GL_563_1"
        number_inputs = 1
        reference_names = ["1", "2"]

        self.BaseTest(code, number_inputs, references_names=reference_names)

    def test_GH_537_1(self):
        """
        Solves bug report in github issue #537: sub fails whith scalar casting
        """
        code = "GH_537_1"
        number_inputs = 1
        reference_names = ["1"]

        self.BaseTest(code, number_inputs, references_names=reference_names)


# ===========================================================================
# Comprehensive explicit cast tests (VTL 2.2) - Without mask
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
        [
            ("+123", 123),
            ("-0456", -456),
            ("789", 789),
            ("123", 123),
            ("+456", 456),
            ("-789", -789),
            ("0", 0),
            (None, None),
        ],
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
            ("+123.45", 123.45),
            ("1.23E5", 123000.0),
            ("-0.001", -0.001),
            ("3.14", 3.14),
            ("123", 123.0),
            ("+456", 456.0),
            ("-789", -789.0),
            ("1.5E3", 1500.0),
            ("1.5E-3", 0.0015),
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
        [
            ("2024-03-15", "2024-03-15"),
            ("2024-01-01", "2024-01-01"),
            ("2025-01-15", "2025-01-15"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=Date)
        assert result.value == expected
        assert result.data_type == Date


class TestCastStringToTimePeriod:
    """String -> TimePeriod (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2024-01-01/2024-12-31", "2024A"),
            ("2024-01-01/2024-03-31", "2024-Q1"),
            ("2020A", "2020A"),
            ("2020Q1", "2020Q1"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=TimePeriod)
        assert result.value == expected
        assert result.data_type == TimePeriod

    def test_irregular_interval_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(
                operand=Scalar("x", String, "2020-01-15/2020-03-20"), scalarType=TimePeriod
            )


class TestCastStringToTimeInterval:
    """String -> Time (TimeInterval) (Explicit without mask)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2024-01-01/2024-01-31", "2024-01-01/2024-01-31"),
            ("2020-01-01/2020-12-31", "2020-01-01/2020-12-31"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, input_val), scalarType=TimeInterval)
        assert result.value == expected
        assert result.data_type == TimeInterval


class TestCastStringToDuration:
    """String -> Duration (Explicit without mask)"""

    @pytest.mark.parametrize("shortcode", ["A", "D", "M", "Q", "W", "S"])
    def test_shortcode(self, shortcode: str) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, shortcode), scalarType=Duration)
        assert result.value == shortcode
        assert result.data_type == Duration

    @pytest.mark.parametrize(
        "iso_input, expected_shortcode",
        [("P1Y", "A"), ("P6M", "S"), ("P3M", "Q"), ("P1M", "M"), ("P1W", "W"), ("P1D", "D")],
    )
    def test_iso8601(self, iso_input: str, expected_shortcode: str) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, iso_input), scalarType=Duration)
        assert result.value == expected_shortcode
        assert result.data_type == Duration

    def test_null_returns_null(self) -> None:
        result = Cast.evaluate(operand=Scalar("x", String, None), scalarType=Duration)
        assert result.value is None

    def test_invalid_iso_raises(self) -> None:
        with pytest.raises(RunTimeError):
            Cast.evaluate(operand=Scalar("x", String, "P2Y"), scalarType=Duration)


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
    """Integer -> String (Explicit without mask): digits, optional -, no separators"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(-123, "-123"), (0, "0"), (456789, "456789"), (42, "42"), (-7, "-7"), (None, None)],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Integer, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastNumberToString:
    """Number -> String (Explicit without mask): decimal notation with . separator"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [(-123.45, "-123.45"), (0.001, "0.001"), (3.14, "3.14"), (5.0, "5.0"), (None, None)],
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
    """Date -> String (Explicit without mask): ISO-8601 YYYY-MM-DD"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2024-03-15", "2024-03-15"),
            ("2024-01-01", "2024-01-01"),
            ("2025-01-15", "2025-01-15"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", Date, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastTimePeriodToString:
    """TimePeriod -> String (Explicit without mask, VTL representation)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2024A", "2024A"),
            ("2024-Q1", "2024-Q1"),
            ("2020A", "2020A"),
            ("2020Q1", "2020Q1"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimePeriod, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastTimeIntervalToString:
    """TimeInterval -> String (Explicit without mask): ISO-8601 start/end"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("2024-01-01/2024-01-31", "2024-01-01/2024-01-31"),
            ("2020-01-01/2020-12-31", "2020-01-01/2020-12-31"),
            (None, None),
        ],
    )
    def test_cast(self, input_val: object, expected: object) -> None:
        result = Cast.evaluate(operand=Scalar("x", TimeInterval, input_val), scalarType=String)
        assert result.value == expected
        assert result.data_type == String


class TestCastDurationToString:
    """Duration -> String (Explicit without mask, ISO-8601 output)"""

    @pytest.mark.parametrize(
        "input_val, expected",
        [
            ("A", "P1Y"),
            ("S", "P6M"),
            ("Q", "P3M"),
            ("M", "P1M"),
            ("W", "P1W"),
            ("D", "P1D"),
            (None, None),
        ],
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
# Mask raises NotImplementedError
# ===========================================================================


class TestCastMaskNotImplemented:
    """Any cast with mask raises NotImplementedError."""

    @pytest.mark.parametrize(
        "operand, to_type, mask",
        [
            (Scalar("x", String, "42"), Integer, "DD"),
            (Scalar("x", String, "3.14"), Number, "D.DD"),
            (Scalar("x", Integer, 42), String, "DD"),
            (Scalar("x", String, "2020-01-01"), Date, "YYYY-MM-DD"),
            (Scalar("x", Date, "2020-01-01"), String, "YYYY-MM-DD"),
            (Scalar("x", String, "A"), Duration, "A"),
            (Scalar("x", Duration, "A"), String, "A"),
        ],
    )
    def test_mask_raises_not_implemented(self, operand: Scalar, to_type: type, mask: str) -> None:
        with pytest.raises(NotImplementedError):
            Cast.validate(operand=operand, scalarType=to_type, mask=mask)


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
            # date → time_period
            ('cast(cast("2020-01-15", date), time_period)', "2020D15", TimePeriod),
            # time_period → date (daily period only)
            ('cast(cast("2020D15", time_period), date)', "2020-01-15", Date),
            # time (time_interval) → time_period
            ('cast(cast("2020-01-01/2020-12-31", time), time_period)', "2020A", TimePeriod),
            # time (time_interval) → date (single-date interval only)
            ('cast(cast("2020-01-15/2020-01-15", time), date)', "2020-01-15", Date),
        ],
    )
    def test_cast(self, expr: str, expected_value: object, expected_type: type) -> None:
        result = self._execute_expression(expr)
        assert result.value == expected_value
        assert result.data_type == expected_type

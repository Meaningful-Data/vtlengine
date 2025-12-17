import warnings
from pathlib import Path

import pytest

from tests.Helper import TestHelper
from vtlengine.API import create_ast
from vtlengine.DataTypes import Date, Duration, Number, String, TimeInterval, TimePeriod
from vtlengine.Exceptions import SemanticError
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


evaluate_params = [
    (
        Scalar("40.000", String, "40.000"),
        Number,
        "DD.DDD",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2022-01-01", String, "2022-01-01"),
        Date,
        "YYYY-MM-DD",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2023-01-12", String, "2023-01-12"),
        Date,
        "\\PY\\YDDD\\D",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2000Q1", String, "2000Q1"),
        TimePeriod,
        "YYYY\\QQ",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2022-05-21/2023-05-21", String, "2022-05-21/2023-05-21"),
        TimeInterval,
        "YYYY-MM-DD/YYYY-MM-DD",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2023-02-05", String, "2023-02-05"),
        Duration,
        "P0Y240D",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2021-12-21", Date, "2021-12-21"),
        String,
        "YYYY-MM-DD hh:mm:ss",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        Scalar("2022-05-21/2023-05-21", TimeInterval, "2022-05-21/2023-05-21"),
        String,
        "YYYY-MM-DD/YYYY-MM-DD",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
]

cast_error_params = [
    (
        "40.000",
        String,
        Number,
        "DD.DDD",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        "2022-01-01",
        String,
        Date,
        "YYYY-MM-DD",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        "2023-01-12",
        String,
        Date,
        "\\PY\\YDDD\\D",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        "2000Q1",
        String,
        TimePeriod,
        "YYYY\\QQ",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        "2022-05-21/2023-05-21",
        String,
        TimeInterval,
        "YYYY-MM-DD/YYYY-MM-DD",
        NotImplementedError,
        "How this cast should be implemented is not yet defined.",
    ),
    (
        "2023-02-05",
        String,
        Duration,
        "P0Y240D",
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        "40.000",
        Number,
        String,
        "DD.DDD",
        SemanticError,
        "Impossible to cast 40.000 from type Number to String",
    ),
]
test_params = [
    (
        'cast("40.000", number, "DD.DDD")',
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        'cast("2022-01-01", date, "YYYY-MM-DD")',
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        r'cast("2023-01-12", date, "\\PY\\YDDD\\D")',
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        r'cast ("2000Q1", time_period, "YYYY\\QQ")',
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        'cast ("2022-05-21/2023-05-21", time, "YYYY-MM-DD/YYYY-MM-DD")',
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
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


@pytest.mark.parametrize("text, type_of_error, exception_message", test_params)
def test_errors_validate_cast_scalar(text, type_of_error, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(type_of_error, match=f"{exception_message}"):
        interpreter.visit(ast)


@pytest.mark.parametrize(
    "value, provided_type, to_type, mask, type_of_error, exception_message",
    cast_error_params,
)
def test_errors_cast_scalar(value, provided_type, to_type, mask, type_of_error, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    with pytest.raises(type_of_error, match=exception_message):
        Cast.cast_value(value, provided_type=provided_type, to_type=to_type, mask_value=mask)


@pytest.mark.parametrize(
    "operand, scalar_type, mask, type_of_error, exception_message", evaluate_params
)
def test_errors_cast_scalar_evaluate(operand, scalar_type, mask, type_of_error, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    with pytest.raises(type_of_error, match=exception_message):
        Cast.evaluate(operand=operand, scalarType=scalar_type, mask=mask)

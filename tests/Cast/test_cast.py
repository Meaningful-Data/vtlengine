import warnings
from pathlib import Path

import pytest

from tests.Helper import TestHelper
from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer
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
        r'cast("2023-01-12", date, "\PY\YDDD\D")',
        NotImplementedError,
        "How this mask should be implemented is not yet defined.",
    ),
    (
        r'cast ("2000Q1", time_period, "YYYY\QQ")',
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
        "(\"A mask can't be provided to cast from type String to String. Mask provided: YYYY-MM-DD hh:mm:ss. Please check transformation with output dataset DS_r\", '1-1-5-5')",
    ),
    (
        'cast("P0Y240D", string, "YYYY-MM-DD hh:mm:ss")',
        SemanticError,
        "(\"A mask can't be provided to cast from type String to String. Mask provided: YYYY-MM-DD hh:mm:ss. Please check transformation with output dataset DS_r\", '1-1-5-5')",
    ),
    (
        'cast ("2022-05-21/2023-05-21", string, "YYYY-MM-DD/YYYY-MM-DD")',
        SemanticError,
        "(\"A mask can't be provided to cast from type String to String. Mask provided: YYYY-MM-DD/YYYY-MM-DD. Please check transformation with output dataset DS_r\", '1-1-5-5')",
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


@pytest.mark.parametrize("text, type_of_error, exception_message", test_params)
def test_errors_cast_scalar(text, type_of_error, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(type_of_error, match=f"{exception_message}"):
        interpreter.visit(ast)


def test_errors_cast_scalar_evaluate(operand, scalar_type, mask, type_of_error, exception_message):
    with pytest.raises(type_of_error, match=f"{exception_message}"):
        Cast.evaluate(operand=operand, scalarType=scalar_type, mask=mask)

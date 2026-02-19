import pytest

from vtlengine.API import create_ast
from vtlengine.AST import Assignment, Constant
from vtlengine.DataTypes import Date, TimeInterval, TimePeriod
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer


def _get_constant_node(script: str) -> Constant:
    """Parse a VTL script and extract the Constant node from the first assignment's RHS."""
    ast = create_ast(script)
    assignment = ast.children[0]
    assert isinstance(assignment, Assignment)
    constant = assignment.right
    assert isinstance(constant, Constant)
    return constant


# ---------- AST constant type detection ----------

detection_params = [
    # Date constants
    ("2020-01-15", "DATE_CONSTANT"),
    ("2020-12-31", "DATE_CONSTANT"),
    ("2020-01-01", "DATE_CONSTANT"),
    # Bare 4-digit string stays STRING_CONSTANT (ambiguous with codes)
    ("2020", "STRING_CONSTANT"),
    # TimePeriod constants — VTL compact formats
    ("2020A", "TIME_PERIOD_CONSTANT"),
    ("2020Q1", "TIME_PERIOD_CONSTANT"),
    ("2020S1", "TIME_PERIOD_CONSTANT"),
    ("2020M1", "TIME_PERIOD_CONSTANT"),
    ("2020M12", "TIME_PERIOD_CONSTANT"),
    ("2020W01", "TIME_PERIOD_CONSTANT"),
    ("2020W53", "TIME_PERIOD_CONSTANT"),
    ("2020D1", "TIME_PERIOD_CONSTANT"),
    ("2020D366", "TIME_PERIOD_CONSTANT"),
    # TimePeriod constants — SDMX hyphenated formats
    ("2020-01", "TIME_PERIOD_CONSTANT"),
    ("2020-M01", "TIME_PERIOD_CONSTANT"),
    ("2020-Q1", "TIME_PERIOD_CONSTANT"),
    ("2020-S1", "TIME_PERIOD_CONSTANT"),
    ("2020-W01", "TIME_PERIOD_CONSTANT"),
    ("2020-D001", "TIME_PERIOD_CONSTANT"),
    # TimeInterval constants
    ("2020-01-01/2020-12-31", "TIME_INTERVAL_CONSTANT"),
    ("2020-06-15/2020-06-30", "TIME_INTERVAL_CONSTANT"),
    # String constants — not auto-detected
    ("hello", "STRING_CONSTANT"),
    ("123abc", "STRING_CONSTANT"),
    ("A", "STRING_CONSTANT"),
    ("M", "STRING_CONSTANT"),
    ("D", "STRING_CONSTANT"),
    ("W", "STRING_CONSTANT"),
]


@pytest.mark.parametrize("value, expected_type", detection_params)
def test_constant_detection(value: str, expected_type: str) -> None:
    """Verify that string constants are auto-detected as the correct AST type."""
    script = f'DS_r <- "{value}";'
    constant = _get_constant_node(script)
    assert constant.type_ == expected_type
    assert constant.value == value


def test_number_constant_rename() -> None:
    """Verify that float literals create NUMBER_CONSTANT (not FLOAT_CONSTANT)."""
    constant = _get_constant_node("DS_r <- 1.5;")
    assert constant.type_ == "NUMBER_CONSTANT"
    assert constant.value == 1.5


def test_integer_constant_unchanged() -> None:
    """Verify that bare integer 2020 stays as INTEGER_CONSTANT (not TimePeriod)."""
    constant = _get_constant_node("DS_r <- 2020;")
    assert constant.type_ == "INTEGER_CONSTANT"
    assert constant.value == 2020


# ---------- Invalid time constants raise SemanticError ----------

invalid_params = [
    "2020Q5",
    "2020S3",
    "2020-13-01",
    "2020-00-01",
    "2020-02-30",
]


@pytest.mark.parametrize("value", invalid_params)
def test_invalid_time_constants(value: str) -> None:
    """Verify that format-correct but value-wrong constants raise SemanticError."""
    script = f'DS_r <- "{value}";'
    with pytest.raises(SemanticError):
        create_ast(script)


# ---------- Interpreter creates correctly typed Scalars ----------

interpreter_type_params = [
    ("2020-01-15", Date),
    ("2020Q1", TimePeriod),
    ("2020-01-01/2020-12-31", TimeInterval),
]


@pytest.mark.parametrize("value, expected_data_type", interpreter_type_params)
def test_constant_interpreter_types(value: str, expected_data_type: type) -> None:
    """Verify that the Interpreter creates Scalars with correct data_type."""
    script = f'DS_r <- "{value}";'
    ast = create_ast(script)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    scalar = result["DS_r"]
    assert scalar.data_type == expected_data_type


# ---------- CAST still works on auto-detected constants ----------


def test_cast_with_auto_detected_constant() -> None:
    """Verify that CAST still processes through the full pipeline."""
    script = 'DS_r <- cast("2020Q1", time_period);'
    ast = create_ast(script)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    scalar = result["DS_r"]
    assert scalar.data_type == TimePeriod

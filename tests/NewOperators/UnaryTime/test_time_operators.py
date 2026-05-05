import warnings
from pathlib import Path

import pytest
from pytest import mark

from tests.NewOperators.conftest import run_expression, run_scalar_expression
from vtlengine.DataTypes import Integer
from vtlengine.Exceptions import RunTimeError, SemanticError

pytestmark = mark.input_path(Path(__file__).parent / "data")

ds_param = [
    ("1", "DS_r := DS_1 [calc Me_2 := getyear(Me_1)];"),
    ("2", "DS_r := DS_1[calc Me_2 := getmonth(Me_1)];"),
    ("3", "DS_r := DS_1[calc Me_2 := dayofmonth(Me_1)];"),
    ("4", "DS_r := DS_1[calc Me_2 := dayofyear(Me_1)];"),
    ("5", "DS_r := DS_1[calc Me_2 := daytomonth(Me_1)];"),
    ("6", "DS_r := DS_1[calc Me_2 := daytoyear(Me_1)];"),
    ("7", "DS_r := DS_1[calc Me_2 := monthtoday(Me_1)];"),
    ("8", "DS_r := DS_1[calc Me_2 := yeartoday(Me_1)];"),
]

error_param = [
    ("9", "DS_r := DS_1[calc Me_2 := daytomonth(Me_1)];", RunTimeError, "2-1-19-16"),
    ("10", "DS_r := DS_1[calc Me_2 := daytoyear(Me_1)];", RunTimeError, "2-1-19-16"),
    ("13", "DS_r := DS_1 [calc Me_2 := getyear(Me_1)];", SemanticError, "1-1-19-10"),
    ("14", "DS_r := DS_1 [calc Me_2 := getmonth(Me_1)];", SemanticError, "1-1-19-10"),
    ("15", "DS_r := DS_1 [calc Me_2 := dayofmonth(Me_1)];", SemanticError, "1-1-19-10"),
    ("16", "DS_r := DS_1 [calc Me_2 := dayofyear(Me_1)];", SemanticError, "1-1-19-10"),
]

scalar_time_params = [
    ('getyear(cast("2023-01-12", date))', 2023),
    ('getyear(cast("2022Q1", time_period))', 2022),
    ('getmonth(cast("2023-01-12", date))', 1),
    ('getmonth(cast("2022Q1", time_period))', 1),
    ('dayofmonth(cast("2023-01-12", date))', 12),
    ('dayofmonth(cast("2022Q1", time_period))', 31),
    ('dayofyear(cast("2023-01-12", date))', 12),
    ('dayofyear(cast("2022Q1", time_period))', 90),
]

scalar_time_error_params = [
    ('getyear(cast("2023-01-12/2024-01-03", date))', RunTimeError, "2-1-19-8"),
    ('getmonth(cast("2023-01-12/2024-02-15", date))', RunTimeError, "2-1-19-8"),
    ('dayofmonth(cast("2023-01-12/2024-02-02", date))', RunTimeError, "2-1-19-8"),
    ('dayofyear(cast("2023-01-12/2024-03-06", date))', RunTimeError, "2-1-19-8"),
    ('getyear(cast("2023-01-12/2024-01-31", time))', SemanticError, "1-1-19-10"),
    ('getmonth(cast("2023-01-12/2024-03-25", time))', SemanticError, "1-1-19-10"),
    ('dayofmonth(cast("2023-01-12/2024-05-29", time))', SemanticError, "1-1-19-10"),
    ('dayofyear(cast("2023-01-12/2024-06-08", time))', SemanticError, "1-1-19-10"),
]


@pytest.mark.parametrize("text, reference", scalar_time_params)
def test_unary_time_scalar(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    result = run_scalar_expression(expression)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


@pytest.mark.parametrize("code, expression", ds_param)
def test_unary_time_ds(load_reference, input_paths, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = run_expression(expression, input_paths)
    assert result == load_reference


@pytest.mark.parametrize("code, expression, type_error, error_code", error_param)
def test_errors_ds(input_paths, code, expression, type_error, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    with pytest.raises(type_error) as context:
        run_expression(expression, input_paths)
    result = error_code == str(context.value.args[1])
    if result is False:
        print(f"\n{error_code} != {context.value.args[1]}")
    assert result


@pytest.mark.parametrize("text, exception_type, exception_message", scalar_time_error_params)
def test_errors_time_scalar(text, exception_type, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    with pytest.raises(exception_type, match=f".*{exception_message}"):
        run_scalar_expression(expression)

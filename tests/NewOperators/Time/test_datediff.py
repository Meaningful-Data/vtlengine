import warnings
from pathlib import Path

import pytest
from pytest import mark

from vtlengine.API import create_ast
from vtlengine.DataTypes import Integer
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

pytestmark = mark.input_path(Path(__file__).parent / "data")

ds_param = [
    ("21", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("22", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("23", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("24", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
]

error_param = [
    ("25", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];', "1-1-1-2"),
    ("26", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];', "2-3-6"),
]

scalar_time_params = [
    ('datediff(cast("2020-12-14", date),cast("2021-04-20", date))', 127),
    ('datediff(cast("2020-01-01",date),cast("2021-01-01",date))', 366),
    ('datediff(cast("2022Q1",time_period),cast("2023Q2",time_period))', 456),
    ('datediff(cast("2020D1",time_period),cast("2020D15",time_period))', 14),
]

scalar_time_error_params = [
    ('datediff(cast("2020-12-14", date),cast("2021-04-20", time_period))', "2-1-19-8"),
    ('datediff(cast("2020-01-01",time_period),cast("2021-01-01",date))', "2-1-19-8"),
    ('datediff(cast("2022Q1",date),cast("2023Q2",time_period))', "2-1-19-8"),
    ('datediff(cast("2020D1",time_period),cast("2020D15",date))', "2-1-19-8"),
]

@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(load_input, load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer(load_input)
    result = interpreter.visit(ast)
    assert result == load_reference

@pytest.mark.parametrize("text, reference", scalar_time_params)
def test_unary_time_scalar(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


@pytest.mark.parametrize("code, expression, error_code", error_param)
def test_errors(load_input, code, expression, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    datasets = load_input
    with pytest.raises(SemanticError) as context:
        ast = create_ast(expression)
        interpreter = InterpreterAnalyzer(datasets)
        interpreter.visit(ast)
    result = error_code == str(context.value.args[1])
    if result is False:
        print(f"\n{error_code} != {context.value.args[1]}")
    assert result

@pytest.mark.parametrize("text, exception_message", scalar_time_error_params)
def test_errors_time_scalar(text, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(SemanticError, match=f".*{exception_message}"):
        interpreter.visit(ast)

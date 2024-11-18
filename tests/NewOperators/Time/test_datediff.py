import pytest
import warnings

from pytest import mark
from pathlib import Path

from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

pytestmark = mark.input_path(Path(__file__).parent / "data")

ds_param = [
    ("21", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("22", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("23", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("24", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
    ("25", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];'),
]

error_param = [
    ("26", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_2)];', "1-1-1-2"),
    # ("27", 'DS_r := DS_1[calc Me_3 := datediff(Me_2, Me_1)];', "0-1-1-12"),
    # ("28", 'DS_r := DS_1[calc Me_3 := datediff(Me_1, Me_1)];', "2-3-6"),
]

@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(load_input, load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer(load_input)
    result = interpreter.visit(ast)
    assert result == load_reference


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
from pathlib import Path

import pytest
import warnings

from pytest import mark

from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    ("1", 'DS_r := DS_1[calc Me_1 := random(1, 1)];'),
    ("2", 'DS_r := DS_1[calc Me_1 := random(Id_1, 1)];'),
    ("3", 'DS_r := random(DS_1, 1);'),
    ("4", 'DS_r := random(DS_1, 1);'),
    ("5", 'DS_r := random(DS_1, 0);')
]

error_param = [
    ("6", 'DS_r := DS_1[calc Me_1 := random(Id_1, 0)];', "1-1-1-2"),
    ("7", 'DS_r := random(1, -1);', "2-1-15-2")
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

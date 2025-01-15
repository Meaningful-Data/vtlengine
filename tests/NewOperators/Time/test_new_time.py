import warnings
from pathlib import Path

import pytest
from pytest import mark

from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    ("1", 'DS_r := dateadd(DS_1, 69, "D");'),
    ("2", 'DS_r := dateadd(DS_1, 69, "D");'),
    ("3", 'DS_r := dateadd(DS_1, -13, "W");'),
    ("4", 'DS_r := dateadd(DS_1, -13, "W");'),
    ("5", 'DS_r := dateadd(DS_1, 16, "M");'),
    ("6", 'DS_r := dateadd(DS_1, 16, "M");'),
    ("7", 'DS_r := dateadd(DS_1, -5, "Q");'),
    ("8", 'DS_r := dateadd(DS_1, -5, "Q");'),
    ("9", 'DS_r := dateadd(DS_1, 5, "S");'),
    ("10", 'DS_r := dateadd(DS_1, 5, "S");'),
    ("11", 'DS_r := dateadd(DS_1, 9, "A");'),
    ("12", 'DS_r := dateadd(DS_1, 9, "A");'),
    ("13", 'DS_r := DS_1[calc Me_2 := dateadd(Me_1, 1889432, "D")];'),
    ("14", 'DS_r := DS_1[calc Me_2 := dateadd(Me_1, 1889432, "D")];'),
]

error_param = [
    ("15", 'DS_r := dateadd(DS_1, DS_1, "D");', "2-1-19-12"),
    ("16", 'DS_r := dateadd(DS_1, "D", "D");', "2-1-19-13"),
    ("17", "DS_r := dateadd(DS_1, 1, DS_1);", "2-1-19-12"),
    ("18", "DS_r := dateadd(DS_1, 1, 1);", "2-1-19-13"),
    ("19", 'DS_r := dateadd(DS_1, 1, "D");', "2-1-19-14"),
    ("20", 'DS_r := DS_1[calc Me_2 := dateadd(Me_1, 1, "D")];', "1-1-1-1"),
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

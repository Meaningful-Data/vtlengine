import warnings
from pathlib import Path

import pytest
from pytest import mark

from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    ("1", 'DS_r := DS_1 [calc Me_3 := case when Id_1 = 1 then "X" else Me_2];'),
    (
        "2",
        'DS_r := DS_1 [calc Me_3 := case when Id_1 = 1 then "X" when Id_2 = "B" then "Y" when Id_1 = 3 then "Z" else Me_2];',
    ),
    (
        "3",
        'DS_r := DS_1 [calc Me_3 := case when Id_1 = 1 then case when Id_2 = "A" then Id_2 else "Y" else case when Me_2 = "J" then Me_2 else null];',
    ),
    (
        "4",
        'DS_r := DS_1 [calc Me_3 := case when Id_1 = 1 then case when Id_2 = "A" then Id_2 when Id_2 = "B" then case when Me_2 = "B" then "X" else Me_1 else case when Id_1 = 3 then "U" when Id_1 = 2 then Id_1 else "Y" else case when Me_2 = "J" then Me_2 when Me_1 = 10 then "Z" when Me_1 = 9 then "I" when Id_1 = 5 then "W" else null];',
    ),
    (
        "5",
        'DS_r := DS_1 [calc Me_3 := case when Id_2 = "A" then case when Id_1 = 1 then if Me_1 = 1 then case when Me_2 = "A" then "Y" else Me_2 else Id_1 else Id_2 when Id_2 = "B" then Id_2 else case when Id_1 = 1 then Id_1 when Me_2 = Id_2 then Me_2 when Me_2 = "J" then "J" else if Me_1 = 4 then Id_2 else "X"];',
    ),
    (
        "6",
        'DS_r := DS_1 [calc Me_3 := case when Me_1 > 0 then "P" when Me_1 = 0 then null else "N"];',
    ),
    ("7", "DS_r := case when DS_cond then DS_2 else DS_1;"),
    ("8", "DS_r := case when DS_cond then DS_1 else null;"),
    ("9", "DS_r := case when DS_cond1 then DS_1 when DS_cond2 then DS_2 else null;"),
]

error_param = [
    (
        "10",
        "x := 1; DS_r := case when DS_cond then 1 when x = 2 then 2 else 0;",
        "2-1-9-1",
    ),
    (
        "11",
        "x := 1; DS_r := case when x = 1 then 1 when x = 2 then DS_1 else 0;",
        "2-1-9-3",
    ),
    ("12", "DS_r := DS_1 [calc Me_3 := case when Me_1 then 1 else 0];", "2-1-9-4"),
    ("13", "DS_r := case when DS_1 then DS_1 else null;", "2-1-9-5"),
    ("14", "DS_r := case when DS_cond1 then 1 else null;", "1-1-1-4"),
    ("15", "DS_r := case when DS_cond1 then 1 else null;", "2-1-9-6"),
    ("16", "DS_r := case when DS_cond1 then DS_1 else DS_2;", "2-1-9-7"),
    ("17", "DS_r := case when DS_cond1 then DS_1 else DS_2;", "2-1-9-7"),
    ("18", "x := 1; DS_r := case when x then 1 else 0;", "2-1-9-2"),
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

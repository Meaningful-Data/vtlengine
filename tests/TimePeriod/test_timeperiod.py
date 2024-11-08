import warnings
from pathlib import Path

import pytest
from pytest import mark

from vtlengine.API import create_ast
from vtlengine.Exceptions import SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer

pytestmark = mark.input_path(Path(__file__).parent / "data")


ds_param = [
    ("1", 'DS_r := DSD_EXR[filter TIME_PERIOD = cast("2002M1", time_period)];'),
    ("2", 'DS_r := DSD_EXR[filter TIME_PERIOD <> cast("2002M1", time_period)];'),
    ("3", 'DS_r := DSD_EXR[filter TIME_PERIOD < cast("2002M1", time_period)];'),
    ("4", 'DS_r := DSD_EXR[filter TIME_PERIOD > cast("2002M1", time_period)];'),
    ("5", 'DS_r := DSD_EXR[filter TIME_PERIOD <= cast("2002M1", time_period)];'),
    ("6", 'DS_r := DSD_EXR[filter TIME_PERIOD >= cast("2002M1", time_period)];'),
    ("GL_416", 'test2_1 := BE2_DF_NICP[filter FREQ = "M" and TIME_PERIOD = cast("2020-01", time_period)];'),
    ("GL_417_1", 'test := avg (BE2_DF_NICP group all time_agg ("Q", "M", TIME_PERIOD));'),
    ("GL_417_2", 'test := avg (BE2_DF_NICP group all time_agg ("A", "M", TIME_PERIOD));'),
    ("GL_417_4", 'test := avg (BE2_DF_NICP group all time_agg ("A", "Q", TIME_PERIOD));'),
    ("GL_418", 'test2_1 := BE2_DF_NICP[sub DERIVATION = "INDICES"][filter FREQ = "M"][keep OBS_VALUE]; \
                test2_2 := timeshift(test2_1,-12); \
                test2_result <- inner_join(test2_1[rename OBS_VALUE to CURRENT] as C, test2_2 \
                    [rename OBS_VALUE to PREVIOUS] as P calc GROWTH :=(CURRENT - PREVIOUS) / PREVIOUS * 100, \
                    identifier DERIVATION := "GROWTH_RATE");'
    ),
    ("GL_421_1", 'test2_1 := BE2_DF_NICP[calc FREQ_2 := TIME_PERIOD in {cast("2020-01", time_period), cast("2021-01", time_period)}];'),
    # ("GL_421_2", 'test := avg (BE2_DF_NICP group all time_agg ("A", "M", TIME_PERIOD));'),
    ("GL_440_1", 'DS_r := DS_1;'),

    ("GL_462_1", 'added := demo_data_structure;'),
    ("GL_462_2", 'added := demo_data_structure; DS_r := added+ ds_2;'),
    ("GL_462_3", 'sc_result := sc_1;'),
    ("GL_462_4", 'DS_r := ds_2;'),
]

error_param = [
    ("GL_440_2", 'DS_r := DS_1;', "0-1-1-12"),
]


@pytest.mark.parametrize("code, expression", ds_param)
def test_case_ds(load_input, load_reference, code, expression):
    warnings.filterwarnings("ignore", category=FutureWarning)
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer(load_input)
    result = interpreter.visit(ast)
    assert result == load_reference


@pytest.mark.parametrize("code, expression, error_code", error_param)
def test_errors(load_error, code, expression, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    result = error_code == load_error
    if result is False:
        print(f"\n{error_code} != {load_error}")
    assert result

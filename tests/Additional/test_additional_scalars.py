import warnings
from pathlib import Path

import pandas as pd
import pytest

from tests.Helper import TestHelper
from vtlengine.API import create_ast, run
from vtlengine.DataTypes import Boolean, Integer, Number, String
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer
from vtlengine.Model import Scalar


class AdditionalScalarsTests(TestHelper):
    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"

    ds_input_prefix = "DS_"

    warnings.filterwarnings("ignore", category=FutureWarning)


string_params = [
    ("substr(null, null, null)", None),
    ("substr(null)", None),
    ('substr("abc", null, null)', "abc"),
    ("substr(null, 1, 2)", None),
    ("substr(null, _, 2)", None),
    ('substr("abc", null)', "abc"),
    ('substr("abc", null, 2)', "ab"),
    ('substr("abc", 3, null)', "c"),
    ('substr("abcdefghijklmnopqrstuvwxyz")', "abcdefghijklmnopqrstuvwxyz"),
    ('substr("abcdefghijklmnopqrstuvwxyz", 1)', "abcdefghijklmnopqrstuvwxyz"),
    ('substr("abcdefghijklmnopqrstuvwxyz", _, 3)', "abc"),
    ('substr("abcdefghijklmnopqrstuvwxyz", 300)', ""),
    ('substr("abcdefghijklmnopqrstuvwxyz", 300, 4)', ""),
    ('substr("abcdefghijklmnopqrstuvwxyz", 2, 300)', "bcdefghijklmnopqrstuvwxyz"),
    ('substr("abcdefghijklmnopqrstuvwxyz", _, 300)', "abcdefghijklmnopqrstuvwxyz"),
    ('substr("abcdefghijklmnopqrstuvwxyz", 400, 200)', ""),
    ('substr("", 4, 2)', ""),
    ("replace(null, null, null)", None),
    ("replace(null, null)", None),
    ('replace("abc", null, null)', ""),
    ('replace("abc", null)', ""),
    ('replace(null, "a", "b")', None),
    ('replace(null, null, "b")', None),
    ('replace(null, "a", null)', None),
    ('replace("abc", null, "b")', ""),
    ('replace("abc", "a", null)', "bc"),
    ('replace("Hello world", "Hello", "Hi")', "Hi world"),
    ('replace("Hello world", "Hello")', " world"),
    ('replace ("Hello", "ello", "i")', "Hi"),
]

instr_op_params = [
    ('instr("abcde", "c")', 3),
    ('instr("abcdecfrxcwsd", "c", _, 3)', 10),
    ('instr("abcdecfrxcwsd", "c", 5, 3)', 0),
    ('instr("abcdecfrxcwsd", "c", 5)', 6),
    ('instr("abcde", "x")', 0),
    ('instr("abcde", "a", 67)', 0),
    ('instr("abcde", "a", 1, 67)', 0),
    ("instr(null, null, null, null)", None),
    ('instr(null, "a")', None),
    ('instr("abc", "a", null)', 1),
    ('instr("abc", "a", 1, null)', 1),
    ('instr("abc", "a", null, 3)', 0),
    ('instr("abc", "a", null, null)', 1),
]

numeric_params = [
    ("+null", None),
    ("-null", None),
    ("ceil(null)", None),
    ("floor(null)", None),
    ("abs(null)", None),
    ("exp(null)", None),
    ("ln(null)", None),
    ("sqrt(null)", None),
    ("2 + null", None),
    ("null + 2.0", None),
    ("2 - null", None),
    ("null - 2.0", None),
    ("2 * null", None),
    ("null * 2.0", None),
    ("2 / null", None),
    ("null / 2", None),
    ("2 + 3.3", 5.3),
    ("3.3 + 2", 5.3),
    ("2 - 3.3", -1.3),
    ("3.3 - 2.0", 1.3),
    ("2 * 3.3", 6.6),
    ("3.3 * 2", 6.6),
    ("2 / 1.0", 2),
    ("1.0 / 2", 0.5),
    ("round(null, 0)", None),
    ("round(null)", None),
    ("round(null, 3)", None),
    ("round(null, _)", None),
    ("round(null, null)", None),
    ("round(5.0, null)", 5),
    ("round(3.14159, 2)", 3.14),
    ("round(3.14159, _)", 3),
    ("round(3.14159, 4)", 3.1416),
    ("round(12345.6, 0)", 12346.0),
    ("round(12345.6)", 12346),
    ("round(12345.6, _)", 12346),
    ("round(12345.6, -1)", 12350.0),
    ("trunc(null, 0)", None),
    ("trunc(null)", None),
    ("trunc(null, 3)", None),
    ("trunc(null, _)", None),
    ("trunc(null, null)", None),
    ("trunc(4.0, null)", 4),
    ("trunc(3.14159, 2)", 3.14),
    ("trunc(3.14159, _)", 3),
    ("trunc(3.14159, 4)", 3.1415),
    ("trunc(12345.6, 0)", 12345),
    ("trunc(12345.6)", 12345),
    ("trunc(12345.6, _)", 12345),
    ("trunc(12345.6, -1)", 12340.0),
    ("power(5, 2)", 25),
    ("power(5, 1)", 5),
    ("power(5, 0)", 1),
    ("power(5, -1)", 0.2),
    ("power(-5, 3)", -125),
    ("power(null, null)", None),
    ("power(null, 1)", None),
    ("power(1, null)", None),
    ("log(8, 2)", 3.0),
    ("log(8.0, 2)", 3.0),
    ("log(1024, 2)", 10.0),
    ("log(1024, 10)", 3.01029996),
    ("log(2.0, 2)", 1.0),
    ("log(null, null)", None),
    ("log(null, 1)", None),
    ("log(1, null)", None),
    ("log(0.5, 6)", -0.38685281),
    ("(1 + 2) / 3", 1.0),
    ("random(12, 2)", 0.66641),
]

boolean_params = [
    ("false and false", False),
    ("false and true", False),
    ("false and null", False),
    ("true and false", False),
    ("true and true", True),
    ("true and null", None),
    ("null and null", None),
    ("false or false", False),
    ("false or true", True),
    ("false or null", None),
    ("true or false", True),
    ("true or true", True),
    ("true or null", True),
    ("null or null", None),
    ("false xor false", False),
    ("false xor true", True),
    ("false xor null", None),
    ("true xor false", True),
    ("true xor true", False),
    ("true xor null", None),
    ("null xor null", None),
    ("not false", True),
    ("not true", False),
    ("not null", None),
]

comparison_params = [
    ("3 = null", None),
    ("3 <> null", None),
    ("3 < null", None),
    ("3 > null", None),
    ("3 <= null", None),
    ("3 >= null", None),
    ("3 in { null }", None),
    ("not (3 in { null })", None),
    ("not (null in { 1,2,3 })", None),
    ("between(null, 4, 5)", None),
    ("between(5, null, 5)", None),
    ("between(4, 4, null)", None),
    ("between(null, null, null)", None),
    ('between("a", "a", "z")', True),
    ('between("z", "a", "c")', False),
    ("between(6, 1, 9)", True),
    ("between(12, 1, 9)", False),
]

string_exception_param = [
    ('substr("asdf", -3)', "1-1-18-4"),
    ('substr("asdf", 0)', "1-1-18-4"),
    ('substr("asdf", -2, 3)', "1-1-18-4"),
    ('substr("asdf", 0, 5)', "1-1-18-4"),
    ('substr("asdf", 1, -9)', "1-1-18-4"),
    ('substr("asdf", _, -1)', "1-1-18-4"),
    ('instr("abcdecfrxcwsd", "c", 0)', "1-1-18-4"),
    ('instr("abcdecfrxcwsd", "c", -5, 4)', "1-1-18-4"),
    ('instr("abcdecfrxcwsd", "c", 0, 0)', "1-1-18-4"),
    ('instr("abcdecfrxcwsd", "c", 6, 0)', "1-1-18-4"),
    ('instr("abcdecfrxcwsd", "c", 5, -5)', "1-1-18-4"),
    ('instr("abcdecfrxcwsd", "c", _, -3)', "1-1-18-4"),
]

# TODO: change this for runtime errors
numeric_exception_param = [
    ("log(5.0, -8)", "Out of Range Error: cannot take logarithm of a negative number"),
    ("log(0.0, 6)", "Out of Range Error: cannot take logarithm of zero"),
    ("log(-2, 6)", "Out of Range Error: cannot take logarithm of a negative number"),
]

ds_param = [
    ("3-51", 'DS_1[calc Me_2:=instr(Me_1, "", null, 4)]'),
    ("4-3", "DS_1 + null"),
    ("4-3", "null + DS_1"),
    ("4-3", "DS_1 - null"),
    ("4-3", "null - DS_1"),
    ("4-3", "DS_1 * null"),
    ("4-3", "null * DS_1"),
    ("4-5", "DS_1 / null"),
    ("4-5", "null / DS_1"),
    ("4-6", "DS_1[calc Me_4:= Me_1 + null]"),
    ("4-6", "DS_1[calc Me_4:= null + Me_1]"),
    ("4-6", "DS_1[calc Me_4:= Me_1 - null]"),
    ("4-6", "DS_1[calc Me_4:= null - Me_1]"),
    ("4-6", "DS_1[calc Me_4:= Me_1 * null]"),
    ("4-6", "DS_1[calc Me_4:= null * Me_1]"),
    ("7-27", "DS_1[calc Me_2:=current_date()]"),
    ("13-9", "DS_1[aggr attribute Me_2 := sum(Me_1) group by Id_1]"),
    ("17-1", "cast(DS_1, string)"),
    ("17-2", "DS_1[calc Me_2 := cast(Me_1, string)]"),
    ("17-3", "DS_1[calc Me_1 := cast(Me_1, string)]"),
]

division_zero_exception_param = [("18-1", "DS_1[calc Me_3 := Me_1 / 0]", "2-1-15-6")]


params_scalar_operations = [
    ("Sc_r <- sc_1 + sc_2 + 3 + sc_3;", {"Sc_r": Scalar(name="Sc_r", data_type=Integer, value=21)}),
    (
        'Sc_r <- replace("Hello world", "Hello", "Hi");',
        {"Sc_r": Scalar(name="Sc_r", data_type=String, value="Hi world")},
    ),
    (
        'Sc_r <- instr("abcde", "c");',
        {"Sc_r": Scalar(name="Sc_r", data_type=Integer, value=3)},
    ),
    (
        "Sc_r <- true and false;",
        {"Sc_r": Scalar(name="Sc_r", data_type=Boolean, value=False)},
    ),
    ("Sc_r <- +null;", {"Sc_r": Scalar(name="Sc_r", data_type=Number, value=None)}),
]


@pytest.mark.parametrize("text, reference", string_params)
def test_string_operators(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == String


@pytest.mark.parametrize("text, reference", instr_op_params)
def test_instr_op_test(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference
    assert result["DS_r"].data_type == Integer


@pytest.mark.parametrize("text, exception_message", string_exception_param)
def test_exception_string_op(text, exception_message):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(SemanticError, match=f".*{exception_message}"):
        interpreter.visit(ast)


@pytest.mark.parametrize("text, reference", numeric_params)
def test_numeric_operators(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    if reference is None:
        assert result["DS_r"].value is None
    else:
        assert result["DS_r"].value == reference
        assert result["DS_r"].data_type == Number or result["DS_r"].data_type == Integer


@pytest.mark.parametrize("text, expected", numeric_exception_param)
def test_exception_numeric_op(text, expected):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    if isinstance(expected, str) and expected.count("-") >= 2:
        with pytest.raises(RunTimeError, match=expected):
            interpreter.visit(ast)
    else:
        with pytest.raises(Exception, match=expected):
            interpreter.visit(ast)


@pytest.mark.parametrize("code, text", ds_param)
def test_datasets_params(code, text):
    warnings.filterwarnings("ignore", category=FutureWarning)
    datasets = AdditionalScalarsTests.LoadInputs(code, 1)
    reference = AdditionalScalarsTests.LoadOutputs(code, ["DS_r"])
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    assert result == reference


@pytest.mark.parametrize("text, reference", boolean_params)
def test_bool_op_test(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference


@pytest.mark.parametrize("text, reference", comparison_params)
def test_comp_op_test(text, reference):
    warnings.filterwarnings("ignore", category=FutureWarning)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result["DS_r"].value == reference


@pytest.mark.parametrize("code, text, error_code", division_zero_exception_param)
def test_division_by_zero_exception(code, text, error_code):
    warnings.filterwarnings("ignore", category=FutureWarning)
    datasets = AdditionalScalarsTests.LoadInputs(code, 1)
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer(datasets)
    with pytest.raises(RunTimeError, match=error_code):
        interpreter.visit(ast)


@pytest.mark.parametrize("script, reference", params_scalar_operations)
def test_run_scalars_operations(script, reference, tmp_path):
    scalar_values = {
        "sc_1": 10,
        "sc_2": 5,
        "sc_3": 3,
        "sc_4": "abcdef",
        "sc_5": "apple",
        "sc_6": True,
        "sc_7": False,
    }

    data_structures = {
        "datasets": [
            {
                "name": "DS_3",
                "DataStructure": [
                    {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
                ],
            }
        ],
        "scalars": [
            {"name": "sc_1", "type": "Integer"},
            {"name": "sc_2", "type": "Integer"},
            {"name": "sc_3", "type": "Integer"},
            {"name": "sc_4", "type": "String"},
            {"name": "sc_5", "type": "String"},
            {"name": "sc_6", "type": "Boolean"},
            {"name": "sc_7", "type": "Boolean"},
        ],
    }

    datapoints = {
        "DS_3": pd.DataFrame(
            {
                "Id_1": [1, 2, 3],
                "Me_1": [10.0, 20.5, 30.1],
            }
        )
    }

    run_result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        scalar_values=scalar_values,
        output_folder=tmp_path,
        return_only_persistent=True,
    )
    for k, expected_scalar in reference.items():
        assert k in run_result
        result_scalar = run_result[k]
        assert result_scalar.value == expected_scalar.value
        assert result_scalar.data_type == expected_scalar.data_type

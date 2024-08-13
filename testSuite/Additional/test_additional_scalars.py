import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest
from pandas import read_csv

from API import create_ast
from DataTypes import String, Integer, Number, SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset

base_path = Path(__file__).parent
filepath_json = base_path / "data" / "DataStructure" / "input"
filepath_csv = base_path / "data" / "DataSet" / "input"
filepath_out_json = base_path / "data" / "DataStructure" / "output"
filepath_out_csv = base_path / "data" / "DataSet" / "output"


def LoadDataset(ds_path, dp_path):
    with open(ds_path, 'r') as file:
        structures = json.load(file)

    for dataset_json in structures['datasets']:
        dataset_name = dataset_json['name']
        components = {
            component['name']: Component(name=component['name'],
                                         data_type=SCALAR_TYPES[component['type']],
                                         role=Role(component['role']),
                                         nullable=component['nullable'])
            for component in dataset_json['DataStructure']}
        data = read_csv(dp_path, sep=',')

        return Dataset(name=dataset_name, components=components, data=data)


def LoadInputs(code: str, number_inputs: int) -> Dict[str, Dataset]:
    '''

    '''
    datasets = {}
    for i in range(number_inputs):
        json_file_name = str(filepath_json / f"{code}-DS_{str(i + 1)}.json")
        csv_file_name = str(filepath_csv / f"{code}-DS_{str(i + 1)}.csv")
        dataset = LoadDataset(json_file_name, csv_file_name)
        datasets[dataset.name] = dataset

    return datasets


def LoadOutputs(code: str, references_names: List[str]) -> Dict[str, Dataset]:
    """

    """
    datasets = {}
    for name in references_names:
        json_file_name = str(filepath_out_json / f"{code}-{name}.json")
        csv_file_name = str(filepath_out_csv / f"{code}-{name}.csv")
        dataset = LoadDataset(json_file_name, csv_file_name)
        datasets[dataset.name] = dataset

    return datasets


string_params = [
    ("substr(null, null, null)", ""),
    ("substr(null)", ""),
    ('substr("abc", null, null)', "abc"),
    ('substr(null, 1, 2)', ""),
    ("substr(null, _, 2)", ""),
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
    ('replace(null, null, null)', ""),
    ('replace(null, null)', ""),
    ('replace("abc", null, null)', ""),
    ('replace("abc", null)', ""),
    ('replace(null, "a", "b")', ""),
    ('replace(null, null, "b")', ""),
    ('replace(null, "a", null)', ""),
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
    ('instr(null, null, null, null)', 0),
    ('instr(null, "a")', 0),
    ('instr("abc", "a", null)', 1),
    ('instr("abc", "a", 1, null)', 1),
    ('instr("abc", "a", null, 3)', 0),
    ('instr("abc", "a", null, null)', 1)
]

numeric_params = [
    ('+null', None),
    ('-null', None),
    ('ceil(null)', None),
    ('floor(null)', None),
    ('abs(null)', None),
    ('exp(null)', None),
    ('ln(null)', None),
    ('sqrt(null)', None),
    ('2 + null', None),
    ('null + 2.0', None),
    ('2 - null', None),
    ('null - 2.0', None),
    ('2 * null', None),
    ('null * 2.0', None),
    ('2 / null', None),
    ('null / 2', None),
    ('2 + 3.3', 5.3),
    ('3.3 + 2', 5.3),
    ('2 - 3.3', -1.3),
    ('3.3 - 2.0', 1.3),
    ('2 * 3.3', 6.6),
    ('3.3 * 2', 6.6),
    ('2 / 1.0', 2),
    ('1.0 / 2', 0.5),
    ('round(null, 0)', None),
    ('round(null)', None),
    ('round(null, 3)', None),
    ('round(null, _)', None),
    ('round(null, null)', None),
    ('round(5.0, null)', 5),
    ('round(3.14159, 2)', 3.14),
    ('round(3.14159, _)', 3),
    ('round(3.14159, 4)', 3.1416),
    ('round(12345.6, 0)', 12346.0),
    ('round(12345.6)', 12346),
    ('round(12345.6, _)', 12346),
    ('round(12345.6, -1)', 12350.0),
    ('trunc(null, 0)', None),
    ('trunc(null)', None),
    ('trunc(null, 3)', None),
    ('trunc(null, _)', None),
    ('trunc(null, null)', None),
    ('trunc(4.0, null)', 4),
    ('trunc(3.14159, 2)', 3.14),
    ('trunc(3.14159, _)', 3),
    ('trunc(3.14159, 4)', 3.1415),
    ('trunc(12345.6, 0)', 12345),
    ('trunc(12345.6)', 12345),
    ('trunc(12345.6, _)', 12345),
    ('trunc(12345.6, -1)', 12340.0),
    ('power(5, 2)', 25),
    ('power(5, 1)', 5),
    ('power(5, 0)', 1),
    ('power(5, -1)', 0.2),
    ('power(-5, 3)', -125),
    ('power(null, null)', None),
    ('power(null, 1)', None),
    ('power(1, null)', None),
    ('log(8, 2)', 3.0),
    ('log(8.0, 2)', 3.0),
    ('log(1024, 2)', 10.0),
    ('log(1024, 10)', 3.0102999566398116),
    ('log(2.0, 2)', 1.0),
    ('log(null, null)', None),
    ('log(null, 1)', None),
    ('log(1, null)', None),
    ('log(0.5, 6)', -0.3868528072345416),

]

boolean_params = [
    ('false and false', False),
    ('false and true', False),
    ('false and null', False),
    ('true and false', False),
    ('true and true', True),
    ('true and null', None),
    ('null and null', None),
    ('false or false', False),
    ('false or true', True),
    ('false or null', None),
    ('true or false', True),
    ('true or true', True),
    ('true or null', True),
    ('null or null', None),
    ('false xor false', False),
    ('false xor true', True),
    ('false xor null', None),
    ('true xor false', True),
    ('true xor true', False),
    ('true xor null', None),
    ('null xor null', None),
    ('not false', True),
    ('not true', False),
    ('not null', None)
]

comparison_params = [
    ('3 = null', None),
    ('3 <> null', None),
    ('3 < null', None),
    ('3 > null', None),
    ('3 <= null', None),
    ('3 >= null', None),
    ('3 in { null }', None),
    ('not (3 in { null })', None),
    ('not (null in { 1,2,3 })', None),
    ('between(null, 4, 5)', None),
    ('between(5, null, 5)', None),
    ('between(4, 4, null)', None),
    ('between(null, null, null)', None),
    ('between("a", "a", "z")', True),
    ('between("z", "a", "c")', False),
    ('between(6, 1, 9)', True),
    ('between(12, 1, 9)', False),
]

string_exception_param = [
    ('substr("asdf", -3)', 'param length should be >= 0'),
    ('substr("asdf", 0)', 'zzz'),
    ('substr("asdf", -2, 3)', 'param length should be >= 0'),
    ('substr("asdf", 0, 5)', 'zzz'),
    ('substr("asdf", 1, -9)', 'param length should be >= 0'),
    ('substr("asdf", _, -1)', 'zzz'),
    ('instr("abcdecfrxcwsd", "c", 0)', 'param start should be >= 1'),
    ('instr("abcdecfrxcwsd", "c", -5, 4)', 'param start should be >= 1'),
    ('instr("abcdecfrxcwsd", "c", 0, 0)', 'param start should be >= 1'),
    ('instr("abcdecfrxcwsd", "c", 6, 0)', 'zzz'),
    ('instr("abcdecfrxcwsd", "c", 5, -5)', 'zzz'),
    ('instr("abcdecfrxcwsd", "c", _, -3)', 'zzz'),
]

numeric_exception_param = [
    ('log(5.0, -8)', 'math domain error'),
    ('log(0.0, 6)', 'math domain error'),
]

ds_param = [
    ('3-51', 'DS_1[calc Me_2:=instr(Me_1, "", null, 4)]'),
    ('4-3', 'DS_1 + null'),
    ('4-3', 'null + DS_1'),
    ('4-3', 'DS_1 - null'),
    ('4-3', 'null - DS_1'),
    ('4-3', 'DS_1 * null'),
    ('4-3', 'null * DS_1'),
    ('4-5', 'DS_1 / null'),
    ('4-5', 'null / DS_1'),
    ('4-6', 'DS_1[calc Me_4:= Me_1 + null]'),
    ('4-6', 'DS_1[calc Me_4:= null + Me_1]'),
    ('4-6', 'DS_1[calc Me_4:= Me_1 - null]'),
    ('4-6', 'DS_1[calc Me_4:= null - Me_1]'),
    ('4-6', 'DS_1[calc Me_4:= Me_1 * null]'),
    ('4-6', 'DS_1[calc Me_4:= null * Me_1]')
]


@pytest.mark.parametrize("text, reference", string_params)
def test_string_operators(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result['DS_r'].value == reference
    assert result['DS_r'].data_type == String


@pytest.mark.parametrize("text, reference", instr_op_params)
def test_instr_op_test(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result['DS_r'].value == reference
    assert result['DS_r'].data_type == Integer


@pytest.mark.parametrize('text, exception_message', string_exception_param)
def test_exception_string_op(text, exception_message):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(Exception, match=exception_message):
        interpreter.visit(ast)


@pytest.mark.parametrize('text, reference', numeric_params)
def test_numeric_operators(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    if reference is None:
        assert result['DS_r'].value is None
    else:
        assert result['DS_r'].value == reference
        assert result['DS_r'].data_type == Number or result['DS_r'].data_type == Integer


@pytest.mark.parametrize('text, exception_message', numeric_exception_param)
def test_exception_numeric_op(text, exception_message):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    with pytest.raises(Exception, match=exception_message):
        interpreter.visit(ast)


@pytest.mark.parametrize('code, text', ds_param)
def test_datasets_params(code, text):
    datasets = LoadInputs(code, 1)
    reference = LoadOutputs(code, ["DS_r"])
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer(datasets)
    result = interpreter.visit(ast)
    assert result == reference


@pytest.mark.parametrize('text, reference', boolean_params)
def test_bool_op_test(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result['DS_r'].value == reference


@pytest.mark.parametrize('text, reference', comparison_params)
def test_comp_op_test(text, reference):
    expression = f"DS_r := {text};"
    ast = create_ast(expression)
    interpreter = InterpreterAnalyzer({})
    result = interpreter.visit(ast)
    assert result['DS_r'].value == reference

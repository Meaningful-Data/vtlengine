"""
Add to vs code User settings:

    {"python.unitTest.unittestEnabled": true,
    "python.unitTest.pyTestEnabled": false,
    "python.unitTest.nosetestsEnabled": false,
    }

"""
import json
from pathlib import Path
from unittest import TestCase

import pandas as pd

# Others
import unittest
import os
from typing import List, Dict

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset

"""
Improvements:

"""
"""
TODOS:
    [1]: Add test with Datasets and Componets with null values for Numeric, Comparison and Boolean Operators.
"""


# Path Selection.---------------------------------------------------------------

class ThreeValueHelper(TestCase):
    """

    """

    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_vtl = base_path / "data" / "vtl"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    # File extensions.--------------------------------------------------------------
    JSON = '.json'
    CSV = '.csv'
    VTL = '.vtl'

    @classmethod
    def LoadDataset(cls, ds_path, dp_path):
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
            data = pd.read_csv(dp_path, sep=',')

            return Dataset(name=dataset_name, components=components, data=data)

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> Dict[str, Dataset]:
        '''

        '''
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(cls.filepath_json / f"{code}-DS_{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-DS_{str(i + 1)}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadOutputs(cls, code: str, references_names: List[str]) -> Dict[str, Dataset]:
        """

        """
        datasets = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """

        """
        vtl_file_name = str(cls.filepath_vtl / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, text: str, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''
        if text is None:
            text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets


class ThreeValueTests(ThreeValueHelper):
    """
    Group 1
    """

    classTest = '3VL.AndTest'

    maxDiff = None

    def test_1(self):
        '''
        And logic Test
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 and Me_2];"""
        code = '1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        Component-Scalar Test true
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 and true];"""
        code = '2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        Component-Scalar Test false
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 and false];"""
        code = '3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        Dataset-scalar Test true
        '''
        text = """DS_r := DS_1 and true;"""
        code = '4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''
        Dataset-scalar Test true
        '''
        text = """DS_r := DS_1 and false;"""
        code = '5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''
        Or logic test
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 or Me_2];"""
        code = '6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        '''
        Xor logic test
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 xor Me_2];"""
        code = '7'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

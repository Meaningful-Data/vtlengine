import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestExtraTypeChecking(TestCase):
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
            json_file_name = str(cls.filepath_json / f"{code}-{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-{str(i + 1)}{cls.CSV}")
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
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''

        text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True

class ExtraTypeChecking(TestExtraTypeChecking):
    """
    Group 4
    """

    classTest = 'Extra.ExtraTypeChecking'

    def test_1(self):
        '''
        ADD OPERATOR
        Status: Me_1, Me_2 and Me_3 should be numbers, i changed this in the datastructure
        Alternative Expression: DS_r := DS_2 + DS_1[calc Me_3 := DS_1#Me_2 + 1.0 ] + 1.0 ; (good result)
        Expression: DS_r := DS_2 + (DS_1[calc Me_3 := DS_1#Me_2 + 1.0 ] + 1.0 ); (bad result)
        Description: operations between nulls, numbers and integers.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-5-3-1'
        # 4 For group numeric
        # 5 For group extra
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]


        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)  # TODO : Review this
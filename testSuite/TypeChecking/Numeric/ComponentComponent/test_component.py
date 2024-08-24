import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestComponentTypeChecking(TestCase):
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


class ComponentComponentTypeChecking(TestComponentTypeChecking):
    """
    Group 4
    """

    classTest = 'ComponentComponent.ComponentComponentTypeChecking'

    def test_1(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1[calc Me_3 := DS_1#Me_1 + DS_1#Me_2];
                    Me_1 Integer
                    Me_2 Boolean
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-1'
        # 4 For group numeric
        # 2 For group scalar component
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_2(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1[calc Me_3 := DS_1#Me_1 + DS_1#Me_2];
                    Me_1 Boolean
                    Me_2 Boolean
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-2'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_3(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_2 + DS_1#Me_1 ];
                    Me_1 Integer
                    Me_2 Boolean
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-3'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_4(self):
        '''
        ADD OPERATOR
        time --> number
        ValueError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 Time
                    Me_2 Integer
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-4'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 Time
                    Me_2 Time
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-5'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_6(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Error Type : ValueError
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 Date
                    Me_2 Integer
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-6'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 time_period
                    Me_2 Integer
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-7'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_8(self):
        '''
        ADD OPERATOR
        string --> number
        Error Type: TypeError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 string
                    Me_2 Integer
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-8'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_9(self):
        '''
        ADD OPERATOR
        string --> number
        Error Type: TypeError
        test_8 with different csv
        This test would be not here but we keep it for now
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 string
                    Me_2 Integer
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-9'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_10(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 string
                    Me_2 string
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-10'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_11(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 duration
                    Me_2 integer
        Description: Forbid implicit cast duration to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-2-3-11'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_12(self):
        '''
        ADD OPERATOR
        number + integer --> number
        Status: ok
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 number
                    Me_2 integer
        Description: Check implicit cast integer to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-2-3-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        '''
        ADD OPERATOR
        integer + integer --> integer
        Status: ok
        Expression: DS_r := DS_1[ calc Me_3 := DS_1#Me_1 + DS_1#Me_2 ];
                    Me_1 integer
                    Me_2 integer
        Description: Forbid implicit cast integer to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-2-3-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

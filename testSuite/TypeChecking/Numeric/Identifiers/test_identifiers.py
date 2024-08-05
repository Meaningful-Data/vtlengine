import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestIdentifiersTypeChecking(TestCase):
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


class IdentifiersTypeCheckingAdd(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = 'Identifiers.IdentifiersTypeCheckingAdd'

    def test_1(self):
        '''
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 + DS_2;
        Number: Number + Integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-3-1'
        # 4 For group numeric
        # 6 For group identifiers
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_2 + DS_1 ;
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-3-2'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
        string-integer
        Description: operations between identifiers, string and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-3-3'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
        string-number
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-3-4'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_2 + DS_1 ;
        number-string
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-3-5'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingSubstraction(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = 'Identifiers.IdentifiersTypeCheckingSubstraction'

    # SUBSTRACTION OPERATOR

    def test_1(self):
        '''
        SUBSTRACTION OPERATOR
        Status: OK
        Expression: DS_r := DS_1 - DS_2;
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-4-1'
        # 4 For group numeric
        # 6 For group identifiers
        # 4 For substraction operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        SUBSTRACTION OPERATOR
        Status: OK
        Expression: DS_r := DS_1 - DS_2;
        number - string
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-4-2'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingMultiplication(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = 'Identifiers.IdentifiersTypeCheckingMultiplication'

    # MULTIPLICATION OPERATOR

    def test_1(self):
        '''
        MULTIPLICATION OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 * DS_2;
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-5-1'
        # 4 For group numeric
        # 6 For group identifiers
        # 5 For multiplication operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        MULTIPLICATION OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 * DS_2;
        number - string
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-5-2'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingDivision(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = 'Identifiers.IdentifiersTypeCheckingDivision'

    # DIVISION OPERATOR

    def test_1(self):
        '''
        DIVISION OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 / DS_2;
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-6-1'
        # 4 For group numeric
        # 6 For group identifiers
        # 6 For multiplication operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        DIVISION OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 / DS_2;
        number - string
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-6-2'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingModule(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = 'Identifiers.IdentifiersTypeCheckingModule'

    # MOD OPERATOR

    def test_1(self):
        '''
        MOD OPERATOR
        Status: BUG
        Expression: DS_r := mod ( DS_1, DS_2 );
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-7-1'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        MOD OPERATOR
        Status: BUG
        Expression: DS_r := mod ( DS_1, DS_2 );
        number - string
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        '''
        code = '4-6-7-2'
        number_inputs = 2
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

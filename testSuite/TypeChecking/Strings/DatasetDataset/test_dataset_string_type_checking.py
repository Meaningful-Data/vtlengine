import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestStringTypeChecking(TestCase):
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


class DatasetDatasetStringTypeChecking(TestStringTypeChecking):
    """
    Group 3
    """

    classTest = 'DatasetDataset.DatasetDatasetStringTypeChecking'

    def test_1(self):
        """
        CONCATENATION OPERATOR
        (string || string) || string --> string
        Status: OK
        Expression: DS_r := (DS_1 || DS_2) || DS_3 ;
                            DS_1 Measure String
                            DS_2 Measure String
                            DS_3 Measure String

        Description: Concatenation of two or more strings.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of two or more strings.
        """
        code = '3-4-1-1'

        # 3 For group string
        # 4 For group dataset dataset
        # 1 For concatenation operator in string
        # 1 Number of test

        number_inputs = 3

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_2(self):
        """
        CONCATENATION OPERATOR
        Integer --> string
        Status: BUG
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure Integer
                            DS_2 Measure String


        Description: Concatenation of an Integer with a String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of Integer with String.
        """
        code = '3-4-1-2'
        number_inputs = 2

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_3(self):
        """
        CONCATENATION OPERATOR
        number --> string
        Status: BUG
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure Number
                            DS_2 Measure String


        Description: Concatenation of a Number with a String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a Number with String.
        """
        code = 'Test3'

        code = '3-4-1-3'
        number_inputs = 2

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_4(self):
        """
        CONCATENATION OPERATOR
        Boolean --> string
        Status: BUG
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure Boolean
                            DS_2 Measure String


        Description: Concatenation of a Boolean with a String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a Boolean with String.
        """
        code = '3-4-1-4'
        number_inputs = 2

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        CONCATENATION OPERATOR
        String --> Integer
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure Integer


        Description: Concatenation of a String with an Integer.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with Integer.
        """
        code = '3-4-1-5'
        number_inputs = 2

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        CONCATENATION OPERATOR
        String --> Time_Period
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure time_period


        Description: Concatenation of a String with Time_Period.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with time_period.
        """
        code = '3-4-1-6'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        CONCATENATION OPERATOR
        Time --> String
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure time
                            DS_2 Measure String


        Description: Concatenation of Time with String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of time with String.
        """
        code = '3-4-1-7'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_8(self):
        """
        CONCATENATION OPERATOR
        Date --> String
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure date
                            DS_2 Measure String


        Description: Concatenation of date with String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of Date with String.
        """
        code = '3-4-1-8'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_9(self):
        """
        CONCATENATION OPERATOR
        Time_Period --> String
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure time_period
                            DS_2 Measure String


        Description: Concatenation of time_period with String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of time_period with String.
        """
        code = '3-4-1-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_10(self):
        """
        CONCATENATION OPERATOR
        Duration --> String
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure duration
                            DS_2 Measure String


        Description: Concatenation of duration with String.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of duration with String.
        """
        code = '3-4-1-10'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_11(self):
        """
        CONCATENATION OPERATOR
        String --> Number
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure Number


        Description: Concatenation of String with Number.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with Number.
        """
        code = '3-4-1-11'
        number_inputs = 2

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        CONCATENATION OPERATOR
        String --> Duration
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure Duration


        Description: Concatenation of String with Duration.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with Duration.
        """
        code = '3-4-1-12'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_13(self):
        """
        CONCATENATION OPERATOR
        String --> Boolean
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure Boolean


        Description: Concatenation of String with Boolean.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with Boolean.
        """
        code = '3-4-1-13'
        number_inputs = 2

        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        CONCATENATION OPERATOR
        String --> Time
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure Time


        Description: Concatenation of String with Time.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with Time.
        """
        code = '3-4-1-14'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_15(self):
        """
        CONCATENATION OPERATOR
        String --> Date
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure Date


        Description: Concatenation of String with Date.

        Git Branch: #23 String operators types checking tests.
        Goal: Do the concatenation of a String with Date.
        """
        code = '3-4-1-15'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_16(self):
        """
        WHITESPACE REMOVAL OPERATOR
        String --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        WHITESPACE REMOVAL OPERATOR
        String --> String
        Status: OK
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        WHITESPACE REMOVAL OPERATOR
        String --> String
        Status: OK
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left) from a string.
        """
        code = '3-4-2-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_19(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Integer --> String
        Status: BUG
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Integer


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_20(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Integer --> String
        Status: BUG
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_21(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Integer --> String
        Status: BUG
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Number


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_22(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Number --> String
        Status: BUG
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Number


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_23(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Number --> String
        Status: BUG
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure Number


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-8'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_24(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Number --> String
        Status: BUG
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Number


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-9'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_25(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Boolean --> String
        Status: BUG
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Boolean


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-10'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_26(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Boolean --> String
        Status: BUG
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure Boolean


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-11'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_27(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Boolean --> String
        Status: BUG
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Boolean


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-12'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Time --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Time


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-13'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_29(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Time --> String
        Status: OK
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure Time


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-14'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_30(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Time --> String
        Status: OK
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Time


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-15'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_31(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Date --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Date


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-16'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_32(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Date --> String
        Status: OK
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure Date


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-17'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_33(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Date --> String
        Status: OK
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Date


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-18'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_34(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Time_Period --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Time_Period


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-19'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_35(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Time_Period --> String
        Status: OK
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure Time_Period


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-20'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_36(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Time_Period --> String
        Status: OK
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Time_Period


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-21'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_37(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Duration --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure Duration


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal from a string.
        """
        code = '3-4-2-22'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_38(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Duration --> String
        Status: OK
        Expression: DS_r := rtrim(DS_1)
                            DS_1 Measure Duration


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(right side) from a string.
        """
        code = '3-4-2-23'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_39(self):
        """
        WHITESPACE REMOVAL OPERATOR
        Duration --> String
        Status: OK
        Expression: DS_r := ltrim(DS_1)
                            DS_1 Measure Duration


        Description: Whitespace removal.

        Git Branch: #87 Trim operators type checking tests.
        Goal: Whitespace removal(left side) from a string.
        """
        code = '3-4-2-24'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_40(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_42(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Integer --> String
        Status: BUG
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Integer


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_43(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Integer --> String
        Status: BUG
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Integer


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_44(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Number --> String
        Status: BUG
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Number


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_45(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Number --> String
        Status: BUG
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Number


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


    def test_46(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Boolean --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Boolean


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Boolean --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Boolean


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-8'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Time --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Time


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_49(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Time --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Time


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-10'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_50(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Date --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Date


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-11'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_51(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Date --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Date


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-12'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_52(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Time_Period --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Time_Period


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-13'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_53(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Time_Period --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Time_Period


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-14'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_54(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Duration --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure Duration


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in upper case.
        """
        code = '3-4-3-15'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_55(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        Duration --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure Duration


        Description: upper/lower operators.

        Git Branch: #98 lower-upper operators type checking tests.
        Goal: Converts the character case of a string in lower case.
        """
        code = '3-4-3-16'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_56(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1, start, length)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_57(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1,_,length)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.
        *- If start is omitted, the substring starts from the 1st position.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_58(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1,start)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.
        *- If length is omitted or overcomes the length of the input string,
        the substring ends at the end of the input string.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_59(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.
        *- If start is greater than the length of the input string, an empty
        string is extracted.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_60(self):
        """
        SUB-STRING EXTRACTION
        Integer --> String
        Status: BUG
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Integer


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_61(self):
        """
        SUB-STRING EXTRACTION
        Number --> String
        Status: BUG
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Number


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_62(self):
        """
        SUB-STRING EXTRACTION
        Boolean --> String
        Status: BUG
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Boolean


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_63(self):
        """
        SUB-STRING EXTRACTION
        Time --> String
        Status: OK
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Time


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-8'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_64(self):
        """
        SUB-STRING EXTRACTION
        Date --> String
        Status: OK
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Date


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_65(self):
        """
        SUB-STRING EXTRACTION
        Time_Period --> String
        Status: OK
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Time_Period


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-10'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_66(self):
        """
        SUB-STRING EXTRACTION
        Duration --> String
        Status: OK
        Expression: DS_r := substr (DS_1,start,length)
                            DS_1 Measure Duration


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #112 substr-operators-type-checking-tests.
        Goal: The operator extracts a substring from op, which must be string type
        """
        code = '3-4-4-11'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_67(self):
        """
        STRING PATTERN REPLACEMENT
        String --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure String


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
        """
        code = '3-4-5-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_68(self):
        """
        STRING PATTERN REPLACEMENT
        String --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1)
                            DS_1 Measure String


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).
        *- If pattern2 is omitted then all occurrences of pattern1 are removed.

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_69(self):
        """
        STRING PATTERN REPLACEMENT
        Integer --> String
        Status: BUG
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Integer


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_70(self):
        """
        STRING PATTERN REPLACEMENT
        Number --> String
        Status: BUG
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Number


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_71(self):
        """
        STRING PATTERN REPLACEMENT
        Boolean --> String
        Status: BUG
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Boolean


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_72(self):
        """
        STRING PATTERN REPLACEMENT
        Time --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Time


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-6'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_73(self):
        """
        STRING PATTERN REPLACEMENT
        Date --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Date


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-7'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_74(self):
        """
        STRING PATTERN REPLACEMENT
        Time_Period --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Time_Period


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-8'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_75(self):
        """
        STRING PATTERN REPLACEMENT
        Duration --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure Duration


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #114 replace operator type checking tests.
        Goal: Replaces all the occurrences of a specified string-pattern
         """
        code = '3-4-5-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_76(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1,pattern,start,occurrence)
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_77(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1,pattern,_,occurrence)
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        *- If start is omitted, the search starts from the 1st position.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_78(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern,start)
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        *- If nth occurrence is omitted, the value is 1.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_79(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence )
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        *- If the nth occurrence of the string-pattern after the start character
        is not found in the input string, the returned value is 0.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_80(self):
        """
        STRING PATTERN LOCATION
        Integer --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence )
                            DS_1 Measure Integer


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_81(self):
        """
        STRING PATTERN LOCATION
        Number --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence )
                            DS_1 Measure Number


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_82(self):
        """
        STRING PATTERN LOCATION
        Boolean --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence)
                            DS_1 Measure Boolean


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_83(self):
        """
        STRING PATTERN LOCATION
        Time --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence)
                            DS_1 Measure Time


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-8'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_84(self):
        """
        STRING PATTERN LOCATION
        Date --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence)
                            DS_1 Measure Date


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_85(self):
        """
        STRING PATTERN LOCATION
        Time_Period --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence)
                            DS_1 Measure Time_Period


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-10'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_86(self):
        """
        STRING PATTERN LOCATION
        Duration --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1, pattern, start, occurrence)
                            DS_1 Measure Duration


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #118 instr operator type checking tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern).
         """
        code = '3-4-6-11'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_87(self):
        """
        STRING LENGTH
        String --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure String


        Description: Returns the length of a string.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a string.
         """
        code = '3-4-7-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_88(self):
        """
        STRING LENGTH
        String --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure String


        Description: Returns the length of a string.
        *- For the empty string “” the value 0 is returned.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a string
         """
        code = '3-4-7-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_89(self):
        """
        STRING LENGTH
        Integer --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Integer


        Description: Returns the length of a Integer.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a Integer.
         """
        code = '3-4-7-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_90(self):
        """
        STRING LENGTH
        Number --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Number


        Description: Returns the length of a Number.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a Number.
         """
        code = '3-4-7-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_91(self):
        """
        STRING LENGTH
        Boolean --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Boolean


        Description: Returns the length of a string.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a string.
         """
        code = '3-4-7-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_92(self):
        """
        STRING LENGTH
        Time --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Time


        Description: Returns the length of a Time.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a Time.
         """
        code = '3-4-7-6'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_93(self):
        """
        STRING LENGTH
        Date --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Date


        Description: Returns the length of a Date.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a Date.
         """
        code = '3-4-7-7'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_94(self):
        """
        STRING LENGTH
        Time_Period --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Time_Period


        Description: Returns the length of a Time_Period.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a Time_Period.
         """
        code = '3-4-7-8'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_95(self):
        """
        STRING LENGTH
        Duration --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure Duration


        Description: Returns the length of a Duration.

        Git Branch: #121 length operator type checking tests.
        Goal: Returns the length of a Duration.
         """
        code = '3-4-7-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

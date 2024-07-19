import json
from pathlib import Path
from typing import List, Dict
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Dataset, Component, Role

check_dtype = False  # remove when refactor complete.

classTest = None


class AdditionalHelper(TestCase):
    """

    """

    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
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
    def BaseTest(cls, text: str, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''
        if text is None:
            # TODO: Read vtl file from code and create ast
            text = ''
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets


class StringOperatorsTest(AdditionalHelper):
    """
    Group 3
    """

    classTest = 'Additional.StringOperatorsTest'

    maxDiff = None

    def test_1(self):
        '''
        Basic behaviour for two datasets.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-1'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        Behaviour for two datasets with nulls.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        Behaviour for two datasets with diferent number of rows.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        Behaviour for components with nulls.
        '''
        text = """DS_r := DS_1[calc Me_3:= Me_1 || Me_2];"""

        code = '3-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''
        Behaviour for two datasets when the left one has more identifiers.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''
        Behaviour for two datasets when the right one has more identifiers.
        '''
        text = """DS_r := DS_1 || DS_2;"""

        code = '3-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1, 2, 3);"""

        code = '3-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1, 2);"""

        code = '3-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1, _, 3);"""

        code = '3-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        '''
        Behaviour for dataset.
        '''
        text = """DS_r := substr(DS_1);"""

        code = '3-14'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1)];"""

        code = '3-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, _, 3)];"""

        code = '3-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, 3)];"""

        code = '3-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        '''
        Behaviour for components and null.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, null)];"""

        code = '3-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        '''
        Behaviour for components and null.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, 1, null)];"""

        code = '3-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        '''
        Behaviour for components and null.
        '''
        text = """DS_r := DS_1[calc Me_2:= substr(Me_1, null, null)];"""

        code = '3-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_1:= substr(Me_1, Me_2, Me_3)];"""
        code = '3-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_3:= substr(Me_1, _, Me_2)];"""

        code = '3-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_3:= substr(Me_1, Me_2)];"""

        code = '3-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r := replace(DS_1, "Hello", "Hi");"""

        code = '3-26'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r := replace(DS_1, "Hello");"""

        code = '3-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r := replace(DS_1, null, "abc");"""

        code = '3-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r := replace(DS_1, "abc", null);"""
        code = '3-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, "Hello")];"""
        code = '3-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, "Hello", "Hi")];"""
        code = '3-31'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_3:= replace(Me_1, Me_2)];"""
        code = '3-32'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_33(self):
        '''
        Behaviour for components.
        '''
        text = """DS_r := DS_1[calc Me_4:= replace(Me_1, Me_2, Me_3)];"""
        code = '3-33'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, null)];"""
        code = '3-34'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, null, null)];"""
        code = '3-35'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, "a", null)];"""
        code = '3-36'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_37(self):
        '''
        Behaviour for components with null.
        '''
        text = """DS_r := DS_1[calc Me_2:= replace(Me_1, null, "a")];"""
        code = '3-37'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r:= instr(DS_1, "o", 3);"""
        code = '3-41'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r:= instr(DS_1, "o", _, 2);"""
        code = '3-42'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        '''
        Behaviour for dataset with null values.
        '''
        text = """DS_r:= instr(DS_1, "o", 4, 3);"""
        code = '3-43'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_44(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r:= instr(DS_1, null, 4, 3);"""
        code = '3-44'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_45(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r:= instr(DS_1, "s", null, 3);"""
        code = '3-45'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        '''
        Behaviour for dataset with null.
        '''
        text = """DS_r:= instr(DS_1, "s", 3, null);"""
        code = '3-46'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        '''
        Behaviour for component with null values.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1,"o", 3)];"""
        code = '3-47'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        '''
        Behaviour for component with null values.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1,"o", _, 2)];"""
        code = '3-48'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_49(self):
        '''
        Behaviour for component with null values.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1,"o", 6, 4)];"""
        code = '3-49'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_50(self):
        '''
        Behaviour for component with null.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1, null, 6, 4)];"""
        code = '3-50'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_51(self):
        '''
        Behaviour for component with null.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1, sc_1, null, 4)];"""

        code = '3-51'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_52(self):
        '''
        Behaviour for component with null.
        '''
        text = """DS_r := DS_1[calc Me_2:=instr(Me_1, "o", 4, null)];"""
        code = '3-52'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_53(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_3:=instr(Me_1, Me_2)];"""
        code = '3-53'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_54(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_4:=instr(Me_1, Me_2, Me_3)];"""
        code = '3-54'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_55(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_4:=instr(Me_1, Me_2, _, Me_3)];"""
        code = '3-55'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_56(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1[calc Me_5:=instr(Me_1, Me_2, Me_3, Me_4)];"""
        code = '3-56'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)


class NumericOperatorsTest(AdditionalHelper):
    """
    Group 4
    """

    classTest = 'Additional.NumericOperatorsTest'

    maxDiff = None

    def test_4(self):
        '''
        Null Unary operations ('+') with Datasets.
        '''
        text = """DS_r := +DS_1;"""

        code = '4-4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        '''
        Basic behaviour for dataset.
        '''
        text = """DS_r := round(DS_1);"""

        code = '4-9'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r := round(DS_1, 0);"""

        code = '4-10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        '''
        Basic behaviour for dataset with null.
        '''
        text = """DS_r := round(DS_1, null);"""

        code = '4-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := DS_1 [ calc Me_3:= round(Me_1, Me_2) ];"""

        code = '4-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := trunc(DS_1, 0);"""

        code = '4-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := trunc(DS_1, null);"""

        code = '4-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r := DS_1 [ calc Me_3:= trunc(Me_1, Me_2) ];"""

        code = '4-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        '''
        Basic behaviour for components.
        '''
        text = """DS_r  := DS_1[ calc Me_3 := power(Me_1, Me_2) ];"""

        code = '4-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        '''
        Basic behaviour for dataset and null.
        '''
        text = """DS_r  := power(DS_1, null);"""

        code = '4-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r  := power(DS_1, 2);"""

        code = '4-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        '''
        Basic behaviour for component and null.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := power(Me_1, null) ];"""

        code = '4-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := power(Me_1, 2) ];"""

        code = '4-24'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        '''
        Basic behaviour for components.
        '''
        text = """DS_r  := DS_1[ calc Me_3 := log(Me_1, Me_2) ];"""

        code = '4-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        '''
        Basic behaviour for dataset and null.
        '''
        text = """DS_r  := log(DS_1, null);"""

        code = '4-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        '''
        Basic behaviour for dataset with null values.
        '''
        text = """DS_r  := log(DS_1, 2);"""

        code = '4-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        '''
        Basic behaviour for component and null.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := log(Me_1, null) ];"""

        code = '4-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        '''
        Basic behaviour for components with null values.
        '''
        text = """DS_r  := DS_1[ calc Me_2 := log(Me_1, 2) ];"""

        code = '4-31'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)


class ComparisonOperatorsTest(AdditionalHelper):
    """
    Group 5
    """

    classTest = 'Additional.ComparisonOperatorsTest'

    maxDiff = None

    def test_2(self):
        '''

        '''
        text = """DS_r := exists_in (DS_1, DS_2);"""
        code = '5-2'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        equal to reference manual test but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2, all);"""
        code = '5-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        equal to reference manual test but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2, true);"""
        code = '5-4'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''
        equal to reference manual test but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2, false);"""
        code = '5-5'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''
        equal to test 2 but this at DS_1 contains nulls.
        '''
        text = """DS_r := exists_in (DS_1, DS_2);"""
        code = '5-6'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        '''
        equal to reference manual test but this with reverse order.
        '''
        text = """DS_r := exists_in (DS_2, DS_1, all);"""
        code = '5-7'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        '''
        equal to reference manual test but at this one DS_2 have no Id_4 (different number of Ids).
        '''
        text = """DS_r := exists_in (DS_1, DS_2, all);"""
        code = '5-8'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        '''
        equal to reference manual test but at this one DS_2 have no Id_4 (different number of Ids).
        '''
        text = """DS_r := exists_in (DS_2, DS_1, all);"""
        code = '5-9'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        '''
        Behaviour for dataset with null values and scalars.
        '''
        text = """DS_r:= between(DS_1, 5, 10);"""
        code = '5-10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        '''
        Behaviour for dataset with null scalars.
        '''
        text = """DS_r:= between(DS_1, 5, null);"""
        code = '5-11'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        '''
        Behaviour for dataset with null scalars.
        '''
        text = """DS_r:= between(DS_1, null, 10);"""
        code = '5-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        '''
        Behaviour for dataset with null scalars.
        '''
        text = """DS_r:= between(DS_1, null, null);"""
        code = '5-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)



    def test_16(self):
        '''
        Behaviour for components with null values.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, Me_3) ];"""
        code = '5-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        '''
        Behaviour for components with null values and scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, 100) ];"""
        code = '5-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        '''
        Behaviour for components with null values and null scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, Me_2, null) ];"""
        code = '5-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        '''
        Behaviour for components with null values and null scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, null, Me_2) ];"""
        code = '5-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        '''
        Behaviour for components with null values and scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, 900, Me_2) ];"""
        code = '5-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        '''
        Behaviour for component with null values and scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, 5, 300) ];"""
        code = '5-21'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, null, null) ];"""
        code = '5-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, 4, null) ];"""
        code = '5-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(Me_1, null, 4) ];"""
        code = '5-24'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, Me_1, 4) ];"""
        code = '5-25'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, Me_1, null) ];"""
        code = '5-26'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(4, Me_1, null) ];"""
        code = '5-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, 4, Me_1) ];"""
        code = '5-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, null, Me_1) ];"""
        code = '5-29'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        '''
        Behaviour for component with null values and null scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(4, null, Me_1) ];"""
        code = '5-30'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        '''
        Behaviour for component with null values and scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(6, 1, Me_1) ];"""
        code = '5-31'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        '''
        Behaviour for component with null values and scalars.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(6, Me_1, 10) ];"""
        code = '5-32'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_33(self):
        '''
        Behaviour for components with null values and scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(6, Me_1, Me_2) ];"""
        code = '5-33'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        '''
        Behaviour for components with null values and null scalar.
        '''
        text = """DS_r := DS_1 [ calc Me_4 := between(null, Me_1, Me_2) ];"""
        code = '5-34'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)


class BooleanOperatorsTest(AdditionalHelper):
    """
    Group 6
    """

    classTest = 'Additional.BooleanOperatorsTest'

    maxDiff = None

    pass
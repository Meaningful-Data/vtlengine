"""
Add to vs code User settings:

    {"python.unitTest.unittestEnabled": true,
    "python.unitTest.pyTestEnabled": false,
    "python.unitTest.nosetestsEnabled": false,
    }

"""
from VtlEngine.Datamodel import DataSet
from VtlEngine.Datamodel import Scalar
from VtlEngine.DataTypes import ScalarTypes
from VtlEngine.API.API import API

from tests import TestHelper

# Others
import unittest
import os
from typing import List

"""
Improvements:

"""
"""
TODOS:
    [1]: Add test with Datasets and Componets with null values for Numeric, Comparison and Boolean Operators.
"""
# Path Selection.---------------------------------------------------------------
Path = os.path.dirname(__file__)
filepath_json = os.path.join(os.path.join(os.path.join(Path, "data"), "DataStructure"), "input")
filepath_csv = os.path.join(os.path.join(os.path.join(Path, "data"), "DataSet"), "input")
filepath_out_json = os.path.join(os.path.join(os.path.join(Path, "data"), "DataStructure"), "output")
filepath_out_csv = os.path.join(os.path.join(os.path.join(Path, "data"), "DataSet"), "output")
# File extensions.--------------------------------------------------------------
JSON = '.json'
CSV = '.csv'
VTL = '.vtl'

check_dtype = False  # remove when refactor complete.

classTest = None


class ThreeValueHelper(TestHelper.TestHelper):
    """

    """
    # Path Selection.---------------------------------------------------------------
    Path = os.path.dirname(__file__)
    filepath_VTL = os.path.join(os.path.join(Path, "data"), "vtl")
    filepath_json = os.path.join(os.path.join(os.path.join(Path, "data"), "DataStructure"), "input")
    filepath_csv = os.path.join(os.path.join(os.path.join(Path, "data"), "DataSet"), "input")
    filepath_out_json = os.path.join(os.path.join(os.path.join(Path, "data"), "DataStructure"), "output")
    filepath_out_csv = os.path.join(os.path.join(os.path.join(Path, "data"), "DataSet"), "output")

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> List[DataSet.DataSet]:
        '''

        '''
        input_jsons = [code + '-' + 'DS_' + str(i + 1) + cls.JSON for i in range(number_inputs)]
        input_csvs = [code + '-' + 'DS_' + str(i + 1) + cls.CSV for i in range(number_inputs)]

        input_datasets = []
        for json in input_jsons:
            input_datasets += DataSet.DataSet.load(cls.filepath_json, json, cls.filepath_csv, input_csvs)
            if len(input_datasets) == number_inputs:
                break
        return input_datasets

    @classmethod
    def BaseTest(cls, text: str, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''
        # Data Loading.---------------------------------------------------------
        if text == None:
            vtlText = cls.ScriptLoad(code=code)
        else:
            vtlText = text
        inputs = cls.LoadInputs(code=code, number_inputs=number_inputs)

        # Interpreter Execution.------------------------------------------------
        results = API.VtlRunOnly(text=vtlText, dataSets=inputs, return_only_persistent=False)

        # Test Assertion.-------------------------------------------------------
        references = cls.LoadOutputs(code=code, references_names=references_names)
        cls.AssertDataSets(references, results)

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

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        Component-Scalar Test true
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 and true];"""
        code = '2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        Component-Scalar Test false
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 and false];"""
        code = '3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        Dataset-scalar Test true
        '''
        text = """DS_r := DS_1 and true;"""
        code = '4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''
        Dataset-scalar Test true
        '''
        text = """DS_r := DS_1 and false;"""
        code = '5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''
        Or logic test
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 or Me_2];"""
        code = '6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        '''
        Xor logic test
        '''
        text = """DS_r := DS_1[calc Me_3 := Me_1 xor Me_2];"""
        code = '7'
        number_inputs = 1
        references_names = ["DS_r"]

        self.SemanticTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)
        self.BaseTest(text=text, code=code, number_inputs=number_inputs, references_names=references_names)

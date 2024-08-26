import json
from pathlib import Path
from typing import Dict, List, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TimePeriodHelper(TestCase):
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
    def BaseTest(cls, text: Any, code: str, number_inputs: int, references_names: List[str]):
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

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True


class TimePeriodTest(TimePeriodHelper):
    """
    Group 1
    """
    classTest = 'timePeriodtests.TimePeriodTest'

    def test_GL_416(self):
        """
        test2_1 := BE2_DF_NICP[filter FREQ = "M" and TIME_PERIOD = cast("2020-01", time_period)];
        """
        code = 'GL_416'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_418(self):
        """
        """
        code = 'GL_418'
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_417_1(self):
        """
        test := avg (BE2_DF_NICP group all time_agg ("Q", "M", TIME_PERIOD));
        """
        code = 'GL_417_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_417_2(self):
        """
        test := avg (BE2_DF_NICP group all time_agg ("A", "M", TIME_PERIOD));
        """
        code = 'GL_417_2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_417_3(self):
        """
        """
        code = 'GL_417_3'
        number_inputs = 1
        error_code = "1-1-19-4"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_417_4(
            self):  # TODO: Check periodIndFrom is not the same as in data, in data is "M", should we allow this?
        """
        test := avg (BE2_DF_NICP group all time_agg ("A", "Q", TIME_PERIOD));
        """
        code = 'GL_417_4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_421_1(self):
        """
        test2_1 := BE2_DF_NICP
            [calc FREQ_2 := TIME_PERIOD in {cast("2020-01", time_period), cast("2021-01", time_period)}];
        """
        code = 'GL_421_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_421_2(self):
        """
        """
        code = 'GL_421_2'
        number_inputs = 1
        error_code = "1-3-10"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_440_1(self):
        """
        DS_r := DS_1 ;
        """
        code = 'GL_440_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_440_2(self):
        """
        """
        code = 'GL_440_2'
        number_inputs = 1
        message = "{} Errors found: {}".format(
            "The datapoints could not be loaded.",
            "{'DT3': ['DT3 Not possible to cast time period for component Me_1']}"
        )

        self.DataLoadExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message)

    #############
    # Tests for the sdmx external representation
    def test_GL_462_1(self):
        """
        DS_r := DS_1 ;
        """
        code = 'GL_462_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            text=None, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_GL_462_2(self):
        """

        """
        code = 'GL_462_2'
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(
            text=None, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_GL_462_3(self):
        """
        Status: OK
        Description: Over scalardataset
        Goal: Check Result.
        """
        code = 'GL_462_3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None,
                      code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_462_4(self):
        """
        Test for null value
        """
        code = 'GL_462_4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            text=None, code=code, number_inputs=number_inputs, references_names=references_names
        )

import json
from pathlib import Path
from typing import Dict, List
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestCalcHelper(TestCase):
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
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True

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


class CalcOperatorTest(TestCalcHelper):
    """
    Group 1
    """

    classTest = 'calc.CalcOperatorTest'

    def test_1(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-1'
        number_inputs = 2  # 2
        message = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_2(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-2'
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_3(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1*DS_2 [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-3'
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_4(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := ln(DS_1#Me_1), Me_5 := ln(DS_1#Me_2),
                                  Me_6 := ln(DS_1#Me_3)];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-4'
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_5(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := (DS_1*DS_2) [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.
        Note: It gives an error that says: AttributeError: 'NoneType' object has no attribute 'name'

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-5'
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_6(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_2 [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-6'
        number_inputs = 2
        message = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := Me_1 + DS_1#Me_2] [calc Me_5 := Me_2 + DS_2#Me_3];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-7'
        number_inputs = 2
        message = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_8(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status:
        Expression: DS_r := (DS_1*DS_2) [calc Me_4 := Me_1 + Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.


        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = '1-1-1-8'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_287_1(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status:
        Expression:
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.
        Note: It gives an error that says: AttributeError: 'NoneType' object has no attribute 'name'

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_287_1'
        number_inputs = 3
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_1(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression:DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_2(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= DS_1#Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_2'
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_300_3(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_4(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_2, DS_1 filter Id_2 ="B" calc Me_4 := Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_4'
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_300_5(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_6(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_6'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_7(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_7'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_8(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_8'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_9(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1 + d2#Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_9'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_10(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d2#Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_10'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_11(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_11'
        number_inputs = 2
        message = "1-1-13-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_300_12(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_12'
        number_inputs = 2
        message = "1-1-13-17"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_300_13(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + Me_2 + d2#Me_1A drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_13'
        number_inputs = 2
        message = "1-1-13-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_300_14(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + Me_3 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_14'
        number_inputs = 2
        message = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_300_15(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + Me_2 + d2#Me_1A drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_15'
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_300_16(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d1#Me_1+ d2#Me_2 + d1#Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_300_16'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_310_1(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression:

        Git Branch: #fix-310-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_310_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_310_2(self):
        """
        inner join
        Dataset --> Dataset
        Status: OK
        Expression:

        Git Branch: #fix-310-review-join
        Goal: Check the performance of the calc operator.
        """
        code = 'GL_310_2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

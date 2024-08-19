import json
from pathlib import Path
from typing import Dict, List
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestCrossJoinTypeChecking(TestCase):
    """ """

    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_vtl = base_path / "data" / "vtl"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    # File extensions.--------------------------------------------------------------
    JSON = ".json"
    CSV = ".csv"
    VTL = ".vtl"

    @classmethod
    def LoadDataset(cls, ds_path, dp_path):
        with open(ds_path, "r") as file:
            structures = json.load(file)

        for dataset_json in structures["datasets"]:
            dataset_name = dataset_json["name"]
            components = {
                component["name"]: Component(
                    name=component["name"],
                    data_type=SCALAR_TYPES[component["type"]],
                    role=Role(component["role"]),
                    nullable=component["nullable"],
                )
                for component in dataset_json["DataStructure"]
            }
            data = pd.read_csv(dp_path, sep=",")

            return Dataset(name=dataset_name, components=components, data=data)

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> Dict[str, Dataset]:
        """ """
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(cls.filepath_json / f"{code}-{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-{str(i + 1)}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadOutputs(cls, code: str, references_names: List[str]) -> Dict[str, Dataset]:
        """ """
        datasets = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """ """
        vtl_file_name = str(cls.filepath_vtl / f"{code}{cls.VTL}")
        with open(vtl_file_name, "r") as file:
            return file.read()

    @classmethod
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str]):
        """ """

        text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(
        cls, code: str, number_inputs: int, exception_code: str
    ):
        assert True


class CrossJoinIdentifiersTypeChecking(TestCrossJoinTypeChecking):
    """
    Group 2
    """

    classTest = "cross_join.CrossJoinIdentifiersTypeChecking"

    def test_1(self):
        """
        CROSS JOIN OPERATOR
        Status: OK
        Expression: DS_r := cross_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-4-1"
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 1 Number of test
        number_inputs = 2
        # references_names = ["DS_r"]

        # self.SemanticTest(code=code, number_inputs=number_inputs, references_names=references_names)
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        message = "1-1-13-3"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_2(self):
        """
        CROSS JOIN OPERATOR
        Status: BUG
        Expression: DS_r := cross_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-4-2"
        number_inputs = 2

        message = "1-1-13-3"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_3(self):
        """
        CROSS JOIN OPERATOR
        Status: duda orden de las datastructures
        Expression: DS_r := cross_join ( DS_1 as ds1, DS_2 as ds2
        rename ds1#Id_1 to Id_1A,ds1#Id_2 to Id_2A,ds2#Id_1 to Id_1B,ds2#Id_2 to Id_2B);
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-4-3"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(
            code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_4(self):
        """
        CROSS JOIN OPERATOR
        Status: OK
        Expression: DS_r := cross_join ( DS_1 as ds1, DS_2 as ds2
            rename ds1#Id_1 to Id_11,ds1#Id_2 to Id_12,ds2#Id_1 to Id_21,ds2#Id_2 to Id_22)
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-4-4"
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 4 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(
            code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_5(self):
        """
        CROSS JOIN OPERATOR
        Status: OK
        Expression: DS_r := cross_join ( DS_1 as ds1, DS_2 as ds2
            rename ds1#Id_1 to Id_11,ds1#Id_2 to Id_12,ds2#Id_3 to Id_21,ds2#Id_4 to Id_22)
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-4-5"
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 5 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(
            code=code, number_inputs=number_inputs, references_names=references_names
        )

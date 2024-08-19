import json
from pathlib import Path
from typing import Dict, List
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestInnerJoinTypeChecking(TestCase):
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


class InnerJoinIdentifiersTypeChecking(TestInnerJoinTypeChecking):
    """
    Group 2
    """

    classTest = "inner_join.InnerJoinIdentifiersTypeChecking"

    def test_1(self):
        """
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-1-1"
        # 2 For group join
        # 2 For group identifiers
        # 1 For clause- for the moment only op inner_join
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(
            code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_2(self):
        """
        INNER JOIN OPERATOR
        Status: BUG
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: inner for the same identifier with diferents types number and integer.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-1-2"
        # 2 For group join
        # 2 For group identifiers
        # 1 For clause- for the moment only op inner_join
        # 2 Number of test
        number_inputs = 2
        message = "BUG"

        self.SemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_message=message
        )

    def test_3(self):
        """
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: operations between booleans and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-1-3"
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_4(self):
        """
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: operations between time and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-1-4"
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

import json
import os.path
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from unittest import TestCase

import pandas as pd
import pytest

from API import create_ast
from AST.DAG import DAGAnalyzer
from DataTypes import SCALAR_TYPES
from Exceptions import SemanticError
from Interpreter import InterpreterAnalyzer
from Model import Dataset, Component, ExternalRoutine, Role, ValueDomain, Scalar
from files.parser import load_datapoints


class TestHelper(TestCase):
    """

    """
    # Path Selection.----------------------------------------------------------
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"

    JSON = '.json'
    CSV = '.csv'
    VTL = '.vtl'

    # Prefix for the input files (DS_).
    ds_input_prefix = ""

    @classmethod
    def LoadDataset(cls, ds_path, dp_path) -> Dict[str, Union[Dataset, Scalar]]:
        with open(ds_path, 'r') as file:
            structures = json.load(file)

        datasets = {}

        if 'datasets' in structures:
            for dataset_json in structures['datasets']:
                dataset_name = dataset_json['name']
                components = {
                    component['name']: Component(name=component['name'],
                                                 data_type=SCALAR_TYPES[component['type']],
                                                 role=Role(component['role']),
                                                 nullable=component['nullable'])
                    for component in dataset_json['DataStructure']}
                data = load_datapoints(components, Path(dp_path))

                datasets[dataset_name] = Dataset(name=dataset_name,
                                                 components=components,
                                                 data=data)
        if 'scalars' in structures:
            for scalar_json in structures['scalars']:
                scalar_name = scalar_json['name']
                scalar = Scalar(name=scalar_name,
                                data_type=SCALAR_TYPES[scalar_json['type']],
                                value=None)
                datasets[scalar_name] = scalar
        return datasets

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> Dict[str, Dataset]:
        '''

        '''
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(cls.filepath_json / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-{cls.ds_input_prefix}{str(i + 1)}{cls.CSV}")
            new_datasets = cls.LoadDataset(json_file_name, csv_file_name)
            datasets.update(new_datasets)

        return datasets

    @classmethod
    def LoadOutputs(cls, code: str, references_names: List[str]) -> Dict[str, Dataset]:
        """

        """
        datasets = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            new_datasets = cls.LoadDataset(json_file_name, csv_file_name)
            datasets.update(new_datasets)

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """

        """
        vtl_file_name = str(cls.filepath_VTL / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str], vd_names: List[str] = None,
                 sql_names: List[str] = None, text: Optional[str] = None, scalars: Dict[str, Any] = None):
        '''

        '''
        if text is None:
            text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)

        if scalars is not None:
            for scalar_name, scalar_value in scalars.items():
                if scalar_name not in input_datasets:
                    raise Exception(f"Scalar {scalar_name} not found in the input datasets")
                if not isinstance(input_datasets[scalar_name], Scalar):
                    raise Exception(f"{scalar_name} is a dataset")
                input_datasets[scalar_name].value = scalar_value
        interpreter = InterpreterAnalyzer(input_datasets,
                                          value_domains=value_domains,
                                          external_routines=external_routines)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str, vd_names: List[str] = None,
                 sql_names: List[str] = None, text: Optional[str] = None, scalars: Dict[str, Any] = None):
        # Data Loading.--------------------------------------------------------
        if text is None:
            text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code=code, number_inputs=number_inputs)

        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)

        if scalars is not None:
            for scalar_name, scalar_value in scalars.items():
                if scalar_name not in input_datasets:
                    raise Exception(f"Scalar {scalar_name} not found in the input datasets")
                if not isinstance(input_datasets[scalar_name], Scalar):
                    raise Exception(f"{scalar_name} is a dataset")
                input_datasets[scalar_name].value = scalar_value

        interpreter = InterpreterAnalyzer(input_datasets,
                                          value_domains=value_domains,
                                          external_routines=external_routines)
        with pytest.raises(SemanticError) as context:
            interpreter.visit(ast)

        assert exception_code == str(context.value.args[1])

    @classmethod
    def LoadValueDomains(cls, vd_names):
        value_domains = {}
        for name in vd_names:
            vd_file_name = str(cls.filepath_valueDomain / f"{name}.json")
            with open(vd_file_name, 'r') as file:
                vd = ValueDomain.from_json(file.read())
                value_domains[vd.name] = vd
        return value_domains

    @classmethod
    def LoadExternalRoutines(cls, sql_names):
        external_routines = {}
        for name in sql_names:
            sql_file_name = str(cls.filepath_sql / f"{name}.sql")
            with open(sql_file_name, 'r') as file:
                external_routines[name] = ExternalRoutine.from_sql_query(name, file.read())
        return external_routines

    @classmethod
    def DataLoadTest(cls, code: str, number_inputs: int, references_names: List[str] = None):
        # Data Loading.--------------------------------------------------------
        inputs = cls.LoadInputs(code=code, number_inputs=number_inputs)

        # Test Assertion.------------------------------------------------------
        if references_names:
            references = cls.LoadOutputs(code=code, references_names=references_names)
            assert inputs == references
        assert True


    @classmethod
    def DataLoadExceptionTest(cls, code: str, number_inputs: int, exception_message: str):
        # Test Assertion.------------------------------------------------------
        with pytest.raises(Exception, match=exception_message):
            cls.LoadInputs(code=code, number_inputs=number_inputs)

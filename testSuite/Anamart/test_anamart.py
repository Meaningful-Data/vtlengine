import json
import os.path
from pathlib import Path
from typing import List, Dict, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Dataset, Component, ExternalRoutine, Role, ValueDomain


class BugsHelper(TestCase):
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
            if not os.path.exists(dp_path):
                data = pd.DataFrame(columns=list(components.keys()))
            else:
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
        vtl_file_name = str(cls.filepath_VTL / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str], vd_names: List[str] = None, sql_names:List[str]=None):
        '''

        '''
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
        interpreter = InterpreterAnalyzer(input_datasets,
                                          value_domains=value_domains,
                                          external_routines=external_routines)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True

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


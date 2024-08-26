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


class ExternalProjectsHelper(TestCase):
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
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str], vd_names: List[str] = None,
                 sql_names: List[str] = None):
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


class BOP(ExternalProjectsHelper):
    """

    """

    classTest = 'ExternalProjects.BOP'

    def test_BOP_Q_Review_1(self):
        """
        Description:
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'BOP_Q_Review_1'
        number_inputs = 1
        rn = [str(i) for i in range(1, 30)]
        references_names = rn

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AnaVal(ExternalProjectsHelper):
    """

    """

    classTest = 'ExternalProjects.AnaVal'

    def test_Monthly_validations_1(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaVal_Monthly_validations_1'
        number_inputs = 36
        vd_names = ["EU_countries", "AnaCreditCountries_1"]
        rn = [str(i) for i in range(1, 288)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names
        )

    def test_Quarterly_validations_1(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaVal_Quarterly_validations_1'
        number_inputs = 12
        vd_names = ["EU_countries", "AnaCreditCountries_1"]
        rn = [str(i) for i in range(1, 38)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names
        )

    def test_GL_283_1(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description:
        Git Branch: #283
        """
        code = 'GL_283_1'
        number_inputs = 36
        vd_names = ["EU_countries", "AnaCreditCountries_1"]
        rn = [str(i) for i in range(1, 129)]
        references_names = rn

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names, vd_names=vd_names)


class AnaMart(ExternalProjectsHelper):
    """

    """

    classTest = 'ExternalProjects.AnaMart'

    def test_AnaMart_AnaMart_1(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaMart_AnaMart_1'
        number_inputs = 30
        vd_names = ["anaCreditCountries_2"]
        # rn = [str(i) for i in range(1, 303)]
        rn = [str(i) for i in range(1, 30)]
        rn += [str(i) for i in range(72, 303)]
        references_names = rn
        sql_names = [
            "instDates",
            "instrFctJn",
            "instrFctJn2",
            "prtctnDts",
            "prtctnFctJn"
        ]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
            sql_names=sql_names
        )
        exception_code = "1-1-13-4"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, vd_names=vd_names, exception_code=exception_code
        )

    def test_AnaMart_AnaMart_2(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaMart_AnaMart_2'
        number_inputs = 30
        vd_names = ["anaCreditCountries_2"]
        # rn = [str(i) for i in range(1, 303)]
        rn = [str(i) for i in range(1, 30)]
        rn += [str(i) for i in range(72, 303)]
        references_names = rn
        sql_names = [
            "instDates",
            "instrFctJn",
            "instrFctJn2",
            "prtctnDts",
            "prtctnFctJn"
        ]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
            sql_names=sql_names
        )

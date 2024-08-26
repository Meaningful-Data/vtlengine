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


class AnamartHelper(TestCase):
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


class ASeries(AnamartHelper):
    """

    """

    classTest = 'Anamart.ASeries'

    def test_2(self):
        '''

        '''
        code = 'A02'
        number_inputs = 2
        references_names = ["NLE", "LE", "LE2", "LE_3", "LE_FLL", "LE_JN", "NLE2", "DBTR", "ANAMART_ENTTY_LE"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''

        '''
        code = 'A03'
        number_inputs = 1
        references_names = ["DBTR_FCT_NN_PRFRMNG", "DBTR_FCT_TTL_NN_PFRMNG", "DBTR_FCT", "DBTR_FCT_TTL", "DBTR_FCT_JN",
                            "ANAMART_ENTTY_PRFRMNG_P", "ANAMART_ENTTY_PRFRMNG"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''

        '''
        code = 'A06'
        number_inputs = 1
        references_names = ["DBTR_GGRPHCL", "DBTR_GGRPHCL_T", "ANAMART_ENTTY_GGRPHCL_P", "ANAMART_ENTTY_GGRPHCL"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class BSeries(AnamartHelper):
    """

    """

    classTest = 'Anamart.BSeries'

    def test_3(self):
        '''

        '''
        code = 'B03'
        number_inputs = 3
        references_names = ["INSTRMNT_FCT_CRDTR_NLL", "INSTRMNT_FCT_CRDTR", "INSTRMNT_FCT_CRDTR_NOA",
                            "NMBR_CRDTRS_NT_OA_P", "NMBR_CRDTRS_NT_OA_D", "NMBR_CRDTRS_PR_INSTRMNT_P",
                            "NMBR_CRDTRS_PR_INSTRMNT", "INSTRMNT_FCT_TRNSFRRD_AMNT", "ANAMART_INSTRMNT_SHR_ONA_CRDTR"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''

        '''
        code = 'B04'
        number_inputs = 7
        references_names = ["INSTRMNT_FCT_CV_1", "INSTRMNT_PRTCTN_TTL_INSTRMNT", "INSTRMNT_FCT_CV_2", "ANAMART_CV",
                            "ANAMART_DV", "ANAMART_CV_DV"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''

        '''
        code = 'B05'
        number_inputs = 3
        references_names = ["INSTRMNT_PRTCTN_RCVD_K", "INSTRMT_PRTCN_3PTPC_P", "INSTRMT_PRTCN_3PTPC",
                            "INSTRMT_PRTCN_3PSPC_P", "INSTRMT_PRTCN_3PSPC", "INSTRMNT_PRTCTN_3PPC",
                            "INSTRMNT_PRTCTN_3PPC_K", "INSTRMNT_PRTCTN_3PPC_INSTR", "INSTRMNT_3PPC"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''

        '''
        code = 'B06'
        number_inputs = 2
        references_names = []

        # TODO Generate data for this test
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        '''

        '''
        code = 'B07'
        number_inputs = 3
        references_names = []

        # TODO Generate data for this test
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        '''

        '''
        code = 'B08'
        number_inputs = 3
        references_names = []

        # TODO Generate data for this test
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class CSeries(AnamartHelper):
    """

    """

    classTest = 'Anamart.CSeries'

    def test_2(self):
        '''

        '''
        code = 'C02'
        number_inputs = 2
        references_names = ["INSTRMNT_INFO", "INSTRMNT_INFO_K", "ANAMART_PRTCTN_INSTRMNT"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''

        '''
        code = 'C03'
        number_inputs = 2
        references_names = ["PRTCTN_FCT_AGGR_FV", "INSTRMNT_FCT_AGGR_KP", "INSTRMNT_FCT_AGGR_P", "INSTRMNT_FCT_AGGR",
                            "ANAMART_PRTCTN_AGGR"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''

        '''
        code = 'C04'
        number_inputs = 2
        references_names = ["PRTCTN_RCVD_KP", "PRTCTN_RCVD_AGG", "INSTRMTN_PRTCT_RCVD_AGG", "RTS_DNMNTR",
                            "INSTRMNT_PRTCTN_ALLCTN_DS", "ANAMART_PRTCTN_PV_ALLCTN"]
        message = "2-1-15-6"

        self.NewExceptionTest(text=None, code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        '''

        '''
        code = 'C05'
        number_inputs = 2
        references_names = ["THRD_PRTY_TTL_CLM_DS", "ALLCTBL_PRTCN_VL_P", "ALLCTBL_PRTCN_VL_DS",
                            "ANAMART_PRTCTN_ALLCTBL"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        '''

        '''
        code = 'C06'
        number_inputs = 2
        references_names = []

        # TODO Generate data for this test
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        '''

        '''
        code = 'C08'
        number_inputs = 3
        references_names = ["SHR_ONA_CRDTR_MIN", "SHR_ONA_CRDTR_AGG", "PAV_AGG_INSTRMNT_P",
                            "PAV_AGG_INSTRMNT", "PAV_AGG_PRTCN_P", "PAV_AGG_PRTCN",
                            "SHR_PV_JN_P", "SHR_PV_JN", "SHR_PV_DS",
                            "SHR_PV_DS_DRP", "ONA_JN", "ONA_DS",
                            "ANAMART_PRTCTN_ALLCTN"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        '''

        '''
        code = 'C09'
        number_inputs = 1
        references_names = ["MX_3PPC_DS", "NMBR_INSTRMNT_SCRD_DS_P", "NMBR_INSTRMNT_SCRD_DS",
                            "INCRMNTL_3PPC_JN", "INCRMNTL_3PPC_DS", "ANAMART_PRTCTN_OTHER"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AnamartFull(AnamartHelper):
    """

    """

    classTest = 'Anamart.AnamartFull'

    def test_1(self):
        '''

        '''
        code = 'AFULL'
        number_inputs = 12
        references_names = []
        vd_names = ["AFULL-1"]

        # TODO Generate data for this test
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names, vd_names=vd_names)
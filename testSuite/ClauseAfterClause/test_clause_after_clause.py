import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestClauseAfterClause(TestCase):
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

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True


class ClauseAfterClauseOperatorsTest(TestClauseAfterClause):
    """
    Group 1
    """

    classTest = 'clause_after_clause.ClauseAfterClauseOperatorsTest'

    def test_1(self):
        """
        FILTER: filter after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := DS_1 [filter Id_1 = 2021]
                                         [filter At_1 >= 1.0 and At_1 < 3.0];
                            DS_1 Dataset

        Description: filter after filter in one statement.

        Git Branch: #test-213-clause_after-clause.
        Goal: Check the result of filter after filter in one statement.
        """
        code = '1-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        FILTER: filter after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [filter Id_1 = 2021 and At_1 >= 1.0]
                                 [calc attribute At_3:= 2022, At_4:= 2020];
                            DS_1 Dataset

        Description: filter after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of filter after calc in one statement.
        """
        code = '1-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        FILTER: filter after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [filter Id_1 = 2021] [aggr Me_1:= sum( Me_1 )
                                  group by Id_1];
                            DS_1 Dataset

        Description: filter after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of filter after aggr in one statement.
        """
        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        FILTER: filter after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [filter Id_1 = 2021] [keep At_1, At_2 ];
                            DS_1 Dataset

        Description: filter after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of filter after keep in one statement.
        """
        code = '1-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        FILTER: filter after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [filter Id_1 = 2021] [drop At_1, At_2 ];
                            DS_1 Dataset

        Description: filter after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of filter after drop in one statement.
        """
        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        FILTER: filter after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [filter Id_1 = 2021] [rename Me_1 to Me11,
                                  Me_2 to Me12, At_1 to At11, At_2 to At12];
                            DS_1 Dataset

        Description: filter after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of filter after rename in one statement.
        """
        code = '1-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        FILTER: filter after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := DS_1 [filter Id_1 = 2021] [sub Id_2 = "Denmark"];
                            DS_1 Dataset

        Description: filter after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of filter after sub in one statement.
        """
        code = '1-1-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        CALC: calc after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0]
                                 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0];
                            DS_1 Dataset

        Description: calc after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after calc in one statement.
        """
        code = '1-1-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        CALC: calc after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0]
                                 [filter Id_1 = 2021 and Me_1 > 15.0];
                            DS_1 Dataset

        Description: calc after filter in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after filter in one statement.
        """
        code = '1-1-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        CALC: calc after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc attribute At_3:= 2022, At_4:= 2020]
                                 [aggr Me_5:= sum( Me_1 ), Me_7:= min(Me_2) group by Id_1];
                            DS_1 Dataset

        Description: calc after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after aggr in one statement.
        """
        code = '1-1-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        CALC: calc after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0]
                                 [keep Me_1, Me_2];
                            DS_1 Dataset

        Description: calc after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after keep in one statement.
        """
        code = '1-1-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        CALC: calc after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := DS_1 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0]
                                         [drop Me_2, At_1, At_2 ];
                            DS_1 Dataset

        Description: calc after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after drop in one statement.
        """
        code = '1-1-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        CALC: calc after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_1:= Me_1 * 3.0, Me_2:= Me_2 * 2.0]
                                 [rename Me_1 to Me_11, Me_2 to Me_22];
                            DS_1 Dataset

        Description: calc after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after rename in one statement.
        """
        code = '1-1-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        CALC: calc after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc attribute At_3:= 2022, At_4:= 2020]
                                 [sub Id_2 = "Denmark"];
                            DS_1 Dataset

        Description: calc after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of calc after sub in one statement.
        """
        code = '1-1-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        KEEP: keep after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep At_1, At_2 ] [keep At_1, At_2];
                            DS_1 Dataset

        Description: keep after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after keep in one statement.
        """
        code = '1-1-1-15'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        KEEP: keep after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep At_1, At_2 ] [filter At_1 > 1.0];
                            DS_1 Dataset

        Description: keep after filter in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after filter in one statement.
        """
        code = '1-1-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        KEEP: keep after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep Me_1, At_2] [calc Me_1:= Me_1 * 2];
                            DS_1 Dataset

        Description: keep after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after calc in one statement.
        """
        code = '1-1-1-17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        KEEP: keep after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep Me_1] [aggr Me_1:= sum( Me_1 ) group by Id_1];
                            DS_1 Dataset

        Description: keep after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after aggr in one statement.
        """
        code = '1-1-1-18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        KEEP: keep after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep Me_1,At_1,At_2] [drop At_1];
                            DS_1 Dataset

        Description: keep after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after drop in one statement.
        """
        code = '1-1-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        KEEP: keep after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep Me_1,At_1,At_2] [rename Me_1 to Me_11,
                                  At_1 to At_11, At_2 to At_22];
                            DS_1 Dataset

        Description: keep after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after rename in one statement.
        """
        code = '1-1-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        KEEP: keep after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [keep Me_1,At_1,At_2] [sub Id_2 = "Denmark"];
                            DS_1 Dataset

        Description: keep after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of keep after sub in one statement.
        """
        code = '1-1-1-21'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        DROP: drop after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_1,At_1,At_2] [drop Me_2];
                            DS_1 Dataset

        Description: drop after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after drop in one statement.
        """
        code = '1-1-1-22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        DROP: drop after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_1,At_2] [filter At_1 > 1.0];
                            DS_1 Dataset

        Description: drop after filter in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after filter in one statement.
        """
        code = '1-1-1-23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        DROP: drop after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_1,At_2] [calc Me_2:= Me_2 * 5];
                            DS_1 Dataset

        Description: drop after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after calc in one statement.
        """
        code = '1-1-1-24'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        DROP: drop after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop At_2] [aggr Me_1:= sum( Me_1 ) group by Id_1];
                            DS_1 Dataset

        Description: drop after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after aggr in one statement.
        """
        code = '1-1-1-25'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        DROP: drop after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_2,At_2] [keep Me_1,At_1];
                            DS_1 Dataset

        Description: drop after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after keep in one statement.
        """
        code = '1-1-1-26'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        DROP: drop after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_2,At_2] [rename Me_1 to Me_11,
                                  Me_3 to Me_33, At_1 to At_11];
                            DS_1 Dataset

        Description: drop after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after rename in one statement.
        """
        code = '1-1-1-27'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        DROP: drop after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [drop Me_2,At_2] [sub Id_2 = "Denmark"];
                            DS_1 Dataset

        Description: drop after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of drop after sub in one statement.
        """
        code = '1-1-1-28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        """
        RENAME: rename after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Me_1 to Me_11, Me_3 to Me_33,
                                  At_1 to At_11] [rename Me_11 to Me_111,
                                  Me_33 to Me_333, At_11 to At_111];
                            DS_1 Dataset

        Description: rename after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after rename in one statement.
        """
        code = '1-1-1-29'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        RENAME: rename after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Me_1 to Me_11, Me_3 to Me_33,
                                  At_1 to At_11] [filter At_11 > 1.0 and Me_11 > 5.0];
                            DS_1 Dataset

        Description: rename after filter in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after filter in one statement.
        """
        code = '1-1-1-30'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        RENAME: rename after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Me_1 to Me_11, Me_3 to Me_33,
                                  At_1 to At_11] [calc Me_33:= Me_33 * 5, At_11:= At_11 * 2.0];
                            DS_1 Dataset

        Description: rename after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after calc in one statement.
        """
        code = '1-1-1-31'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        """
        RENAME: rename after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename At_1 to At_11] [aggr Me_1:= sum(Me_1)
                                  group by Id_1];
                            DS_1 Dataset

        Description: rename after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after aggr in one statement.
        """
        code = '1-1-1-32'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_33(self):
        """
        RENAME: rename after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Me_1 to Me_11, Me_3 to Me_33,
                                  At_1 to At_11] [keep Me_11, Me_33, At_11];
                            DS_1 Dataset

        Description: rename after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after keep in one statement.
        """
        code = '1-1-1-33'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        """
        RENAME: rename after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Me_1 to Me_11, Me_3 to Me_33,
                                  At_1 to At_11] [drop At_11];
                            DS_1 Dataset

        Description: rename after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after drop in one statement.
        """
        code = '1-1-1-34'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        """
        RENAME: rename after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Me_1 to Me_11, Me_3 to Me_33,
                                  At_1 to At_11] [sub Id_2 = "Denmark"];
                            DS_1 Dataset

        Description: rename after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of rename after sub in one statement.
        """
        code = '1-1-1-35'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        SUB: sub after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = "2021"] [sub Id_2 = "Denmark"];
                            DS_1 Dataset

        Description: sub after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after sub in one statement.
        """
        code = '1-1-1-36'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_37(self):
        """
        SUB: sub after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = 2021] [filter Me_1 = 10.0];
                            DS_1 Dataset

        Description: sub after filter in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after filter in one statement.
        """
        code = '1-1-1-37'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        SUB: sub after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = 2021] [calc Me_2:= Me_2 * 5];
                            DS_1 Dataset

        Description: sub after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after calc in one statement.
        """
        code = '1-1-1-38'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        SUB: sub after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = 2021] [aggr Me_1:= sum( Me_1 ) group by Id_1];
                            DS_1 Dataset

        Description: sub after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after aggr in one statement.
        """
        code = '1-1-1-39'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        SUB: sub after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = 2021] [keep Me_1, Me_2, Me_3];
                            DS_1 Dataset

        Description: sub after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after keep in one statement.
        """
        code = '1-1-1-40'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        """
        SUB: sub after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = 2021] [drop At_1, At_2, Me_3];
                            DS_1 Dataset

        Description: sub after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after drop in one statement.
        """
        code = '1-1-1-41'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        SUB: sub after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [sub Id_1 = 2021] [rename Me_1 to Me_11,
                                  Me_3 to Me_33, At_1 to At_11, At_2 to At_22];
                            DS_1 Dataset

        Description: sub after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of sub after rename in one statement.
        """
        code = '1-1-1-42'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        """
        AGGR: aggr after aggr
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1]
                                 [aggr Me_4:= min( Me_1 ) group by Id_1];
                            DS_1 Dataset

        Description: aggr after aggr in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after aggr in one statement.
        """
        code = '1-1-1-43'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_44(self):
        """
        AGGR: aggr after filter
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1]
                                 [filter Me_1 > 20.0];
                            DS_1 Dataset

        Description: aggr after filter in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after filter in one statement.
        """
        code = '1-1-1-44'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_45(self):
        """
        AGGR: aggr after calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1]
                                 [calc Me_1 := Me_1 / 2];
                            DS_1 Dataset

        Description: aggr after calc in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after calc in one statement.
        """
        code = '1-1-1-45'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        """
        AGGR: aggr after keep
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1]
                                 [keep Me_1];
                            DS_1 Dataset

        Description: aggr after keep in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after keep in one statement.
        """
        code = '1-1-1-46'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        """
        AGGR: aggr after drop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ), Me_2:= sum( Me_2 )
                                  group by Id_1] [drop Me_1];
                            DS_1 Dataset

        Description: aggr after drop in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after drop in one statement.
        """
        code = '1-1-1-47'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        """
        AGGR: aggr after rename
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ), Me_2:= sum( Me_2 )
                                  group by Id_1] [rename Me_1 to Me_11, Me_2 to Me_22];
                            DS_1 Dataset

        Description: aggr after rename in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after rename in one statement.
        """
        code = '1-1-1-48'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_49(self):
        """
        AGGR: aggr after sub
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= sum( Me_1 ), Me_2:= sum( Me_2 )
                                  group by Id_1] [sub Id_1 = 2021];
                            DS_1 Dataset

        Description: aggr after sub in one statement.

        Git Branch: #test-213-clause-after-clause.
        Goal: Check the result of aggr after sub in one statement.
        """
        code = '1-1-1-49'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

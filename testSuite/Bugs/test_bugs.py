import json
from pathlib import Path
from typing import List, Dict, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Dataset, Component, Role


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
    def BaseTest(cls, text: Any, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''
        if text is None:
            text = cls.LoadVTL(code)
        else:
            text = ''
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets


class GeneralBugs(BugsHelper):
    """

    """

    classTest = 'Bugs.GeneralBugs'

    def test_GL_22(self):
        """
        Description: cast zero value to number-Integer.
        Git Branch: bug-22-improve-cast-zero-to-number-integer.
        Goal: Interpreter results.
        """
        code = 'GL_22'
        number_inputs = 1
        references_names = ['1']

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class JoinBugs(BugsHelper):
    """

    """

    classTest = 'Bugs.JoinBugs'

    def test_VTLEN_569(self):
        """

        """
        code = 'VTLEN_569'
        number_inputs = 2

        error_code = "1-1-13-6"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_VTLEN_572(self):
        """
        Description:
        Jira issue: VTLEN 572.
        Git Branch: bug-VTLEN-572-Inner-join-with-using-clause.
        Goal: Check semantic result.
        """
        code = 'VTLEN_572'
        number_inputs = 2
        references_names = ['1']

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_40(self):
        """

        """
        code = 'GL_40'
        number_inputs = 2
        references_names = ['1']

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_24(self):
        """

        """
        code = 'GL_24'
        number_inputs = 2
        message = "1-1-13-11"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_32(self):
        """
        Status: OK
        Expression: DS_r := inner_join ( AMOUNTS [ sub measure_ = "O" ] [ rename OBS_VALUE to O ] [ drop OBS_STATUS ]
                            as A ,  AMOUNTS [ sub measure_ = "V" ] [ rename OBS_VALUE to V ] [ drop OBS_STATUS ] as B);
        Description: Inner join on same dataset
        Git Branch: fix-32-names-joins
        Goal: Check Result.
        """
        code = 'GL_32'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_63(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_63'
        number_inputs = 2
        references_names = ['1', '2', '3', '4', '5']

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_14(self):
        """
        Description:
        Git Branch: bug-14-left_join-interpreter-error.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_14'
        number_inputs = 6
        message = "1-1-13-3"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_133_1(self):
        """
        Description: Fails on line 79-83: NLE2 := inner_join(NLE, LE_JN using DT_RFRNC);
                     This is not allowed as fails in case B2. (VTL Reference line 2269)
        Git Branch: fix-197-inner-using.
        Goal: Check exception.
        """
        code = 'GL_133_1'
        number_inputs = 1
        vd_names = ["GL_133_1-1"]

        message = "1-1-13-4"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message, vd_names=vd_names)

    def test_GL_133_2(self):
        """
        Description: NLE2 := inner_join(NLE, LE_JN using DT_RFRNC);
             This is not allowed as fails in case B2. (VTL Reference line 2269)
        Git Branch: fix-197-inner-using.
        Goal: Check exception.
        """
        code = 'GL_133_2'
        number_inputs = 2

        message = "1-1-13-4"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_133_3(self):
        """
        Description: DS_3 := inner_join(DS_1, DS_2 using Id_1);
            This is not allowed as fails in case B2. (VTL Reference line 2269)
        Git Branch: fix-197-inner-using.
        Goal: Check exception.
        """
        code = 'GL_133_3'
        number_inputs = 2
        message = "1-1-13-4"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_161_1(self):
        """
        Description: inner join with duplicated attributes.
        Git Branch: bug-161-inner-join-not-working-properly-attributes-duplicated.
        Goal: Check Exception.
        """
        code = 'GL_161_1'
        number_inputs = 2

        message = "1-1-13-3"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_161_2(self):
        """
        Description: inner join with duplicated attributes.
        Git Branch: bug-161-inner-join-not-working-properly-attributes-duplicated.
        Goal: Check Result.
        """
        code = 'GL_161_2'
        number_inputs = 2
        references_names = ['1']

        self.BaseTest(text=None,
                      code=code,
                      number_inputs=number_inputs,
                      references_names=references_names
                      )


    def test_GL_47_4(self):
        """
        Description: Two same rename.
        Git Branch: #47.
        Goal: Check Result.
        """
        code = 'GL_47_4'
        number_inputs = 2
        message = "1-3-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_47_5(self):
        """
        Description: Two same rename.
        Git Branch: #47.
        Goal: Check Result.
        """
        code = 'GL_47_5'
        number_inputs = 2
        # message = "Join conflict with duplicated names for column reference_date from original datasets."
        message = "1-1-13-3"  # "1-3-4"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_GL_47_6(self):
        """
        Description: Two duplicated components.
        Git Branch: #47.
        Goal: Check Result.
        """
        code = 'GL_47_6'
        number_inputs = 2
        message = "1-1-13-3"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_47_8(self):
        """
        Description:
        Git Branch: #47.
        Goal: Check Result.
        """
        code = 'GL_47_8'
        number_inputs = 2
        message = "1-1-13-3"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)


    def test_GL_64_1(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_64_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_2(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_64_2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_3(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_64_3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_4(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_64_4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_5(self):
        """
        Description: inner join
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_64_5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


    def test_GL_64_7(self):
        """
        Description: inner join
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_64_7'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_239_1(self):
        """
        Description: keep after keep(ds_2) inside a inner join. / Semantic error but the expression is correct.
        Git feat-234-new-grammar-parser.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_239_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


    def test_GL_250(self):
        """
        Description: Alias symbol and identifier in validate types
        Git fix-250-rename-sameDS.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_250'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_255(self):
        """
        Description: Drop inside a join
        Git fix-255-drop-join.
        Goal: Check semantic result and interpreter results.
        """
        code = 'GL_255'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_253(self):
        """
        Description: Duplicated component names on result join
        Git fix-253-duplicated-inner.
        Goal: Check semantic result (BKAR is duplicated).
        """
        code = 'GL_253'
        number_inputs = 2
        message = "1-1-13-3"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_279(self):
        """
        Description: Aggr with join and other clause
        Git fix-279-aggr-join.
        Goal: Check result.
        """
        code = 'GL_279'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

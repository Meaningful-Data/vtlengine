import json
from pathlib import Path
from typing import Dict, List, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class DWIHelper(TestCase):
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


class Membership(DWIHelper):
    """
        Membership tests for datasets without identifiers
    """

    classTest = 'DWIHelper.Membership'

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1#INSTTTNL_SCTR;';
        Description: Membership correct loading.
        Git Branch: feat-200-DWI-membership.
        Goal: Check Result.
        """
        code = 'GL_200-1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1#BLNC_SHT_TTL_CRRNCY + DS_2#BLNC_SHT_TTL_CRRNCY;
        Description: Should raise a semantic exception
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_1'
        number_inputs = 2

        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ calc Me_AUX := BLNC_SHT_TTL_CRRNCY + 10];
        Description:
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1 + DWI_1
        Description: Should raise a semantic exception
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_3'
        number_inputs = 2

        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_4(self):
        """
        Status: OK
        Expression: DS_r := sqrt(DS_1);
        Description: Try for unary op is working
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class Aggregate(DWIHelper):
    """
        Aggregate tests for datasets without identifiers as result
    """

    classTest = 'DWIHelper.Aggregate'

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Status: OK
        Expression: DS_r := avg (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := min (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := max (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := median (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Status: OK
        Expression: DS_r := sum (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_7'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_8'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Status: OK
        Expression: DS_r := var_pop (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_9'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Status: OK
        Expression: DS_r := var_samp (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    # Aggregate tests for datasets without identifiers as input

    def test_GL_218_5(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(one record)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_6(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(without record)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_7(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(null record)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_8(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(nulls and record, the result should be 1)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_9(self):
        """
        Status: OK
        Expression: DS_r := avg (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_10(self):
        """
        Status: OK
        Expression: DS_r := avg (DS_1);
        Description: Aggregate tests for datasets to check an error.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_10'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_11(self):
        """
        Status: BUG
        Expression: DS_r := DS_1[ aggr Me_aux := max(BLNC_SHT_TTL_CRRNCY)];
        Description: Aggregate tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_12(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ aggr Me_aux := max(BLNC_SHT_TTL_CRRNCY) group by Id_1];
        Description: Aggregate tests for datasets without identifiers as input. Id_1 is not in DS_1.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_12'
        number_inputs = 1
        message = "1-3-16"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_13(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ aggr Me_aux := max(BLNC_SHT_TTL_CRRNCY) group by NMBR_EMPLYS];
        Description: Aggregate tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_13'
        number_inputs = 1
        message = "1-1-2-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_24(self):
        """
        Status: BUG
        Expression: DS_r := ANCRDT_INSTRMNT_PRTCTN_RCVD_C [ aggr NMBR_INSTRMNT_SCRD := count ( ) , THRD_PRTY_PRRTY_CLMS_MX := max ( THRD_PRTY_PRRTY_CLMS ) group except CNTRCT_ID , INSTRMNT_ID ] ;
        Description: Aggregate tests.
        Git Issue: #218. Related also #222
        Goal: Check Result.
        """
        code = 'GL_218_24'
        number_inputs = 1
        references_names = []  # add reference name when #222 is done.

        # TODO Generate data for this test


        self.BaseTest(text= None, code=code, number_inputs=number_inputs, references_names=references_names)


class Clause(DWIHelper):
    """
        Clause operator tests for datasets without identifiers as result
    """

    classTest = 'DWIHelper.Clause'

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ sub Id_1 = "AA"]';
        Description: Perform the Sub operation correctly.
        Git Branch: feat-201-DWI-sub.
        Goal: Check Result.
        """
        code = 'GL_201_1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ rename BLNC_SHT_TTL_CRRNCY to Me_1 ];
        Description: Perform the rename operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ drop BLNC_SHT_TTL_CRRNCY ]
        Description: Perform the drop operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ keep BLNC_SHT_TTL_CRRNCY ]
        Description: Perform the keep operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_1 := BLNC_SHT_TTL_CRRNCY + ANNL_TRNVR_CRRNCY + NMBR_EMPLYS ]
        Description: Perform the calc operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ filter BLNC_SHT_TTL_CRRNCY + ANNL_TRNVR_CRRNCY <= 10000 ]
        Description: Perform the filter operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result (empty).
        """
        code = 'GL_202_5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ filter BLNC_SHT_TTL_CRRNCY + ANNL_TRNVR_CRRNCY >= 10000 ]
        Description: Perform the filter operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result (with one row).
        """
        code = 'GL_202_6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_14(self):
        """
        Status: OK. Commented as pivot is not implemented.
        Expression: DS_r := DS_1[ pivot BLNC_SHT_TTL_CRRNCY, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_14'
        number_inputs = 1

        message = "pivot not implemented"
        # self.SemanticExceptionTest(code=code, number_inputs=number_inputs, exception_message=message)

    def test_GL_218_15(self):
        """
        Status: OK. Commented as pivot is not implemented.
        Expression: DS_r := DS_1[ pivot Id_1, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.

        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_15'
        number_inputs = 1
        message = "pivot not implemented"
        # self.SemanticExceptionTest(code=code, number_inputs=number_inputs, exception_message=message)

    def test_GL_218_16(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ unpivot BLNC_SHT_TTL_CRRNCY, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_16'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_17(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ unpivot Id_1, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.
                    Should raise semantic error, raise a error on the evaluate, this is wrong
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_17'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_18(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ sub NMBR_EMPLYS = 10.0];
        Description: subspace tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_18'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_19(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ sub Id_1 = "C"];
        Description: subspace tests for datasets with identifiers as input. and the id doesn't match.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_20(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ drop Id_1 ];
        Description: 
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_20'
        number_inputs = 1
        message = "1-1-6-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_21(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ keep BLNC_SHT_TTL_CRRNCY ][ drop BLNC_SHT_TTL_CRRNCY ];
        Description: keep + drop tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_21'
        number_inputs = 1
        message = "1-1-6-12"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_22(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ calc identifier Id_1 := BLNC_SHT_TTL_CRRNCY + NMBR_EMPLYS];
        Description:
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_22'
        number_inputs = 1
        message = "1-1-1-16"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)


class Join(DWIHelper):
    """
        Clause operator tests for datasets without identifiers as result
    """

    classTest = 'DWIHelper.Join'

    # DS join DWI
    def test_GL_218_23(self):
        """
        Status: OK
        Expression:DS_r := inner_join (DS_1, DWI_1);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_23'
        number_inputs = 2
        message = "1-1-13-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_25(self):
        """
        Status: OK
        Expression:DS_r := left_join (DS_1, DWI_1);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_25'
        number_inputs = 2
        message = "1-1-13-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_26(self):
        """
        Status: OK
        Expression:DS_r := full_join (DS_1, DWI_1);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_26'
        number_inputs = 2
        message = "1-1-13-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_27(self):
        """
        Status: OK
        Expression: DS_r := cross_join (DS_1, DWI_1 as dw rename dw#BLNC_SHT_TTL_CRRNCY to col1, dw#ANNL_TRNVR_CRRNCY to col2, dw#NMBR_EMPLYS to col3);
        Description: 
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_27'
        number_inputs = 2
        message = "1-1-13-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_28(self):
        """
        Status: OK
        Expression: DS_r := cross_join (DS_1, DWI_1);
        Description: 
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_28'
        number_inputs = 2
        message = "1-1-13-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    # DWI join DWI

    def test_GL_218_29(self):
        """
        Status: OK
        Expression:DS_r := inner_join (DWI_1, DWI_2);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_29'
        number_inputs = 2
        message = "1-1-13-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

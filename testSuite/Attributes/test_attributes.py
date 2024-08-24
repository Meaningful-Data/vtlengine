import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestAttributesHelper(TestCase):
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
        datasets_references = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets_references[dataset.name] = dataset

        return datasets_references

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """

        """
        vtl_file_name = str(cls.filepath_vtl / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, text: Optional[str], code: str, number_inputs: int, references_names: List[str]):
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


class GeneralPurposeOperatorsTest(TestAttributesHelper):
    """
    Group 1
    """

    classTest = 'test_attributes.GeneralPurposeOperatorsTest'

    def test_16(self):
        """
        PARENTHESES: ()
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := (DS_1 + DS_2) * DS_3 ;
                            DS_1 Dataset
                            DS_2 Dataset
                            DS_3 Dataset

        Description: The operations enclosed in the parentheses are evaluated first.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: Do the operations using parentheses and check their attributes.
        """
        code = '1-4-1-1'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        PERSISTENT ASSIGNMENT: <-
        Dataset --> Dataset
        Status: OK
        Expression: DS_r <- DS_1 ;
                            DS_1 Measure Dataset

        Description: The input operand op is assigned to the persistent result re,
        which assumes the same value as op.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The input operand op is assigned to the persistent result re,
        which assumes the same value as op and check their attributes.
        """
        code = '1-4-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        NON-PERSISTENT ASSIGNMENT: :=
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 ;
                            DS_1 Measure Dataset

        Description: The value of the operand op is assigned to the result re,
        which is non-persistent and therefore is not stored. As mentioned, the
        operand op can be obtained through an expression as complex as needed
        (for example op can be the expression DS_1 - DS_2). The result re is a
        non-persistent Data Set that has the same data structure as the Operand.
        For example in DS_r := DS_1 the data structure of DS_r is the same as
        the one of DS_1.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The value of the operand op is assigned to the result re, which
        is non-persistent and therefore is not stored and check their attributes.
        """
        code = '1-4-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        NON-PERSISTENT ASSIGNMENT: :=
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 - DS_2 ;
                            DS_1 Measure Dataset
                            DS_1 Measure Dataset

        Description: The value of the operand op is assigned to the result re,
        which is non-persistent and therefore is not stored. As mentioned, the
        operand op can be obtained through an expression as complex as needed
        (for example op can be the expression DS_1 - DS_2). The result re is a
        non-persistent Data Set that has the same data structure as the Operand.
        For example in DS_r := DS_1 the data structure of DS_r is the same as
        the one of DS_1.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The value of the operand op is assigned to the result re, which
        is non-persistent and therefore is not stored and check their attributes.
        """
        code = '1-4-1-4'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        MEMBERSHIP: #
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1#Me_1 ;
                            DS_1 Measure Dataset

        Description: The membership operator returns a Data Set having the same
        Identifier Components of ds and a single Measure.
        Note: If comp is a Measure in ds, then comp is maintained in the result
        while all other Measures are dropped.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The membership operator returns a Data Set having the same
        Identifier Components of ds and a single Measure and check their attributes.
        """
        code = '1-4-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        MEMBERSHIP: #
        Dataset --> Dataset
        Expression: DS_r := DS_1#Id_1 ;
                            DS_1 Measure Dataset

        Description: The membership operator returns a Data Set having the same
        Identifier Components of ds and a single Measure.
        Note: f comp is an Identifier or an Attribute Component in ds, then all
        the existing Measures of ds are dropped in the result and a new Measure
        is added. The Data Points’ values for the new Measure are the same as
        the values of comp in ds.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The membership operator returns a Data Set having the same
        Identifier Components of ds and a single Measure and check their attributes.
        """
        code = '1-4-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        MEMBERSHIP: #
        Dataset --> Dataset
        Expression: DS_r := DS_1#At_1 ;
                            DS_1 Measure Dataset

        Description: The membership operator returns a Data Set having the same
        Identifier Components of ds and a single Measure.
        Note: f comp is an Identifier or an Attribute Component in ds, then all
        the existing Measures of ds are dropped in the result and a new Measure
        is added. The Data Points’ values for the new Measure are the same as
        the values of comp in ds.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The membership operator returns a Data Set having the same
        Identifier Components of ds and a single Measure and check their attributes.
        """
        code = '1-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r <- Test_8(DS_1); ;
                            DS_1 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := Test_9(DS_1); ;
                            DS_1 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := Test_10(DS_1); ;
                            DS_1 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Expression: DS_r := Test_11(DS_1); ;
                            DS_1 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := Test_12(DS_1); ;
                            DS_1 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := Test_13(DS_1,DS_2);
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-13'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := Test_14(DS_1,DS_2);
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-14'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_15(self):
        """
        USER-DEFINED OPERATOR CALL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := Test_15(DS_1,1.0);
                            DS_1 Measure Dataset

        Description: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order, the first argument as the value of the first
        parameter, the second argument as the value of the second parameter, and so on.

        Git Branch: #179 General purpose operators attributes tests.
        Goal: The invoked user-defined operator is evaluated. The arguments
        passed to the operator in the invocation are associated to the corresponding
        parameters in positional order and check their attributes.
        """
        code = '1-4-1-15'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class JoinOperatorsTest(TestAttributesHelper):
    """
    Group 2
    """

    classTest = 'test_attributes.JoinOperatorsTest'

    def test_1(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 filter Id_1 = 2021 and At_1 = 1.0 );
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 filter Id_1 = 2021 and At_1 = 1.0 );
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 filter Id_1 = 2021 and At_1 > 1.0 );
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [filter Id_1 = 2021 and At_1 > 1.0];
                    DS_r := cross_join ( DS_r1 as d1 );
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-4'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 calc attribute At_3:= 2022, At_4:= 2020 );
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (calc attribute At_3:= 2022);
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (calc attribute At_3:= 2022);
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 [calc attribute At_3:= 2022, At_4:= 2020] as d1);
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := inner_join ( DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1] as d1 );
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := left_join ( DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1] as d1 );
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := full_join ( DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1] as d1);
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 [aggr Me_1:= sum( Me_1 ) group by Id_1] as d1);
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 [keep At_1, At_2 ] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 [keep At_1, At_2 ] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 [keep At_1, At_2 ] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-15'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 [keep At_1, At_2 ] as d1);
                    DS_1 Measure Dataset


        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 [drop At_2] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 [drop At_1, At_2] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 [drop At_1, At_2] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 [drop At_1, At_2] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [rename Me_1 to Me11, Me_2 to Me12,
                                        At_1 to At11, At_2 to At12] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-21'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 [rename Me_1 to Me11, Me_2 to Me12,
                                        At_1 to At11, At_2 to At12] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 [rename Me_1 to Me11, Me_2 to Me12,
                                        At_1 to At11, At_2 to At12] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join (DS_1 [rename Me_1 to Me11, Me_2 to Me12,
                                        At_1 to At11, At_2 to At12] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-24'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join (DS_1 [rename Id_1 to Id_11, Id_2 to Id_12,
                                        Me_1 to Me11, Me_2 to Me12, At_1 to At11,
                                        At_2 to At12] as d1);
                    DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-25'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 [sub Id_1 = 2021, Id_2 = "Spain"] as d1 );
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-26'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 [sub Id_1 = 2021, Id_2 = "Spain"] as d1 )
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-27'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 [sub Id_1 = 2021, Id_2 = "Spain"] as d1)
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-28'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 sub Id_1 = 2021, Id_2 = "Spain" )
                            DS_1 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-29'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1 as d1, DS_2 as d2 );
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-30'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 as d1, DS_2 as d2 );
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-31'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 as d1, DS_2 as d2 );
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-32'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_33(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 as d1, DS_2 as d2 );
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-33'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [aggr Me_11:= sum( Me_11 ) group by Id_1 ];
                    DS_r2 := DS_2 [aggr Me_12:= sum( Me_21 ) group by Id_1 ];
                    DS_r := inner_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-34'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [aggr Me_11:= sum( Me_11 ) group by Id_1 ];
                    DS_r2 := DS_2 [aggr Me_12:= sum( Me_21 ) group by Id_1 ];
                    DS_r := left_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-35'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [aggr Me_11:= sum( Me_11 ) group by Id_1 ];
                    DS_r2 := DS_2 [aggr Me_12:= sum( Me_21 ) group by Id_1 ];
                    DS_r := full_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-36'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_37(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [aggr Me_11:= sum( Me_11 ) group by Id_1 ];
                    DS_r2 := DS_2;
                    DS_r := cross_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-37'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [filter Id_1 = 2021 and At_11 = 1.0 ];
                    DS_r2 := DS_2;
                    DS_r := inner_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-38'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [filter Id_1 = 2021 and At_11 < 3.0 ];
                    DS_r2 := DS_2 [filter Id_1 = 2021 and At_21 > 1.0 ];
                    DS_r := left_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-39'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [filter Id_1 = 2021 and At_11 < 3.0 ];
                    DS_r2 := DS_2 [filter Id_1 = 2021 and At_21 > 1.0 ];
                    DS_r := full_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-40'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [filter Id_1 = 2021 and At_11 < 3.0 ];
                    DS_r2 := DS_2 [filter Id_1 = 2021 and At_21 > 1.0 ];
                    DS_r := cross_join (DS_r1 as d1, DS_r2 as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-41'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [calc attribute At_3:= 2022];
                    DS_r2 := DS_2 [calc attribute At_4:= 2021];
                    DS_r := inner_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-42'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [calc attribute At_3:= 2022];
                    DS_r2 := DS_2 [calc attribute At_4:= 2021];
                    DS_r := left_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-43'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_44(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [calc attribute At_3:= 2022];
                    DS_r2 := DS_2 [calc attribute At_4:= 2021];
                    DS_r := full_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-44'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_45(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [calc attribute At_3:= 2022];
                    DS_r2 := DS_2 [calc attribute At_4:= 2021];
                    DS_r := cross_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-45'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [sub Id_1 = 2021, Id_2 = "Spain"];
                    DS_r2 := DS_2;
                    DS_r := inner_join ( DS_r1 as d1, DS_r2 as d2 );
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-46'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [sub Id_1 = 2021, Id_2 = "Spain"];
                    DS_r2 := DS_2 [sub Id_1 = 2021, Id_2 = "Denmark"];
                    DS_r := left_join ( DS_r1 as d1, DS_r2 as d2 );
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-47'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [sub Id_1 = 2021, Id_2 = "Spain"];
                    DS_r2 := DS_2 [sub Id_1 = 2021, Id_2 = "Denmark"];
                    DS_r := full_join ( DS_r1 as d1, DS_r2 as d2 );
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-48'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_49(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [sub Id_11 = 2021, Id_12 = "Spain"];
                    DS_r2 := DS_2 [sub Id_21 = 2021, Id_22 = "Denmark"];
                    DS_r := cross_join ( DS_r1 as d1, DS_r2 as d2 );
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-49'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_50(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [keep At_1];
                    DS_r2 := DS_2 [keep At_2];
                    DS_r := inner_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-50'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_51(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [keep At_1];
                    DS_r2 := DS_2 [keep At_2];
                    DS_r := left_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-51'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_52(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [keep At_2];
                    DS_r2 := DS_2 [keep At_1];
                    DS_r := full_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-52'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_53(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [keep At_1];
                    DS_r2 := DS_2 [keep At_2];
                    DS_r := cross_join ( DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset


        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-53'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_54(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [drop At_1,Me_11];
                    DS_r2 := DS_2 [drop At_2];
                    DS_r := inner_join (DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-54'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_55(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [drop At_1,Me_11];
                    DS_r2 := DS_2 [drop At_2];
                    DS_r := left_join (DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-55'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_56(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [drop At_1,Me_11];
                    DS_r2 := DS_2 [drop At_2];
                    DS_r := full_join (DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-56'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_57(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r1 := DS_1 [drop At_1,Me_11];
                    DS_r2 := DS_2 [drop At_2];
                    DS_r := cross_join (DS_r1 as d1, DS_r2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-57'
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_58(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id11,
                            d1#Id_2 to Id12, d2#Id_1 to Id21, d2#Id_2 to Id22,
                            d1#Me_1 to Me11, d1#Me_2 to Me12, d1#Me_3 to Me13,
                            d2#Me_1 to Me21, d2#Me_2 to Me22, d1#At_1 to At11,
                            d1#At_2 to At12);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset



        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-58'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_59(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id11,
                            d1#Id_2 to Id12, d2#Id_1 to Id21, d2#Id_2 to Id22,
                            d1#Me_1 to Me11, d1#Me_2 to Me12, d1#Me_3 to Me13,
                            d2#Me_1 to Me21, d2#Me_2 to Me22, d1#At_1 to At11,
                            d1#At_2 to At12);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset



        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-59'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_60(self):
        """
        JOIN: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id11,
                            d1#Id_2 to Id12, d2#Id_1 to Id21, d2#Id_2 to Id22,
                            d1#Me_1 to Me11, d1#Me_2 to Me12, d1#Me_3 to Me13,
                            d2#Me_1 to Me21, d2#Me_2 to Me22, d1#At_1 to At11,
                            d1#At_2 to At12);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-60'
        number_inputs = 2
        message = "1-1-13-17"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_61(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id11,
                            d1#Id_2 to Id12, d2#Id_1 to Id21, d2#Id_2 to Id22,
                            d1#Me_1 to Me11, d1#Me_2 to Me12, d1#Me_3 to Me13,
                            d2#Me_1 to Me21, d2#Me_2 to Me22,
                            d1#At_1 to At11, d1#At_2 to At12)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset


        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-61'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_62(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id11,
                            d1#Id_2 to Id12, d2#Id_1 to Id21, d2#Id_2 to Id22, d1#Me_1 to Me11,
                            d1#Me_2 to Me12, d1#Me_3 to Me13, d2#Me_1 to Me21, d2#Me_2 to Me22)
                            DS_1 Measure Dataset
                            DS_2 Measure Dataset


        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-62'
        number_inputs = 2
        message = "1-1-13-3"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_63(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [filter Id_1 = 2021 and At_11 < 3.0] as d1,
                    DS_2 [filter Id_1 = 2021 and At_21 > 1.0] as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-63'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_64(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [aggr Me_11:= sum( Me_11 ) group by Id_1 ] as d1,
                    DS_2 [aggr Me_12:= sum( Me_21 ) group by Id_1 ] as d2)
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-64'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_65(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [calc attribute At_3:= 2022] as d1,
                    DS_2 [calc attribute At_4:= 2021] as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-65'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_66(self):
        """
        JOIN: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [sub Id_1 = 2021, Id_2 = "Spain"] as d1, DS_2 as d2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-66'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_67(self):
        """
        JOIN: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1 as d1, DS_2 as d2 keep d1#At_1, d2#At_2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-67'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_68(self):
        """
        JOIN: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join ( DS_1 as d1, DS_2 as d2 drop d1#At_1, d1#Me_11, d2#At_2);
                    DS_1 Measure Dataset
                    DS_2 Measure Dataset

        Description: Join operators serve to combine data points from two or
        more datasets, based on related components between them.

        Git Branch: #180 Join operators attributes tests.
        Goal: Combine data points from two or more datasets, based on related
        components between them and check their attributes.
        """
        code = '2-4-1-68'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class StringOperatorsTest(TestAttributesHelper):
    """
    Group 3
    """

    classTest = 'test_attributes.StringOperatorsTest'

    def test_1(self):
        """
        CONCATENATION OPERATOR
        string --> string
        Status: OK
        Expression: DS_r := (DS_1 || DS_2) || DS_3 ;
                            DS_1 Measure String
                            DS_2 Measure String
                            DS_3 Measure String

        Description: Concatenation of two or more strings.

        Git Branch: #123 String operators attributes tests.
        Goal: Do the concatenation of two or more strings and check their attributes.
        """
        code = '3-4-1-1'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        CONCATENATION OPERATOR
        string --> string
        Status: OK
        Expression: DS_r := (DS_1 || DS_2) || DS_3 ;
                            DS_1 Measure String
                            DS_2 Measure Integer
                            DS_3 Measure String

        Description: Concatenation of two or more strings.

        Git Branch: #123 String operators attributes tests.
        Goal: Do the concatenation of two or more strings and check their attributes.
        """
        code = '3-4-1-2'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        CONCATENATION OPERATOR
        string --> string
        Status: OK
        Expression: DS_r := (DS_1 || DS_2) || DS_3 ;
                            DS_1 Measure String
                            DS_2 Measure String
                            DS_3 Measure String

        Description: Concatenation of two or more strings.

        Git Branch: #123 String operators attributes tests.
        Goal: Do the concatenation of two or more strings.
        Note: All roles are Attributes.
        """
        code = '3-4-1-3'
        number_inputs = 3
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_4(self):
        """
        CONCATENATION OPERATOR
        Integer --> string
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure Integer
                            DS_2 Measure String


        Description: Concatenation of an Integer with a String.

        Git Branch: #123 String operators attributes tests.
        Goal: Do the concatenation of Integer with String and check their attributes.
        """
        code = '3-4-1-4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        CONCATENATION OPERATOR
        string --> string
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure String


        Description: Concatenation of two Strings.

        Git Branch: #123 String operators attributes tests.
        Goal: Do the concatenation of two Strings and check their attributes.
        """
        code = '3-4-1-5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        CONCATENATION OPERATOR
        string --> string
        Status: OK
        Expression: DS_r := DS_1 || DS_2
                            DS_1 Measure String
                            DS_2 Measure String


        Description: Concatenation of two Strings.

        Git Branch: #123 String operators attributes tests.
        Goal: Do the concatenation of two Strings and check their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-6'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        WHITESPACE REMOVAL OPERATOR
        String --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #123 String operators attributes tests.
        Goal: Whitespace removal from a string and check their attributes.
        """
        code = '3-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        WHITESPACE REMOVAL OPERATOR
        String --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #123 String operators attributes tests.
        Goal: Whitespace removal from a string and check their attributes.
        Note: Output Dataset with two measures.
        """
        code = '3-4-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        WHITESPACE REMOVAL OPERATOR
        String --> String
        Status: OK
        Expression: DS_r := trim(DS_1)
                            DS_1 Measure String


        Description: Whitespace removal.

        Git Branch: #123 String operators attributes tests.
        Goal: Whitespace removal from a string and check their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-9'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_10(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #123 String operators attributes tests.
        Goal: Converts the character case of a string in upper case and check
        their attributes.
        """
        code = '3-4-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.
            -- Exception raised, should it be raised? Manual does not specify to have one measure --

        Git Branch: #123 String operators attributes tests.
        Goal: Converts the character case of a string in upper case and check
        their attributes.
        Note: Output Dataset whit two measures.
        """
        code = '3-4-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := upper(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #123 String operators attributes tests.
        Goal: Converts the character case of a string in upper case and check
        their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-12'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_13(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #123 String operators attributes tests.
        Goal: Converts the character case of a string in lower case and check
        their attributes.
        """
        code = '3-4-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #123 String operators attributes tests.
        Goal: Converts the character case of a string in lower case and check
        their attributes.
        Note: Output Dataset with two measures.
        """
        code = '3-4-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        CHARACTER CASE CONVERSION: UPPER/LOWER
        String --> String
        Status: OK
        Expression: DS_r := lower(DS_1)
                            DS_1 Measure String


        Description: upper/lower operators.

        Git Branch: #123 String operators attributes tests.
        Goal: Converts the character case of a string in lower case and check
        their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-15'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_16(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1, start, length)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #123 String operators attributes tests.
        Goal: The operator extracts a substring from op, which must be string
        type and check their attributes.
        """
        code = '3-4-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1, start, length)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #123 String operators attributes tests.
        Goal: The operator extracts a substring from op, which must be string
        type and check their attributes.
        Note: Output Dataset with two measures.
        """
        code = '3-4-1-17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        SUB-STRING EXTRACTION
        String --> String
        Status: OK
        Expression: DS_r := substr (DS_1, start, length)
                            DS_1 Measure String


        Description: Substr operators. The operator extracts a substring from
        the operand, which must be string type. The substring starts from the
        start character of the input string and has a number of characters equal
        to the length parameter.

        Git Branch: #123 String operators attributes tests.
        Goal: The operator extracts a substring from op, which must be string
        type and check their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-18'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_19(self):
        """
        STRING PATTERN REPLACEMENT
        String --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure String


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #123 String operators attributes tests.
        Goal: Replaces all the occurrences of a specified string-pattern and check their attributes.
        """
        code = '3-4-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        STRING PATTERN REPLACEMENT
        String --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure String


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #123 String operators attributes tests.
        Goal: Replaces all the occurrences of a specified string-pattern and check their attributes.
        Note: Output Dataset with two measures.
        """
        code = '3-4-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        STRING PATTERN REPLACEMENT
        String --> String
        Status: OK
        Expression: DS_r := replace (op , pattern1, pattern2)
                            DS_1 Measure String


        Description: Replaces all the occurrences of a specified string-pattern
        (pattern1) with another one (pattern2).

        Git Branch: #123 String operators attributes tests.
        Goal: Replaces all the occurrences of a specified string-pattern and check their attributes.
        Note: Output Dataset with two measures.
        Note: All roles are Attributes.
        """
        code = '3-4-1-21'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_22(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1,pattern,start,occurrence)
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #123 String operators attributes tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern) and check their attributes.
        """
        code = '3-4-1-22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1,pattern,start,occurrence)
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #123 String operators attributes tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern) and check their attributes.
        Note: Output Dataset with two measures.
        """
        code = '3-4-1-23'
        number_inputs = 1
        message = "1-1-18-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_24(self):
        """
        STRING PATTERN LOCATION
        String --> Integer
        Status: OK
        Expression: DS_r := instr (DS_1,pattern,start,occurrence)
                            DS_1 Measure String


        Description: The operator returns the position in the input string of a
        specified string (pattern). The search starts from the start character
        of the input string and finds the nth occurrence of the pattern,
        returning the position of its first character.

        Git Branch: #123 String operators attributes tests.
        Goal: The operator returns the position in the input string of a
        specified string (pattern) and check their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-24'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_25(self):
        """
        STRING LENGTH
        String --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure String


        Description: Returns the length of a string.

        Git Branch: #123 String operators attributes tests.
        Goal: Returns the length of a string. and check their attributes.
        """
        code = '3-4-1-25'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        STRING LENGTH
        String --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure String


        Description: Returns the length of a string.

        Git Branch: #123 String operators attributes tests.
        Goal: Returns the length of a string. and check their attributes.
        Note: Output Dataset with two measures.
        """
        code = '3-4-1-26'
        number_inputs = 1
        message = "1-1-18-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_27(self):
        """
        STRING LENGTH
        String --> Integer
        Status: OK
        Expression: DS_r := length(DS_1)
                            DS_1 Measure String


        Description: Returns the length of a string.

        Git Branch: #123 String operators attributes tests.
        Goal: Returns the length of a string. and check their attributes.
        Note: All roles are Attributes.
        """
        code = '3-4-1-27'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)


class NumericOperatorsTest(TestAttributesHelper):
    """
    Group 4
    """

    classTest = 'test_attributes.NumericOperatorsTest'

    def test_1(self):
        """
        UNARY PLUS: +
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := + DS_1
                    DS_r2 := + DS_r
                    DS_1 Measure Number

        Description: The operator + returns the operand unchanged.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator + returns the operand unchanged and check their attributes.
        """
        code = '4-4-1-1'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        UNARY MINUS: -
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := - DS_1
                    DS_r2 := - DS_r
                    DS_1 Measure Number

        Description: The operator - inverts the sign of op.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator - inverts the sign of op and check their attributes.
        """
        code = '4-4-1-2'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        ADDITION: +
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 + 5.0
                    DS_1 Measure Number

        Description: The operator addition returns the sum of two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator addition returns the sum of two numbers and check their attributes.
        """
        code = '4-4-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        ADDITION: +
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 + DS_2
                    DS_1 Measure Number

        Description: The operator addition returns the sum of two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator addition returns the sum of two numbers and check their attributes.
        """
        code = '4-4-1-4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        SUBSTRACTION: -
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 - 5.0
                    DS_1 Measure Number

        Description: The operator subtraction returns the difference of two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator subtraction returns the difference of two numbers and
        check their attributes.
        """
        code = '4-4-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        SUBSTRACTION: -
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r1 - DS_r2
                    DS_1 Measure Number

        Description: The operator subtraction returns the difference of two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator subtraction returns the difference of two numbers and
        check their attributes.
        """
        code = '4-4-1-6'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        MULTIPLICATION: *
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 * 5.0
                    DS_1 Measure Number

        Description: The operator multiplication returns the product of two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator multiplication returns the product of two numbers and
        check their attributes.
        """
        code = '4-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        MULTIPLICATION: *
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 * DS_2
                    DS_1 Measure Number

        Description: The operator multiplication returns the product of two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator multiplication returns the product of two numbers and
        check their attributes.
        """
        code = '4-4-1-8'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        DIVISION: /
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 / 5.0
                    DS_1 Measure Number

        Description: The operator division divides two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator division divides two numbers and check their attributes.
        """
        code = '4-4-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        DIVISION: /
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 / DS_2
                    DS_1 Measure Number

        Description: The operator division divides two numbers.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator division divides two numbers and check their attributes.
        """
        code = '4-4-1-10'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        MODULO: MOD
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := mod ( DS_1, 12 )
                    DS_1 Measure Number

        Description: The operator mod returns the remainder of op1 divided by op2.
        It returns op1 if divisor op2 is 0.
        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator mod returns the remainder of op1 divided by op2 and
        check their attributes.
        """
        code = '4-4-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        MODULO: MOD
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := mod (DS_1,DS_2)
                    DS_1 Measure Number

        Description: The operator mod returns the remainder of op1 divided by op2.
        It returns op1 if divisor op2 is 0.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator mod returns the remainder of op1 divided by op2 and
        check their attributes.
        """
        code = '4-4-1-12'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        ROUNDING: ROUND
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := round (DS_1,2)
                    DS_1 Measure Number
                    numDigit (2) Measure Integer
                    numDigit: the number of positions to round to

        Description: The operator round rounds the operand to a number of positions
        at the right of the decimal point equal to the numDigit parameter.
        The decimal point is assumed to be at position 0.
        ***Note: Attributes are strings

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator round rounds the operand to a number of positions
        at the right of the decimal point equal to the numDigit parameter.
        The decimal point is assumed to be at position 0 and check their attributes.
        """
        code = '4-4-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        TRUNCATION: TRUNC
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := round (DS_1,1)
                    DS_1 Measure Number
                    numDigit (1) Measure Integer
                    numDigit: the number of position from which to trunc

        Description: The operator trunc truncates the operand to a number of
        positions at the right of the decimal point equal to the numDigit parameter.
        The decimal point is assumed to be at position 0.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator trunc truncates the operand to a number of
        positions at the right of the decimal point equal to the numDigit parameter.
        The decimal point is assumed to be at position 0 and check their attributes.
        """
        code = '4-4-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        CEILING: CEIL
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := ceil (DS_1)
                    DS_1 Measure Number


        Description: The operator ceil returns the smallest integer greater
        than or equal to op.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator ceil returns the smallest integer greater
        than or equal to op and check their attributes.
        """
        code = '4-4-1-15'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        FLOOR: FLOOR
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := floor (DS_1)
                    DS_1 Measure Number


        Description: The operator floor returns the greatest integer which is
        smaller than or equal to op.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator floor returns the greatest integer which is smaller
        than or equal to op and check their attributes.
        """
        code = '4-4-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        ABSOLUTE VAUE: ABS
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := abs (DS_1)
                    DS_1 Measure Number


        Description: The operator abs calculates the absolute value of a number.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator abs calculates the absolute value of a number and
        check their attributes.
        """
        code = '4-4-1-17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        EXPONENTIAL: EXP
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := exp (DS_1)
                    DS_1 Measure Number


        Description: The operator exp returns e (base of the natural logarithm)
        raised to the op-th power.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator exp returns e (base of the natural logarithm) raised
        to the op-th power and check their attributes.
        """
        code = '4-4-1-18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        NATURAL LOGARITHM: LN
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := ln (DS_1)
                    DS_1 Measure Number


        Description: The operator ln calculates the natural logarithm of a number.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator ln calculates the natural logarithm of a number and
        check their attributes.
        """
        code = '4-4-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        POWER: POWER
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := power (DS_1,2)
                    DS_1 Measure Number


        Description: The operator power raises a number (the base) to another
        one (the exponent).
        Note: power ( base , exponent ): base is the operand, exponent the exponent of the power.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator power raises a number (the base) to another one
        (the exponent) and check their attributes.
        """
        code = '4-4-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        LOGARITHM: LOG
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := log (DS_1,10)
                    DS_1 Measure Number


        Description: The operator log calculates the logarithm of num base op.
        Note: log ( op , num ): op is the base of the logarithm, num is the number
        to which the logarithm is applied.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator log calculates the logarithm of num base op and check their attributes.
        """
        code = '4-4-1-21'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        SQUARE ROOT: SQRT
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sqrt (DS_1)
                    DS_1 Measure Number


        Description: The operator sqrt calculates the square root of a number.

        Git Branch: #138 Numeric operators attributes tests.
        Goal: The operator sqrt calculates the square root of a number and check their attributes.
        """
        code = '4-4-1-22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class ComparisonOperatorsTest(TestAttributesHelper):
    """
    Group 5
    """

    classTest = 'test_attributes.ComparisonOperatorsTest'

    def test_1(self):
        """
        EQUAL TO
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 = 0.08
                            DS_1 Measure Number



        Description: The operator returns TRUE if the left is equal to right,
        FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns TRUE if the left is equal to right,
        FALSE otherwise and check their attributes.
        """
        code = '5-4-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        NOT EQUAL TO
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 <> DS_2
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator returns FALSE if the left is equal to right,
        TRUE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns FALSE if the left is equal to right, TRUE
        otherwise and check their attributes.
        """
        code = '5-4-1-2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        GREATER THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 > DS_2
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator > returns TRUE if left is greater than right,
        FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator > returns TRUE if left is greater than right,
        FALSE otherwise and check their attributes.
        """
        code = '5-4-1-3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        GREATER THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 >= DS_2
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator >= returns TRUE if left is greater than or
        equal to right, FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator >= returns TRUE if left is greater than or equal to
        right, FALSE otherwise and check their attributes.
        """
        code = '5-4-1-4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        LESS THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 < DS_2
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator < returns TRUE if left is smaller than right,
        FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator < returns TRUE if left is smaller than right, FALSE
        otherwise and check their attributes.
        """
        code = '5-4-1-5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        LESS THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 <= DS_2
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator <= returns TRUE if left is smaller than or
        equal to right, FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator <= returns TRUE if left is smaller than or equal to
        right, FALSE otherwise and check their attributes.
        """
        code = '5-4-1-6'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        BETWEEN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := between (DS_1, from, to)
                            DS_1 Measure Number


        Description: The operator returns TRUE if op is greater than or equal
        to from and lower than or equal to to. In other terms, it is a shortcut
        for the following: op >= from and op <= to

        "op" the Data Set to be checked
        "from the left delimiter
        "to" the right delimiter

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns TRUE if op is greater than or equal
        to from and lower than or equal to to and check their attributes.
        """
        code = '5-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        ELEMENT OF: IN / NOT_IN
        Scalar --> Boolean
        Status: BUG
        Expression: DS_r := DS_1 in collection
                            DS_1 Measure Number
                            collection ::= set | valueDomainName


        Description: The in operator returns TRUE if op belongs to the
        collection, FALSE otherwise.

        "op" the operand to be tested
        "collection" the the Set or the Value Domain which contains the values
        "set" the Set which contains the values (it can be a Set name or a Set literal)
        "valueDomainName" the name of the Value Domain which contains the values

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The in operator returns TRUE if op belongs to the
        collection, FALSE otherwise and check their attributes.
        """
        code = '5-4-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_9(self):
        """
        ELEMENT OF: IN / NOT_IN
        Scalar --> Boolean
        Status: BUG
        Expression: DS_r := DS_1 in collection
                            DS_1 Measure Number
                            collection ::= set | valueDomainName


        Description: The not_in operator returns FALSE if op belongs to the
        collection, TRUE otherwise.

        "op" the operand to be tested
        "collection" the the Set or the Value Domain which contains the values
        "set" the Set which contains the values (it can be a Set name or a Set literal)
        "valueDomainName" the name of the Value Domain which contains the values

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The not_in operator returns FALSE if op belongs to the
        collection, TRUE otherwise and check their attributes.
        """
        code = '5-4-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        IS NULL
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := isnull(DS_1)
                            DS_1 Measure Number


        Description: The operator returns TRUE if the value of the operand is
        NULL, FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns TRUE if the value of the operand is NULL,
        FALSE otherwise. and check their attributes.
        """
        code = '5-4-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        EXIST IN:
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := exists_in (DS_1, DS_2,all)
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2.
        The result has the same Identifiers as op1 and a boolean Measure bool_var
        whose value, for each Data Point of op1, is TRUE if the combination of
        values of the common Identifier Components in op1 is found in a Data Point of
        op2, FALSE otherwise.
        If retain is all then both the Data Points having bool_var = TRUE and
        bool_var = FALSE are returned.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2 and check their attributes. If the
        retain parameter is omitted, the default is all.
        """
        code = '5-4-1-11'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        EXIST IN:
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := exists_in (DS_1, DS_2,true)
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2.
        The result has the same Identifiers as op1 and a boolean Measure bool_var
        whose value, for each Data Point of op1, is TRUE if the combination of
        values of the common Identifier Components in op1 is found in a Data Point of
        op2, FALSE otherwise.
        If retain is true then only the data points with bool_var = TRUE are returned.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2 and check their attributes.
        """
        code = '5-4-1-12'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        EXIST IN:
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := exists_in (DS_1, DS_2,false)
                            DS_1 Measure Number
                            DS_2 Measure Number


        Description: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2.
        The result has the same Identifiers as op1 and a boolean Measure bool_var
        whose value, for each Data Point of op1, is TRUE if the combination of
        values of the common Identifier Components in op1 is found in a Data Point of
        op2, FALSE otherwise.
        If retain is false then only the Data Points with bool_var = FALSE are returned.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2 and check their attributes.
        """
        code = '5-4-1-13'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        EQUAL TO
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 = 0.08
                            DS_1 Attribute



        Description: The operator returns TRUE if the left is equal to right,
        FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns TRUE if the left is equal to right,
        FALSE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-14'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_15(self):
        """
        NOT EQUAL TO
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 <> DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator returns FALSE if the left is equal to right,
        TRUE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns FALSE if the left is equal to right, TRUE
        otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-15'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_16(self):
        """
        GREATER THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 > DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator > returns TRUE if left is greater than right,
        FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator > returns TRUE if left is greater than right,
        FALSE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-16'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_17(self):
        """
        GREATER THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 >= DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator >= returns TRUE if left is greater than or
        equal to right, FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator >= returns TRUE if left is greater than or equal to
        right, FALSE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-17'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_18(self):
        """
        LESS THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 < DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator < returns TRUE if left is smaller than right,
        FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator < returns TRUE if left is smaller than right, FALSE
        otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-18'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_19(self):
        """
        LESS THAN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 <= DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator <= returns TRUE if left is smaller than or
        equal to right, FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator <= returns TRUE if left is smaller than or equal to
        right, FALSE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-19'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_20(self):
        """
        BETWEEN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := between (DS_1, from, to)
                            DS_1 Attribute


        Description: The operator returns TRUE if op is greater than or equal
        to from and lower than or equal to to. In other terms, it is a shortcut
        for the following: op >= from and op <= to

        "op" the Data Set to be checked
        "from the left delimiter
        "to" the right delimiter

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns TRUE if op is greater than or equal
        to from and lower than or equal to to and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-20'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_21(self):
        """
        ELEMENT OF: IN / NOT_IN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 in collection
                            DS_1 Attribute
                            collection ::= set | valueDomainName


        Description: The in operator returns TRUE if op belongs to the
        collection, FALSE otherwise.

        "op" the operand to be tested
        "collection" the the Set or the Value Domain which contains the values
        "set" the Set which contains the values (it can be a Set name or a Set literal)
        "valueDomainName" the name of the Value Domain which contains the values

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The in operator returns TRUE if op belongs to the
        collection, FALSE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-21'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_22(self):
        """
        ELEMENT OF: IN / NOT_IN
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := DS_1 in collection
                            DS_1 Attribute
                            collection ::= set | valueDomainName


        Description: The not_in operator returns FALSE if op belongs to the
        collection, TRUE otherwise.

        "op" the operand to be tested
        "collection" the the Set or the Value Domain which contains the values
        "set" the Set which contains the values (it can be a Set name or a Set literal)
        "valueDomainName" the name of the Value Domain which contains the values

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The not_in operator returns FALSE if op belongs to the
        collection, TRUE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-22'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_23(self):
        """
        IS NULL
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := isnull(DS_1)
                            DS_1 Attribute


        Description: The operator returns TRUE if the value of the operand is
        NULL, FALSE otherwise.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator returns TRUE if the value of the operand is NULL,
        FALSE otherwise. and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-23'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_24(self):
        """
        EXIST IN:
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := exists_in (DS_1, DS_2,all)
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2.
        The result has the same Identifiers as op1 and a boolean Measure bool_var
        whose value, for each Data Point of op1, is TRUE if the combination of
        values of the common Identifier Components in op1 is found in a Data Point of
        op2, FALSE otherwise.
        If retain is all then both the Data Points having bool_var = TRUE and
        bool_var = FALSE are returned.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2 and check their attributes. If the
        retain parameter is omitted, the default is all.
        Note: All measures are attributes.
        """
        code = '5-4-1-24'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        EXIST IN:
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := exists_in (DS_1, DS_2,true)
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2.
        The result has the same Identifiers as op1 and a boolean Measure bool_var
        whose value, for each Data Point of op1, is TRUE if the combination of
        values of the common Identifier Components in op1 is found in a Data Point of
        op2, FALSE otherwise.
        If retain is true then only the data points with bool_var = TRUE are returned.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2 and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-25'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        EXIST IN:
        Scalar --> Boolean
        Status: OK
        Expression: DS_r := exists_in (DS_1, DS_2,false)
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2.
        The result has the same Identifiers as op1 and a boolean Measure bool_var
        whose value, for each Data Point of op1, is TRUE if the combination of
        values of the common Identifier Components in op1 is found in a Data Point of
        op2, FALSE otherwise.
        If retain is false then only the Data Points with bool_var = FALSE are returned.

        Git Branch: #134 Comparison operators attributes tests.
        Goal: The operator takes under consideration the common Identifiers
        of op1 and op2 and checks if the combinations of values of these Identifiers
        which are in op1 also exist in op2 and check their attributes.
        Note: All measures are attributes.
        """
        code = '5-4-1-26'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        Time_Period DataSet Comparison
        Time_Period --> Boolean
        Status: OK
        Expression: DS_r := DS_1 <= DS_2
        """
        code = '5-4-1-27'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        Time_Period Component Comparison
        Time_Period --> Boolean
        Status: OK
        Expression: DS_r := DS_1[calc Me_3 := Me_1 = Me_2];
        """
        code = '5-4-1-28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class BooleanOperatorsTest(TestAttributesHelper):
    """
    Group 6
    """

    classTest = 'test_attributes.BooleanOperatorsTest'

    def test_1(self):
        """
        LOGICAL CONJUNCTION: AND
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := DS_1 and DS_2
                            DS_1 Boolean
                            DS_2 Boolean


        Description: The and operator returns TRUE if both operands are TRUE,
        otherwise FALSE.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The and operator returns TRUE if both operands are TRUE,
        otherwise FALSE and check their attributes.
        """
        code = '6-4-1-1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        LOGICAL CONJUNCTION: AND
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := DS_1 and DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The and operator returns TRUE if both operands are TRUE,
        otherwise FALSE.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The and operator returns TRUE if both operands are TRUE,
        otherwise FALSE and check their attributes.
        Note: All measures are attributes.
        """
        code = '6-4-1-2'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_3(self):
        """
        LOGICAL DISJUNCTION: OR
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := DS_1 or DS_2
                            DS_1 Boolean
                            DS_2 Boolean


        Description: The or operator returns TRUE if at least one of the operands
        is TRUE, otherwise FALSE.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The or operator returns TRUE if at least one of the operands
        is TRUE, otherwise FALSE and check their attributes.
        """
        code = '6-4-1-3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        LOGICAL DISJUNCTION: OR
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := DS_1 or DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The or operator returns TRUE if at least one of the operands
        is TRUE, otherwise FALSE.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The or operator returns TRUE if at least one of the operands
        is TRUE, otherwise FALSE and check their attributes.
        Note: All measures are attributes.
        """
        code = '6-4-1-4'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        """
        EXCLUSIVE DISJUNCTION: XOR
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := DS_1 xor DS_2
                            DS_1 Boolean
                            DS_2 Boolean


        Description: The xor operator returns TRUE if only one of the operand
        is TRUE (but not both), FALSE otherwise.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The xor operator returns TRUE if only one of the operand
        is TRUE (but not both), FALSE otherwise and check their attributes.
        """
        code = '6-4-1-5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        EXCLUSIVE DISJUNCTION: XOR
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := DS_1 xor DS_2
                            DS_1 Attribute
                            DS_2 Attribute


        Description: The xor operator returns TRUE if only one of the operand
        is TRUE (but not both), FALSE otherwise.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The xor operator returns TRUE if only one of the operand
        is TRUE (but not both), FALSE otherwise and check their attributes.
        Note: All measures are attributes.
        """
        code = '6-4-1-6'
        number_inputs = 2
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        LOGICAL NEGATION: NOT
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := not DS_1
                            DS_1 Boolean

        Description: The not operator returns TRUE if op is FALSE, otherwise TRUE.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The not operator returns TRUE if op is FALSE, otherwise TRUE and
        check their attributes.
        """
        code = '6-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        LOGICAL NEGATION: NOT
        Boolean --> Boolean
        Status: OK
        Expression: DS_r := not DS_1
                            DS_1 Boolean

        Description: The not operator returns TRUE if op is FALSE, otherwise TRUE.

        Git Branch: #135 Boolean operators attributes tests.
        Goal: The not operator returns TRUE if op is FALSE, otherwise TRUE and
        check their attributes.
        Note: All measures are attributes.
        """
        code = '6-4-1-8'
        number_inputs = 1
        message = "1-1-1-8"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)


class TimeOperatorsTest(TestAttributesHelper):
    """
    Group 7
    """

    classTest = 'test_attributes.TimeOperatorsTest'

    pass


class SetOperatorsTest(TestAttributesHelper):
    """
    Group 8
    """

    classTest = 'test_attributes.SetOperatorsTest'

    def test_1(self):
        """
        UNION: union
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := union (DS_1, DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The union operator implements the union of functions
        (i.e., Data Sets). The resulting Data Set has the same Identifier,
        Measure and Attribute Components of the operand Data Sets specified
        in the dsList, and contains the Data Points belonging to any of the
        operand Data Sets.

        Git Branch: #208 Set operators attributes tests.
        Goal: The union operator implements the union of functions
        (i.e., Data Sets) and check their attributes.
        """
        code = '8-4-1-1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        UNION: union
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := union (DS_1, DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The union operator implements the union of functions
        (i.e., Data Sets). The resulting Data Set has the same Identifier,
        Measure and Attribute Components of the operand Data Sets specified
        in the dsList, and contains the Data Points belonging to any of the
        operand Data Sets.

        Git Branch: #208 Set operators attributes tests.
        Goal: The union operator implements the union of functions
        (i.e., Data Sets) and check their attributes.
        """
        code = '8-4-1-2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        UNION: union
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := union (DS_1, DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The union operator implements the union of functions
        (i.e., Data Sets). The resulting Data Set has the same Identifier,
        Measure and Attribute Components of the operand Data Sets specified
        in the dsList, and contains the Data Points belonging to any of the
        operand Data Sets.

        Git Branch: #208 Set operators attributes tests.
        Goal: The union operator implements the union of functions
        (i.e., Data Sets) and check their attributes.
        """
        code = '8-4-1-3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        INTERSECTION: intersect
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := intersect(DS_1,DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The intersect operator implements the intersection of
        functions (i.e., Data Sets). The resulting Data Set has the same
        Identifier, Measure and Attribute Components of the operand Data Sets
        specified in the dsList, and contains the Data Points belonging to all
        the operand Data Sets.

        Git Branch: #208 Set operators attributes tests.
        Goal: The intersect operator implements the intersection of
        functions (i.e., Data Sets) and check their attributes.
        """
        code = '8-4-1-4'
        number_inputs = 2
        message = '[At_1,At_2] not in index'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        """
        INTERSECTION: intersect
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := intersect(DS_1,DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The intersect operator implements the intersection of
        functions (i.e., Data Sets). The resulting Data Set has the same
        Identifier, Measure and Attribute Components of the operand Data Sets
        specified in the dsList, and contains the Data Points belonging to all
        the operand Data Sets.

        Git Branch: #208 Set operators attributes tests.
        Goal: The intersect operator implements the intersection of
        functions (i.e., Data Sets) and check their attributes.
        """
        code = '8-4-1-5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        INTERSECTION: intersect
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := intersect(DS_1,DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The intersect operator implements the intersection of
        functions (i.e., Data Sets). The resulting Data Set has the same
        Identifier, Measure and Attribute Components of the operand Data Sets
        specified in the dsList, and contains the Data Points belonging to all
        the operand Data Sets.

        Git Branch: #208 Set operators attributes tests.
        Goal: The intersect operator implements the intersection of
        functions (i.e., Data Sets) and check their attributes.
        """
        code = '8-4-1-6'
        number_inputs = 2
        message = '[At_1,At_2] not in index'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        SET DIFFERENCE: setdiff
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := setdiff(DS_1,DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The operator implements the set difference of functions
        (i.e. Data Sets), interpreting the Data Points of the input Data Sets
        as the elements belonging to the operand sets, the minuend and the
        subtrahend, respectively. The operator returns one single Data Set,
        with the same Identifier, Measure and Attribute Components as the
        operand Data Sets, containing the Data Points that appear in the first
        Data Set but not in the second.

        Git Branch: #208 Set operators attributes tests.
        Goal: Implements the set difference of functions (i.e. Data Sets),
        interpreting the Data Points of the input Data Sets as the elements
        belonging to the operand sets, the minuend and the subtrahend,
        respectively and check their attributes.
        """
        code = '8-4-1-7'
        number_inputs = 2
        message = '[At_1,At_2] not in index'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_8(self):
        """
        SET DIFFERENCE: setdiff
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := setdiff(DS_1,DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The operator implements the set difference of functions
        (i.e. Data Sets), interpreting the Data Points of the input Data Sets
        as the elements belonging to the operand sets, the minuend and the
        subtrahend, respectively. The operator returns one single Data Set,
        with the same Identifier, Measure and Attribute Components as the
        operand Data Sets, containing the Data Points that appear in the first
        Data Set but not in the second.

        Git Branch: #208 Set operators attributes tests.
        Goal: Implements the set difference of functions (i.e. Data Sets),
        interpreting the Data Points of the input Data Sets as the elements
        belonging to the operand sets, the minuend and the subtrahend,
        respectively and check their attributes.
        """
        code = '8-4-1-8'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        SIMMETRIC DIFFERENCE: symdiff
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := symdiff(DS_1, DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The operator implements the symmetric set difference between
        functions (i.e. Data Sets), interpreting the Data Points of the input
        Data Sets as the elements in the operand Sets. The operator returns one
        Data Set, with the same Identifier, Measure and Attribute Components as
        the operand Data Sets, containing the Data Points that appear in the
        first Data Set but not in the second and the Data Points that appear in
        the second Data Set but not in the first one.

        Git Branch: #208 Set operators attributes tests.
        Goal: implements the symmetric set difference between
        functions (i.e. Data Sets), interpreting the Data Points of the input
        Data Sets as the elements in the operand Sets and check their attributes.
        """
        code = '8-4-1-9'
        number_inputs = 2
        message = '[At_1,At_2] not in index'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_10(self):
        """
        SIMMETRIC DIFFERENCE: symdiff
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := symdiff(DS_1, DS_2)
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The operator implements the symmetric set difference between
        functions (i.e. Data Sets), interpreting the Data Points of the input
        Data Sets as the elements in the operand Sets. The operator returns one
        Data Set, with the same Identifier, Measure and Attribute Components as
        the operand Data Sets, containing the Data Points that appear in the
        first Data Set but not in the second and the Data Points that appear in
        the second Data Set but not in the first one.

        Git Branch: #208 Set operators attributes tests.
        Goal: implements the symmetric set difference between
        functions (i.e. Data Sets), interpreting the Data Points of the input
        Data Sets as the elements in the operand Sets and check their attributes.
        """
        code = '8-4-1-10'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class ConditionalOperatorsTest(TestAttributesHelper):
    """
    Group 12
    """

    classTest = 'test_attributes.ConditionalOperatorsTest'

    def test_1(self):
        """
        IF-THEN-ELSE: IF
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if ( DS_cond#Id_4 = "F" ) then DS_1 else DS_2
                            DS_cond Dataset
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The if operator returns thenOperand if condition evaluates
        to true, elseOperand otherwise.

        Git Branch: #137 Conditional operators attributes tests.
        Goal: The if operator returns thenOperand if condition evaluates to true,
        elseOperand otherwise and check their attributes.
        """
        code = '12-4-1-1'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        IF-THEN-ELSE: IF
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if ( DS_cond#Id_4 = "M" ) then DS_2 else DS_1
                            DS_cond Dataset
                            DS_1 Dataset
                            DS_2 Dataset

        Description: The if operator returns thenOperand if condition evaluates
        to true, elseOperand otherwise.

        Git Branch: #137 Conditional operators attributes tests.
        Goal: The if operator returns thenOperand if condition evaluates to true,
        elseOperand otherwise.
        """
        code = '12-4-1-2'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        NVL: NVL
        Dataset --> Scalar
        Status: OK
        Expression: DS_r := nvl ( DS_1, 0 )
                    DS_1 Dataset
        Syntax: nvl ( op1 , op2 )
                op1 the first operand
                op2 the second operand

        Description: The operator nvl returns the value from op2 when the value
        from op1 is null, otherwise it returns the value from op1.

        Git Branch: #137 Conditional operators attributes tests.
        Goal: The operator nvl returns the value from op2 when the value
        from op1 is null, otherwise it returns the value from op1 and check
        their attributes.
        """
        code = '12-4-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        NVL: NVL
        Dataset --> Scalar
        Status: OK
        Expression: DS_r := nvl ( DS_1, 2021 )
                    DS_1 Dataset
        Syntax: nvl ( op1 , op2 )
                op1 the first operand
                op2 the second operand

        Description: The operator nvl returns the value from op2 when the value
        from op1 is null, otherwise it returns the value from op1.

        Git Branch: #137 Conditional operators attributes tests.
        Goal: The operator nvl returns the value from op2 when the value
        from op1 is null, otherwise it returns the value from op1 and check
        their attributes.
        """
        code = '12-4-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)


class ClauseOperatorsTest(TestAttributesHelper):
    """
    Group 13
    """

    classTest = 'test_attributes.ClauseOperatorsTest'

    def test_1(self):
        """
        FILTERING DATA POINTS: FILTER
        Boolean --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ filter Id_1 = 1 and Me_1 < 10 ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and a boolean
        Component expression (filterCondition) and filters the input Data Points
        according to the evaluation of the condition. When the expression is TRUE
        the Data Point is kept in the result, otherwise it is not kept
        (in other words, it filters out the Data Points of the operand Data Set
        for which filterCondition condition evaluates to FALSE or NULL).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator takes as input a Data Set (op) and a boolean
        Component expression (filterCondition) and filters the input Data Points
        according to the evaluation of the condition. When the expression is TRUE
        the Data Point is kept in the result, otherwise it is not kept
        (in other words, it filters out the Data Points of the operand Data Set
        for which filterCondition condition evaluates to FALSE or NULL) and check
        their attributes.
        """
        code = '13-4-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        FILTERING DATA POINTS: FILTER
        Boolean --> Dataset
        Status: OK
        Expression: DS_r := DS_1 DS_1 [ filter Me_1 ]
                            DS_1 Boolean

        Description: The operator takes as input a Data Set (op) and a boolean
        Component expression (filterCondition) and filters the input Data Points
        according to the evaluation of the condition. When the expression is TRUE
        the Data Point is kept in the result, otherwise it is not kept
        (in other words, it filters out the Data Points of the operand Data Set
        for which filterCondition condition evaluates to FALSE or NULL).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator takes as input a Data Set (op) and a boolean
        Component expression (filterCondition) and filters the input Data Points
        according to the evaluation of the condition. When the expression is TRUE
        the Data Point is kept in the result, otherwise it is not kept
        (in other words, it filters out the Data Points of the operand Data Set
        for which filterCondition condition evaluates to FALSE or NULL) and check
        their attributes.
        """
        code = '13-4-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        CALCULATION OF A COMPONENT: CALC
        Scalar --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_1:= Me_1 * 2 ]
                            DS_1 Scalar

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level. Each
        Component is calculated through an independent sub-expression. It is
        possible to specify the role of the calculated Component among measure,
        identifier, attribute, or viral attribute, therefore the calc clause can
        be used also to change the role of a Component when possible.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level. Each
        Component is calculated through an independent sub-expression. It is
        possible to specify the role of the calculated Component among measure,
        identifier, attribute, or viral attribute, therefore the calc clause can
        be used also to change the role of a Component when possible and check
        their attributes.
        """
        code = '13-4-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        CALCULATION OF A COMPONENT: CALC
        Scalar --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc attribute At_2:= “EP” ]
                            DS_1 Scalar

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level. Each
        Component is calculated through an independent sub-expression. It is
        possible to specify the role of the calculated Component among measure,
        identifier, attribute, or viral attribute, therefore the calc clause can
        be used also to change the role of a Component when possible.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level. Each
        Component is calculated through an independent sub-expression. It is
        possible to specify the role of the calculated Component among measure,
        identifier, attribute, or viral attribute, therefore the calc clause can
        be used also to change the role of a Component when possible and check
        their attributes.
        """
        code = '13-4-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        CALCULATION OF A COMPONENT: CALC
        Scalar --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc attribute At_2:= Me_1 ]
                            DS_1 Scalar

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level. Each
        Component is calculated through an independent sub-expression. It is
        possible to specify the role of the calculated Component among measure,
        identifier, attribute, or viral attribute, therefore the calc clause can
        be used also to change the role of a Component when possible.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level. Each
        Component is calculated through an independent sub-expression. It is
        possible to specify the role of the calculated Component among measure,
        identifier, attribute, or viral attribute, therefore the calc clause can
        be used also to change the role of a Component when possible and check
        their attributes.
        """
        code = '13-4-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        AGGREGATION: AGGR
        Scalar --> Dataset
        Dataset: OK
        Expression: DS_r := DS_1 [ aggr Me_1:= sum( Me_1 ) group by Id_1 , Id_2 ]
                            DS_1 Scalar

        Description: The operator aggr calculates aggregations of dependent
        Components (Measures or Attributes) on the basis of sub-expressions at
        Component level. Each Component is calculated through an independent
        sub-expression.
        **group by: the Data Points are grouped by the values of the specified
        Identifiers. The Identifiers not specified are dropped in the result.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator aggr calculates aggregations of dependent
        Components (Measures or Attributes) on the basis of sub-expressions at
        Component level. Each Component is calculated through an independent
        sub-expression and check their attributes.
        """
        code = '13-4-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        AGGREGATION: AGGR
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ aggr Me_3:= min( Me_1 ) group except Id_3 ]
                            DS_1 Dataset

        Description: The operator aggr calculates aggregations of dependent
        Components (Measures or Attributes) on the basis of sub-expressions at
        Component level. Each Component is calculated through an independent sub-expression.
        **group except: the Data Points are grouped by the values of the Identifiers
        not specified in the clause. The specified Identifiers are dropped in the result.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator aggr calculates aggregations of dependent
        Components (Measures or Attributes) on the basis of sub-expressions at
        Component level. Each Component is calculated through an independent
        sub-expression and check their attributes.
        """
        code = '13-4-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        MAINTAINING COMPONENTS: KEEP
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ keep Me_1 ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and some Component
        names of such a Data Set (comp). These Components can be Measures or
        Attributes of op but not Identifiers. The operator maintains the specified
        Components, drops all the other dependent Components of the Data Set
        (Measures and Attributes) and maintains the independent Components (Identifiers)
        unchanged. This operation corresponds to a projection in the usual relational
        join semantics (specifying which columns will be projected in among
        Measures and Attributes).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator maintains the specified
        Components, drops all the other dependent Components of the Data Set
        (Measures and Attributes) and maintains the independent Components (Identifiers)
        unchanged and check their attributes.
        """
        code = '13-4-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        MAINTAINING COMPONENTS: KEEP
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ keep Me_1, Me_2 ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and some Component
        names of such a Data Set (comp). These Components can be Measures or
        Attributes of op but not Identifiers. The operator maintains the specified
        Components, drops all the other dependent Components of the Data Set
        (Measures and Attributes) and maintains the independent Components (Identifiers)
        unchanged. This operation corresponds to a projection in the usual relational
        join semantics (specifying which columns will be projected in among
        Measures and Attributes).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator maintains the specified
        Components, drops all the other dependent Components of the Data Set
        (Measures and Attributes) and maintains the independent Components (Identifiers)
        unchanged and check their attributes.
        """
        code = '13-4-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        MAINTAINING COMPONENTS: KEEP
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ keep At_1 ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and some Component
        names of such a Data Set (comp). These Components can be Measures or
        Attributes of op but not Identifiers. The operator maintains the specified
        Components, drops all the other dependent Components of the Data Set
        (Measures and Attributes) and maintains the independent Components (Identifiers)
        unchanged. This operation corresponds to a projection in the usual relational
        join semantics (specifying which columns will be projected in among
        Measures and Attributes).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator maintains the specified
        Components, drops all the other dependent Components of the Data Set
        (Measures and Attributes) and maintains the independent Components (Identifiers)
        unchanged and check their attributes.
        """
        code = '13-4-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        REMOVAL OF COMPONENTS: DROP
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ drop At_1 ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and some Component
        names of such a Data Set (comp). These Components can be Measures or
        Attributes of op but not Identifiers. The operator drops the specified
        Components and maintains all the other Components of the Data Set. This
        operation corresponds to a projection in the usual relational join semantics
        (specifying which columns will be projected out).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator takes as input a Data Set (op) and some Component names
        of such a Data Set (comp). These Components can be Measures or Attributes
        of op but not Identifiers. The operator drops the specified Components
        and maintains all the other Components of the Data Set and check their
        attributes.
        """
        code = '13-4-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        REMOVAL OF COMPONENTS: DROP
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ drop At_1, Me_1 ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and some Component
        names of such a Data Set (comp). These Components can be Measures or
        Attributes of op but not Identifiers. The operator drops the specified
        Components and maintains all the other Components of the Data Set. This
        operation corresponds to a projection in the usual relational join semantics
        (specifying which columns will be projected out).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator takes as input a Data Set (op) and some Component names
        of such a Data Set (comp). These Components can be Measures or Attributes
        of op but not Identifiers. The operator drops the specified Components
        and maintains all the other Components of the Data Set and check their
        attributes.
        """
        code = '13-4-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_13(self):
        """
        REMOVAL OF COMPONENTS: DROP
        Is a BUG because semantic drops the attribute and has to keep it.
        Dataset --> Dataset
        Status: OK
        ExpressionOK: DS_r := DS_1 [ drop Me_1, Me_2  ]
                            DS_1 Dataset

        Description: The operator takes as input a Data Set (op) and some Component
        names of such a Data Set (comp). These Components can be Measures or
        Attributes of op but not Identifiers. The operator drops the specified
        Components and maintains all the other Components of the Data Set. This
        operation corresponds to a projection in the usual relational join semantics
        (specifying which columns will be projected out).

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator takes as input a Data Set (op) and some Component names
        of such a Data Set (comp). These Components can be Measures or Attributes
        of op but not Identifiers. The operator drops the specified Components
        and maintains all the other Components of the Data Set and check their
        attributes.
        """
        code = '13-4-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        CHANGE OF COMPONENT NAME: RENAME
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ rename Me_1 to Me_2, At_1 to At_2 ]
                            DS_1 Dataset

        Description: The operator assigns new names to one or more Components
        (Identifier, Measure or Attribute Components). The resulting Data Set,
        after renaming the specified Components, must have unique names of all
        its Components (otherwise a runtime error is raised). Only the Component
        name is changed and not the Component Values, therefore the new Component
        must be defined on the same Value Domain and Value Domain Subset as the
        original Component

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator assigns new names to one or more Components
        (Identifier, Measure or Attribute Components). The resulting Data Set,
        after renaming the specified Components, must have unique names of all
        its Components and check their attributes.
        """
        code = '13-4-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        CHANGE OF COMPONENT NAME: RENAME
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ rename Me_1 to Me_2 ]
                            DS_1 Dataset

        Description: The operator assigns new names to one or more Components
        (Identifier, Measure or Attribute Components). The resulting Data Set,
        after renaming the specified Components, must have unique names of all
        its Components (otherwise a runtime error is raised). Only the Component
        name is changed and not the Component Values, therefore the new Component
        must be defined on the same Value Domain and Value Domain Subset as the
        original Component

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator assigns new names to one or more Components
        (Identifier, Measure or Attribute Components). The resulting Data Set,
        after renaming the specified Components, must have unique names of all
        its Components and check their attributes.
        """
        code = '13-4-1-15'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        CHANGE OF COMPONENT NAME: RENAME
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ rename At_1 to At_3 ]
                            DS_1 Dataset

        Description: The operator assigns new names to one or more Components
        (Identifier, Measure or Attribute Components). The resulting Data Set,
        after renaming the specified Components, must have unique names of all
        its Components (otherwise a runtime error is raised). Only the Component
        name is changed and not the Component Values, therefore the new Component
        must be defined on the same Value Domain and Value Domain Subset as the
        original Component

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator assigns new names to one or more Components
        (Identifier, Measure or Attribute Components). The resulting Data Set,
        after renaming the specified Components, must have unique names of all
        its Components and check their attributes.
        """
        code = '13-4-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    # TODO: Uncomment when Pivot is redefined
    # def test_17(self):
    #     """
    #     PIVOTING: PIVOT
    #     Dataset --> Dataset
    #     Status: OK
    #     Expression: DS_r := DS_1 [ pivot Id_2, Me_1 ]
    #                         DS_1 Dataset
    #
    #     Description: The operator transposes several Data Points of the operand
    #     Data Set into a single Data Point of the resulting Data Set.
    #
    #     Git Branch: #136 Clause operators attributes tests.
    #     Goal: The operator transposes several Data Points of the operand
    #     Data Set into a single Data Point of the resulting Data Set and check
    #     their attributes.
    #     """
    #     code = '13-4-1-17'
    #     number_inputs = 1
    #     references_names = ["1"]
    #
    #     self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        UNPIVOTING: UNPIVOT
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_r := DS_1 [ unpivot Id_2, Me_1]
                            DS_1 Dataset

        Description: The unpivot operator transposes a single Data Point of the
        operand Data Set into several Data Points of the result Data set.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The unpivot operator transposes a single Data Point of the
        operand Data Set into several Data Points of the result Data set and
        check their attributes.
        """
        code = '13-4-1-18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        SUBSPACE: SUB
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ sub Id_1 = 1, Id_2 = "A" ]
                            DS_1 Dataset

        Description: The operator returns a Data Set in a subspace of the one of
        the input Dataset.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator returns a Data Set in a subspace of the one of
        the input Dataset and check their attributes.
        """
        code = '13-4-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        SUBSPACE: SUB
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ sub Id_2 = "A" ]
                            DS_1 Dataset

        Description: The operator returns a Data Set in a subspace of the one of
        the input Dataset.

        Git Branch: #136 Clause operators attributes tests.
        Goal: The operator returns a Data Set in a subspace of the one of
        the input Dataset and check their attributes.
        """
        code = '13-4-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

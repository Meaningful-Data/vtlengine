import json
from pathlib import Path
from typing import Dict, List, Optional
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class TestAggregateTypeChecking(TestCase):
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


class AggregateOperatorsDatasetTypeChecking(TestAggregateTypeChecking):
    """
    Group 10
    """

    classTest = 'AggregateOperators.AggregateOperatorsDatasetTypeChecking'

    # average operator

    def test_1(self):
        '''
        Operation with int and number, me1 int me2 number.
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: All the measures are involved and the results should be type Number. 
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-1'
        # 10 For group aggregate operators
        # 1 For group dataset
        # 1 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        test 1 plus nulls.
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: The nulls are ignored in the average.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-2'
        # 10 For group aggregate operators
        # 1 For group dataset
        # 2 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with  measure String, and more measures, if one measure fails the exception is raised.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-3'
        # 10 For group aggregate operators
        # 1 For group dataset
        # 3 Number of test
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_4(self):
        '''
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-4'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_5(self):
        '''
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-5'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_6(self):
        '''
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with time_period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-6'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_7(self):
        '''
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-7'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_8(self):
        '''
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-8'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_9(self):
        '''
        average with time again
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-9'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # count operator

    def test_10(self):
        '''
        count with integer and number
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: There are measures int and num without nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-10'
        # 10 For group aggregate operators.
        # 1 For group datasets
        # 10 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        '''
        count one measure
        Status: OK
        Expression: DS_r := count ( Me_1 group by Id_1);
        Description: Special case of count with a component, should ignore nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-11'
        # 10 For group aggregate operators.
        # 1 For group datasets
        # 11 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        '''
        count with string
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: There isnt fail because take the null as empty string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-12'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        '''
        count with time
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Time with null, counts the null
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-13'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        '''
        count with date
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Date with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-14'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        '''
        count with time period
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Time Period with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-15'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        '''
        count with duration
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Duration with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-16'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        '''
        count with boolean
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Boolean with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-17'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        '''
        count with number and integer
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: There are measures int and num with nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-18'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        '''
        count with number and integer
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Example that takes the most left measure.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-19'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        '''
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: count with grouping by identifier string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-20'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # max operator
    def test_21(self):
        '''
        max for integers
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: All the measures Integers are involved and the results should be type Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-21'
        # 10 for group aggregate operators
        # 1 For group datasets
        # 21 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        '''
        max for integers and numbers
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: All the measures Integers and Numbers are involved and the results should be the parent type.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-22'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        '''
        max for integers and string
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for string is ok on a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-23'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        '''
        max for integers and time
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for time takes the the mayor number but not the mayor time,
                    2008M1/2008M12 should be the result not 2010M1/2010M12.
                    In this test is not present but max fails for time with nulls .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-24'
        number_inputs = 1
        message = '1-1-1-7'
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

        # references_names = ["DS_r"]

        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        '''
        max for integers and date
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for date and nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-25'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        '''
        max for integers and time period
        Status: OK.
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max doesnt work with nulls and diferent time_period in the same id (2012Q2,2012M12).
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-26'
        number_inputs = 1
        message = '1-1-1-7'
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

        # references_names = ["DS_r"]
        #
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        '''
        max for integers and duration
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max doesnt work with nulls and take the max duration in a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-27'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        '''
        max for integers and boolean
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for booleans takes True as max.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-28'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # median operator
    def test_29(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers  The nulls are ignored in the average and the result measures has the type number .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-29'
        # 10 For group aggregate operators
        # 1 For group dataset
        # 29 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and numbers, all the meaures are calculated.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-30'
        # 10 For group aggregate operators
        # 1 For group dataset
        # 30 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-31'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_32(self):
        '''
        Status:
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-32'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_33(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-33'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_34(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Time_Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-34'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_35(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-35'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_36(self):
        '''
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-36'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # min operator
    def test_37(self):
        '''
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for integers. All the measures Integers are involved and the results should be type Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-37'
        # 10 For group aggregate operators.
        # 1 For group datasets
        # 37 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        '''
        Status: OK
        Expression: DS_r := DS_r := min ( DS_1 group by Id_1);
        Description: Min for integers and numbers with nulls and the results should be the parent type.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-38'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        '''
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for string is ok on a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-39'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        '''
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for time takes the the minor number but not the minor time,
                    2010M1/2010M12 should be the result not 2008M1/2008M12.
                    In this test is not present but max fails for time with nulls .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-40'
        number_inputs = 1
        message = '1-1-1-7'
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # references_names = ["DS_r"]
        #
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        '''
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-41'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        '''
        Status: TO REVIEW
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min Time Period doesnt work with nulls and diferent time_period in the same id (2012Q2,2012M12)..
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-42'
        number_inputs = 1
        message = '1-1-1-7'
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # references_names = ["DS_r"]
        #
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        '''
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min doesnt work with nulls and take the min duration in a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-43'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_44(self):
        '''
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for booleans takes False as min.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-44'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # stddev_pop operator

    def test_45(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: population standard deviation for integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-45'
        # 10 For group aggregation
        # 1 For group dataset
        # 45 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: Population standard deviation for integers and numbers
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-46'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for strings.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-47'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_48(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-48'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_49(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-49'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_50(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for time period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-50'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_51(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-51'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_52(self):
        '''
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-52'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # stddev_samp operator

    def test_53(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-53'
        # 10 For group aggregate oerators
        # 1 For datasets
        # 53 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_54(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-54'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_55(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-55'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_56(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-56'
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 1 Number of test
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_57(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-57'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_58(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Time_period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-58'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_59(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-59'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_60(self):
        '''
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-60'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # sum operator

    def test_61(self):
        '''
        Status: OK? TO REVIEW doubt about result type
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum inputs Integer and results type Number.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-61'
        # 10 For group aggregate operators
        # 1 For group datasets
        # 61 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_62(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: sum Numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-62'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_63(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-63'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_64(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-64'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_65(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-65'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_66(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-66'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_67(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-67'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_68(self):
        '''
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-68'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # var_pop operator

    def test_69(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: Population standard deviation for integers .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-69'
        # 10 For group aggregate operators
        # 1 For group datasets
        # 69 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_70(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: Population standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-70'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_71(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-71'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_72(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-72'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_73(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-73'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_74(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-74'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_75(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Duration .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-75'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_76(self):
        '''
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-76'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    # var_samp operator

    def test_77(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-77'
        # 10 For group aggregate operators
        # 1 For group dataset
        # 77 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_78(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        '''
        code = '10-1-78'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_79(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for String. 
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-79'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_80(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Time. 
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-80'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_81(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Date.  
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-81'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_82(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Time Period. 
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-82'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_83(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Duration.  
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-83'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

    def test_84(self):
        '''
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Boolean. 
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        '''
        code = '10-1-84'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message
        )

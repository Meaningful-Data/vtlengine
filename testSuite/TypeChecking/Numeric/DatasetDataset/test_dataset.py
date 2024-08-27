import json
from pathlib import Path
from typing import Dict, List, Optional
from pathlib import Path

from testSuite.Helper import TestHelper


class TestDatasetNumericTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class DatasetDatasetTypeChecking(TestDatasetNumericTypeChecking):
    """
    Group 4
    """

    classTest = 'DatasetDataset.DatasetDatasetTypeChecking'

    def test_1(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure Integer
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-1'
        # 4 For group numeric
        # 4 For group dataset dataset
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_2(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-2'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_3(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time
                            DS_2 Measure Integer
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-3'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_4(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time
                            DS_2 Measure time
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-4'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure integer
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-5'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_6(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure date
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-6'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure integer
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-7'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_8(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure string
                            DS_2 Measure integer
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-8'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_9(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure integer
        Description: Forbid implicit cast duration to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-9'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_10(self):
        '''
        ADD OPERATOR
        integer + integer --> integer
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure integer
                            DS_2 Measure integer
        Description: Forbid implicit cast integer to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-10'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        '''
        ADD OPERATOR
        integer + number --> number
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure integer
                            DS_2 Measure number
        Description: check implicit cast integer to number in plus operator.
        UPDATED: (Discrepancy between semantic and interpreter, output measure type is Integer, not Number)
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-11'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        '''
        ADD OPERATOR
        number + integer --> number
        Status: OK
        DS_r should be number not integer
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure number
                            DS_2 Measure integer
        Description: check implicit cast integer to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-12'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        '''
        ADD OPERATOR
        number + number --> number
        Status: OK
        DS_r should be number
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure number
                            DS_2 Measure number
        Description: check implicit cast number to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-13'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        '''
        ADD OPERATOR
        number + boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Number
                            DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-14'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_15(self):
        '''
        ADD OPERATOR
        number + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Number
                            DS_2 Measure Time
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-15'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_16(self):
        '''
        ADD OPERATOR
        number + date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Number
                            DS_2 Measure date
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-16'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_17(self):
        '''
        ADD OPERATOR
        number + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Number
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-17'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_18(self):
        '''
        ADD OPERATOR
        number + string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Number
                            DS_2 Measure string
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-18'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_19(self):
        '''
        ADD OPERATOR
        number + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Number
                            DS_2 Measure duration
        Description: Forbid implicit cast duration to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-19'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_20(self):
        '''
        ADD OPERATOR
        boolean + number --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure Number
        Description: Forbid implicit cast number to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-20'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_21(self):
        '''
        ADD OPERATOR
        time + number --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Time
                            DS_2 Measure Number
        Description: Forbid implicit cast number to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-21'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_22(self):
        '''
        ADD OPERATOR
        date + number --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure Number
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-22'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_23(self):
        '''
        ADD OPERATOR
        time_period + number --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure Number
        Description: Forbid implicit cast number to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-23'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_24(self):
        '''
        ADD OPERATOR
        string + number --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure String
                            DS_2 Measure Number
        Description: Forbid implicit cast number to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-24'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_25(self):
        '''
        ADD OPERATOR
        duration + number --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure Number
        Description: Forbid implicit cast number to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-25'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_26(self):
        '''
        ADD OPERATOR
        integer + boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Integer
                            DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to integer in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-26'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_27(self):
        '''
        ADD OPERATOR
        time + boolean--> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Time
                            DS_2 Measure boolean
        Description: Forbid implicit cast boolean to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-27'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_28(self):
        '''
        ADD OPERATOR
        date + boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-28'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_29(self):
        '''
        ADD OPERATOR
        time_period + boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-29'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_30(self):
        '''
        ADD OPERATOR
        string + boolean --> number
        Status: Improvement
        Expression: DS_r := DS_1 + DS_2 ;
                           DS_1 Measure String
                           DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-30'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_31(self):
        '''
        ADD OPERATOR
        duration + boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure Boolean
        Description: Forbid implicit cast boolean to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-31'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_32(self):
        '''
        ADD OPERATOR
        integer + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Integer
                            DS_2 Measure Time
        Description: Forbid implicit cast integer to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-32'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_33(self):
        '''
        ADD OPERATOR
        boolean + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure Time
        Description: Forbid implicit cast time to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-33'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_34(self):
        '''
        ADD OPERATOR
        date + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure time
        Description: Forbid implicit cast time to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-34'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_35(self):
        '''
        ADD OPERATOR
        time_period + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure Time
        Description: Forbid implicit cast time to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-35'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_36(self):
        '''
        ADD OPERATOR
        string + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                           DS_1 Measure String
                           DS_2 Measure Time
        Description: Forbid implicit cast time to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-36'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_37(self):
        '''
        ADD OPERATOR
        duration + time --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure Time
        Description: Forbid implicit cast time to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-37'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_38(self):
        '''
        ADD OPERATOR
        integer + date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Integer
                            DS_2 Measure Date
        Description: Forbid implicit cast date to integer in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-38'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_39(self):
        '''
        ADD OPERATOR
        boolean + date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure Date
        Description: Forbid implicit cast date to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-39'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_40(self):
        '''
        ADD OPERATOR
        time + date--> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Time
                            DS_2 Measure Date
        Description: Forbid implicit cast date to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-40'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_41(self):
        '''
        ADD OPERATOR
        time_period + date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure Date
        Description: Forbid implicit cast date to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-41'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_42(self):
        '''
        ADD OPERATOR
        string + date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                           DS_1 Measure String
                           DS_2 Measure Date
        Description: Forbid implicit cast date to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-42'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_43(self):
        '''
        ADD OPERATOR
        duration + date --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure Date
        Description: Forbid implicit cast date to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-43'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_44(self):
        '''
        ADD OPERATOR
        integer + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Integer
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period to integer in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-44'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_45(self):
        '''
        ADD OPERATOR
        boolean + time_period  --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period  to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-45'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_46(self):
        '''
        ADD OPERATOR
        time + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Time
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period  to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-46'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_47(self):
        '''
        ADD OPERATOR
        date + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-47'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_48(self):
        '''
        ADD OPERATOR
        time_period + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-48'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_49(self):
        '''
        ADD OPERATOR
        string + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                           DS_1 Measure String
                           DS_2 Measure time_period
        Description: Forbid implicit cast time_period to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-49'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_50(self):
        '''
        ADD OPERATOR
        duration + time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure time_period
        Description: Forbid implicit cast time_period to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-50'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_51(self):
        '''
        ADD OPERATOR
        integer + string --> number
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Integer
                            DS_2 Measure String
        Description: Forbid implicit cast string to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-51'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_52(self):
        '''
        ADD OPERATOR
        boolean + string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure String
        Description: Forbid implicit cast string to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-52'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_53(self):
        '''
        ADD OPERATOR
        time + string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time
                            DS_2 Measure String
        Description: Forbid implicit cast string to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-53'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_54(self):
        '''
        ADD OPERATOR
        date + string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure String
        Description: Forbid implicit cast string to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-54'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_55(self):
        '''
        ADD OPERATOR
        time_period + string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure String
        Description: Forbid implicit cast string to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-55'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_56(self):
        '''
        ADD OPERATOR
        string + string --> number
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
                           DS_1 Measure String
                           DS_2 Measure String
        Description: Forbid implicit cast string to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-56'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_57(self):
        '''
        ADD OPERATOR
        duration + string --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure String
        Description: Forbid implicit cast string to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-57'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_58(self):
        '''
        ADD OPERATOR
        integer + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Integer
                            DS_2 Measure Duration
        Description: Forbid implicit cast duration to integer in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-58'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_59(self):
        '''
        ADD OPERATOR
        boolean + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure Boolean
                            DS_2 Measure Duration
        Description: Forbid implicit cast duration to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-4-3-59'

        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_60(self):
        '''
        ADD OPERATOR
        time + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time
                            DS_2 Measure Duration
        Description: Forbid implicit cast duration to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-60'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_61(self):
        '''
        ADD OPERATOR
        date + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure date
                            DS_2 Measure Duration
        Description: Forbid implicit cast duration to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-54'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_62(self):
        '''
        ADD OPERATOR
        time_period + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure time_period
                            DS_2 Measure Duration
        Description: Forbid implicit cast duration to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-62'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_63(self):
        '''
        ADD OPERATOR
        string + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                           DS_1 Measure String
                           DS_2 Measure Duration
        Description: Forbid implicit cast duration to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-63'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_64(self):
        '''
        ADD OPERATOR
        duration + duration --> number
        Status: OK
        Expression: DS_r := DS_1 + DS_2 ;
                            DS_1 Measure duration
                            DS_2 Measure Duration
        Description: Forbid implicit cast duration to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-4-3-64'
        number_inputs = 2
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
from pathlib import Path

from testSuite.Helper import TestHelper


class TestScalarDatasetTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class ScalarDatasetTypeChecking(TestScalarDatasetTypeChecking):
    """
    Group 4
    """

    classTest = 'ScalarDataset.ScalarDatasetTypeChecking'

    def test_1(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-1'
        # 4 For group numeric
        # 3 For group scalar dataset
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_2(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-2'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_3(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-3'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_4(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-4'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-5'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_6(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-6'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        '''
        ADD OPERATOR
        integer + integer --> integer
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Check implicit cast integer to integer in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-3-3-7'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        '''
        ADD OPERATOR
        integer + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1 ;
        Description: Check implicit cast integer to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-3-3-8'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        '''
        ADD OPERATOR
        integer + boolean --> number
        Status: OK
        Expression: DS_r := 1 + DS_1 ;
        Description: Forbid implicit cast integer to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-9'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_10(self):
        '''
        ADD OPERATOR
        integer + time --> number
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-10'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_11(self):
        '''
        ADD OPERATOR
        integer + date --> number
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-11'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_12(self):
        '''
        ADD OPERATOR
        integer + time_period --> number
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-12'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_13(self):
        '''
        ADD OPERATOR
        integer + string --> number
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-13'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_14(self):
        '''
        ADD OPERATOR
        integer + duration --> number
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-14'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_15(self):
        '''
        ADD OPERATOR
        integer + integer --> integer
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to integer in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-15'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        '''
        ADD OPERATOR
        integer + number --> number
        Status: OK
        Expression: DS_r := 1 + DS_1;
        Description: Forbid implicit cast integer to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-16'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        '''
        ADD OPERATOR
        number + boolean --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to boolean in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-17'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_18(self):
        '''
        ADD OPERATOR
        number + time --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to time in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-18'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_19(self):
        '''
        ADD OPERATOR
        numebr + date --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to date in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-19'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_20(self):
        '''
        ADD OPERATOR
        number + time_period --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to time_period in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-20'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_21(self):
        '''
        ADD OPERATOR
        number + string --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to string in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-21'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_22(self):
        '''
        ADD OPERATOR
        number + duration --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to duration in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-22'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_23(self):
        '''
        ADD OPERATOR
        number + integer --> integer
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to integer in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-23'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        '''
        ADD OPERATOR
        number + number --> number
        Status: OK
        Expression: DS_r := 1.0 + DS_1 ;
        Description: Forbid implicit cast number to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-24'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        '''
        ADD OPERATOR
        boolean + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0;
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-25'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_26(self):
        '''
        ADD OPERATOR
        time + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0 ;
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-26'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_27(self):
        '''
        ADD OPERATOR
        date + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0 ;
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-27'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_28(self):
        '''
        ADD OPERATOR
        time_period + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0 ;
        Description: Forbid implicit cast time_period to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-28'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_29(self):
        '''
        ADD OPERATOR
        string + number--> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0 ;
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-29'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_30(self):
        '''
        ADD OPERATOR
        duration + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0 ;
        Description: Forbid implicit cast duration to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-30'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_31(self):
        '''
        ADD OPERATOR
        integer + number --> Number
        Status: OK
        Expression: DS_r := DS_1 + 1.0;
        Description: Forbid implicit cast integer to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-31'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        '''
        ADD OPERATOR
        number + number --> number
        Status: OK
        Expression: DS_r := DS_1 + 1.0 ;
        Description: Forbid implicit cast number to number in plus operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-32'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_33(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast boolean to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-33'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_34(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast time to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-34'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_35(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast date to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-35'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_36(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast time_period to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-36'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_37(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast string to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-37'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_38(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast duration to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-38'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_39(self):
        '''
        ADD OPERATOR
        integer * integer --> integer
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast integer to integer in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-39'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        '''
        ADD OPERATOR
        integer * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1 ;
        Description: Forbid implicit cast integer to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-40'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        '''
        ADD OPERATOR
        integer * boolean --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to boolean in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-41'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_42(self):
        '''
        ADD OPERATOR
        integer * time --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to time in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-42'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_43(self):
        '''
        ADD OPERATOR
        integer * date --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to date in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-43'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_44(self):
        '''
        ADD OPERATOR
        integer * time_period --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to time_period in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-44'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_45(self):
        '''
        ADD OPERATOR
        integer * string --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to string in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-45'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_46(self):
        '''
        ADD OPERATOR
        integer * duration --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to duration in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-46'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_47(self):
        '''
        ADD OPERATOR
        integer * integer --> integer
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to integer in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-47'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        '''
        ADD OPERATOR
        integer * number --> number
        Status: OK
        Expression: DS_r := 1 * DS_1 ;
        Description: Forbid implicit cast integer to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-48'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_49(self):
        '''
        ADD OPERATOR
        number * boolean --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to boolean in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-49'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_50(self):
        '''
        ADD OPERATOR
        number * time --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to time in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-50'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_51(self):
        '''
        ADD OPERATOR
        numebr * date --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to date in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-51'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_52(self):
        '''
        ADD OPERATOR
        number * time_period --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to time_period in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-52'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_53(self):
        '''
        ADD OPERATOR
        number * string --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to string in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-53'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_54(self):
        '''
        ADD OPERATOR
        number * duration --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to duration in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-54'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_55(self):
        '''
        ADD OPERATOR
        number * integer --> Integer
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to integer in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-55'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_56(self):
        '''
        ADD OPERATOR
        number * number --> number
        Status: OK
        Expression: DS_r := 1.0 * DS_1 ;
        Description: Forbid implicit cast number to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-56'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_57(self):
        '''
        ADD OPERATOR
        boolean * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast boolean to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-57'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_58(self):
        '''
        ADD OPERATOR
        time * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast time to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-58'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_59(self):
        '''
        ADD OPERATOR
        date * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast date to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-59'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_60(self):
        '''
        ADD OPERATOR
        time_period * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast time_period to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-60'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_61(self):
        '''
        ADD OPERATOR
        string * number--> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast string to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-61'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_62(self):
        '''
        ADD OPERATOR
        duration * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast duration to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-62'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_63(self):
        '''
        ADD OPERATOR
        integer * number --> Number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast integer to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-63'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_64(self):
        '''
        ADD OPERATOR
        number * number --> number
        Status: OK
        Expression: DS_r := DS_1 * 1.0;
        Description: Forbid implicit cast number to number in multiplication operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-64'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_65(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast boolean to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-65'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_66(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast time to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-66'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_67(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast date to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-67'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_68(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast time_period to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-68'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_69(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast string to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-69'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_70(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast duration to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-70'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_71(self):
        '''
        ADD OPERATOR
        integer / integer --> integer
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast integer to integer in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-71'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_72(self):
        '''
        ADD OPERATOR
        integer / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1 ;
        Description: Forbid implicit cast integer to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-72'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_73(self):
        '''
        ADD OPERATOR
        integer / boolean --> number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to boolean in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-73'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_74(self):
        '''
        ADD OPERATOR
        integer / time --> number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to time in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-74'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_75(self):
        '''
        ADD OPERATOR
        integer / date --> number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to date in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-75'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_76(self):
        '''
        ADD OPERATOR
        integer / time_period --> number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to time_period in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-76'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_77(self):
        '''
        ADD OPERATOR
        integer / string --> not allower
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to string in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-77'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_78(self):
        '''
        ADD OPERATOR
        integer / duration --> number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to duration in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-78'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_79(self):
        '''
        ADD OPERATOR
        integer / integer --> Number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: Forbid implicit cast integer to integer in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-79'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_80(self):
        '''
        ADD OPERATOR
        integer / number --> number
        Status: OK
        Expression: DS_r := 1 / DS_1 ;
        Description: .
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-80'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_81(self):
        '''
        ADD OPERATOR
        number / boolean --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: Forbid implicit cast number to boolean in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-81'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_82(self):
        '''
        ADD OPERATOR
        number / time --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: Forbid implicit cast number to time in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-82'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_83(self):
        '''
        ADD OPERATOR
        numebr / date --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: Forbid implicit cast number to date in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-83'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_84(self):
        '''
        ADD OPERATOR
        number / time_period --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: Forbid implicit cast number to time_period in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-84'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_85(self):
        '''
        ADD OPERATOR
        number / string --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: Forbid implicit cast number to string in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-85'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_86(self):
        '''
        ADD OPERATOR
        number / duration --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: Forbid implicit cast number to duration in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-86'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_87(self):
        '''
        ADD OPERATOR
        number / integer --> Number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: .
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-87'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_88(self):
        '''
        ADD OPERATOR
        number / number --> number
        Status: OK
        Expression: DS_r := 1.0 / DS_1 ;
        Description: .
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-88'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_89(self):
        '''
        ADD OPERATOR
        boolean / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast boolean to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-89'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_90(self):
        '''
        ADD OPERATOR
        time / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast time to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-90'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_91(self):
        '''
        ADD OPERATOR
        date / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast date to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-91'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_92(self):
        '''
        ADD OPERATOR
        time_period / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast time_period to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-92'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_93(self):
        '''
        ADD OPERATOR
        string / number--> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast string to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-93'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_94(self):
        '''
        ADD OPERATOR
        duration / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast duration to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-94'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_95(self):
        '''
        ADD OPERATOR
        integer / number --> Number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast integer to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-95'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_96(self):
        '''
        ADD OPERATOR
        number / number --> number
        Status: OK
        Expression: DS_r := DS_1 / 1.0;
        Description: Forbid implicit cast number to number in division operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-96'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_97(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast boolean to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-97'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_98(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast time to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-98'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_99(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast date to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-99'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_100(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast time_period to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-100'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_101(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast string to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-101'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_102(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast duration to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-102'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_103(self):
        '''
        ADD OPERATOR
        integer  mod integer --> integer
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast integer to integer in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-103'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_104(self):
        '''
        ADD OPERATOR
        integer mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1) ;
        Description: Forbid implicit cast integer to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-104'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_105(self):
        '''
        ADD OPERATOR
        integer mod boolean --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to boolean in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-105'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_106(self):
        '''
        ADD OPERATOR
        integer mod time --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to time in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-106'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_107(self):
        '''
        ADD OPERATOR
        integer mod date --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to date in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-107'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_108(self):
        '''
        ADD OPERATOR
        integer mod time_period --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to time_period in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-108'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_109(self):
        '''
        ADD OPERATOR
        integer mod string --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to string in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-109'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_110(self):
        '''
        ADD OPERATOR
        integer mod duration --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to duration in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-110'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_111(self):
        '''
        ADD OPERATOR
        integer mod integer --> integer
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: check because the results are wrong.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-111'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_112(self):
        '''
        ADD OPERATOR
        integer mod number --> number
        Status: OK
        Expression: DS_r := mod(1 , DS_1) ;
        Description: Forbid implicit cast integer to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-112'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_113(self):
        '''
        ADD OPERATOR
        number mod boolean --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to boolean in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-113'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_114(self):
        '''
        ADD OPERATOR
        number mod time --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-114'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_115(self):
        '''
        ADD OPERATOR
        numebr mod date --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to date in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-115'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_116(self):
        '''
        ADD OPERATOR
        number mod time_period --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time_period in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-116'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_117(self):
        '''
        ADD OPERATOR
        number mod string --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to string in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-117'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_118(self):
        '''
        ADD OPERATOR
        number mod duration --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to duration in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-118'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_119(self):
        '''
        ADD OPERATOR
        number mod integer --> Number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to integer in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-119'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_120(self):
        '''
        ADD OPERATOR
        number mod number --> number
        Status: OK
        Expression: DS_r := mod(1.0 , DS_1) ;
        Description: Forbid implicit cast number to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-120'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_121(self):
        '''
        ADD OPERATOR
        boolean mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast boolean to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-121'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_122(self):
        '''
        ADD OPERATOR
        time mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast time to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-122'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_123(self):
        '''
        ADD OPERATOR
        date mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast date to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-123'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_124(self):
        '''
        ADD OPERATOR
        time_period mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast time_period to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-124'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_125(self):
        '''
        ADD OPERATOR
        string mod number--> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast string to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-125'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_126(self):
        '''
        ADD OPERATOR
        duration mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast duration to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-126'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_127(self):
        '''
        ADD OPERATOR
        integer mod number --> Number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast integer to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-127'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_128(self):
        '''
        ADD OPERATOR
        number mod number --> number
        Status: OK
        Expression: DS_r := mod(DS_1 , 1.0) ;
        Description: Forbid implicit cast number to number in modulo operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-128'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_129(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast boolean to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-129'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_130(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast time to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-130'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_131(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast date to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-131'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_132(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast time_period to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-132'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_133(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast string to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-133'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_134(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast duration to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-134'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_135(self):
        '''
        ADD OPERATOR
        integer  round integer --> Number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast integer to integer in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-135'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_136(self):
        '''
        ADD OPERATOR
        integer round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1) ;
        Description: Forbid implicit cast integer to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-136'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_137(self):
        '''
        ADD OPERATOR
        integer round boolean --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to boolean in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-137'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_138(self):
        '''
        ADD OPERATOR
        integer round time --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to time in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-138'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_139(self):
        '''
        ADD OPERATOR
        integer round date --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to date in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-139'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_140(self):
        '''
        ADD OPERATOR
        integer round time_period --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to time_period in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-140'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_141(self):
        '''
        ADD OPERATOR
        integer round string --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to string in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-141'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_142(self):
        '''
        ADD OPERATOR
        integer round duration --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to duration in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-142'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_143(self):
        '''
        ADD OPERATOR
        integer round integer --> integer
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to integer in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-143'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_144(self):
        '''
        ADD OPERATOR
        integer round number --> number
        Status: OK
        Expression: DS_r := round(1 , DS_1) ;
        Description: Forbid implicit cast integer to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-144'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_145(self):
        '''
        ADD OPERATOR
        number round boolean --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to boolean in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-145'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_146(self):
        '''
        ADD OPERATOR
        number round time --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-146'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_147(self):
        '''
        ADD OPERATOR
        numebr round date --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to date in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-147'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_148(self):
        '''
        ADD OPERATOR
        number round time_period --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time_period in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-148'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_149(self):
        '''
        ADD OPERATOR
        number round string --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to string in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-149'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_150(self):
        '''
        ADD OPERATOR
        number round duration --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to duration in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-150'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_151(self):
        '''
        ADD OPERATOR
        number round integer --> integer
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to integer in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-151'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_152(self):
        '''
        ADD OPERATOR
        number round number --> number
        Status: OK
        Expression: DS_r := round(1.0 , DS_1) ;
        Description: Forbid implicit cast number to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-152'
        number_inputs = 1
        message = '1-1-1-14'

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_153(self):
        '''
        ADD OPERATOR
        boolean round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast boolean to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-153'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_154(self):
        '''
        ADD OPERATOR
        time round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast time to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-154'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_155(self):
        '''
        ADD OPERATOR
        date round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast date to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-155'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_156(self):
        '''
        ADD OPERATOR
        time_period round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast time_period to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-156'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_157(self):
        '''
        ADD OPERATOR
        string round number--> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast string to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-157'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_158(self):
        '''
        ADD OPERATOR
        duration round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast duration to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-158'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_159(self):
        '''
        ADD OPERATOR
        integer round number --> integer
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast integer to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-159'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_160(self):
        '''
        ADD OPERATOR
        number round number --> number
        Status: OK
        Expression: DS_r := round(DS_1 , 1.0) ;
        Description: Forbid implicit cast number to number in round operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-160'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_161(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast boolean to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-161'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_162(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast time to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-162'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_163(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast date to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-163'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_164(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast time_period to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-164'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_165(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast string to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-165'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_166(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast duration to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-166'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_167(self):
        '''
        ADD OPERATOR
        integer  truncation integer --> Number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast integer to integer in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-167'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_168(self):
        '''
        ADD OPERATOR
        integer truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1) ;
        Description: Forbid implicit cast integer to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-168'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_169(self):
        '''
        ADD OPERATOR
        integer truncation boolean --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to boolean in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-169'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_170(self):
        '''
        ADD OPERATOR
        integer truncation time --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to time in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-170'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_171(self):
        '''
        ADD OPERATOR
        integer truncation date --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to date in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-171'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_172(self):
        '''
        ADD OPERATOR
        integer truncation time_period --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to time_period in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-172'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_173(self):
        '''
        ADD OPERATOR
        integer truncation string --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to string in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-174'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_174(self):
        '''
        ADD OPERATOR
        integer truncation duration --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to duration in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-174'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_175(self):
        '''
        ADD OPERATOR
        integer truncation integer --> integer
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to integer in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-175'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_176(self):
        '''
        ADD OPERATOR
        integer truncation number --> number
        Status: OK
        Expression: DS_r := trunc(1 , DS_1) ;
        Description: Forbid implicit cast integer to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-176'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_177(self):
        '''
        ADD OPERATOR
        number truncation boolean --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to boolean in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-177'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_178(self):
        '''
        ADD OPERATOR
        number truncation time --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-178'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_179(self):
        '''
        ADD OPERATOR
        numebr truncation date --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to date in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-179'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_180(self):
        '''
        ADD OPERATOR
        number truncation time_period --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time_period in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-180'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_181(self):
        '''
        ADD OPERATOR
        number truncation string --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to string in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-181'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_182(self):
        '''
        ADD OPERATOR
        number truncation duration --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to duration in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-182'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_183(self):
        '''
        ADD OPERATOR
        number truncation integer --> integer
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to integer in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-183'
        number_inputs = 1

        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_184(self):
        '''
        ADD OPERATOR
        number truncation number --> number
        Status: OK
        Expression: DS_r := trunc(1.0 , DS_1) ;
        Description: Forbid implicit cast number to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-184'
        number_inputs = 1
        message = "1-1-1-14"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_185(self):
        '''
        ADD OPERATOR
        boolean truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast boolean to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-185'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_186(self):
        '''
        ADD OPERATOR
        time truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast time to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-186'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_187(self):
        '''
        ADD OPERATOR
        date truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast date to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-187'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_188(self):
        '''
        ADD OPERATOR
        time_period truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast time_period to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-188'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_189(self):
        '''
        ADD OPERATOR
        string truncation number--> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast string to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-189'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_190(self):
        '''
        ADD OPERATOR
        duration truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast duration to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-190'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_191(self):
        '''
        ADD OPERATOR
        integer truncation number --> integer
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast integer to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-191'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_192(self):
        '''
        ADD OPERATOR
        number truncation number --> number
        Status: OK
        Expression: DS_r := trunc(DS_1 , 1.0) ;
        Description: Forbid implicit cast number to number in truncation operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-192'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_193(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast boolean to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-193'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_194(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast time to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-194'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_195(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast date to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-195'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_196(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast time_period to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-196'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_197(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast string to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-197'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_198(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast duration to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-198'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_199(self):
        '''
        ADD OPERATOR
        integer  ceiling integer --> integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to integer in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-199'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_200(self):
        '''
        ADD OPERATOR
        integer ceiling number --> Integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-200'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_201(self):
        '''
        ADD OPERATOR
        integer ceiling boolean --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to boolean in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-201'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_202(self):
        '''
        ADD OPERATOR
        integer ceiling time --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to time in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-202'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_203(self):
        '''
        ADD OPERATOR
        integer ceiling date --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to date in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-203'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_204(self):
        '''
        ADD OPERATOR
        integer ceiling time_period --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to time_period in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-204'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_205(self):
        '''
        ADD OPERATOR
        integer ceiling string --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to string in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-205'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_206(self):
        '''
        ADD OPERATOR
        integer ceiling duration --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to duration in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-206'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_207(self):
        '''
        ADD OPERATOR
        integer ceiling integer --> integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to integer in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-207'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_208(self):
        '''
        ADD OPERATOR
        integer ceiling number --> Integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-208'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_209(self):
        '''
        ADD OPERATOR
        number ceiling boolean --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to boolean in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-209'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_210(self):
        '''
        ADD OPERATOR
        number ceiling time --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to time in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-210'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_211(self):
        '''
        ADD OPERATOR
        numebr ceiling date --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to date in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-211'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_212(self):
        '''
        ADD OPERATOR
        number ceiling time_period --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to time_period in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-212'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_213(self):
        '''
        ADD OPERATOR
        number ceiling string --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to string in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-213'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_214(self):
        '''
        ADD OPERATOR
        number ceiling duration --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to duration in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-214'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_215(self):
        '''
        ADD OPERATOR
        number ceiling integer --> integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to integer in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-215'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_216(self):
        '''
        ADD OPERATOR
        number ceiling number --> Integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-216'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_217(self):
        '''
        ADD OPERATOR
        boolean ceiling number --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast boolean to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-217'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_218(self):
        '''
        ADD OPERATOR
        time ceiling number --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast time to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-218'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_219(self):
        '''
        ADD OPERATOR
        date ceiling number --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast date to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-219'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_220(self):
        '''
        ADD OPERATOR
        time_period ceiling number --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast time_period to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-220'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_221(self):
        '''
        ADD OPERATOR
        string ceiling number--> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast string to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-221'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_222(self):
        '''
        ADD OPERATOR
        duration ceiling number --> number
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast duration to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-222'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_223(self):
        '''
        ADD OPERATOR
        integer ceiling number --> integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast integer to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-223'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_224(self):
        '''
        ADD OPERATOR
        number ceiling number --> Integer
        Status: OK
        Expression: DS_r := ceil(DS_1) ;
        Description: Forbid implicit cast number to number in ceiling operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-224'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_225(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast boolean to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-225'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_226(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast time to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-226'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_227(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast date to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-227'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_228(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast time_period to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-228'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_229(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast string to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-229'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_230(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast duration to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-230'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_231(self):
        '''
        ADD OPERATOR
        integer  floor integer --> integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to integer in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-231'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_232(self):
        '''
        ADD OPERATOR
        integer floor number --> Integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-232'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_233(self):
        '''
        ADD OPERATOR
        integer floor boolean --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to boolean in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-233'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_234(self):
        '''
        ADD OPERATOR
        integer floor time --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to time in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-234'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_235(self):
        '''
        ADD OPERATOR
        integer floor date --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to date in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-235'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_236(self):
        '''
        ADD OPERATOR
        integer floor time_period --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to time_period in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-236'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_237(self):
        '''
        ADD OPERATOR
        integer floor string --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to string in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-237'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_238(self):
        '''
        ADD OPERATOR
        integer floor duration --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to duration in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-238'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_239(self):
        '''
        ADD OPERATOR
        integer floor integer --> integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to integer in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-239'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_240(self):
        '''
        ADD OPERATOR
        integer floor number --> Integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-240'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_241(self):
        '''
        ADD OPERATOR
        number floor boolean --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to boolean in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-241'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_242(self):
        '''
        ADD OPERATOR
        number floor time --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to time in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-242'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_243(self):
        '''
        ADD OPERATOR
        numebr floor date --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to date in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-243'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_244(self):
        '''
        ADD OPERATOR
        number floor time_period --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to time_period in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-244'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_245(self):
        '''
        ADD OPERATOR
        number floor string --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to string in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-245'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_246(self):
        '''
        ADD OPERATOR
        number floor duration --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to duration in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-246'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_247(self):
        '''
        ADD OPERATOR
        number floor integer --> integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to integer in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-247'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_248(self):
        '''
        ADD OPERATOR
        number floor number --> Integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-248'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_249(self):
        '''
        ADD OPERATOR
        boolean floor number --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast boolean to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-249'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_250(self):
        '''
        ADD OPERATOR
        time floor number --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast time to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-250'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_251(self):
        '''
        ADD OPERATOR
        date floor number --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast date to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-251'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_252(self):
        '''
        ADD OPERATOR
        time_period floor number --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast time_period to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-252'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_253(self):
        '''
        ADD OPERATOR
        string floor number--> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast string to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-253'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_254(self):
        '''
        ADD OPERATOR
        duration floor number --> number
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast duration to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-254'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_255(self):
        '''
        ADD OPERATOR
        integer floor number --> integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast integer to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-255'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_256(self):
        '''
        ADD OPERATOR
        number floor number --> Integer
        Status: OK
        Expression: DS_r := floor(DS_1) ;
        Description: Forbid implicit cast number to number in floor operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-256'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_257(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast boolean to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-257'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_258(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast time to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-258'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_259(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast date to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-259'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_260(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast time_period to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-260'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_261(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast string to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-261'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_262(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast duration to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-262'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_263(self):
        '''
        ADD OPERATOR
        integer  absolute value integer --> integer
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to integer in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-263'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_264(self):
        '''
        ADD OPERATOR
        integer absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-264'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_265(self):
        '''
        ADD OPERATOR
        integer absolute value boolean --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to boolean in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-265'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_266(self):
        '''
        ADD OPERATOR
        integer absolute value time --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to time in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-266'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_267(self):
        '''
        ADD OPERATOR
        integer absolute value date --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to date in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-267'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_268(self):
        '''
        ADD OPERATOR
        integer absolute value time_period --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to time_period in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-268'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_269(self):
        '''
        ADD OPERATOR
        integer absolute value string --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to string in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-269'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_270(self):
        '''
        ADD OPERATOR
        integer absolute value duration --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to duration in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-270'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_271(self):
        '''
        ADD OPERATOR
        integer absolute value integer --> integer
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to integer in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-271'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_272(self):
        '''
        ADD OPERATOR
        integer absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-272'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_273(self):
        '''
        ADD OPERATOR
        number absolute value boolean --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to boolean in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-273'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_274(self):
        '''
        ADD OPERATOR
        number absolute value time --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to time in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-274'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_275(self):
        '''
        ADD OPERATOR
        numebr absolute value date --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to date in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-275'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_276(self):
        '''
        ADD OPERATOR
        number absolute value time_period --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to time_period in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-276'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_277(self):
        '''
        ADD OPERATOR
        number absolute value string --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to string in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-277'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_278(self):
        '''
        ADD OPERATOR
        number absolute value duration --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to duration in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-278'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_279(self):
        '''
        ADD OPERATOR
        number absolute value integer --> integer
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to integer in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-279'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_280(self):
        '''
        ADD OPERATOR
        number absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-280'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_281(self):
        '''
        ADD OPERATOR
        boolean absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast boolean to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-281'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_282(self):
        '''
        ADD OPERATOR
        time absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast time to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-282'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_283(self):
        '''
        ADD OPERATOR
        date absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast date to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-283'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_284(self):
        '''
        ADD OPERATOR
        time_period absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast time_period to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-284'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_285(self):
        '''
        ADD OPERATOR
        string absolute value number--> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast string to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-285'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_286(self):
        '''
        ADD OPERATOR
        duration absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast duration to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-286'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_287(self):
        '''
        ADD OPERATOR
        integer absolute value number --> integer
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast integer to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-287'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_288(self):
        '''
        ADD OPERATOR
        number absolute value number --> number
        Status: OK
        Expression: DS_r := abs(DS_1) ;
        Description: Forbid implicit cast number to number in absolute value operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-288'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_289(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast boolean to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-289'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_290(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast time to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-290'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_291(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast date to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-291'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_292(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast time_period to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-292'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_293(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast string to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-293'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_294(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast duration to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-294'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_295(self):
        '''
        ADD OPERATOR
        integer  exponential integer --> integer
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to integer in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-295'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_296(self):
        '''
        ADD OPERATOR
        integer exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-296'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_297(self):
        '''
        ADD OPERATOR
        integer exponential boolean --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to boolean in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-297'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_298(self):
        '''
        ADD OPERATOR
        integer exponential time --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to time in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-298'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_299(self):
        '''
        ADD OPERATOR
        integer exponential date --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to date in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-299'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_300(self):
        '''
        ADD OPERATOR
        integer exponential time_period --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to time_period in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-300'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_301(self):
        '''
        ADD OPERATOR
        integer exponential string --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to string in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-301'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_302(self):
        '''
        ADD OPERATOR
        integer exponential duration --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to duration in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-302'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_303(self):
        '''
        ADD OPERATOR
        integer exponential integer --> Number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to integer in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-303'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_304(self):
        '''
        ADD OPERATOR
        integer exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-304'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_305(self):
        '''
        ADD OPERATOR
        number exponential boolean --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to boolean in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-305'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_306(self):
        '''
        ADD OPERATOR
        number exponential time --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to time in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-306'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_307(self):
        '''
        ADD OPERATOR
        numebr exponential date --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to date in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-307'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_308(self):
        '''
        ADD OPERATOR
        number exponential time_period --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to time_period in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-308'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_309(self):
        '''
        ADD OPERATOR
        number exponential string --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to string in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-309'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_310(self):
        '''
        ADD OPERATOR
        number exponential duration --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to duration in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-310'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_311(self):
        '''
        ADD OPERATOR
        number exponential integer --> Number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to integer in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-311'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_312(self):
        '''
        ADD OPERATOR
        number exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-312'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_313(self):
        '''
        ADD OPERATOR
        boolean exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast boolean to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-313'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_314(self):
        '''
        ADD OPERATOR
        time exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast time to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-314'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_315(self):
        '''
        ADD OPERATOR
        date exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast date to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-315'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_316(self):
        '''
        ADD OPERATOR
        time_period exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast time_period to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-316'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_317(self):
        '''
        ADD OPERATOR
        string exponential number--> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast string to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-317'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_318(self):
        '''
        ADD OPERATOR
        duration exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast duration to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-318'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_319(self):
        '''
        ADD OPERATOR
        integer exponential number --> Number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast integer to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-319'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_320(self):
        '''
        ADD OPERATOR
        number exponential number --> number
        Status: OK
        Expression: DS_r := exp(DS_1) ;
        Description: Forbid implicit cast number to number in exponential operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-320'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_321(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast boolean to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-321'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_322(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast time to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-322'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_323(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast date to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-323'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_324(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast time_period to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-324'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_325(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast string to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-325'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_326(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast duration to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-326'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_327(self):
        '''
        ADD OPERATOR
        integer  natural logarithm integer --> Number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to integer in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-327'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_328(self):
        '''
        ADD OPERATOR
        integer natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Sensibility error in the comparission.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-3-3-328'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_329(self):
        '''
        ADD OPERATOR
        integer natural logarithm boolean --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to boolean in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-329'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_330(self):
        '''
        ADD OPERATOR
        integer natural logarithm time --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to time in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-330'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_331(self):
        '''
        ADD OPERATOR
        integer natural logarithm date --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to date in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-331'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_332(self):
        '''
        ADD OPERATOR
        integer natural logarithm time_period --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to time_period in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-332'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_333(self):
        '''
        ADD OPERATOR
        integer natural logarithm string --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to string in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-333'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_334(self):
        '''
        ADD OPERATOR
        integer natural logarithm duration --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to duration in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-334'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_335(self):
        '''
        ADD OPERATOR
        integer natural logarithm integer --> Number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Sensibility in the comparission results.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Result.
        '''
        code = '4-3-3-335'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_336(self):
        '''
        ADD OPERATOR
        integer natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-336'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_337(self):
        '''
        ADD OPERATOR
        number natural logarithm boolean --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to boolean in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-337'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_338(self):
        '''
        ADD OPERATOR
        number natural logarithm time --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to time in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-338'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_339(self):
        '''
        ADD OPERATOR
        numebr natural logarithm date --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to date in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-339'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_340(self):
        '''
        ADD OPERATOR
        number natural logarithm time_period --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to time_period in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-340'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_341(self):
        '''
        ADD OPERATOR
        number natural logarithm string --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to string in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-341'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_342(self):
        '''
        ADD OPERATOR
        number natural logarithm duration --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to duration in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-342'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_343(self):
        '''
        ADD OPERATOR
        number natural logarithm integer --> Number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to integer in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-343'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_344(self):
        '''
        ADD OPERATOR
        number natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-344'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_345(self):
        '''
        ADD OPERATOR
        boolean natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast boolean to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-345'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_346(self):
        '''
        ADD OPERATOR
        time natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast time to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-346'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_347(self):
        '''
        ADD OPERATOR
        date natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast date to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-347'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_348(self):
        '''
        ADD OPERATOR
        time_period natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast time_period to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-348'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_349(self):
        '''
        ADD OPERATOR
        string natural logarithm number--> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast string to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-349'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_350(self):
        '''
        ADD OPERATOR
        duration natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast duration to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-350'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_351(self):
        '''
        ADD OPERATOR
        integer natural logarithm number --> Number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast integer to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Resault.
        '''
        code = '4-3-3-351'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_352(self):
        '''
        ADD OPERATOR
        number natural logarithm number --> number
        Status: OK
        Expression: DS_r := ln(DS_1) ;
        Description: Forbid implicit cast number to number in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-352'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_353(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast boolean to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-353'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_354(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast time to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-354'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_355(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast date to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-355'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_356(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast time_period to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-356'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_357(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast string to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-357'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_358(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast duration to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-358'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_359(self):
        '''
        ADD OPERATOR
        integer  power integer --> Number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast integer to integer in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-359'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_360(self):
        '''
        ADD OPERATOR
        integer power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1) ;
        Description: Forbid implicit cast integer to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-360'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_361(self):
        '''
        ADD OPERATOR
        integer power boolean --> number
        Status: OK
        Expression: DS_r := power(1 , DS_1) ;
        Description: Forbid implicit cast integer to boolean in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-361'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_362(self):
        '''
        ADD OPERATOR
        integer power time --> number
        Status: OK
        Expression: DS_r := power(1 , DS_1) ;
        Description: Forbid implicit cast integer to time in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-362'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_363(self):
        '''
        ADD OPERATOR
        integer power date --> number
        Status: OK
        Expression: DS_r := power(1 , DS_1) ;
        Description: Forbid implicit cast integer to date in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-363'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_364(self):
        '''
        ADD OPERATOR
        integer power time_period --> number
        Status: OK
        Expression: DS_r := power(1 , DS_1) ;
        Description: Forbid implicit cast integer to time_period in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-364'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_365(self):
        '''
        ADD OPERATOR
        integer power string --> number
        Status: OK
        Expression: DS_r := power(1 , DS_1) ;
        Description: Forbid implicit cast integer to string in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-365'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_366(self):
        '''
        ADD OPERATOR
        integer power duration --> number
        Status: OK
        Expression: DS_r := power(1 , DS_1) ;
        Description: Forbid implicit cast integer to duration in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-366'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    # def test_367(self):
    #     '''
    #     ADD OPERATOR
    #     integer power integer --> integer
    #     Status: OK
    #     Expression: DS_r := power(1 , DS_1) ;
    #     Description: Forbid implicit cast integer to integer in power operator.
    #     Jira issue: VTLEN 562.
    #     Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
    #     Goal: Check Exception.
    #     '''
    #     code = '4-3-3-367'
    #     number_inputs = 1
    #     references_names = ["DS_r"]

    #     
    #     self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # def test_368(self):
    #     '''
    #     ADD OPERATOR
    #     integer power number --> number
    #     Status: OK
    #     Expression: DS_r := power(1 , DS_1) ;
    #     Description: Forbid implicit cast integer to number in power operator.
    #     Jira issue: VTLEN 562.
    #     Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
    #     Goal: Check Exception.
    #     '''
    #     code = '4-3-3-368'
    #     number_inputs = 1
    #     references_names = ["DS_r"]

    #     
    #     self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_369(self):
        '''
        ADD OPERATOR
        number power boolean --> number
        Status: OK
        Expression: DS_r := power(1.0 , DS_1) ;
        Description: Forbid implicit cast number to boolean in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-369'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_370(self):
        '''
        ADD OPERATOR
        number power time --> number
        Status: OK
        Expression: DS_r := power(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-370'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_371(self):
        '''
        ADD OPERATOR
        numebr power date --> number
        Status: OK
        Expression: DS_r := power(1.0 , DS_1) ;
        Description: Forbid implicit cast number to date in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-371'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_372(self):
        '''
        ADD OPERATOR
        number power time_period --> number
        Status: OK
        Expression: DS_r := power(1.0 , DS_1) ;
        Description: Forbid implicit cast number to time_period in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-372'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_373(self):
        '''
        ADD OPERATOR
        number power string --> number
        Status: OK
        Expression: DS_r := power(1.0 , DS_1) ;
        Description: Forbid implicit cast number to string in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-373'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_374(self):
        '''
        number power duration --> number
        Status: OK
        Expression: DS_r := power(1.0 , DS_1) ;
        Description: Forbid implicit cast number to duration in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-374'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    # def test_375(self):
    #     '''
    #     ADD OPERATOR
    #     number power integer --> integer
    #     Status: OK
    #     Expression: DS_r := power(1.0 , DS_1) ;
    #     Description: Forbid implicit cast number to integer in power operator.
    #     Jira issue: VTLEN 562.
    #     Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
    #     Goal: Check Exception.
    #     '''
    #     code = '4-3-3-375'
    #     number_inputs = 1
    #     references_names = ["DS_r"]

    #     
    #     self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # def test_376(self):
    #     '''
    #     ADD OPERATOR
    #     number power number --> number
    #     Status: OK
    #     Expression: DS_r := power(1.0 , DS_1) ;
    #     Description: Forbid implicit cast number to number in power operator.
    #     Jira issue: VTLEN 562.
    #     Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
    #     Goal: Check Exception.
    #     '''
    #     code = '4-3-3-376'
    #     number_inputs = 1
    #     references_names = ["DS_r"]

    #     
    #     self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_377(self):
        '''
        ADD OPERATOR
        boolean power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast boolean to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-377'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_378(self):
        '''
        ADD OPERATOR
        time power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast time to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-378'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_379(self):
        '''
        ADD OPERATOR
        date power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast date to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-379'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_380(self):
        '''
        ADD OPERATOR
        time_period power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast time_period to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-380'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_381(self):
        '''
        ADD OPERATOR
        string power number--> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast string to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-381'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_382(self):
        '''
        ADD OPERATOR
        duration power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast duration to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-382'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_383(self):
        '''
        ADD OPERATOR
        integer power number --> integer
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast integer to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-383'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_384(self):
        '''
        ADD OPERATOR
        number power number --> number
        Status: OK
        Expression: DS_r := power(DS_1 , 1.0) ;
        Description: Forbid implicit cast number to number in power operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-384'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_385(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast boolean to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-385'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_386(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast time to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-386'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_387(self):
        '''
        Log OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast date to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-387'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_388(self):
        '''
        Log OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast time_period to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-388'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_389(self):
        '''
        Log OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast string to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-389'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_390(self):
        '''
        Log OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast duration to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-390'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_391(self):
        '''
        Log OPERATOR
        integer  logarithm integer --> integer
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast integer to integer in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-391'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_392(self):
        '''
        Log OPERATOR
        integer logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2) ;
        Description: Forbid implicit cast integer to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-392'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_393(self):
        '''
        Log OPERATOR
        integer logarithm boolean --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to boolean in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-393'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_394(self):
        '''
        Log OPERATOR
        integer logarithm time --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to time in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-394'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_395(self):
        '''
        Log OPERATOR
        integer logarithm date --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to date in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-395'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_396(self):
        '''
        Log OPERATOR
        integer logarithm time_period --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to time_period in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-396'
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_397(self):
        '''
        Log OPERATOR
        integer logarithm string --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to string in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-397'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_398(self):
        '''
        Log OPERATOR
        integer logarithm duration --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to duration in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-398'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_399(self):
        '''
        WRONG OPERATOR THE SECOND OP cant be a dataset
        integer logarithm integer --> integer
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to integer in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-399'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_400(self):
        '''
        ADD OPERATOR
        integer logarithm number --> number
        Status: OK
        Expression: DS_r := log(2 , DS_1) ;
        Description: Forbid implicit cast integer to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-400'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_401(self):
        '''
        ADD OPERATOR
        number logarithm boolean --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to boolean in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-401'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_402(self):
        '''
        ADD OPERATOR
        number logarithm time --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to time in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-402'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_403(self):
        '''
        ADD OPERATOR
        numebr logarithm date --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to date in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-403'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_404(self):
        '''
        ADD OPERATOR
        number logarithm time_period --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to time_period in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-404'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_405(self):
        '''
        ADD OPERATOR
        number logarithm string --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to string in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-405'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_406(self):
        '''
        ADD OPERATOR
        number logarithm duration --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to duration in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-406'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_407(self):
        '''
        ADD OPERATOR
        number logarithm integer --> integer
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to integer in natural logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-407'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_408(self):
        '''
        ADD OPERATOR
        number logarithm number --> number
        Status: OK
        Expression: DS_r := log(2.0 , DS_1) ;
        Description: Forbid implicit cast number to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-408'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_409(self):
        '''
        ADD OPERATOR
        boolean logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Logarithm test that the second operator is not an integer. Not allow.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-409'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_410(self):
        '''
        ADD OPERATOR
        time logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Logarithm test that the second operator is not an integer.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-410'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_411(self):
        '''
        ADD OPERATOR
        date logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Logarithm test that the second operator is not an integer.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-411'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_412(self):
        '''
        ADD OPERATOR
        time_period logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Logarithm test that the second operator is not an integer.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-412'
        number_inputs = 1

        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_413(self):
        '''
        ADD OPERATOR
        string logarithm number--> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Logarithm test that the second operator is not an integer.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-413'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_414(self):
        '''
        ADD OPERATOR
        duration logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Logarithm test that the second operator is not an integer.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-414'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_415(self):
        '''
        ADD OPERATOR
        integer logarithm number --> integer
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Forbid implicit cast integer to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-415'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_416(self):
        '''
        ADD OPERATOR
        number logarithm number --> number
        Status: OK
        Expression: DS_r := log(DS_1 , 2.0) ;
        Description: Forbid implicit cast number to number in logarithm operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-416'
        number_inputs = 1
        message = "1-1-1-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_417(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast boolean to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-417'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_418(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast time to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-418'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_419(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast date to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-419'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_420(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast time_period to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-420'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_421(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast string to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-421'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_422(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast duration to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-422'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_423(self):
        '''
        ADD OPERATOR
        integer  square root integer --> Number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to integer in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-423'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_424(self):
        '''
        ADD OPERATOR
        integer square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-424'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_425(self):
        '''
        ADD OPERATOR
        integer square root boolean --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to boolean in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-425'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_426(self):
        '''
        ADD OPERATOR
        integer square root time --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to time in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-426'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_427(self):
        '''
        ADD OPERATOR
        integer square root date --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to date in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-427'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_428(self):
        '''
        ADD OPERATOR
        integer square root time_period --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to time_period in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exceptidon.
        '''
        code = '4-3-3-428'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_429(self):
        '''
        ADD OPERATOR
        integer square root string --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to string in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-429'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_430(self):
        '''
        ADD OPERATOR
        integer square root duration --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to duration in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-430'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_431(self):
        '''
        ADD OPERATOR
        integer square root integer --> Number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to integer in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-431'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_432(self):
        '''
        ADD OPERATOR
        integer square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-432'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_433(self):
        '''
        ADD OPERATOR
        number square root boolean --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to boolean in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-433'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_434(self):
        '''
        ADD OPERATOR
        number square root time --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to time in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-434'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_435(self):
        '''
        ADD OPERATOR
        numebr square root date --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to date in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-435'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_436(self):
        '''
        ADD OPERATOR
        number square root time_period --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to time_period in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-436'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_437(self):
        '''
        ADD OPERATOR
        number square root string --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to string in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-437'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_438(self):
        '''
        ADD OPERATOR
        number square root duration --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to duration in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-438'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_439(self):
        '''
        ADD OPERATOR
        number square root integer --> Number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to integer in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-439'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_440(self):
        '''
        ADD OPERATOR
        number square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-440'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_441(self):
        '''
        ADD OPERATOR
        boolean square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast boolean to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-441'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_442(self):
        '''
        ADD OPERATOR
        time square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast time to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-442'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_443(self):
        '''
        ADD OPERATOR
        date square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast date to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-443'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_444(self):
        '''
        ADD OPERATOR
        time_period square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast time_period to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-444'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_445(self):
        '''
        ADD OPERATOR
        string square root number--> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast string to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-445'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_446(self):
        '''
        ADD OPERATOR
        duration square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast duration to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-446'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_447(self):
        '''
        ADD OPERATOR
        integer square root number --> Number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast integer to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-447'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_448(self):
        '''
        ADD OPERATOR
        number square root number --> number
        Status: OK
        Expression: DS_r := sqrt(DS_1) ;
        Description: Forbid implicit cast number to number in square root operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-448'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_449(self):
        '''
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Forbid implicit cast boolean to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-449'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_450(self):
        '''
        ADD OPERATOR
        time --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Forbid implicit cast boolean to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-450'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_451(self):
        '''
        ADD OPERATOR
        date --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Forbid implicit cast date to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-451'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_452(self):
        '''
        ADD OPERATOR
        time_period --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Forbid implicit cast time_period to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-452'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_453(self):
        '''
        ADD OPERATOR
        string --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Forbid implicit cast time_period to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-453'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_454(self):
        '''
        ADD OPERATOR
        duration --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Forbid implicit cast time_period to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-454'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_455(self):
        '''
        ADD OPERATOR
        integer - integer --> integer
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Check implicit cast integer to integer in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-455'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_456(self):
        '''
        ADD OPERATOR
        integer - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1 ;
        Description: Check implicit cast integer to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-456'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_457(self):
        '''
        ADD OPERATOR
        integer - boolean --> number
        Status: OK
        Expression: DS_r := 1 - DS_1 ;
        Description: Forbid implicit cast integer to boolean in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-457'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_458(self):
        '''
        ADD OPERATOR
        integer - time --> number
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to time in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-458'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_459(self):
        '''
        ADD OPERATOR
        integer - date --> number
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to date in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-459'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_460(self):
        '''
        ADD OPERATOR
        integer - time_period --> number
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to time_period in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-460'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_461(self):
        '''
        ADD OPERATOR
        integer - string --> number
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to string in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-461'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_462(self):
        '''
        ADD OPERATOR
        integer - duration --> number
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to duration in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-462'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_463(self):
        '''
        ADD OPERATOR
        integer - integer --> integer
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to integer in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-463'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_464(self):
        '''
        ADD OPERATOR
        integer - number --> number
        Status: OK
        Expression: DS_r := 1 - DS_1;
        Description: Forbid implicit cast integer to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-464'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_465(self):
        '''
        ADD OPERATOR
        number - boolean --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to boolean in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-465'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_466(self):
        '''
        ADD OPERATOR
        number - time --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to time in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-466'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_467(self):
        '''
        ADD OPERATOR
        numebr - date --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to date in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-467'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_468(self):
        '''
        ADD OPERATOR
        number - time_period --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to time_period in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-468'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_469(self):
        '''
        ADD OPERATOR
        number - string --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to string in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-469'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_470(self):
        '''
        ADD OPERATOR
        number - duration --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to duration in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-470'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_471(self):
        '''
        ADD OPERATOR
        number - integer --> Number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to integer in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-471'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_472(self):
        '''
        ADD OPERATOR
        number - number --> number
        Status: OK
        Expression: DS_r := 1.0 - DS_1 ;
        Description: Forbid implicit cast number to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-472'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_473(self):
        '''
        ADD OPERATOR
        boolean - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0;
        Description: Forbid implicit cast boolean to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-473'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_474(self):
        '''
        ADD OPERATOR
        time - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0 ;
        Description: Forbid implicit cast time to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-474'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_475(self):
        '''
        ADD OPERATOR
        date - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0 ;
        Description: Forbid implicit cast date to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-475'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_476(self):
        '''
        ADD OPERATOR
        time_period - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0 ;
        Description: Forbid implicit cast time_period to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-476'
        number_inputs = 1

        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_477(self):
        '''
        ADD OPERATOR
        string - number--> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0 ;
        Description: Forbid implicit cast string to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-477'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_478(self):
        '''
        ADD OPERATOR
        duration - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0 ;
        Description: Forbid implicit cast duration to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-478'
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_479(self):
        '''
        ADD OPERATOR
        integer - number --> Number
        Status: OK
        Expression: DS_r := DS_1 - 1.0;
        Description: Forbid implicit cast integer to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-479'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_480(self):
        '''
        ADD OPERATOR
        number - number --> number
        Status: OK
        Expression: DS_r := DS_1 - 1.0 ;
        Description: Forbid implicit cast number to number in subtraction operator.
        Jira issue: VTLEN 562.
        Git Branch: feat-VTLEN-562-Binary-Add-Numeric-tests.
        Goal: Check Exception.
        '''
        code = '4-3-3-480'
        number_inputs = 1
        references_names = ["DS_r"]

        
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
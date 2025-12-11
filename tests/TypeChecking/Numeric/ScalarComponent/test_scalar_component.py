from pathlib import Path

from tests.Helper import TestHelper


class TestScalarComponentTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class ScalarComponentTypeChecking(TestScalarComponentTypeChecking):
    """
    Group 4
    """

    classTest = "ScalarComponent.ScalarComponentTypeChecking"

    def test_1(self):
        """
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-1"
        # 4 For group numeric
        # 1 For group scalar component
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_2(self):
        """
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1.0 + DS_1#Me_1 ];
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-2"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_3(self):
        """
        ADD OPERATOR
        time --> number
        Error Type: ValueError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast time to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-3"
        number_inputs = 1
        message = "0-3-1-6"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_4(self):
        """
        ADD OPERATOR
        date --> number
        Error Type: ValueError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast date to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-4"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_5(self):
        """
        ADD OPERATOR
        time_period --> number
        Error Type: Should return an error
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-5"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_6(self):
        """
        ADD OPERATOR
        time_period --> number
        test_5 with different csv also fails
        Error Type: Should return an error
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-6"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_7(self):
        """
        ADD OPERATOR
        string --> number
        Error Type: TypeError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-7"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_8(self):
        """
        ADD OPERATOR
        string --> number
        test_7 with different csv
        Error Type: TypeError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-8"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_9(self):
        """
        ADD OPERATOR
        string --> number
        test_7 with different csv
        Error Type: TypeError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast string to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-9"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_10(self):
        """
        ADD OPERATOR
        duration --> number
        Error Type: TypeError
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#duration_var ];
        Description: Forbid implicit cast duration to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-10"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_11(self):
        """
        ADD OPERATOR
        integer --> integer (!number)
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast integer to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        """
        code = "4-1-3-11"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        ADD OPERATOR
        number --> number (!integer)
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast number to integer in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        """
        code = "4-1-3-12"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        ADD OPERATOR
        number --> number (!integer)
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := DS_1#Me_1 + 1 ];
        Description: Forbid implicit cast number to integer in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        """
        code = "4-1-3-13"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        ADD OPERATOR
        boolean --> number
        Status: OK
        Expression: DS_r := DS_1[ calc Me_2 := 1 + DS_1#Me_1 ];
        Description: Forbid implicit cast boolean to number in plus operator.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Exception.
        """
        code = "4-1-3-14"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

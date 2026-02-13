from pathlib import Path

from tests.Helper import TestHelper


class TestLeftJoinTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class LeftJoinIdentifiersTypeChecking(TestLeftJoinTypeChecking):
    """
    Group 2
    """

    classTest = "left_join.LeftJoinIdentifiersTypeChecking"

    def test_1(self):
        """
        LEFT JOIN OPERATOR
        Status: OK
        Expression: DS_r := left_join ( DS_1, DS_2 );
        Description: operations integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-2-1"
        # 2 For group join
        # 2 For group identifiers
        # 2 For clause- for the moment only op left_join
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # def test_2(self):
    #     '''
    #     LEFT JOIN OPERATOR
    #     Status: BUG
    #     Expression: DS_r := left_join ( DS_1, DS_2 );
    #     Description: operations numbers and integers.
    #     Jira issue: VTLEN 564.
    #     Git Branch: feat-VTLEN-564-Join-operators-type-checking.
    #     Goal: Check Exception.
    #     '''
    #     code = '2-2-2-2'
    #     # 2 For group join
    #     # 2 For group identifiers
    #     # 2 For clause- for the moment only op left_join
    #     # 1 Number of test
    #     number_inputs = 2
    #     message = "BUG"
    #
    #     self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_message=message)

    def test_3(self):
        """
        LEFT JOIN OPERATOR
        Status: OK
        Expression: DS_r := left_join ( DS_1, DS_2 );
        Description: operations booleans and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-2-3"
        number_inputs = 2
        # TODO: Fix in #501 - Boolean/Integer implicit casting
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code="1-1-13-18"
        )

    def test_4(self):
        """
        LEFT JOIN OPERATOR
        Status: OK
        Expression: DS_r := left_join ( DS_1, DS_2 );
        Description: operations time and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-2-4"
        number_inputs = 2
        message = "0-3-1-6"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

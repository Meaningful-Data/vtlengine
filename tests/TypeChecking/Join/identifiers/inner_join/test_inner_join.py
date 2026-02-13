from pathlib import Path

from tests.Helper import TestHelper


class TestInnerJoinTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class InnerJoinIdentifiersTypeChecking(TestInnerJoinTypeChecking):
    """
    Group 2
    """

    classTest = "inner_join.InnerJoinIdentifiersTypeChecking"

    def test_1(self):
        """
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-1-1"
        # 2 For group join
        # 2 For group identifiers
        # 1 For clause- for the moment only op inner_join
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # def test_2(self):
    #     '''
    #     INNER JOIN OPERATOR
    #     Status: BUG
    #     Expression: DS_r := inner_join ( DS_1, DS_2 );
    #     Description: inner for the same identifier with diferents types number and integer.
    #     Jira issue: VTLEN 564.
    #     Git Branch: feat-VTLEN-564-Join-operators-type-checking.
    #     Goal: Check Exception.
    #     '''
    #     code = '2-2-1-2'
    #     # 2 For group join
    #     # 2 For group identifiers
    #     # 1 For clause- for the moment only op inner_join
    #     # 2 Number of test
    #     number_inputs = 2
    #     message = "BUG"
    #
    #     self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_message=message)

    def test_3(self):
        """
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: operations between booleans and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-1-3"
        number_inputs = 2
        # TODO: Fix in #501 - Boolean/Integer implicit casting
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code="1-1-13-18"
        )

    def test_4(self):
        """
        INNER JOIN OPERATOR
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 );
        Description: operations between time and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-1-4"
        number_inputs = 2
        message = "0-3-1-6"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

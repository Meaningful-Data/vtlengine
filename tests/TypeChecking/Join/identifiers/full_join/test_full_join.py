from pathlib import Path

from tests.Helper import TestHelper


class TestFullJoinTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class FullJoinIdentifiersTypeChecking(TestFullJoinTypeChecking):
    """
    Group 2
    """

    classTest = "full_join.FullJoinIdentifiersTypeChecking"

    def test_1(self):
        """
        FULL JOIN OPERATOR
        Status: OK
        Expression: DS_r := full_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-3-1"
        # 2 For group join
        # 2 For group identifiers
        # 3 For clause- for the moment only op full_join
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        FULL JOIN OPERATOR
        Status: BUG
        Expression: DS_r := full_join ( DS_1, DS_2 );
        Description: operations between numbers and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-3-2"
        number_inputs = 2
        references_names = ["DS_r"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        FULL JOIN OPERATOR
        Status: OK
        Expression: DS_r := full_join ( DS_1, DS_2 );
        Description: operations between booleans and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        """
        code = "2-2-3-3"
        number_inputs = 2
        references_names = ["DS_r"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        FULL JOIN OPERATOR
        Status: OK
        Expression: DS_r := full_join ( DS_1, DS_2 );
        Description: operations between time and integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        """
        code = "2-2-3-4"
        number_inputs = 2
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

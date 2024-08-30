from pathlib import Path

from testSuite.Helper import TestHelper


class TestExtraTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class ExtraTypeChecking(TestExtraTypeChecking):
    """
    Group 4
    """

    classTest = 'Extra.ExtraTypeChecking'

    def test_1(self):
        '''
        ADD OPERATOR
        Status: Me_1, Me_2 and Me_3 should be numbers, i changed this in the datastructure
        Alternative Expression: DS_r := DS_2 + DS_1[calc Me_3 := DS_1#Me_2 + 1.0 ] + 1.0 ; (good result)
        Expression: DS_r := DS_2 + (DS_1[calc Me_3 := DS_1#Me_2 + 1.0 ] + 1.0 ); (bad result)
        Description: operations between nulls, numbers and integers.
        Jira issue: VTLEN 551.
        Git Branch: feat-VTLEN-551-Numeric-operators-type-checking-tests.
        Goal: Check Result.
        '''
        code = '4-5-3-1'
        # 4 For group numeric
        # 5 For group extra
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]


        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)  # TODO : Review this
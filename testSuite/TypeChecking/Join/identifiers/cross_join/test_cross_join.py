from pathlib import Path

from testSuite.Helper import TestHelper


class TestCrossJoinTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class CrossJoinIdentifiersTypeChecking(TestCrossJoinTypeChecking):
    """
    Group 2
    """

    classTest = 'cross_join.CrossJoinIdentifiersTypeChecking'

    def test_1(self):
        '''
        CROSS JOIN OPERATOR
        Status: OK
        Expression: DS_r := cross_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        '''
        code = '2-2-4-1'
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''
        CROSS JOIN OPERATOR
        Status: BUG
        Expression: DS_r := cross_join ( DS_1, DS_2 );
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Exception.
        '''
        code = '2-2-4-2'
        number_inputs = 2
        references_names = ["DS_r"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''
        CROSS JOIN OPERATOR
        Status: duda orden de las datastructures
        Expression: DS_r := cross_join ( DS_1 as ds1, DS_2 as ds2
        rename ds1#Id_1 to Id_1A,ds1#Id_2 to Id_2A,ds2#Id_1 to Id_1B,ds2#Id_2 to Id_2B);
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        '''
        code = '2-2-4-3'
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        CROSS JOIN OPERATOR
        Status: OK
        Expression: DS_r := cross_join ( DS_1 as ds1, DS_2 as ds2
            rename ds1#Id_1 to Id_11,ds1#Id_2 to Id_12,ds2#Id_1 to Id_21,ds2#Id_2 to Id_22)
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        '''
        code = '2-2-4-4'
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 4 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        '''
        CROSS JOIN OPERATOR
        Status: OK
        Expression: DS_r := cross_join ( DS_1 as ds1, DS_2 as ds2
            rename ds1#Id_1 to Id_11,ds1#Id_2 to Id_12,ds2#Id_3 to Id_21,ds2#Id_4 to Id_22)
        Description: operations between integers.
        Jira issue: VTLEN 564.
        Git Branch: feat-VTLEN-564-Join-operators-type-checking.
        Goal: Check Result.
        '''
        code = '2-2-4-5'
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 5 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

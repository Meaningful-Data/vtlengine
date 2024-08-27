from pathlib import Path

import pytest

from testSuite.Helper import TestHelper


class TestEval(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class SQLliteEval(TestEval):
    """

    """
    classTest = 'Evaltests.SQLliteEval'

    def test_1(self):
        '''

        '''
        code = 'SQL1'
        number_inputs = 1
        references_names = ['DS_r']
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        '''

        '''
        code = 'SQL2'
        number_inputs = 2
        references_names = ['DS_r']
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        '''

        '''
        code = 'SQL3'
        number_inputs = 3
        references_names = ['DS_r']
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        '''
        Semantic Error on Dataset not found in the SQL Query
        that does not match the operands in Eval.
        '''
        code = 'SQL_DS_NOT_FOUND'
        number_inputs = 1
        references_names = ['DS_r']
        with pytest.raises(ValueError, match="External Routine dataset DS_X is not present in Eval operands"):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

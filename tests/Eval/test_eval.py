from pathlib import Path

import pandas as pd
import pytest

from tests.Helper import TestHelper
from vtlengine.Exceptions import SemanticError
from vtlengine.Operators.General import Eval


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
    """ """

    classTest = "Evaltests.SQLliteEval"

    def test_1(self):
        """ """
        code = "SQL1"
        number_inputs = 1
        references_names = ["DS_r"]
        sql_names = ["SQL1"]
        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            sql_names=sql_names,
        )

    def test_2(self):
        """ """
        code = "SQL2"
        number_inputs = 2
        references_names = ["DS_r"]
        sql_names = ["SQL2"]
        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            sql_names=sql_names,
        )

    def test_3(self):
        """ """
        code = "SQL3"
        number_inputs = 3
        references_names = ["DS_r"]
        sql_names = ["SQL3"]
        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            sql_names=sql_names,
        )

    def test_4(self):
        """
        Semantic Error on Dataset not found in the SQL Query
        that does not match the operands in Eval.
        """
        code = "SQL_DS_NOT_FOUND"
        number_inputs = 1
        references_names = ["DS_r"]
        sql_names = ["SQL_DS_NOT_FOUND"]
        with pytest.raises(
            ValueError,
            match="External Routine dataset DS_X is not present in Eval operands",
        ):
            self.BaseTest(
                code=code,
                number_inputs=number_inputs,
                references_names=references_names,
                sql_names=sql_names,
            )

    def test_5(self):
        """
        Semantic Error on Dataset not found in the SQL Query
        that does not match the operands in Eval.
        """
        code = "SQL_INVALID_LANGUAGE"
        number_inputs = 1
        references_names = ["DS_r"]
        sql_names = ["SQL_INVALID_LANGUAGE"]
        with pytest.raises(
            SemanticError,
            match="1-3-6",
        ):
            self.BaseTest(
                code=code,
                number_inputs=number_inputs,
                references_names=references_names,
                sql_names=sql_names,
            )


def test_execute_query_valid():
    query = "SELECT A, B FROM DS_1;"
    datasets = {"DS_1": pd.DataFrame([{"A": 1, "B": 2}])}
    result = Eval._execute_query(query, ["DS_1"], datasets)
    assert result.shape == (1, 2)
    assert result.loc[0, "A"] == 1
    assert result.loc[0, "B"] == 2


def test_execute_query_empty_row():
    query = "SELECT CNTRCT_ID, DT_RFRNC FROM MSMTCH_BL_DS;"
    datasets = {"MSMTCH_BL_DS": pd.DataFrame([{"CNTRCT_ID": None, "DT_RFRNC": None}])}
    result = Eval._execute_query(query, ["MSMTCH_BL_DS"], datasets)
    assert result.shape[0] == 1
    assert pd.isna(result.loc[0, "CNTRCT_ID"])
    assert pd.isna(result.loc[0, "DT_RFRNC"])


def test_execute_query_forbid_install():
    query = "INSTALL some_extension;"
    datasets = {"DS_1": pd.DataFrame([{"A": 1}])}
    with pytest.raises(Exception, match="Query contains forbidden command: INSTALL"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_forbid_load():
    query = "LOAD 'some_file';"
    datasets = {"DS_1": pd.DataFrame([{"A": 1}])}
    with pytest.raises(Exception, match="Query contains forbidden command: LOAD"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_forbid_url_in_from():
    query = "SELECT column_a FROM 'https://domain.tld/file.parquet';"
    datasets = {"DS_1": pd.DataFrame([{"column_a": 1}])}
    with pytest.raises(Exception, match="Query contains forbidden URL in FROM clause"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_sql_error():
    query = "SELECT NONEXISTENT_FUNC(A) FROM DS_1;"
    datasets = {"DS_1": pd.DataFrame([{"A": 1}])}
    with pytest.raises(Exception, match="Error executing SQL query:"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_duckdb_function():
    query = "SELECT ABS(A) AS abs_a FROM DS_1;"
    datasets = {"DS_1": pd.DataFrame([{"A": -10}])}
    result = Eval._execute_query(query, ["DS_1"], datasets)
    assert result.loc[0, "abs_a"] == 10


def test_execute_query_empty_row_with_function_error():
    # On duckdb, julianday does not exist. It is called julian.
    query = """
    SELECT
        julianday(DT_LGL_FNL_MTRTY) - julianday(DT_MTRTY_PRTCTN) AS PRTCTN_RSDL_MTRTY_DYS
    FROM MSMTCH_BL_DS;
    """
    datasets = {"MSMTCH_BL_DS": pd.DataFrame([{"DT_LGL_FNL_MTRTY": None, "DT_MTRTY_PRTCTN": None}])}
    with pytest.raises(Exception, match="Error executing SQL query:"):
        Eval._execute_query(query, ["MSMTCH_BL_DS"], datasets)

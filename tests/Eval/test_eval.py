from pathlib import Path

import pandas as pd
import pytest

from tests.Helper import TestHelper, _use_duckdb_backend, engine_params
from vtlengine import run, semantic_analysis
from vtlengine.Exceptions import RunTimeError, SemanticError
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
            SemanticError,
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

    def test_GH_1276_3(self):
        """
        Expression: DS_r1 <- eval ( GH_1276_3 ( DS_1 ) ... ) [ calc Me_2 := Me_1 * 2 ];
                    DS_r2 <- eval ( GH_1276_3 ( DS_1 ) ... ) * 2;
                    DS_r3 <- isnull ( eval ( GH_1276_3 ( DS_1 ) ... ) );
        Description: the result of an eval is the operand of a clause or of another operator.
                     The DuckDB engine had no structure for it and wrote its query inside the
                     expression of the outer operator: the clause raised an AttributeError, the
                     product a BinderException and isnull returned no Data Points, while the
                     pandas engine gave these results.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1276
        """
        code = "GH_1276_3"
        number_inputs = 1
        references_names = ["1", "2", "3"]
        sql_names = ["GH_1276_3"]
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
    with pytest.raises(SemanticError, match="forbidden command: INSTALL"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_forbid_load():
    query = "LOAD 'some_file';"
    datasets = {"DS_1": pd.DataFrame([{"A": 1}])}
    with pytest.raises(SemanticError, match="forbidden command: LOAD"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_forbid_url_in_from():
    query = "SELECT column_a FROM 'https://domain.tld/file.parquet';"
    datasets = {"DS_1": pd.DataFrame([{"column_a": 1}])}
    with pytest.raises(SemanticError, match="forbidden URL in FROM clause"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_sql_error():
    query = "SELECT NONEXISTENT_FUNC(A) FROM DS_1;"
    datasets = {"DS_1": pd.DataFrame([{"A": 1}])}
    with pytest.raises(RunTimeError, match="DuckDB runtime error"):
        Eval._execute_query(query, ["DS_1"], datasets)


def test_execute_query_infinite_values():
    query = "SELECT A, CAST('Infinity' AS DOUBLE) AS B FROM DS_1;"
    datasets = {"DS_1": pd.DataFrame([{"A": 1}])}
    with pytest.raises(RunTimeError) as context:
        Eval._execute_query(query, ["DS_1"], datasets)
    assert context.value.args[1] == "2-1-3-1"


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
    with pytest.raises(RunTimeError, match="DuckDB runtime error"):
        Eval._execute_query(query, ["MSMTCH_BL_DS"], datasets)


def test_eval_julian_with_date_columns():
    """Date columns stored as string[pyarrow] should work with DuckDB date functions."""

    script = """
        DS_r <- eval(
            dateDiff(DS_1)
            language "sql"
            returns dataset {
                identifier<string> Id_1,
                measure<integer> DIFF_DAYS
            }
        );
    """

    er = {
        "name": "dateDiff",
        "query": (
            "SELECT Id_1, CAST(julian(Me_1) - julian(Me_2) AS INTEGER) AS DIFF_DAYS FROM DS_1;"
        ),
    }

    data_structures = {
        "datasets": [
            {
                "name": "DS_1",
                "DataStructure": [
                    {"name": "Id_1", "type": "String", "role": "Identifier", "nullable": False},
                    {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
                    {"name": "Me_2", "type": "Date", "role": "Measure", "nullable": True},
                ],
            }
        ]
    }

    df = pd.DataFrame(
        {
            "Id_1": ["A", "B"],
            "Me_1": ["2024-01-15", "2024-06-01"],
            "Me_2": ["2024-01-01", "2024-05-01"],
        }
    )
    datapoints = {"DS_1": df}

    expected_diff_days = (pd.to_datetime(df["Me_1"]) - pd.to_datetime(df["Me_2"])).dt.days

    result = run(
        script=script,
        data_structures=data_structures,
        datapoints=datapoints,
        external_routines=er,
        use_duckdb=_use_duckdb_backend(),
    )

    assert result["DS_r"] is not None
    assert result["DS_r"].data["DIFF_DAYS"].tolist() == expected_diff_days.tolist()


EVAL_IN_CLAUSE_STRUCTURES = {
    "datasets": [
        {
            "name": "DS_1",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        }
    ]
}


@pytest.mark.parametrize(
    "script",
    [
        'DS_r <- DS_1[calc Me_2 := eval(SQL_1(Me_1) language "SQL" returns number)];',
        'DS_r <- DS_1[calc Me_2 := eval(SQL_1(Me_1, Me_1) language "SQL" returns number)];',
    ],
)
def test_eval_inside_a_clause_is_not_supported(script: str) -> None:
    """Eval only applies to Data Sets: inside a clause it is a SemanticError, where an
    IndexError was raised."""
    external_routines = [{"name": "SQL_1", "query": "SELECT Me_1 FROM DS_1"}]
    with pytest.raises(SemanticError) as context:
        semantic_analysis(
            script=script,
            data_structures=EVAL_IN_CLAUSE_STRUCTURES,
            external_routines=external_routines,
        )
    assert context.value.args[1] == "1-3-2-5"
    with pytest.raises(SemanticError) as context:
        run(
            script=script,
            data_structures=EVAL_IN_CLAUSE_STRUCTURES,
            datapoints={"DS_1": pd.DataFrame({"Id_1": [1], "Me_1": [1.0]})},
            external_routines=external_routines,
            use_duckdb=_use_duckdb_backend(),
        )
    assert context.value.args[1] == "1-3-2-5"


_STRUCTURES_1356 = {
    "datasets": [
        {
            "name": "DS_2",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Number", "role": "Measure", "nullable": True},
            ],
        },
        {
            "name": "DS_4",
            "DataStructure": [
                {"name": "Id_1", "type": "Integer", "role": "Identifier", "nullable": False},
                {"name": "Me_1", "type": "Date", "role": "Measure", "nullable": True},
            ],
        },
    ]
}


def _eval_1356(use_duckdb, query, returns, operand="DS_2"):
    """Runs eval of the query on the operand, DS_2 (Id_1 1, 2 and the Number Me_1 1.5, null)
    or DS_4 (the Date Me_1 2020-01-31, null), after checking it in semantic_analysis, and
    returns the result sorted by Id_1."""
    script = (
        f'DS_r <- eval(SQL_1({operand}) language "SQL" returns dataset'
        f" {{ identifier<integer> Id_1, {returns} }});"
    )
    routine = {"name": "SQL_1", "query": query}
    semantic_analysis(script, _STRUCTURES_1356, external_routines=routine)
    datapoints = {
        "DS_2": pd.DataFrame({"Id_1": [1, 2], "Me_1": [1.5, None]}),
        "DS_4": pd.DataFrame({"Id_1": [1, 2], "Me_1": ["2020-01-31", None]}),
    }
    result = run(
        script,
        _STRUCTURES_1356,
        datapoints,
        external_routines=routine,
        use_duckdb=use_duckdb,
    )["DS_r"]
    return result.data.sort_values("Id_1").reset_index(drop=True)


def _eval_error_1356(use_duckdb, query, returns):
    with pytest.raises(RunTimeError) as context:
        _eval_1356(use_duckdb, query, returns)
    return context.value


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1356_49(use_duckdb):
    """A query that reads DS_2 twice (a self join) runs as written on both engines. The DuckDB
    engine quoted the second DS_2 twice and raised a raw ParserException (issue #1356)."""
    result = _eval_1356(
        use_duckdb,
        "SELECT a.Id_1, a.Me_1 FROM DS_2 a JOIN DS_2 b ON a.Id_1 = b.Id_1",
        "measure<number> Me_1",
    )
    assert result["Me_1"].tolist()[0] == 1.5
    assert pd.isna(result["Me_1"].tolist()[1])


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1356_50(use_duckdb):
    """The string literal 'DS_2' is not a table: it is returned as written on both engines.
    The DuckDB engine returned '"DS_2"' (issue #1356)."""
    result = _eval_1356(use_duckdb, "SELECT Id_1, 'DS_2' AS Me_2 FROM DS_2", "measure<string> Me_2")
    assert result["Me_2"].tolist() == ["DS_2", "DS_2"]


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1356_51(use_duckdb):
    """A query that ends with a -- comment runs on both engines. On the DuckDB engine the
    comment swallowed the parenthesis around the query: raw ParserException (issue #1356)."""
    result = _eval_1356(
        use_duckdb, "SELECT Id_1, Me_1 FROM DS_2 -- DS_2 comment", "measure<number> Me_1"
    )
    assert result["Me_1"].tolist()[0] == 1.5
    assert pd.isna(result["Me_1"].tolist()[1])


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1356_52(use_duckdb):
    """The name t of a WITH clause is defined by the query, it is not an operand: the query
    runs on both engines and in semantic_analysis, which raised 1-1-1-24 (issue #1356)."""
    result = _eval_1356(
        use_duckdb,
        "WITH t AS (SELECT Id_1, Me_1 FROM DS_2) SELECT Id_1, Me_1 FROM t",
        "measure<number> Me_1",
    )
    assert result["Me_1"].tolist()[0] == 1.5
    assert pd.isna(result["Me_1"].tolist()[1])


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1356_53(use_duckdb):
    """Double quotes delimit an identifier, as in DuckDB SQL: "Me_1" is the Measure Me_1 on
    both engines. They turned it into the text Me_1 and raised a raw ArrowInvalid (pandas) or
    ConversionException (DuckDB) (issue #1356)."""
    result = _eval_1356(use_duckdb, 'SELECT Id_1, "Me_1" AS Me_2 FROM DS_2', "measure<number> Me_2")
    assert result["Me_2"].tolist()[0] == 1.5
    assert pd.isna(result["Me_2"].tolist()[1])


@pytest.mark.parametrize("use_duckdb", engine_params())
def test_GH_1356_54(use_duckdb):
    """The query is checked on the declared types of its operands, so a date function on the
    Date Me_1 runs on both engines and in semantic_analysis, which raised 1-1-1-23 (issue
    #1356)."""
    result = _eval_1356(
        use_duckdb,
        "SELECT Id_1, Me_1 - INTERVAL 1 DAY AS Me_3 FROM DS_4",
        "measure<date> Me_3",
        operand="DS_4",
    )
    assert result["Me_3"].tolist()[0] == "2020-01-30"
    assert pd.isna(result["Me_3"].tolist()[1])

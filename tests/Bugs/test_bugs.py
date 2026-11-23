from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pandas as pd
import pytest

from tests.Helper import TestHelper, _use_duckdb_backend
from vtlengine.API import create_ast, run
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Interpreter import InterpreterAnalyzer


class BugHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class GeneralBugs(BugHelper):
    """ """

    classTest = "Bugs.GeneralBugs"

    def test_GL_22(self):
        """
        Description: cast zero value to number-Integer.
        Git Branch: bug-22-improve-cast-zero-to-number-integer.
        Goal: Interpreter results.
        """
        code = "GL_22"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_408(self):
        """ """
        code = "GL_408"
        number_inputs = 2
        references_names = ["1", "2", "3", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_314_1(self):
        """ """
        script = """
        a <- cast("2020-M01", time_period);
        b := cast("2020-M01", time_period);
        c <- a;
        d := a;
        e <- b;
        f := b;
        """

        references = {
            "a": True,
            "b": False,
            "c": True,
            "d": False,
            "e": True,
            "f": False,
        }

        if _use_duckdb_backend():
            result = run(
                script=script,
                data_structures={"datasets": []},
                datapoints={},
                return_only_persistent=False,
                use_duckdb=True,
            )
        else:
            ast = create_ast(script)
            interpreter = InterpreterAnalyzer(datasets={})
            result = interpreter.visit(ast)
        for sc in result.values():
            assert sc.persistent == references[sc.name]

    def test_GH_917(self):
        """
        Status: OK
        Description: The DuckDB path applied ``scalar_values`` to the input
            scalars without casting them to the declared data type. A string
            value for an Integer scalar (``{"sc1": "1"}``) was inlined into
            the SQL as a string literal, so ``99 + sc1`` failed with a DuckDB
            Binder Error. The pandas path casts the values and works.
        Git issue: https://github.com/Meaningful-Data/vtlengine/issues/917
        Goal: Check Result.
        """
        code = "GH_917"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc1": "1"},
        )

    def test_GH_945_1(self):
        """
        Expression: ds_from_membership_3 := max(DS_1#Me_int)#Me_int;
                    in_member_one_row <- ds_from_membership_3 = 9;
        Description: a derived scalar (membership on an ungrouped aggregation)
            referenced by a later statement must resolve in both engines; the
            DuckDB engine stored the scalar table with the measure column name
            instead of ``value`` and raised ``BinderException``.
        Git Issue: GH_945.
        Goal: Check scalar result values.
        """
        code = "GH_945_1"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            return_only_persistent=False,
            use_duckdb=_use_duckdb_backend(),
        )
        assert result["ds_from_membership_3"].value == 9
        assert result["in_member_one_row"].value is True

    def test_GH_945_2(self):
        """
        Expression: x := max(DS_1#Me_int)#Me_int;
                    DS_r <- DS_1[calc Me_2 := Me_int + x];
        Description: a derived scalar (membership on an ungrouped aggregation)
            referenced inside a dataset expression must resolve in both
            engines; the DuckDB engine raised ``BinderException`` because the
            scalar table exposed no ``value`` column.
        Git Issue: GH_945.
        Goal: Check component result values.
        """
        code = "GH_945_2"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        data = result["DS_r"].data.sort_values("Id_1")
        assert data["Me_2"].tolist() == [16, 17, 18]

    def test_GH_945_3(self):
        """
        Expression: define viral propagation VP_VAt_1 (variable VAt_1)
                        is aggregate max end viral propagation;
                    x := max(DS_1#Me_int)#Me_int;
                    y <- x + 1;
        Description: membership on an ungrouped aggregation that kept a viral
            attribute is a scalar extraction, so the viral attribute is
            dropped (a scalar has no attributes, mirroring
            ``Membership.validate``); the DuckDB engine emitted the viral
            column too and the scalar subquery raised ``BinderException``
            (Subquery returns 2 columns).
        Git Issue: GH_945.
        Goal: Check scalar result values.
        """
        code = "GH_945_3"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            return_only_persistent=False,
            use_duckdb=_use_duckdb_backend(),
        )
        assert result["x"].value == 9
        assert result["y"].value == 10

    def test_GH_1163_1(self):
        """
        Status: OK
        Expression: DS_r <- timeshift(DS_1, DS_2);
        Description: A Data Set as the shift number of timeshift raises a SemanticError instead of a Python error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1163
        Goal: Check Exception.
        """
        code = "GH_1163_1"
        number_inputs = 2
        message = "1-1-18-4"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1163_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := timeshift(Id_1, 1)];
        Description: timeshift over a Component raises a SemanticError instead of a Python error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1163
        Goal: Check Exception.
        """
        code = "GH_1163_2"
        number_inputs = 1
        message = "1-1-19-7"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1163_3(self):
        """
        Status: OK
        Expression: DS_r <- eval(GH_1163_3(DS_1) language "SQL" returns integer);
        Description: An eval that returns a scalar type raises a SemanticError instead of a Python error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1163
        Goal: Check Exception.
        """
        code = "GH_1163_3"
        number_inputs = 1
        message = "1-3-2-1"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            sql_names=["GH_1163_3"],
        )

    def test_GH_1163_4(self):
        """
        Status: OK
        Expression: DS_r <- inner_join(DS_1 as d1, DS_2 as d2 apply d1 + 1);
        Description: A constant in the apply expression of a join raises a SemanticError instead of a Python error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1163
        Goal: Check Exception.
        """
        code = "GH_1163_4"
        number_inputs = 2
        message = "1-1-13-22"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_1163_5(self):
        """
        Status: OK
        Expression: DS_r <- inner_join(DS_1 apply DS_1 + DS_1);
        Description: An apply over the only operand of a join raises a SemanticError instead of a Python error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1163
        Goal: Check Exception.
        """
        code = "GH_1163_5"
        number_inputs = 1
        message = "1-1-13-22"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
        )

    def test_GH_976_1(self):
        """
        Expression: DS_r <- check_datapoint(DS_1, DR_1);
        Description: check_datapoint result values must match the reference
            in both engines (the errorlevel component is declared as Number).
        Git Issue: GH_976.
        Goal: Check Result.
        """
        code = "GH_976_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_976_1_dtypes(self):
        """
        Expression: DS_r <- check_datapoint(DS_1, DR_1);
        Description: in-memory outputs must materialize the declared component
            dtypes in both engines; the DuckDB engine returned the physical
            SQL type from fetchdf() (errorlevel INTEGER -> int32) instead of
            the declared type, Integer (int64[pyarrow]) since issue #1256.
        Git Issue: GH_976, GH_1256.
        Goal: Check returned DataFrame dtypes.
        """
        code = "GH_976_1"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        assert str(result["DS_r"].data["errorlevel"].dtype) == "int64[pyarrow]"

    @pytest.mark.skipif(not _use_duckdb_backend(), reason="parquet output_format is DuckDB-only")
    def test_GH_976_1_parquet(self):
        """
        Expression: DS_r <- check_datapoint(DS_1, DR_1);
        Description: parquet files written to output_folder must carry the
            declared component type for physical/declared mismatches; the
            errorlevel column (declared Number) was written as INT32 instead
            of DOUBLE. Since issue #1256 errorlevel is declared Integer.
        Git Issue: GH_976, GH_1256.
        Goal: Check written parquet schema.
        """
        code = "GH_976_1"
        script = self.LoadVTL(code)
        with TemporaryDirectory() as tmpdir:
            run(
                script=script,
                data_structures=self.filepath_json / f"{code}-1.json",
                datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
                output_folder=Path(tmpdir),
                output_format="parquet",
                use_duckdb=True,
            )
            with duckdb.connect() as conn:
                rows = conn.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{Path(tmpdir) / 'DS_r.parquet'}')"
                ).fetchall()
        types = {name: sql_type for name, sql_type, *_ in rows}
        assert types["errorlevel"] in ("INTEGER", "BIGINT")

    def test_GH_976_2(self):
        """
        Expression: DS_r <- check_datapoint(DS_1, DR_1 all_measures);
        Description: check_datapoint all_measures result values (bool_var,
            errorcode and errorlevel nulls included) must match the reference
            in both engines.
        Git Issue: GH_976.
        Goal: Check Result.
        """
        code = "GH_976_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_976_2_csv_text(self):
        """
        Expression: DS_r <- check_datapoint(DS_1, DR_1 all_measures);
        Description: dataset CSVs written to output_folder must render Number
            values with the shared significant-digits format in both engines;
            the DuckDB engine wrote the DECIMAL(28,10) text (10.5000000000)
            via COPY TO where the pandas engine writes 10.5.
        Git Issue: GH_976.
        Goal: Check written CSV text.
        """
        code = "GH_976_2"
        script = self.LoadVTL(code)
        with TemporaryDirectory() as tmpdir:
            run(
                script=script,
                data_structures=self.filepath_json / f"{code}-1.json",
                datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
                output_folder=Path(tmpdir),
                use_duckdb=_use_duckdb_backend(),
            )
            content = (Path(tmpdir) / "DS_r.csv").read_text()
        lines = content.strip().splitlines()
        assert lines[0] == "Id_1,ruleid,Me_1,bool_var,errorcode,errorlevel"
        assert sorted(lines[1:]) == [
            "1,R_1,10.5,True,,",
            "2,R_1,-5,False,R_1,2",
            "3,R_1,,,,",
        ]

    def test_GH_976_3(self):
        """
        Expression: DS_r <- DS_1;
        Description: a passthrough of every scalar type (Number, Integer,
            String, Boolean, Date) with nulls must match the reference in
            both engines.
        Git Issue: GH_976.
        Goal: Check Result.
        """
        code = "GH_976_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_976_3_dtypes(self):
        """
        Expression: DS_r <- DS_1;
        Description: in-memory outputs must materialize the declared component
            dtypes (DTYPE_MAPPING, pyarrow-backed) for every scalar type in
            both engines; the DuckDB engine returned numpy-backed dtypes from
            fetchdf() (object/float64/int32).
        Git Issue: GH_976.
        Goal: Check returned DataFrame dtypes.
        """
        code = "GH_976_3"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        dtypes = {name: str(dtype) for name, dtype in result["DS_r"].data.dtypes.items()}
        assert dtypes == {
            "Id_1": "int64[pyarrow]",
            "Me_1": "double[pyarrow]",
            "Me_2": "int64[pyarrow]",
            "Me_3": "string",
            "Me_4": "bool[pyarrow]",
            "Me_5": "string",
        }

    def test_GH_1322(self):
        """
        Status: OK
        Expression: DS_r <- 1DS_1[calc Me_2 := 1Me_1 * 10];
        Description: A bare name that starts with a digit and contains a letter is a regular
                     VTL name. Since #1231 it raised a VTLSyntaxError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1322
        Goal: Check Result.
        """
        code = "GH_1322"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class JoinBugs(BugHelper):
    """ """

    classTest = "Bugs.JoinBugs"

    def test_VTLEN_569(self):
        """ """
        code = "VTLEN_569"
        number_inputs = 2

        error_code = "1-1-13-6"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_VTLEN_572(self):
        """
        Description:
        Jira issue: VTLEN 572.
        Git Branch: bug-VTLEN-572-Inner-join-with-using-clause.
        Goal: Check semantic result.
        """
        code = "VTLEN_572"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_40(self):
        """ """
        code = "GL_40"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_24(self):
        """ """
        code = "GL_24"
        number_inputs = 2
        message = "1-1-13-11"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_32(self):
        """
        Status: OK
        Expression: DS_r := inner_join ( AMOUNTS [ sub measure_ = "O" ] [ rename OBS_VALUE to O ] [ drop OBS_STATUS ]
                            as A ,  AMOUNTS [ sub measure_ = "V" ] [ rename OBS_VALUE to V ] [ drop OBS_STATUS ] as B);
        Description: Inner join on same dataset
        Git Branch: fix-32-names-joins
        Goal: Check Result.
        """
        code = "GL_32"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_63(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_63"
        number_inputs = 2
        references_names = ["1", "2", "3", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_14(self):
        """
        Description:
        Git Branch: bug-14-left_join-interpreter-error.
        Goal: Check semantic result and interpreter results.
        """
        # code = "GL_14"
        # number_inputs = 6
        # message = "1-1-13-3"
        # TODO: check up this error test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_GL_133_1(self):
        """
        Description: Fails on line 79-83: NLE2 := inner_join(NLE, LE_JN using DT_RFRNC);
                     This is not allowed as fails in case B2. (VTL Reference line 2269)
        Git Branch: fix-197-inner-using.
        Goal: Check exception.
        """
        code = "GL_133_1"
        number_inputs = 1
        vd_names = ["GL_133_1-1"]

        message = "1-1-13-9"

        # HUH!!!!!!!!!
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_GL_133_2(self):
        """
        Description: NLE2 := inner_join(NLE, LE_JN using DT_RFRNC);
             This is not allowed as fails in case B2. (VTL Reference line 2269)
        Git Branch: fix-197-inner-using.
        Goal: Check exception.
        """
        code = "GL_133_2"
        number_inputs = 2

        message = "1-1-13-9"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_133_3(self):
        """
        Description: DS_3 := inner_join(DS_1, DS_2 using Id_1);
            This is not allowed as fails in case B2. (VTL Reference line 2269)
        Git Branch: fix-197-inner-using.
        Goal: Check exception.
        """
        code = "GL_133_3"
        number_inputs = 2
        message = "1-1-13-9"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_161_1(self):
        """
        Description: inner join with duplicated attributes.
        Git Branch: bug-161-inner-join-not-working-properly-attributes-duplicated.
        Goal: Check Exception.
        """
        # code = "GL_161_1"
        # number_inputs = 2
        # message = "1-1-13-3"
        # TODO: check up this error test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_GL_161_2(self):
        """
        Description: inner join with duplicated attributes.
        Git Branch: bug-161-inner-join-not-working-properly-attributes-duplicated.
        Goal: Check Result.
        """
        code = "GL_161_2"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(
            text=None,
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
        )

    def test_GL_47_4(self):
        """
        Description: Two same rename.
        Git Branch: #47.
        Goal: Check Result.
        """
        code = "GL_47_4"
        number_inputs = 2
        message = "1-2-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_47_5(self):
        """
        Description: Two same rename.
        Git Branch: #47.
        Goal: Check Result.
        """
        # code = "GL_47_5"
        # number_inputs = 2
        # message = "Join conflict with duplicated names for column reference_date from original datasets."
        # message = "1-1-13-3"
        # TODO: check up this error test
        # "1-3-4"
        # self.NewSemanticExceptionTest(
        #     code=code,
        #     number_inputs=number_inputs,
        #     exception_code=message
        # )
        assert True

    def test_GL_47_6(self):
        """
        Description: Two duplicated components.
        Git Branch: #47.
        Goal: Check Result.
        """
        # code = "GL_47_6"
        # number_inputs = 2
        # message = "1-1-13-3"
        # TODO: check up this error test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_GL_47_8(self):
        """
        Description:
        Git Branch: #47.
        Goal: Check Result.
        """
        code = "GL_47_8"
        number_inputs = 2
        message = "1-1-6-8"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_64_1(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_64_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_2(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_64_2"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_3(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_64_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_4(self):
        """
        Description:
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_64_4"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_5(self):
        """
        Description: inner join
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_64_5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_64_7(self):
        """
        Description: inner join
        Git Branch: bug-63-max-after-cross-join-not-working-properly.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_64_7"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_239_1(self):
        """
        Description: keep after keep(ds_2) inside a inner join. / Semantic error but the expression is correct.
        Git feat-234-new-grammar-parser.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_239_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_250(self):
        """
        Description: Alias symbol and identifier in validate types
        Git fix-250-rename-sameDS.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_250"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_255(self):
        """
        Description: Drop inside a join
        Git fix-255-drop-join.
        Goal: Check semantic result and interpreter results.
        """
        code = "GL_255"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_253(self):
        """
        Description: Duplicated component names on result join
        Git fix-253-duplicated-inner.
        Goal: Check semantic result (BKAR is duplicated).
        """
        # code = "GL_253"
        # number_inputs = 2
        # message = "1-1-13-3"
        # TODO: check up this error test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_GL_279(self):
        """
        Description: Aggr with join and other clause
        Git fix-279-aggr-join.
        Goal: Check result.
        """
        code = "GL_279"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_23(self):
        """
        Status: OK
        Expression: DS_r <- inner_join(DS_1 as Me_1, DS_2 as d2);
        Description: An alias named as a Component of its Data Set is rejected with a VTL
                     error, where the engine raised a KeyError on a missing message code.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_23"
        number_inputs = 2
        message = "1-1-6-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1318_1(self):
        """
        Status: OK
        Expression: F <- DS_2[rename Id_2 to K];
                    B <- left_join(DS_1[calc measure K := Me_1] as d, F as f using K);
                    C <- left_join(DS_1[calc measure K := Me_1] as d, F as f using K);
        Description: K is a Measure of the left-most operand and the Identifier of F. The
                     join turned K into a Measure in F itself, so the same join failed the
                     second time with 1-2-10 and F was returned without an Identifier.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1318
        Goal: Check Result.
        """
        code = "GH_1318_1"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class NumericBugs(BugHelper):
    """ """

    classTest = "Bugs.NumericBugs"

    def test_GL_27_2(self):
        """
        Expression: DS_r := DS_1 + DS_1;
        Description: Basic Op + with Identifier type Time_Period.
        Git Issue: GL_27-Pandas merge over dates reference columns.
        Goal: Check Result.
        """
        code = "GL_27_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_27_3(self):
        """
        Expression: DS_r := DS_1 + DS_1;
        Description: Basic Op + with Identifier type Date.
        Git Issue: GL_27-Pandas merge over dates reference columns.
        Goal: Check Result.
        """
        code = "GL_27_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_413(self):
        """
        Expression: A := cast(BIS_LBS_DISS#OBS_VALUE, integer) * 2;
        Description: The cast of the Number Measure OBS_VALUE to integer is refused, as the VTL 2.2
                     manual marks number to integer as not feasible. It used to drop the decimals.
        Git Issue: GL_413-cast-with-integer, https://github.com/Meaningful-Data/vtlengine/issues/1388
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GL_413", number_inputs=1, exception_code="1-1-5-4")

    def test_GH_763_1(self):
        """
        Expression: DS_A <- abs(DS_1);
        Description: Unary abs on a dataset with no datapoints failed because the
            pyarrow fast-path expected ``series.values`` to be an
            ArrowExtensionArray, but pandas returns a numpy ndarray for empty
            non-arrow columns.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_763_2(self):
        """
        Expression: DS_A <- ceil(DS_1);
        Description: Same empty-dataset pyarrow fast-path bug as GH_763_1, for ceil.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_763_3(self):
        """
        Expression: DS_A <- floor(DS_1);
        Description: Same empty-dataset pyarrow fast-path bug as GH_763_1, for floor.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_763_4(self):
        """
        Expression: DS_A <- exp(DS_1);
        Description: Same empty-dataset pyarrow fast-path bug as GH_763_1, for exp.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_763_5(self):
        """
        Expression: DS_A <- ln(DS_1);
        Description: Same empty-dataset pyarrow fast-path bug as GH_763_1, for ln.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_763_6(self):
        """
        Expression: DS_A <- sqrt(DS_1);
        Description: Same empty-dataset pyarrow fast-path bug as GH_763_1, for sqrt.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_763_7(self):
        """
        Expression: DS_A <- -DS_1;
        Description: Same empty-dataset pyarrow fast-path bug as GH_763_1, for the
            unary minus operator.
        Git Issue: GH_763.
        Goal: Check Result.
        """
        code = "GH_763_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_766_1(self):
        """
        Expression: DS_A <- round(DS_1);
        Description: ``round`` without param promotes Number measures to Integer,
            renaming the measure column (e.g. ``Me_1`` -> ``int_var``).
            ``Parameterized.dataset_evaluation`` iterated the result's renamed
            measures over the operand's data (still using the original names),
            raising ``KeyError: 'int_var'``.
        Git Issue: GH_766.
        Goal: Check Result.
        """
        code = "GH_766_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_766_2(self):
        """
        Expression: DS_A <- trunc(DS_1);
        Description: Same KeyError as GH_766_1 for ``trunc`` without param.
        Git Issue: GH_766.
        Goal: Check Result.
        """
        code = "GH_766_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_766_3(self):
        """
        Expression: DS_A <- round(DS_1, 1);
        Description: ``round`` with an explicit decimal param keeps the Number
            type and the original measure name. Regression guard for the
            no-rename branch of the same fix.
        Git Issue: GH_766.
        Goal: Check Result.
        """
        code = "GH_766_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_766_4(self):
        """
        Expression: DS_A <- trunc(DS_1, 1);
        Description: ``trunc`` with an explicit decimal param keeps the Number
            type and the original measure name. Regression guard for the
            no-rename branch of the same fix.
        Git Issue: GH_766.
        Goal: Check Result.
        """
        code = "GH_766_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_1(self):
        """
        Expression: DS_A <- round(DS_1);
        Description: When the operand has an attribute, ``round`` without param
            renames the measure (``Me_1`` -> ``int_var``) and
            ``modify_measure_column`` copied the attribute column into the new
            measure name, raising a numeric cast error on the string values
            (e.g. ``Failed to parse string: 'D' as a scalar of type int64``).
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_2(self):
        """
        Expression: DS_A <- trunc(DS_1);
        Description: Same attribute-leak bug as GH_778_1, for ``trunc`` without param.
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_3(self):
        """
        Expression: DS_A <- round(DS_1, 1);
        Description: ``round`` with param on a dataset with an attribute keeps
            the Number measure and must still drop the attribute from the
            result. Regression guard for the no-rename branch.
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_4(self):
        """
        Expression: DS_A <- trunc(DS_1, 1);
        Description: ``trunc`` with param on a dataset with an attribute keeps
            the Number measure and must still drop the attribute from the
            result. Regression guard for the no-rename branch.
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_5(self):
        """
        Expression: DS_A <- sum(DS_1 over(partition by Id_1));
        Description: ``Analytic.validate`` kept all components (including
            non-viral attributes) in ``result.components`` while the underlying
            SQL query selected only identifiers and measures, leaving
            ``result.data`` without the attribute column. Per VTL 2.1
            ("Attribute propagation rule" applied at Data Set level), non-viral
            attributes must be dropped from analytic results.
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_6(self):
        """
        Expression: DS_A <- avg(DS_1 over(order by Id_1));
        Description: Same attribute-propagation bug as GH_778_5, for ``avg``
            with an ``order by`` analytic clause.
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_778_7(self):
        """
        Expression: DS_A <- time_agg("A", _, DS_1);
        Description: ``Time_Aggregation.dataset_validation`` dropped attributes
            from ``result.components`` but ``dataset_evaluation`` copied
            ``operand.data`` verbatim, leaving the attribute column in
            ``result.data``. Per VTL 2.1 (time_agg "returns a Data Set having
            the same structure" + attribute propagation rule), non-viral
            attributes must be dropped from both components and data.
        Git Issue: GH_778.
        Goal: Check Result.
        """
        code = "GH_778_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_985_1(self):
        """
        Expression: DS_copy := DS_1; DS_r := DS_1[calc Me_1 := Me_1 / 3];
        Description: the DuckDB engine stored Number as DECIMAL(28,10), so a
            plain copy already truncated 15-significant-digit values to 10
            decimal places (0.123456789012345 -> 0.123456789, small values lost
            almost every digit) and |x| >= 1e18 failed to load. Number is now
            DOUBLE end to end and both engines keep 15 significant digits.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/985
        Goal: Check Result.
        """
        code = "GH_985_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_985_2(self):
        """
        Expression: DS_r := DS_1[calc Me_1 := (Me_1 / 3) * 3, Me_2 := 1/3];
        Description: per-operation rounding to 15 significant digits is
            observable ((1/3)*3 is 0.999999999999999, not 1) and previously only
            the pandas engine applied it. Both engines now share the same
            float64 + round-half-even kernel, so the chained result and the
            literal division pin identical values on both backends.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/985
        Goal: Check Result.
        """
        code = "GH_985_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := random(Id_1, -1)];
        Description: A negative literal index is rejected by the semantic analysis.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_5"
        number_inputs = 1
        message = "1-1-15-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_6(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := random(Id_1, sc_1)];
        Description: A negative index given as a scalar value is a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_6"
        number_inputs = 1
        message = "2-1-15-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": -1}
        )

    def test_GH_1245(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_5 := mod(Me_1, Me_3), Me_6 := mod(Me_2, Me_4), ...];
        Description: The remainder of mod takes the sign of the divisor (mod(5, -2) is -1,
                     not 1) and a zero divisor gives the dividend back (mod(9, 0) is 9, where
                     both engines raised 2-1-15-6), on Integer and Number components, with
                     another component or a scalar, and with null values.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1245
        Goal: Check Result.
        """
        code = "GH_1245"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class ComparisonBugs(BugHelper):
    """ """

    classTest = "Bugs.ComparisonBugs"

    def test_VTLEN_346(self):
        """ """
        code = "VTLEN_346"
        number_inputs = 2
        references_names = ["RI0110"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # Comparison operators mix measures and attributes
    def test_GL_56_1(self):
        """
        Status: OK
        Expression: A:= BOP >0;
        Description: datasetScalarEvaluation
        Git Issue: GL_56-Comparison operators mix measures and attributes.
        Goal: Check Result.
        """
        code = "GL_56_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_56_2(self):
        """
        Status: OK.
        Expression: A:= BOP[calc Me_1 := OBS_VALUE > 0];
        Description: Comparison at component level, componentScalarEvaluation.
            This works at component level, so we can keep the
            attributes on the dataset. But read RM lines(1191-1195) says
            in different way, but do not care about this, calc has a specific
            behaviour, so this is the good result.
        Git Issue: GL_56-Comparison operators mix measures and attributes.
        Goal: Check Result.
        """
        code = "GL_56_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_56_3(self):
        """
        Status: OK
        Expression: A:= BOP > DS_1;
        Description: datasetEvaluation.
        Git Issue: GL_56-Comparison operators mix measures and attributes.
        Goal: Check Result.
        """
        code = "GL_56_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_56_4(self):
        """
        Status: BUG the attributes should not appear in the result.
        Expression: A:= BOP[calc Me_1 := OBS_VALUE > OBS_VALUE];
        Description: Comparison component level, componentEvaluation.
            This works at component level, so we can keep the
            attributes on the dataset. But read RM lines(1191-1195) says
            in different way, but do not care about this, calc has a specific
            behaviour, so this is the good result.
            Other bug related is the difference between semantic and
            evaluate on a Me_1 isNull = None on semantic result.
        Git Issue: GL_56-Comparison operators mix measures and attributes.
        Goal: Check Result.
        """
        code = "GL_56_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_86(self):
        """
        Status: OK.
        Git Branch: fix-86-comp-scalar
        Goal: Check Exception.
        """
        code = "GL_86"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_88_1(self):
        """
        Status: OK
        Expression: CC0010_DS := dsPrep.ENTTY
                                [calc CC0010:= CNTRY in AnaCreditCountries]
                                [keep CC0010];
        Description: If there is a null, the result is null.

        Git Issue: bug-88-treatment-of-null-with-in-operation-not-correct.
        Goal: Check Result.
        """
        code = "GL_88_1"
        number_inputs = 1
        vd_names = ["GL_88-1"]
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )
        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )

    def test_GL_88_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_5:= Me_2 in { "0", "3", "6", "12" } ]
        Description: If there is a null, the result is null.
        Git Issue: bug-88-treatment-of-null-with-in-operation-not-correct.
        Goal: Check Exception.
        """
        code = "GL_88_2"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_88_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_5:= Me_2 in { 0, 3, 6, 12 } ]
        Description: If there is a null, the result is null.
        Git Issue: bug-88-treatment-of-null-with-in-operation-not-correct.
        Goal: Check Result.
        """
        code = "GL_88_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_88_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_5:= Me_4 in { "2000-01-01/2009-12-31", "2001-01-01/2001-12-31" } ]
        Description: If there is a null, the result is null.
        Git Issue: bug-88-treatment-of-null-with-in-operation-not-correct.
        Goal: Check Exception.
        """
        code = "GL_88_4"
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_169_1(self):
        """
        Status: BUG
        Expression: DS_r:=match_characters(DS_1, "[A-Z]{2}[0-9]{3}");
        Description: match_characters at Dataset level.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_2(self):
        """
        Status: BUG
        Expression: DS_r := DS_1 [ calc m1 := match_characters(DS_1#Me_1, "[A-Z]{2}[0-9]{3}") ];
        Description: match_characters at Component level.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_3(self):
        """
        Status: BUG
        Expression: DS_r := DS_1 [ calc m1 := match_characters(DS_1#Me_1, "[A-Z]{2}[0-9]{3}") ];
        Description: match_characters at Component level.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_4(self):
        """
        Status: BUG
        Expression: DS_r := DS_1 [ calc m1 := match_characters(DS_1#Me_1, DS_1#Me_2) ];
        Description: Two components on match characters.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_5(self):
        """
        Status:
        Expression: DS_r := match_characters(DS_1, "[A-Z]{2}[0-9]{3}");
        Description: More than one measure
        Git Issue: feat-169-implement-match.
        Goal: Check Exception.
        """
        code = "GL_169_5"
        number_inputs = 1
        message = "1-1-1-4"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_169_6(self):
        """
        Status:
        Expression: DS_r := match_characters(DS_1, "[A-Z]{2}[0-9]{3}");
        Description: With Attributes at dataset level.
        Git Issue: feat-169-implement-match.
        Goal: Check Exception.
        """
        code = "GL_169_6"
        number_inputs = 1
        references_names = ["1"]
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_7(self):
        """
        Status:
        Expression: DS_r := DS_1 [ calc m1 := match_characters(DS_1#At_1, "[A-Z]{2}[0-9]{3}") ];
        Description: With Attribute at component level.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_8(self):
        """
        Status:
        Expression: DS_r := match_characters(DS_1, "[A-Z]{2}[0-9]{3}");
        Description: Implicit cast string for number.
        Git Issue: feat-169-implement-match.
        Goal: Check Exception.
        """
        code = "GL_169_8"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_169_9(self):
        """
        Status:
        Expression: DS_r := match_characters(DS_1, "[A-Z]{2}[0-9]{3}");
        Description: Implicit cast string for time, not allowed.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_9"
        number_inputs = 1
        message = "1-1-1-2"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_169_10(self):
        """
        Status:
        Expression: DS_r := match_characters(DS_1, "[A-Z]{2}[0-9]{3}");
        Description: Empty Data points.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_11(self):
        """
        Status: BUG
        Expression: DS_r := DS_1 [ calc m1 := match_characters(DS_1#Me_1, "[A-Z]{2}[0-9]{3}") ];
        Description: Empty Dataset.
        Git Issue: feat-169-implement-match.
        Goal: Check Result.
        """
        code = "GL_169_11"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_169_12(self):
        """
        Status: BUG
        Expression: DS_r := DS_1 [ calc m1 := match_characters(DS_1#Me_1, r"^\\d{4}\\-(0[1-9]|1[012])\\-(0[1-9]|[12][0-9]|3[01])") ];
        Description: Check unicode regex.
        Git Issue: bug-185-match-unicode.
        Goal: Check Result.
        """
        code = "GL_169_12"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_792(self):
        """
        Status: BUG
        Expression: pc22 <- DS_1[filter COUNTRY_INCORPORATION = "CA"]
            [calc PC_VALID := match_characters(POSTAL_CODE, "^((?=[^DdFfIiOoQqUu\\d\\s])...")];
        Description: match_characters patterns using PCRE/Python lookaround assertions
            (a Canadian postal-code rule) are rejected by DuckDB's RE2 engine. They
            must fall back to the Python ``re`` engine and yield the same result.
        Git Issue: #792.
        Goal: Check Result.
        """
        code = "GL_792"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_193_1(self):
        """
        Status: OK
        Expression:
        Description: If then else inside a calc.
        Git Issue: bug-193-evaluate-for-if-then-inside-a-calc.
        Goal: Check Result.
        """
        code = "GL_193_1"
        number_inputs = 3
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # comparission type checking
    def test_GL_165_1(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD#DFLT_STTS<> 14;
        Description: Comparison between date and integer.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_1"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_2(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD#DFLT_STTS<> 14;
        Description: Comparison between string and integer.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_2"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_3(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD#DFLT_STTS<> 14.0;
        Description: Comparison between string and number.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_3"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_4(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD#DFLT_STTS<> 14.05;
        Description: Comparison between string and number.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_4"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_5(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD#DFLT_STTS<> "True";
        Description: Comparison between string and boolean.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Result.
        """
        code = "GL_165_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_165_6(self):
        """
        Status: OK
        Expression: temp := ;
        Description: Comparison between number and string.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_6"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_7(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD[calc Me_2 := DFLT_STTS<> Me_1];
        Description: Comparison between string and number for components.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_7"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_8(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD[calc Me_2 := DFLT_STTS >= Me_1];
        Description: Comparison between string and number for components.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_8"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_165_9(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD[calc Me_2 := DFLT_STTS < Me_1];
        Description: Comparison between integer and number for components.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Result.
        """
        code = "GL_165_9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_165_10(self):
        """
        Status: OK
        Expression: temp := dsPrep.OVR_1_DBTR_ALL_DFLTD[calc Me_2 := DFLT_STTS < Me_1];
        Description: Comparison between boolean and number for components.
        Git Issue: fix-gl-165-force-df-string-type-cast.
        Goal: Check Exception.
        """
        code = "GL_165_10"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_281_1(self):
        """
        Status: OK
        Description: Fix handling of scalar values with null on time types
        Git Branch: cr-281
        Goal: Check Result.
        """
        code = "GH_281_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_281_2(self):
        """
        Status: OK
        Description: Fix handling of scalar values with null on time types
        Git Branch: cr-281
        Goal: Check Result.
        """
        code = "GH_281_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_20(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 > DS_1;
        Description: Time values cannot be ordered, so the semantic analysis rejects the comparison
                     even when the Data Set holds no data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_20"
        number_inputs = 1
        message = "1-1-19-18"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_33(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[filter Me_1 in {1, "a"}];
        Description: A set literal that mixes types is a structure error, where a plain Exception
                     was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_33"
        number_inputs = 1
        message = "1-2-25"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_34(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[filter Me_1 in {1, 1}];
        Description: A set literal that repeats a value is a structure error, where a plain
                     Exception was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_34"
        number_inputs = 1
        message = "1-2-27"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1276_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 = abs ( DS_1 );
        Description: The operand abs ( DS_1 ) of a comparison was built with the structure of
                     the statement's result, so its Measure was named bool_var and the DuckDB
                     engine failed with a BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1276
        Goal: Check Result.
        """
        code = "GH_1276_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1282(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := isnull ( not Me_1 ),
                                        Me_3 := between ( not Me_1 , true , true ),
                                        Me_4 := cast ( not Me_1 , string ) ];
        Description: The DuckDB engine wrote not as NOT x, without parentheses, so it read
                     NOT x IS NULL and NOT x BETWEEN ... as NOT (x IS NULL) and
                     NOT (x BETWEEN ...), which inverted isnull, between and the cast to string
                     of a not.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1282
        Goal: Check Result.
        """
        code = "GH_1282"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class TimeBugs(BugHelper):
    """ """

    classTest = "Bugs.TimeBugs"

    def test_GH_437(self):
        """
        Status: OK
        Description: Referential integrity error with time_agg over Time_Period inside Aggregation.
            The second operation was modifying the results of the first operation.
        Git Branch: fix-issue-437
        Goal: Check that time_agg in group all does not modify previous results.
        """
        code = "GH_437"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_tp": "1995D1"},
        )

    def test_GH_497(self):
        """
        Status: OK
        Description: TimeShift converts annual time periods to int, breaking inner_join.
            After timeshift on annual periods, the join keys became int instead of str,
            causing pd.merge to produce empty results.
        Git Branch: cr-497
        Goal: Check that timeshift on annual periods preserves string format and inner_join works.
        """
        code = "GH_497"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_duckdb_derived_scalar_chain(self):
        """
        Status: OK
        Description: DuckDB transpiler inlined scalar references as their
            transpile-time value. Derived (output-only) scalars have no value
            at transpile time, so chained scalar arithmetic such as
            ``b := a + 1; c := b + 1;`` produced ``NULL + 1 = NULL`` and the
            chain collapsed to NULL.
        Goal: pandas and DuckDB return the same scalar values across a chain
            that mixes input scalars, derived scalars and persistent assigns.
        """
        script = """
        a := sc1 + 1;
        b := a + 1;
        c <- b;
        """
        scalar_values = {"sc1": 5}
        for engine in (False, True):
            result = run(
                script=script,
                data_structures={"datasets": [], "scalars": [{"name": "sc1", "type": "Integer"}]},
                datapoints={},
                scalar_values=scalar_values,
                return_only_persistent=False,
                use_duckdb=engine,
            )
            assert result["a"].value == 6, f"engine={engine}"
            assert result["b"].value == 7, f"engine={engine}"
            assert result["c"].value == 7, f"engine={engine}"

    def test_GH_935_1(self):
        """
        Status: OK
        Description: flow_to_stock accumulates with SUM() OVER (), and DuckDb
                     widens SUM() over an integer to HUGEINT. HUGEINT has no pandas
                     integer counterpart, so an Integer measure came back as a float
                     (1.0, 3.0, 6.0) even though the component is still declared
                     Integer. stock_to_flow was unaffected because it subtracts.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/935
        Goal: Check Result.
        """
        code = "GH_935_1"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_949_1(self):
        """
        Status: OK
        Description: fill_time_series only adds Data Points, so an operand Data
                     Point whose date misses the generated grid still reaches the
                     result. The grid runs from 2020-01-30 one month at a time, so it
                     holds 02-29 and 03-30 and the operand's 2020-03-31 falls outside
                     it. The original dates for this test no longer leave anything off
                     the grid, because #961 stopped the grid drifting off period ends.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/949
        Goal: Check Result.
        """
        code = "GH_949_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_962_1(self):
        """
        Status: OK
        Description: datediff takes a time operand on either side, so a Date and a
                     Time_Period can be mixed. The manual lists
                     ``datediff (2021Q2, 2021-11-04)`` as a valid syntax, states no
                     Additional Constraints, and mixes the two in its own example.
                     SimpleBinaryTime rejected the pair outright even though py_op
                     already resolved a Time_Period to the last day of its period.
                     The rows and the expected 546, 639 and 90 are the manual's.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/962
        Goal: Check Result.
        """
        code = "GH_962_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_1(self):
        """
        Status: OK
        Description: the time operators accept an identifier of type Time
                     (TimeInterval), not only Date and Time_Period. The DuckDB
                     backend used to reject those Data Sets before generating any
                     SQL, so flow_to_stock, stock_to_flow, timeshift and
                     fill_time_series all raised a TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_2(self):
        """
        Status: OK
        Description: on a Date identifier, timeshift adds calendar periods without
                     snapping a month end to the month end of the target month, and
                     only number measures accumulate, so a number Attribute passes
                     through flow_to_stock and stock_to_flow untouched.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_2"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_3(self):
        """
        Status: OK
        Description: fill_time_series adds every Data Point the grid expects even
                     when the Data Set has Attribute components. The added Data
                     Points carry no values, so their Attributes are null too.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_3"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_4(self):
        """
        Status: OK
        Description: fill_time_series with the single limits method works whatever
                     the type of the other identifiers, which used to be looked up
                     by their string representation.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_5(self):
        """
        Status: OK
        Description: a Data Set may hold a single series, with the time identifier
                     as its only identifier. The time operators then have nothing
                     to group by, which used to raise a Pandas ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_5"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_6(self):
        """
        Status: OK
        Description: same single series as GH_918_5, on a Date identifier.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_6"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_7(self):
        """
        Status: OK
        Description: fill_time_series only adds Data Points, so intervals that
                     overlap keep their own values. Their two endpoint grids come
                     out different lengths, which used to rewrite the operand's
                     intervals into ones it never held.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_7"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_918_8(self):
        """
        Status: OK
        Description: a Time value may carry a time component, which the operators
                     read past when they measure an interval's duration and keep in
                     the intervals they add.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/918
        Goal: Check Result.
        """
        code = "GH_918_8"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_961_1(self):
        """
        Status: OK
        Description: the expected grid is measured from the lower limit rather than from
                     the previous date. A month-based step out of a period end clamps in
                     the shorter month, and carrying that clamped day forward drifted the
                     grid off the period ends, so 2020-01-31 reached 2020-03-29 instead
                     of 2020-03-31. The manual identifies a date-typed period by its last
                     day. DS_3 keeps a mid-month series covered, since it must stay where
                     it was.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/961
        Goal: Check Result.
        """
        code = "GH_961_1"
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_936_1(self):
        """
        Status: OK
        Description: pins the period inferred for a Date series by timeshift. Every
                     point of DS_1 falls on the first of a month and every gap is a
                     whole number of months, so the period is a month and shifting by
                     -1 moves back one month. Before #934 the day-count heuristic read
                     these gaps as days, because none of them is an exact multiple of
                     365, and shifted by a single day instead. DS_2 keeps a day-long
                     period covered, so a future change cannot trade one for the other.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/936
        Goal: Check Result.
        """
        code = "GH_936_1"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_960_1(self):
        """
        Status: OK
        Description: timeshift reads the period of the reference time Identifier from
                     each time series. DS_1 interleaves two monthly series, whose
                     combined column shows two-week gaps and used to move the Data Set
                     by a day. DS_2 adds a series of a single Data Point, which shows
                     no gap and takes the period the rest of the Data Set agrees on.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/960
        Goal: Check Result.
        """
        code = "GH_960_1"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1032_1(self):
        """
        Status: OK
        Expression: DS_r1 := dateadd(DS_1, 2, "A");
                    DS_r2 := timeshift(DS_1, 2);
        Description: dateadd on a Time_Period Identifier retyped the operand's
                     shared Component to Date in place, so every structure
                     sharing it (the operand and the sibling results) reported a
                     Date Identifier holding period strings, the Time_Period
                     output formatting was skipped, and reloading the operand
                     from disk crashed the pandas engine
                     (ValueError: Invalid isoformat string).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1032
        Goal: Check Result.
        """
        code = "GH_1032_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_960_2(self):
        """
        Status: OK
        Description: the reference time Identifier must be periodical, so a Data Set
                     holding a daily series beside a monthly one has no period to shift
                     by and timeshift rejects it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/960
        Goal: Check Exception.
        """
        code = "GH_960_2"
        with pytest.raises(RunTimeError, match="2-1-19-23"):
            run(
                script=self.LoadVTL(code),
                data_structures=self.filepath_json / f"{code}-1.json",
                datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
                use_duckdb=_use_duckdb_backend(),
            )

    def test_GH_1032_2(self):
        """
        Status: OK
        Expression: DS_r1 := timeshift(DS_1, 2);
                    DS_r2 := dateadd(DS_1, 2, "A");
        Description: with dateadd placed after the timeshift, the shared
                     Component mutation retroactively retyped the already
                     computed timeshift result to Date, so both engines skipped
                     its Time_Period output formatting and reported a Date
                     Identifier holding period strings.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1032
        Goal: Check Result.
        """
        code = "GH_1032_2"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1034(self):
        """
        Status: OK
        Expression: DS_r <- timeshift(DS_1, 1);
        Description: timeshift over a Time (interval) identifier read the first
                     Data Point to infer the shift frequency, so an operand with
                     no Data Points raised IndexError: single positional indexer
                     is out-of-bounds on the Pandas engine, while the DuckDB
                     engine returned the empty dataset.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1034
        Goal: Check Result.
        """
        code = "GH_1034"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_10(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := dateadd(Me_1, 1, "X")];
        Description: An invalid literal period indicator is rejected by the semantic analysis.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_10"
        number_inputs = 1
        message = "1-1-19-17"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_11(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := dateadd(Me_1, 1, sc_1)];
        Description: An invalid period indicator given as a scalar value is a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_11"
        number_inputs = 1
        message = "2-1-19-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": "X"}
        )

    def test_GH_1192_18(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[aggr Me_2 := sum(Me_1) group all time_agg(sc_1)];
        Description: The scalar that gives time_agg its period holds no value: a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_18"
        number_inputs = 1
        message = "2-1-19-24"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": None}
        )

    def test_GH_1192_21(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := datediff(Me_1, Me_1)];
        Description: datediff reads dates and time periods, so a Time value is a runtime error on
                     both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_21"
        number_inputs = 1
        message = "2-1-19-25"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_24(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := datediff(Me_1, sc_1)];
        Description: A Time value given as a scalar is a runtime error on both engines, where
                     DuckDB read the first date of the interval.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_24"
        number_inputs = 1
        message = "2-1-19-25"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            scalars={"sc_1": "2020-01-01/2020-12-31"},
        )

    def test_GH_1192_25(self):
        """
        Status: OK
        Expression: DS_r <- inner_join(DS_1 as d1, DS_2 as d2
                        calc Me_3 := datediff(d1#Me_1, d2#Me_2));
        Description: A Time value read through a join alias is a runtime error on both
                     engines, where DuckDB failed to bind the SQL.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_25"
        number_inputs = 2
        message = "2-1-19-25"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_26(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := datediff(Me_1, sc_1)];
        Description: A Time scalar without value gives a null on both engines, where DuckDB
                     failed to bind the SQL.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_26"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_1": None},
        )

    def test_GH_1276_2(self):
        """
        Status: OK
        Expression: DS_r1 <- cast ( time_agg ( "A" , _ , DS_1 ) , string );
                    DS_r2 <- min ( time_agg ( "A" , _ , DS_1 ) );
                    DS_r3 <- inner_join ( time_agg ( "A" , _ , DS_1 ) as a , DS_2 as b );
        Description: A time_agg on a Data Set is the operand of another operator. The DuckDB
                     engine wrote its query inside the expression of cast and min and failed
                     with a ParserException. As on the pandas engine, the time_agg drops the
                     Attribute At_1 of DS_1 that is not viral, so the join takes At_1 from DS_2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1276
        Goal: Check Result.
        """
        code = "GH_1276_2"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1309(self):
        """
        Status: OK
        Expression: DS_r1 <- DS_1 [calc Me_2 := time_agg("A", Id_1)]
                                  [calc Me_3 := cast(Me_2, string)] [filter Me_1 > 0];
                    DS_r2 <- DS_1 [calc Me_2 := time_agg("A", Id_1)]
                                  [calc Me_2 := cast(Me_2, string)];
        Description: On the DuckDB engine, a calc read the time_agg result of an earlier calc
                     as a Number when it typed the operand without the statement's result
                     (DS_r1 raised 2-1-5-1), and as the String the statement's result has after
                     the last calc (DS_r2 gave 2004A instead of 2004).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1309
        Goal: Check Result.
        """
        code = "GH_1309"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class SetBugs(BugHelper):
    """ """

    classTest = "Bugs.SetBugs"

    def test_GL_20_1(self):
        """
        Status: OK
        Expression: DS_r := union(DS_1,DS_2);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_2(self):
        """
        Status: OK
        Expression: SCRTS_RPR_1 := union(
            SCRTY_SHRT_PSTN_FLTRD_implct, G_OWND_SCRTS_INFO.OWND_SCRTS_SUB);
        Description: Example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_2"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_3(self):
        """
        Status: OK
        Expression: SCRTS_RPR_2 := union(
            G_OWND_SCRTS_INFO.OWND_SCRTS_SUB,SCRTY_SHRT_PSTN_FLTRD_implct);
        Description: Example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_4(self):
        """
        Status: OK
        Expression: DS_r := union(DS_1,DS_2);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Result.
        """
        code = "GL_20_4"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_5(self):
        """
        Status: OK
        Expression: DS_r := union(DS_2, DS_1);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Result.
        """
        code = "GL_20_5"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_6(self):
        """
        Status: OK
        Expression: DS_r := union(DS_1, DS_2, DS_3);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_6"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_7(self):
        """
        Status: OK
        Expression: DS_r := union(DS_1, DS_2, DS_3);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Result.
        """
        code = "GL_20_7"
        number_inputs = 3
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_8(self):
        """
        Status: OK
        Expression: DS_r := intersect(DS_1, DS_2);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_8"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_9(self):
        """
        Status: OK
        Expression: DS_r := setdiff(DS_1, DS_2);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_9"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_20_10(self):
        """
        Status: OK
        Expression: DS_r := setdiff(DS_1, DS_2);
        Description: Basic example with different isNull in measures.
        Git Branch: GL_20-improve-at-nullable-calculations.
        Goal: Check Semantic Result.
        """
        code = "GL_20_10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_796(self):
        """
        Expression: res <- union(count(cd # ID group by CI), count(cd group by CI));
        Description: ``Membership.validate`` added the synthetic ``str_var`` measure
            (from ``cd # ID``) directly into the shared operand ``cd``. The mutation
            leaked into the later ``count(cd ...)`` over the multi-measure ``cd``,
            so the DuckDB transpiler emitted ``COUNT(CASE WHEN ... "str_var" ...)``
            against a table without that column, raising ``BinderException``. The
            union of the two count branches surfaced it.
        Git Issue: GH_796.
        Goal: Check Result (``cd`` keeps only its real measures; the union runs).
        """
        code = "GH_796"
        number_inputs = 1
        references_names = ["cd", "a", "c", "res"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1057_1(self):
        """
        Status: OK
        Description: symdiff matches the Measures of the second operand by name, so a
                     Data Point taken from it keeps its values on the right Measures
                     even when the second operand declares the same components in a
                     different order. The DuckDb engine glued the two ANTI JOIN branches
                     with a positional UNION ALL, silently swapping the values.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1057
        Goal: Check Result.
        """
        code = "GH_1057_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1057_2(self):
        """
        Status: OK
        Description: intersect accepts more than two operands and keeps only the Data
                     Points present in every one of them. The DuckDb engine used the
                     first two operands and silently ignored the rest.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1057
        Goal: Check Result.
        """
        code = "GH_1057_2"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1057_3(self):
        """
        Status: OK
        Description: the set operators promote the result structure (data type and
                     nullability) across their operands, but must not write the
                     promotion back onto the first operand. DS_1 declares Me_1 as a
                     non-nullable Integer and setdiff promoted it in place to a nullable
                     Number, so DS_s <- DS_1 carried the poisoned structure.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1057
        Goal: Check Result.
        """
        code = "GH_1057_3"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1057_4(self):
        """
        Status: OK
        Description: a component present in one operand of a set operator but missing
                     in the other raised a bare Exception instead of a SemanticError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1057
        Goal: Check Exception.
        """
        code = "GH_1057_4"
        number_inputs = 2
        message = "1-1-17-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1057_5(self):
        """
        Status: OK
        Description: a component holding different roles across the operands of a set
                     operator raised a bare Exception instead of a SemanticError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1057
        Goal: Check Exception.
        """
        code = "GH_1057_5"
        number_inputs = 2
        message = "1-1-17-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_35(self):
        """
        Status: OK
        Expression: DS_r <- union(DS_1, 1);
        Description: A set operator over a Scalar is a structure error, where a ValueError was
                     raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_35"
        number_inputs = 1
        message = "1-2-23"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1225_4(self):
        """
        Expression: DS_r1 <- union(symdiff(DS_1, DS_3), DS_2);
                    DS_r2 <- setdiff(f_hierarchy(DS_1), DS_2);
                    DS_r3 <- intersect(check_datapoint(DS_1[filter Id_1 > 0], DPR_1 all),
                                       check_datapoint(DS_2, DPR_1 all));
        Description: The first operand of each set operator is an inline operator whose query
            starts with WITH: symdiff, a user defined operator returning a hierarchy and
            check_datapoint over an expression with two rules.
        Git Issue: GH_1225.
        Goal: Check the set operators read any inline operand built from CTEs.
        """
        code = "GH_1225_4"
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1225_5(self):
        """
        Expression: DS_r <- union(fill_time_series(DS_1, all), DS_2);
        Description: The first operand of union is an inline fill_time_series.
        Git Issue: GH_1225.
        Goal: Check the set operators read an inline fill_time_series.
        """
        code = "GH_1225_5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1283_1(self):
        """
        Expression: DS_r1 <- union((sum(DS_1 group by Id_1) / 2)[filter Me_1 > 2],
                                   (sum(DS_1 group by Id_1) / 2)[filter Me_1 <= 2]);
                    DS_r2 <- union((sum(DS_1 group by Id_1) / 2)[filter Me_1 > 5],
                                   (sum(DS_1 group by Id_1) / 2)[filter Me_1 > 1]
                                       [calc Me_1 := Me_1 + 100],
                                   (sum(DS_1 group by Id_1) / 2)[calc Me_1 := Me_1 + 1000]);
        Description: The operands repeat the same division, which DuckDB computes in a
            MATERIALIZED CTE, and the union kept the first Data Point of each Identifier with a
            row-number filter over the UNION ALL of the operands, where DuckDB 1.5 returned
            0.0 or other wrong values. Each operand gives the Data Points whose Identifiers are
            in no operand before it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1283
        Goal: Check Result.
        """
        code = "GH_1283_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1304_1(self):
        """
        Expression: DS_r <- union(DS_1 [filter Id_1 <= 1] [calc Me_1 := Me_1 + 1],
                                  DS_1 [filter Id_1 <= 2] [calc Me_1 := Me_1 + 2],
                                  ...
                                  DS_1 [filter Id_1 <= 10] [calc Me_1 := Me_1 + 10]);
        Description: DuckDB wrote the union with an ANTI JOIN of each operand against every
            operand before it, so its SQL and planning grew with the square of the number of
            operands. Each operand repeats the Data Points of the operands before it with other
            values and adds one Identifier, so each Data Point has to come from the left-most
            operand that has it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1304
        Goal: Check Result.
        """
        code = "GH_1304_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1304_4(self):
        """
        Expression: DS_r1 <- case when DS_1 > 4 then (case when DS_2 > 3 then DS_2 else DS_1)
                        else DS_1;
                    DS_r2 <- if DS_1 > 4 then (if DS_3 then DS_2 * 10 else DS_1 * 10) else DS_1;
        Description: A Data Set case or if inside a branch of another one. On the pandas
            engine, the inner condition was cut by the row labels of the Data Points of the
            outer branch, which belong to the frame of another Data Set: the condition on DS_2
            raised a KeyError (DS_r1), and DS_3, which lists its Data Points in another order,
            cut the inner branches to other Data Points and lost Id_1 1 (DS_r2). In both, Id_1
            3 is left out, as the inner case or if has no Data Point 3.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1304
        Goal: Check Result.
        """
        code = "GH_1304_4"
        number_inputs = 3
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1304_5(self):
        """
        Expression: DS_r1 <- case when DS_2 > 3 then DS_2 when DS_3 then DS_1 else DS_1;
                    DS_r2 <- case when DS_1 > 6 then DS_1 when DS_2 > 3 then DS_2
                        else DS_1 * 10;
        Description: The first condition of a Data Set case gives the Data Points. On the
            pandas engine, a later condition added its own true Data Points (DS_r1 Id_1 3),
            and the else operand only had the Data Points of the last condition, so the Data
            Points that the last condition lacks were lost (DS_r2 Id_1 4).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1304
        Goal: Check Result.
        """
        code = "GH_1304_5"
        number_inputs = 3
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1304_7(self):
        """
        Expression: DS_r1 <- (case when DS_1 > 0 then 1.0 else DS_2)[calc Me_2 := Me_1 * 2];
                    DS_r2 <- (case when DS_1 > 0 then 1.0 else DS_2) + DS_1;
        Description: A Data Set case whose first then-operand is a scalar, as the operand of a
            clause or of another operator. On the DuckDB engine, the structure of a case came
            from its first then-operand only and its type from its then-operands only, so the
            calc raised an AttributeError (DS_r1) and the case was taken for a scalar (DS_r2).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1304
        Goal: Check Result.
        """
        code = "GH_1304_7"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1276_7(self):
        """
        Status: OK
        Expression: DS_r <- sum ( union ( DS_1 + DS_1 , DS_1 + DS_1 ) group by Id_1 );
        Description: A union used as an operand took the columns of the statement's result, so
                     inside the aggregation it lost Id_2 and the sum was silently wrong on the
                     DuckDB engine, and its computed operands made DuckDB fail with an internal
                     error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1276
        Goal: Check Result.
        """
        code = "GH_1276_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AggregationBugs(BugHelper):
    """ """

    classTest = "Bugs.AggregationBugs"

    def test_GL_11(self):
        """ """
        code = "GL_11"
        number_inputs = 33
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_85(self):
        """
        Description:
        Issue: GL_85.
        Git Branch: bug-85-error-in-count-without-datapoints.
        Goal: Check interpreter result.
        """
        code = "GL_85"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_243_1(self):
        """
        Description: count in aggr with others operators
        Issue: GL_243_1.
        Git Branch: fix-243-Anamart-count.
        Goal: Check result.
        """
        code = "GL_243_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_243_2(self):
        """
        Description: count in aggr with others operators
        Issue: GL_243.
        Git Branch: fix-243-Anamart-count.
        Goal: Check result.
        """
        code = "GL_243_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_243_3(self):
        """
        Description: count in aggr with others operators
        Issue: GL_243.
        Git Branch: fix-243-Anamart-count.
        Goal: Check result.
        """
        code = "GL_243_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_254_1(self):
        """
        Description: count in aggr after filter
        Issue: GL_254.
        Git Branch: fix-254-aggr-after filter.
        Goal: Check result.
        """
        code = "GL_254_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_1(self):
        """
        Description: aggr(sum, avg, var_pop...) over a component, other components not numeric
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_2(self):
        """
        Description: aggr(sum, avg, var_pop...) over two components, other components not numeric
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_3(self):
        """
        Description: aggr(sum, avg, var_pop...) over two components, other components not numeric
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_4(self):
        """
        Description: aggr(sum, avg, var_pop...) over two components, other components not numeric
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_5(self):
        """
        Description: DOUBT about nested Operator
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_6(self):
        """
        Description:
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_7(self):
        """
        Description: with group except
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_259_8(self):
        """
        Description: several statements
        Issue: GL_259.
        Git Branch: fix-259-aggr-after-aggr.
        Goal: Check result.
        """
        code = "GL_259_8"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_270_1(self):
        """
        Description: basic example for min if then else
        Issue: GL_270.
        Git Branch: fix-270-min-with-if-then-else.
        Goal: Check result.
        """
        code = "GL_270_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_270_2(self):
        """
        Description: example with more elements in aggr
        Issue: GL_270.
        Git Branch: fix-270-min-with-if-then-else.
        Goal: Check result.
        """
        code = "GL_270_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_270_3(self):
        """
        Description: basic example for min if then else
        Issue: Quick fix over GL_270.
        Git Branch: test-270-empty-dataset
        Goal: Check result.
        """
        code = "GL_270_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_312(self):
        """
        Description: Fix on Aggr after filter
        Issue: #313.
        Git Branch: fix-313-aggr-order
        Goal: Check result.
        """
        code = "GL_312"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_313(self):
        """
        Description: Fix on Aggr order (isOneLineNested)
        Issue: #313.
        Git Branch: fix-313-aggr-order
        Goal: Check result.
        """
        code = "GL_313"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_329(self):
        """
        Description: Fix on Aggr with if-then-else
        Issue: #330.
        Git Branch: fix-337-AST-aggr
        Goal: Check result.
        """
        code = "GL_329"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_330(self):
        """
        Description: Fix on Aggr
        Issue: #330.
        Git Branch: fix-337-AST-aggr
        Goal: Check result.
        """
        code = "GL_330"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_327(self):
        """
        Description: Fix on Aggr
        Issue: #327.
        Git Branch: fix-327-aggr-null
        Goal: Check result.
        """
        code = "GL_327"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_410(self):
        """
        Description: Aggr inside a calc error (AggregationComp)
        Issue: #410.
        Git Branch: fix-410-aggr-calc
        Goal: Check result.
        """

        code = "GL_410"
        number_inputs = 1
        message = "1-2-14"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_959_1(self):
        """
        Status: OK
        Description: count over a Data Set holding a single Measure counts that Measure
                     and renames it to int_var, so it reports the same number as count
                     over that Measure as a Component. Only count() with no operand
                     reports the number of Data Points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/959
        Goal: Check Result.
        """
        code = "GH_959_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_959_2(self):
        """
        Status: OK
        Description: count over a Data Set holding several Measures reports the Data
                     Points holding at least one non-null Measure, as the single
                     Measure int_var (issue #1049). A group whose Measures are all
                     null counts 0 and not null. The aggr Component form still counts
                     each Measure on its own.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/959
        Goal: Check Result.
        """
        code = "GH_959_2"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_996_1(self):
        """
        Status: OK
        Description: the count cases reported at once. A Data Set operand counts the
                     Data Points holding at least one non-null Measure, as the single
                     Measure int_var (DS_r1, DS_r2, issue #1049); a Component operand
                     counts its values and reports 0 where a group holds none (DS_r3);
                     count() with no operand counts the Data Points (DS_r4);
                     an analytic count reports 0 for a partition of nulls (DS_r5) and,
                     with an order clause and no window clause, counts up to the
                     current Data Point (DS_r6, issue #1249); a having condition keeps
                     the viral attributes (DS_r7);
                     an analytic count over a String Measure counts its values
                     (DS_r8); and count() reports the Data Points of a Data Set that
                     is not grouped (DS_r9).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/996
        Goal: Check Result.
        """
        code = "GH_996_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_978_1(self):
        """
        Status: OK
        Description: a clause reads its operand as the operand actually comes out. An
                     operator applied over the Measures leaves the plain Attributes out
                     of its query while still reporting them in its structure, so naming
                     the Components one by one asked DuckDb for a column the subquery
                     never selected and the binder rejected the statement. DS_r3 keeps a
                     calc overwriting an existing Measure covered.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/978
        Goal: Check Result.
        """
        code = "GH_978_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_988_1(self):
        """
        Status: OK
        Description: an operator nested inside another names its Measures for itself.
                     Only the one producing the assignment's result takes the name the
                     target declares, so trim inside length used to rename the Measure
                     to length's int_var and the length around it then looked for a
                     column that was no longer there. DS_r3 keeps a lone operator
                     covered, where the target's name does apply.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/988
        Goal: Check Result.
        """
        code = "GH_988_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_981_1(self):
        """
        Status: OK
        Description: ceil, floor and the parameterless round and trunc give an Integer
                     Measure. DuckDb hands back the type it was given, so a Number
                     Measure stayed a float while the Component said Integer. round and
                     trunc with a number of digits keep giving a Number, which DS_r4
                     covers.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/981
        Goal: Check Result.
        """
        code = "GH_981_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_937_1(self):
        """
        Status: OK
        Description: count only reports the number of Data Points, so it does not
                     aggregate the Measures it is given: the reference manual types its
                     operand as a plain dataset, states no Additional Constraints, and
                     counts a String Measure in its own example. A Time Measure was
                     rejected anyway, because the guard against aggregating one ran
                     before count replaced the Measures with int_var.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/937
        Goal: Check Result.
        """
        code = "GH_937_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1049_1(self):
        """
        Status: OK
        Description: count over a Data Set returns the single Integer Measure int_var,
                     whatever the number of Measures of the operand. A Data Point whose
                     Measures are all null is ignored; one holding at least one non-null
                     Measure is counted (DS_r1, DS_r2). The Component forms keep their
                     behaviour: count over a Component counts its non-null values
                     (DS_r3) and count() with no operand counts the Data Points (DS_r4).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1049
        Goal: Check Result.
        """
        code = "GH_1049_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1225_7(self):
        """
        Expression: DS_r1 <- count((DS_1 / 2) group by Id_1);
                    DS_r2 <- count(DS_1 + DS_2 group by Id_1);
        Description: The operand of count is an expression, whose measure keeps its name
            while count reports int_var.
        Git Issue: GH_1225.
        Goal: Check count reads the measures of an expression operand on DuckDB.
        """
        code = "GH_1225_7"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_VTLEN_503(self):
        """ """
        code = "VTLEN_503"
        number_inputs = 1
        message = "1-2-6"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_19(self):
        """
        Description: Check 3VL inside define datapoint ruleset.
        Issue: #19.
        Git Branch: bug-19-and-or-unexpected-resultshen-then.
        Goal: Interpreter SUCCESS.
        """
        code = "GL_19"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_53(self):
        """
        Description: Check 3VL inside define datapoint ruleset.
        Issue: #53.
        Git Branch: bug-53-nulls-behavior-at-when-then-rule.
        Goal: Interpreter SUCCESS.
        """
        code = "GL_53"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_13(self):
        """
        Description: Check Exists in when left datapoint are null.
        Issue: #13.
        Git Branch: bug-13-nullable-field-with-null-value.
        Goal: Interpreter SUCCESS.
        """
        code = "GL_13"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_117_1(self):
        """
        Description: .
        Issue: #117.
        Git Branch: bug-117-null-value-for-then-condition-in-datapoint-ruleset
        Goal: Check result.
        """
        code = "GL_117_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_117_2(self):
        """
        Description: illustrative example, output parameter=invalid.
        Issue: #117.
        Git Branch: bug-117-null-value-for-then-condition-in-datapoint-ruleset
        Goal: Check result.
        """
        code = "GL_117_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_117_3(self):
        """
        Description: illustrative example, output parameter=all.
        Issue: #117.
        Git Branch: bug-117-null-value-for-then-condition-in-datapoint-ruleset
        Goal: Check result.
        """
        code = "GL_117_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_117_4(self):
        """
        Description: illustrative example, output parameter=all_measures.
        Issue: #117.
        Git Branch: bug-117-null-value-for-then-condition-in-datapoint-ruleset
        Goal: Check result.
        """
        code = "GL_117_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_443_1(self):
        """ """
        code = "GL_443_1"
        number_inputs = 1
        vd_names = ["GL_443_1"]
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_GL_443_2(self):
        """ """
        code = "GL_443_2"
        number_inputs = 1
        vd_names = ["GL_443_2"]
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_GL_443_3(self):
        """ """
        code = "GL_443_3"
        number_inputs = 1
        vd_names = ["GL_443_3"]
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_GH_1192_32(self):
        """
        Status: OK
        Expression: define datapoint ruleset dpr1 ... (twice); DS_r <- check_datapoint(DS_1, dpr1);
        Description: A datapoint ruleset defined twice is a structure error, where a ValueError
                     was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_32"
        number_inputs = 1
        message = "1-2-24"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_36(self):
        """
        Status: OK
        Expression: DS_r <- check(1 > 0);
        Description: check over a Scalar is a structure error, where a ValueError was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_36"
        number_inputs = 1
        message = "1-2-23"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )


class ConditionalBugs(BugHelper):
    """ """

    classTest = "Bugs.ConditionalOperatorsTest"

    def test_VTLEN_476(self):
        """ """
        code = "VTLEN_476"
        number_inputs = 3
        vd_names = []
        references_names = [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
        ]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )

    def test_VTLEN_573(self):
        """ """
        code = "VTLEN_573"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_44(self):
        """
        Status: OK
        Description: Nvl check types
        Git Branch: fix-44-nvl.
        Goal: Check Result.
        """
        code = "GL_44"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_59_1(self):
        """
        Status: OK
        Description: If then not working for null values
                    (on a user define operator in this case).
        Git Branch: bug-59-if-then-not-working-properly-inside-a-udo.
        Goal: Check Result.
        """
        code = "GL_59_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_59_2(self):
        """
        Status: OK
        Description: If then not working for null values
                    (on a user define operator in this case).
        Git Branch: bug-59-if-then-not-working-properly-inside-a-udo.
        Goal: Check Result.
        """
        code = "GL_59_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_59_3(self):
        """
        Status: OK
        Description: If then not working for null values
                    (on a user define operator in this case).
        Git Branch: bug-59-if-then-not-working-properly-inside-a-udo.
        Goal: Check Result.
        """
        code = "GL_59_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_191_1(self):
        """
        Status: OK
        Description: nvl operators with attributes.
        Git Branch: bug-191-evaluate-review-on-nvl-operator.
        Goal: Check Result.
        """
        code = "GL_191_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_191_2(self):
        """
        Status: OK
        Description: nvl operators with attributes without datapoints.
        Git Branch: bug-191-evaluate-review-on-nvl-operator.
        Goal: Check Result.
        """
        code = "GL_191_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_195_1(self):
        """
        Status: OK
        Description: nvl operators with attributes without datapoints.
        Git Branch: bug-191-evaluate-review-on-nvl-operator.
        Goal: Check Result.
        """
        code = "GL_195_1"
        number_inputs = 1
        references_names = ["1"]
        # message = "2-1-15-6"

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_196_1(self):
        """
        Status: OK
        Description: if-then-else inside a calc identifier.
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Exception.
        """
        code = "GL_196_1"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_196_2(self):
        """
        Status: OK
        Description: if-then-else inside a calc measure but isnull=False.
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Result.
        """
        code = "GL_196_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_196_3(self):
        """
        Status: OK
        Description: if-then-else inside a calc measure but else=null.
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Result.
        """
        code = "GL_196_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_196_4(self):
        """
        Status: OK
        Description: calc identifier whose else branch is null, so the expression does
                     produce nulls and an Identifier cannot hold them. Checked through
                     run(), as that is decided on the data (issue #1010). The script is
                     also invalid for a second reason, it ends with a keep of that same
                     Identifier, so the DuckDB engine reports that one first.
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Exception.
        """
        self._assert_null_identifier_rejected(
            "GL_196_4", "comp_cred.CD0060_P", expected=("1-1-1-16", "1-1-6-2")
        )

    def test_GL_196_5(self):
        """
        Status: OK
        Description: if-then-else inside a calc measure but else=null.
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Result.
        """
        code = "GL_196_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_196_6(self):
        """
        Status: OK
        Description: calc identifier that returns the operand itself when it is null, so
                     the expression does produce nulls. Checked through run(), as that is
                     decided on the data (issue #1010).
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Exception.
        """
        self._assert_null_identifier_rejected("GL_196_6", "dsPrep.ENTTY_INSTRMNT_HDQRTR")

    def _assert_null_identifier_rejected(self, code, dataset_name, expected=("1-1-1-16",)):
        """Run a calc identifier whose expression produces nulls and check it is rejected."""
        with pytest.raises(SemanticError) as context:
            run(
                script=self.LoadVTL(code),
                data_structures=self.filepath_json / f"{code}-1.json",
                datapoints={dataset_name: self.filepath_csv / f"{code}-1.csv"},
                use_duckdb=_use_duckdb_backend(),
            )
        assert context.value.args[1] in expected

    def test_GH_1024(self):
        """
        Status: OK
        Expression: DS_r1 := if H_c then H_a else H_b;
                    DS_r2 := case when H_c then H_a else H_b;
        Description: a dataset-level if mutated its condition dataset in place,
                     popping the boolean measure column out of the DataFrame the
                     stored variable shares, so any later statement reading the
                     condition dataset (the case here, but another if fails the
                     same way) raised KeyError: 'c'.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1024
        Goal: Check Result.
        """
        code = "GH_1024"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1225_6(self):
        """
        Expression: DS_r1 <- if DS_1 > 1 then DS_2 else DS_3;
                    DS_r2 <- if DS_1[keep Me_1] > 1 and DS_1 < 5 then DS_2 else DS_3;
                    DS_r3 <- if hierarchy(DS_1, HR_1 rule Id_2) > 4 then DS_2 else DS_3;
                    DS_r4 <- case when DS_1 > 6 then DS_1 when DS_1[keep Me_1] <= 1 then DS_2
                              else DS_3;
        Description: The conditions of the dataset if and case are expressions other than a
            plain dataset or a comparison of two datasets.
        Git Issue: GH_1225.
        Goal: Check the condition is evaluated as a boolean dataset on DuckDB.
        """
        code = "GH_1225_6"
        number_inputs = 3
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1251(self):
        """
        Expression: DS_r1 <- case when DS_1 > 20 then DS_2 when DS_1 > 10 then DS_3 else DS_1;
                    DS_r2 <- case when DS_C1 then DS_2 when DS_C2 then DS_3 else DS_1;
                    DS_r3 <- case when DS_1 > 20 then DS_4 when DS_1 > 10 then DS_3 else DS_2;
                    DS_r4 <- case when DS_1 > 20 then 1000 when DS_1 > 10 then DS_3 else 0;
        Description: case on Data Sets returned the Data Point of the last true condition
            instead of the first (Id_1 3 and 5). The Data Point comes from the operand of
            the first true condition also when its Measure is null (DS_r1 Id_1 5), and it is
            left out when that operand does not have it (DS_r3 Id_1 3). A null condition is
            not true (DS_r2 Id_1 2 and 3), and DS_C2 lists its Data Points in another order.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1251
        Goal: Check Result.
        """
        code = "GH_1251"
        number_inputs = 6
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1276_8(self):
        """
        Status: OK
        Expression: DS_r1 <- abs ( case when DS_1 > 0 then DS_1 else DS_1 * -1 );
                    DS_r2 <- abs ( if DS_1 > 0 then DS_1 * 2 else DS_1 * -1 );
                    DS_r3 <- if DS_1 [ filter Me_1 > 0 or isnull ( Me_1 ) ] > 0
                        then DS_1 [ filter Me_1 > 0 or isnull ( Me_1 ) ] else DS_1;
        Description: On the DuckDB engine, a case (DS_r1) and an if (DS_r2) used as operands
                     took the bool_var Measure of their condition. On the pandas engine, an if
                     whose operands are clauses (DS_r3) lost the Data Points of its true branch.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1276
        Goal: Check Result.
        """
        code = "GH_1276_8"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_2(self):
        """
        Expression: DS_r1 <- if DS_1 < 1 then 0 else DS_1;
                    DS_r3 <- if DS_1 > 2 then DS_1[calc Me_1 := floor(Me_1)] else 0.5;
                    DS_r4 <- case when DS_2 > 1 then DS_2 else 0.5;
                    DS_r5 <- case when DS_2 > 2 then DS_2 else DS_3;
                    DS_r6 <- if DS_4 > 1 then 0.5 else DS_4;
                    DS_r7 <- if DS_5 > 1 then DS_5 else DS_3;
                    DS_r8 <- DS_4 + DS_5;
                    DS_r9 <- DS_6[calc Me_2 := if 1 > 0 then 1 else 2.5];
                    DS_r10 <- DS_6[calc Me_2 := case when 1 > 0 then 1 else 2.5];
                    DS_r11 <- DS_6 * (if 1 >= 0.5 then 0.5 else 2);
                    DS_r12 <- DS_7[calc Me_2 := if Me_1 then "yes" else Me_1];
                    DS_r13 <- if DS_8 then DS_7 else "yes";
                    DS_r14 <- case when DS_8 then DS_7 else "yes";
                    DS_r15 <- (if DS_8 then DS_7 else "yes") || "!";
                    DS_r16 <- inner_join(DS_7 as a, DS_8 as b calc Me_3 := case when b#Me_1
                        then a#Me_1 else "yes" keep Me_3);
        Description: An if or a case whose operands have different types gives their common
            type, at Data Set, Component and scalar level: Number for Integer and Number,
            String for Boolean and String.
            DS_r1, DS_r3: on the pandas engine, both operands of a Data Set if took the dtype of
            the then operand: under the Integer scalar 0, the Number values of the else operand
            were cut (DS_r1 Id_1 3 gave 2.0), and the Number scalar 0.5 raised an ArrowInvalid
            under the Integer then operand (DS_r3).
            DS_r4, DS_r5: the structure of a Data Set case was the one of its first Data Set
            operand, DS_2, so Me_1 was Integer: the DuckDB engine rounded the Number values
            (0.5 to 0, 1.5 to 2) and the pandas engine raised an ArrowInvalid.
            DS_r6 to DS_r8: a Data Set if types its result without retyping its operands. The
            semantic analysis wrote the common type, Number, into the Integer Measures of DS_4
            (DS_r6) and DS_5 (DS_r7), so DS_r8, which comes after them, was Number on both
            engines.
            DS_r9 to DS_r11: a scalar if or case gave the type of the operand that its
            condition selects: the pandas engine typed Me_2 Integer (DS_r9, DS_r10), and the
            DuckDB engine, whose semantic analysis takes the else operand when the value of
            the condition is unknown, typed DS_r11 Integer and rounded 0.5 and 2.5.
            DS_r12 to DS_r16: a Boolean operand with a String operand is promoted to String,
            as cast to string renders it ('true', 'false'). The DuckDB engine cast the String
            operand to Boolean instead, so "yes" came back as 'true'. The pandas engine wrote
            'True' and 'False' (DS_r12, DS_r16) and could not read "yes" as a Boolean (DS_r13
            to DS_r15).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_2"
        number_inputs = 8
        references_names = [str(i) for i in range(1, 17) if i != 2]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_S1_2(self):
        """
        Expression: DS_r <- DS_1[calc Me_2 := if 1 then 2 else 3];
        Description: The condition of a scalar if must be Boolean, as at Component and Data
            Set level. Both engines took the Integer condition, pandas as a true one.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_S1_2"
        number_inputs = 1
        error_code = "1-1-9-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1314_N1(self):
        """
        Expression: DS_r1 <- DS_1[calc Me_2 := if 1 < 3 then 1.5 else null];
                    DS_r2 <- DS_1[calc Me_2 := if 1 > 3 then null else 2];
                    DS_r3 <- DS_1[calc Me_2 := if Me_1 > 1 then sc_1 else 0.0];
                    DS_r4 <- DS_1[calc Me_2 := case when Me_1 > 1 then sc_1 else 0.0];
                    DS_r5 <- (if DS_1 > 1 then DS_1 else null)[calc Me_2 := Me_1];
                    DS_r6 <- (case when DS_1 > 1 then DS_1 else DS_2)[calc Me_2 := Me_1];
        Description: An if or a case can be null where any operand can be.
            DS_r1, DS_r2: a scalar if can be null when either operand can be, as a scalar case.
            It took the nullability of the operand selected: on the pandas engine Me_2 was not
            nullable in both, and on the DuckDB engine, whose semantic analysis takes the else
            operand when the value of the condition is unknown, in DS_r2; in DS_r1 it typed
            Me_2 Null.
            DS_r3 to DS_r6: the operand that can be null is the nullable input scalar sc_1,
            here null, the null literal, or the nullable Measure of DS_2, while the Measure of
            DS_1 is not nullable. Their results were declared not nullable, so the pandas
            engine raised 2-1-1-16 in the calc and the DuckDB engine returned nulls in a
            Component it declared not nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_N1"
        number_inputs = 2
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_1": None},
        )

    def test_GH_1314_I2_1(self):
        """
        Expression: DS_r <- if DS_1 > 1 then DS_1 else DS_2;
        Description: The else operand of a Data Set if must have the Identifiers of the
            condition, as the then operand: Id_1 is a String in DS_2. The check compared the
            then operand twice, so the pandas engine failed in the merge and the DuckDB
            engine ran with an implicit cast.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_I2_1"
        number_inputs = 2
        error_code = "1-1-9-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1314_I1(self):
        """
        Expression: DS_r <- case when DS_1 > 3 then DS_1 when DS_3 then DS_2 else DS_1;
        Description: Every condition of a Data Set case must have the Identifiers of the
            first one, which gives the Data Points: DS_3 is on Id_1 alone, and the operands
            on Id_1 and Id_2. The case passed the semantic analysis, then the pandas engine
            raised a KeyError and the DuckDB engine matched DS_3 on Id_1 alone.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_I1"
        number_inputs = 3
        error_code = "1-1-9-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1314_10(self):
        """
        Expression: DS_r1 <- if DS_1 then DS_2 else 0.5;
                    DS_r2 <- isnull(DS_3);
                    DS_r3 <- if isnull(DS_3) xor DS_3 then DS_4 else 0.5;
                    DS_r4 <- case when isnull(DS_3) xor DS_3 then DS_4 else 0.5;
                    DS_r5 <- if DS_6 < 0.5 then DS_5 - DS_5 else -DS_5;
                    DS_r6 <- case when (case when DS_7 > 0 and DS_8 > 0 then true
                        else DS_7 = DS_9) then DS_9 else DS_9;
                    DS_r7 <- case when DS_7 > 1 then DS_9 when (case when DS_7 > 0 and DS_8 > 0
                        then true else DS_7 = DS_9) then 0 else DS_9 * 10;
        Description: The Data Points of an if or a case.
            DS_r1: a null condition takes the else operand. On the DuckDB engine, a Data Set if
            with a Data Set then operand and a scalar else operand dropped the Data Points
            whose condition is null when the then operand lacked them (Id_1 2).
            DS_r2 to DS_r4: isnull keeps the name of the lone Boolean Measure of DS_3, which it
            does not retype. The DuckDB engine named it bool_var: DS_r2 had no Me_1 column, the
            if read its condition as always true (Id_1 1 gave 10.5) and the case raised an
            IndexError.
            DS_r5: the then operand gives 0.0 on Id_1 3 and the else operand -0.0 on Id_1 1.
            On the pandas engine, the Data Points of both operands were outer merged on all
            their columns, the Measure too, and the merge raised "values should be unique if
            codes is not None" on 0.0 and -0.0.
            DS_r6, DS_r7: DS_7 and DS_8 share no Data Point, so the inner case has none. As a
            condition of the outer case, first (DS_r6) or later (DS_r7), it gives no Data
            Point. On the pandas engine, its Measure was an empty column of object dtype,
            which selected no column instead of no row, and the outer case raised a KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_10"
        number_inputs = 9
        references_names = ["1", "2", "3", "4", "5", "6", "7"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_23(self):
        """
        Expression: DS_r1 <- max(DS_1);
                    DS_r2 <- DS_1[aggr Me_2 := sum(Me_1)];
                    DS_r4 <- DS_2[aggr Me_2 := sum(Me_1)];
                    DS_r5 <- DS_2[aggr Me_2 := count()];
        Description: Without a grouping clause, all the Data Points of the operand make a
            single group, and the result is a single Data Point. Over the null Measures of
            DS_1 it holds a null (DS_r1, DS_r2): the pandas engine dropped it, so the result
            had no Data Point, while the DuckDB engine kept it. DS_2 has no Data Points, so
            the result of an aggr clause has none (DS_r4), where the DuckDB engine gave a null
            Data Point. A count still reports 0 (DS_r5).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_23"
        number_inputs = 2
        references_names = ["1", "2", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_24(self):
        """
        Expression: DS_r1 <- DS_1[aggr Me_2 := avg(1) group by Id_1];
        Description: In an aggr clause, an aggregate operator takes a Component of the operand:
            it cannot be applied to a scalar value (VTL 2.2, Aggregate invocation). Both
            engines raised AttributeError: 'Scalar' object has no attribute 'role'.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_24"
        number_inputs = 1
        error_code = "1-2-17"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1314_13(self):
        """
        Expression: DS_r1 <- DS_1[calc Me_3 := Me_2 = Me_1];
                    DS_r2 <- DS_2 = "true";
                    DS_r3 <- DS_3 = true;
                    DS_r4 <- DS_3 <> DS_2;
                    DS_r5 <- nvl(DS_2, "none");
                    DS_r6 <- DS_2[calc Me_2 := nvl(Me_1, "none"), Me_3 := Me_1 in {"true", "x"},
                        Me_4 := match_characters(Me_1, "t.*")];
                    DS_r7 <- DS_2[calc Me_2 := if Me_1 then true else false]
                        [calc Me_3 := Me_2 = "false"];
                    DS_r8 <- DS_2 in {"true", "x"};
                    DS_r9 <- DS_1[calc Me_3 := if Me_1 then nvl(Me_2, "none")
                        else nvl(Me_1, true)];
                    DS_r10 <- DS_1[calc Me_3 := upper(nvl(Me_1, false)),
                        Me_4 := cast(if Me_1 then Me_2 else "true", boolean)];
        Description: A Boolean promoted to String is "true" or "false", as the VTL 2.2 manual
            casts it.
            DS_r1 to DS_r4: a Boolean compared with a String is compared as text, so "True"
            differs from a true value. The DuckDB engine cast the String to Boolean, so a String
            that is not a Boolean raised a Conversion Error (DS_r1, DS_r3, DS_r4), and the
            pandas engine compared the Boolean as "True" or "False".
            DS_r5 to DS_r8: the same holds in nvl, in, match_characters and the comparisons.
            The pandas engine wrote "True" and "False", and did not convert the numpy Booleans
            of a calc before comparing them with a String (DS_r7). The DuckDB engine cast the
            String to Boolean in nvl and in, and match_characters had no function for a
            Boolean. Both engines failed on in with a String that is not a Boolean.
            DS_r9, DS_r10: nvl has the common type of its operands, as an if or a case: a
            String with a String operand, a Boolean with Boolean operands. In DS_r9 the if is a
            String, so its Boolean operand nvl(Me_1, true) is promoted to String: the pandas
            engine wrote "False" and "True", and the DuckDB engine, which did not type nvl,
            mixed a BOOLEAN and a VARCHAR in its CASE and raised a Binder Error. In DS_r10,
            upper promotes the Boolean nvl to String, and the String if casts to Boolean as a
            String does ("yes" is false). The DuckDB engine had no upper for a BOOLEAN, and cast
            the String if as a value of unknown type, which made "yes" true.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_13"
        number_inputs = 3
        references_names = [str(i) for i in range(1, 11)]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_14(self):
        """
        Expression: DS_r <- (if DS_1#Me_1 > 1 then floor(DS_1) else 0)[filter Me_1 > 0];
        Description: floor drops the Attribute At_1 of DS_1, which is not viral. On the DuckDB
            engine, the structure of floor(DS_1) kept it, so the if used as the operand of the
            filter selected At_1 from it and raised a Binder Error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_14"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_31(self):
        """
        Expression: DS_r1 <- DS_1[calc Me_2 := if Me_1 > 0 then ln(Me_1) else 0];
                    DS_r2 <- DS_1[calc Me_2 := if Me_3 <> "abc" then cast(Me_3, integer)
                        else 0];
                    DS_r3 <- DS_1[calc Me_2 := case when Me_1 > 0 then sqrt(Me_1)
                        when Me_1 = 0 then 0 else ln(abs(Me_1))];
                    DS_r7 <- DS_1[calc Me_2 := case when Me_1 > -10 then Me_1
                        when ln(Me_1) > 0 then ln(0) else 3];
                    DS_r8 <- DS_1[calc Me_2 := if Me_1 > 0 then sum(Me_1 + 1 over ())
                        else 0];
                    DS_r9 <- DS_1[calc Me_2 := (if Id_1 >= 2 then Me_1 else Id_1) + Me_1];
                    DS_r10 <- DS_1[filter (if Id_1 >= 2 then Me_1 else Id_1) <= Me_1];
        Description: At Component level, an operand of an if or a case is computed only on
            the Data Points that take its branch, as on the DuckDB engine. On the pandas
            engine, only a binary operator cut its Components to the Data Points of the
            branch: a unary or parameterised operator on a Component (ln, sqrt, abs, cast)
            ran on every Data Point and raised for the ones of another branch, in an if
            (DS_r1, DS_r2) and a case (DS_r3). A condition or an operand that no Data Point
            reaches is not computed (DS_r7), and an analytic operator in a branch reads every
            Data Point of the clause (DS_r8, where the pandas engine summed only the Data
            Points of the branch and gave 5.0).
            DS_r9, DS_r10: on the pandas engine, a Component level if gave the values of its
            then operand first and those of its else operand after them, and the operator
            that takes the result pairs its values with the other operand by position. The
            value of Id_1 2 was added to the Me_1 of Id_1 1 (DS_r9 gave 0, 4 and 3 instead of
            1, 8 and -2), and the filter dropped Id_1 2 (DS_r10).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_31"
        number_inputs = 1
        references_names = ["1", "2", "3", "7", "8", "9", "10"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_32(self):
        """
        Expression: DS_r1 <- DS_1 + (if sc_1 > 0 then 1 else sc_1 / 0);
                    DS_r2 <- DS_1 + (if sc_2 > 0 then sc_1 / 0 else 3);
                    DS_r3 <- DS_1 + (case when sc_1 < 0 then sc_1 / 0 when sc_1 > 5 then 7
                        when sc_1 / 0 > 1 then 8 else ln(0));
                    DS_r4 <- DS_1[calc Me_2 := if sc_1 > 0 then 1 else sc_1 / 0];
                    DS_r5 <- if DS_1 > 0 then (case when sc_1 > 0 then 1 else -1 / 0)
                        else DS_1;
                    DS_r6 <- DS_1 + cast(if isnull(sc_2) then "1" else "yes", integer);
                    DS_r7 <- DS_1 + cast(if sc_3 >= "b" then "1" else sc_3, number);
        Description: A scalar if or case computes only the operand it selects, as on the
            DuckDB engine: the then operand of the first true condition, or else the else
            operand, which a null condition selects (sc_2 is null). The pandas engine computed
            every condition and operand, so the division by zero of an operand that is not
            selected raised 2-1-15-6, also in a calc (DS_r4) and in a Data Set if (DS_r5).
            The result still has the common type of all the operands, Number.
            DS_r6, DS_r7: the value of a scalar if depends on the data, so the semantic
            analysis does not know it. It took the value of the else operand when it did not
            know the value of the condition, so the DuckDB engine checked the cast of the
            operand that is not selected: "yes" raised 1-1-5-6 (DS_r6) and the value "b" of
            sc_3 raised 2-1-5-1 (DS_r7), where the selected operand "1" casts.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_32"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6", "7"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_1": 10, "sc_2": None, "sc_3": "b"},
        )

    def test_GH_1314_34(self):
        """
        Expression: DS_r <- DS_1 + cast(if sc_2 >= 0.0 then "a" else null, integer);
        Description: A scalar if is a literal, which the semantic analysis checks, only when
            all its operands are literals with the same value, unless its conditions are
            literals (GH_1314_44): which operand it gives depends on the data. The pandas
            engine took the literal flag of the operand selected, so "a" raised the semantic
            error 1-1-5-6, while it is the runtime error 2-1-5-1 that the DuckDB engine raises.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_34"
        number_inputs = 1
        error_code = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=error_code,
            scalars={"sc_1": None, "sc_2": 3, "sc_3": "b"},
        )

    def test_GH_1314_44(self):
        """
        Expression: DS_r <- DS_1[calc Me_2 := cast(if true then "a" else "1", integer)];
                    DS_r <- DS_1[calc Me_2 := cast(case when false then "1" when true then "a"
                        else "2", integer)];
        Description: While the conditions of a scalar if or case are literals, the script
            selects the operand, so the result is that operand, with its value and its literal
            flag, and the cast of the literal "a" is the semantic error 1-1-5-6. The case
            skips a false literal condition. The rule of GH_1314_34 treated a literal
            condition as one that the data decide, so both casts raised the runtime error
            2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_44"
        number_inputs = 1
        error_code = "1-1-5-6"

        for statement in self.LoadVTL(code).splitlines():
            with self.subTest(statement=statement):
                self.NewSemanticExceptionTest(
                    code=code,
                    number_inputs=number_inputs,
                    exception_code=error_code,
                    text=statement,
                )

    def test_GH_1314_45(self):
        """
        Expression: DS_r1 <- DS_1[calc Me_3 := true or (1 / Me_1 > 0)];
                    DS_r2 <- DS_1[calc Me_3 := sc_f and (1 / Me_1 > 0)];
                    DS_r3 <- DS_1 * (if true or (1 / sc_0 > 1) then 2 else 3);
                    DS_r4 <- sum(DS_1 group by Id_1 having count() > 0 or sum(Me_1) / 0 > 1);
                    DS_r5 <- sum(DS_1 group by Id_1 having count() < 2 and 100 / max(Me_1) > 1);
                    DS_r6 <- DS_1[calc Me_3 := nvl(Me_1, 1 / Me_2)];
                    DS_r7 <- DS_1[calc Me_3 := nvl(Me_2, 1 / sc_0)];
                    DS_r8 <- DS_1[keep Me_1] + nvl(sc_1, 1 / sc_0);
        Description: The right operand of an or is not computed where the left one is true,
            nor the right operand of an and where the left one is false, as on the DuckDB
            engine: with a scalar left operand, at Component (DS_r1, DS_r2) and at scalar level
            (DS_r3), and in a having clause, on the groups (DS_r4, DS_r5). Likewise, at
            Component and scalar level, the second operand of an nvl is computed only where
            the first one is null, as the COALESCE of the DuckDB engine (DS_r6 to DS_r8). The
            pandas engine computed them and raised 2-1-15-6.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_45"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6", "7", "8"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_0": 0, "sc_1": 1, "sc_f": False},
        )

    def test_GH_1314_41(self):
        """
        Expression: DS_r <- symdiff(if DS_1 > 1 then (if DS_2 > 1 then 1 / DS_2 else -0.5)
                        else null, DS_3);
        Description: Each Data Set operand of an if is computed on all its Data Points, so
            the inner if divides by zero at Id_1 3 of DS_2, although the outer if takes no
            Data Point from it: DS_1 and DS_2 share none. The DuckDB engine computed 1 / DS_2
            in a MATERIALIZED CTE nested in the inner if, and dropped it with the inner if,
            whose Data Points its statistics showed the outer if does not need, so symdiff
            returned DS_3. The CTE is now at the top of the statement, here before the WITH
            clause of symdiff.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_41"
        number_inputs = 3
        error_code = "2-1-15-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1314_42(self):
        """
        Expression: DS_r2 <- DS_2[calc Me_3 := Me_1 or Me_2, Me_4 := Me_1 and Me_2];
                    DS_r3 <- DS_1[calc Me_4 := Me_2 or (Me_1 / Me_1 > 11) or ...
                        or (Me_1 / Me_1 > 2) or (Me_1 / Me_1 > 0.5)];
                    DS_r4 <- DS_1[calc Me_4 := (Me_2 or (Me_1 / Me_1 > 0.5))
                        and (Me_2 or (Me_1 / Me_1 < 2)),
                        Me_5 := (Me_2 and Me_3) or (not Me_2 and (Me_1 / Me_1 > 0.5))];
                    DS_r5 <- DS_1[calc Me_4 := Me_2 xor Me_3 xor Me_3 xor Me_3 xor Me_3
                        or (Me_1 / Me_1 > 0.5), Me_5 := Me_2 xor Me_3 xor Me_3
                        or (Me_1 / Me_1 > 0.5) xor Me_3 xor Me_2 xor Me_3];
                    DS_r6 <- DS_1[filter (Me_2 or (Me_1 / Me_1 > 0.5))
                        and (not Me_2 and (Me_1 / Me_1 > 0.5))];
                    DS_r7 <- DS_1[aggr Me_4 := count(Me_2 or (Me_1 / Me_1 > 0.5)),
                        Me_5 := max(not Me_2 and (Me_1 / Me_1 > 0.5)) group by Id_1];
        Description: At Component level, the right operand of an or is computed only where
            the left one is false or null, and the right operand of an and only where it is
            true or null. Me_1 / Me_1 divides by zero at Id_1 1, where no statement needs it.
            The pandas engine computed it on every Data Point and raised 2-1-15-6. The DuckDB
            engine computed both operands of an and or an or that gives a value, in a calc
            (DS_r3, DS_r4) or an aggregate operand (DS_r7), and raised 2-1-15-6; in a chain
            with xor (DS_r5) it folded the or link into a list, which computes all its
            elements. DS_r2 gives the three-valued results on the nine pairs of true, false
            and null, DS_r3 is a chain of twelve or operands, written as one CASE, DS_r4 nests
            and and or, and DS_r6 is a filter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Result.
        """
        code = "GH_1314_42"
        number_inputs = 2
        references_names = ["2", "3", "4", "5", "6", "7"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1314_47(self):
        """
        Expression: DS_r <- DS_1[calc Me_2 := cast(if Me_1 > 0 then "3.25" else "1", integer)];
        Description: The if is a String, so it casts to Integer as a String does: "3.25",
            which Id_1 1 takes, is not an Integer and raises 2-1-5-1, as on the pandas engine.
            The DuckDB engine converted the String if to a Number and truncated it to 3.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_47"
        number_inputs = 1
        error_code = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1314_48(self):
        """
        Expression: DS_r <- DS_1[filter Me_b and (1 / Me_1 > 0)];
        Description: In a filter, the right operand of an and is computed where the left one
            is null, as in a calc: null and false is false. At Id_1 1, Me_b is null and Me_1
            is 0, so 1 / Me_1 raises 2-1-15-6, as on the pandas engine. The DuckDB engine
            filtered on the left operand first and did not compute the right one there, so
            the result had no Data Point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1314
        Goal: Check Exception.
        """
        code = "GH_1314_48"
        number_inputs = 1
        error_code = "2-1-15-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )


class ClauseBugs(BugHelper):
    """ """

    classTest = "Bugs.ClauseOperatorsTest"

    def test_VTLEN_466(self):
        """ """
        code = "VTLEN_466"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_VTLEN_467(self):
        """ """
        code = "VTLEN_467"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_VTLEN_469(self):
        """ """
        code = "VTLEN_469"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_VTLEN_523(self):
        """ """
        code = "VTLEN_523"
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_VTLEN_587(self):
        """
        Description:
        Jira issue: VTLEN 587.
        Git Branch: bug-VTLEN-587-Filter-with-null-values.
        Goal: Check interpreter result.
        """
        code = "VTLEN_587"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_25_1(self):
        """
        Description: Calc identifier from non existent component.
        Git issue: GL-25 Calc identifier from non existent component.
        Git Branch: bug-gl-25-calc-identifier-from-non-existent-component.
        Goal: Check Exception.
        """
        code = "GL_25_1"
        number_inputs = 2
        message = "1-1-1-10"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_25_2(self):
        """
        Description: Recalculate an identifier.
        Git issue: GL-25 Calc identifier from non existent component.
        Git Branch: bug-gl-25-calc-identifier-from-non-existent-component.
        Goal: Check interpreter result.
        """
        code = "GL_25_2"
        number_inputs = 1

        message = "1-1-6-13"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # unpivot
    def test_GL_124_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are strings there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_124_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are numbers there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_124_3(self):
        """
        Status: BUG
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are integers there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
            Semantic and interpreter shows difference between types on Me_3.
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_124_4(self):
        """
        Status:
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are time there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
            Semantic and interpreter shows difference between types on Me_3.
            The output in interpreter structure is wrong.
            Error in the CSV output (representation).
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_124_5(self):
        """
        Status:
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are date there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
            KeyError: <class 'numpy.datetime64'>.
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_5"
        number_inputs = 1
        reference_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=reference_names)

    # BUG
    def test_GL_124_6(self):
        """
        Status:
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are time_period there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
            Semantic and interpreter shows difference between types on Me_3.
            The output in interpreter structure is wrong.
            Error in the CSV output (representation).
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_124_7(self):
        """
        Status:
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are duration there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
            Semantic and interpreter shows difference between types on Me_3.
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_124_8(self):
        """
        Status:
        Expression: DS_r := DS_1 [unpivot Id_3, Me_3];
        Description: Measures are boolean there are nulls also.
            line RM 7200: " When a Measure is NULL then unpivot does not create
                            a Data Point for that Measure."
            Semantic and interpreter shows difference between types on Me_3.
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Result.
        """
        code = "GL_124_8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_124_9(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [unpivot Id_3, Me_4];
        Description: Measures with different types.
        Git Branch: bug-124-fix-unpivot-for null-records.
        Goal: Check Exception.
        """
        code = "GL_124_9"
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # Drop
    def test_GL_161_3(self):
        """
        Description: drop with attributes.
        Git Branch: bug-161-inner-join-not-working-properly-attributes-duplicated.
        Goal: Check Result.
        """
        code = "GL_161_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_161_4(self):
        """
        Description: drop with attributes.
        Git Branch: bug-161-inner-join-not-working-properly-attributes-duplicated.
        Goal: Check Result.
        """
        code = "GL_161_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_125_1(self):
        """
        Status:
        Expression: A:= BOP[calc Me_1 := OBS_VALUE > OBS_VALUE];
        Description: check is null semantic interpreter
        Git Branch:
        Goal: Check Result.
        """
        code = "GL_125_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_30_1(self):
        """
        Status: OK
        Expression: result := ANAMART_ENTTY_TM[keep DT_BRTH,DT_CLS][rename ENTTY_RIAD_ID to LEI];
        Description: Rename after keep is working, when the new name is not in the input dataset
        Git Issue: #30
        Goal: Check Result.
        """
        code = "GL_30_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_30_3(self):
        """
        Status: OK
        Expression: result := ANAMART_ENTTY_TM[keep DT_BRTH,DT_CLS][rename ENTTY_RIAD_ID to LEI];
        Description: Rename after keep is working, when any name that is not present in the actual dataset
        Git Issue: #30
        Goal: Check Result.
        """
        code = "GL_30_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_30_4(self):
        """
        Status: OK
        Expression: result := ANAMART_ENTTY_TM[keep DT_BRTH,DT_CLS,LEI]][rename ENTTY_RIAD_ID to LEI];
        Description: check for duplicates
        Git Issue: #30
        Goal: Check Exception.
        """
        code = "GL_30_4"
        number_inputs = 1

        message = "1-1-6-8"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_31_1(self):
        """
        Status:
        Expression: FNL_STTNG2 := FNL_STTNG1 [keep ENTRPRS_SZ_CLCLTD_T][rename ENTRPRS_SZ_CLCLTD_T to ENTRPRS_SZ_CLCLTD];
        Description: Similar to 30. but now the var_from rename is a measure not and identifier
        Git Issue: #31
        Goal: Check Result.
        """
        code = "GL_31_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_31_2(self):
        """
        Status:
        Expression: FNL_STTNG2 := FNL_STTNG1 [keep ENTRPRS_SZ_CLCLTD_T][rename ENTRPRS_SZ_CLCLTD_T to ENTRPRS_SZ_CLCLTD];
        Description: Similar to 30. but now the var_from rename is a measure not and identifier
        Git Issue: #31
        Goal: Check Result.
        """
        code = "GL_31_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_31_3(self):
        """
        Status:
        Expression: FNL_STTNG2 := FNL_STTNG1 [keep ENTRPRS_SZ_CLCLTD_T][rename ENTRPRS_SZ_CLCLTD_T to ENTRPRS_SZ_CLCLTD,RLTD_ID to TYP_ENTRPRS];
        Description:
        Git Issue: #31
        Goal: Check Result.
        """
        code = "GL_31_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_31_4(self):
        """
        Status: OK
        Expression: FNL_STTNG2 := FNL_STTNG1 [keep ENTRPRS_SZ_CLCLTD_T][rename ENTRPRS_SZ_CLCLTD_T to ENTRPRS_SZ_CLCLTD,RLTD_ID to ENTRPRS_SZ_CLCLTD];
        Description: check for duplicates
        Git Issue: #31
        Goal: Check Exception.
        """
        code = "GL_31_4"
        number_inputs = 1

        message = "1-2-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_31_5(self):
        """
        Status: OK
        Expression: CN0869 <- check ( dsPrep.ENTTS_USD_BRTH [ rename DT_RFRNC_USD to COMP_DT ] [ keep COMP_DT ] >= dsPrep.ENTTS_USD_BRTH [ rename DT_BRTH to COMP_DT ] [ keep COMP_DT ] errorcode "CN0869" errorlevel 2 invalid ) ;
        Description:
        Git Issue: #31
        Goal: Check Result.
        """
        code = "GL_31_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_31_6(self):
        """
        Status: OK
        Expression: CN0869 <- check ( dsPrep.ENTTS_USD_BRTH [ rename DT_RFRNC_USD to COMP_DT ] [ keep COMP_DT ] >= dsPrep.ENTTS_USD_BRTH [ rename DT_RFRNC_USD to COMP_DT, ENTTY_RIAD_CD to COMP_DT] [ keep COMP_DT ] errorcode "CN0869" errorlevel 2 invalid ) ;
        Description: check for duplicates, The error is raised in the keep should be raised in the rename
        Git Issue: #31
        Goal: Check Exception.
        """
        code = "GL_31_6"
        number_inputs = 1

        message = "1-2-1"  # 1-1-6-2 the error code was wrong this is better
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_35_1(self):
        """
        Status: OK
        Expression:
        Description: rename after rename inside a join
        Git Issue: #35
        Goal: Check Result.
        """
        code = "GL_35_1"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_35_2(self):
        """
        Status: OK
        Expression: OBSRVD_AGNT_LE := ENTTY_LE[rename ENTTY_RIAD_ID to OBSRVD_AGNT_ID][rename ENTTY_RIAD_CD_LE to LE_OBSRVD_AGNT] ;
        Description: rename after rename in one statement.
        Git Issue: #35
        Goal: Check Result.
        """
        code = "GL_35_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_35_3(self):
        """
        Status: OK
        Expression: OBSRVD_AGNT_LE:= inner_join(CNTRPRTY_LE, ENTTY_LE[rename ENTTY_RIAD_ID to OBSRVD_AGNT_ID][rename ENTTY_RIAD_CD_LE to LE_OBSRVD_AGNT] as B);
        Description: rename after rename in a join.
        Git Issue: #35
        Goal: Check Result.
        """
        code = "GL_35_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_35_4(self):
        """
        Status: OK
        Expression: OBSRVD_AGNT_LE:= inner_join(CNTRPRTY_LE [rename ENTTY_RIAD_ID to AUX_1] as A, ENTTY_LE[rename ENTTY_RIAD_ID to OBSRVD_AGNT_ID][rename ENTTY_RIAD_CD_LE to LE_OBSRVD_AGNT] as B);
        Description: rename after rename in a join.
        Git Issue: #35
        Goal: Check Result.
        """
        code = "GL_35_4"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_35_5(self):
        """
        Status: OK
        Expression: OBSRVD_AGNT_LE:= inner_join(CNTRPRTY_LE [rename ENTTY_RIAD_ID to AUX_1] as A, ENTTY_LE[rename ENTTY_RIAD_ID to OBSRVD_AGNT_ID][rename ENTTY_RIAD_CD_LE to LE_OBSRVD_AGNT] as B);
        Description: rename after rename in a join.
        Git Issue: #35
        Goal: Check Result.
        """
        code = "GL_35_5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_292(self):
        """
        Status: OK
        Description: calc with two child, using in the second the result of the first.
        Git Issue: #292
        Goal: Check Result.
        """
        code = "GL_292"
        number_inputs = 1
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_444_1(self):
        """
        Description: unpivot with a empty dataset
        Git Branch: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/444
        Goal: Check interpreter result.
        """
        code = "GL_444_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_444_2(self):
        """
        Description: unpivot with a scalar dataset
        Git Branch: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/444
        Goal: Check Exception.
        """
        code = "GL_444_2"
        number_inputs = 1
        error_code = "1-1-1-20"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_575(self):
        """
        Description: Rename clause should allow swapping component names.
        Git Branch: cr-575.
        Goal: Check Result.
        """
        code = "GH_575"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_728(self):
        """
        Description: Window aggregate over an identifier (e.g.
            ``max(Id_2 over(partition by Id_1))``) inside a filter or calc clause
            was broken in both engines. Pandas silently returned the row's
            identifier value (filter became a no-op); DuckDB raised
            ``BinderException: WHERE clause cannot contain window functions``.
        Git Branch: cr-728.
        Goal: Check Result.
        """
        code = "GH_728"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1028_1(self):
        """
        Status: OK
        Expression: DS_r1 <- DS_1[unpivot Id_x, Me_v];
        Description: unpivot stacks the Measures into a single one, which took the type
                     of the first Measure rather than the type they all fit in. An
                     Integer Measure declared before a Number one typed the result
                     Integer, so the pandas engine aborted with an ArrowInvalid and the
                     DuckDb engine rounded 1.1 to 1 and 7.7 to 8. DS_2 declares the same
                     Measures the other way round, where the type already came out
                     right.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1028
        Goal: Check Result.
        """
        code = "GH_1028_1"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1119_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter Me_1 = sum(Me_1) ];
        Description: an aggregation inside a filter clause was only rejected when it
                     was the whole condition. Combined with a Component, the binary
                     operator had no branch for a Component against a Data Set and
                     returned None, so both engines aborted with AttributeError:
                     'NoneType' object has no attribute 'data'. The aggregation is now
                     rejected where it is resolved, whatever surrounds it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1119
        Goal: Check Exception.
        """
        code = "GH_1119_1"
        number_inputs = 1
        error_code = "1-2-14"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1119_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 + sum(Me_1) ];
        Description: the same construction inside a calc clause reached the
                     assignment with a None operand and aborted with AttributeError:
                     'NoneType' object has no attribute 'name'. Only an aggregation
                     standing alone on the right-hand side was rejected before.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1119
        Goal: Check Exception.
        """
        code = "GH_1119_2"
        number_inputs = 1
        error_code = "1-2-14"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1119_3(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1, DS_2 filter Me_1 = sum(Me_1) );
        Description: a filter clause inside a join body took the same path and
                     aborted with AttributeError: 'NoneType' object has no attribute
                     'data'.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1119
        Goal: Check Exception.
        """
        code = "GH_1119_3"
        number_inputs = 2
        error_code = "1-2-14"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1119_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter if Me_1 = sum(Me_1) then true else false ];
        Description: an aggregation nested inside an if condition of a filter clause
                     aborted with AttributeError: 'NoneType' object has no attribute
                     'data_type' while building the then/else Data Sets.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1119
        Goal: Check Exception.
        """
        code = "GH_1119_4"
        number_inputs = 1
        error_code = "1-2-14"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1192_16(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc identifier Id_2 := null];
        Description: A null literal cannot be an Identifier: the semantic analysis rejects it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_16"
        number_inputs = 1
        message = "1-1-1-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_17(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc identifier Id_2 := sc_1];
        Description: A scalar without value is a null read from the data: a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_17"
        number_inputs = 1
        message = "2-1-1-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": None}
        )

    def test_GH_1192_37(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[filter Me_1];
        Description: A filter condition that is not Boolean is a structure error, where a
                     ValueError was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_37"
        number_inputs = 1
        message = "1-1-9-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_38(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[pivot Id_1, Me_1];
        Description: pivot raises a SemanticError instead of a NotImplementedError. It raised
                     1-3-13 until pivot takes an Identifier defined on a value domain, which
                     Id_1 is not (issue #1234).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_38"
        number_inputs = 1
        message = "1-1-6-18"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1234_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
        Description: Id_2 is defined on VD_AB, so the result has a Measure for A and one for
                     B, B null where Id_1 2 has no data point for it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1234
        Goal: Check Result.
        """
        code = "GH_1234_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1234-VD_AB"],
        )

    def test_GH_1234_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
        Description: The example of the reference manual: the Measures created are on VD_INT,
                     the value domain of Me_1, and Me_2 and the Attribute At_1 are dropped.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1234
        Goal: Check Result.
        """
        code = "GH_1234_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1234-VD_ABC", "GH_1234-VD_INT"],
        )

    def test_GH_1234_3(self):
        """
        Status: OK
        Expression: define viral propagation CONF ( variable VAt_1 ) is ... end viral propagation;
                    DS_r <- DS_1 [ pivot Id_2 , Me_1 ] [ calc Me_2 := A + B ];
        Description: The viral attribute of the data points of a group is combined by its
                     rule, and the Measures created are read by a later clause.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1234
        Goal: Check Result.
        """
        code = "GH_1234_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1234-VD_AB"],
        )

    def test_GH_1234_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
        Description: Id_2 is a String on no value domain, whose values would give the
                     Measures of the result.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1234
        Goal: Check Exception.
        """
        code = "GH_1234_4"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-6-18",
            exception_message="The Identifier Id_2 is not defined on a value domain",
        )

    def test_GH_1234_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
        Description: Id_1, a value of VD_NAMES, would name a Measure as the Identifier the
                     result keeps.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1234
        Goal: Check Exception.
        """
        code = "GH_1234_5"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-6-9",
            vd_names=["GH_1234-VD_NAMES"],
        )

    def test_GH_1234_6(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
        Description: Pivoting the only Identifier leaves a single data point, with no
                     Identifiers.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1234
        Goal: Check Result.
        """
        code = "GH_1234_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1234-VD_ABC"],
        )

    def test_GH_1265(self):
        """
        Status: OK
        Expression: SC_1 <- 5;
                    DS_r <- DS_1[calc Me_2 := SC_1];
        Description: The DuckDB engine dropped the table of a persistent Scalar once it
                     returned it, so a clause of a later statement could not read it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1265
        Goal: Check Result.
        """
        code = "GH_1265"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1252(self):
        """
        Status: OK
        Expression: DS_r1 <- DS_1 [ calc At_1 := "X" ];
                    DS_r2 <- DS_2 [ aggr Me_1 := sum ( Me_1 ), At_1 := max ( At_1 ) group by Id_1 ];
                    DS_r3 <- DS_2 [ calc At_1 := "X" ] [ aggr Me_1 := sum ( Me_1 ), At_1 := max ( At_1 ) group by Id_1 ] * 2;
        Description: calc and aggr turned the Attribute At_1, computed with no role, into a
                     Measure. A Component of the operand keeps its role when the role is omitted.
                     In DS_r3 the String Attribute At_1 is dropped by *, which DuckDB only does
                     when its own structures of the two clauses keep At_1 an Attribute.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1252
        Goal: Check Result.
        """
        code = "GH_1252"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1318_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[filter not isnull(Me_1)][calc identifier Id_2 := Me_1];
        Description: Me_1 is declared nullable, which a filter does not change, so an
                     Identifier cannot be calculated from it. The message described null
                     values in the data, which the check never reads, and did not name
                     the Identifier.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1318
        Goal: Check Exception.
        """
        code = "GH_1318_2"
        number_inputs = 1
        message = "1-1-1-16"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            exception_message=(
                "Identifier Id_2 cannot be calculated from an expression that is declared nullable."
            ),
        )


class DefinedBugs(BugHelper):
    """ """

    classTest = "Bugs.DefinedOperatorsTest"

    def test_VTLEN_410(self):
        """ """
        code = "VTLEN_410"
        number_inputs = 3

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_252(self):
        """
        Status: OK #
        Expression: IS_INTRCMPNY_DS := drop_identifier ( inner_join ( drop_identifier ( B10.ENTTY_INSTRMNT_CRDTR_LE , ENTTY_RIAD_CD ) [ drop ENTTY_RIAD_CD_LE ] as A , drop_identifier ( B10.ENTTY_INSTRMNT_DBTR_LE , ENTTY_RIAD_CD ) [ drop ENTTY_RIAD_CD_LE ] as B calc IS_INTRCMPNY := "T" ) , LE_CD ) ;
        Description: DO in a join with a calc
        Git Issue: #252
        Goal: Check Result.
        """
        code = "GL_252"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_274(self):
        """
        Status: OK
        Expression: drop_identifier(drop_identifier(INSTTTNL_SCTR_DS_CJ[filter DT_RFRNC >= VLD_FRM and DT_RFRNC <= VLD_T], VLD_T), VLD_FRM);
        Description: DO without a join
        Git Issue: #252
        Goal: Check Result.
        """
        code = "GL_274"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_278(self):
        """
        Status: OK
        Expression: drop_identifier(RIAD_IS_OWNR_C
                        [calc measure IMMDT_PRNT_CD := ENTTY_RIAD_CD]
                        [keep IMMDT_PRNT_CD], ENTTY_RIAD_CD);

        Description: DO without a join
        Git Issue: #252
        Goal: Check Result.
        """
        code = "GL_278"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_269(self):
        """
        Status: OK
        Expression: table4_currencies := check_hierarchy(hier_prep(table4, column, OC, entity, reference_date), metrics rule OC);
        Description: calc in an aggr inside a DO
        Git Issue: #269
        Goal: Check Result.
        """
        code = "GL_269"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_282(self):
        """
        Status: OK
        Expression: DS_r2 := get_dp_rc(table1,row, column, "", "");
                    DS_r3 := get_dp_rc(table1,row, column);
                    DS_r4 := get_dp_rc(table1,row, column);

        Description: default in an UDO
        Git Issue: #282
        Goal: Check Result.
        """
        code = "GL_282"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_317(self):
        """
        Description: Cast + Substr inside UDO
        Issue: #317
        Git Branch: fix-317-cast-sub-UDO
        Goal: Check result
        """

        code = "GL_317"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_316(self):
        """
        Status: OK
        Expression: C08.ONA_JN := drop_identifier ( C08.SHR_PV_DS [ keep SHR_PAV_INSTRMNT ] , PRTCTN_ID )
        Description: Keep in UDO call
        Git Issue: #316
        Goal: Check Result.
        """
        code = "GL_316"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_399(self):
        """
        Status: OK
        Description: Join inside UDO with automatic aliasing
        Git Issue: #399
        Goal: Check Result.
        """
        code = "GL_399"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_30(self):
        """
        Status: OK
        Expression: define operator f ... (twice); DS_r <- f(DS_1);
        Description: A user defined operator defined twice is a structure error, where a
                     ValueError was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_30"
        number_inputs = 1
        message = "1-2-24"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_31(self):
        """
        Status: OK
        Expression: define operator f (x dataset, x dataset) ...; DS_r <- f(DS_1, DS_1);
        Description: A user defined operator with a repeated parameter is a structure error,
                     where a ValueError was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_31"
        number_inputs = 1
        message = "1-3-1-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_39(self):
        """
        Status: OK
        Expression: define operator f (x VD_1) returns number is x end operator; DS_r <- DS_1[calc Me_2 := f(Me_1)];
        Description: VD_1 types the parameter, but no value domain has its name: a script
                     error, where a SyntaxError was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_39"
        number_inputs = 1
        message = "0-5-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_40(self):
        """
        Status: OK
        Expression: DS_r <- eval(GH_1192_40(DS_1) language "SQL" returns dataset {...});
        Description: An External Routine whose result holds infinite values raises 2-1-3-1 on
                     both engines: pandas reported it as 2-1-1-1 and DuckDB returned the
                     infinite values.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_40"
        number_inputs = 1
        message = "2-1-3-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, sql_names=[code]
        )

    def test_GH_1192_41(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := nvl(Me_1, -(1))];
        Description: A sign over a parenthesised number is a literal too, so Me_2 is not
                     nullable, as with -1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_41"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_41_semantic(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := nvl(Me_1, -(1))];
        Description: The semantic analysis gives the structure of GH_1192_41, without data.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_41"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GH_1192_42(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := substr(Me_1, 1, -(2))];
        Description: A sign over a parenthesised number is a literal, so its negative length
                     is rejected by the semantic analysis, as -2 is.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_42"
        number_inputs = 1
        message = "1-1-18-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_43(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := - -Me_1, Me_3 := -(-1), Me_4 := - - 1.5];
        Description: A sign before a signed operand keeps them apart on DuckDB, where -- starts
                     an SQL comment (it raised a ParserException or a BinderException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_43"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_44(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_3 := between(Me_1, Me_2, Me_2)];
        Description: between orders its operands, so it rejects Time values as < and > do,
                     instead of comparing their text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_44"
        number_inputs = 1
        message = "1-1-19-18"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_45(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := time_agg(sc_1, _, Me_1, first)];
        Description: The scalar that gives time_agg its period holds no value: a runtime error
                     with an operand too, where the pandas engine raised an AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_45"
        number_inputs = 1
        message = "2-1-19-24"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": None}
        )

    def test_GH_1192_46(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := time_agg(sc_1, "D", Me_1, first)];
        Description: With a period to aggregate from, the period of the reference is only known
                     at run time: semantic_analysis() raised a KeyError, and the null value is
                     a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_46"
        number_inputs = 1
        message = "2-1-19-24"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": None}
        )

    def test_GH_1192_47(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := random(Id_1, 0 - 1)];
        Description: A computed negative random index is a runtime error on both engines: the
                     DuckDB engine did not check it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_47"
        number_inputs = 1
        message = "2-1-15-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_48(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := dateadd(Me_1, 1, substr("XD", 1, 1))];
        Description: A computed invalid dateadd period is a runtime error on both engines: the
                     DuckDB engine returned nulls.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_48"
        number_inputs = 1
        message = "2-1-19-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_49(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, integer)];
        Description: Text that is no integer is a runtime error on both engines: DuckDB raised a raw
                     ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_49"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_50(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, number)];
        Description: Text that is no number is a runtime error on both engines: DuckDB raised a raw
                     ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_50"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_51(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast("abc", integer)];
        Description: A literal that is no integer is rejected by the semantic analysis, as a literal
                     that is no date is.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_51"
        number_inputs = 1
        message = "1-1-5-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_52(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(sc_1, number)];
        Description: A scalar value that is no number is a runtime error on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_52"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": "abc"}
        )

    def test_GH_1192_53(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, date)]; (Me_1 = "2022-1-5")
        Description: A date written unpadded is read as the loaders read a Date, the Date
                     2022-01-05, on both engines (issue #1349). It was a runtime error
                     (2-1-19-8) on both engines since #1192.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1192_53", number_inputs=1, references_names=["1"])

    def test_GH_1192_54(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, date)]; (Me_1 = "2022-02-30")
        Description: A day that does not exist is a runtime error on both engines: the pandas
                     engine raised a ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_54"
        number_inputs = 1
        message = "2-1-19-8"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_55(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, time_period)]; (Me_1 = "2020X1")
        Description: Text that is no time period is a runtime error of the cast on both
                     engines: DuckDB read it as 2020D1, and pandas only failed when writing the
                     result.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_55"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_56(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast("2020X1", time_period)];
        Description: A literal that is no time period is rejected by the semantic analysis on
                     both engines: pandas raised the input error 0-1-2-7 at run time and DuckDB
                     read it as 2020D1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_56"
        number_inputs = 1
        message = "1-1-5-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_57(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(sc_1, time_period)]; (sc_1 = "2020X1")
        Description: A scalar value that is no time period is a runtime error on both engines:
                     pandas raised the input error 0-1-2-7.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_57"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            scalars={"sc_1": "2020X1"},
        )

    def test_GH_1192_58(self):
        """
        Status: OK
        Description: A null dateadd period or shift, or a null random index or seed, gives null:
                     the pandas engine raised a KeyError or a TypeError, and DuckDB returned a
                     number for random.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_58"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            scalars={"sc_1": None, "sc_2": None},
        )

    def test_GH_1192_59(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_3 := random(Me_1, Me_2)];
        Description: A negative random index read from a Component is a runtime error on both
                     engines, as a negative scalar value is: both engines accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_59"
        number_inputs = 1
        message = "2-1-15-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_60(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, duration)];
        Description: Text cast to a duration gives the same code on both engines (an ISO
                     duration or a code in any case): DuckDB kept the text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_60"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_61(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, duration)]; (Me_1 = "X")
        Description: Text that is no duration is a runtime error on both engines: pandas raised
                     the input error 0-1-1-35 and DuckDB kept the text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_61"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_62(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, time)]; (Me_1 = "2020X1")
        Description: Text that is no time is a runtime error on both engines, which kept it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_62"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_63(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, time)]; (Me_1 = "2020-12-31/2020-01-01")
        Description: A time interval ending before it starts is a runtime error on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_63"
        number_inputs = 1
        message = "2-1-5-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_64(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := time_agg(sc_1, _, Me_1, first)]; (sc_1 = "X")
        Description: A period read from a scalar that is no period indicator is a runtime error
                     on both engines, where the structure error 1-1-19-3 was raised at run time.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_64"
        number_inputs = 1
        message = "2-1-19-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": "X"}
        )

    def test_GH_1192_65(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := time_agg(sc_1, "M", Me_1, first)]; (sc_1 = "D")
        Description: A period read from a scalar that is not coarser than the period to
                     aggregate from is a runtime error on both engines, where the structure
                     error 1-1-19-4 was raised at run time.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_65"
        number_inputs = 1
        message = "2-1-19-26"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": "D"}
        )

    def test_GH_1276_6(self):
        """
        Status: OK
        Expression: define operator to_abs ( ds dataset ) returns dataset is abs ( ds )
                        end operator;
                    DS_r <- to_abs ( DS_1 ) + 1;
        Description: A user defined operator call used as an operand was typed from the body
                     of the operator with its arguments unbound, so the DuckDB engine wrote the
                     call inside the outer expression and failed with a ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1276
        Goal: Check Result.
        """
        code = "GH_1276_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class OtherBugs(BugHelper):
    """ """

    classTest = "Bugs.OtherTest"

    def test_VTLEN_495(self):
        """ """
        code = "VTLEN_495"
        number_inputs = 1

        message = "1-2-6"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_VTLEN_456(self):
        """ """
        # code = "VTLEN_456"
        # number_inputs = 18
        # vd_names = ["VTLEN_456-1", "VTLEN_456-2"]
        # message = "1-1-10-7"
        # TODO: Check if the error code is correct
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message, vd_names=vd_names)
        assert True

    def test_VTLEN_563(self):
        """ """
        code = "VTLEN_563"
        number_inputs = 1
        references_names = ["DS_r"]
        vd_names = []

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )

    def test_Fail_GL_67(self):
        """ """
        code = "GL_67_Fail"
        number_inputs = 39
        message = "1-1-6-10"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_Ok_GL_67(self):
        """ """
        code = "GL_67_Ok"
        number_inputs = 39
        # vd_names = ["GL_67_Ok-1", "GL_67_Ok-6", "GL_67_Ok-7", "GL_67_Ok-8"]

        message = "1-1-6-10"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_39(self):
        """ """
        code = "GL_39"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_52(self):
        """
        Status: OK
        Expression:
        Description: Check count on multiple measures
        Git Branch: test-52-count-multiple
        Goal: Check Result.
        """
        code = "GL_52"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # BUG
    def test_GL_60(self):
        """
        Status: KO
        Expression:
        Description: Check data loading casting numeric values to string type.
        Git Branch: bug-60-comparison-operator-not-working-properly-after
                    -membership-operator.
        Goal: Check Result.
        """
        code = "GL_60"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_61(self):
        """
        Status: OK
        Expression:
        Description: Check if condition as nan
        Git Branch: bug-61-if-nan.
        Goal: Check Result.
        """
        code = "GL_61"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # TODO Move to HR tests
    def test_GL_120(self):
        """
        Status: OK
        Expression:
        Description: Semantic Error on HR with Identifier as ruleComp (VTL Reference line 6413)
        Git Branch: fix-120-HR_Identifier.
        Goal: Check Exception.
        """
        code = "GL_120"
        number_inputs = 1

        message = "1-2-7"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_157_1(self):
        """
        Expression: A01.AUX := ANCRDT_ENTTY_DFLT_C [ calc identifier DT_RFRNC := cast ( "2018-12-31" , date ) ][ filter VLD_FRM <= DT_RFRNC and VLD_T >= DT_RFRNC ]  ;
                    A01.FINAL := inner_join (A01.AUX, ANCRDT_ENTTY_RSK_C using DT_RFRNC, ENTTY_RIAD_CD)
        Description: Calc and cast for dates.
        Issue: #157 fail over dates references columns.
        Git Branch: bug-157-operator-fails-over-dates-references-columns.
        Goal: Check result.
        """
        code = "GL_157_1"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_157_2(self):
        """
        Expression: A01.AUX := ANCRDT_ENTTY_DFLT_C [ calc identifier DT_RFRNC := cast ( "2018-12-31" , date ) ][ filter VLD_FRM <= DT_RFRNC and VLD_T >= DT_RFRNC ]  ;
                    A01.FINAL := inner_join (A01.AUX, ANCRDT_ENTTY_RSK_C using DT_RFRNC, ENTTY_RIAD_CD)
        Description: The same as earlier but with datapoints and attributes.
        Issue: #157 fail over dates references columns.
        Git Branch: bug-157-operator-fails-over-dates-references-columns.
        Goal: Check result.
        """
        code = "GL_157_2"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_171_1(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check result.
        """
        code = "GL_171_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_171_2(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check result.
        """
        code = "GL_171_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_171_3(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check result.
        """
        code = "GL_171_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_171_4(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check result.
        """
        code = "GL_171_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_171_5(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check result.
        """
        code = "GL_171_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_171_6(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check Exception.
        """
        code = "GL_171_6"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_171_7(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check Exception.
        """
        code = "GL_171_7"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_171_8(self):
        """
        Description: Is related with comparison, set and clause bugs.
        Issue: #171 in and not_in Not working properly inside a calc.
        Git Branch: bug-171-in-and-not-inside-a-calc.
        Goal: Check Exception.
        """
        code = "GL_171_8"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_262(self):
        """
        Description: udo with a join inside
        Issue: #262 fix udo plus drop on an inner join
        Git Branch: fix-262-udo-inner
        Goal: Check result
        """

        code = "GL_262"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_281(self):
        """
        Description: nullability in union
        Issue: #281 review nullability in union operator
        Git Branch: fix-281-union-null
        Goal: Check result
        """

        code = "GL_281"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_294(self):
        """
        Description: Comparison operator discrepancy
        Issue: #294 comparison operator mismatch in identifiers
        Git Branch: fix-294-comp-id-mismatch
        Goal: Check result
        """

        code = "GL_294"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_364(self):
        """
        Description: NewComponent symbol in Partition ID in analytic
        Issue: #364
        Git Branch: fix-364-partition-nc
        Goal: Check exception code
        """

        code = "GL_364"
        number_inputs = 1

        error_code = "1-1-1-10"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_366(self):
        """
        Description: Analytic inside if with empty data
        Issue: #366
        Git Branch: fix-366-analytic-empty
        Goal: Check result
        """

        code = "GL_366"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_377(self):
        """
        Description: Several datasets on same JSON
        Issue: #377
        Git Branch: fix-377-datastructure-refactor
        Goal: Check result
        """

        code = "GL_377"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_566_1(self):
        code = "GL_566_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(
            code=code,
            vd_names=["GL_566_1-COUNTRY", "GL_566_1-TIME_PERIOD"],
            number_inputs=number_inputs,
            references_names=references_names,
        )

    def test_GH_969_1(self):
        """
        Description: hierarchy roll-up per the reference-manual roll-up examples
                     (non_null default / non_zero / partial_null) over a dataset
                     with a viral attribute. partial_null must keep a computed
                     node whose children all exist with null measures (H = D + E),
                     and a missing child data point (R in D = R + S) contributes a
                     null attribute value to the propagation rule, so the
                     when-null clause wins instead of the sibling's value.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/969
        Goal: Check Result.
        """
        code = "GH_969_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(
            code=code,
            vd_names=["GH_969_1-VD_1"],
            number_inputs=number_inputs,
            references_names=references_names,
        )

    def test_GH_969_2(self):
        """
        Description: same roll-up examples as GH_969_1 but the rule component is
                     the only identifier. This used to crash on both backends
                     (pandas IndexError from a merge on an empty key list; DuckDB
                     'USING (1=1)' parse error) instead of aggregating over the
                     single global group.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/969
        Goal: Check Result.
        """
        code = "GH_969_2"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(
            code=code,
            vd_names=["GH_969_2-VD_1"],
            number_inputs=number_inputs,
            references_names=references_names,
        )

    def test_564(self):
        """
        Description: Can't subspace a component named unit
        """

        code = "GL_564_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := nvl(Me_1, -1)];
        Description: A negative literal is a literal, so nvl replaces every null of Me_1 and
                     Me_2 is not nullable, as with a positive literal.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := nvl(Me_1, -1)];
        Description: The semantic analysis gives the structure of GH_1192_1, without data.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GH_1192_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := substr(Me_1, 1, -9)];
        Description: A negative literal length is rejected by the semantic analysis, as a
                     literal length of 0 is.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_3"
        number_inputs = 1
        message = "1-1-18-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := -1, Me_3 := +1.5, Me_4 := Me_1 * -2];
        Description: Signed literals keep their values and types.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Result.
        """
        code = "GH_1192_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1192_7(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_3 := substr(Me_1, 1, Me_2)];
        Description: A negative length read from a Component is a runtime error on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_7"
        number_inputs = 1
        message = "2-1-18-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_8(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_3 := instr(Me_1, "l", Me_2)];
        Description: A start of 0 read from a Component is a runtime error on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_8"
        number_inputs = 1
        message = "2-1-18-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_9(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := substr(Me_1, 1, sc_1)];
        Description: A negative length given as a scalar value is a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_9"
        number_inputs = 1
        message = "2-1-18-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": -3}
        )

    def test_GH_1192_14(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := f(Me_1, -2)]; (f calls substr(x, 1, n))
        Description: A negative literal passed to a user defined operator is rejected by the
                     semantic analysis, as when it is written in place.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_14"
        number_inputs = 1
        message = "1-1-18-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_15(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := string_distance(hamming, Me_1, "fooo")];
        Description: Strings of different lengths read from the data are a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_15"
        number_inputs = 1
        message = "2-1-18-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )


class ExternalRoutineBugs(BugHelper):
    """ """

    classTest = "Bugs.ExternalRoutineTest"

    def test_GL_156_1(self):
        """
        Status: OK
        Expression: C07.ANAMART_PRTCTN_MSMTCH := eval ( prtctnDts ( C07.MSMTCH_BL_DS ) language "sqlite" returns dataset { identifier < string > CNTRCT_ID , identifier < date > DT_RFRNC , identifier < string > OBSRVD_AGNT_CD , identifier < string > INSTRMNT_ID , identifier < string > PRTCTN_ID , measure < string > IS_PRTCTN_MTRTY_MSMTCH , measure < number > MTRTY_MSMTCH , measure < number > PRTCTN_RSDL_MTRTY_DYS } ) ;
        Description: Example for externals routines with dot in the name in a eval.
        Git Branch: GL_156-quick fix for allow module dot name on eval operator.
        Goal: Check Result.
        """
        code = "GL_156_1"
        number_inputs = 2
        references_names = ["1"]
        sql_names = ["prtctnDts"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            sql_names=sql_names,
        )

    def test_GL_159_1(self):
        """
        Status: OK
        Expression:
        Description: Example for externals routines with correct type for
                    dataPoints type date.
        Git Branch: GL_159- fix wrong result type for eval.
        Goal: Check Result.
        """
        code = "GL_159_1"
        number_inputs = 6
        references_names = ["1", "2"]
        sql_names = ["instrFctJn"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            sql_names=sql_names,
        )

    def test_GH_1192_22(self):
        """
        Status: OK
        Expression: DS_r <- eval(SQL_GH_1192_22(DS_1) language "SQL" returns dataset {...});
        Description: A query that reads a Component the operand does not have fails on the
                     structures, so the semantic analysis rejects it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_22"
        number_inputs = 1
        message = "1-1-1-23"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            sql_names=["SQL_GH_1192_22"],
        )


class CastBugs(BugHelper):
    classTest = "Bugs.CastTest"

    def test_GH_923(self):
        """
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, string), Me_3 := upper(Me_1), ...];
        Description: Boolean-to-string results are "true"/"false" in both engines, as
            the VTL 2.2 manual casts a Boolean to String (issue #1314). Component-level
            string operators (upper, substr, replace, length, instr) on a Boolean raised
            ``BinderException`` on the DuckDB engine because no implicit coercion was
            applied.
        Git Issue: GH_923.
        Goal: Check Result matches the pandas engine output.
        """
        code = "GH_923"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_923_2(self):
        """
        Expression: DS_r <- DS_1;
        Description: dataset CSVs written to output_folder must serialize
            booleans Python-style (True/False) in both engines; the DuckDB
            engine wrote SQL-style lowercase (true/false) via COPY TO.
        Git Issue: GH_923.
        Goal: Check written CSV text.
        """
        code = "GH_923_2"
        script = self.LoadVTL(code)
        with TemporaryDirectory() as tmpdir:
            run(
                script=script,
                data_structures=self.filepath_json / f"{code}-1.json",
                datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
                output_folder=Path(tmpdir),
                use_duckdb=_use_duckdb_backend(),
            )
            content = (Path(tmpdir) / "DS_r.csv").read_text()
        lines = content.strip().splitlines()
        assert lines[0] == "Id_1,Me_1"
        assert sorted(lines[1:]) == ["1,True", "2,False", "3,"]

    def test_GH_923_2_in_memory(self):
        """
        Expression: DS_r <- DS_1;
        Description: in-memory results must keep boolean values (not 'True'
            strings) in both engines; only the CSV text serialization is
            formatted.
        Git Issue: GH_923.
        Goal: Check returned DataFrame values.
        """
        code = "GH_923_2"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        me_1 = result["DS_r"].data["Me_1"]
        assert me_1.tolist()[:2] == [True, False]
        assert pd.isna(me_1.iloc[2])

    def test_GH_923_3(self):
        """
        Expression: sc_r <- true || "_x";
        Description: scalar boolean || string gives the same text in both engines,
            "true_x", as the VTL 2.2 manual casts a Boolean to String (issue #1314).
        Git Issue: GH_923.
        Goal: Check scalar result value.
        """
        code = "GH_923_3"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={},
            use_duckdb=_use_duckdb_backend(),
        )
        assert result["sc_r"].value == "true_x"

    def test_GH_923_4(self):
        """
        Expression: DS_r <- DS_1[calc Me_2 := Me_1 || "_x"];
        Description: component-level boolean || string gives the same text in both
            engines, "true_x" (issue #1314); the pandas engine raised TypeError from
            the series-scalar path (issue GH_940).
        Git Issue: GH_923.
        Goal: Check component result values.
        """
        code = "GH_923_4"
        script = self.LoadVTL(code)
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        me_2 = result["DS_r"].data["Me_2"]
        assert me_2.tolist()[:2] == ["true_x", "false_x"]
        assert pd.isna(me_2.iloc[2])

    def test_GH_940(self):
        """
        Expression: DS_r <- DS_1[calc Me_2 := "x_" || Me_1];
        Description: component-level string || boolean (component on the
            right) must implicitly cast the Boolean component to text ("true",
            "false"); the pandas engine raised TypeError from the series-scalar
            path.
        Git Issue: GH_940.
        Goal: Check Result.
        """
        code = "GH_940"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_940_2(self):
        """
        Expression: DS_r <- DS_1 || "_x";
        Description: dataset-level boolean || string must implicitly cast the
            Boolean measure to text ("true", "false"); the pandas engine raised
            TypeError from the series-scalar path.
        Git Issue: GH_940.
        Goal: Check Result.
        """
        code = "GH_940_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_449_2(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        Note: In VTL 2.2, TimePeriod->Date is allowed without mask but only
              for daily periods. Monthly periods fail at runtime.
        """
        code = "GL_449_2"
        number_inputs = 1
        message = "2-1-5-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_449_3(self):
        """
        Status: OK
        Description:
        Goal: Check Result.
        Note: Cast with mask raises NotImplementedError (not yet implemented).
        """
        code = "GL_449_3"
        number_inputs = 1
        with pytest.raises((NotImplementedError, Exception)):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=["1"])

    def test_GL_449_6(self):
        """
        Status: OK
        Description: Over dataset
        Goal: Check Result.
        Note: Cast with mask raises NotImplementedError (not yet implemented).
        """
        code = "GL_449_6"
        number_inputs = 1
        with pytest.raises((NotImplementedError, Exception)):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=["1"])

    def test_GL_449_7(self):
        """
        Status: OK
        Description: Over scalar
        Goal: Check Result.
        Note: Cast with mask raises NotImplementedError (not yet implemented).
        """
        code = "GL_449_7"
        number_inputs = 1
        with pytest.raises((NotImplementedError, Exception)):
            self.BaseTest(
                code=code,
                number_inputs=number_inputs,
                references_names=["1"],
                scalars={"sc_1": "2000Q2"},
            )

    def test_GL_448_1(self):
        """
        Status: OK
        Description: stock to flow
        Goal: Check Result.
        """
        code = "GL_448_1"
        number_inputs = 1
        message = "1-1-19-7"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_448_2(self):
        """
        Status: OK
        Description: flow to stock
        Goal: Check Result.
        """
        code = "GL_448_2"
        number_inputs = 1
        message = "1-1-19-7"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_90_1(self):
        """
        Status: OK
        Description: aggr and analytic ops ceil instead of round.
        Git Branch: cr-90
        Goal: Check Result.
        """
        code = "GL_90_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_90_2(self):
        """
        Status: OK
        Description: aggr and analytic ops ceil instead of round.
        Git Branch: cr-90
        Goal: Check Result.
        """
        code = "GL_90_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_931_1(self):
        """
        Status: OK
        Description: stock_to_flow and flow_to_stock only compute over number
                     measures; Boolean and String measures pass through, as the
                     reference manual types the operand as measure<number>.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/931
        Goal: Check Result.
        """
        code = "GH_931_1"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_990_1(self):
        """
        Status: OK
        Expression: DS_r <- count ( DS_1#Me_1 group by Id_1 having count() > 2 );
        Description: an aggregation with a having clause over an operand that
                     carries a viral attribute failed on the Pandas engine: the
                     having analysis dropped the viral attribute from the data
                     while keeping it as a component, and the propagated values
                     were then assigned positionally to the groups the having
                     clause had already filtered out.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/990
        Goal: Check Result.
        """
        code = "GH_990_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_990_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := count ( Me_1 over ( partition by Id_1 ) ) ];
        Description: the analytic count replaced the nulls of every measure with
                     the numeric sentinel -1 before querying, which corrupts a
                     String measure and made the Pandas engine fail; it also made
                     the count include the null values, disagreeing with the
                     DuckDB engine.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/990
        Goal: Check Result.
        """
        code = "GH_990_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_994_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr m1 := min(Id_1), m2 := min(Id_2),
                                        m3 := min(Id_date), m4 := min(Id_period),
                                        m5 := min(At_1), m6 := max(Id_1),
                                        m7 := min(Id_3)
                                   group by Id_3 ];
        Description: an aggregate operator inside an aggr clause is applied to the
                     operand Component whatever its role is, so that Component is
                     the Measure of the Data Set it aggregates. Keeping the original
                     role left that Data Set without Measures, and both engines
                     raised IndexError whenever the aggregated Component was an
                     Identifier or an Attribute. An aggregated Identifier is held
                     under an internal name so that it stays an Identifier of the
                     operand and can still be used as a grouping key (m7).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/994
        Goal: Check Result.
        """
        code = "GH_994_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_995_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m1 := min(Me_1 over (order by Id_2, Id_1)),
                                        m2 := max(Me_1 over (order by Id_2 desc, Id_1 desc)),
                                        m3 := median(Me_1 over (order by Id_2, Id_1)),
                                        m4 := sum(Me_1 over (order by Id_1)) ];
        Description: an analytic invocation that omits the partition clause is
                     partitioned by the Identifiers the order clause does not name.
                     The DuckDB engine emitted no PARTITION BY at all, so it
                     analysed the whole Data Set as a single partition and
                     disagreed with the Pandas engine for every analytic operator.
                     No window clause is given either, so each value covers the
                     partition up to the current Data Point (see issue #1249).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/995
        Goal: Check Result.
        """
        code = "GH_995_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1002_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m1 := sum(Me_1 over (partition by Id_1
                                                             order by Id_2)),
                                        m2 := count(...), m3 := last_value(...),
                                        m4 := sum(... data points between
                                              unbounded preceding and current
                                              data point) ];
        Description: with an order clause, an omitted window clause defaults to
                     range between unbounded preceding and current data point
                     (VTL 2.2, issue #1249), so m1, m2 and m3 run from the first
                     Data Point of the partition to the current one, as m4 does
                     with its explicit window.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1002
        Goal: Check Result.
        """
        code = "GH_1002_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1010_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier i := if Me_1 > 1 then Me_1 else sc_1 ];
        Description: an expression declared not nullable can still hold a null, which
                     only the data shows: the input scalar sc_1 is declared not nullable
                     and holds a null. A calculated Identifier is checked on its data as
                     well, and reports it at run time. A null literal operand would make
                     the if nullable, which the semantic analysis rejects (issue #1314).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1010
        Goal: Check Exception.
        """
        code = "GH_1010_1"
        number_inputs = 1
        message = "2-1-1-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message, scalars={"sc_1": None}
        )

    def test_GH_1027_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1[keep Me_1, Me_1];
        Description: a Component named twice in a keep clause came out of it twice, so
                     the Data Set carried more columns than Components and the CSV it
                     wrote could not be read back. The repeated name is rejected, as
                     the rename clause already rejects one.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1027
        Goal: Check Exception.
        """
        code = "GH_1027_1"
        number_inputs = 1
        message = "1-1-6-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1027_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Me_1, Me_1];
        Description: the same repeated name in a drop clause, which the DuckDb engine
                     reported as a raw parser error over its EXCLUDE list.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1027
        Goal: Check Exception.
        """
        code = "GH_1027_2"
        number_inputs = 1
        message = "1-1-6-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1027_3(self):
        """
        Status: OK
        Expression: DS_r := left_join(DS_1 as a, DS_2 as b keep Me_1, Me_1);
        Description: a join clause quietly deduplicated the names it was given, so the
                     repeated one never reached the clause that rejects it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1027
        Goal: Check Exception.
        """
        code = "GH_1027_3"
        number_inputs = 2
        message = "1-1-6-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1027_4(self):
        """
        Status: OK
        Expression: DS_r := left_join(DS_1 as a, DS_2 as b drop Me_1, Me_1);
        Description: the same repeated name in the drop of a join clause.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1027
        Goal: Check Exception.
        """
        code = "GH_1027_4"
        number_inputs = 2
        message = "1-1-6-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1010_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier i := Me_1 * 2 ];
        Description: an Identifier cannot be nullable, so an expression declared
                     nullable is rejected on the structure, whatever its values are.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1010
        Goal: Check Exception.
        """
        code = "GH_1010_2"
        number_inputs = 1
        message = "1-1-1-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1007_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m1 := if Me_bool then 1 else 2.5,
                                        m2 := case when Me_bool then 1 else 2.5,
                                        m3 := if Me_bool then Me_int else Me_num,
                                        m4 := nvl(Me_int, 2.5),
                                        m5 := nvl(Me_int, 7) ];
        Description: the implicit promotion of two types was resolved in favour of
                     the left one, so an Integer on the left of a Number gave an
                     Integer result. The conditionals then truncated or rounded the
                     Number branch, differently on each engine (m1, m2, m3). nvl
                     also declared the type of its first operand and filled in that
                     type, so the applicable value was truncated as well (m4). m5
                     keeps two Integers as an Integer, so nothing is over promoted.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1007
        Goal: Check Result.
        """
        code = "GH_1007_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1005_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr m := count(Me_1) group by Id_2
                                   having rank(over (order by Id_2)) > 1 ];
        Description: a having condition refers to the groups, so the reference
                     manual requires it to invoke aggregate operators, and an
                     analytic invocation cannot be nested in an aggregate one.
                     The analytic was evaluated anyway and the having clause then
                     read it as a Data Set, raising AttributeError: 'DataComponent'
                     object has no attribute 'get_measures' on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1005
        Goal: Check Exception.
        """
        code = "GH_1005_1"
        number_inputs = 1
        error_code = "1-1-3-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1004_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m1 := lag(Me_1 over (partition by Id_2
                                                             order by Id_1)),
                                        m2 := lead(Me_1 over (...)),
                                        m3 := lag(Id_1, 1 over (...)),
                                        m4 := lead(Id_1, 1 over (...)) ];
        Description: the offset of lag and lead is optional, but an absent
                     parameter list was rendered as LAG("Me_1", ) and the query
                     failed to parse (m1, m2). Both operators also have no value
                     to shift in at the edges of the partition, so their result
                     is nullable whatever the operand declared; taking it from a
                     non nullable Identifier was rejected as null values in a non
                     nullable structure (m3, m4).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1004
        Goal: Check Result.
        """
        code = "GH_1004_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1000_1(self):
        """
        Status: OK
        Expression: DS_r <- first_value ( DS_1 over (partition by Id_2 order by Id_1) );
        Description: an analytic invocation over a Data Set that carries viral
                     attributes re-attaches them by merging on the Identifiers.
                     The merge took them from the operand, which still holds the
                     values as they were loaded, while the result carries the
                     dates as the query returned them, so a Date Identifier made
                     the Pandas engine raise "You are trying to merge on
                     datetime64[us] and string columns".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1000
        Goal: Check Result.
        """
        code = "GH_1000_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_998_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m := sum(Me_1 over (partition by Id_1)) ];
        Description: the analytic sum takes measure<number>, but its operator
                     declared no type to check, unlike the aggregate sum and
                     unlike the other numeric analytic operators. A String
                     Measure reached the query and both engines surfaced a raw
                     BinderException for sum(VARCHAR) instead of a SemanticError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/998
        Goal: Check Exception.
        """
        code = "GH_998_1"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1009_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter sum(Me_1) > 1 ];
        Description: a filter condition is evaluated for each Data Point, so an
                     aggregation cannot be its operand. The aggregation came back as
                     a Data Set and the clause then read a Component name off it,
                     raising AttributeError: 'Aggregation' object has no attribute
                     'value' on both engines. The calc clause already rejected the
                     same construction, and filter now does too.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1009
        Goal: Check Exception.
        """
        code = "GH_1009_1"
        number_inputs = 1
        error_code = "1-2-14"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1011_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter true ];
        Description: a filter condition is a Component expression evaluated for each
                     Data Point, so it cannot be a constant. The Pandas engine raised
                     AttributeError: 'Scalar' object has no attribute 'data' for every
                     constant condition, while the DuckDB engine accepted four of them
                     and raised the same AttributeError for 1 = 1. Both engines now
                     reject them with the same SemanticError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1011
        Goal: Check Exception.
        """
        code = "GH_1011_1"
        number_inputs = 1
        error_code = "1-2-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1008_1_if_component_order(self):
        """
        Expression: DS_r <- if H_c then H_a else H_b;
        Description: the branches of if were required to declare their Components in
                     the same order, as the check compared the ordered lists of names.
                     A calc moves the Component it recalculates to the end, so two
                     Data Sets holding the same Components were rejected. Written
                     inline because the branches must be named Data Sets: an inline
                     clause chain as a branch fails on the DuckDB engine for an
                     unrelated reason.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1008
        Goal: Check Result.
        """
        self._assert_1008_branch_order("if H_c then H_a else H_b")

    def test_GH_1008_2_case_component_order(self):
        """
        Expression: DS_r <- case when H_c then H_a else H_b;
        Description: same ordered comparison as GH_1008_1, in the operand check of
                     case instead of the then-else check of if.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1008
        Goal: Check Result.
        """
        self._assert_1008_branch_order("case when H_c then H_a else H_b")

    def _assert_1008_branch_order(self, expression):
        """Run a conditional whose branches hold the same Components in a different
        order, and check the result."""
        code = "GH_1008_3"
        script = (
            "H_a := DS_1 [keep Me_1, Me_2];\n"
            "H_b := H_a [calc Me_1 := 0];\n"
            "H_c := DS_1 [keep Me_bool] [rename Me_bool to c];\n"
            f"DS_r <- {expression};"
        )
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        data = result["DS_r"].data.sort_values("Id_1").reset_index(drop=True)
        assert sorted(result["DS_r"].components) == ["Id_1", "Me_1", "Me_2"]
        assert list(data["Me_1"].astype("string").fillna("")) == ["1", "0", ""]
        assert list(data["Me_2"].astype("string").fillna("")) == ["a", "", "c"]

    def test_GH_1008_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m2 := nvl(Me_1, null),
                                        m3 := nvl(Me_2, null) ];
        Description: a null applicable value replaces nothing, but nvl filled with
                     it, which the Pandas engine rejected with ValueError: Must
                     specify a fill 'value' or 'method'. The operand is now returned
                     as it is, and the result keeps the nullability of the operand
                     since its nulls survive.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1008
        Goal: Check Result.
        """
        code = "GH_1008_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1008_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc m := nvl(2.5, Me_1) ];
        Description: nvl raised a bare Python ValueError when its operands were at
                     levels that cannot be combined, instead of a SemanticError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1008
        Goal: Check Exception.
        """
        code = "GH_1008_4"
        number_inputs = 1
        error_code = "1-1-9-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1008_5_case_scalar_condition(self):
        """
        Expression: DS_r <- DS_1 [calc m := case when 1 > 0 then 10 else 20];
        Description: case never took a value when its condition was a Scalar, so the
                     Pandas engine returned null for every Data Point while the
                     DuckDB engine returned 10. Checked on the values alone: the
                     engines still disagree on the declared nullability of a calc
                     Component built from a scalar case, which is not part of this
                     issue.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1008
        Goal: Check Result.
        """
        code = "GH_1008_3"
        script = "DS_r <- DS_1 [calc m := case when 1 > 0 then 10 else 20];"
        result = run(
            script=script,
            data_structures=self.filepath_json / f"{code}-1.json",
            datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
            use_duckdb=_use_duckdb_backend(),
        )
        assert list(result["DS_r"].data["m"]) == [10, 10, 10]

    def test_GH_1192_12(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast("2022Q1", date)];
        Description: A literal that cannot be cast to a date is rejected by the semantic analysis.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_12"
        number_inputs = 1
        message = "1-1-5-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_13(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(sc_1, date)];
        Description: A scalar value that cannot be cast to a date is a runtime error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_13"
        number_inputs = 1
        message = "2-1-19-8"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            scalars={"sc_1": "2022Q1"},
        )

    def test_GH_1192_29(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := cast(Me_1, VD_1)];
        Description: A cast to a value domain when no value domains are given is a script
                     error, where a NotImplementedError was raised. It raised 0-5-1-8 until
                     a cast takes a value domain (issue #1241).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_29"
        number_inputs = 1
        message = "1-2-22"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1241_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , VD_1 ) ];
        Description: A cast to the value domain VD_1 gives Me_2 on it, as A and B are values
                     of it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1241
        Goal: Check Result.
        """
        code = "GH_1241_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1241-VD_1"],
        )

    def test_GH_1241_2(self):
        """
        Status: OK
        Expression: DS_r1 <- cast ( DS_1 , VD_1 );
                    DS_r2 <- cast ( DS_2 , VD_2 );
        Description: The Measure of a Data Set cast to a value domain is defined on it, from
                     its basic type (a null stays null) and from another value domain on it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1241
        Goal: Check Result.
        """
        code = "GH_1241_2"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1241-VD_1", "GH_1241-VD_2"],
        )

    def test_GH_1241_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , VD_2 ) ];
        Description: C, a value of VD_1, is not a value of VD_2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1241
        Goal: Check Exception.
        """
        code = "GH_1241_3"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="2-1-5-4",
            exception_message="At op cast: Cannot cast C of String to Value Domain VD_2",
            vd_names=["GH_1241-VD_1", "GH_1241-VD_2"],
        )

    def test_GH_1241_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , VD_1 ) ];
        Description: The Integer 1 is cast to the String "1", the basic type of VD_1, which is
                     not a value of VD_1. It raised 1-1-5-1 before the cast to a value domain
                     applied the casts of the basic types (issue #1342).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1241
        Goal: Check Exception.
        """
        code = "GH_1241_4"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="2-1-5-4",
            exception_message="At op cast: Cannot cast 1 of String to Value Domain VD_1",
            vd_names=["GH_1241-VD_1"],
        )

    def test_GH_1241_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "D" , VD_1 ) ];
        Description: The literal D is not a value of VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1241
        Goal: Check Exception.
        """
        code = "GH_1241_5"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-5-6",
            exception_message="Impossible to cast D from type String to VD_1",
            vd_names=["GH_1241-VD_1"],
        )

    def test_GH_1241_6(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , VD_3 ) ];
        Description: VD_3 is not one of the value domains given.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1241
        Goal: Check Exception.
        """
        code = "GH_1241_6"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-2-8",
            exception_message="Missing value domain 'VD_3' definition",
            vd_names=["GH_1241-VD_1"],
        )

    vd_1342 = [
        "GH_1342-VD_INT",
        "GH_1342-VD_NUM",
        "GH_1342-VD_CODE",
        "GH_1342-VD_DATE",
        "GH_1342-VD_INT2",
    ]

    def test_GH_1342_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_1 := cast ( Me_s , VD_INT ) ,
                                        Me_2 := cast ( Me_i , VD_NUM ) ,
                                        Me_3 := cast ( Me_i , VD_CODE ) ,
                                        Me_4 := cast ( Me_b , VD_INT ) ,
                                        Me_5 := cast ( Me_d , VD_DATE ) ,
                                        Me_7 := cast ( Me_v , VD_NUM ) ,
                                        Me_8 := cast ( "3" , VD_INT ) ] [ keep ... ];
        Description: A cast to a value domain casts to its basic type first, from any type that
                     casts to it: String, Integer and Boolean, and another value domain (VD_INT2)
                     of another basic type. They raised 1-1-5-1. A Number does not cast to the
                     integers of VD_INT (issue #1388).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1342
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1342_1",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1342,
        )

    def test_GH_1342_2(self):
        """
        Status: OK
        Expression: DS_r1 <- cast ( DS_1 , VD_INT ); DS_r2 <- cast ( DS_2 , VD_CODE );
                    DS_r3 <- cast ( DS_2 , VD_NUM ) + 1; DS_r4 <- cast ( DS_2 , string );
        Description: The Measure of a Data Set cast to a value domain is named as in a cast to
                     its basic type: int_var and str_var, which the DuckDB engine left empty.
                     The Integers of DS_2, with a null, are cast to "2" and not "2.0" on the
                     pandas engine.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1342
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1342_2",
            number_inputs=2,
            references_names=["1", "2", "3", "4"],
            vd_names=self.vd_1342,
        )

    def test_GH_1342_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_1 := cast ( Me_s , VD_INT2 ) ];
        Description: The String "3" is cast to the Integer 3, which is not a value of VD_INT2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1342
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1342_3",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op cast: Cannot cast 3 of Integer to Value Domain VD_INT2",
            vd_names=self.vd_1342,
        )

    def test_GH_1342_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_1 := cast ( Me_t , VD_INT ) ];
        Description: A Date does not cast to an Integer, so it does not cast to VD_INT either.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1342
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1342_4",
            number_inputs=1,
            exception_code="1-1-5-1",
            exception_message="Type Date, cannot be cast to VD_INT",
            vd_names=self.vd_1342,
        )


class AnalyticBugs(BugHelper):
    """Bugs on analytic (window) operators."""

    classTest = "Bugs.AnalyticBugs"

    def test_GH_1026_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Va_str][calc m := ratio_to_report(
                        ratio_to_report(Me_num over (partition by Id_2))
                        over (partition by Id_3))];
                    and the same nesting with sum(ratio_to_report(...) over ...),
                    first_value(lag(...) over ...), and a calc chain whose second
                    clause nests an analytic over the first clause's output.
        Description: nested analytic invocations landed in a single SELECT on the
                     DuckDB engine and SQL forbids nesting window functions, so any
                     analytic over the result of another raised BinderException; the
                     inner analytic is now materialised in a subquery. The pandas
                     engine already computed it correctly.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1026
        Goal: Check Result.
        """
        code = "GH_1026_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1026_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Va_str][calc m := ratio_to_report(
                        Me_num - Me_num over (partition by Id_2))];
        Description: a partition whose values sum to zero makes ratio_to_report
                     divide 0/0; the pandas engine only caught infinity (x/0 with
                     x <> 0) and quietly returned null for the whole partition,
                     while the DuckDB engine raised 2-1-3-1. Both engines now
                     raise 2-1-3-1 on any zero-sum partition.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1026
        Goal: Check Exception.
        """
        code = "GH_1026_2"
        number_inputs = 1
        message = "2-1-3-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1026_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Va_str][calc m := ratio_to_report(
                        Me_num - 5.5 over (partition by Id_2))];
        Description: on a genuine division by zero (x/0 with x <> 0) both engines
                     raised 2-1-3-1 but blamed different operators: pandas said
                     ratio_to_report and DuckDB said division, because the generic
                     division-by-zero pattern in the query error mapping shadowed
                     the specific ratio_to_report one. Both engines now blame
                     ratio_to_report.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1026
        Goal: Check Exception message.
        """
        code = "GH_1026_3"
        script = self.LoadVTL(code)

        with pytest.raises(RunTimeError) as context:
            run(
                script=script,
                data_structures=self.filepath_json / f"{code}-1.json",
                datapoints={"DS_1": self.filepath_csv / f"{code}-1.csv"},
                return_only_persistent=False,
                use_duckdb=_use_duckdb_backend(),
            )
        assert str(context.value.args[1]) == "2-1-3-1"
        assert "ratio_to_report" in str(context.value.args[0])

    def test_GH_1026_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Va_str][calc m := ratio_to_report(Me_num
                        over (partition by Me_bool))];
        Description: partitioning by a Measure inside a calc clause reported
                     "Component Me_bool not found in Dataset __VDS_1__" because the
                     virtual dataset built for the component-level analytic keeps
                     only the identifiers and the operand, so the role check that
                     the dataset-level branch reaches (1-1-3-2, as rank gives) never
                     ran. The partition components are now checked against the real
                     dataset first.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1026
        Goal: Check Exception.
        """
        code = "GH_1026_4"
        number_inputs = 1
        message = "1-1-3-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1026_5(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Va_str][filter ratio_to_report(
                        ratio_to_report(Me_num over (partition by Id_2))
                        over (partition by Id_3)) >= 0.3];
        Description: an analytic nested inside another analytic of a filter
                     condition raised BinderException on the DuckDB engine, as the
                     whole predicate landed in a single SELECT; the nested analytic
                     is now materialised in a subquery below the predicate, as the
                     calc clause already does. The pandas engine already computed
                     it correctly.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1026
        Goal: Check Result.
        """
        code = "GH_1026_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1026_6(self):
        """
        Status: OK
        Expression: DS_r := DS_1[drop Va_str][calc m := first_value(
                        Me_num over (partition by Id_2 order by Me_bool, Id_1))];
        Description: ordering by a Measure inside a calc clause reported "Component
                     Me_bool not found in Dataset __VDS_1__" because the virtual
                     dataset built for the component-level analytic keeps only the
                     identifiers and the operand, while the dataset-level branch
                     orders by any existing component (e.g. rank over a Measure).
                     The components named in the ordering are now carried into the
                     virtual dataset, so both forms behave the same.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1026
        Goal: Check Result.
        """
        code = "GH_1026_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1069_1(self):
        """
        Status: OK
        Expression: DS_r1 <- nvl(DS_1, DS_2);
                    DS_r2 <- nvl(DS_1, DS_3);
                    DS_r3 <- DS_1[calc Me_2 := nvl(Me_1, Me_1)];
        Description: nvl declared its result as not nullable whatever the replacement
                     was, so a null replaced by another null came out as a null in a
                     Measure the structure said could not hold one. The result now
                     stays nullable where the replacement is itself nullable (DS_r1
                     and DS_r3), and is not nullable where the replacement cannot be
                     null (DS_r2).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1069
        Goal: Check Result.
        """
        code = "GH_1069_1"
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1130_1(self):
        """
        Status: OK
        Expression: DS_r1 <- nvl(DS_1, DS_2);
                    DS_r2 <- nvl(DS_1, DS_3);
        Description: nvl between two datasets filled the nulls of DS_1 with the value
                     found in the same row position of the second operand, not with
                     the value of the data point with the same identifiers. The
                     operands are now paired on their common identifiers and only the
                     data points found in both are kept, as in the other binary
                     operators. DS_2 holds the same data points as DS_1 in another
                     order (DS_r1); DS_3 misses Id_1 = 3 and adds Id_1 = 4 (DS_r2).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1130
        Goal: Check Result.
        """
        code = "GH_1130_1"
        number_inputs = 3
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1131_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 / DS_2 / DS_3;
        Description: DS_1 / DS_2 divides by zero at Id_1 = 1, which DS_3 does not have.
                     The DuckDB engine computed the division only for the data points
                     the join with DS_3 keeps and returned a result, while the pandas
                     engine raised. The division is now checked over all its data
                     points on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1131
        Goal: Check Exception.
        """
        code = "GH_1131_1"
        number_inputs = 3
        message = "2-1-15-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1131_2(self):
        """
        Status: OK
        Expression: DS_r <- (DS_1 / DS_2)[filter Id_1 = 2];
        Description: Same as GH_1131_1 with a filter dropping the data point that
                     divides by zero.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1131
        Goal: Check Exception.
        """
        code = "GH_1131_2"
        number_inputs = 2
        message = "2-1-15-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1131_3(self):
        """
        Status: OK
        Expression: DS_r <- sqrt(DS_1) + DS_2;
        Description: Same as GH_1131_1 with the square root of a negative value at the
                     data point that DS_2 does not have.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1131
        Goal: Check Exception.
        """
        code = "GH_1131_3"
        number_inputs = 2
        message = "2-1-15-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1131_4(self):
        """
        Status: OK
        Expression: DS_r <- ln(DS_1) + DS_2;
        Description: Same as GH_1131_1 with the logarithm of a negative value at the
                     data point that DS_2 does not have.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1131
        Goal: Check Exception.
        """
        code = "GH_1131_4"
        number_inputs = 2
        message = "2-1-15-8"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1131_5(self):
        """
        Status: OK
        Expression: DS_r <- intersect(DS_1 / DS_2, DS_3);
        Description: Same as GH_1131_1 with a set operation dropping the data point
                     that divides by zero.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1131
        Goal: Check Exception.
        """
        code = "GH_1131_5"
        number_inputs = 3
        message = "2-1-15-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1131_6(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := Me_1 / 0][filter Id_1 = 3];
        Description: The filter keeps no data point, so DuckDB could drop the whole
                     calc instead of computing its division by zero.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1131
        Goal: Check Exception.
        """
        code = "GH_1131_6"
        number_inputs = 1
        message = "2-1-15-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1191_1(self):
        """
        Status: OK
        Expression: DS_A <- DS_1;
                    DS_r <- sqrt(DS_1);
        Description: The square root of a negative value fails while the second statement runs.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1191
        Goal: Check the error names the output of the failing statement, and that the name is
              cleared afterwards.
        """
        code = "GH_1191_1"
        number_inputs = 1
        message = "2-1-15-2"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            exception_message="Please check transformation with output Dataset DS_r",
        )
        # An error raised after the failing run names no transformation
        assert "Please check transformation" not in SemanticError("1-2-5").args[0]

    def test_GH_1191_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1[calc Me_2 := random(Id_1, -1)];
        Description: A negative random index, which the DuckDB engine rejects while it
                     generates the SQL.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1191
        Goal: Check the error names the output of the failing statement.
        """
        code = "GH_1191_2"
        number_inputs = 1
        message = "1-1-15-9"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            exception_message="Please check transformation with output Dataset DS_r",
        )

    def test_GH_1192_27(self):
        """
        Status: OK
        Expression: DS_r <- lag(DS_1 over (order by Id_1));
        Description: lag without its offset is a script error, where a TypeError was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_27"
        number_inputs = 1
        message = "0-5-1-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GH_1192_28(self):
        """
        Status: OK
        Expression: DS_r <- sum(DS_1 over (order by Id_1 data points between unbounded preceding and unbounded preceding));
        Description: An analytic window bounded by two unbounded preceding is a script error,
                     where a plain Exception was raised.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1192
        Goal: Check Exception.
        """
        code = "GH_1192_28"
        number_inputs = 1
        message = "0-5-1-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )


class ValueDomainBugs(BugHelper):
    """Components defined on value domains (issue #1301)."""

    vd_flow = ["GH_1301-VD_FLOW"]
    vd_all = ["GH_1301-VD_FLOW", "GH_1301-VD_FLOW_2", "GH_1301-VD_INT"]

    def test_GH_1301_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter Me_1 > 0 ] [ calc Me_2 := Id_2 || "_X", Me_3 := Id_2 ];
        Description: Id_2 is defined on the value domain VD_FLOW, which the type of the
                     Component names. It keeps it through the clauses, as Me_3 does, which
                     is written alone, while Me_2 is a String.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_flow,
        )

    def test_GH_1301_2(self):
        """
        Status: OK
        Description: OTHER is not a value of VD_FLOW, the value domain of Id_2, so DS_1
                     does not load.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Exception.
        """
        code = "GH_1301_2"
        number_inputs = 1

        self.DataLoadExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="0-3-1-6",
            exception_message="Value OTHER is not one of the values of Value Domain VD_FLOW",
            vd_names=self.vd_flow,
        )

    def test_GH_1301_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1;
        Description: The type VD_UNKNOWN of Id_2 is neither a basic type nor a value domain
                     given.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Exception.
        """
        code = "GH_1301_3"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="0-2-1-1",
            exception_message="'VD_UNKNOWN' is not one of",
            vd_names=self.vd_flow,
        )

    def test_GH_1301_4(self):
        """
        Status: OK
        Expression: DS_r1 <- DS_1 + 1;
                    DS_r2 <- sum ( DS_1 group by Id_2 );
                    DS_r3 <- DS_1 [ keep Me_1 ] [ rename Me_1 to Me_3 ];
                    DS_r4 <- DS_1 # Me_1;
        Description: The values an operator computes are of the basic type of the value
                     domain, the Components it keeps as they are keep it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_4"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_all,
        )

    def test_GH_1301_5(self):
        """
        Status: OK
        Expression: DS_r1 <- union ( DS_1 , DS_2 );
                    DS_r2 <- full_join ( DS_1 as A , DS_2 as B rename B#Me_1 to Me_2 );
        Description: Id_2 is on VD_FLOW in DS_1 and a String in DS_2: the result is a String,
                     as a value domain is upcast to its basic type (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_5"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_flow,
        )

    def test_GH_1301_6(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_1 , DS_2 );
        Description: Id_2 is on VD_FLOW in DS_1 and a String in DS_2, so the result is a
                     String, and OTHER of DS_2 is kept. It raised 2-1-5-4, the union keeping
                     VD_FLOW (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_flow,
        )

    def test_GH_1301_7(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_1 , DS_2 );
        Description: Id_2 is on two value domains of Strings, so the result is a String, and
                     Me_1, on a value domain of Integers and a Number, a Number.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_all,
        )

    def test_GH_1301_8(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_1 , 2 ) ,
                                        Me_3 := if Me_1 > 1 then Me_1 else 3 ,
                                        Me_4 := case when Me_1 = 1 then 10 else Me_1 ,
                                        Me_5 := nvl ( Me_1 , 2.5 ) ];
        Description: nvl, if-then-else and case give an Integer for Me_1, on VD_INT, and an
                     Integer literal, and a Number for a Number (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_all,
        )

    def test_GH_1301_9(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_1 , 5 ) ];
        Description: nvl gives an Integer for Me_1, on VD_INT, and the literal 5, which is not
                     a value of VD_INT. It raised 2-1-5-4, nvl keeping VD_INT (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_all,
        )

    def test_GH_1301_10(self):
        """
        Status: OK
        Expression: DS_r <- left_join ( DS_2 as A , DS_1 as B rename B#Me_1 to Me_2 );
        Description: The join matches Id_2, a String in DS_2 and on VD_FLOW in DS_1, so Id_2 is
                     a String, and OTHER of DS_2 is kept. It raised 2-1-5-4, the join keeping
                     VD_FLOW (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_flow,
        )

    def test_GH_1301_11(self):
        """
        Status: OK
        Expression: define operator u ( a dataset , b dataset )
                      returns dataset is union ( a , b ) end operator;
                    DS_r1 <- u ( DS_1 , DS_2 );
                    DS_r2 <- u ( DS_1 , DS_3 );
        Description: The union of the operator gives a String for the two value domains of
                     the first call, and for VD_FLOW and a String in the second one
                     (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_11"
        number_inputs = 3
        references_names = ["1", "2"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.vd_all,
        )

    def test_GH_1301_12(self):
        """
        Status: OK
        Expression: DS_r <- if DS_1 then DS_2 else DS_3;
        Description: DS_2 is on VD_DATE and DS_3 a Date, so the result is a Date, and
                     2022-03-31 of DS_3, not a value of VD_DATE, is kept. It raised 2-1-5-4,
                     the if keeping VD_DATE (issue #1341).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1301
        Goal: Check Result.
        """
        code = "GH_1301_12"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1301-VD_DATE"],
        )

    vd_1341 = ["GH_1341-VD_CTY"]

    def test_GH_1341_1(self):
        """
        Status: OK
        Expression: DS_r1 <- union ( DS_1 , DS_2 ); DS_r2 <- union ( DS_2 , DS_1 );
                    DS_r3 <- setdiff ( DS_2 , DS_1 ); DS_r4 <- symdiff ( DS_1 , DS_2 );
                    DS_r5 <- intersect ( DS_1 , DS_2 ); DS_r6 <- setdiff ( DS_1 , DS_2 );
                    DS_r7 <- union ( DS_1 , DS_1 [ calc Me_1 := Me_1 * 10 ] );
        Description: Id_1 is on VD_CTY in DS_1 and a String in DS_2, so the set operators give
                     a String and keep IT, of DS_2, which is not a value of VD_CTY. Two operands
                     on VD_CTY keep it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1341
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1341_1",
            number_inputs=2,
            references_names=["1", "2", "3", "4", "5", "6", "7"],
            vd_names=self.vd_1341,
        )

    def test_GH_1341_2(self):
        """
        Status: OK
        Expression: DS_r1 <- left_join ( DS_2 as A , DS_1 as B rename B#Me_1 to Me_2 );
                    DS_r2 <- full_join ( DS_1 as A , DS_2 as B rename B#Me_1 to Me_2 );
                    DS_r3 <- inner_join ( DS_1 as A , DS_2 as B rename B#Me_1 to Me_2 );
                    DS_r4 <- left_join ( DS_1 as A , DS_2 as B rename B#Me_1 to Me_2 );
        Description: The joins match Id_1, on VD_CTY in DS_1 and a String in DS_2, so Id_1 is a
                     String and IT of DS_2 is kept.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1341
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1341_2",
            number_inputs=2,
            references_names=["1", "2", "3", "4"],
            vd_names=self.vd_1341,
        )

    def test_GH_1341_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_v , "XX" ) ,
                                        Me_3 := if Me_1 > 1 then Me_s else Me_v ,
                                        Me_4 := case when Me_1 > 1 then Me_s else Me_v ,
                                        Me_5 := nvl ( Me_v , Me_v ) ];
        Description: Me_v is on VD_CTY and Me_s and "XX" are Strings, so nvl, if-then-else and
                     case give a String and keep XX, which is not a value of VD_CTY. Me_v with
                     itself keeps VD_CTY.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1341
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1341_3",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1341,
        )

    def test_GH_1341_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 + DS_2;
        Description: The Identifier Id_1 is on VD_CTY in DS_1 and a String in DS_2, so it is a
                     String in the result.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1341
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1341_4",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1341,
        )

    def test_GH_1236_1(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_FLOW , VD_AMOUNT ) is ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Id_2 , Me_1 all );
        Description: The components of the call are bound in order to the value domains of the
                     ruleset, which the rules name. Id_2 and Me_1 are defined on them (#1338).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Result.
        """
        code = "GH_1236_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1236-VD_FLOW", "GH_1236-VD_AMOUNT"],
        )

    def test_GH_1236_2(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_FLOW as F , VD_AMOUNT as A )
                    is r1 : when F = "CREDIT" then A >= 0 errorcode "credit below zero" ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Id_2 , Me_1 );
        Description: The rules name the value domains by their aliases, and the components of
                     the call are defined on them.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Result.
        """
        code = "GH_1236_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1236-VD_FLOW", "GH_1236-VD_AMOUNT"],
        )

    def test_GH_1236_3(self):
        """
        Status: OK
        Expression: define datapoint ruleset DR_1 ( valuedomain num , str ) is
                    r1 : when str = "X" then num > 0 end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DR_1 components Me_1 , Me_2 all );
        Description: The value domains num and str of the ruleset are not given, which the
                     ruleset needs (issue #1338).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_3"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code="1-2-22"
        )

    def test_GH_1236_4(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , DPR_1 components Id_2 , Me_1 );
        Description: Id_2, a String, and Me_1, an Integer, are of the basic types of VD_FLOW
                     and VD_AMOUNT, the value domains they are bound to, so the ruleset applies
                     to them and they keep their basic types (issue #1348). It raised 1-1-10-12,
                     as they are not defined on the value domains (issue #1338).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1236_4",
            number_inputs=1,
            references_names=["1"],
            vd_names=["GH_1236-VD_FLOW", "GH_1236-VD_AMOUNT"],
        )

    def test_GH_1236_5(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 , Id_2 );
        Description: Me_1, an Integer, is bound to VD_FLOW, a value domain of Strings.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_5"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-10-12",
            exception_message="Component Me_1 of type Integer is not defined on the Value Domain "
            "VD_FLOW",
            vd_names=["GH_1236-VD_FLOW", "GH_1236-VD_AMOUNT"],
        )

    def test_GH_1236_6(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , DPR_1 components Id_2 );
        Description: The call names one component for the two value domains of the ruleset.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_6"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-10-15",
            vd_names=["GH_1236-VD_FLOW", "GH_1236-VD_AMOUNT"],
        )

    def test_GH_1236_7(self):
        """
        Status: OK
        Expression: define operator with_suffix ( x component < VD_FLOW > ) ... end operator;
                    define operator flag ( x VD_FLOW ) ... end operator;
                    DS_r <- DS_1 [ calc Me_2 := with_suffix ( Id_2 ) , Me_3 := flag ( "DEBIT" ) ];
        Description: A value domain types a component and a scalar parameter, which take the
                     values of its basic type that are values of it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Result.
        """
        code = "GH_1236_7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1236-VD_FLOW"],
        )

    def test_GH_1236_8(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := with_suffix ( Id_2 ) ];
        Description: OTHER, a value of Id_2, is not one of VD_FLOW, which types the parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_8"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="2-1-5-4",
            exception_message="At op with_suffix: Cannot cast OTHER of String to Value Domain",
            vd_names=["GH_1236-VD_FLOW"],
        )

    def test_GH_1236_9(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := with_suffix ( Me_1 ) ];
        Description: Me_1, an Integer, is no argument of a parameter typed by VD_FLOW, a value
                     domain of Strings.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_9"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-2-18",
            exception_message="Expected String, got Integer on UDO with_suffix, parameter x",
            vd_names=["GH_1236-VD_FLOW"],
        )

    def test_GH_1236_10(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := flag ( "OTHER" ) ];
        Description: The literal OTHER is not one of VD_FLOW, which types the parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_10"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="2-1-5-4",
            exception_message="At op flag: Cannot cast OTHER of String to Value Domain VD_FLOW",
            vd_names=["GH_1236-VD_FLOW"],
        )

    def test_GH_1236_11(self):
        """
        Status: OK
        Expression: define datapoint ruleset dpr ( valuedomain Me_1 , Me_2 ) is
                    r1 : when Me_2 = "X" then Me_1 > 0 end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , dpr components Me_2 , Me_1 all );
        Description: The value domains Me_1 and Me_2 are named as components of DS_1 but not
                     given. They were read as those components for backwards compatibility,
                     which the spec does not allow (issue #1338).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_11"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code="1-2-22"
        )

    def test_GH_1236_12(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr components Me_1 all );
        Description: The call lists fewer components than the value domains of the ruleset,
                     which were read as the components named as them for backwards
                     compatibility, which the spec does not allow (issue #1338).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_12"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code="1-1-10-15"
        )

    def test_GH_1236_13(self):
        """
        Status: OK
        Expression: define hierarchical ruleset hr ( valuedomain condition At_1 rule Id_2 ) is
                    when At_1 = "x" then A = B + C end hierarchical ruleset;
                    DS_r <- check_hierarchy ( DS_1 , hr condition At_1 rule Id_2 all );
        Description: The value domains At_1 and Id_2 are named as components of DS_1 but not
                     given. They were read as those components for backwards compatibility,
                     which the spec does not allow (issue #1339).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1236
        Goal: Check Exception.
        """
        code = "GH_1236_13"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code="1-2-22"
        )

    error_vds = ["GH_1316-errorcode_vd", "GH_1316-errorlevel_vd"]

    def test_GH_1316_1(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is
                    R1 : Me_1 > 0 errorcode 100 errorlevel "ERROR" ;
                    R2 : Me_1 < 100 errorcode 200 errorlevel "WARNING" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 );
        Description: errorcode_vd and errorlevel_vd are given, so errorcode is an Integer and
                     errorlevel a String, their basic types, and the codes and levels are
                     values of them.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1316
        Goal: Check Result.
        """
        code = "GH_1316_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.error_vds,
        )

    def test_GH_1316_2(self):
        """
        Status: OK
        Expression: R2 : Me_1 < 100 errorcode 300 errorlevel "FATAL"
        Description: The error code 300 of R2 is not a value of errorcode_vd.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1316
        Goal: Check Exception.
        """
        code = "GH_1316_2"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-10-13",
            exception_message="The errorcode 300 of rule R2 of DPR_1 is not one of the values of "
            "the Value Domain errorcode_vd",
            vd_names=self.error_vds,
        )

    def test_GH_1316_3(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( valuedomain rule Id_2 ) is
                    R1 : A = B + C errorcode 200 errorlevel 2 end hierarchical ruleset;
                    DS_r <- check_hierarchy ( DS_1 , HR_1 rule Id_2 );
        Description: check_hierarchy gives errorcode and errorlevel the Integer types of
                     errorcode_vd and errorlevel_vd.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1316
        Goal: Check Result.
        """
        code = "GH_1316_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1316-errorcode_vd", "GH_1316-errorlevel_vd_int", "GH_1316_3-Id_2"],
        )

    def test_GH_1316_4(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 errorcode 200 errorlevel "ERROR" invalid );
        Description: check gives errorcode and errorlevel the types of errorcode_vd and
                     errorlevel_vd.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1316
        Goal: Check Result.
        """
        code = "GH_1316_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=self.error_vds,
        )

    def test_GH_1316_5(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 errorcode "E1" errorlevel "ERROR" invalid );
        Description: The error code E1 of check is not a value of errorcode_vd.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1316
        Goal: Check Exception.
        """
        code = "GH_1316_5"
        number_inputs = 1

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code="1-1-10-13",
            exception_message='The errorcode "E1" of check is not one of the values of the Value '
            "Domain errorcode_vd",
            vd_names=self.error_vds,
        )

    def test_GH_1316_6(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is
                    R1 : Me_1 > 0 errorcode 100 ; R2 : Me_1 < 100 errorcode 200
                    end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 );
        Description: Only errorcode_vd is given: errorcode is an Integer, and errorlevel keeps
                     the type it has without errorlevel_vd.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1316
        Goal: Check Result.
        """
        code = "GH_1316_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=["GH_1316-errorcode_vd"],
        )

    vd_1338 = ["GH_1338-VD_CTY", "GH_1338-VD_CTY2", "GH_1338-VD_INT"]

    def test_GH_1338_1(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr_cty all );
        Description: dpr_cty is defined on value domains, so the call must list the components,
                     although DS_1 has a component named as the value domain VD_CTY.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1338
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1338_1",
            number_inputs=1,
            exception_code="1-1-10-14",
            exception_message="The ruleset dpr_cty is defined on value domains, so the call must "
            "list the components bound to them",
            vd_names=self.vd_1338,
        )

    def test_GH_1338_2(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr_cty components Me_1 all );
        Description: The rules read Me_1, the component of the call, and not the component
                     named as the value domain VD_CTY. The constant consequent of the rule
                     raised an AttributeError on the pandas engine.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1338
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1338_2", number_inputs=1, references_names=["1"], vd_names=self.vd_1338
        )

    def test_GH_1338_3(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr_cty components Me_3 all );
        Description: Me_3, a String, is of the basic type of VD_CTY, the value domain it is
                     bound to, so the ruleset applies to it (issue #1348); it raised 1-1-10-12.
                     The value IT, outside VD_CTY, is checked as any other.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1338
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1338_3",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1338,
        )

    def test_GH_1338_4(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr_cty components Me_2 all );
        Description: Me_2 is defined on VD_CTY2, a value domain of the same basic type as
                     VD_CTY, the one it is bound to.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1338
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1338_4",
            number_inputs=1,
            exception_code="1-1-10-12",
            exception_message="Component Me_2 of type VD_CTY2 is not defined on the Value Domain "
            "VD_CTY",
            vd_names=self.vd_1338,
        )

    def test_GH_1338_5(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr_nope components Me_1 all );
        Description: VD_NOPE, the value domain of dpr_nope, is not given.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1338
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1338_5",
            number_inputs=1,
            exception_code="1-2-8",
            exception_message="Missing value domain 'VD_NOPE' definition",
            vd_names=self.vd_1338,
        )

    def test_GH_1338_6(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , dpr_two components Me_1 all );
        Description: dpr_two is defined on two value domains, and the call lists one component.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1338
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1338_6",
            number_inputs=1,
            exception_code="1-1-10-15",
            exception_message="The ruleset dpr_two is defined on 2 value domains, but the call "
            "lists 1 components",
            vd_names=self.vd_1338,
        )

    vd_1339 = ["GH_1339-VD_CTY", "GH_1339-VD_CTY2", "GH_1339-VD_YEAR"]

    def test_GH_1339_1(self):
        """
        Status: OK
        Expression: DS_r <- check_hierarchy ( DS_1 , hr_cty rule Id_1 all );
        Description: hr_cty is defined on the value domain VD_CTY, and Id_1 is a String
                     component, of the basic type of VD_CTY, so the ruleset applies to it and
                     Id_1 stays a String (issue #1348); the value IT, outside VD_CTY, is not a
                     code item of it. It raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1339
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1339_1",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1339,
        )

    def test_GH_1339_2(self):
        """
        Status: OK
        Expression: DS_r <- hierarchy ( DS_1 , hr_cty rule Id_1 );
        Description: Id_1 is defined on VD_CTY2, which has the basic type of VD_CTY and more
                     values. hierarchy kept the values outside VD_CTY.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1339
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1339_2",
            number_inputs=1,
            exception_code="1-1-10-12",
            exception_message="Component Id_1 of type VD_CTY2 is not defined on the Value Domain "
            "VD_CTY of the ruleset hr_cty",
            vd_names=self.vd_1339,
        )

    def test_GH_1339_4(self):
        """
        Status: OK
        Expression: DS_r <- check_hierarchy ( DS_1 , hr_cond condition Id_2 rule Id_1 all );
        Description: The condition component Id_2 is an Integer component, of the basic type
                     of the value domain VD_YEAR of the condition of hr_cond, so the ruleset
                     applies to it and Id_2 stays an Integer (issue #1348). It raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1339
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1339_4",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1339,
        )

    def test_GH_1339_7(self):
        """
        Status: OK
        Expression: DS_r <- check_hierarchy ( DS_1 , hr_cond condition Id_9 rule Id_1 all );
        Description: The condition component Id_9 is not in DS_1. The message named the alias Y
                     of the condition instead.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1339
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1339_7",
            number_inputs=1,
            exception_code="1-1-1-10",
            exception_message="Component Id_9 not found in Dataset DS_1",
            vd_names=self.vd_1339,
        )

    vd_1350 = [
        "GH_1350-VD_1",
        "GH_1350-VD_2",
        "GH_1350-VD_TP",
        "GH_1350-VD_TP1",
        "GH_1350-VD_TI",
    ]

    def test_GH_1350_1a(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := cast ( Me_1 , VD_TP ) ];
        Description: A String cast into VD_TP, a Time_Period value domain of two values, gives
                     Me_3 on VD_TP. On DuckDB, the commas of the setlist split the argument of
                     the macro that checks it into several arguments (ParserException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_1a",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_1b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := cast ( Me_2 , VD_TI ) ];
        Description: A Time_Period cast into VD_TI, a Time value domain of two values, gives Me_3 on
                     VD_TI. On DuckDB, it raised a ParserException as 1a.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_1b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_1c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter Id_1 = 1 ] [ calc Me_3 := cast ( Me_1 , VD_TP1 ) ];
        Description: A String cast into VD_TP1, a Time_Period value domain of one value, gives Me_3
                     on VD_TP1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_1c",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_1d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := cast ( Me_1 , time_period ) ];
        Description: A String cast into the basic type of VD_TP gives a Time_Period.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_1d",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_2a(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier Id_2 := cast ( Me_2 , VD_1 ) ]
                                 [ pivot Id_2 , Me_1 ];
        Description: Id_2, created on VD_1 by the calc of the same statement, is pivoted into a
                     Measure per value of VD_1. On DuckDB, the calc lost the value domain of Id_2
                     (AttributeError).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_2a",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_2b(self):
        """
        Status: OK
        Expression: DS_a <- DS_1 [ calc identifier Id_2 := cast ( Me_2 , VD_1 ) ];
                    DS_r <- DS_a [ pivot Id_2 , Me_1 ];
        Description: The same pivot as 2a, in two statements.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_2b",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3a(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : between ( X , 1 , 2 ) errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 invalid );
        Description: between reads the alias X of VD_2, bound to Me_1. On DuckDB, X was written into
                     the SQL instead of Me_1 (BinderException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3a",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3b(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_1 as X ) is
                        r1 : substr ( X , 1 , 1 ) <> "C" errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_2 invalid );
        Description: substr reads the alias X of VD_1, bound to Me_2 (BinderException on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3c(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : not ( X = 3 ) errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 invalid );
        Description: not of a parenthesised condition reads the alias X of VD_2, bound to Me_1
                     (BinderException on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3c",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3d(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 as X ) is
                        r1 : between ( X , 1 , 2 ) errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: between reads X, the alias of the variable Me_1 (BinderException on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3d",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3e(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is
                        r1 : between ( Me_1 , 1 , 2 ) errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: between reads the variable Me_1, which has no alias.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3e",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3f(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X >= 1 and X <= 2 errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 invalid );
        Description: Comparisons and and read the alias X of VD_2, bound to Me_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3f",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3g(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 , VD_1 ) is
                        r1 : cast ( VD_2 , string ) <> "3" ...;
                        r2 : case when VD_2 = 3 then false else true ...;
                        r3 : round ( VD_2 * 1.2 , 0 ) <> 4 ...;
                        r4 : trunc ( VD_2 * 1.5 , 0 ) <> 4 ...;
                        r5 : replace ( VD_1 , "C" , "Z" ) <> "Z" ...;
                        r6 : instr ( VD_1 , "C" ) = 0 ...;
                        r7 : ( VD_2 < 3 ) ...
                    end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 , Me_2 invalid );
        Description: cast, case, round, trunc, replace, instr and a parenthesised condition read
                     the value domains VD_2 and VD_1, which have no alias, bound to Me_1 and Me_2
                     (BinderException on DuckDB). Each rule fails on the data point 3 only.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3g",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3h(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : period_indicator ( Me_1 ) = cast ( "Q" , duration ) ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: period_indicator of the Time_Period Me_1, in a ruleset on variables.
                     DuckDB wrote it as a plain function, which does not exist
                     (CatalogException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3h",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3i(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : getyear ( Me_1 ) > 2020 ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: getyear of the Time_Period Me_1. DuckDB applied year to its text
                     (BinderException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3i",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3j(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : time_agg ( "A" , _ , Me_1 ) = cast ( "2020A" , time_period ) ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: time_agg of the Time_Period Me_1 (BinderException on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3j",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3k(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : random ( Me_2 , 1 ) >= 0 ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: random of the Integer Me_2: no data point is invalid. DuckDB wrote it as
                     its own random function (BinderException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3k",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3l(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : cast ( Me_1 , string ) = "2020Q1" ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: cast of the Time_Period Me_1 to string gives 2020Q1, as in calc. DuckDB
                     gave its internal text 2020-Q1, so both data points were invalid.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3l",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3m(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : when cast ( Me_1 , string ) = "2020Q1" then Me_2 > 5 ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: The same cast in the when condition. DuckDB gave no invalid data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3m",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3n(self):
        """
        Status: OK
        Expression: define operator f ( x component ) returns component is x > 5 end operator;
                    define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 as X ) is
                        r1 : f ( X ) ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: A user defined operator called on X, the alias of Me_2. DuckDB wrote X
                     into the SQL (BinderException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3n",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3o(self):
        """
        Status: OK
        Expression: define operator f ( x component ) returns component is x > 5 end operator;
                    define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : f ( Me_2 ) ...
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: A user defined operator called on the variable Me_2, which has no alias.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3o",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_3p(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 , Me_2 ) is
                        r1 : getmonth ( Me_1 ) < 6 ...;
                        r2 : dayofmonth ( Me_1 ) <> 31 ...;
                        r3 : dayofyear ( Me_1 ) < 100 ...
                    end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 invalid );
        Description: getmonth, dayofmonth and dayofyear of the Time_Period Me_1 read the end of
                     the period, 2020-03-31 and 2021-12-31, as in calc. DuckDB applied its own
                     functions to the text of Me_1 (BinderException).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_3p",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4a(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 [ calc Me_3 := Me_1 ] ,
                                              DPR_1 components Me_3 invalid );
        Description: Me_3, written alone in calc from Me_1, keeps VD_2, the value domain of the
                     ruleset. DuckDB gave Me_3 the basic type and raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4a",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4b(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( inner_join ( DS_1 as d calc Me_3 := Me_1 ) ,
                                                DPR_1 components Me_3 invalid );
        Description: The same as 4a, with the calc inside a join.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4c(self):
        """
        Status: OK
        Expression: DS_a <- DS_1 [ calc Me_3 := Me_1 ];
                    DS_r <- check_datapoint ( DS_a , DPR_1 components Me_3 invalid );
        Description: The same as 4a, in two statements.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4c",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_5a(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_2 ) is aggregate avg ...;
                    DS_r <- DS_1 + DS_2;
        Description: At_1 is on VD_2 in both operands, so the rule on VD_2 averages it into a
                     Number. DuckDB looked the rule up on the result At_1, without the value
                     domain, and gave null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_5a",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_5b(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_2 ) is aggregate avg ...;
                    DS_r <- DS_1 + DS_2 + DS_1;
        Description: The first + averages At_1 into a Number on no value domain, so the second
                     + combines it with At_1 on VD_2. The rule on VD_2 does not apply when an
                     operand is not on VD_2, and there is no rule on the variable At_1. Both
                     engines gave null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1350_5b",
            number_inputs=2,
            exception_code="1-3-3-6",
            exception_message="Viral attribute At_1 has no viral propagation rule",
            vd_names=self.vd_1350,
        )

    def test_GH_1350_5c(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( variable At_1 ) is aggregate avg ...;
                    DS_r <- DS_1 + DS_2;
        Description: The rule on the variable At_1 averages it into a Number.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_5c",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_5d(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( variable At_1 ) is aggregate avg ...;
                    DS_r <- DS_1 + DS_2 + DS_1;
        Description: The rule on the variable At_1 applies to both +: the average of the average.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_5d",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_2c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier Me_2 := Me_2 ] [ pivot Me_2 , Me_4 ];
        Description: Me_2, a Measure on VD_1, becomes an Identifier on VD_1 in the calc of the same
                     statement, and is pivoted into a Measure per value of VD_1. On DuckDB, the calc
                     rebuilt Me_2 without its value domain (AttributeError, as 2a).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_2c",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4d(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 [ calc attribute Me_4 := Me_1 ] ,
                                                DPR_1 components Me_4 invalid );
        Description: Me_4, an Integer Measure, becomes an Attribute written from Me_1 alone, so
                     it is on VD_2. DuckDB rebuilt it with its old type and no value domain and
                     raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4d",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4e(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 [ calc Me_5 := cast ( Me_5 , VD_2 ) ] ,
                                                DPR_1 components Me_5 invalid );
        Description: Me_5, a String Measure, is overwritten by its cast into VD_2. DuckDB kept the
                     String of the operand and raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4e",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_5e(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_2 ) is aggregate avg ...;
                    DS_r <- case when DS_1 > 1 then DS_1 else DS_2;
        Description: At_1 is on VD_2 in both operands of the case, so the rule on VD_2 averages it
                     into a Number. DuckDB looked the rule up on the result At_1, without the value
                     domain, and gave null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_5e",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4f(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 [ calc Me_5 := nvl ( Me_1 , Me_5 ) ] ,
                                                DPR_1 components Me_5 invalid );
        Description: Me_5 is overwritten by nvl of Me_1 and Me_5, both on VD_2, so it keeps VD_2.
                     DuckDB gave it the basic type and raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4f",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4g(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r1 <- check_datapoint ( DS_1 [ calc Me_5 := if Me_1 > 1 then Me_5
                                                                       else Me_1 ] ,
                                                 DPR_1 components Me_5 invalid );
                    DS_r2 <- check_datapoint ( DS_1 [ calc Me_5 := case when Me_1 > 1 then Me_5
                                                                         else Me_1 ] , ... );
        Description: The same overwrite with if-then-else and with case: their operands Me_5
                     and Me_1 are on VD_2, so Me_5 keeps VD_2 (1-1-10-12 on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4g",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4h(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 [ calc Me_6 := nvl ( Me_1 , Me_5 ) ] ,
                                                DPR_1 components Me_6 invalid );
        Description: Me_6, a new Component from nvl of Me_1 and Me_5, both on VD_2, is on VD_2
                     (1-1-10-12 on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4h",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4i(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( inner_join ( DS_1 as d , DS_2 as e
                                                             calc Me_5 := d#Me_1
                                                             drop d#Me_1 , e#Me_1 ) ,
                                                DPR_1 components Me_5 invalid );
        Description: Me_5 is overwritten by d#Me_1, a Component on VD_2 written alone in the calc of
                     a join, so it keeps VD_2 (1-1-10-12 on DuckDB).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4i",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1350,
        )

    def test_GH_1350_4j(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_2 as X ) is
                        r1 : X > 1 errorcode "E1" end datapoint ruleset;
                    DS_r1 <- check_datapoint ( DS_1 [ calc Me_5 := nvl ( Me_1 , SC_1 ) ] ,
                                               DPR_1 components Me_5 invalid );
                    DS_r2 <- check_datapoint ( DS_1 [ calc Me_6 := SC_1 ] ,
                                               DPR_1 components Me_6 invalid );
        Description: SC_1 is a scalar on VD_2, as are Me_1 and Me_5, so nvl of Me_1 and SC_1
                     and SC_1 written alone keep VD_2. DuckDB gave Me_5 and Me_6 the basic type
                     and raised 1-1-10-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1350
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1350_4j",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1350,
            scalars={"SC_1": 1},
        )

    vd_1348 = ["GH_1348-VD_TP", "GH_1348-VD_INT", "GH_1348-VD_STR"]

    def test_GH_1348_1(self):
        """
        Status: OK
        Expression: DS_r <- sum ( DS_1 group all time_agg ( "A" ) );
        Description: time_agg in the grouping writes the aggregated periods into Id_2, which
                     is on VD_TP. The period 2024 is not a value of VD_TP, so Id_2 has the
                     basic type Time_Period; it kept VD_TP.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_1",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_2(self):
        """
        Status: OK
        Expression: DS_r <- sum ( DS_1 group all time_agg ( "A" ) ) [ pivot Id_2 , Me_1 ];
        Description: Id_2 holds the aggregated period 2024 with the basic type Time_Period, so
                     pivot raises as in case 3. While Id_2 kept VD_TP, pivot gave the Measures
                     2024Q1 and 2024Q3, both null, and the data point was lost.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_2",
            number_inputs=1,
            exception_code="1-1-6-18",
            exception_message="The Identifier Id_2 is not defined on a value domain",
            vd_names=self.vd_1348,
        )

    def test_GH_1348_3(self):
        """
        Status: OK
        Expression: DS_r <- sum ( DS_1 group all time_agg ( "A" ) ) [ pivot Id_2 , Me_1 ];
        Description: Control of case 2 with the basic types: Id_2 is a Time_Period, which is
                     on no value domain, so pivot raises.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_3",
            number_inputs=1,
            exception_code="1-1-6-18",
            exception_message="The Identifier Id_2 is not defined on a value domain",
            vd_names=self.vd_1348,
        )

    def test_GH_1348_4(self):
        """
        Status: OK
        Expression: DS_r <- time_agg ( "A" , _ , DS_1 );
        Description: time_agg on a Data Set writes the aggregated periods into the time
                     Measure Me_1, which is on VD_TP, so Me_1 has the basic type Time_Period;
                     it kept VD_TP.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_4",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_5(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 1 );
        Description: timeshift writes the shifted periods into the time Identifier Id_2, which
                     is on VD_TP, so Id_2 has the basic type Time_Period; it kept VD_TP.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_5",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_6(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , all );
        Description: fill_time_series writes the filled periods into the time Identifier Id_2,
                     which is on VD_TP, so Id_2 has the basic type Time_Period; it kept VD_TP.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_6",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_7(self):
        """
        Status: OK
        Expression: DS_r <- dateadd ( DS_1 , 1 , "M" );
        Description: dateadd on a Data Set writes Dates into the time Identifier Id_2, which is
                     on VD_TP, so Id_2 has the basic type Date; it kept VD_TP.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_7",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_8(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( variable rule Id_2 ) is
                        r1 : C = A + B end hierarchical ruleset;
                    DS_r <- hierarchy ( DS_1 , HR_1 rule Id_2 non_null all )
                                [ calc Me_2 := Id_2 in VD_STR ];
        Description: hierarchy writes the computed code item C into the rule Identifier Id_2,
                     which is on VD_STR, so Id_2 has the basic type String; it kept VD_STR,
                     while Id_2 in VD_STR is false on C.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_8",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_8b(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( valuedomain rule VD_STR ) is
                        r1 : C = A + B end hierarchical ruleset;
                    DS_r <- hierarchy ( DS_1 , HR_1 rule Id_2 non_null all );
        Description: The ruleset is defined on VD_STR, whose values are A and B, and its rule
                     computes the code item C, which is not one of them, so the ruleset gives
                     values outside its value domain. hierarchy computed C on Id_2, on VD_STR.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_8b",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message="The code item C of the hierarchical ruleset HR_1 is not one of "
            "the values of the Value Domain VD_STR",
            vd_names=self.vd_1348,
        )

    def test_GH_1348_8c(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( valuedomain rule VD_STR ) is
                        r1 : A = B + C end hierarchical ruleset;
                    DS_r <- check_hierarchy ( DS_1 , HR_1 rule Id_2 always_zero all );
        Description: Case 8b for check_hierarchy, with the code item C, not a value of
                     VD_STR, on the right of the rule: it was read as a missing 0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_8c",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message="The code item C of the hierarchical ruleset HR_1 is not one of "
            "the values of the Value Domain VD_STR",
            vd_names=self.vd_1348,
        )

    vd_1348_n = [*vd_1348, "GH_1348-VD_STR2", "GH_1348-VD_NUM"]

    def test_GH_1348_8d(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( valuedomain condition VD_NUM as Y
                        rule VD_STR ) is r1 : when Y > 2020 then A = B
                    end hierarchical ruleset;
                    DS_r <- hierarchy ( DS_1 , HR_1 condition Id_3 rule Id_2 all );
        Description: The condition Component Id_3 is a String, a basic type that does not
                     promote to Number, the basic type of the value domain VD_NUM it is bound
                     to.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_8d",
            number_inputs=1,
            exception_code="1-1-10-12",
            exception_message="Component Id_3 of type String is not defined on the Value Domain "
            "VD_NUM of the ruleset HR_1, nor of a basic type that promotes to its type Number",
            vd_names=self.vd_1348_n,
        )

    def test_GH_1348_8e(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( valuedomain condition VD_NUM as Y
                        rule VD_STR ) is r1 : when Y > 2020 then A = B
                    end hierarchical ruleset;
                    DS_r <- hierarchy ( DS_1 , HR_1 condition Id_3 rule Id_2 all );
        Description: The condition Component Id_3 is an Integer, which promotes to Number,
                     the basic type of VD_NUM, so the ruleset applies to it: A is computed in
                     2021 only, and Id_3 stays an Integer. It raised 1-1-10-12, as Id_3 is not
                     defined on VD_NUM.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_8e",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348_n,
        )

    def test_GH_1348_8f(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_STR as S , VD_INT as I ) is
                        r1 : when S = "A" then I > 1 errorcode "A_low"
                    end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 , Me_2 all_measures );
        Description: Me_1 is a String, the basic type of VD_STR, and Me_2 is on VD_INT, so the
                     ruleset applies to both, and Me_1 stays a String. It raised 1-1-10-12, as
                     Me_1 is not defined on VD_STR.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_8f",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348_n,
        )

    def test_GH_1348_8g(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_STR as S , VD_INT as I ) is
                        r1 : when S = "A" then I > 1 errorcode "A_low"
                    end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 , Me_2 all_measures );
        Description: Me_1 is on VD_STR2, another value domain of the basic type of VD_STR, so
                     it is rejected, as a basic String would not be.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_8g",
            number_inputs=1,
            exception_code="1-1-10-12",
            exception_message="Component Me_1 of type VD_STR2 is not defined on the Value Domain "
            "VD_STR of the ruleset DPR_1",
            vd_names=self.vd_1348_n,
        )

    def test_GH_1348_9(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( variable rule Id_2 ) is
                        r1 : C = A + B end hierarchical ruleset;
                    DS_r <- check_hierarchy ( DS_1 , HR_1 rule Id_2 always_zero all_measures );
        Description: check_hierarchy writes the computed code item C into the rule Identifier
                     Id_2, on VD_STR, and the 0 of the missing C into the Measure Me_1, on
                     VD_INT, so Id_2 is a String and Me_1 an Integer, as in hierarchy; they
                     kept their value domains.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_9",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_10(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_INT ) is
                        aggregate sum end viral propagation;
                    DS_r <- DS_1 + DS_2;
        Description: At_1 is on VD_INT in both operands, so the rule on VD_INT sums it. The
                     sums 5 and 6 are not values of VD_INT, so At_1 has the basic type Integer;
                     it kept VD_INT.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_10",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_9b(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( variable rule Id_2 ) is
                        r1 : C = A + B end hierarchical ruleset;
                    DS_r <- check_hierarchy ( DS_1 , HR_1 rule Id_2 non_zero all_measures );
        Description: non_zero also writes the 0 of the missing C into the Measure Me_1, on
                     VD_INT, so Me_1 is an Integer, as in case 9; it kept VD_INT.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_9b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_11(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b apply nvl ( a , b ) );
        Description: Me_1 is on VD_INT in both operands and nvl only puts their values into it,
                     so Me_1 keeps VD_INT, as in nvl ( DS_1 , DS_2 ); it had the basic type
                     Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_11",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_12(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Control of case 11: Me_1 is on VD_INT in both operands, so it keeps VD_INT.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_12",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_13(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b
                                calc Me_2 := nvl ( a#Me_1 , b#Me_1 ) keep Me_2 );
        Description: Control of case 11: both operands of nvl are on VD_INT, so Me_2 is on
                     VD_INT.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_13",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_14(self):
        """
        Status: OK
        Expression: DS_r <- flow_to_stock ( DS_1 );
        Description: flow_to_stock only accumulates the number Measures, so the String Measures
                     Me_1 and Me_2 keep VD_STR, and Me_3, on VD_INT, holds the sums with the
                     basic type Integer. Me_1 and Me_2 had the basic type String.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_14",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_15(self):
        """
        Status: OK
        Expression: DS_r <- stock_to_flow ( DS_1 );
        Description: stock_to_flow only takes the differences of the number Measures, so the
                     String Measures Me_1 and Me_2 keep VD_STR, and Me_3, on VD_INT, has the
                     basic type Integer. Me_1 and Me_2 had the basic type String.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_15",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_14b(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_STR as X ) is
                        r1 : X = "A" errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( flow_to_stock ( DS_1 ) , DPR_1 components Me_1 all );
        Description: Me_1 keeps VD_STR through flow_to_stock (case 14), so the ruleset on VD_STR
                     applies to it; it raised 1-1-10-12, while it works on the input.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_14b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_16(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ unpivot Id_3 , Me_3 ];
        Description: Both unpivoted Measures are on VD_STR, so the new Measure Me_3 keeps VD_STR;
                     it had the basic type String. The new Identifier Id_3 is a String.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_16",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_16b(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_STR as X ) is
                        r1 : X = "A" errorcode "E1" end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 [ unpivot Id_3 , Me_3 ] ,
                                DPR_1 components Me_3 all );
        Description: Me_3 keeps VD_STR (case 16), so the ruleset on VD_STR applies to it; it
                     raised 1-1-10-12, while it works on the input Measures.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_16b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_17(self):
        """
        Status: OK
        Expression: DS_r <- if DS_1 then DS_2 else DS_3;
        Description: Id_1 is on VD_INT in the condition and an Integer in the then and else
                     operands, so the Identifiers match on the basic type and Id_1 is an Integer;
                     Me_1 is on VD_INT in both branches and keeps it. It raised 1-1-9-10.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_17",
            number_inputs=3,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    def test_GH_1348_18(self):
        """
        Status: OK
        Expression: DS_r <- case when DS_1 then DS_2 else DS_3;
        Description: As in case 17, the Identifiers match on the basic type and Id_1 is an
                     Integer; Me_1 keeps VD_INT. It raised 1-1-9-10.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_18",
            number_inputs=3,
            references_names=["1"],
            vd_names=self.vd_1348,
        )

    vd_1348_c = [*vd_1348, "GH_1348-VD_STR2"]

    def test_GH_1348_19(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_STR ) is
                        when "A" then "A"; else "B" end viral propagation;
                    define viral propagation VP_2 ( valuedomain VD_STR2 ) is
                        when "A" then "X"; else "X" end viral propagation;
                    DS_r <- DS_1 + DS_2;
        Description: At_1 is on VD_STR in DS_1 and on VD_STR2 in DS_2. The mix is promoted to
                     String, on neither value domain, and At_1 has no rule on the variable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_19",
            number_inputs=2,
            exception_code="1-3-3-6",
            exception_message="Viral attribute At_1 has no viral propagation rule",
            vd_names=self.vd_1348_c,
        )

    def test_GH_1348_20(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_STR ) is
                        when "A" then "A"; else "B" end viral propagation;
                    define viral propagation VP_2 ( valuedomain VD_STR2 ) is
                        when "A" then "X"; else "X" end viral propagation;
                    DS_r <- DS_2 + DS_1;
        Description: Case 19 in the other operand order raises the same error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_20",
            number_inputs=2,
            exception_code="1-3-3-6",
            exception_message="Viral attribute At_1 has no viral propagation rule",
            vd_names=self.vd_1348_c,
        )

    def test_GH_1348_21(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_STR ) is
                        when "A" then "A"; else "B" end viral propagation;
                    DS_r <- DS_1 + DS_2;
        Description: At_1 is on VD_STR in DS_1 and a String in DS_2. The mix is promoted to
                     String, so the rule on VD_STR does not apply and At_1 has no rule.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_21",
            number_inputs=2,
            exception_code="1-3-3-6",
            exception_message="Viral attribute At_1 has no viral propagation rule",
            vd_names=self.vd_1348_c,
        )

    def test_GH_1348_22(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_STR ) is
                        when "A" then "A"; else "B" end viral propagation;
                    DS_r <- DS_2 + DS_1;
        Description: Case 21 in the other operand order raises the same error; it returned an
                     all-null At_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1348_22",
            number_inputs=2,
            exception_code="1-3-3-6",
            exception_message="Viral attribute At_1 has no viral propagation rule",
            vd_names=self.vd_1348_c,
        )

    def test_GH_1348_23(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_STR ) is
                        when "A" then "A"; else "B" end viral propagation;
                    define viral propagation VP_2 ( valuedomain VD_STR2 ) is
                        when "A" then "X"; else "X" end viral propagation;
                    define viral propagation VP_3 ( variable At_1 ) is
                        when "A" then "V"; else "W" end viral propagation;
                    DS_r <- DS_2 + DS_1;
        Description: Control of cases 19 and 20: the rule on the variable At_1 applies to the
                     mix of VD_STR and VD_STR2, promoted to String.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_23",
            number_inputs=2,
            references_names=["1"],
            vd_names=self.vd_1348_c,
        )

    def test_GH_1348_24(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( variable rule Id_2 ) is
                        r1 : A = B end hierarchical ruleset;
                    DS_r1 <- hierarchy ( DS_1 , HR_1 rule Id_2 all );
                    DS_r2 <- check_hierarchy ( DS_1 , HR_1 rule Id_2 all );
        Description: The ruleset is on a variable and the item A that its rule computes and
                     checks is a value of VD_STR, so the rule Identifier Id_2 keeps VD_STR in
                     both results, unlike cases 8 and 9, whose item C is not a value of it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1348
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1348_24",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1348,
        )

    vd_1349 = ["GH_1349-VD_TP", "GH_1349-VD_DATE", "GH_1349-VD_B", "GH_1349-VD_1"]

    def test_GH_1349_7(self):
        """
        Status: OK
        Expression: define operator in_tp ( x time_period , s set < time_period > )
                        returns boolean is x in s end operator;
                    DS_r <- DS_1 [ calc Me_4 := Me_1 in VD_TP , Me_5 := Me_2 not_in VD_TP ,
                                        Me_6 := Me_3 in VD_DATE ,
                                        Me_7 := cast ( "2024-Q1" , time_period ) in VD_TP ,
                                        Me_8 := in_tp ( cast ( "2024-Q2" , time_period ) ,
                                                        VD_TP ) ] [ keep ... ];
                    DS_r2 <- DS_1 [ filter Me_2 in VD_TP ] [ keep Me_2 ];
                    DS_r3 <- DS_1 [ keep Me_2 ] in VD_TP;
        Description: in and not_in compare a Time_Period or a Date as a value of its type with
                     the values of the value domain, whatever form the setlist (2024Q1,
                     2024-1-5) or the value (2024-Q2, 2024-1-6) is written in, for a Component,
                     in filter, at Data Set level, for a scalar and for a value domain passed
                     to a set parameter. pandas compared their text, so Me_4 was false, false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_7",
            number_inputs=1,
            references_names=["1", "2", "3"],
            vd_names=self.vd_1349,
        )

    def test_GH_1349_8(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
                    DS_r2 <- DS_1 [ pivot Id_2 , Me_1 ] [ keep 'true' ];
        Description: pivot on Id_2, on the Boolean value domain VD_B, names the Measures true
                     and false, as cast ( Id_2 , string ) writes the values; they were True and
                     False, so keep 'true' raised 1-1-1-10.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_8",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1349,
        )

    def test_GH_1349_9(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := cast ( Id_2 , string ) ];
        Description: Control of case 8: the String of a Boolean of VD_B is true or false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_9",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1349,
        )

    def test_GH_1349_10(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_1 ) is
                        when "A" then "Z"; else "Q" end viral propagation;
                    DS_r <- DS_1 + DS_1;
        Description: The constants of a rule on a value domain must be values of it: Z is not
                     a value of VD_1. At_1 was on VD_1 holding Z and Q.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_10",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message='The constant "Z" of viral propagation rule VP_1 is not one of the '
            "values of the Value Domain VD_1",
            vd_names=self.vd_1349,
        )

    def test_GH_1349_11(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_X ) is
                        when "A" then "A"; else "B" end viral propagation;
                    DS_r <- DS_1 + DS_1;
        Description: The value domain VD_X of the rule is not given, which the definition
                     reports. It raised a misleading 1-3-3-6 for At_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_11",
            number_inputs=1,
            exception_code="1-2-8",
            exception_message="Missing value domain 'VD_X'",
            vd_names=self.vd_1349,
        )

    def test_GH_1349_11b(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_X ) is
                        when "A" then "A"; else "B" end viral propagation;
                    DS_r <- DS_1;
        Description: As case 11 with no statement using the rule: the definition itself
                     raises. It was accepted.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_11b",
            number_inputs=1,
            exception_code="1-2-8",
            exception_message="Missing value domain 'VD_X'",
            vd_names=self.vd_1349,
        )

    def test_GH_1349_12(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( valuedomain VD_X ) is
                        R1 : VD_X > 0 end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , DPR_1 components Me_1 );
        Description: Control of case 11: a datapoint ruleset on a value domain that is not
                     given raises 1-2-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_12",
            number_inputs=1,
            exception_code="1-2-8",
            exception_message="Missing value domain 'VD_X'",
            vd_names=self.vd_1349,
        )

    vd_1349_int = ["GH_1349-errorcode_vd_int", "GH_1349-errorlevel_vd_int"]

    def test_GH_1349_13(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 1 errorcode "100" errorlevel "1" invalid );
        Description: The String "100" is not a value of the Integer errorcode_vd, which does
                     not convert it. It gave the errorcode 100 and the errorlevel 1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_13",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message='The errorcode "100" of check is not one of the values of the '
            "Value Domain errorcode_vd",
            vd_names=self.vd_1349_int,
        )

    def test_GH_1349_13b(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 1 errorcode 100 errorlevel "1" invalid );
        Description: As case 13 for the error level alone: the String "1" is not a value of
                     the Integer errorlevel_vd, while the errorcode 100 is one of errorcode_vd.
                     It gave the errorlevel 1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_13b",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message='The errorlevel "1" of check is not one of the values of the '
            "Value Domain errorlevel_vd",
            vd_names=self.vd_1349_int,
        )

    def test_GH_1349_14(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 1 errorlevel 1 invalid );
        Description: The Integer 1 is not a value of the String errorlevel_vd, which holds
                     "1". It gave the errorlevel "1".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_14",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message="The errorlevel 1 of check is not one of the values of the Value "
            "Domain errorlevel_vd",
            vd_names=["GH_1349-errorlevel_vd_str"],
        )

    def test_GH_1349_15(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 1 errorcode 100.0 invalid );
        Description: Control of cases 13 and 14: the Number 100.0 is not a value of the
                     Integer errorcode_vd.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_15",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message="The errorcode 100.0 of check is not one of the values of the "
            "Value Domain errorcode_vd",
            vd_names=self.vd_1349_int,
        )

    def test_GH_1349_21(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter between ( Id_2 , cast ( "2024Q2" , time_period ) ,
                                                    cast ( "2024Q3" , time_period ) ) ];
        Description: between orders Time_Period values as periods. pandas compared the text
                     2024-Q2 with 2024Q2 and kept no data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_21", number_inputs=1, references_names=["1"])

    def test_GH_1349_21b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := between ( Me_1 , SC_1 , SC_1 ) ];
        Description: between stands for >= and <=, which do not compare periods of different
                     period indicators (2024M01 and 2024Q2): both engines raise 2-1-19-19, as
                     >= and <= do. DuckDB returned false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_21b",
            number_inputs=1,
            exception_code="2-1-19-19",
            exception_message="Time Period comparison (>, <, >=, <=) with different period "
            "indicator is not supported",
            scalars={"SC_1": "2024Q2"},
        )

    def test_GH_1349_22(self):
        """
        Status: OK
        Expression: DS_r <- between ( DS_1 , SC_2 , SC_2 );
        Description: At Data Set level the period equal to SC_2 (2024Q1) is between SC_2 and
                     SC_2; pandas gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_22", number_inputs=1, references_names=["1"], scalars={"SC_2": "2024Q1"}
        )

    def test_GH_1349_23(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := between ( Me_1 , SC_2 , SC_2 ) ];
        Description: As case 22 on a Component; pandas gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_23", number_inputs=1, references_names=["1"], scalars={"SC_2": "2024Q1"}
        )

    def test_GH_1349_24(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 >= SC_2 and Me_1 <= SC_2 ];
        Description: Control of cases 21 to 23: the same check with >= and <=.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_24", number_inputs=1, references_names=["1"], scalars={"SC_2": "2024Q1"}
        )

    def test_GH_1349_38(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 in { cast ( "2024Q2" , time_period ) } ,
                                   Me_3 := Me_1 not_in { cast ( "2024Q2" , time_period ) } ];
                    DS_r2 <- DS_1 [ keep Me_1 ] in { cast ( "2024Q2" , time_period ) };
                    DS_r3 <- DS_2 in { cast ( "2024-01-05T00:00:00" , date ) ,
                                       cast ( "2024-01-06 10:00:00" , date ) };
        Description: The elements of a literal set are values of their type: a Time_Period or
                     a Date is in the set holding its value in any form, for a Component and
                     at Data Set level. Both engines compared the text of the elements, so
                     Me_2 was false for the data point holding 2024Q2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_38", number_inputs=2, references_names=["1", "2", "3"])

    def test_GH_1349_40(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 = cast ( "2024Q2" , time_period ) ];
        Description: Control of case 38: the same value compared with =.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_40", number_inputs=1, references_names=["1"])

    vd_1349_d = ["GH_1349-VD_TP", "GH_1349-VD_D"]

    def test_GH_1349_41(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_TP ) is
                        when "2024Q1" then "2024Q2"; else "2024Q1" end viral propagation;
                    define viral propagation VP_2 ( valuedomain VD_D ) is
                        when "2024-1-5" then "2024-1-6"; else "2024-1-5"
                    end viral propagation;
                    DS_r <- DS_1 + DS_1;
        Description: The constants of a rule on a Time_Period or Date value domain are values
                     of its type: the when matches the data holding that value in any form, and
                     the result is written as a value of the type (2024Q2, 2024-01-06). pandas
                     took the else for both data points and DuckDB gave null for At_1 and
                     wrote the constants of At_2 as written.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_41", number_inputs=1, references_names=["1"], vd_names=self.vd_1349_d
        )

    def test_GH_1349_42(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_TP ) is
                        when "2024-Q1" then "2024-Q2"; else "2024-Q1" end viral propagation;
                    define viral propagation VP_2 ( valuedomain VD_D ) is
                        when "2024-01-05" then "2024-01-06"; else "2024-01-05"
                    end viral propagation;
                    DS_r <- DS_1 + DS_1;
        Description: Control of case 41: the constants written in the form the engines store.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_42", number_inputs=1, references_names=["1"], vd_names=self.vd_1349_d
        )

    def test_GH_1349_43(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := between ( Me_1 , SC_2 , SC_2 ) ];
        Description: between stands for >= and <=, which do not compare periods of different
                     period indicators (2024M01, 2025A and 2024Q1): both engines raise
                     2-1-19-19, as >= and <= do. Both gave false for the other indicators.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_43",
            number_inputs=1,
            exception_code="2-1-19-19",
            exception_message="Time Period comparison (>, <, >=, <=) with different period "
            "indicator is not supported",
            scalars={"SC_2": "2024Q1"},
        )

    def test_GH_1349_44(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := between ( Me_1 , SC_2 , null ) ];
        Description: between compares the operand with the bound that is not null, as
                     Me_1 >= SC_2 and Me_1 <= null does, so periods of different indicators
                     raise 2-1-19-19 (case 45). Both engines gave null without comparing.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_44",
            number_inputs=1,
            exception_code="2-1-19-19",
            exception_message="Time Period comparison (>, <, >=, <=) with different period "
            "indicator is not supported",
            scalars={"SC_2": "2024Q1"},
        )

    def test_GH_1349_44b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := between ( Me_1 , SC_2 , null ) ,
                                   Me_3 := between ( Me_1 , null , SC_2 ) ];
        Description: As case 44 with periods of one indicator: as op >= from and op <= to,
                     a null bound gives false where the other comparison is false (2024Q1 is
                     not >= 2024Q2, 2024Q3 is not <= 2024Q2) and null elsewhere (case 56).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_44b", number_inputs=1, references_names=["1"], scalars={"SC_2": "2024Q2"}
        )

    def test_GH_1349_45(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 >= SC_2 and Me_1 <= null ];
        Description: Control of case 44: >= raises 2-1-19-19 for periods of different
                     indicators also when the other bound is null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_45",
            number_inputs=1,
            exception_code="2-1-19-19",
            exception_message="Time Period comparison (>, <, >=, <=) with different period "
            "indicator is not supported",
            scalars={"SC_2": "2024Q1"},
        )

    def test_GH_1349_46(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
                    DS_r2 <- DS_1 [ calc Me_2 := cast ( Id_2 , string ) ] [ keep Me_2 ];
        Description: pivot on Id_2, on the Date value domain VD_D whose setlist writes
                     2024-1-5, names the Measures as cast ( Id_2 , string ) writes the values:
                     2024-01-05 and 2024-01-06 (the reference structure names them, case 57).
                     The first one was named 2024-1-5.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_46",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=["GH_1349-VD_D"],
        )

    def test_GH_1349_47(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
                    DS_r2 <- DS_1 [ calc Me_2 := cast ( Id_2 , string ) ] [ keep Me_2 ];
        Description: pivot on Id_2, on the Time_Period value domain VD_TP2 whose setlist
                     writes 2024-Q1, names the Measures as cast ( Id_2 , string ) writes the
                     values: 2024Q1 and 2024Q2. The first one was named 2024-Q1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_47",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=["GH_1349-VD_TP2"],
        )

    def test_GH_1349_48(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
                    DS_r2 <- DS_1 [ calc Me_2 := cast ( Id_2 , string ) ] [ keep Me_2 ];
        Description: pivot on Id_2, on the Duration value domain VD_DUR, keeps the names A and
                     Q, the form the data and the setlist write the values in, while
                     cast ( Id_2 , string ) writes P1Y and P3M (not decided: pinned as is).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_48",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=["GH_1349-VD_DUR"],
        )

    def test_GH_1349_49(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_1 as b apply random ( a , b ) );
        Description: random in a join apply gives on DuckDB the numbers pandas gives, those
                     of random ( Me_1 , Me_1 ) in calc (case 50). DuckDB wrote a plain SQL
                     random ( a , b ) and raised a BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_49", number_inputs=1, references_names=["1"])

    def test_GH_1349_50(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := random ( Me_1 , Me_1 ) ];
        Description: Control of case 49: random in calc.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_50", number_inputs=1, references_names=["1"])

    def test_GH_1349_53(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( variable At_1 ) is
                        when "2024Q1" then "2024Q2"; else "2024Q1" end viral propagation;
                    define viral propagation VP_2 ( variable At_2 ) is
                        when "2024-1-5" then "2024-1-6"; else "2024-1-5"
                    end viral propagation;
                    DS_r <- DS_1 + DS_1;
                    DS_r2 <- DS_2 + DS_2;
        Description: As case 41 for rules on a variable: the constants are values of the type
                     of the Attribute the rule applies to (the Time_Period At_1, the Date
                     At_2), whatever form they are written in, and the result is written as a
                     value of that type. pandas took the else, DuckDB gave null for At_1 and
                     wrote the constants of At_2 as written.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_53", number_inputs=2, references_names=["1", "2"])

    def test_GH_1349_53b(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( variable At_1 ) is
                        when "2024Q1" then "2024Q2"; when "2024-Q1" then "2024Q1";
                        else "2024Q1" end viral propagation;
                    DS_r <- DS_1 + DS_1;
        Description: "2024Q1" and "2024-Q1" are one Time_Period, so the two when are the same
                     combination for the Time_Period At_1 and raise 1-3-3-4. Only the text was
                     compared, so the rule was accepted.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_53b",
            number_inputs=1,
            exception_code="1-3-3-4",
            exception_message="Duplicate enumeration combination",
        )

    def test_GH_1349_54(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "2024-1-5" , date ) ,
                                   Me_3 := cast ( Me_1 , date ) ,
                                   Me_4 := cast ( SC_1 , date ) ];
                    DS_r2 <- cast ( DS_1 , date );
        Description: A String cast to date reads the text as the loaders read a Date:
                     2024-1-5, 20240105, 2024-1-5 00:00:00 and 2024-01-05T00:00:00 are
                     2024-01-05, for a literal, a Component, a scalar and a Data Set (a time
                     other than midnight is kept: test_GH_1349_54b). The literal raised
                     1-1-5-6 and the other forms 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1349_54",
            number_inputs=1,
            references_names=["1", "2"],
            scalars={"SC_1": "2024-1-5"},
        )

    def test_GH_1349_55(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := Me_1 in { cast ( "2024Q1" , time_period ) ,
                                                         cast ( "2024-Q1" , time_period ) } ];
        Description: 2024Q1 and 2024-Q1 are one Time_Period, so the set holds a repeated
                     value and raises 1-2-27, as { 1 , 1 } does, on both engines and in the
                     semantic analysis. The set was accepted.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_55",
            number_inputs=1,
            exception_code="1-2-27",
            exception_message="A set must not contain duplicates",
        )

    def test_GH_1349_55b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := Me_2 in { cast ( "2024-01-05" , date ) ,
                                                         cast ( "2024-01-05T00:00:00" , date ) } ];
        Description: As case 55 for two forms of one Date.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1349_55b",
            number_inputs=1,
            exception_code="1-2-27",
            exception_message="A set must not contain duplicates",
        )

    def test_GH_1349_55c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := Me_2 in { cast ( "05/01/2024" , date ,
                                                         "DD/MM/YYYY" ) ,
                                                         cast ( "2024-01-07" , date ) } ];
        Description: A cast with a mask inside a set reads its mask as a cast outside a set
                     does, in the semantic analysis and on both engines. The mask of a cast
                     inside a set was kept as a parser node, so every path failed with an
                     internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_55c", number_inputs=1, references_names=["1"])
        self.BaseTest(
            code="GH_1349_55c", number_inputs=1, references_names=["1"], only_semantic=True
        )

    def test_GH_1349_56(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_5 := between ( Me_1 , 2 , null ) , ... ,
                                   Me_13 := between ( 3 , null , 2 ) , ... ] [ drop ... ];
                    DS_r2 <- between ( DS_1 [ keep Me_1 ] , 2 , null );
                    DS_r3 <- between ( DS_1 [ keep Me_4 ] , null ,
                                       cast ( "2024Q2" , time_period ) );
        Description: between is op >= from and op <= to, as the VTL 2.2 manual defines it,
                     for every type (Integer, String, Date, Time_Period), for a Component, a
                     scalar and a Data Set: a null bound gives false where the other
                     comparison is false, null elsewhere, and a null operand gives null. Both
                     engines gave null whenever a bound was null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_56", number_inputs=1, references_names=["1", "2", "3"])

    def test_GH_1349_56b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := between ( Me_1 , Me_2 , Me_3 ) ,
                                   Me_5 := Me_1 >= Me_2 and Me_1 <= Me_3 , ... ] [ drop ... ];
                    DS_r2 <- between ( DS_1 [ keep Me_1 ] , cast ( "D" , duration ) ,
                                       cast ( "M" , duration ) );
        Description: between of Duration operands is op >= from and op <= to: the values are
                     ordered as durations (D < W < M < Q < S < A), as >= and <= order them, for
                     a Component, a scalar and a Data Set, also with a null bound. Both engines
                     ordered them as text, so between(A, A, M) was true where the pair is false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1349
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1349_56b", number_inputs=1, references_names=["1", "2"])

    vd_1351 = ["GH_1351-VD_1", "GH_1351-VD_2", "GH_1351-VD_S"]

    def test_GH_1351_1(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset , s set < VD_1 > ) ... ds [ filter Me_1 in s ]
                        ...
                    DS_r <- f ( DS_2 , VD_1 );
        Description: A value domain types the values of a set parameter, which takes VD_1 itself. In
                     the body s holds the values of VD_1, and Me_1 stays an Integer. The value
                     domain raised 0-5-1-10, as a name that is no type.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_1",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_2(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_2 , VD_2 ); (s set < VD_1 >)
        Description: 4, a value of VD_2, is not a value of VD_1, which types the set parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_2",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op f: Cannot cast 4 of Integer to Value Domain VD_1",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_3(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_2 , VD_S ); (s set < VD_1 >)
        Description: VD_S, a value domain of Strings, is no argument of a set of VD_1, a value
                     domain of Integers, as for set < integer >.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_3",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected set<VD_1>, got set<string> on UDO f, parameter s",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_4(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_2 , VD_1 ); (s set < integer >)
        Description: Control: a set of a basic type takes a value domain of that type.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_4",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_5(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_2 , VD_S ); (s set < integer >)
        Description: Control: a value domain of Strings is no argument of a set of integers.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_5",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected set<integer>, got set<string> on UDO f, parameter s",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_6(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset { measure < VD_1 > Me_1 } ) ... ds ...
                    DS_r <- f ( DS_1 );
        Description: A value domain types a component of a dataset parameter. Me_1 of DS_1 is on
                     VD_1, and the argument is passed as it is. The value domain raised 0-5-1-10.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_6",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_7(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset { identifier < integer > Id_1 , identifier < VD_1
                        > Id_2 ,
                    measure < VD_1 > Me_1 , attribute < VD_1 > At_1 } ) ... ds ...
                    DS_r <- f ( DS_1 );
        Description: Value domains type an Identifier, a Measure and an Attribute of a dataset
                     parameter, and the components of DS_1 on VD_1 keep it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_7",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_8(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset { measure < VD_1 > _+ } ) ... ds ...
                    DS_r <- f ( DS_1 );
        Description: A value domain types a component of a dataset parameter written with the
                     multiplicity modifier _+, which takes Me_1 of DS_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_8",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_9(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_2 ); (ds dataset { measure < VD_1 > Me_1 })
        Description: Me_1 of DS_2 is an Integer, which the constraint takes, as x component < VD_1 >
                     does, but 5 is not a value of VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_9",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op f: Cannot cast 5 of Integer to Value Domain VD_1",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_10(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_1 ); (ds dataset { measure < integer > Me_1 })
        Description: Control: a constraint of a basic type takes Me_1 on VD_1, which keeps VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_10",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_11(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_3 ); (ds dataset { measure < integer > Me_1 })
        Description: Control: Me_1 of DS_3, a String, does not have the type of the constraint.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_11",
            number_inputs=1,
            exception_code="1-3-1-12",
            exception_message=(
                "component Me_1 of parameter ds is measure<string>, but its type "
                "dataset {measure<integer> Me_1} declares it as measure<integer>"
            ),
            vd_names=self.vd_1351,
        )

    def test_GH_1351_12(self):
        """
        Status: OK
        Expression: define operator f ( x VD_1 ) returns integer is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := f ( 2 ) ];
        Description: Control (#1236): a scalar parameter typed by VD_1 is on it, and a returns of
                     its basic type keeps it, so Me_2 is on VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_12",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_13(self):
        """
        Status: OK
        Expression: define operator f ( x component < VD_1 > ) returns component < integer > is x
                        ...
                    DS_r <- DS_1 [ calc Me_2 := f ( Me_1 ) ];
        Description: Control (#1236): a component parameter typed by VD_1 takes Me_1 on VD_1, and
                     Me_2 keeps VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_13",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_14(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns VD_1 is x end operator;
                    SC_r <- f ( 2 );
                    DS_r <- DS_2 [ calc Me_2 := SC_r ];
        Description: A value domain types the returns: the result 2 is on VD_1, which Me_2 keeps.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_14",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_15(self):
        """
        Status: OK
        Expression: SC_r <- f ( 5 ); (returns VD_1)
        Description: The result 5 is not a value of VD_1, which the returns names.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_15",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op f: Cannot cast 5 of Integer to Value Domain VD_1",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_16(self):
        """
        Status: OK
        Expression: SC_r <- f ( "A" ); (x string, returns VD_1)
        Description: The result is a String, which is not the basic type of VD_1, as for a returns
                     integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_16",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="User defined output declared as VD_1, found string",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_17(self):
        """
        Status: OK
        Expression: define operator f ( x component ) returns measure < VD_1 > is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := f ( Me_1 ) ];
        Description: A value domain types a component returns: Me_2 is on VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_17",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_18(self):
        """
        Status: OK
        Expression: DS_r <- DS_2 [ calc Me_2 := f ( Me_1 ) ]; (returns measure < VD_1 >)
        Description: Me_1 of DS_2 is an Integer, the basic type of VD_1, but 5 is not a value of
                     VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_18",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op f: Cannot cast 5 of Integer to Value Domain VD_1",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_19(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset ) returns dataset { measure < VD_1 > Me_1 } is ds
                        ...
                    DS_r <- f ( DS_1 );
        Description: A value domain types a component of a dataset returns: Me_1 of the result is on
                     VD_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_19",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_20(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_3 ); (returns dataset { measure < VD_1 > Me_1 })
        Description: Me_1 of DS_3 is a String, not the basic type of VD_1, as for a dataset returns
                     of integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_20",
            number_inputs=1,
            exception_code="1-3-1-12",
            exception_message=(
                "component Me_1 of the output is measure<string>, but its type dataset "
                "{measure<VD_1> Me_1} declares it as measure<VD_1>"
            ),
            vd_names=self.vd_1351,
        )

    def test_GH_1351_21(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns integer is x end operator;
                    SC_r <- f ( 2 );
                    DS_r <- DS_2 [ calc Me_2 := SC_r ];
        Description: Control: a returns of a basic type gives that type.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_21",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_22(self):
        """
        Status: OK
        Expression: SC_r <- f ( "A" ); (x string, returns integer)
        Description: Control: the result is a String, not the type the returns declares.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_22",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="User defined output declared as integer, found string",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_23(self):
        """
        Status: OK
        Expression: define operator f ( x component ) returns measure < integer > is x end operator;
                    DS_r <- DS_2 [ calc Me_2 := f ( Me_1 ) ];
        Description: Control: a component returns of a basic type gives that type.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_23",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_24(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_2 := f ( Me_1 ) ]; (returns measure < integer >)
        Description: Control: Me_1 of DS_3 is a String, not the type the returns declares.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_24",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message=(
                "User defined output declared as measure<integer>, found measure<string>"
            ),
            vd_names=self.vd_1351,
        )

    def test_GH_1351_25(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := f ( Me_1 ) ]; (returns measure < integer >)
        Description: Control: a component returns of a basic type keeps VD_1, the value domain of
                     Me_1 of DS_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_25",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_26(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset ) returns dataset { measure < integer > Me_1 } is
                        ds
                    ... DS_r <- f ( DS_1 );
        Description: Control: a dataset returns of a basic type keeps VD_1, the value domain of Me_1
                     of DS_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_26",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_27(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_3 ); (returns dataset { measure < integer > Me_1 })
        Description: Control: Me_1 of DS_3 is a String, not the type the returns declares.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_27",
            number_inputs=1,
            exception_code="1-3-1-12",
            exception_message=(
                "component Me_1 of the output is measure<string>, but its type dataset "
                "{measure<integer> Me_1} declares it as measure<integer>"
            ),
            vd_names=self.vd_1351,
        )

    def test_GH_1351_28(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset ) returns dataset { identifier < VD_1 > Id_1 ,
                        measure < integer > Me_1 } is ds end operator;
                    DS_r <- f ( DS_2 ) [ pivot Id_1 , Me_1 ];
        Description: Beyond the cases of the issue: pivot reads the value domain that a dataset
                     returns puts an Identifier on. The DuckDB engine kept the basic type in its
                     own structure of the call, so its pivot failed with an internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_28",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_28b(self):
        """
        Status: OK
        Expression: define operator g ( x integer ) returns VD_1 is x end operator;
                    DS_r2 <- DS_1 [ calc identifier Id_3 := g ( 1 ) ] [ keep Me_1 ]
                        [ sub Id_1 = 1 , Id_2 = 1 ] [ pivot Id_3 , Me_1 ];
        Description: Beyond the cases of the issue: pivot reads the value domain that the returns
                     of an operator puts an Identifier on in a calc. The DuckDB engine kept the
                     basic type in its own structure of the calc, so its pivot failed with an
                     internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_28b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_29(self):
        """
        Status: OK
        Expression: define operator flag ( ds dataset ) returns dataset { identifier < integer >
                        Id_1 , measure < integer > Me_1 , measure < VD_S > Me_2 } is
                        ds [ calc Me_2 := if Me_1 > 1 then "A" else "B" ] end operator;
                    DS_r <- flag ( DS_2 ) [ filter Id_1 > 0 ];
        Description: The String Measure that the body computes is on VD_S, also when the call is
                     the operand of a clause. The DuckDB engine checked the returns against the
                     structure it guessed for the call, where Me_2 was a Number, and raised
                     1-3-1-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_29",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_30(self):
        """
        Status: OK
        Expression: define operator flag ( ds dataset ) returns dataset { identifier < integer >
                        Id_1 , measure < integer > Me_1 , measure < VD_B > Me_2 } is
                        ds [ calc Me_2 := Me_1 > 1 ] end operator;
                    DS_r <- union ( flag ( DS_2 [ filter Id_1 = 1 ] ) , flag ( DS_2 ) );
        Description: The Boolean Measure that the body computes is on VD_B, also when the calls
                     are the operands of union. The DuckDB engine checked the returns against
                     the structure it guessed for each call, where Me_2 was a Number, and raised
                     1-3-1-12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_30",
            number_inputs=1,
            references_names=["1"],
            vd_names=[*self.vd_1351, "GH_1351-VD_B"],
        )

    def test_GH_1351_31(self):
        """
        Status: OK
        Expression: define operator f ( x number ) returns VD_1 is x end operator;
                    SC_r <- f ( 2.5 );
        Description: A Number is not implicitly cast to Integer, so the Number result does not
                     have the basic type of VD_1, as with returns integer (issue #1364). The
                     DuckDB engine and the semantic analysis cast the result to Integer first
                     and raised 0-1-1-27.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_31",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined output declared as VD_1, found number.",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_32a(self):
        """
        Status: OK
        Expression: define operator f ( x VD_S ) returns string is x end operator;
                    SC_r <- f ( true );
        Description: A Boolean argument is promoted to String as the text true, which is not a
                     value of VD_S. The DuckDB engine failed with an internal error comparing
                     the Boolean with the Strings of VD_S.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_32a",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op f: Cannot cast true of String to Value Domain VD_S",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_32b(self):
        """
        Status: OK
        Expression: define operator f ( x component < VD_S > ) returns component is x end operator;
                    DS_r <- DS_B [ calc Me_2 := f ( Me_1 ) ];
        Description: The values of a Boolean Component are promoted to String as the text true
                     or false, which are not values of VD_S. The DuckDB engine failed with an
                     internal error comparing the Booleans with the Strings of VD_S.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_32b",
            number_inputs=1,
            exception_code="2-1-5-4",
            exception_message="At op f: Cannot cast true of String to Value Domain VD_S",
            vd_names=self.vd_1351,
        )

    def test_GH_1351_32c(self):
        """
        Status: OK
        Expression: define operator f ( x VD_SB ) returns string is x end operator;
                    define operator g ( x component < VD_SB > ) returns component is x
                        end operator;
                    SC_r <- f ( true );
                    DS_r <- DS_B [ calc Me_2 := g ( Me_1 ) , Me_3 := SC_r , Me_4 := f ( false ) ];
        Description: VD_SB holds the Strings true and false, so a Boolean argument and the
                     values of a Boolean Component, promoted to String as that text, are values
                     of it. The pandas engine compared the Booleans themselves and raised
                     2-1-5-4 for the Component, and the DuckDB engine failed with an internal
                     error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_32c",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=[*self.vd_1351, "GH_1351-VD_SB"],
        )

    def test_GH_1351_32d(self):
        """
        Status: OK
        Expression: define operator f ( x VD_N ) returns string is cast ( x , string )
                        end operator;
                    define operator g ( x component < VD_N > ) returns component is x
                        end operator;
                    DS_r <- DS_I [ calc Me_2 := f ( 2 ) , Me_3 := g ( Me_1 ) ];
        Description: An Integer argument is promoted to the Number of VD_N, which its body
                     casts to the String 2.0, and the values of an Integer Component are values
                     of VD_N. The DuckDB engine kept the argument an Integer and gave 2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_32d",
            number_inputs=1,
            references_names=["1"],
            vd_names=[*self.vd_1351, "GH_1351-VD_N"],
        )

    def test_GH_1351_32e(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset ) returns dataset { measure < VD_SB > Me_1 } is
                        ds end operator;
                    DS_r <- f ( DS_B );
        Description: The Boolean Measure of the result is promoted to String on VD_SB, as the
                     text true or false. The pandas engine compared the Booleans themselves and
                     raised 2-1-5-4, and the DuckDB engine failed with an internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_32e",
            number_inputs=1,
            references_names=["1"],
            vd_names=[*self.vd_1351, "GH_1351-VD_SB"],
        )

    def test_GH_1351_32f(self):
        """
        Status: OK
        Expression: define operator f ( x component ) returns component < VD_SB > is x
                        end operator;
                    DS_r <- DS_B [ calc Me_2 := f ( Me_1 ) ];
        Description: The Boolean Component result is promoted to String on VD_SB, as the text
                     true or false. The pandas engine compared the Booleans themselves and raised
                     2-1-5-4, and the DuckDB engine failed with an internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_32f",
            number_inputs=1,
            references_names=["1"],
            vd_names=[*self.vd_1351, "GH_1351-VD_SB"],
        )

    def test_GH_1351_33(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns VD_N is x end operator;
                    define operator g ( x integer ) returns number is x / 2 end operator;
                    SC_r <- f ( 2 );
                    DS_r <- DS_2 [ calc Me_2 := cast ( SC_r , string ) ,
                        Me_3 := cast ( f ( 2 ) , string ) , Me_4 := cast ( g ( 2 ) , string ) ];
        Description: A scalar result is a value of its type: the Integer 2 returned on VD_N is
                     the Number 2.0, and 2 / 2 is the Number 1.0, which cast to String give 2.0
                     and 1.0. The DuckDB engine kept the Integer 2 and the pandas engine gave
                     the whole Number 1 as an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_33",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=[*self.vd_1351, "GH_1351-VD_N"],
        )

    def test_GH_1351_34a(self):
        """
        Status: OK
        Expression: define operator f ( x VD_1 ) returns integer is x end operator;
                    DS_r <- DS_2 [ calc identifier Id_3 := f ( 1 ) ] [ pivot Id_3 , Me_1 ];
        Description: The result of f is its parameter, on VD_1, which returns integer keeps, so
                     pivot reads the values of VD_1. The DuckDB engine did not see the value
                     domain through the call and failed with an internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_34a",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_34b(self):
        """
        Status: OK
        Expression: define operator f ( x component ) returns component < integer > is x
                        end operator;
                    DS_r <- DS_1 [ calc identifier Id_3 := f ( Id_2 ) ] [ keep Me_1 ]
                        [ sub Id_2 = 1 ] [ pivot Id_3 , Me_1 ];
        Description: The result of f is the Component Id_2, on VD_1, which returns
                     component < integer > keeps, so pivot reads the values of VD_1. The DuckDB
                     engine did not see the value domain through the call and failed with an
                     internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1351_34b",
            number_inputs=1,
            references_names=["1"],
            vd_names=self.vd_1351,
        )

    def test_GH_1351_35(self):
        """
        Status: OK
        Expression: define operator f ( s component < string > , k number )
                        returns component < string > is substr ( s , 1 , k ) end operator;
                    define operator g ( s component < string > , k number )
                        returns component < integer > is instr ( s , "c" , k ) end operator;
                    DS_r <- DS_1 [ calc Me_2 := f ( Me_1 , 3 ) , Me_3 := g ( Me_1 , 2 ) ];
        Description: A number parameter is the length of substr and the start of instr. The
                     DuckDB engine read the Number parameter as a DOUBLE, which substr and
                     instr do not take, and failed with an internal error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1351_35", number_inputs=1, references_names=["1"])

    def test_GH_1351_36(self):
        """
        Status: OK
        Expression: define operator f ( x number ) returns number is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := substr ( Me_1 , f ( 2 ) , f ( 3 ) ) ,
                        Me_3 := substr ( Me_1 , f ( 1 ) , f ( 2.5 ) ) ];
        Description: The Number results of f are the start and the length of substr, and a
                     length with decimals is truncated, 2.5 to 2, as for any Number. The
                     DuckDB engine failed with an internal error on the DOUBLE results.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1351_36", number_inputs=1, references_names=["1"])

    def test_GH_1351_37(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns integer is x end operator;
                    SC_r <- f ( 7 / 2 );
        Description: The Number 3.5 is not an Integer, which the parameter x declares: the
                     call raises the RunTimeError 2-3-10 where it is computed (issue #1364), as
                     the semantic analysis does not know the value and gives the structure. The
                     DuckDB engine rounded 3.5 to the Integer 4.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        with pytest.raises(RunTimeError) as context:
            self.BaseTest(code="GH_1351_37", number_inputs=1, references_names=["1"])
        self._check_error(
            context.value,
            "2-3-10",
            "At op f: The value 3.5 given to the parameter x is not an Integer.",
        )
        self.BaseTest(
            code="GH_1351_37", number_inputs=1, references_names=["1"], only_semantic=True
        )

    def test_GH_1351_38(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns integer is x end operator;
                    define operator g ( x integer ) returns string is cast ( x , string )
                        end operator;
                    SC_r <- f ( 4 / 2 );
                    DS_r <- DS_2 [ calc Me_2 := cast ( SC_r , string ) ,
                        Me_3 := cast ( f ( 2.0 ) , string ) , Me_4 := g ( 4 / 2 ) ,
                        Me_5 := f ( 4 / 2 ) ];
        Description: A whole Number given to an integer parameter is the Integer 2, also
                     where the body reads it, so cast to String it gives 2. The DuckDB engine
                     kept the Number 2.0 inside the body.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1351_38", number_inputs=1, references_names=["1", "2"])

    def test_GH_1351_39(self):
        """
        Status: OK
        Expression: define operator f ( x component < integer > )
                        returns component < integer > is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := f ( Me_1 ) ];
        Description: A Number Component is accepted by a component < integer > parameter, but
                     a Number is not implicitly cast to Integer, so the Number result does not
                     have the type the returns declares (issue #1364).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_39",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined output declared as component<integer>, "
            "found measure<number>.",
        )

    def test_GH_1351_40(self):
        """
        Status: OK
        Expression: define operator f ( x number ) returns number is x * -1 end operator;
                    SC_r <- f ( 0 );
                    DS_r <- DS_2 [ calc Me_2 := cast ( SC_r , string ) ,
                        Me_3 := cast ( f ( 0 ) , string ) ];
        Description: The Number result 0 * -1 is 0.0, which cast to String gives 0.0. The
                     DuckDB engine gave the negative zero -0.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1351_40", number_inputs=1, references_names=["1", "2"])

    def test_GH_1351_41(self):
        """
        Status: OK
        Expression: define operator f ( x integer default 2.5 ) returns number is x * 2
                        end operator;
                    SC_r <- f ( );
        Description: Extra case: the default 2.5 of an integer parameter is not an Integer, a
                     Number with decimals, and raises 1-3-1-1 where the operator is defined
                     (issue #1356), on both engines and in the semantic analysis. The pandas
                     engine and the semantic analysis took the default as it was written, so
                     pandas gave 5.0, while the DuckDB engine raised 1-2-18.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1351_41",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined default of parameter x declared as "
            "Integer, found Number.",
        )

    def test_GH_1351_42(self):
        """
        Status: OK
        Expression: define operator f ( x integer default 2.0 ) returns string is
                        cast ( x , string ) end operator;
                    define operator g ( x integer default 2.0 ) returns integer is x
                        end operator;
                    DS_r <- DS_2 [ calc Me_2 := f ( ) , Me_3 := g ( ) ];
        Description: Extra case: the whole Number default 2.0 of an integer parameter is the
                     Integer 2, as an argument 2.0 passed to it is, so cast to String it gives
                     2 and g returns the Integer 2. The pandas engine kept the Number 2.0 and
                     gave 2.0 as the String, and both engines typed Me_3 as a Number.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1351_42", number_inputs=1, references_names=["1"])

    def test_GH_1351_43(self):
        """
        Status: OK
        Expression: define operator f ( x integer , y integer ) returns integer is y
                        end operator;
                    SC_r <- f ( 7 / 2 , 1 );
        Description: Extra case: the Number 3.5 given to the integer parameter x raises the
                     RunTimeError 2-3-10 where the call is computed (issue #1364), on both
                     engines, also when the body never reads x (test_GH_1351_37 reads it). The
                     semantic analysis does not know the value and gives the structure.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1351
        Goal: Check Exception.
        """
        with pytest.raises(RunTimeError) as context:
            self.BaseTest(code="GH_1351_43", number_inputs=1, references_names=["1"])
        self._check_error(
            context.value,
            "2-3-10",
            "At op f: The value 3.5 given to the parameter x is not an Integer.",
        )
        self.BaseTest(
            code="GH_1351_43", number_inputs=1, references_names=["1"], only_semantic=True
        )


class TimeValueBugs(BugHelper):
    """Date, Time_Period, Time and Duration values handled as values of their type (issue
    #1352)."""

    def test_GH_1352_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := Me_1 = Me_2 ];
        Description: A Date and a Time_Period have no common type: the comparison raises
                     1-1-1-1, as for a Date and an Integer, and the script must cast one side.
                     Both were promoted to Time, so the pandas engine compared them and the
                     DuckDB engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_1",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_1b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := nvl ( Me_2 , Me_2 ) = Me_3 ] [ keep Me_4 ];
        Description: Extra of case 1: the nvl of the Time_Period Me_2 is a Time_Period, which =
                     casts to Time to compare it with the Time Me_3, so 2020Q1 equals
                     2020-01-01/2020-03-31. The DuckDB engine did not know the type of the nvl and
                     compared the texts, giving false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_1b", number_inputs=1, references_names=["1"])

    def test_GH_1352_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := Me_1 < Me_2 ];
        Description: A Date and a Time_Period have no common type: the comparison raises
                     1-1-1-1. The pandas engine raised 2-1-19-19 and the DuckDB engine a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_2",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := between ( Me_1 , Me_2 , Me_2 ) ];
        Description: between compares the operand with each bound, so a Date with Time_Period
                     bounds raises 1-1-1-1, as a comparison does. The pandas engine raised
                     2-1-19-19 and the DuckDB engine a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_3",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := Me_3 < Me_3 ];
        Description: Control: a Time value allows only = and <>, so < raises 1-1-19-18.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_4",
            number_inputs=1,
            exception_code="1-1-19-18",
            exception_message="At op <: Time operators comparison are only support = and <>",
        )

    def test_GH_1352_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := nvl ( Me_1 , Me_2 ) ];
        Description: The operands of nvl must be of the same type, so a Date and a Time_Period
                     raise 1-1-1-1. The pandas engine copied the Date and Time_Period texts
                     into a Time Component and the DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_5",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_6(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := if Id_1 = 1 then Me_1 else Me_2 ];
        Description: The branches of if must be of the same type, so a Date and a Time_Period
                     raise 1-1-1-1. The pandas engine copied the texts into a Time Component
                     and the DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_6",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_6b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := if Id_1 = 1 then Me_1 else Me_3 ];
        Description: Extra of case 6: the Date and the Time branches have the common type
                     Time, so the Date 2020-01-15 is the Time 2020-01-15/2020-01-15. The
                     DuckDB engine raised a raw BinderException and the pandas engine copied
                     the Date text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_6b", number_inputs=1, references_names=["1"])

    def test_GH_1352_6c(self):
        """
        Status: OK
        Expression: DS_r <- if DS_1 = DS_1 then DS_1 else DS_2;
        Description: Extra of case 6: the Time Measure of DS_1 and the Date Measure of DS_2
                     have the common type Time. The DuckDB engine raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_6c", number_inputs=2, references_names=["1"])

    def test_GH_1352_6d(self):
        """
        Status: OK
        Expression: DS_r <- case when DS_1 = DS_1 then DS_2 else DS_1;
        Description: Extra of case 6: the Date Measure of DS_2 that case takes is cast to
                     Time, so 2020-01-15 is 2020-01-15/2020-01-15. The DuckDB engine raised a
                     raw BinderException and the pandas engine copied the Date text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_6d", number_inputs=2, references_names=["1"])

    def test_GH_1352_6e(self):
        """
        Status: OK
        Expression: SC_r <- if true then cast ( "2020-01-15" , date )
                        else cast ( "2020-01-01/2020-03-31" , time );
                    DS_r <- DS_1 [ calc Me_2 := SC_r ];
        Description: Extra of case 6: the scalar if of a Date and a Time is the Time
                     2020-01-15/2020-01-15. The DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_6e", number_inputs=1, references_names=["1", "2"])

    def test_GH_1352_6f(self):
        """
        Status: OK
        Expression: DS_r <- ( if DS_2 = DS_2 then DS_1 else DS_2 ) = DS_2;
        Description: Extra of case 6: the if of the Date DS_1 and the Time DS_2 is a Time, which
                     = compares with DS_2 as Time values. The DuckDB engine took the if as a Date
                     and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_6f", number_inputs=2, references_names=["1"])

    def test_GH_1352_6g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := ( if Id_1 = 2 then Me_2
                        else cast ( "2020Q1" , time_period ) ) = Me_3 ] [ keep Me_4 ];
        Description: Extra of case 6: the if of two Time_Periods is a Time_Period, which = casts to
                     Time to compare it with the Time Me_3, so 2020Q1 equals
                     2020-01-01/2020-03-31. The DuckDB engine did not know the type of the if and
                     compared the texts, giving false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_6g", number_inputs=1, references_names=["1"])

    def test_GH_1352_7(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 = DS_2;
        Description: The Date Measure of DS_1 and the Time_Period Measure of DS_2 have no
                     common type, so = raises 1-1-1-1. The engines returned values that
                     differed (DuckDB true for null = null).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_7",
            number_inputs=2,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_8(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 > DS_2;
        Description: The Date Measure of DS_1 and the Time_Period Measure of DS_2 have no
                     common type, so > raises 1-1-1-1. The pandas engine raised the
                     RunTimeError 2-1-19-17 and the DuckDB engine returned values.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_8",
            number_inputs=2,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_9(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: The Date Measure of DS_1 and the Time_Period Measure of DS_2 have no
                     common type, so nvl raises 1-1-1-1. The pandas engine copied the texts
                     into a Time Measure and the DuckDB engine wrote Python struct texts.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_9",
            number_inputs=2,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Date to Time_Period.",
        )

    def test_GH_1352_10(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 = DS_2;
        Description: A Date is implicitly cast to a Time with its start and end equal to it,
                     so the Date 2020-01-15 equals the Time 2020-01-15/2020-01-15. The pandas
                     engine raised a raw TypeError on the Date text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10", number_inputs=2, references_names=["1"])

    def test_GH_1352_10b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 = cast ( "2020-01-15" , date );
        Description: Extra case: the Date Scalar is cast to the Time 2020-01-15/2020-01-15,
                     which the second Time of DS_1 equals. The pandas engine compared the
                     Date text with the Time and gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10b", number_inputs=1, references_names=["1"])

    def test_GH_1352_10c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 in { cast ( "2020-01-15" , date ) };
        Description: Extra of case 10: the Date of the set is cast to the Time
                     2020-01-15/2020-01-15, which the second Time of DS_1 is. The DuckDB engine
                     raised 2-1-19-8 and the pandas engine compared the texts and gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10c", number_inputs=1, references_names=["1"])

    def test_GH_1352_10d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 in { cast ( "2020-01-15/2020-01-15" , time ) };
        Description: Extra of case 10: the Date 2020-01-15 of DS_1 is cast to Time, so it is in
                     the set of the Time 2020-01-15/2020-01-15. The DuckDB engine raised 2-1-19-8
                     and the pandas engine gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10d", number_inputs=1, references_names=["1"])

    def test_GH_1352_10e(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 in { cast ( "2020-01-01/2020-03-31" , time ) };
        Description: Extra of case 10: the Time_Period 2020Q1 of DS_1 is cast to the Time
                     2020-01-01/2020-03-31, so it is in the set. Both engines gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10e", number_inputs=1, references_names=["1"])

    def test_GH_1352_10f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := nvl ( Me_3 , Me_1 )
                        in { cast ( "2020-01-15" , date ) } ] [ keep Me_4 ];
        Description: Extra of case 10: the nvl of the Time Me_3 and the Date Me_1 is a Time, so the
                     Date of the set is cast to the Time 2020-01-15/2020-01-15. The DuckDB engine
                     did not know the type of the nvl and raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10f", number_inputs=1, references_names=["1"])

    def test_GH_1352_10g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := nvl ( Me_2 , Me_2 )
                        in { cast ( "2020-01-01/2020-03-31" , time ) } ] [ keep Me_4 ];
        Description: Extra of case 10: the nvl of the Time_Period Me_2 is a Time_Period, cast to
                     Time to look for it in a set of Times, so 2020Q1 is in the set. The DuckDB
                     engine did not know the type of the nvl and compared the texts, giving false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10g", number_inputs=1, references_names=["1"])

    def test_GH_1352_10h(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_5 := nvl ( Me_3 , Me_1 ) ]
                        [ calc Me_4 := Me_5 in { cast ( "2020-01-15" , date ) } ] [ keep Me_4 ];
        Description: Extra of case 10: Me_5, computed by the first calc as the nvl of the Time Me_3
                     and the Date Me_1, is a Time, so the Date of the set is cast to Time. The
                     DuckDB engine took Me_5 as a Number and raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10h", number_inputs=1, references_names=["1"])

    def test_GH_1352_10i(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 in { true } ] [ keep Me_2 ];
        Description: Extra of case 10: in with no time type among the operand and the set is
                     unchanged by the cast to Time, so the String Me_1 is not in the Boolean set,
                     as before. The DuckDB engine dropped the conversion of the set to text and
                     raised a raw ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_10i", number_inputs=1, references_names=["1"])

    def test_GH_1352_11(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 + DS_2;
        Description: The Date Identifier Id_1 of DS_1 and the Time_Period Identifier Id_1 of
                     DS_2 have no common type, so + raises 1-1-13-18, as for a Date and an
                     Integer Identifier. The pandas engine matched no data points and the
                     DuckDB engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_11",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Date and "
            "Time_Period.",
        )

    def test_GH_1352_11b(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_1 , DS_2 );
        Description: Extra case: exists_in matches the Data Points on the common Identifiers,
                     so the Date Id_1 and the Time_Period Id_1 raise 1-1-13-18, as in +. The
                     pandas engine gave false for both and the DuckDB engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_11b",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Date and "
            "Time_Period.",
        )

    def test_GH_1352_11c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 + DS_2;
        Description: Extra of case 11: the Date Identifier of DS_1 is cast to the Time of the
                     Identifier of DS_2, so 2020-01-01 matches 2020-01-01/2020-01-01 and the
                     result Identifier is a Time. The pandas engine matched nothing and the DuckDB
                     engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_11c", number_inputs=2, references_names=["1"])

    def test_GH_1352_11d(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_1 , DS_2 );
        Description: Extra of case 11: the Date Identifiers of DS_1 are matched as the Time values
                     of the Identifiers of DS_2, so 2020-01-01 exists. The pandas engine gave
                     false for both and the DuckDB engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_11d", number_inputs=2, references_names=["1"])

    def test_GH_1352_11e(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Extra of case 11: nvl matches the Data Points on the Identifiers as Time values,
                     so the Date 2020-01-01 matches the Time 2020-01-01/2020-01-01. The pandas
                     engine matched nothing and the DuckDB engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_11e", number_inputs=2, references_names=["1"])

    def test_GH_1352_11f(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( exists_in ( DS_1 , DS_2 ) , DS_1 );
        Description: Extra of case 11: exists_in of a Data Set that has a bool_var Measure, the
                     result of the inner exists_in, gives a new bool_var. The pandas engine
                     merged both bool_var columns and raised a raw KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_11f", number_inputs=2, references_names=["1"])

    def test_GH_1352_11g(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Extra of case 11: nvl matches the Data Points on the common Identifiers, so
                     the Date Id_1 and the Time_Period Id_1 raise 1-1-13-18, as in +. The pandas
                     engine matched no data points, the DuckDB engine raised 2-1-19-8 and the
                     semantic analysis accepted the script.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_11g",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Date and "
            "Time_Period.",
        )

    def test_GH_1352_11h(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Extra of case 11: the Integer Id_1 of DS_1 and the String Id_1 of DS_2 have no
                     common type, so nvl raises 1-1-13-18, as + and exists_in do. The engines
                     matched the Identifiers by their text.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_11h",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Integer and String.",
        )

    def test_GH_1352_12(self):
        """
        Status: OK
        Expression: DS_r <- full_join ( DS_1 as a , DS_2 as b keep a#Me_1 );
        Description: The Date Identifier Id_1 of DS_1 and the Time_Period Identifier Id_1 of
                     DS_2 have no common type, so the join raises 1-1-13-18. The pandas engine
                     wrote 2020D1 twice as an Identifier and the DuckDB engine raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_12",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Date and "
            "Time_Period.",
        )

    def test_GH_1352_12b(self):
        """
        Status: OK
        Expression: DS_r <- full_join ( DS_1 as a , DS_2 as b keep a#Me_1 );
        Description: Extra of case 12: the Date Identifier Id_1 of DS_1 is cast to the Time
                     of the Identifier of DS_2, so 2020-01-01 matches 2020-01-01/2020-01-01.
                     The DuckDB engine raised a raw BinderException and the pandas engine
                     matched nothing, writing the Date and the Time texts as four Identifiers.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_12b", number_inputs=2, references_names=["1"])

    def test_GH_1352_12c(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b keep a#Me_1 );
        Description: Extra of case 12: the join key takes the common type Time of the Time
                     Identifier of DS_1 and the Date one of DS_2, whatever their order. The result
                     typed it as the Date of the last operand: the pandas engine matched nothing
                     and the DuckDB engine wrote a Time value in a Date Identifier.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_12c", number_inputs=2, references_names=["1"])

    def test_GH_1352_12d(self):
        """
        Status: OK
        Expression: DS_r <- left_join ( DS_1 as a , DS_2 as b keep a#Me_1 );
        Description: Extra of case 12: the Time_Period keys 2020D1 and 2020Q1 of DS_2 are cast to
                     Time and match the Time keys of DS_1. The result typed the key as a
                     Time_Period: pandas wrote 2020M1 twice and DuckDB null twice.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_12d", number_inputs=2, references_names=["1"])

    def test_GH_1352_12e(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b , DS_3 as c keep a#Me_1 );
        Description: Extra of case 12: with a Date operand last, the key is still a Time and the
                     Date keys are cast to it. The pandas engine matched nothing and the DuckDB
                     engine wrote a Time value in a Date Identifier.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_12e", number_inputs=3, references_names=["1"])

    def test_GH_1352_41(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_1 , dateadd ( DS_1 , 1 , "M" ) );
        Description: dateadd turns the Time_Period Identifier Id_2 into a Date one, and a
                     Time_Period and a Date have no common type, so union raises 1-1-1-1. Both
                     engines returned a Time Identifier holding the raw texts.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_41",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Time_Period to Date.",
        )

    def test_GH_1352_42(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := nvl ( Me_3 , Me_2 ) ];
        Description: A Time_Period is implicitly cast to a Time with its start and end, so the
                     Time_Period 2021M12 that fills the null Time is 2021-12-01/2021-12-31.
                     Both engines copied the Time_Period text into the Time Component.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_42", number_inputs=1, references_names=["1"])

    def test_GH_1352_42b(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Extra case: the Time_Period 2020Q2 of DS_2 that fills the null Time of
                     DS_1 is the Time 2020-04-01/2020-06-30. Both engines copied the
                     Time_Period text 2020-Q2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_42b", number_inputs=2, references_names=["1"])

    def test_GH_1352_43(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_4 := nvl ( Me_1 , Me_3 ) ];
        Description: A Date is implicitly cast to a Time with its start and end equal to it,
                     so the Date 2020-01-15 is the Time 2020-01-15/2020-01-15. The pandas
                     engine copied the Date text into the Time Component and the DuckDB engine
                     raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43", number_inputs=1, references_names=["1"])

    def test_GH_1352_43b(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Extra case: the Date Measure of DS_1 is cast to Time, so 2020-01-15 is
                     2020-01-15/2020-01-15, and its nulls take the Time of DS_2. The pandas
                     engine kept the Date text and the DuckDB engine raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43b", number_inputs=2, references_names=["1"])

    def test_GH_1352_43c(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_1 , DS_2 );
        Description: Extra of case 43: the Date Measure of DS_1 is cast to the Time of DS_2, so
                     2020-01-15 is 2020-01-15/2020-01-15. The pandas engine kept the Date text and
                     DuckDB wrote 2020-01-15 00:00:00.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43c", number_inputs=2, references_names=["1"])

    def test_GH_1352_43d(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_1 , DS_2 );
        Description: Extra of case 43: the Date Identifiers of DS_1 are cast to Time, so 2020-01-01
                     is the Data Point 2020-01-01/2020-01-01 that DS_2 repeats. Both engines kept
                     the Date texts and four Data Points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43d", number_inputs=2, references_names=["1"])

    def test_GH_1352_43e(self):
        """
        Status: OK
        Expression: DS_r <- setdiff ( DS_1 , DS_2 );
        Description: Extra of case 43: the Date 2020-01-01 of DS_2 is the Time
                     2020-01-01/2020-01-01, which setdiff takes out of DS_1. The pandas engine kept
                     both Data Points and the DuckDB engine raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43e", number_inputs=2, references_names=["1"])

    def test_GH_1352_43f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ keep Me_2 , Me_3 ] [ unpivot Id_2 , Me_4 ];
        Description: Extra of case 43: the Time_Period Measure Me_2 unpivoted into the Time Measure
                     Me_4 holds the Time values 2020-01-01/2020-03-31 and 2021-12-01/2021-12-31.
                     Both engines kept the Time_Period texts 2020-Q1 and 2021-M12.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43f", number_inputs=1, references_names=["1"])

    def test_GH_1352_43g(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 ) = DS_2;
        Description: Extra of case 43: the nvl of the Date DS_1 and the Time DS_2 is a Time, which
                     = compares with DS_2. The DuckDB engine took the nvl as a Date and raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_43g", number_inputs=2, references_names=["1"])

    def test_GH_1352_13(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := time_agg ( "A" , _ , SC_1 ) ];
        Description: The input Scalar SC_1 is a Time_Period inside the clause too, so time_agg
                     gives 2020 for 2020Q2. The DuckDB engine typed only the Components of the
                     clause and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1352_13", number_inputs=1, references_names=["1"], scalars={"SC_1": "2020Q2"}
        )

    def test_GH_1352_14(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b
                        calc Me_2 := a#Me_1 > b#Me_1 keep Me_2 );
        Description: Durations named by their qualified name are ordered by their length, A > M,
                     as plain Components are. The DuckDB engine compared their letters.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_14", number_inputs=2, references_names=["1"])

    def test_GH_1352_15(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b
                        calc Me_2 := getmonth ( a#Me_1 ) keep Me_2 );
        Description: A Time_Period named by its qualified name keeps its type, so getmonth gives
                     1 for 2020Q1 and 12 for 2021M12. The DuckDB engine raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_15", number_inputs=2, references_names=["1"])

    def test_GH_1352_40e(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := time_agg ( "A" , _ , nvl ( Me_1 , Me_1 ) ) ,
                        Me_3 := time_agg ( "A" , _ , if Id_1 = 1 then Me_1 else Me_1 ) ,
                        Me_4 := time_agg ( "A" , _ , case when Id_1 = 1 then Me_1 else Me_1 ) ];
        Description: Extra of case 40: the nvl, the if and the case of the Time_Period Me_1 are
                     Time_Periods, which time_agg aggregates to 2020 and 2021. The DuckDB engine
                     read them as Dates and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40e", number_inputs=1, references_names=["1"])

    def test_GH_1352_16(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := period_indicator ( Id_2 ) ]
                        [ aggr Me_3 := max ( Me_2 ) , Me_4 := min ( Me_2 ) group by Id_1 ];
        Description: The Duration that a calc computes is ordered by its length, so max gives A
                     and min gives D. The DuckDB engine typed the Component as a Number and took
                     the max and the min of the letters, M and A.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_16", number_inputs=1, references_names=["1"])

    def test_GH_1352_40b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_1 , Me_1 ) ,
                        Me_3 := if Id_1 = 1 then Me_1 else Me_1 ,
                        Me_4 := case when Id_1 = 1 then Me_1 else Me_1 ]
                        [ calc Me_5 := cast ( Me_2 , string ) , Me_6 := cast ( Me_3 , string ) ,
                        Me_7 := cast ( Me_4 , string ) ] [ keep Me_5 , Me_6 , Me_7 ];
        Description: Extra of case 40: a Component that a calc computes as the nvl, the if or the
                     case of the String Me_1 is a String, which cast to string keeps. The DuckDB
                     engine typed it as a Number and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40b", number_inputs=1, references_names=["1"])

    def test_GH_1352_40c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_1 , Me_1 ) ,
                        Me_3 := if Id_1 = 1 then Me_1 else Me_1 ,
                        Me_4 := case when Id_1 = 1 then Me_1 else Me_1 ]
                        [ calc Me_5 := cast ( Me_2 , string ) , Me_6 := cast ( Me_3 , string ) ,
                        Me_7 := cast ( Me_4 , string ) ] [ keep Me_5 , Me_6 , Me_7 ];
        Description: Extra of case 40: a Component that a calc computes as the nvl, the if or the
                     case of the Integer Me_1 is an Integer, which cast to string writes 1. The
                     DuckDB engine typed it as a Number and wrote 1.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40c", number_inputs=1, references_names=["1"])

    def test_GH_1352_40d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_1 , Me_1 ) ,
                        Me_3 := if Id_1 = 1 then Me_1 else Me_1 ,
                        Me_4 := case when Id_1 = 1 then Me_1 else Me_1 ]
                        [ calc Me_5 := cast ( Me_2 , string ) , Me_6 := cast ( Me_3 , string ) ,
                        Me_7 := cast ( Me_4 , string ) ] [ keep Me_5 , Me_6 , Me_7 ];
        Description: Extra of case 40: a Component that a calc computes as the nvl, the if or the
                     case of the Boolean Me_1 is a Boolean, which cast to string writes true. The
                     DuckDB engine typed it as a Number and wrote 1.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40d", number_inputs=1, references_names=["1"])

    def test_GH_1352_17(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := max ( Me_1 over ( ) ) ,
                        Me_3 := min ( Me_1 over ( ) ) ];
        Description: The analytic max and min of Durations order them by their length, so max
                     gives A and min gives M. Both engines took the max and the min of the
                     letters.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_17", number_inputs=1, references_names=["1"])

    def test_GH_1352_18(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := rank ( over ( order by Id_1 ) ) ,
                        Me_3 := lag ( Id_1 , 1 over ( order by Id_1 ) ) ];
        Description: An analytic ordered by a Duration orders it by its length, D, M, A, so the
                     ranks of A, D and M are 3, 1 and 2. Both engines ordered the letters.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_18", number_inputs=1, references_names=["1"])

    def test_GH_1352_35(self):
        """
        Status: OK
        Expression: SC_r <- cast ( "2020-01-01/2020-03-31" , time_period );
                    DS_r <- DS_1 [ calc Me_2 := SC_r ];
        Description: A start/end text is the Time_Period of its interval when that is a year, a
                     half, a quarter or a month, so it casts to 2020Q1. The DuckDB engine read its
                     leading 2020-01 as the month 2020M1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_35", number_inputs=1, references_names=["1", "2"])

    def test_GH_1352_36(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ];
        Description: A String Component holding start/end texts casts to the Time_Periods of
                     their intervals, 2020Q1 and 2020. The DuckDB engine raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_36", number_inputs=1, references_names=["1"])

    def test_GH_1352_37(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ];
        Description: A String Component holding the date 2020-1-1, written without zeros, casts
                     to 2020D1, as the literal does. The DuckDB engine raised a raw
                     ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_37", number_inputs=1, references_names=["1"])

    def test_GH_1352_37b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1;
        Description: Extra of case 37: a Time_Period Identifier loaded as a date written without
                     zeros, 2020-1-5, is the day 2020D5, as the pandas engine reads it. The DuckDB
                     engine raised 0-3-1-6.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_37b", number_inputs=1, references_names=["1"])

    def test_GH_1352_38(self):
        """
        Status: OK
        Expression: SC_r <- cast ( "2020-1-1" , time_period );
                    DS_r <- DS_1 [ calc Me_2 := SC_r ];
        Description: Control: the literal date 2020-1-1, written without zeros, casts to 2020D1
                     on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_38", number_inputs=1, references_names=["1", "2"])

    def test_GH_1352_40(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( nvl ( Me_1 , SC_1 ) , string ) ];
        Description: The nvl of the Time_Period Me_1 and the Time_Period Scalar SC_1 is a
                     Time_Period, whose text is 2020Q1, as for cast ( Me_1 , string ). The DuckDB
                     engine wrote the internal text 2020-Q1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1352_40", number_inputs=1, references_names=["1"], scalars={"SC_1": "2020Q2"}
        )

    def test_GH_1352_13b(self):
        """
        Status: OK
        Expression: define operator f ( x time_period ) returns time_period is x end operator;
                    define operator g ( x component < time_period > )
                        returns component < time_period > is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := time_agg ( "A" , _ , f ( SC_1 ) ) ,
                        Me_3 := time_agg ( "A" , _ , g ( Me_1 ) ) ,
                        Me_4 := getmonth ( g ( Me_1 ) ) ];
        Description: Extra of case 13: the result of a user defined operator that returns a
                     Time_Period is a Time_Period, so time_agg gives 2020 and 2021 and getmonth
                     gives 1 and 12. The DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1352_13b", number_inputs=1, references_names=["1"], scalars={"SC_1": "2020Q2"}
        )

    def test_GH_1352_40f(self):
        """
        Status: OK
        Expression: define operator f ( x component < time_period > )
                        returns component < time_period > is x end operator;
                    define operator g ( x component < date > ) returns component < date > is x
                        end operator;
                    DS_r <- DS_1 [ calc Me_3 := cast ( f ( Me_1 ) , string ) ,
                        Me_4 := cast ( g ( Me_2 ) , string ) ];
        Description: Extra of case 40: cast to string of the Time_Period and the Date that user
                     defined operators return writes 2020Q1 and 2020-01-15, as cast of the
                     Components does. The DuckDB engine wrote 2020-Q1 and 2020-01-15 00:00:00.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40f", number_inputs=1, references_names=["1"])

    def test_GH_1352_18b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := lag ( Me_1 , 1 , "Q" over ( order by Id_1 ) ) ];
        Description: Extra of case 18: lag ordered by the Duration Id_1 (D, M, A) gives the
                     default Q to the first Data Point, D. The pandas engine read the default as a
                     column name and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_18b", number_inputs=1, references_names=["1"])

    def test_GH_1352_18c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := lag ( Me_1 , 1 , "z" over ( order by Id_1 ) ) ,
                        Me_4 := lag ( Me_2 , 1 , cast ( "2019-01-01" , date )
                            over ( order by Id_1 ) ) ,
                        Me_5 := lead ( Me_1 , 1 , null over ( order by Id_1 ) ) ]
                        [ keep Me_3 , Me_4 , Me_5 ];
        Description: Extra of case 18: the default of lag and lead is a value, a String, a Date
                     given by cast or null. The pandas engine read a text and null as column
                     names and wrote the cast into its query, raising raw DuckDB errors.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_18c", number_inputs=1, references_names=["1"])

    def test_GH_1352_17b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := max ( nvl ( Me_1 , Me_1 ) over ( ) ) ,
                        Me_3 := min ( if Id_1 = 1 then Me_1 else Me_1 over ( ) ) ,
                        Me_4 := max ( case when Id_1 = 1 then Me_1 else Me_1 over ( ) ) ];
        Description: Extra of case 17: the nvl, the if and the case of a Duration are Durations,
                     so the analytic max and min order them by their length and give A and M.
                     The DuckDB engine ordered the letters and gave M and A.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_17b", number_inputs=1, references_names=["1"])

    def test_GH_1352_17c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( nvl ( Me_1 , Me_1 ) , string ) ];
        Description: Extra of case 17: the nvl of a Duration is a Duration, which cast to string
                     writes as P1Y and P1M, as cast ( Me_1 , string ) does. The DuckDB engine
                     wrote the letters A and M.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_17c", number_inputs=1, references_names=["1"])

    def test_GH_1352_17d(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b
                        aggr Me_2 := max ( nvl ( a#Me_1 , b#Me_1 ) ) ,
                        Me_3 := min ( nvl ( a#Me_1 , b#Me_1 ) ) );
        Description: Extra of case 17: the max and the min in the aggr of a join of the nvl of
                     two Durations order them by their length and give A and M. The DuckDB
                     engine ordered the letters and gave M and A.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_17d", number_inputs=2, references_names=["1"])

    def test_GH_1352_17e(self):
        """
        Status: OK
        Expression: define operator f ( x component < duration > )
                        returns component < duration > is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := f ( Me_1 ) ]
                        [ aggr Me_3 := max ( Me_2 ) , Me_4 := min ( Me_2 ) ];
        Description: Extra of case 17: a Component that a calc computes with a user defined
                     operator returning a Duration is a Duration, so the max and the min of the
                     aggr order it by its length and give A and M. The DuckDB engine typed it as
                     a Number, ordered the letters and gave M and A.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_17e", number_inputs=1, references_names=["1"])

    def test_GH_1352_40g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "2020Q2" , time_period ) ]
                        [ calc Me_3 := getyear ( Me_2 ) ] [ keep Me_3 ];
        Description: Extra of case 40: a Component that a calc computes with a cast to
                     time_period is a Time_Period, also when a later keep drops it, so getyear
                     gives 2020. The DuckDB engine typed it as a Number and raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40g", number_inputs=1, references_names=["1"])

    def test_GH_1352_40h(self):
        """
        Status: OK
        Expression: define operator f ( x component < time_period > )
                        returns component < time_period > is x end operator;
                    DS_r <- DS_1 [ calc Me_2 := f ( Me_1 ) ] [ calc Me_3 := cast ( Me_2 , string ) ]
                        [ keep Me_3 ];
        Description: Extra of case 40: a Component that a calc computes with a user defined
                     operator returning a Time_Period is a Time_Period, also when a later keep
                     drops it, so cast to string writes 2020Q1 and 2021M12. The DuckDB engine
                     typed it as a Number and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40h", number_inputs=1, references_names=["1"])

    def test_GH_1352_40i(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , string ) ]
                        [ calc Me_3 := cast ( Me_2 , string ) ] [ keep Me_3 ];
        Description: Extra of case 40: a Component that a calc computes with a cast to string is
                     a String, also when a later keep drops it, so a second cast to string keeps
                     2020-01-15. The DuckDB engine typed it as a Number and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40i", number_inputs=1, references_names=["1"])

    def test_GH_1352_40j(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "2020-01-15" , date ) ]
                        [ calc Me_3 := cast ( Me_2 , string ) ] [ keep Me_3 ];
        Description: Extra of case 40: a Component that a calc computes with a cast to date is a
                     Date, also when a later keep drops it, so cast to string writes 2020-01-15.
                     The DuckDB engine typed it as a Number and raised 2-1-19-8.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_40j", number_inputs=1, references_names=["1"])

    def test_GH_1352_36b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ];
        Description: Extra of case 36: the year of a Time_Period is written with four digits,
                     from 1000, so the text 0001 is no Time_Period and the cast raises 2-1-5-1,
                     as for 10000. The pandas engine gave 1 and the DuckDB engine raised a raw
                     ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36b",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message="Impossible to cast 0001 from type String to Time_Period.",
        )

    def test_GH_1352_36c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ,
                        Me_3 := cast ( " 2020Q1" , time_period ) ];
        Description: Extra of case 36: a text cast to time_period is read without the space
                     around it, as the loaders read a Time_Period and every path reads a Date,
                     so " 2020Q1" is 2020Q1 and " 2020-01-15" is 2020D15. The pandas engine
                     raised 2-1-19-2 and a raw ValueError, the DuckDB engine 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_36c", number_inputs=1, references_names=["1"])

    def test_GH_1352_13d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := max ( ( if Id_1 = 1
                        then cast ( "2021Q1" , time_period ) else Me_1 ) over ( ) ) ]
                        [ keep Me_2 ];
        Description: Extra of case 13: a Time_Period given by cast is held in the one form the
                     loaded ones are held in, so the analytic max orders the periods and gives
                     2021Q2. The pandas engine held the cast text as written and ordered the
                     texts, giving 2021Q1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_13d", number_inputs=1, references_names=["1"])

    def test_GH_1352_36d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ];
        Description: Extra of case 36: the start/end text of a year below 1000,
                     0001-01-01/0001-12-31, is no Time_Period, whose year is written with four
                     digits from 1000, so the cast raises 2-1-5-1. The pandas engine raised a raw
                     ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36d",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message=(
                "Impossible to cast 0001-01-01/0001-12-31 from type String to Time_Period."
            ),
        )

    def test_GH_1352_36e(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "2020" , time ) ,
                        Me_3 := cast ( "2020-01" , time ) ]
                        [ filter Me_2 = Me_2 and Me_3 = Me_3 ]
                        [ calc Me_4 := cast ( Me_2 , string ) , Me_5 := cast ( Me_3 , string ) ]
                        [ keep Me_4 , Me_5 ];
        Description: Extra of case 36: a Time written as the year or the month it covers is
                     equal to itself, so the filter keeps the Data Point with 2020 and 2020-01.
                     The pandas engine read it as a start/end interval and raised a raw
                     TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_36e", number_inputs=1, references_names=["1"])

    def test_GH_1352_36f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "0001" , time_period ) ];
        Description: Extra of case 36: the literal 0001 is no Time_Period, whose year is written
                     with four digits from 1000, so the cast raises 1-1-5-6. The pandas
                     engine gave the period 1 and the DuckDB engine raised a raw ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36f",
            number_inputs=1,
            exception_code="1-1-5-6",
            exception_message="Impossible to cast 0001 from type String to Time_Period.",
        )

    def test_GH_1352_36g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "2020M13" , time_period ) ];
        Description: Control of case 36: the literal 2020M13 is no Time_Period, so the cast
                     raises 1-1-5-6 on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36g",
            number_inputs=1,
            exception_code="1-1-5-6",
            exception_message="Impossible to cast 2020M13 from type String to Time_Period.",
        )

    def test_GH_1352_36h(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "0001-01-01/0001-12-31" ,
                        time_period ) ];
        Description: Extra of case 36: the literal start/end text of a year below 1000 is no
                     Time_Period, so the cast raises 1-1-5-6. Both engines and the semantic
                     analysis raised a raw ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36h",
            number_inputs=1,
            exception_code="1-1-5-6",
            exception_message=(
                "Impossible to cast 0001-01-01/0001-12-31 from type String to Time_Period."
            ),
        )

    def test_GH_1352_13e(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier Id_9 := cast ( Id_T , time_period ) ]
                        [ filter Id_9 = cast ( "2020Q1" , time_period ) ];
        Description: Extra of case 13: a Time_Period given by the cast of a Time is held in the
                     one form the loaded ones and the cast literals are held in, so the filter
                     keeps the Data Point of 2020Q1. The DuckDB engine held the cast Time in
                     another form, so the filter kept nothing.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_13e", number_inputs=1, references_names=["1"])

    def test_GH_1352_36i(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ]
                        [ filter Id_1 = 3 ];
        Description: Extra of case 36: a Time of a year below 1000, 0500-01-01/0500-12-31 or
                     0999-01-01/0999-03-31, is no Time, whose years run from 1000 to 9999 as the
                     ones of a Date and a Time_Period, so loading it raises 0-3-1-6 before the
                     cast. The DuckDB engine raised a raw ValueError or 2-1-19-10 for the cast.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.DataLoadExceptionTest(
            code="GH_1352_36i",
            number_inputs=1,
            exception_code="0-3-1-6",
            exception_message=(
                "On Dataset DS_1 loading: not possible to cast column Me_1 to Time. Error found: "
                '"Time 0500-01-01/0500-12-31 is invalid. Year must be between 1000 and 9999."'
            ),
        )

    def test_GH_1352_36l(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ];
        Description: Extra of case 36: the Time 9999-12-27/9999-12-31 is no period, as the week
                     that starts on 9999-12-27 ends in the year 10000, so the cast raises
                     2-1-5-1. The pandas engine raised a raw ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36l",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message=(
                "Impossible to cast 9999-12-27/9999-12-31 from type Time to Time_Period."
            ),
        )

    def test_GH_1352_36m(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , -2000 );
        Description: Extra of case 36j: timeshift back by more periods than a year has moves the
                     year back as many years as they cover, so 2020Q1 back by 2000 quarters is
                     1520Q1. The pandas engine moved the year back by one and raised a raw
                     ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_36m", number_inputs=1, references_names=["1"])

    def test_GH_1352_36n(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time_period ) ]
                        [ filter Id_1 = 1 ];
        Description: Extra of case 36: the text abc is no Time_Period, so the cast raises 2-1-5-1,
                     also for a Data Point that the filter drops. The DuckDB engine cast only the
                     Data Point the filter keeps.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36n",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message="Impossible to cast abc from type String to Time_Period.",
        )

    def test_GH_1352_34(self):
        """
        Status: OK
        Expression: DS_r <- max ( DS_1 group by Id_1 );
        Description: max of a Time_Period Measure grouped by an Integer Identifier gives the
                     latest period of each group, 2020Q2 and 2021Q1. The pandas engine raised a
                     raw InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_34", number_inputs=1, references_names=["1"])

    def test_GH_1352_19(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , SC_2 );
        Description: The Integer Scalar SC_2 that gives timeshift its shift number holds null,
                     so no period can be shifted and timeshift raises 2-1-19-24, as time_agg
                     does for a null Scalar. The pandas engine raised a raw TypeError and the
                     DuckDB engine a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_19",
            number_inputs=1,
            exception_code="2-1-19-24",
            exception_message="At op timeshift: The scalar SC_2 that gives the shift number has "
            "no value.",
            scalars={"SC_2": None},
        )

    def test_GH_1352_21(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_1 , timeshift ( DS_1 , 10 ) );
        Description: timeshift of a Date Identifier gives Dates, written as the bare date, so
                     union writes 2020-11-30 and 2020-12-29. The pandas engine wrote
                     2020-11-30T00:00:00 and 2020-12-29T00:00:00.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_21", number_inputs=1, references_names=["1"])

    def test_GH_1352_22(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( timeshift ( DS_1 , 1 ) , 1 );
        Description: timeshift reads the Dates that another timeshift gives, so 2020-01-31 and
                     2020-02-29 move two months to 2020-03-29 and 2020-04-29. The pandas engine
                     raised a raw TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_22", number_inputs=1, references_names=["1"])

    def test_GH_1352_30(self):
        """
        Status: OK
        Expression: SC_r <- dateadd ( cast ( "2020-02-29" , date ) , 8000 , "A" );
        Description: dateadd of 2020-02-29 by 8000 years gives the year 10020, which no Date
                     has, so it raises 2-1-19-10. The pandas engine raised a raw ValueError and
                     the DuckDB engine a raw NotImplementedError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_30",
            number_inputs=1,
            exception_code="2-1-19-10",
            exception_message="Invalid year 10020, must be between 1000 and 9999.",
        )

    def test_GH_1352_23(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( dateadd ( Me_1 , 1 , "M" ) , string ) ];
        Description: The Date that dateadd gives is cast to string as any Date, 2020-02-15. The
                     DuckDB engine wrote 2020-02-15 00:00:00.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_23", number_inputs=1, references_names=["1"])

    def test_GH_1352_24(self):
        """
        Status: OK
        Expression: DS_r <- cast ( dateadd ( DS_1 , -1 , "Q" ) , string );
        Description: dateadd of a Time_Period Data Set gives a Date Measure, which is cast to
                     string as a Date: 2019-12-31 and 2020-03-30. The DuckDB engine read it as a
                     Time_Period and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_24", number_inputs=1, references_names=["1"])

    def test_GH_1352_25(self):
        """
        Status: OK
        Expression: DS_r <- dateadd ( dateadd ( DS_1 , -1 , "Q" ) , -1 , "Q" );
        Description: The outer dateadd reads the Date Measure that the inner one gives:
                     2019-09-30 and 2019-12-30. The DuckDB engine read it as a Time_Period and
                     raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_25", number_inputs=1, references_names=["1"])

    def test_GH_1352_26(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 1 );
        Description: 2020 has 53 ISO weeks, so 2020W53 moves one week to 2021W1 and 2021W1 to
                     2021W2. Both engines counted 52 weeks in every year and gave 2021W2 twice.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_26", number_inputs=1, references_names=["1"])

    def test_GH_1352_27(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single ) [ filter Id_2 >= cast ( "2020W52" ,
                        time_period ) and Id_2 <= cast ( "2021W1" , time_period ) ];
        Description: fill_time_series adds the week 2020W53 between 2020W52 and 2021W1. The
                     DuckDB engine counted 52 weeks in every year and left it out.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_27", number_inputs=1, references_names=["1"])

    def test_GH_1352_28(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := time_agg ( "Q" , _ , Id_2 ) ,
                        Me_3 := time_agg ( "A" , _ , Id_2 ) ];
        Description: A week stays in the year it is numbered in, so 2020W53 (2020-12-28 to
                     2021-01-03) goes to 2020Q4 and 2020. The pandas engine gave 2020Q1 and the
                     DuckDB engine 2021Q1 and 2021.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_28", number_inputs=1, references_names=["1"])

    def test_GH_1352_28b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := time_agg ( "A" , _ , Me_1 ) ,
                        Me_3 := time_agg ( "S" , _ , Me_1 ) , Me_4 := time_agg ( "Q" , _ , Me_1 ) ,
                        Me_5 := time_agg ( "M" , _ , Me_1 ) ];
        Description: Extra of case 28: a week that ends in the next year, 2021W52 or 9999W52,
                     goes to the last period of its own year, so time_agg of the last week of
                     9999 stays in the years 1000 to 9999 of a Time_Period. The pandas engine
                     raised a raw ValueError for 9999W52 and gave the first period of 2021 for
                     2021W52, and the DuckDB engine wrote a null for 9999W52 and 2022 for
                     2021W52.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_28b", number_inputs=1, references_names=["1"])

    def test_GH_1352_32b(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single );
        Description: Extra of case 32: over the period indicators 2020Q1, 2020-04, 2020W20,
                     2020-1-5 and 2020, fill_time_series fills each indicator over the year 2020,
                     which has 53 ISO weeks and 366 days. The DuckDB engine counted 52 weeks and
                     365 days in every year and left 2020W53 and 2020D366 out.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_32b", number_inputs=1, references_names=["1"])

    def test_GH_1352_31(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single );
        Description: fill_time_series only adds Data Points, so 2020-01-02, whose Measure is
                     null, stays and 2020-01-03 is added. The pandas engine removed 2020-01-02.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_31", number_inputs=1, references_names=["1"])

    def test_GH_1352_32(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , all );
        Description: fill_time_series in all mode fills a single Data Point, 2020Q2, to the
                     whole of its year, 2020Q1 to 2020Q4, as it fills two. The pandas engine
                     returned the Data Point alone.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_32", number_inputs=1, references_names=["1"])

    def test_GH_1352_33(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , all );
        Description: Control of case 32: fill_time_series in all mode fills 2020Q1 and 2020Q3
                     to the whole of their year, 2020Q1 to 2020Q4, on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_33", number_inputs=1, references_names=["1"])

    def test_GH_1352_36o(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := cast ( Me_1 , time_period ) ] [ keep Me_2 ];
        Description: Extra of case 36: the text abc is no Time_Period, so the cast raises
                     2-1-5-1, also when the keep that follows removes the Measure it gives. The
                     DuckDB engine did not compute the removed Measure and returned the Data Set.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36o",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message="Impossible to cast abc from type String to Time_Period.",
        )

    def test_GH_1352_20(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , -5 );
        Description: timeshift moves 2020Q1 and 2020Q2 back five quarters, more than a year, to
                     2018Q4 and 2019Q1. The pandas engine raised 2-1-19-7.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_20", number_inputs=1, references_names=["1"])

    def test_GH_1352_29(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 8000 );
        Description: timeshift of 2020A by 8000 years gives the year 10020, which no Time_Period
                     has, so it raises 2-1-19-10. The DuckDB engine wrote null Identifiers.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_29",
            number_inputs=1,
            exception_code="2-1-19-10",
            exception_message="Invalid year 10020, must be between 1000 and 9999.",
        )

    def test_GH_1352_31b(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single );
        Description: Extra of case 31: the Measure of a Date Data Set may have any name, also
                     _added, and fill_time_series keeps it: 1.0, null, null, 2.0. The pandas
                     engine marked the Data Points it added in a column of that name and removed
                     the Measure.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_31b", number_inputs=1, references_names=["1"])

    def test_GH_1352_31c(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single );
        Description: Extra of case 31: the Measure of a Time_Period Data Set may have any name,
                     also _freq, and fill_time_series keeps it: 1.0, null, null, 2.0. The pandas
                     engine grouped the periods by a column of that name and removed the Measure.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_31c", number_inputs=1, references_names=["1"])

    def test_GH_1352_19b(self):
        """
        Status: OK
        Expression: SC_n := SC_2 + 1 ; DS_r <- timeshift ( DS_1 , SC_n ) ;
        Description: Extra of case 19: the Scalar SC_n that the script computes from the null
                     SC_2 holds null, so timeshift raises 2-1-19-24 as for an input Scalar. The
                     DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_19b",
            number_inputs=1,
            exception_code="2-1-19-24",
            exception_message="At op timeshift: The scalar SC_n that gives the shift number has "
            "no value.",
            scalars={"SC_2": None},
        )

    def test_GH_1352_19c(self):
        """
        Status: OK
        Expression: define operator f ( x dataset , n integer ) returns dataset is
                        timeshift ( x , n ) end operator ; DS_r <- f ( DS_1 , SC_2 ) ;
        Description: Extra of case 19: the null Scalar SC_2 given to the parameter of a user
                     defined operator that gives timeshift its shift number raises 2-1-19-24,
                     naming SC_2. The DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_19c",
            number_inputs=1,
            exception_code="2-1-19-24",
            exception_message="At op timeshift: The scalar SC_2 that gives the shift number has "
            "no value.",
            scalars={"SC_2": None},
        )

    def test_GH_1352_19d(self):
        """
        Status: OK
        Expression: define operator f ( x dataset , n integer ) returns dataset is
                        timeshift ( x , n ) end operator ; DS_r <- f ( DS_1 , 1 ) ;
        Description: Extra of case 19: the parameter of a user defined operator gives timeshift
                     its shift number, so 2020Q1 and 2020Q2 move to 2020Q2 and 2020Q3. The DuckDB
                     engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_19d", number_inputs=1, references_names=["1"])

    def test_GH_1352_19e(self):
        """
        Status: OK
        Expression: define operator f ( x dataset , n integer ) returns dataset is
                        timeshift ( x , n ) end operator ; DS_r <- f ( DS_1 , 1 ) ;
        Description: Extra of case 19: the parameter of a user defined operator gives timeshift
                     of a Date Identifier its shift number, so 2020-01-31 and 2020-02-29 move one
                     month to 2020-02-29 and 2020-03-29. The DuckDB engine raised a raw
                     ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_19e", number_inputs=1, references_names=["1"])

    def test_GH_1352_19f(self):
        """
        Status: OK
        Expression: define operator f ( x dataset , n integer ) returns dataset is
                        timeshift ( x , n ) end operator ; DS_r <- f ( DS_1 , null ) ;
        Description: Extra of case 19: a null given to the parameter of a user defined operator
                     that gives timeshift its shift number raises 2-1-19-24, naming the parameter
                     n. The pandas engine named the scalar None and the DuckDB engine raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_19f",
            number_inputs=1,
            exception_code="2-1-19-24",
            exception_message="At op timeshift: The scalar n that gives the shift number has "
            "no value.",
        )

    def test_GH_1352_36p(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , date ) ] [ keep Me_3 ];
        Description: Extra of case 36: the text notadate is no Date, so the cast raises
                     2-1-19-8, also when the keep that follows removes the Measure it gives. The
                     DuckDB engine did not compute the removed Measure and returned the Data Set.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p",
            number_inputs=1,
            exception_code="2-1-19-8",
            exception_message="Invalid date format, must be YYYY-MM-DD: notadate",
        )

    def test_GH_1352_27b(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 1 );
        Description: Extra of case 27: 2020 has 366 days, so 2020D365 moves one day to 2020D366,
                     2020D366 to 2021D1 and 2021D1 to 2021D2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_27b", number_inputs=1, references_names=["1"])

    def test_GH_1352_26e(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 1 );
        Description: Extra of case 26: timeshift of the monthly Dates 9999-11-30 and 9999-12-31
                     by one month gives the year 10000, which no Date has, so it raises
                     2-1-19-10, as dateadd does.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_26e",
            number_inputs=1,
            exception_code="2-1-19-10",
            exception_message="Invalid year 10000, must be between 1000 and 9999.",
        )

    def test_GH_1352_26f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , date ) ,
                        Me_3 := dateadd ( Id_2 , 1 , "M" ) ];
        Description: Extra of case 26: a Date has the years 1000 to 9999, as a Time_Period, so
                     the Dates 1500-01-31 and 1000-12-31 load, the texts 1500-06-15 and
                     1799-12-31 cast to Dates, and dateadd moves the loaded ones one month, to
                     1500-02-28 (1500 is no leap year) and 1001-01-31. Both engines read the
                     years 1800 to 9999 only and refused the data.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_26f", number_inputs=1, references_names=["1"])

    def test_GH_1352_36q(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier Id_9 := cast ( Me_1 , time_period ) ]
                        [ sub Id_9 = cast ( "2020Q1" , time_period ) ];
        Description: Extra of case 36: an Identifier cannot be calculated from the cast of the
                     nullable Me_1, which the structures alone tell, so the calc raises 1-1-1-16
                     before the text abc is cast. The pandas engine computed the cast first and
                     raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36q",
            number_inputs=1,
            exception_code="1-1-1-16",
            exception_message="Identifier Id_9 cannot be calculated from an expression that is "
            "declared nullable.",
        )

    def test_GH_1352_36r(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ keep Me_1 ] [ calc Me_2 := max ( Me_1 over ( order by Id_1 ) )
                        + 1 ];
        Description: Extra of case 36: a Time_Period plus an Integer has no type, which the
                     structures alone tell, so the calc raises 1-1-1-2 before max compares
                     2020Q1 with 2020M1. The pandas engine computed max first and raised
                     2-1-19-20.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36r",
            number_inputs=1,
            exception_code="1-1-1-2",
            exception_message="Invalid implicit cast from Time_Period and Integer to Number.",
        )

    def test_GH_1352_36s(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter cast ( Me_1 , integer ) > 0 and Me_1 ];
        Description: Extra of case 36: and takes no String, which the structures alone tell, so
                     the filter raises 1-1-1-2 before the text abc is cast. The pandas engine
                     computed the cast first and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36s",
            number_inputs=1,
            exception_code="1-1-1-2",
            exception_message="Invalid implicit cast from Boolean and String to Boolean.",
        )

    def test_GH_1352_36t(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := sum ( cast ( Me_1 , integer ) ) ,
                        Me_3 := sum ( Me_1 ) group by Id_1 ];
        Description: Extra of case 36: sum takes no String, which the structures alone tell, so
                     the aggr raises 1-1-1-1 before the text abc is cast. The pandas engine
                     computed the cast first and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36t",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from String to Number.",
        )

    def test_GH_1352_36u(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , integer ) , Id_1 := 1 ];
        Description: Extra of case 36: a calc cannot overwrite the Identifier Id_1, which the
                     structures alone tell, so it raises 1-1-6-13 before the text abc is cast.
                     The pandas engine computed the cast first and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36u",
            number_inputs=1,
            exception_code="1-1-6-13",
            exception_message="At op calc: Not allowed to overwrite an Identifier: Id_1",
        )

    def test_GH_1352_26h(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 1 );
        Description: Extra of case 26: timeshift of the Time 9999-01-01/9999-12-31 by one year
                     gives the year 10000, which no Time has, so it raises 2-1-19-10, as for a
                     Date. The pandas engine raised a raw OverflowError and the DuckDB engine
                     wrote 10000-01-01/10000-12-31.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_26h",
            number_inputs=1,
            exception_code="2-1-19-10",
            exception_message="Invalid year 10000, must be between 1000 and 9999.",
        )

    def test_GH_1352_26i(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , -1 );
        Description: Extra of case 26: timeshift of the Time 1000-01-01/1000-12-31 back one year
                     gives the year 999, which no Time has, so it raises 2-1-19-10. Both engines
                     wrote a Time of the year 999.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_26i",
            number_inputs=1,
            exception_code="2-1-19-10",
            exception_message="Invalid year 999, must be between 1000 and 9999.",
        )

    def test_GH_1352_32c(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single );
        Description: Extra of case 32: fill_time_series adds 9999-11-30 between the monthly
                     Dates 9999-10-31 and 9999-12-31, the last month a Date has. The pandas
                     engine raised a raw ValueError computing the month after the last one.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_32c", number_inputs=1, references_names=["1"])

    def test_GH_1352_32d(self):
        """
        Status: OK
        Expression: DS_r <- fill_time_series ( DS_1 , single );
        Description: Extra of case 32: fill_time_series adds 9999-11-01/9999-11-30 between the
                     monthly Times of October and December 9999, the last month a Time has. The
                     pandas engine raised a raw OverflowError reading the frequency of the Time
                     that ends on 9999-12-31.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_32d", number_inputs=1, references_names=["1"])

    def test_GH_1352_26j(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := dateadd ( Me_1 , -1 , "W" ) ];
        Description: Extra of case 26: the week 9999W52 ends on 10000-01-02, so dateadd back one
                     week gives 9999-12-26, a Date of the years 1000 to 9999, as 2020W1 gives
                     2019-12-29.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_26j", number_inputs=1, references_names=["1"])

    def test_GH_1352_36v(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , integer ) ]
                        [ calc Me_3 := Me_1 + 1 ];
        Description: Extra of case 36: a String plus an Integer has no type, which the
                     structures alone tell, so the second calc raises 1-1-1-2 before the first
                     casts the text abc. The pandas engine computed the cast first and raised
                     2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36v",
            number_inputs=1,
            exception_code="1-1-1-2",
            exception_message="Invalid implicit cast from String and Integer to Number.",
        )

    def test_GH_1352_36w(self):
        """
        Status: OK
        Expression: DS_a := DS_1 [ calc Me_2 := cast ( Me_1 , integer ) ] ;
                    DS_r <- DS_a [ calc Me_3 := Me_1 + 1 ];
        Description: Extra of case 36: a String plus an Integer has no type, which the
                     structures alone tell, so the second statement raises 1-1-1-2 before the
                     first casts the text abc. The pandas engine computed the first statement
                     and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36w",
            number_inputs=1,
            exception_code="1-1-1-2",
            exception_message="Invalid implicit cast from String and Integer to Number.",
        )

    def test_GH_1352_36x(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as a , DS_2 as b
                        filter cast ( a#Me_1 , integer ) > 0 calc Me_3 := a#Me_1 + 1 );
        Description: Extra of case 36: a String plus an Integer has no type, which the
                     structures alone tell, so the calc of the join raises 1-1-1-2 before its
                     filter casts the text abc. The pandas engine computed the filter first and
                     raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36x",
            number_inputs=2,
            exception_code="1-1-1-2",
            exception_message="Invalid implicit cast from String and Integer to Number.",
        )

    def test_GH_1352_36y(self):
        """
        Status: OK
        Expression: DS_r <- cast ( DS_1#Me_1 , integer ) + DS_1#Me_1;
        Description: Extra of case 36: the cast gives the Measure int_var and the other operand
                     the Measure Me_1, which the structures alone tell, so the addition raises
                     1-1-14-1 before the text abc is cast. The pandas engine computed the cast
                     first and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36y",
            number_inputs=1,
            exception_code="1-1-14-1",
            exception_message="At op +: Measure names don't match: ['int_var'] - ['Me_1'].",
        )

    def test_GH_1352_36z(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := time_agg ( "A" , _ , Me_1 ) ];
        Description: Extra of case 36: time_agg of a Date needs first or last, which the
                     structures alone tell, so the script raises 1-1-19-11 before the Date
                     0999-01-15, out of the years 1000 to 9999, is loaded. The pandas engine
                     loaded the data first and raised 0-3-1-6.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36z",
            number_inputs=1,
            exception_code="1-1-19-11",
            exception_message="At op time_agg: If used over a Date, first/last parameter must be "
            "declared.",
        )

    def test_GH_1352_36p2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , date , "YYYY-MM-DD" ) ]
                        [ keep Me_3 ];
        Description: Extra of case 36: the text notadate does not follow the mask, so the cast
                     raises 2-1-5-3, also when the keep that follows removes the Measure it
                     gives. The DuckDB engine did not compute the removed Measure and returned
                     the Data Set.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p2",
            number_inputs=1,
            exception_code="2-1-5-3",
            exception_message="Impossible to cast notadate from type String to Date using mask "
            "YYYY-MM-DD.",
        )

    def test_GH_1352_36p3(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , time ) ] [ keep Me_3 ];
        Description: Extra of case 36: the text abc is no Time, so the cast raises 2-1-5-1, also
                     when the keep that follows removes the Measure it gives. The DuckDB engine
                     did not compute the removed Measure and returned the Data Set.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p3",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message="Impossible to cast abc from type String to Time.",
        )

    def test_GH_1352_36p4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := ratio_to_report ( Me_1 over
                        ( partition by Id_1 ) ) ] [ keep Me_3 ];
        Description: Extra of case 36: the values of Me_1 in the partition Id_1 = 1 add up to
                     zero, so ratio_to_report raises 2-1-3-1, also when the keep that follows
                     removes the Measure it gives. The DuckDB engine did not compute the removed
                     Measure and returned the Data Set.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p4",
            number_inputs=1,
            exception_code="2-1-3-1",
            exception_message="At op ratio_to_report: Division by zero or a value beyond the range "
            "of a Number produced infinite values.",
        )

    def test_GH_1352_36p5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := max ( Me_1 ) , Me_3 := sum ( Me_4 )
                        group by Id_1 ] [ keep Me_3 ];
        Description: Extra of case 36: the group Id_1 = 1 holds 2020Q1 and 2020M1, which max
                     cannot compare, so the aggr raises 2-1-19-20, also when the keep that
                     follows removes the Measure it gives. Both engines already raised it: the
                     DuckDB engine computes every aggregate of an aggr, whatever the keep reads.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p5",
            number_inputs=1,
            exception_code="2-1-19-20",
            exception_message="Time Period operands with different period indicators do not "
            "support < and > Comparison operations, unable to get the max",
        )

    def test_GH_1352_36p6(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := max ( Me_1 over ( partition by Id_1 ) ) ];
        Description: Extra of case 36: Me_1 holds 2020Q1, 2020M1 and 2020Q2, periods of
                     different indicators, which max cannot compare, so the analytic max raises
                     2-1-19-20, as the max of a Data Set does. The DuckDB engine compared the
                     texts and gave 2020Q1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p6",
            number_inputs=1,
            exception_code="2-1-19-20",
            exception_message="Time Period operands with different period indicators do not "
            "support < and > Comparison operations, unable to get the max",
        )

    def test_GH_1352_36p7(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := min ( Me_1 over ( partition by Id_1 ) ) ]
                        [ keep Me_4 ];
        Description: Extra of case 36: the periods of Me_1 have different indicators, so the
                     analytic min raises 2-1-19-20, also when the keep that follows removes the
                     Measure it gives. The DuckDB engine compared the texts and returned the Data
                     Set.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_36p7",
            number_inputs=1,
            exception_code="2-1-19-20",
            exception_message="Time Period operands with different period indicators do not "
            "support < and > Comparison operations, unable to get the min",
        )

    def test_GH_1352_26k(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( "0999-01-15" , date , "YYYY-MM-DD" ) ];
        Description: Extra of case 26: a Date has the years 1000 to 9999 also when a mask reads
                     it, so the cast of the literal 0999-01-15 raises 1-1-5-6, as the cast
                     without a mask does. The pandas engine raised the input error 0-1-2-7 and
                     the DuckDB engine and the semantic analysis gave the Date.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_26k",
            number_inputs=1,
            exception_code="1-1-5-6",
            exception_message="Impossible to cast 0999-01-15 from type String to Date.",
        )

    def test_GH_1352_26l(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 , date , "YYYY-MM-DD" ) ];
        Description: Extra of case 26: a Date has the years 1000 to 9999 also when a mask reads
                     it, so the cast of the text 0999-01-15 raises 2-1-5-3. Both engines gave the
                     Date 0999-01-15.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_26l",
            number_inputs=1,
            exception_code="2-1-5-3",
            exception_message="Impossible to cast 0999-01-15 from type String to Date using mask "
            "YYYY-MM-DD.",
        )

    def test_GH_1352_19g(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , SC_N ) ;
        Description: Extra of case 19: the shift of timeshift is a whole number of periods, so
                     the Number Scalar SC_N raises 1-1-19-15, as a Number shift of dateadd does.
                     Both engines took the Number 2.7: the pandas engine shifted 2020Q1 by two
                     quarters and the DuckDB engine by three.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1352_19g",
            number_inputs=1,
            exception_code="1-1-19-15",
            exception_message="At op timeshift: Invalid param data_type Number for param "
            "shiftNumber, expected Integer.",
            scalars={"SC_N": 2.7},
        )

    def test_GH_1352_26m(self):
        """
        Status: OK
        Expression: DS_r <- timeshift ( DS_1 , 1 );
        Description: Extra of case 26: timeshift of the yearly Times of 1500 and 1501, and of
                     9997 and 9998, by one year gives the Times of 1501 and 1502, and of 9998 and
                     9999. The pandas engine raised a raw OutOfBoundsDatetime with pandas 2, whose
                     nanosecond timestamps hold only the years 1677 to 2262.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1352
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1352_26m", number_inputs=1, references_names=["1"])


class BasicTypeValueBugs(BugHelper):
    """Values of the basic types converted, written and compared alike by both engines (issue
    #1353)."""

    def test_GH_1353_1(self):
        """
        Status: OK
        Expression: DS_r <- DS_2 + DS_1;
        Description: The Number Identifier Id_1 of DS_2 (1.0, 3.5) and the Integer one of DS_1
                     (1, 3) are matched as Numbers, so 3.5 does not match 3. The pandas engine
                     truncated 3.5 to 3 and matched it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_1", number_inputs=2, references_names=["1"])

    def test_GH_1353_1b(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: Extra of case 1: the Integer Id_1 of DS_1 and the Number Id_1 of DS_2 are
                     matched as Numbers and Id_1 is a Number in the result, whatever the order of
                     the operands, as in a join. Both engines typed Id_1 as the first operand,
                     and the pandas engine matched 3 with 3.5.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_1b", number_inputs=2, references_names=["1"])

    def test_GH_1353_1c(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_1 , DS_2 );
        Description: Extra of case 1: the Integer Id_1 of DS_1 is matched as a Number with the
                     Number Id_1 of DS_2, and it is a Number in the result. Both engines typed it
                     as an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_1c", number_inputs=2, references_names=["1"])

    def test_GH_1353_1d(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_2 , DS_1 );
        Description: Extra of case 1: the Number Id_1 of DS_2 is matched with the Integer Id_1
                     of DS_1 as Numbers, so 3.5 does not exist in DS_1. The pandas engine raised
                     a raw ArrowInvalid.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_1d", number_inputs=2, references_names=["1"])

    def test_GH_1353_2(self):
        """
        Status: OK
        Expression: DS_r <- intersect ( DS_2 , DS_1 );
        Description: The Number Id_1 of DS_2 and the Integer Id_1 of DS_1 are matched as Numbers:
                     only 1.0 is in both. The pandas engine raised a raw ArrowInvalid.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_2", number_inputs=2, references_names=["1"])

    def test_GH_1353_3(self):
        """
        Status: OK
        Expression: DS_r <- full_join ( DS_2 as A , DS_1 as B keep A#Me_1 );
        Description: The join key Id_1, a Number in DS_2 and an Integer in DS_1, is a Number in
                     the result and keeps 3.5. The pandas engine raised a raw ArrowInvalid, and
                     the DuckDB engine and the semantic analysis typed Id_1 as an Integer and
                     rounded 3.5 to 4.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_3", number_inputs=2, references_names=["1"])

    def test_GH_1353_3b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 , DS_1 );
        Description: Extra of case 3: the join key Id_1, a Number in DS_2 and an Integer in
                     DS_1, is a Number and its values are compared as Numbers, so 3.5 does not
                     match 3. The pandas engine raised a raw ArrowInvalid and the DuckDB engine
                     typed Id_1 as an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_3b", number_inputs=2, references_names=["1"])

    def test_GH_1353_3c(self):
        """
        Status: OK
        Expression: DS_r <- left_join ( DS_2 , DS_1 );
        Description: Extra of case 3: the join key Id_1 is a Number, so the data point 3.5 of
                     DS_2 keeps its key and finds no match. The pandas engine raised a raw
                     ArrowInvalid and the DuckDB engine typed Id_1 as an Integer and gave 4.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_3c", number_inputs=2, references_names=["1"])

    def test_GH_1353_3d(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 , DS_1 using Id_1 );
        Description: Extra of case 3: with using, the key Id_1 is a Number too and 3.5 does not
                     match 3. The pandas engine truncated 3.5 to 3 and matched it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_3d", number_inputs=2, references_names=["1"])

    def test_GH_1353_4(self):
        """
        Status: OK
        Expression: DS_a <- DS_1 + DS_2;
                    DS_r <- DS_2;
        Description: Matching the keys of DS_1 and DS_2 does not change the data of DS_2, which
                     the next statement reads with its key 3.5. The pandas engine wrote the
                     truncated key 3.0 into the data of DS_2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_4", number_inputs=2, references_names=["1", "2"])

    def test_GH_1353_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 + DS_4;
        Description: The Boolean Id_1 of DS_4 is cast to the String "true" to match the String
                     Id_1 of DS_3, as the implicit cast from boolean to string of the VTL 2.2
                     reference manual gives it, so only "true" matches. The pandas engine
                     matched nothing and the DuckDB engine cast "1" to true and matched it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5", number_inputs=2, references_names=["1"])

    def test_GH_1353_5b(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 = DS_4;
        Description: Extra of case 5: the Boolean Id_1 of DS_4 is matched as a String with the
                     String Id_1 of DS_3, which also holds "abc". The pandas engine matched
                     nothing and the DuckDB engine raised a raw ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5b", number_inputs=2, references_names=["1"])

    def test_GH_1353_5c(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_3 , DS_4 );
        Description: Extra of case 5: nvl matches the String Id_1 of DS_3 with the Boolean Id_1
                     of DS_4 as Strings. The pandas engine matched nothing and the DuckDB engine
                     raised a raw ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5c", number_inputs=2, references_names=["1"])

    def test_GH_1353_5d(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_3 , DS_4 );
        Description: Extra of case 5: "true" of the String Id_1 of DS_3 exists in DS_4, whose
                     Boolean Id_1 is matched as a String. The pandas engine gave false for all
                     and the DuckDB engine raised a raw ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5d", number_inputs=2, references_names=["1"])

    def test_GH_1353_5e(self):
        """
        Status: OK
        Expression: DS_r <- intersect ( DS_3 , DS_4 );
        Description: Extra of case 5: the String Id_1 of DS_3 and the Boolean Id_1 of DS_4 are
                     matched as Strings, so "true" is in both. The pandas engine matched nothing
                     and the DuckDB engine raised a raw ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5e", number_inputs=2, references_names=["1"])

    def test_GH_1353_5f(self):
        """
        Status: OK
        Expression: DS_r <- DS_4 + DS_3;
        Description: Extra of case 5: with the Boolean operand first, Id_1 is a String in the
                     result too and only "true" matches. The pandas engine raised a raw
                     ArrowInvalid, the DuckDB engine a raw ConversionException, and both typed
                     Id_1 as a Boolean.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5f", number_inputs=2, references_names=["1"])

    def test_GH_1353_5g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Me_1 in { true } ];
        Description: Extra of case 5: the String Me_1 is compared with the set of Booleans as
                     Strings, so only "true" is in it. The pandas engine gave false for "true"
                     and the DuckDB engine true.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5g", number_inputs=1, references_names=["1"])

    def test_GH_1353_5h(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 in { true };
        Description: Extra of case 5: the String Measure of DS_1 is compared with the set of
                     Booleans as Strings. The pandas engine gave false for "true" and the DuckDB
                     engine raised a raw ConversionException on "abc".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_5h", number_inputs=1, references_names=["1"])

    def test_GH_1353_6(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_3 as A , DS_4 as B keep A#Me_1 );
        Description: The join key Id_1, a String in DS_3 and a Boolean in DS_4, is a String and
                     only "true" matches the Boolean true. Both engines and the semantic analysis
                     typed Id_1 as a Boolean, and matched "1" too.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_6", number_inputs=2, references_names=["1"])

    def test_GH_1353_6b(self):
        """
        Status: OK
        Expression: DS_r <- full_join ( DS_4 , DS_3 );
        Description: Extra of case 6: the join key Id_1, a Boolean in DS_4 and a String in DS_3,
                     is a String, and the Boolean true is the String "true". The pandas engine
                     wrote it as "True" and the DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_6b", number_inputs=2, references_names=["1"])

    def test_GH_1353_7(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_1 , DS_3 );
        Description: The Integer Id_1 of DS_1 and the String Id_1 of DS_3 have no common type,
                     so exists_in raises 1-1-13-18, as the binary operators do.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1353_7",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Integer and String.",
        )

    def test_GH_1353_8(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_3 );
        Description: The Integer Id_1 of DS_1 and the String Id_1 of DS_3 have no common type,
                     so nvl raises 1-1-13-18.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1353_8",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Integer and String.",
        )

    def test_GH_1353_9(self):
        """
        Status: OK
        Expression: DS_r <- full_join ( DS_1 as A , DS_3 as B keep A#Me_1 );
        Description: The join key Id_1, an Integer in DS_1 and a String in DS_3, has no common
                     type, so full_join raises 1-1-13-18, as inner_join and left_join do.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1353_9",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="Incompatible types for common identifier Id_1: Integer and String.",
        )

    def test_GH_1353_10(self):
        """
        Status: OK
        Expression: DS_r <- symdiff ( DS_1 , DS_5 );
        Description: The Number Me_1 of DS_1 and the Integer Me_1 of DS_5 give a Number Me_1,
                     and the Integer values of DS_5 are cast to Number. The pandas engine wrote
                     the Number values into the Integer column and raised a raw ArrowInvalid.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_10", number_inputs=2, references_names=["1"])

    def test_GH_1353_11(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := substr ( Me_1 , 1 , Me_3 ) ];
        Description: The Number length Me_3 (2.0) is implicitly cast to Integer. The DuckDB
                     engine passed the DOUBLE to SUBSTR and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_11", number_inputs=1, references_names=["1"])

    def test_GH_1353_12(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := instr ( Me_1 , "b" , 1 , Me_3 ) ];
        Description: The Number occurrence Me_3 (2.0) is implicitly cast to Integer. The pandas
                     engine raised 2-1-18-1, naming it the start parameter, and the DuckDB engine
                     raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_12", number_inputs=1, references_names=["1"])

    def test_GH_1353_13(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := instr ( "abc" , Me_2 ) ];
        Description: instr of a String literal with a Component pattern gives a value per data
                     point. The pandas engine raised a raw AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_13", number_inputs=1, references_names=["1"])

    def test_GH_1353_14(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := replace ( Me_2 , "b" , Me_2 ) ];
        Description: The same Component is the operand and the replacement of replace. The
                     pandas engine raised a raw ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_14", number_inputs=1, references_names=["1"])

    def test_GH_1353_15(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := lag ( Me_3 , 1 , 0.5 over ( order by Id_1 ) ) ];
        Description: The Number default 0.5 of lag on the Integer Me_3 gives a Number Me_4, as
                     nvl ( Me_3 , 0.5 ) does, and the default is not rounded. The pandas engine
                     rounded it to 1 and the DuckDB engine to 0, and both typed Me_4 as an
                     Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_15", number_inputs=1, references_names=["1"])

    def test_GH_1353_16(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := lag ( Me_1 , 1 , "x" over ( order by Id_1 ) ) ];
        Description: The String default "x" of lag on the Number Me_1 has no common type with
                     it, so lag raises 1-1-1-1, as nvl ( Me_1 , "x" ) does. The semantic analysis
                     accepted it, the pandas engine raised a raw ConversionException and the
                     DuckDB engine 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1353_16",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Number to String.",
        )

    def test_GH_1353_17(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( SC_1 * 2 , string ) ];
        Description: The whole Number 3.0 computed from the Number Scalar SC_1 (1.5) is a float,
                     so its cast to string is "3.0" on both engines. The pandas engine held a
                     Python int and wrote "3".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1353_17", number_inputs=1, references_names=["1"], scalars={"SC_1": 1.5}
        )

    def test_GH_1353_18(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := cast ( Me_1 * 2 , string ) ];
        Description: Control of case 17: the whole Number 3.0 computed from a Component is
                     written "3.0" on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_18", number_inputs=1, references_names=["1"])

    def test_GH_1353_19(self):
        """
        Status: OK
        Expression: DS_r <- union ( DS_8 , DS_9 );
        Description: The Boolean Me_1 of DS_8 promoted to the String of DS_9 is "true" or
                     "false". The pandas engine wrote "True" and "False".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_19", number_inputs=2, references_names=["1"])

    def test_GH_1353_20(self):
        """
        Status: OK
        Expression: DS_r <- DS_8 [ unpivot Id_2 , Me_9 ];
        Description: The Boolean Me_1 unpivoted with the String Me_2 into the String Me_9 is
                     "true" or "false". The pandas engine wrote "True" and "False".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_20", number_inputs=1, references_names=["1"])

    def test_GH_1353_22(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := lag ( Me_2 , 1 , null over ( order by Id_1 ) ) ];
        Description: The null default of lag gives null. The pandas engine wrote it into its SQL
                     as the column name None and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_22", number_inputs=1, references_names=["1"])

    def test_GH_1353_23(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := lag ( Me_2 , 1 , "x" over ( order by Id_1 ) ) ];
        Description: The String default "x" of lag is the text x. The pandas engine wrote it
                     into its SQL as a column name and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_23", number_inputs=1, references_names=["1"])

    def test_GH_1353_24(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := lead ( Me_2 , 1 , "Me_2" over ( order by Id_1 ) ) ];
        Description: The String default "Me_2" of lead is the text Me_2, not the value of the
                     Component Me_2. The pandas engine gave the value of Me_2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_24", number_inputs=1, references_names=["1"])

    def test_GH_1353_25(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := SC_2 ];
        Description: The Number Scalar SC_2 with the value inf is inf. The DuckDB engine wrote it
                     into its SQL as the column name inf and raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1353_25",
            number_inputs=1,
            references_names=["1"],
            scalars={"SC_2": float("inf")},
        )

    def test_GH_1353_26(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := cast ( SC_2 , string ) ];
        Description: The Number Scalar SC_2 with the value -0.0 keeps its sign: its cast to
                     string is "-0.0". The DuckDB engine wrote it into its SQL as 0.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1353_26", number_inputs=1, references_names=["1"], scalars={"SC_2": -0.0}
        )

    def test_GH_1353_28(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := 1 not_in { null } ];
        Description: not_in is the negation of in, so 1 not_in { null } is null, as
                     1 in { null }, and Me_4 is nullable. The pandas engine gave true, and both
                     engines declared Me_4 non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_28", number_inputs=1, references_names=["1"])

    def test_GH_1353_29(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := Me_1 in { 0.0 } ];
        Description: -0.0 is 0.0, so it is in { 0.0 }, as Me_1 = 0.0 gives true. The pandas
                     engine told -0.0 from 0.0 and gave false.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_29", number_inputs=1, references_names=["1"])

    def test_GH_1353_30(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := Me_1 = 0.0 ];
        Description: Control of case 29: -0.0 = 0.0 is true on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_30", number_inputs=1, references_names=["1"])

    def test_GH_1353_31(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := replace ( Me_1 , "" , "-" ) ];
        Description: An empty pattern matches nothing, so replace leaves the value unchanged.
                     The pandas engine inserted the replacement between the characters.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_31", number_inputs=1, references_names=["1"])

    def test_GH_1353_32(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := instr ( Me_1 , "" , 1 , 2 ) ,
                                         Me_7 := instr ( "" , "" , 1 , 2 ) ];
        Description: The second occurrence of an empty pattern is at position 2 of "abcb" and
                     there is none in an empty string, which gives 0. The DuckDB engine gave
                     the position 2, past the end of the empty string.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_32", number_inputs=1, references_names=["1"])

    def test_GH_1353_33(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := match_characters ( Me_1 , "[[:alpha:]]+" ) ];
        Description: The pattern is a POSIX Extended Regular Expression, so [[:alpha:]] is a
                     letter on both engines. The pandas engine read it as a set of the
                     characters [, :, a, l, p and h.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_33", number_inputs=1, references_names=["1"])

    def test_GH_1353_35(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := string_distance ( levenshtein , Me_4 , "ab" ) ];
        Description: The Levenshtein distance counts characters, so "áb" is one edit from "ab".
                     The DuckDB engine counted UTF-8 bytes and gave 2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_35", number_inputs=1, references_names=["1"])

    def test_GH_1353_36(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := string_distance ( hamming , Me_4 , "ab" ) ];
        Description: The Hamming distance compares characters, so "áb" and "ab" have the same
                     length and differ in one position. The DuckDB engine counted UTF-8 bytes
                     and raised a raw InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_36", number_inputs=1, references_names=["1"])

    def test_GH_1353_37(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := string_distance ( hamming , "" , "" ) ];
        Description: The Hamming distance of two empty strings is 0. The DuckDB engine raised a
                     raw InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_37", number_inputs=1, references_names=["1"])

    def test_GH_1353_40(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_6 := lag ( Me_2 , 1 , cast ( 1 , string ) over
                                                      ( order by Id_1 ) ) ];
        Description: The default cast ( 1 , string ) of lag is the text 1. The pandas engine
                     wrote it into its SQL in its Python form and raised a raw ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_40", number_inputs=1, references_names=["1"])

    def test_GH_1353_41(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := cast ( f ( 0.1 ) + f ( 0.2 ) , string ) ];
        Description: The sum of the Number results of the user defined operator f, which
                     returns its parameter, is rounded to 15 significant digits, as 0.1 + 0.2,
                     so it is "0.3" on both engines. The DuckDB engine gave
                     0.30000000000000004; both engines already give "0.3" on the code of this
                     fix.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_41", number_inputs=1, references_names=["1"])

    def test_GH_1353_42(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := ln ( 0 - SC_1 ) ];
        Description: The Number 0 - SC_1 (SC_1 2.0) is -2.0, so both engines raise 2-1-15-8
                     with the value -2.0 in the message. The pandas engine wrote -2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1353_42",
            number_inputs=1,
            exception_code="2-1-15-8",
            exception_message="Value -2.0 must be greater than 0",
            scalars={"SC_1": 2.0},
        )

    def test_GH_1353_43(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ calc Me_4 := cast ( g ( 0 ) * -1 , string ) ];
        Description: The Number result 0 of the user defined operator g times -1 is 0.0, written
                     "0.0" on both engines. The pandas engine held a Python int and wrote "0",
                     and the DuckDB engine wrote "-0.0".
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_43", number_inputs=1, references_names=["1"])

    def test_GH_1353_15b(self):
        """
        Status: OK
        Expression: DS_r <- lag ( DS_1 [ keep T ] , 1 , "2023Q4" over ( order by Id_1 ) );
        Description: The String default "2023Q4" of lag on the Time_Period T is the period
                     2023Q4, given to the first Data Point on both engines. The DuckDB engine
                     gave it a null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1353
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1353_15b", number_inputs=1, references_names=["1"])


class SemanticCheckBugs(BugHelper):
    """Invalid scripts refused by the semantic analysis on both engines (issue #1356)."""

    def test_GH_1356_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := between ( Me_3 , 1 , 10 ) ];
        Description: The String Me_3 and the Integer bounds of between have no common type, so
                     between raises 1-1-1-1, as Me_3 > 1 does. Both engines raised a raw error and
                     the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_4",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from String to Integer.",
        )

    def test_GH_1356_5(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := between ( Me_2 , true , 10 ) ];
        Description: The Number Me_2 and the Boolean bound true have no common type, so between
                     raises 1-1-1-1, as Me_2 > true does. The pandas engine returned a result and
                     the DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_5",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Number to Boolean.",
        )

    def test_GH_1356_6(self):
        """
        Status: OK
        Expression: SC_r <- between ( "a" , 1 , 10 );
        Description: The String "a" and the Integer bounds have no common type, so between raises
                     1-1-1-1. Both engines raised a raw error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_6",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from String to Integer.",
        )

    def test_GH_1356_7(self):
        """
        Status: OK
        Expression: DS_r <- between ( DS_2 , DS_2 , 10 );
        Description: A bound of between is a Component or a Scalar, so the Data Set DS_2 as a bound
                     raises 1-1-15-8, as round ( DS_2 , DS_2 ) does. Both engines and the semantic
                     analysis raised a raw AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_7",
            number_inputs=1,
            exception_code="1-1-15-8",
            exception_message="At op between: between operator cannot have a Dataset as parameter.",
        )

    def test_GH_1356_8(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_9 := sum ( Me_2 ) group by Id_1 having between ( max (
                    Me_2 ) , 0 , sum ( Me_2 ) ) ];
        Description: In a having clause the aggregates sum ( Me_2 ) and max ( Me_2 ) are valid
                     bounds of between, which keeps the groups as max ( Me_2 ) >= 0 and max ( Me_2 )
                     <= sum ( Me_2 ) does (case 8b). Both engines and the semantic analysis raised a
                     raw AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_8", number_inputs=1, references_names=["1"])

    def test_GH_1356_8b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_9 := sum ( Me_2 ) group by Id_1 having max ( Me_2 ) >= 0
                    and max ( Me_2 ) <= sum ( Me_2 ) ];
        Description: Control of case 8: the same condition written with >= and <=.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_8b", number_inputs=1, references_names=["1"])

    def test_GH_1356_9(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( order by Me_4 range between 1
                    preceding and current data point ) ) ];
        Description: A range window ordered by the Boolean Me_4 raises 1-1-19-13, as for a String
                     order. Both engines raised a raw BinderException and the semantic analysis
                     accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_9",
            number_inputs=1,
            exception_code="1-1-19-13",
            exception_message="At op sum: RANGE window is not supported for Boolean data "
            "type in component Me_4.",
        )

    def test_GH_1356_10(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( order by Id_1 , Me_1 range between
                    1 preceding and current data point ) ) ];
        Description: A range window with a limit of 1 measures the distance on one order by
                     Component, so two raise 1-1-3-3. Both engines raised a raw BinderException and
                     the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_10",
            number_inputs=1,
            exception_code="1-1-3-3",
            exception_message="At op sum: A range window with a limit other than unbounded "
            "or current data point needs one order by Component, found 2.",
        )

    def test_GH_1356_10b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( partition by Id_1 range between 1
                    preceding and current data point ) ) ];
        Description: A range window with a limit of 1 and no order by raises 1-1-3-3. Both engines
                     raised a raw BinderException and the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_10b",
            number_inputs=1,
            exception_code="1-1-3-3",
            exception_message="At op sum: A range window with a limit other than unbounded "
            "or current data point needs one order by Component, found 0.",
        )

    def test_GH_1356_11(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( partition by Id_1 order by Id_2
                    data points between 1 following and 1 preceding ) ) ];
        Description: A window that starts after it ends raises 1-1-3-4. Both engines raised a raw
                     ParserException and the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_11",
            number_inputs=1,
            exception_code="1-1-3-4",
            exception_message="Invalid analytic window 'data points between 1 following and "
            "1 preceding' (line 1, column 79): the window starts after it "
            "ends.",
        )

    def test_GH_1356_12(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( partition by Id_1 order by Id_2
                    data points between SC_1 preceding and current data point ) ) ];
        Description: The Scalar SC_1 that gives a limit of the window is null, so the window has no
                     limit and raises 2-1-18-1. Both engines raised a raw error.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_12",
            number_inputs=1,
            exception_code="2-1-18-1",
            exception_message="At op sum: Window bound parameter should be a non-negative Integer.",
            scalars={"SC_1": None},
        )

    def test_GH_1356_13(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( partition by Id_1 order by Id_2
                    data points between SC_1 preceding and current data point ) ) ];
        Description: The Scalar SC_1 = 0 as a limit is the current data point, as the literal 0
                     (case 13b). The pandas engine raised a raw ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1356_13",
            number_inputs=1,
            references_names=["1"],
            scalars={"SC_1": 0},
        )

    def test_GH_1356_13b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( partition by Id_1 order by Id_2
                    data points between 0 preceding and current data point ) ) ];
        Description: Control of case 13: the literal limit 0 is the current data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_13b", number_inputs=1, references_names=["1"])

    def test_GH_1356_14(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sum ( Me_1 over ( partition by Id_1 order by Id_2
                    data points between -2 preceding and current data point ) ) ];
        Description: A negative literal limit raises 1-1-3-5. The pandas engine returned null values
                     and the DuckDB engine took the limit as unbounded.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_14",
            number_inputs=1,
            exception_code="1-1-3-5",
            exception_message="Invalid limit '-2 preceding' (line 1, column 99) of an "
            "analytic window: a limit cannot be negative.",
        )

    def test_GH_1356_15(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := yeartoday ( Me_1 ) ];
        Description: yeartoday takes the text of a duration PnYnD, a String, so the Integer Me_1
                     raises 1-1-1-1, as length ( Me_1 ) does. Both engines raised a raw error and
                     the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_15",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Integer to String.",
        )

    def test_GH_1356_16(self):
        """
        Status: OK
        Expression: SC_r <- daytoyear ( 400.5 );
        Description: daytoyear takes an Integer, so the Number 400.5 raises 1-1-1-1. Each engine
                     returned a different result.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_16",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Number to Integer.",
        )

    def test_GH_1356_17(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := getyear ( Me_5 ) ];
        Description: getyear takes a time value, so the Duration Me_5 raises 1-1-19-10, as getyear (
                     Me_3 ) does. Both engines raised a raw error and the semantic analysis accepted
                     it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_17",
            number_inputs=1,
            exception_code="1-1-19-10",
            exception_message="getyear can only be applied to operands with data type as "
            "Date or Time Period",
        )

    def test_GH_1356_18(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is R1 : true errorcode "E1" end
                    datapoint ruleset; DS_r <- check_datapoint ( DS_2 , DPR_1 invalid );
        Description: A rule that reads no Component holds its constant for every data point: true,
                     so no data point is invalid. Both engines and the semantic analysis raised a
                     raw TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_18", number_inputs=1, references_names=["1"])

    def test_GH_1356_19(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is R1 : when true then Me_1 > 1
                    end datapoint ruleset; DS_r <- check_datapoint ( DS_2 , DPR_1 all );
        Description: A when condition that reads no Component applies the rule to every data point.
                     The pandas engine raised a raw AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_19", number_inputs=1, references_names=["1"])

    def test_GH_1356_20(self):
        """
        Status: OK
        Expression: define hierarchical ruleset HR_1 ( variable rule Id_2 ) is R1 : when true then A
                    = B + C end hierarchical ruleset; DS_r <- check_hierarchy ( DS_3 , HR_1 rule
                    Id_2 all );
        Description: A when condition that reads no Component applies the hierarchical rule to every
                     data point. The pandas engine raised a raw AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_20", number_inputs=1, references_names=["1"])

    def test_GH_1356_21(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is R1 : when Me_1 > 0 then Me_9
                    > 0 end datapoint ruleset; DS_r <- check_datapoint ( DS_2 , DPR_1 all );
        Description: The then part of the rule reads Me_9, which is not in the signature: 1-1-10-7
                     on both engines and in the semantic analysis. The DuckDB engine raised a raw
                     BinderException and the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_21",
            number_inputs=1,
            exception_code="1-1-10-7",
            exception_message="Not found component Me_9 on signature.",
        )

    def test_GH_1356_22(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is R1 : when Me_1 > 100 then
                    Me_9 > 0 end datapoint ruleset; DS_r <- check_datapoint ( DS_2 , DPR_1 all );
        Description: The then part of the rule is checked also when no data point meets the
                     condition: 1-1-10-7. The pandas engine returned a result and the DuckDB engine
                     raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_22",
            number_inputs=1,
            exception_code="1-1-10-7",
            exception_message="Not found component Me_9 on signature.",
        )

    def test_GH_1356_23(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is R1 : when Me_1 > 0 then Me_1
                    > "a" end datapoint ruleset; DS_r <- check_datapoint ( DS_2 , DPR_1 all );
        Description: The then part compares the Number Me_1 with a String: 1-1-1-1 on both engines
                     and in the semantic analysis. The DuckDB engine raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_23",
            number_inputs=1,
            exception_code="1-1-1-1",
            exception_message="Invalid implicit cast from Number to String.",
        )

    def test_GH_1356_24(self):
        """
        Status: OK
        Expression: define datapoint ruleset DPR_1 ( variable Me_1 ) is R1 : when Me_1 > 0 then f (
                    Me_1 ) > 0 end datapoint ruleset; DS_r <- check_datapoint ( DS_2 , DPR_1 all );
        Description: The then part calls the operator f, which is not defined: 1-2-22 on both
                     engines and in the semantic analysis. The DuckDB engine raised a raw KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_24",
            number_inputs=1,
            exception_code="1-2-22",
            exception_message="No User Defined Operators have been defined.",
        )

    def test_GH_1356_25(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_9 := sum ( Me_1 ) , Me_9 := max ( Me_1 ) group by Id_1 ];
        Description: aggr computes Me_9 twice: 1-1-6-9, as keep Me_1 , Me_1 raises. The pandas
                     engine raised a raw KeyError and the DuckDB engine kept the last one.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_25",
            number_inputs=1,
            exception_code="1-1-6-9",
            exception_message="At op aggr: The following components are repeated: {'Me_9'}.",
        )

    def test_GH_1356_26(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := Me_1 , Me_9 := Me_1 * 10 ];
        Description: calc computes Me_9 twice: 1-1-6-9. Both engines kept the last one.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_26",
            number_inputs=1,
            exception_code="1-1-6-9",
            exception_message="At op calc: The following components are repeated: {'Me_9'}.",
        )

    def test_GH_1356_27(self):
        """
        Status: OK
        Expression: DS_r <- union ( sum ( DS_2 ) , sum ( DS_2 * 10 ) );
        Description: union refuses a Data Set without Identifiers with 1-2-10, as the joins do (case
                     29b). The pandas engine raised a raw ValueError and the DuckDB engine returned
                     both data points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_27",
            number_inputs=1,
            exception_code="1-2-10",
            exception_message="Dataset without Identifiers are not allowed in union operator.",
        )

    def test_GH_1356_28(self):
        """
        Status: OK
        Expression: DS_r <- intersect ( sum ( DS_2 ) , sum ( DS_2 * 10 ) );
        Description: intersect refuses a Data Set without Identifiers with 1-2-10. The pandas engine
                     raised a raw IndexError and the DuckDB engine returned a data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_28",
            number_inputs=1,
            exception_code="1-2-10",
            exception_message="Dataset without Identifiers are not allowed in intersect operator.",
        )

    def test_GH_1356_29(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( sum ( DS_2 ) , sum ( DS_2 * 10 ) );
        Description: exists_in refuses a Data Set without Identifiers with 1-2-10. The pandas engine
                     raised a raw IndexError and the DuckDB engine a raw ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_29",
            number_inputs=1,
            exception_code="1-2-10",
            exception_message="Dataset without Identifiers are not allowed in exists_in operator.",
        )

    def test_GH_1356_29b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( sum ( DS_2 ) as a , sum ( DS_2 * 10 ) as b );
        Description: Control of cases 27 to 29: the joins refuse a Data Set without Identifiers.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_29b",
            number_inputs=1,
            exception_code="1-2-10",
            exception_message="Dataset without Identifiers are not allowed in inner_join operator.",
        )

    def test_GH_1356_30(self):
        """
        Status: OK
        Expression: DS_r <- sum ( DS_2 ) [ calc Me_2 := rank ( over ( order by Me_1 ) ) ];
        Description: An analytic function on a Data Set without Identifiers ranks its data point.
                     The pandas engine raised a raw ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_30", number_inputs=1, references_names=["1"])

    def test_GH_1356_55(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := match_characters ( Me_3 , "[" ) ];
        Description: The pattern "[" is not a regular expression: 1-1-18-8. The pandas engine raised
                     a raw re.error, the DuckDB engine a raw InvalidInputException and the semantic
                     analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_55",
            number_inputs=1,
            exception_code="1-1-18-8",
            exception_message="At op match_characters: unterminated character set in "
            "regexp: [,  in position 0.",
        )

    def test_GH_1356_55b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := match_characters ( Me_3 , "[[:alpha:]" ) ];
        Description: The pattern "[[:alpha:]" is checked as it is evaluated, with its POSIX class
                     read: its bracket expression is not closed, 1-1-18-8. The semantic analysis
                     accepted it, the pandas engine raised a raw re.error and the DuckDB engine a
                     raw InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_55b",
            number_inputs=1,
            exception_code="1-1-18-8",
            exception_message="At op match_characters: unterminated character set in "
            "regexp: [[:alpha:],  in position 0.",
        )

    def test_GH_1356_55c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := match_characters ( Me_3 , "[[:foo:]]" ) ];
        Description: [:foo:] is not a POSIX character class: 1-1-18-8. The semantic analysis
                     accepted it, the pandas engine gave false for every data point and the DuckDB
                     engine raised a raw InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_55c",
            number_inputs=1,
            exception_code="1-1-18-8",
            exception_message="At op match_characters: unknown POSIX character class [:foo:] "
            "in regexp: [[:foo:]],  in position 1.",
        )

    def test_GH_1356_56(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset , x component < integer > ) returns dataset is ds
                    [ calc Me_9 := x ] end operator; DS_r <- f ( DS_1 , Me_3 );
        Description: The String Component Me_3 given to the parameter x component < integer > raises
                     1-2-18, as the String "a" given to an integer parameter (case 56b). Both
                     engines took it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_56",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected Integer, got String on UDO f, parameter x.",
        )

    def test_GH_1356_56b(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset , x integer ) returns dataset is ds [ calc Me_9
                    := x ] end operator; DS_r <- f ( DS_1 , "a" );
        Description: Control of case 56: the String "a" given to an integer parameter raises 1-2-18.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_56b",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected Integer, got String on UDO f, parameter x.",
        )

    def test_GH_1356_57(self):
        """
        Status: OK
        Expression: define operator f ( x component < integer > ) returns component is x end
                    operator; DS_r <- DS_1 [ calc Me_9 := f ( Me_3 ) ];
        Description: The String Component Me_3 given to the parameter x component < integer > in a
                     calc raises 1-2-18. Both engines took it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_57",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected Integer, got String on UDO f, parameter x.",
        )

    def test_GH_1356_58(self):
        """
        Status: OK
        Expression: define operator f ( ds dataset , x component < VD_1 > ) returns dataset is ds [
                    calc Me_9 := x ] end operator; DS_r <- f ( DS_1 , Me_3 );
        Description: The String Component Me_3 given to a parameter on the Integer value domain VD_1
                     raises 1-2-18. The pandas engine raised 2-1-5-4 and the DuckDB engine a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_58",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected Integer, got String on UDO f, parameter x.",
            vd_names=["GH_1356-VD_1"],
        )

    def test_GH_1356_59(self):
        """
        Status: OK
        Expression: define operator f ( x integer default 2.5 ) returns integer is x end operator;
                    SC_r <- f ( 1 );
        Description: The default 2.5 of the integer parameter x is not an Integer: 1-3-1-1 where the
                     operator is defined, whether the default is used or not. Both engines accepted
                     it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_59",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined default of parameter x declared as "
            "Integer, found Number.",
        )

    def test_GH_1356_60(self):
        """
        Status: OK
        Expression: define operator f ( x integer default 2.5 ) returns integer is x end operator;
                    SC_r <- f ( _ );
        Description: The default 2.5 of the integer parameter x raises 1-3-1-1 where the operator is
                     defined, not 1-2-18 at the call.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_60",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined default of parameter x declared as "
            "Integer, found Number.",
        )

    def test_GH_1356_60b(self):
        """
        Status: OK
        Expression: define operator f ( x integer default "a" ) returns integer is x end operator;
                    SC_r <- f ( 1 );
        Description: Control of cases 59 and 60: the default "a" of an integer parameter raises
                     1-3-1-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_60b",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined default of parameter x declared as "
            "Integer, found String.",
        )

    def test_GH_1356_61(self):
        """
        Status: OK
        Expression: define operator f ( x VD_1 default 5 ) returns integer is x end operator; SC_r
                    <- f ( 1 );
        Description: The default 5 is not a value of the value domain VD_1 of the parameter x:
                     1-1-10-13 where the operator is defined, whether the default is used or not.
                     Both engines accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_61",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message="The default 5 of parameter x of the operator f is not one of "
            "the values of the Value Domain VD_1.",
            vd_names=["GH_1356-VD_1"],
        )

    def test_GH_1356_62(self):
        """
        Status: OK
        Expression: define operator f ( x VD_1 default 5 ) returns integer is x end operator; SC_r
                    <- f ( _ );
        Description: The default 5 outside VD_1 raises 1-1-10-13 where the operator is defined. Both
                     engines raised 2-1-5-4 at run time and the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_62",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message="The default 5 of parameter x of the operator f is not one of "
            "the values of the Value Domain VD_1.",
            vd_names=["GH_1356-VD_1"],
        )

    def test_GH_1356_63(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( valuedomain VD_DUR ) is when "A" then "XYZ" ;
                    else "A" end viral propagation; DS_r <- DS_7 + DS_7;
        Description: The constant "XYZ" of the rule is not a Duration, so not a value of VD_DUR:
                     1-1-10-13, an error of the script. Both engines and the semantic analysis
                     raised the input error 0-1-1-35.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_63",
            number_inputs=1,
            exception_code="1-1-10-13",
            exception_message='The constant "XYZ" of viral propagation rule VP_1 is not one '
            "of the values of the Value Domain VD_DUR.",
            vd_names=["GH_1356-VD_DUR"],
        )

    def test_GH_1356_64(self):
        """
        Status: OK
        Expression: DS_r <- DS_2 [ rename Me_1 to 'x"y' ];
        Description: A Component name cannot hold a double quote, as the input structures refuse
                     it, so the new name x"y raises 1-1-6-19 on both engines and in the semantic
                     analysis. The pandas engine returned the Component x"y, the DuckDB engine
                     raised a raw ParserException and the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_64",
            number_inputs=1,
            exception_code="1-1-6-19",
            exception_message='At op rename: Invalid Component name x"y',
        )


class RunTimeErrorMappingBugs(BugHelper):
    """Errors raised while the data is computed are VTL errors, the same on both engines (issue
    #1356)."""

    def test_GH_1356_31(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := Me_1 * Me_1 ];
        Description: The product 10^24 of the Integer 10^12 by itself does not fit in an Integer
                     (64 bits): both engines raise 2-1-5-1. The pandas engine raised a raw
                     ArrowInvalid and the DuckDB engine a raw OutOfRangeException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_31",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message="Impossible to cast 1000000000000000000000000 from type Number "
            "to Integer.",
        )

    def test_GH_1356_32(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := ceil ( Me_2 ) ];
        Description: ceil of the Number 1e20 does not fit in an Integer: both engines raise
                     2-1-5-1. The pandas engine raised a raw ArrowInvalid and the DuckDB engine a
                     raw ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_32",
            number_inputs=1,
            exception_code="2-1-5-1",
            exception_message="Impossible to cast 1e+20 from type Number to Integer.",
        )

    def test_GH_1356_33(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := cast ( Me_2 , integer ) ];
        Description: The cast of the Number Me_2 to integer is refused by the semantic analysis on
                     both engines, as the VTL 2.2 manual marks number to integer as not feasible
                     (issue #1388). It raised 2-1-5-1 for the value 1e20, which does not fit in an
                     Integer, and dropped the decimals of the other values.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_33",
            number_inputs=1,
            exception_code="1-1-5-4",
            exception_message="Invalid mask to cast from type Number to Integer.",
        )

    def test_GH_1356_34(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := power ( Me_2 , 20 ) ];
        Description: 1e20 to the power of 20 is beyond the range of a Number: both engines raise
                     2-1-3-1, as for the infinite values of a division. The pandas engine raised
                     a raw OverflowError and the DuckDB engine returned inf.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GH_1356_34", number_inputs=1, exception_code="2-1-3-1")

    def test_GH_1356_34b(self):
        """
        Status: OK
        Expression: SC_r <- exp ( 1000 );
        Description: exp(1000) is beyond the range of a Number: both engines raise 2-1-3-1. The
                     pandas engine raised a raw OverflowError and the DuckDB engine returned inf.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GH_1356_34b", number_inputs=1, exception_code="2-1-3-1")

    def test_GH_1356_35(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := round ( Me_2 , -400 ) ];
        Description: Rounded to the 400th digit before the decimal point, 2.5 and 1e20 are 0.0
                     on both engines. The pandas engine raised a raw ZeroDivisionError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_35", number_inputs=1, references_names=["1"])

    def test_GH_1356_36(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := round ( Me_2 , null ) ];
        Description: round with a null number of digits rounds to a whole Number on both
                     engines: 3.0 and 1e20. The pandas engine raised a raw OverflowError for
                     1e20.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_36", number_inputs=1, references_names=["1"])

    def test_GH_1356_37(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := trunc ( Me_2 , 400 ) ];
        Description: Truncated at the 400th decimal, 2.5 and 1e20 are unchanged on both engines.
                     The pandas engine raised a raw OverflowError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1356_37", number_inputs=1, references_names=["1"])

    def test_GH_1356_38(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := log ( Me_2 , 1 ) ];
        Description: A logarithm in base 1 does not exist: both engines raise 2-1-15-3 from the
                     check of the base. The pandas engine raised a raw ZeroDivisionError and the
                     DuckDB engine a raw OutOfRangeException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_38",
            number_inputs=1,
            exception_code="2-1-15-3",
            exception_message="At op log: Base value 1",
        )

    def test_GH_1356_38b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := log ( Me_1 , Me_2 ) ];
        Description: Case 38 with the base in a Component: the base 1 of the second data point
                     raises 2-1-15-3 on both engines. The pandas engine raised a raw
                     ZeroDivisionError and the DuckDB engine a raw OutOfRangeException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_38b",
            number_inputs=1,
            exception_code="2-1-15-3",
            exception_message="At op log: Base value 1",
        )

    def test_GH_1356_65(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := Me_2 / Me_1 ];
        Description: A divisor 0 raises 2-1-15-6 on both engines, also when the dividend is
                     null. The pandas engine returned null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GH_1356_65", number_inputs=1, exception_code="2-1-15-6")

    def test_GH_1356_66(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := log ( Me_2 , Me_1 ) ];
        Description: A base 0 raises 2-1-15-3 on both engines, also when the operand is null.
                     The pandas engine returned null.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GH_1356_66", number_inputs=1, exception_code="2-1-15-3")

    def test_GH_1356_67(self):
        """
        Status: OK
        Expression: DS_r <- DS_6 [ pivot Id_2 , Me_1 ];
        Description: The Measures of pivot are the values of VD_X, the value domain of Id_2, and
                     its value a"b holds a double quote, which a Component name cannot hold, so
                     pivot raises 1-1-6-19 on both engines and in the semantic analysis, as
                     test_GH_1356_64. The pandas engine returned the Measure a"b, the DuckDB
                     engine raised a raw ParserException and the semantic analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_67",
            number_inputs=1,
            exception_code="1-1-6-19",
            exception_message='At op pivot: Invalid Component name a"b',
            vd_names=["GH_1356-VD_X"],
        )

    def test_GH_1356_43(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := sqrt ( -1 ) + length ( substr ( Me_3 , 0 ) ) ];
        Description: The semantic error of substr (start 0) is raised before the data error of
                     sqrt(-1), on both engines and in the semantic analysis. The pandas engine
                     raised 2-1-15-2 for sqrt first.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1356
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1356_43",
            number_inputs=1,
            exception_code="1-1-18-4",
            exception_message="At op substr: Start parameter should be >= 1.",
        )


class ValueDomainFollowUpBugs(BugHelper):
    """Value domains, returns types, viral Attributes and error messages that results and checks
    keep (issue #1364)."""

    vd_1364 = ["GH_1364-VD_INT", "GH_1364-VD_NUM", "GH_1364-VD_STR", "GH_1364-VD_STR2"]

    def test_GH_1364_1(self):
        """
        Status: OK
        Expression: DS_r <- max ( DS_1 group by Id_1 );
        Description: The aggregate max of Me_1, on the value domain VD_INT, is one of its values,
                     so Me_1 keeps VD_INT. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_1", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_2(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := min ( Me_1 ) group by Id_1 ];
        Description: The aggregate min of Me_1, on VD_INT, in an aggr gives Me_2 on VD_INT. It was
                     an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_2", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_3(self):
        """
        Status: OK
        Expression: DS_r <- max ( DS_1 over ( partition by Id_1 ) );
        Description: The analytic max of Me_1, on VD_INT, keeps VD_INT. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_3", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_4(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := min ( Me_1 over ( partition by Id_1 ) ) ];
        Description: The analytic min of Me_1, on VD_INT, in a calc gives Me_2 on VD_INT. It was
                     an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_4", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_5(self):
        """
        Status: OK
        Expression: DS_r <- first_value ( DS_1 over ( partition by Id_1 order by Id_2 ) );
        Description: first_value of Me_1, on VD_INT, keeps VD_INT. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_5", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_6(self):
        """
        Status: OK
        Expression: DS_r <- last_value ( DS_1 over ( partition by Id_1 order by Id_2 ) );
        Description: last_value of Me_1, on VD_INT, keeps VD_INT. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_6", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_7(self):
        """
        Status: OK
        Expression: DS_r <- lag ( DS_1 , 1 over ( partition by Id_1 order by Id_2 ) );
        Description: lag of Me_1, on VD_INT, without a default keeps VD_INT, with nulls at the
                     edges of the partitions. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_7", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_8(self):
        """
        Status: OK
        Expression: DS_r <- lead ( DS_1 , 1 over ( partition by Id_1 order by Id_2 ) );
        Description: lead of Me_1, on VD_INT, without a default keeps VD_INT. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_8", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_9(self):
        """
        Status: OK
        Expression: DS_r <- lag ( DS_1 , 1 , 9 over ( partition by Id_1 order by Id_2 ) );
        Description: Control: the default 9 is a literal of the basic type, so lag of Me_1, on
                     VD_INT, gives an Integer, as nvl ( DS_1 , 9 ) does.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_9", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_10(self):
        """
        Status: OK
        Expression: DS_r <- sum ( DS_1 group by Id_1 );
        Description: Control: sum computes a new value, so the sum of Me_1, on VD_INT, is an
                     Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_10", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_11(self):
        """
        Status: OK
        Expression: SC_r <- f ( 2 ); (x integer, returns number)
                    DS_r <- DS_2 [ calc Me_3 := SC_r ];
        Description: The Integer result 2 is cast to the Number the returns declares, 2.0. It was
                     the Integer 2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_11", number_inputs=1, references_names=["1", "2"], vd_names=self.vd_1364
        )

    def test_GH_1364_12(self):
        """
        Status: OK
        Expression: SC_r <- f ( true ); (x boolean, returns string)
                    DS_r <- DS_2 [ calc Me_3 := SC_r ];
        Description: The Boolean result true is cast to the String the returns declares, "true".
                     It was the Boolean true.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_12", number_inputs=1, references_names=["1", "2"], vd_names=self.vd_1364
        )

    def test_GH_1364_13(self):
        """
        Status: OK
        Expression: SC_r <- f ( 2.5 ); (x number, returns integer)
        Description: A Number is not implicitly cast to Integer, so the Number result does not
                     have the type the returns declares. It was returned as the Number 2.5.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_13",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined output declared as integer, found number.",
            vd_names=self.vd_1364,
        )

    def test_GH_1364_14(self):
        """
        Status: OK
        Expression: SC_r <- f ( "1" ); (x string, returns integer)
        Description: Control: the String result does not have the type the returns declares.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_14",
            number_inputs=1,
            exception_code="1-3-1-1",
            exception_message="At op f: User defined output declared as integer, found string.",
            vd_names=self.vd_1364,
        )

    def test_GH_1364_15(self):
        """
        Status: OK
        Expression: DS_r <- DS_2 [ calc Me_3 := f ( Me_1 ) ]; (returns component < number >)
        Description: The Integer Component result is cast to the Number the returns declares. It
                     was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_15", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_16(self):
        """
        Status: OK
        Expression: DS_r <- f ( DS_2 ); (returns dataset { measure < number > Me_1 },
                    is x [ keep Me_1 ])
        Description: The Integer Measure Me_1 of the result is cast to the Number the dataset type
                     of the returns declares for it. It was an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_16", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_17(self):
        """
        Status: OK
        Expression: SC_r <- f ( 2 ); (x integer, returns VD_NUM)
                    DS_r <- DS_2 [ calc Me_3 := SC_r ];
        Description: Control: the Integer result 2 is on VD_NUM, a value domain of Numbers, as
                     2.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_17", number_inputs=1, references_names=["1", "2"], vd_names=self.vd_1364
        )

    def test_GH_1364_18(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_3 as a , DS_4 as b apply a + b );
                    (rules VP_1 on VD_STR, VP_2 on VD_STR2, VP_3 on the variable At_1)
        Description: The viral Attribute At_1, on VD_STR in DS_3 and on VD_STR2 in DS_4, is
                     combined with the rule VP_3 of the variable in the join step, as in
                     DS_3 + DS_4, and kept by apply. The result had no At_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_18", number_inputs=2, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_19(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 + DS_4;
                    (rules VP_1 on VD_STR, VP_2 on VD_STR2, VP_3 on the variable At_1)
        Description: Control: the viral Attribute At_1 is combined with the rule VP_3 of the
                     variable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_19", number_inputs=2, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_20(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_3 as a , DS_3 as b apply a + b ); (rule VP_1 on
                    VD_STR)
        Description: The viral Attribute At_1, on VD_STR in both operands, is combined with the
                     rule VP_1 of its value domain and keeps VD_STR, as in DS_3 + DS_3. The result
                     had no At_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1364_20", number_inputs=1, references_names=["1"], vd_names=self.vd_1364
        )

    def test_GH_1364_21(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_5 as a , DS_5 as b apply a + b );
        Description: Control: the viral Attribute At_1 of both operands has no propagation rule.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_21",
            number_inputs=1,
            exception_code="1-3-3-6",
            exception_message="Viral attribute At_1 has no viral propagation rule",
            vd_names=self.vd_1364,
        )

    def test_GH_1364_22(self):
        """
        Status: OK
        Expression: DS_r <- if DS_6 then DS_7 else DS_2 [ keep Me_1 ];
        Description: The Identifier Id_1 is a String in DS_7 and an Integer in the condition: the
                     message names the operator, the Identifier and both types. It said that
                     DS_7 had a different number of Identifiers, at op None.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_22",
            number_inputs=3,
            exception_code="1-1-9-10",
            exception_message=(
                "At op if: The Identifiers of DS_7 (Id_1 String) are not the Identifiers of the "
                "condition (Id_1 Integer)."
            ),
            vd_names=self.vd_1364,
        )

    def test_GH_1364_23(self):
        """
        Status: OK
        Expression: DS_r <- case when DS_6 then DS_7 else DS_2 [ keep Me_1 ];
        Description: As 22, for case: the message names case, Id_1 and both types.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_23",
            number_inputs=3,
            exception_code="1-1-9-10",
            exception_message=(
                "At op case: The Identifiers of DS_7 (Id_1 String) are not the Identifiers of "
                "the condition (Id_1 Integer)."
            ),
            vd_names=self.vd_1364,
        )

    def test_GH_1364_24(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 + DS_2 [ keep Me_1 ];
        Description: Control: the String Id_1 of DS_7 and the Integer Id_1 of DS_2 have no common
                     type.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_24",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="At op +: Incompatible types for common identifier Id_1: String "
            "and Integer.",
            vd_names=self.vd_1364,
        )

    def test_GH_1364_25(self):
        """
        Status: OK
        Expression: SC_r <- f ( 3.5 ); (x integer)
        Description: Control: the literal 3.5 given to the integer parameter x is known from the
                     script, so the semantic analysis raises too.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_25",
            number_inputs=1,
            exception_code="1-2-18",
            exception_message="Expected Integer, got Number on UDO f, parameter x.",
            vd_names=self.vd_1364,
        )

    def test_GH_1364_26(self):
        """
        Status: OK
        Expression: SC_r <- f ( 7 / 2 ); (x integer)
        Description: The value 3.5 given to the integer parameter x is computed when the script
                     runs, so it is an error of the data, which the semantic analysis accepts. It
                     raised the SemanticError 1-2-18.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_26",
            number_inputs=1,
            exception_code="2-3-10",
            exception_message="At op f: The value 3.5 given to the parameter x is not an Integer.",
            vd_names=self.vd_1364,
        )

    def test_GH_1364_27(self):
        """
        Status: OK
        Expression: SC_r <- f ( SC_1 / 2 ); (x integer, SC_1 = 7)
        Description: As 26, with the value of the input scalar SC_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1364
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1364_27",
            number_inputs=2,
            exception_code="2-3-10",
            exception_message="At op f: The value 3.5 given to the parameter x is not an Integer.",
            vd_names=self.vd_1364,
            scalars={"SC_1": 7},
        )


class StructureDataBugs(BugHelper):
    """Results whose data do not match their declared structure (issue #1354)."""

    def test_GH_1354_1(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 all );
        Description: check combines no data points, so the viral Attribute At_1 of DS_1 is
                     copied to the result: H, null, null. Both engines declared it and gave
                     no data column for it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_1", number_inputs=1, references_names=["1"])

    def test_GH_1354_2(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 all );
        Description: The Date viral Attribute At_2 of DS_1 is copied to the result:
                     2020-01-01, null, null. Both engines raised KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_2", number_inputs=1, references_names=["1"])

    def test_GH_1354_2b(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 all );
        Description: Extra of case 2: DS_1 has a viral Attribute named imbalance, like a Measure
                     check computes. The computed Measure keeps the name, as before the viral
                     Attributes were copied to the result, and the viral Attribute is not in it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_2b", number_inputs=1, references_names=["1"])

    def test_GH_1354_3(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , DPR_1 all );
        Description: Control: check_datapoint copies the viral Attribute At_1 of DS_1 to each
                     rule of a data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_3", number_inputs=1, references_names=["1"])

    def test_GH_1354_4(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 imbalance DS_2 # Me_1 all );
        Description: The imbalance is matched with the data points of DS_1 by the Identifiers,
                     and all keeps the data point (1, A), which has none in DS_2, with a null
                     imbalance: null, 20, 30. The pandas engine paired them by position, and
                     the DuckDB engine dropped (1, A).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_4", number_inputs=2, references_names=["1"])

    def test_GH_1354_5(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 errorcode "E1" all );
        Description: The result Measure of check is always bool_var, also when DS_1 has one
                     Boolean Measure Me_1. Both engines and the semantic analysis declared
                     Me_1, which the DuckDB engine wrote as bool_var.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_5", number_inputs=1, references_names=["1"])

    def test_GH_1354_6(self):
        """
        Status: OK
        Expression: DS_r <- sum ( check_datapoint ( DS_1 , DPR_1 invalid ) [ keep errorlevel ]
                        group by Id_1 );
        Description: The errorlevel of R2, which gives none, is a null Integer, as the
                     structure declares, so sum reads it. The DuckDB engine wrote it as text
                     and raised BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_6", number_inputs=1, references_names=["1"])

    def test_GH_1354_7(self):
        """
        Status: OK
        Expression: DS_r <- sum ( check_hierarchy ( DS_1 , HR_1 rule Id_2 all )
                        [ keep errorlevel ] group by Id_1 );
        Description: As test_GH_1354_6 for a hierarchical ruleset.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_7", number_inputs=1, references_names=["1"])

    def test_GH_1354_8(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 errorcode "E1" invalid )
                        [ calc Me_9 := errorlevel + 1 ];
        Description: The errorlevel of a check that gives none is a null Integer, so
                     Me_9 is a null Integer. The DuckDB engine raised BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_8", number_inputs=1, references_names=["1"])

    def test_GH_1354_9(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 [ calc errorcode := "x" , errorlevel := 9 ] ,
                        DPR_1 all_measures );
        Description: The operand Measure errorcode collides with the one check_datapoint
                     computes. The pandas engine gave two errorcode and two errorlevel
                     columns, and the DuckDB engine replaced the operand's.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_9",
            number_inputs=1,
            exception_code="1-1-6-14",
            exception_message="The component errorcode in Dataset",
        )

    def test_GH_1354_9b(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 [ rename Me_1 to bool_var ] , DPR_2 all );
        Description: Extra of case 9: with all the result does not keep the Measures of the
                     operand, so its Measure bool_var collides with nothing and the result is
                     given.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_9b", number_inputs=1, references_names=["1"])

    def test_GH_1354_9c(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 [ calc imbalance := Me_1 ] , DPR_1
                        all_measures );
        Description: Extra of case 9: the operand Measure imbalance collides with the name of
                     a validation Measure. The pandas engine gave two imbalance columns, the
                     DuckDB engine one.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_9c",
            number_inputs=1,
            exception_code="1-1-6-14",
            exception_message="The component imbalance in Dataset",
        )

    def test_GH_1354_10(self):
        """
        Status: OK
        Expression: DS_r <- check_hierarchy ( DS_1 [ rename Me_1 to imbalance ] , HR_1
                        rule Id_2 all_measures );
        Description: The operand Measure imbalance collides with the one check_hierarchy
                     computes. The pandas engine gave two imbalance columns, and the DuckDB
                     engine the operand's value.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_10",
            number_inputs=1,
            exception_code="1-1-6-14",
            exception_message="The component imbalance in Dataset",
        )

    def test_GH_1354_11(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 [ rename Me_1 to bool_var ] , DPR_2
                        invalid );
        Description: The operand Measure bool_var collides with the one check_datapoint
                     computes. The pandas engine raised KeyError, and the DuckDB engine
                     returned the operand's bool_var.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_11",
            number_inputs=1,
            exception_code="1-1-6-14",
            exception_message="The component bool_var in Dataset",
        )

    def test_GH_1354_11b(self):
        """
        Status: OK
        Expression: DS_r <- check_hierarchy ( DS_1 [ rename Me_1 to imbalance ] , HR_1
                        rule Id_2 all );
        Description: Extra of case 10: with all the result does not keep the Measure of the
                     operand, so its name imbalance collides with nothing and the computed
                     imbalance is given.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_11b", number_inputs=1, references_names=["1"])

    def test_GH_1354_26(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 [ calc Me_2 := Me_1 ] , DS_1 );
        Description: The Data Set operands of nvl must have the same Components. Both engines
                     and the semantic analysis raised KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_26",
            number_inputs=1,
            exception_code="1-1-9-20",
            exception_message="At op nvl",
        )

    def test_GH_1354_27(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_1 [ calc Me_2 := Me_1 ] );
        Description: As test_GH_1354_26 with the extra Measure in the second operand, which
                     both engines dropped.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_27",
            number_inputs=1,
            exception_code="1-1-9-20",
            exception_message="At op nvl",
        )

    def test_GH_1354_28(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_9 := sum ( Me_2 ) group by Id_1
                        having nvl ( min ( Me_2 ) , count ( Me_2 ) ) > 0 ];
        Description: In having the aggregates are values of the groups, so nvl reads count
                     however its Measure is named, as in test_GH_1354_28b. Both engines and
                     the semantic analysis raised KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_28", number_inputs=1, references_names=["1"])

    def test_GH_1354_28b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_9 := sum ( Me_2 ) group by Id_1
                        having nvl ( min ( Me_2 ) , 0 ) > 0 ];
        Description: Control: nvl of an aggregate and a constant in having.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_28b", number_inputs=1, references_names=["1"])

    def test_GH_1354_11c(self):
        """
        Status: OK
        Expression: DS_r <- check_datapoint ( DS_1 , DPR all );
        Description: Extra of case 11: DS_1 has a viral Attribute named imbalance, which
                     check_datapoint does not compute, so it is copied to the result as any
                     viral Attribute. The pandas engine raised KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_11c", number_inputs=1, references_names=["1"])

    def test_GH_1354_12(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := sum ( Me_1 ) group by Id_1 ] [ keep Me_2 ];
        Description: The viral Attribute At_1 that aggr propagates with VP_1 is kept by keep:
                     H, X. The DuckDB engine declared it and gave no data column for it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_12", number_inputs=1, references_names=["1"])

    def test_GH_1354_13(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := max ( Id_2 ) group by Id_1 ]
                        [ calc Me_3 := cast ( Me_2 , string ) ];
        Description: Me_2 is the String that max gives, so cast reads it as a String: B, A.
                     The DuckDB engine took it as a Number and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13", number_inputs=1, references_names=["1"])

    def test_GH_1354_13b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := max ( upper ( Id_2 ) ) group by Id_1 ]
                        [ calc Me_3 := cast ( Me_2 , string ) ];
        Description: Extra of case 13: Me_2 is the String that max gives of a String
                     function, so cast reads it as a String: B, A. The DuckDB engine took it as
                     a Number and raised 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13b", number_inputs=1, references_names=["1"])

    def test_GH_1354_13c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := max ( length ( Id_2 ) ) group by Id_1 ]
                        [ calc Me_3 := cast ( Me_2 , string ) ];
        Description: Extra of case 13: Me_2 is the Integer that max gives of length, so cast
                     writes it as 1. The DuckDB engine took it as a Number and wrote 1.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13c", number_inputs=1, references_names=["1"])

    def test_GH_1354_13d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := max ( Id_2 = "A" ) group by Id_1 ]
                        [ calc Me_3 := cast ( Me_2 , string ) ];
        Description: Extra of case 13: Me_2 is the Boolean that max gives of a comparison, so
                     cast writes it as true. The DuckDB engine took it as a Number and wrote
                     1.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13d", number_inputs=1, references_names=["1"])

    def test_GH_1354_13e(self):
        """
        Status: OK
        Expression: DS_r <- max ( DS_1 group by Id_1 );
        Description: Extra of case 13: max of a Time_Period Measure grouped by an Integer
                     Identifier: 2020Q2, 2021Q1. The pandas engine raised
                     InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13e", number_inputs=1, references_names=["1"])

    def test_GH_1354_13f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := min ( Me_1 ) group by Id_1 ];
        Description: Extra of case 13: min of a Time_Period Measure in an aggr clause grouped
                     by an Integer Identifier: 2020Q1, 2021Q1. The pandas engine raised
                     InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13f", number_inputs=1, references_names=["1"])

    def test_GH_1354_13g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := max ( Me_4 over ( order by Id_1 ) ) ];
        Description: Extra of case 13: the analytic max of Time_Periods of different period
                     indicators raises 2-1-19-20, as the aggregate max does. The DuckDB engine
                     compared them and returned 2020M1, 2020Q1, 2020M3.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_13g",
            number_inputs=1,
            exception_code="2-1-19-20",
            exception_message="Time Period operands with different period indicators do not "
            "support < and > Comparison operations, unable to get the max",
        )

    def test_GH_1354_13h(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_5 := max ( Me_1 ) ] [ calc Me_6 := Me_5 + 1 ];
        Description: Extra of case 13: an aggr without grouping gives a single Data Point, also
                     when another clause follows it: Me_5 3.0, Me_6 4.0. The DuckDB engine
                     grouped it by the Identifiers of DS_1 and gave three Data Points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13h", number_inputs=1, references_names=["1"])

    def test_GH_1354_13i(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_2 := max ( Me_1 ) , Me_3 := min ( Me_1 )
                        group by Id_1 ];
                    DS_r2 <- DS_2 [ aggr Me_2 := max ( Me_1 ) , Me_3 := min ( Me_1 )
                        group by Id_1 ];
        Description: Extra of case 13: max and min order the Time_Periods by their value, not
                     by their text: 2020M10 and 2020M2, 2021M1 and 2020M12 for the months,
                     2020W10 and 2020W9, 2021W1 and 2020W52 for the weeks.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13i", number_inputs=2, references_names=["1", "2"])

    def test_GH_1354_13j(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_5 := max ( Me_1 ) ]
                        [ calc Me_6 := sum ( Me_5 over ( ) ) ];
        Description: Extra of case 13: an analytic function over the result of an aggr without
                     grouping, which has no Identifiers: Me_5 3.0, Me_6 3.0. The pandas engine
                     raised ParserException, and the DuckDB engine gave three Data Points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13j", number_inputs=1, references_names=["1"])

    def test_GH_1354_13k(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 [ keep Me_4 ] as a , DS_1 [ keep Me_1 ] as b
                        calc Me_10 := max ( a#Me_4 over ( order by Id_1 ) ) );
        Description: Extra of case 13: the analytic max of the Time_Periods of a join,
                     named with their alias, raises 2-1-19-20 for different period indicators.
                     The DuckDB engine compared them and returned 2020M1, 2020Q1, 2020Q1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_13k",
            number_inputs=1,
            exception_code="2-1-19-20",
            exception_message="Time Period operands with different period indicators do not "
            "support < and > Comparison operations, unable to get the max",
        )

    def test_GH_1354_13l(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_10 := max ( nvl ( Me_4 , cast ( "2020M1" ,
                        time_period ) ) over ( order by Id_1 ) ) ];
        Description: Extra of case 13: the analytic max of a Time_Period expression raises
                     2-1-19-20 for different period indicators, as of a Time_Period Measure.
                     The DuckDB engine compared them and returned 2020M1, 2020Q1, 2020Q1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_13l",
            number_inputs=1,
            exception_code="2-1-19-20",
            exception_message="Time Period operands with different period indicators do not "
            "support < and > Comparison operations, unable to get the max",
        )

    def test_GH_1354_13m(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr Me_5 := max ( Me_1 ) group all ]
                        [ calc Me_6 := Me_5 + 1 ];
        Description: Extra of case 13: an aggr grouped by group all with no time_agg gives a
                     single Data Point, also when another clause follows it: Me_5 3.0, Me_6
                     4.0. The DuckDB engine grouped it by the Identifiers of DS_1 and gave
                     three Data Points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_13m", number_inputs=1, references_names=["1"])

    def test_GH_1354_14(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc viral attribute At_9 := "v" ] [ keep Me_1 ];
        Description: At_9 is a viral Attribute, which keep does not drop: v, v, v. The DuckDB
                     engine took it as a Measure and gave no data column for it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_14", number_inputs=1, references_names=["1"])

    def test_GH_1354_14b(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 [ calc viral attribute errorcode := "x" ] > 0 );
        Description: Extra of case 14: the comparison reads the Measure Me_1 only, not the
                     viral Attribute errorcode, and the computed Measure errorcode of check
                     keeps the name. The DuckDB engine compared errorcode as a Measure and
                     raised BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_14b", number_inputs=1, references_names=["1"])

    def test_GH_1354_15(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ drop Me_1 ] [ unpivot Id_3 , Me_9 ];
        Description: unpivot of a Data Set without Measures gives no data points, with the
                     new Identifier Id_3 and Measure Me_9. The DuckDB engine returned the data
                     points of the operand without Id_3 and Me_9.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_15", number_inputs=1, references_names=["1"])

    def test_GH_1354_16(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := max ( At_3 over ( partition by Id_2 ) ) ];
        Description: An analytic function over the Attribute At_3 gives the String Measure
                     Me_3: p, r, r. Both engines raised KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_16", number_inputs=1, references_names=["1"])

    def test_GH_1354_17(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := count ( Me_1 over ( order by Me_2 ) ) ];
        Description: count ordered by another Measure gives an Integer, as ordered by the
                     counted one (test_GH_1354_18): 1, 1, 1. The pandas engine raised
                     KeyError, and the DuckDB engine and the semantic analysis gave a Number.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_17", number_inputs=1, references_names=["1"])

    def test_GH_1354_18(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_3 := count ( Me_1 over ( order by Me_1 ) ) ];
        Description: Control: count ordered by the counted Measure gives an Integer.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_18", number_inputs=1, references_names=["1"])

    def test_GH_1354_19(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_1 [ sub Id_2 = "A" ] , DS_1 , all );
                    DS_r2 <- count ( exists_in ( DS_1 [ sub Id_2 = "A" ] , DS_1 , all ) );
        Description: The result has one data point for each data point of the first operand,
                     also when the second operand has more Identifiers: Id_1 1, 2, bool_var
                     true, true, so count gives 2. The pandas engine repeated Id_1 1 for each
                     data point of DS_1 that matches it, and count gave 3.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_19", number_inputs=1, references_names=["1", "2"])

    def test_GH_1354_20(self):
        """
        Status: OK
        Expression: DS_r <- exists_in ( DS_1 > 0 , DS_2 );
        Description: The first operand already has a Measure bool_var, which the bool_var of
                     exists_in replaces: false, true, true. The pandas engine raised KeyError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_20", number_inputs=2, references_names=["1"])

    def test_GH_1354_21(self):
        """
        Status: OK
        Expression: DS_r <- nvl ( DS_1 , DS_2 );
        Description: nvl copies the viral Attribute At_1 of the first operand and runs no
                     propagation rule: H, null, null. The DuckDB engine filled the nulls from
                     DS_2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_21", number_inputs=2, references_names=["1"])

    def test_GH_1354_21b(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( variable At_1 ) is
                        when "2020-01-01" then "2021-01-01" ; else "2022-01-01"
                    end viral propagation;
                    DS_r <- sum ( DS_1 group by Id_1 );
        Description: Extra of case 21: a rule on the Date viral Attribute At_1 written with
                     String literals combines the values of each group as Dates: 2021-01-01,
                     2022-01-01. The DuckDB engine raised BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_21b", number_inputs=1, references_names=["1"])

    def test_GH_1354_21c(self):
        """
        Status: OK
        Expression: define viral propagation VP_1 ( variable At_1 ) is
                        when "2020-01-01" then "2021-01-01" ; else "2022-01-01"
                    end viral propagation;
                    DS_r <- sum ( DS_1 over ( partition by Id_2 ) );
        Description: Extra of case 21: an analytic function on a Data Set with the Date viral
                     Attribute At_1 combines it over each partition with VP_1: 2021-01-01,
                     2022-01-01, 2021-01-01. The pandas engine raised KeyError, and the DuckDB
                     engine raised BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_21c", number_inputs=1, references_names=["1"])

    def test_GH_1354_22(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := Id_1 + null ];
        Description: A Component combined with null is nullable: Me_2 null, null, null. Both
                     engines and the semantic analysis declared it non-nullable, so the pandas
                     engine raised 2-1-1-16.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22", number_inputs=1, references_names=["1"])

    def test_GH_1354_22b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier Id_3 := Id_1 + SC_1 ];
        Description: Extra of case 22: the input Scalar SC_1 is given the value 5, so Id_1 + SC_1
                     is not null and calc identifier accepts it: 6, 6, 7.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1354_22b", number_inputs=1, references_names=["1"], scalars={"SC_1": 5}
        )

    def test_GH_1354_22c(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 + null;
        Description: Extra of case 22: a Data Set combined with null has nullable Measures, also
                     where the Measure of DS_1 is not nullable: null, null, null. Both engines and
                     the semantic analysis declared it non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22c", number_inputs=1, references_names=["1"])

    def test_GH_1354_22d(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := between ( Me_1 , 1 , null ) ];
        Description: Extra of case 22: between with a null bound is nullable, also over a
                     Measure that is not: null, false, null. The pandas engine raised 2-1-1-16,
                     and the DuckDB engine returned nulls in a Measure declared non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22d", number_inputs=1, references_names=["1"])

    def test_GH_1354_22e(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := instr ( Me_2 , cast ( null , string ) ) ];
        Description: Extra of case 22: instr with a null pattern is nullable, also over a
                     Measure that is not: null, null, null. The pandas engine raised 2-1-1-16,
                     and the DuckDB engine returned nulls in a Measure declared non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22e", number_inputs=1, references_names=["1"])

    def test_GH_1354_22f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := replace ( Me_2 , cast ( null , string ) , "a" ) ];
        Description: Extra of case 22: replace with a null pattern is nullable, also over a
                     Measure that is not: null, null, null. The pandas engine raised 2-1-1-16,
                     and the DuckDB engine returned nulls in a Measure declared non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22f", number_inputs=1, references_names=["1"])

    def test_GH_1354_22g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := dateadd ( Me_5 , 1 , cast ( null , string ) ) ];
        Description: Extra of case 22: dateadd with a null period indicator is nullable, also
                     over a Measure that is not: null, null, null. The pandas engine raised
                     2-1-1-16, and the DuckDB engine returned nulls in a Measure declared
                     non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22g", number_inputs=1, references_names=["1"])

    def test_GH_1354_22h(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := instr ( Me_2 , "a" , 1 , Me_7 ) ];
        Description: Extra of case 22: instr takes the occurrence from the Component Me_7 in
                     each Data Point, a null as the default occurrence: 1, 2, 0. The pandas
                     engine raised 2-1-18-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22h", number_inputs=1, references_names=["1"])

    def test_GH_1354_22i(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := substr ( Me_2 , Me_7 , Me_7 ) ];
        Description: Extra of case 22: substr takes the start and the length from the
                     Component Me_7 in each Data Point, a null as the default: a, ba, d. The
                     pandas engine raised ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22i", number_inputs=1, references_names=["1"])

    def test_GH_1354_22j(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_9 := replace ( Me_2 , Me_6 , Me_6 ) ];
        Description: Extra of case 22: replace takes the pattern and the replacement from the
                     nullable Component Me_6 in each Data Point, so the result is nullable: ab,
                     null, cd. The pandas engine raised ValueError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22j", number_inputs=1, references_names=["1"])

    def test_GH_1354_22k(self):
        """
        Status: OK
        Expression: DS_r <- instr ( DS_1 , null );
        Description: Extra of case 22: instr of a Data Set with a null pattern has a nullable
                     Measure, also over a Measure that is not: null, null, null. Both engines
                     declared it non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_22k", number_inputs=1, references_names=["1"])

    def test_GH_1354_23(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := nvl ( Me_1 , cast ( "2020-01-01" , date ) ) ];
        Description: A cast of a literal is a constant that is not null, so the nvl that it
                     replaces nulls with is non-nullable. The DuckDB engine and the semantic
                     analysis declared it nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_23", number_inputs=1, references_names=["1"])

    def test_GH_1354_24(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc identifier Id_3 := nvl ( Me_1 ,
                        cast ( "2020-01-01" , date ) ) ];
        Description: As test_GH_1354_23, calc identifier accepts the nvl: 2020-05-01,
                     2020-01-01, 2020-06-01. The DuckDB engine and the semantic analysis
                     raised 1-1-1-16.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_24", number_inputs=1, references_names=["1"])

    def test_GH_1354_25(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := sum ( Id_1 over ( order by Id_1 , Id_2
                        data points between 1 preceding and 1 preceding ) ) ];
        Description: The window of an analytic function can be empty, so its result over an
                     Identifier is nullable: null, 1, 1. Both engines and the semantic analysis
                     declared it non-nullable, so the pandas engine raised 2-1-1-16.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_25", number_inputs=1, references_names=["1"])

    def test_GH_1354_25b(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ filter Id_1 = 3 ]
                        [ calc Me_6 := stddev_samp ( Me_1 over ( partition by Id_1 ) ) ];
        Description: Extra of case 25: a window of a single Data Point has no sample standard
                     deviation, so the result is nullable, also over a Measure that is not:
                     null. The pandas engine raised 2-1-1-16, and the DuckDB engine returned a
                     null in a Measure declared non-nullable.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_25b", number_inputs=1, references_names=["1"])

    def test_GH_1354_22l(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_8 := substr ( "abc" , Me_7 , 1 ) ,
                                        Me_9 := substr ( SC_1 , Me_3 , Me_3 ) ];
        Description: Extra of case 22: substr of a Scalar takes the start and the length from
                     Components in each Data Point, a null as the default start: Me_8 a, a, b;
                     with SC_1 = "abcabc", Me_9 a, bc, cab, nullable as the input Scalar SC_1.
                     The pandas engine raised AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1354_22l",
            number_inputs=1,
            references_names=["1"],
            scalars={"SC_1": "abcabc"},
        )

    def test_GH_1354_22m(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_8 := instr ( "abc" , Me_6 ) ,
                                        Me_9 := instr ( SC_1 , Me_5 , Me_3 , Me_3 ) ];
        Description: Extra of case 22: instr of a Scalar takes the pattern, the start and the
                     occurrence from Components in each Data Point: Me_8 1, null, 3, nullable
                     as the pattern Me_6; with SC_1 = "abcabc", Me_9 1, 5, 0. The pandas engine
                     raised AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1354_22m",
            number_inputs=1,
            references_names=["1"],
            scalars={"SC_1": "abcabc"},
        )

    def test_GH_1354_22n(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_8 := replace ( "abc" , Me_6 , "z" ) ,
                                        Me_9 := replace ( SC_1 , Me_5 ) ];
        Description: Extra of case 22: replace of a Scalar takes the pattern from Components in
                     each Data Point: Me_8 zbc, null, abz, nullable as the pattern Me_6; with
                     SC_1 = "abcabc", Me_9 bcbc, acac, abab. The pandas engine raised
                     AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1354_22n",
            number_inputs=1,
            references_names=["1"],
            scalars={"SC_1": "abcabc"},
        )

    def test_GH_1354_29b(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 [ aggr Me_2 := sum ( Me_1 ) ] > 3
                        imbalance DS_1 [ aggr Me_2 := sum ( Me_1 ) ] );
        Description: An operand and an imbalance without Identifiers have one Data Point each,
                     which match: bool_var true, imbalance 5. The pandas engine raised
                     IndexError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1354_29b", number_inputs=1, references_names=["1"])

    def test_GH_1354_29c(self):
        """
        Status: OK
        Expression: DS_r <- check ( DS_1 > 0 imbalance DS_2 );
        Description: The Integer Id_1 of the operand and the String Id_1 of the imbalance have
                     no common type, so check raises 1-1-13-18, as the binary operators do. The
                     pandas engine raised ValueError, the DuckDB engine and the semantic
                     analysis accepted it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1354
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1354_29c",
            number_inputs=2,
            exception_code="1-1-13-18",
            exception_message="At op check: Incompatible types for common identifier Id_1: "
            "Integer and String.",
        )


class NameResolutionBugs(BugHelper):
    """Names inside user defined operators resolved to what they name (issue #1355)."""

    vd_1355 = ["GH_1355-VD_1"]

    def test_GH_1355_1(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , c component ) returns dataset is
                        d [ rename Me_1 to c ] end operator;
                    DS_r <- f ( DS_1 , Me_9 );
        Description: The component parameter c, as the new name of rename, stands for its
                     argument Me_9, on both engines and in the semantic analysis. The result
                     component was named c, and the DuckDB engine lost its column.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_1", number_inputs=1, references_names=["1"])
        self.BaseTest(code="GH_1355_1", number_inputs=1, references_names=["1"], only_semantic=True)

    def test_GH_1355_2(self):
        """
        Status: OK
        Expression: define operator f ( value integer ) returns integer [ value > 0 ] is
                        value + 10 end operator;
                    SC_r <- f ( 3 );
                    DS_r <- DS_3 [ calc Me_3 := SC_r ];
        Description: value in the constraint of the result is the result being checked, and
                     value in the body is the parameter, so f ( 3 ) is 13. The DuckDB engine
                     raised a RecursionError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_2", number_inputs=1, references_names=["1", "2"])

    def test_GH_1355_2b(self):
        """
        Status: OK
        Expression: define operator f ( value integer ) returns VD_2 [ value > 0 ] is
                        value + 10 end operator;
                    define operator g ( value component ) returns component < VD_2 [ value > 0 ]
                        > is value + 10 end operator;
                    SC_r <- f ( 3 );
                    DS_r <- DS_3 [ calc Me_3 := SC_r , Me_4 := g ( Me_2 ) ];
        Description: Extra case: as test_GH_1355_2, for operators that return a value of the
                     value domain VD_2, a scalar and a Component. The DuckDB engine raised a
                     RecursionError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_2b",
            number_inputs=1,
            references_names=["1", "2"],
            vd_names=["GH_1355-VD_2"],
        )

    def test_GH_1355_3(self):
        """
        Status: OK
        Expression: define operator f ( value component )
                        returns component < integer [ value > 0 ] > is value - 10 end operator;
                    DS_r <- DS_3 [ calc Me_3 := f ( Me_2 ) ];
        Description: The result 3 - 10 does not meet the constraint of the result, where value
                     is the result, so the call raises 2-3-9. The DuckDB engine raised a
                     RecursionError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_3",
            number_inputs=1,
            exception_code="2-3-9",
            exception_message=(
                "At op f: the output has the value -7, which its type constraint "
                "[value > 0] does not allow."
            ),
        )

    def test_GH_1355_4(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns integer [ value > 0 ] is x - 10
                        end operator;
                    SC_r <- f ( 3 );
        Description: Control: with the parameter named x, the constraint of the result raises
                     2-3-9 on both engines.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_4",
            number_inputs=1,
            exception_code="2-3-9",
            exception_message=(
                "At op f: the output has the value -7, which its type constraint "
                "[value > 0] does not allow."
            ),
        )

    def test_GH_1355_5(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , r component ) returns dataset is
                        hierarchy ( d , HR_1 rule r all ) end operator;
                    DS_r <- f ( DS_5 , Id_2 );
        Description: The component parameter r, as the rule component of a value domain
                     hierarchical ruleset, stands for its argument Id_2. The DuckDB engine
                     read the column r, which does not exist.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_5", number_inputs=1, references_names=["1"], vd_names=self.vd_1355
        )

    def test_GH_1355_6(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , r component ) returns dataset is
                        check_hierarchy ( d , HR_1 rule r all ) end operator;
                    DS_r <- f ( DS_5 , Id_2 );
        Description: As test_GH_1355_5, for check_hierarchy: the result is the one of
                     test_GH_1355_7. The DuckDB engine read the column r, which does not exist.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_6", number_inputs=1, references_names=["1"], vd_names=self.vd_1355
        )

    def test_GH_1355_7(self):
        """
        Status: OK
        Expression: DS_r <- check_hierarchy ( DS_5 , HR_1 rule Id_2 all );
        Description: Control: the same check_hierarchy outside an operator.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_7", number_inputs=1, references_names=["1"], vd_names=self.vd_1355
        )

    def test_GH_1355_8(self):
        """
        Status: OK
        Expression: define operator f ( x dataset ) returns dataset is x + DS_4 end operator;
                    DS_r <- f ( DS_3 );
        Description: The body of f reads the input DS_4, which it does not receive as a
                     parameter, so the statement reads DS_4. The DuckDB engine never loaded it.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_8", number_inputs=2, references_names=["1"])

    def test_GH_1355_9(self):
        """
        Status: OK
        Expression: DS_9 <- DS_4 * 10;
                    define operator f ( x dataset ) returns dataset is x + DS_9 end operator;
                    DS_r <- f ( DS_3 );
        Description: The body of f reads DS_9, which an earlier statement computes, so the
                     statement that calls f depends on it. The DuckDB engine dropped DS_9
                     before the call.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_9", number_inputs=2, references_names=["1", "2"])

    def test_GH_1355_20(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x number ) returns dataset is
                        d [ aggr Me_2 := sum ( Me_1 * x ) group by Id_1 ] end operator;
                    DS_r <- f ( DS_1 , 2.5 );
        Description: The scalar parameter x of an operator whose body has an aggr clause is
                     2.5 in the aggregation, as in DS_1 [ aggr Me_2 := sum ( Me_1 * 2.5 ) group
                     by Id_1 ]. Both engines and the semantic analysis raised an internal
                     AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_20", number_inputs=1, references_names=["1"])
        self.BaseTest(
            code="GH_1355_20", number_inputs=1, references_names=["1"], only_semantic=True
        )

    def test_GH_1355_21(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x integer ) returns dataset is
                        d [ aggr Me_2 := sum ( Me_1 ) group by Id_1 ] end operator;
                    DS_r <- f ( DS_1 , 2 );
        Description: The aggr clause gives the result of test_GH_1355_22 also when the
                     operator has a scalar parameter that it does not read. Both engines raised
                     an internal AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_21", number_inputs=1, references_names=["1"])

    def test_GH_1355_22(self):
        """
        Status: OK
        Expression: define operator f ( d dataset ) returns dataset is
                        d [ aggr Me_2 := sum ( Me_1 ) group by Id_1 ] end operator;
                    DS_r <- f ( DS_1 );
        Description: Control: the aggr clause in an operator without a scalar parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_22", number_inputs=1, references_names=["1"])

    def test_GH_1355_23(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x integer ) returns dataset is
                        d [ calc Me_3 := lag ( Me_2 , x over ( order by Id_1 ) ) ] end operator;
                    DS_r <- f ( DS_3 , 1 );
        Description: The scalar parameter x is the offset 1 of lag, as in test_GH_1355_26.
                     The DuckDB engine raised a KeyError with the name of the parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_23", number_inputs=1, references_names=["1"])
        self.BaseTest(
            code="GH_1355_23", number_inputs=1, references_names=["1"], only_semantic=True
        )

    def test_GH_1355_23b(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x integer ) returns dataset is
                        d [ calc Me_3 := lag ( Me_2 , x over ( order by Id_1 ) ) ] end operator;
                    DS_r <- f ( DS_3 , -1 );
        Description: Extra of test_GH_1355_23: the negative offset -1 given to x, which lag
                     takes as lead 1. The DuckDB engine raised a KeyError with the name of the
                     parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_23b", number_inputs=1, references_names=["1"])

    def test_GH_1355_23c(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x integer ) returns dataset is
                        d [ calc Me_3 := lag ( Me_2 , x over ( order by Id_1 ) ) ] end operator;
                    DS_r <- f ( DS_3 , 1 + 0 );
        Description: Extra of test_GH_1355_23: the offset given to x is the expression 1 + 0.
                     The DuckDB engine raised a KeyError with the name of the parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_23c", number_inputs=1, references_names=["1"])

    def test_GH_1355_23d(self):
        """
        Status: OK
        Expression: SC_1 <- 1;
                    DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , SC_1 over ( order by Id_1 ) ) ];
        Description: Extra of test_GH_1355_23: the offset of lag is SC_1, a Scalar that an
                     earlier statement computes. The DuckDB engine raised a KeyError with the
                     name of the Scalar.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_23d", number_inputs=1, references_names=["1", "2"])

    def test_GH_1355_23e(self):
        """
        Status: OK
        Expression: SC_1 <- cast ( null , integer );
                    DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , SC_1 over ( order by Id_1 ) ) ];
        Description: Extra of test_GH_1355_23: SC_1, a Scalar that an earlier statement
                     computes, gives lag a null offset, which raises 2-1-18-1 on both engines.
                     The pandas engine raised an internal BinderException and the DuckDB engine
                     gave null data points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_23e",
            number_inputs=1,
            exception_code="2-1-18-1",
            exception_message="At op lag: Offset parameter should be Integer.",
        )

    def test_GH_1355_23f(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , SF_1 over ( order by Id_1 ) ) ];
                    (SF_1 an input Number Scalar, 0.5)
        Description: Extra of test_GH_1355_23: a Number offset is rounded half away from
                     zero, so 0.5 is the offset 1, on both engines. The DuckDB engine rounded it
                     half to even, to 0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_23f", number_inputs=1, references_names=["1"], scalars={"SF_1": 0.5}
        )

    def test_GH_1355_23g(self):
        """
        Status: OK
        Expression: SC_1 <- 2.5;
                    DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , SC_1 over ( order by Id_1 ) ) ];
        Description: Extra of test_GH_1355_23: the offset 2.5 that a statement computes is
                     rounded half away from zero, to 3, on both engines. The DuckDB engine
                     rounded it half to even, to 2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_23g", number_inputs=1, references_names=["1", "2"])

    def test_GH_1355_23h(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , Me_2 over ( order by Id_1 ) ) ];
        Description: Extra of test_GH_1355_23: the offset of lag is an Integer Scalar, so the
                     Component Me_2 raises 1-1-18-4 on both engines and in the semantic
                     analysis. The pandas engine raised an internal ParserException and the
                     DuckDB engine gave null data points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_23h",
            number_inputs=1,
            exception_code="1-1-18-4",
            exception_message="At op lag: Offset parameter should be an Integer Scalar.",
        )

    def test_GH_1355_23i(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , SP_1 over ( order by Id_1 ) ) ];
                    (SP_1 an input String Scalar)
        Description: Extra of test_GH_1355_23: the String Scalar SP_1 is no Integer, which
                     raises 1-1-18-4 on both engines and in the semantic analysis. The pandas
                     engine raised an internal BinderException and the DuckDB engine 2-1-5-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_23i",
            number_inputs=1,
            exception_code="1-1-18-4",
            exception_message="At op lag: Offset parameter should be an Integer Scalar.",
            scalars={"SP_1": "A"},
        )

    def test_GH_1355_24(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x integer ) returns dataset is
                        d [ calc Me_3 := lead ( Me_2 , x over ( order by Id_1 ) ) ] end operator;
                    DS_r <- f ( DS_3 , 1 );
        Description: The scalar parameter x is the offset 1 of lead. The DuckDB engine raised
                     a KeyError with the name of the parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_24", number_inputs=1, references_names=["1"])

    def test_GH_1355_24b(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , x integer ) returns dataset is
                        d [ calc Me_3 := lead ( Me_2 , x over ( order by Id_1 ) ) ] end operator;
                    SC_1 <- cast ( null , integer );
                    DS_r <- f ( DS_3 , SC_1 );
        Description: Extra of test_GH_1355_24: the parameter x, given the null Scalar SC_1, is
                     a null offset of lead, which raises 2-1-18-1 on both engines. The pandas
                     engine raised an internal BinderException and the DuckDB engine gave null
                     data points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_24b",
            number_inputs=1,
            exception_code="2-1-18-1",
            exception_message="At op lead: Offset parameter should be Integer.",
        )

    def test_GH_1355_25(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , p string ) returns dataset is
                        d [ calc Me_3 := time_agg ( p , _ , Tp ) ] end operator;
                    DS_r <- f ( DS_7 , "A" );
        Description: The scalar parameter p is the period "A" of time_agg, as in
                     test_GH_1355_27. The DuckDB engine raised a KeyError with the name of the
                     parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_25", number_inputs=1, references_names=["1"])
        self.BaseTest(
            code="GH_1355_25", number_inputs=1, references_names=["1"], only_semantic=True
        )

    def test_GH_1355_25b(self):
        """
        Status: OK
        Expression: define operator f ( d dataset , p string ) returns dataset is
                        d [ calc Me_3 := time_agg ( p , _ , Tp ) ] end operator;
                    DS_r <- f ( DS_7 , "A" || "" );
        Description: Extra of test_GH_1355_25: the period given to p is the expression
                     "A" || "". The DuckDB engine raised a KeyError with the name of the
                     parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_25b", number_inputs=1, references_names=["1"])

    def test_GH_1355_25c(self):
        """
        Status: OK
        Expression: SC_2 <- "A";
                    DS_r <- DS_7 [ calc Me_3 := time_agg ( SC_2 , _ , Tp ) ];
        Description: Extra of test_GH_1355_25: the period of time_agg is SC_2, a Scalar that
                     an earlier statement computes. The DuckDB engine raised a KeyError with the
                     name of the Scalar.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_25c", number_inputs=1, references_names=["1", "2"])

    def test_GH_1355_25d(self):
        """
        Status: OK
        Expression: SC_2 <- cast ( null , string );
                    DS_r <- DS_7 [ calc Me_3 := time_agg ( SC_2 , _ , Tp ) ];
        Description: Extra of test_GH_1355_25: SC_2, a Scalar that an earlier statement
                     computes, gives time_agg no period, which raises 2-1-19-24 on both engines.
                     The DuckDB engine raised a KeyError with the name of the Scalar.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_25d",
            number_inputs=1,
            exception_code="2-1-19-24",
            exception_message=(
                "At op time_agg: The period indicator SC_2 to aggregate to has no value."
            ),
        )

    def test_GH_1355_25e(self):
        """
        Status: OK
        Expression: SC_2 <- cast ( null , string );
                    DS_r <- DS_7 [ calc Me_3 := time_agg ( SC_2 , _ , Tp ) ];
        Description: Extra of test_GH_1355_25: as test_GH_1355_25d, with DS_7 without data
                     points, the null period raises 2-1-19-24 on both engines. The DuckDB
                     engine checked the period on each data point, so it raised nothing.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_25e",
            number_inputs=1,
            exception_code="2-1-19-24",
            exception_message=(
                "At op time_agg: The period indicator SC_2 to aggregate to has no value."
            ),
        )

    def test_GH_1355_25f(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_7 as a , DS_4 as b
                        calc Me_9 := time_agg ( "A" , _ , a#Tp ) drop b#Me_2 );
        Description: Extra of test_GH_1355_25: time_agg of the Time_Period a#Tp in the calc
                     of a join. The DuckDB engine took a#Tp as a Date and raised an internal
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_25f", number_inputs=2, references_names=["1"])

    def test_GH_1355_26(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := lag ( Me_2 , 1 over ( order by Id_1 ) ) ];
        Description: Control: lag with a literal offset.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_26", number_inputs=1, references_names=["1"])

    def test_GH_1355_26b(self):
        """
        Status: OK
        Expression: SC_1 <- 2;
                    DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SC_1 preceding and current data point ) ) ];
        Description: Extra of test_GH_1355_26: SC_1, a Scalar that an earlier statement
                     computes, is the bound of the window. The DuckDB engine raised a KeyError
                     with the name of the Scalar.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_26b", number_inputs=1, references_names=["1", "2"])

    def test_GH_1355_26c(self):
        """
        Status: OK
        Expression: define operator w ( d dataset , n integer ) returns dataset is
                        d [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between n preceding and current data point ) ) ] end operator;
                    DS_r <- w ( DS_3 , 1 + 0 );
        Description: Extra of test_GH_1355_26: the expression 1 + 0 given to the parameter n
                     is the bound of the window. The DuckDB engine raised a KeyError with the
                     name of the parameter.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_26c", number_inputs=1, references_names=["1"])

    def test_GH_1355_26d(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between Me_2 preceding and current data point ) ) ];
        Description: Extra of test_GH_1355_26: the bound of a window is an Integer Scalar, so
                     the Component Me_2 raises 1-1-18-4 on both engines and in the semantic
                     analysis. The pandas engine raised an internal ParserException and the
                     DuckDB engine read the Me_2 of each data point as its bound.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26d",
            number_inputs=1,
            exception_code="1-1-18-4",
            exception_message="At op sum: Window bound parameter should be an Integer Scalar.",
        )

    def test_GH_1355_26e(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SP_1 preceding and current data point ) ) ];
                    (SP_1 an input String Scalar)
        Description: Extra of test_GH_1355_26: the String Scalar SP_1 is no Integer, which
                     raises 1-1-18-4 on both engines and in the semantic analysis. The pandas
                     engine raised an internal ParserException and the DuckDB engine an
                     internal BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26e",
            number_inputs=1,
            exception_code="1-1-18-4",
            exception_message="At op sum: Window bound parameter should be an Integer Scalar.",
            scalars={"SP_1": "A"},
        )

    def test_GH_1355_26f(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SI_0 preceding and current data point ) ) ];
                    (SI_0 an input Integer Scalar, 0)
        Description: Extra of test_GH_1355_26: the bound 0 that a Scalar gives is the bound 0
                     preceding written in the window, the current data point. The pandas engine
                     raised an internal ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_26f", number_inputs=1, references_names=["1"], scalars={"SI_0": 0}
        )

    def test_GH_1355_26g(self):
        """
        Status: OK
        Expression: DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SF_2 preceding and current data point ) ) ];
                    (SF_2 an input Number Scalar, 2.5)
        Description: Extra of test_GH_1355_26: a Number bound is rounded half away from zero,
                     as the offset of lag, so 2.5 is the bound 3. The pandas engine raised an
                     internal ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_26g", number_inputs=1, references_names=["1"], scalars={"SF_2": 2.5}
        )

    def test_GH_1355_26h(self):
        """
        Status: OK
        Expression: SC_1 <- 2.5;
                    DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SC_1 preceding and current data point ) ) ];
        Description: Extra of test_GH_1355_26: the bound 2.5 that a statement computes is
                     rounded half away from zero, to 3, as in test_GH_1355_26g. The pandas
                     engine raised an internal ParserException and the DuckDB engine rounded
                     it half to even, to 2.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_26h", number_inputs=1, references_names=["1", "2"])

    def test_GH_1355_26i(self):
        """
        Status: OK
        Expression: DS_r <- DS_D [ calc Me_3 := sum ( Me_2 over ( order by Id_1 range
                        between SF_2 preceding and current data point ) ) ];
                    (SF_2 an input Number Scalar, 2.5, Id_1 a Date)
        Description: Extra of test_GH_1355_26: the Number bound 2.5 of a range window ordered
                     by a Date is rounded half away from zero, to 3 days. The pandas engine
                     raised an internal ParserException and the DuckDB engine an internal
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(
            code="GH_1355_26i", number_inputs=1, references_names=["1"], scalars={"SF_2": 2.5}
        )

    def test_GH_1355_27(self):
        """
        Status: OK
        Expression: DS_r <- DS_7 [ calc Me_3 := time_agg ( "A" , _ , Tp ) ];
        Description: Control: time_agg with a literal period.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_27", number_inputs=1, references_names=["1"])

    def test_GH_1355_28(self):
        """
        Status: OK
        Expression: define operator w ( d dataset ) returns dataset is
                        d [ calc Z9 := Me_2 + 1 ] [ calc Z8 := cast ( Z9 , string ) ]
                        end operator;
                    DS_r <- w ( DS_3 ) [ calc Z := 1 ];
        Description: Z9, which the body computes and then reads, is an Integer also where the
                     call is the operand of a clause, so cast to String it gives 4, as w ( DS_3 )
                     alone does. The DuckDB engine took Z9 as a Number and gave 4.0.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_28", number_inputs=1, references_names=["1"])
        self.BaseTest(
            code="GH_1355_28", number_inputs=1, references_names=["1"], only_semantic=True
        )

    def test_GH_1355_10(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B drop B#Id_2 );
        Description: Id_2 is an Identifier of DS_1 and a Measure of DS_2, so the join keeps
                     both and B#Id_2 is the Measure of DS_2: dropping it keeps the Identifier
                     Id_2. The DuckDB engine kept a single Id_2 column and dropped it from the
                     data.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_10", number_inputs=2, references_names=["1"])

    def test_GH_1355_10b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 as B , DS_1 as A drop B#Id_2 );
        Description: Extra of test_GH_1355_10: the same with the operand that has Id_2 as a
                     Measure first.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_10b", number_inputs=2, references_names=["1"])

    def test_GH_1355_10c(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 as X , DS_1 as Y , DS_5 as Z drop X#Id_2 );
        Description: Extra of test_GH_1355_10: three operands, the first with Id_2 as a
                     Measure and the others with Id_2 as an Identifier, are joined on the
                     Identifier Id_2 of DS_1 and DS_5. The DuckDB engine joined on the Measure
                     Id_2 of DS_2 and gave no data point.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_10c", number_inputs=3, references_names=["1"])

    def test_GH_1355_10d(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as 'A-B' , DS_2 as B using Id_1 , Id_2
                        calc Me_9 := 'A-B'#Me_1 + B#Me_1 drop 'A-B'#Me_1 , B#Me_1 );
        Description: Extra of test_GH_1355_10: a quoted alias names its operand without the
                     quotes, as a quoted Data Set name, and any name a Component name allows
                     is a valid alias, also one with a character such as - that SQL does not
                     take without quotes. The DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_10d", number_inputs=2, references_names=["1"])

    def test_GH_1355_10e(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as '1A' , DS_2 as B using Id_1 , Id_2
                        rename '1A'#Me_1 to Me_8 , B#Me_1 to Me_9 );
        Description: Extra of test_GH_1355_10: as test_GH_1355_10d, for an alias that starts
                     with a digit. The DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_10e", number_inputs=2, references_names=["1"])

    def test_GH_1355_10h(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as '' , DS_2 as B using Id_1 , Id_2 );
        Description: Extra of test_GH_1355_10d: an empty alias names no operand, as an empty
                     Component name is no name. The pandas engine ran the join and the DuckDB
                     engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_10h", number_inputs=2, exception_code="1-1-13-28"
        )

    def test_GH_1355_10j(self):
        """
        Status: OK
        Expression: define datapoint ruleset dpr1 ( variable Me_1 as 'X' ) is
                        r1 : X > 1 errorcode "e"
                    end datapoint ruleset;
                    DS_r <- check_datapoint ( DS_1 , dpr1 all );
        Description: Extra of test_GH_1355_10d: a quoted alias of a datapoint ruleset signature
                     names the variable without the quotes too, so the rule reads it as X. Both
                     engines and the semantic analysis raised 1-1-10-7 for X.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_10j", number_inputs=1, references_names=["1"])

    def test_GH_1355_11(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B rename B#Id_2 to Me_3 );
        Description: As test_GH_1355_10, B#Id_2 is the Measure of DS_2, so Me_3 takes its
                     values and the Identifier Id_2 stays. The DuckDB engine gave Me_3 the
                     values of the Identifier.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_11", number_inputs=2, references_names=["1"])

    def test_GH_1355_11b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 as B , DS_1 as A rename B#Id_2 to Me_3 );
        Description: Extra of test_GH_1355_11: the same with the operand that has Id_2 as a
                     Measure first.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_11b", number_inputs=2, references_names=["1"])

    def test_GH_1355_11c(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 as B , DS_1 as A rename A#Id_2 to Me_3 );
        Description: Extra of test_GH_1355_11: A#Id_2 is the Identifier of DS_1, so Me_3 takes
                     its values, and the Measure Id_2 of DS_2 keeps its own. The DuckDB engine
                     renamed both to Me_3 and lost the Measure.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_11c", number_inputs=2, references_names=["1"])

    def test_GH_1355_11d(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B rename Id_2 to Me_3 );
        Description: Extra of test_GH_1355_11: Id_2 written alone is the Identifier Id_2 of
                     the join, so only it is renamed, and the Measure Id_2 of DS_2 keeps its
                     own name. The DuckDB engine renamed both to Me_3 and lost the Measure.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_11d", number_inputs=2, references_names=["1"])

    def test_GH_1355_11e(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc 'Me#1' := Me_1 ];
        Description: Extra of test_GH_1355_11: a new Component cannot be named with #, which
                     only separates the alias of a join operand from the name of its Component
                     (Inner Join page of the VTL 2.2 manual). Both engines created Me#1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11e", number_inputs=1, exception_code="1-1-6-19"
        )

    def test_GH_1355_11f(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B rename A#Me_1 to 'x#y' );
        Description: Extra of test_GH_1355_11: the same for the new name of a rename in a
                     join. Both engines named the Component y, the part after the #.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11f", number_inputs=2, exception_code="1-1-6-19"
        )

    def test_GH_1355_11g(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ rename Me_1 to 'x"y' ];
        Description: Extra of test_GH_1355_11: a new Component cannot be named with a double
                     quote either. The pandas engine renamed Me_1 to x"y and the DuckDB engine
                     raised a raw ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11g", number_inputs=1, exception_code="1-1-6-19"
        )

    def test_GH_1355_11h(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B calc A#Me_1 := 1 );
        Description: Extra of test_GH_1355_11: calc names a new Component, never a membership.
                     Both engines read the qualified target as its alias and created a Measure
                     named A.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11h",
            number_inputs=2,
            exception_code="1-1-6-6",
            exception_message="Membership is not allowed inside a clause, found A#Me_1.",
        )

    def test_GH_1355_11i(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B rename A#Me_2 to Me_9 );
        Description: Extra of test_GH_1355_11: A#Me_2 names the Component Me_2 of the operand
                     A, which has none. Both engines renamed Me_2 of the operand B.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11i",
            number_inputs=2,
            exception_code="1-1-1-10",
            exception_message="Component Me_2 not found in Dataset A.",
        )

    def test_GH_1355_11j(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B aggr Me_9 := sum ( Me_1 )
                        group by C#Id_1 );
        Description: Extra of test_GH_1355_11: C is no operand of the join. Both engines grouped
                     by the Identifier Id_1 of the join.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11j",
            number_inputs=2,
            exception_code="1-1-1-10",
            exception_message="Component Id_1 not found in Dataset C.",
        )

    def test_GH_1355_11k(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B calc Me_9 := B#Me_1 );
        Description: Extra of test_GH_1355_11: Me_1 is a Component of the operand A, not of B.
                     Both engines gave Me_9 the values of Me_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11k",
            number_inputs=2,
            exception_code="1-1-1-10",
            exception_message="Component Me_1 not found in Dataset B.",
        )

    def test_GH_1355_11l(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as 'A#B' , DS_2 as C using Id_1 , Id_2 );
        Description: Extra of test_GH_1355_11: the alias of a join operand prefixes the names
                     of its Components in the join, alias#comp, so it cannot contain #
                     (Inner Join page of the VTL 2.2 manual). The pandas engine and the
                     semantic analysis gave a Component B'#Me_1 and the DuckDB engine raised a
                     raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11l",
            number_inputs=2,
            exception_code="1-1-13-28",
            exception_message="At op inner_join: Invalid name A#B of a join operand, the "
            "alias of a join operand, or else its Data Set name, cannot be empty or contain ', \", "
            "#, NUL or carriage return.",
        )

    def test_GH_1355_11m(self):
        """
        Status: OK
        Expression: 'x#y' <- DS_1;
                    DS_r <- inner_join ( 'x#y' , DS_2 using Id_1 , Id_2 );
        Description: Extra of test_GH_1355_11: as test_GH_1355_11l, for the name of a Data Set
                     joined without an alias. The pandas engine and the semantic analysis gave
                     a Component y#Me_1 and the DuckDB engine raised a raw ParserException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11m",
            number_inputs=2,
            exception_code="1-1-13-28",
            exception_message="At op inner_join: Invalid name x#y of a join operand, the "
            "alias of a join operand, or else its Data Set name, cannot be empty or contain ', \", "
            "#, NUL or carriage return.",
        )

    def test_GH_1355_11n(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B using Id_1 , Id_2
                        calc 'A#Me_1' := 1 drop B#Me_1 );
        Description: Extra of test_GH_1355_11: a calc target written as a quoted name holding
                     # is refused as a new name, also where the join has a Component A#Me_1.
                     Both engines and the semantic analysis gave Me_1 the value 1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11n",
            number_inputs=2,
            exception_code="1-1-6-19",
            exception_message="At op calc: Invalid Component name A#Me_1, a Component name "
            "cannot contain ', \", #, NUL or carriage return.",
        )

    def test_GH_1355_11o(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ aggr 'x#y' := sum ( Me_1 ) group by Id_1 ];
        Description: Extra of test_GH_1355_11: as test_GH_1355_11e, for the new Component of
                     an aggr.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11o", number_inputs=1, exception_code="1-1-6-19"
        )

    def test_GH_1355_11p(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ unpivot Id_9 , 'x#y' ];
        Description: Extra of test_GH_1355_11: as test_GH_1355_11e, for the Measure of an
                     unpivot.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11p", number_inputs=1, exception_code="1-1-6-19"
        )

    def test_GH_1355_11q(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ pivot Id_2 , Me_1 ];
        Description: Extra of test_GH_1355_11: as test_GH_1355_11e, for the Measures of a
                     pivot, which the values x#y of the value domain VD_3 of Id_2 name.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11q",
            number_inputs=1,
            exception_code="1-1-6-19",
            vd_names=["GH_1355-VD_3"],
        )

    def test_GH_1355_11r(self):
        """
        Status: OK
        Expression: DS_r <- eval ( GH_1355_11r ( DS_1 ) language "SQL" returns dataset
                        { identifier < integer > Id_1 , identifier < string > Id_2 ,
                        measure < number > 'x#y' } );
        Description: Extra of test_GH_1355_11: as test_GH_1355_11e, for a Component of the
                     result of eval.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11r",
            number_inputs=1,
            exception_code="1-1-6-19",
            sql_names=["GH_1355_11r"],
        )

    def test_GH_1355_11s(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B keep B#Me_1 );
        Description: Extra of test_GH_1355_11: as test_GH_1355_11k, for keep: Me_1 is a
                     Component of the operand A, not of B.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_11s",
            number_inputs=2,
            exception_code="1-1-1-10",
            exception_message="Component Me_1 not found in Dataset B.",
        )

    def test_GH_1355_12(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_3 as A , DS_4 as B apply A + B
                        rename Me_2 to Me_9 );
        Description: rename in the join body after apply renames the Measure that apply
                     computes. The DuckDB engine read the column A#Me_2, which apply no longer
                     has.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_12", number_inputs=2, references_names=["1"])

    def test_GH_1355_13(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_3 as A , DS_4 as B apply A + B )
                        [ rename Me_2 to Me_9 ];
        Description: Control: the same rename as a clause after the join.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_13", number_inputs=2, references_names=["1"])

    def test_GH_1355_13b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B apply A + B keep A#Me_1 );
        Description: Extra of test_GH_1355_12: apply replaces the Measures Me_1 of A and B
                     with one Measure Me_1, so A#Me_1 is no longer in the join after it (Inner
                     Join page of the VTL 2.2 manual: the input Measures do not appear in the
                     result of apply). The pandas engine kept the Me_1 of apply and the DuckDB
                     engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_13b",
            number_inputs=2,
            exception_code="1-1-13-17",
            exception_message="At op keep: A#Me_1 not present in the Dataset(result from join "
            "VDS) at the time it is called",
        )

    def test_GH_1355_13c(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B apply A + B
                        rename A#Me_1 to Me_9 );
        Description: Extra of test_GH_1355_12: as test_GH_1355_13b, for rename. The pandas
                     engine renamed the Me_1 of apply and the DuckDB engine gave data without
                     the column Me_9 that its structure had.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_13c",
            number_inputs=2,
            exception_code="1-1-13-17",
            exception_message="At op rename: A#Me_1 not present in the Dataset(result from "
            "join VDS) at the time it is called",
        )

    def test_GH_1355_13d(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B apply A + B drop A#Me_1 );
        Description: Extra of test_GH_1355_12: as test_GH_1355_13b, for drop. The pandas
                     engine dropped the Me_1 of apply and the DuckDB engine raised a raw
                     BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_13d", number_inputs=2, exception_code="1-1-13-17"
        )

    def test_GH_1355_13e(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B
                        aggr Me_1 := sum ( A#Me_1 ) group by Id_1 keep A#Me_1 );
        Description: Extra of test_GH_1355_12: as test_GH_1355_13b, after an aggr whose new
                     Measure Me_1 replaces the Measures Me_1 of A and B. The pandas engine
                     kept the Me_1 of aggr and the DuckDB engine raised a raw BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_13e", number_inputs=2, exception_code="1-1-13-17"
        )

    def test_GH_1355_13f(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B using Id_1 , Id_2
                        calc Me_1 := 1 keep A#Me_1 );
        Description: Extra of test_GH_1355_12: as test_GH_1355_13b, after a calc whose new
                     Measure Me_1 replaces the Measures Me_1 of A and B (Inner Join page of the
                     VTL 2.2 manual: the input Components homonymous to a Component that calc
                     generates are removed). Both engines kept the Me_1 of A and dropped the one
                     of calc.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_13f",
            number_inputs=2,
            exception_code="1-1-13-17",
            exception_message="At op keep: A#Me_1 not present in the Dataset(result from join "
            "VDS) at the time it is called",
        )

    def test_GH_1355_13g(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B using Id_1 , Id_2
                        calc Me_1 := 1 );
        Description: Extra of test_GH_1355_13f: the Measure Me_1 that calc generates replaces
                     the Measures Me_1 of A and B, so the result has one Me_1 (Inner Join page
                     of the VTL 2.2 manual). Both engines and the semantic analysis kept A#Me_1
                     and B#Me_1 next to it and raised 1-1-13-9 when the join removed the prefixes.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_13g", number_inputs=2, references_names=["1"])

    def test_GH_1355_14(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_3 as B
                        calc Me_3 := sum ( B#Me_2 over ( partition by A#Id_2 ) ) );
        Description: In a join, the qualified Identifier A#Id_2 in partition by is Id_2 of
                     the operand A, as partition by Id_2. Both engines and the semantic
                     analysis raised an internal AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_14", number_inputs=2, references_names=["1"])
        self.BaseTest(
            code="GH_1355_14", number_inputs=2, references_names=["1"], only_semantic=True
        )

    def test_GH_1355_14b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_8 as a , DS_9 as b
                        calc Me_9 := rank ( over ( order by a#Dt ) ) );
        Description: Extra of test_GH_1355_14: in a join, the qualified Identifier a#Dt in
                     order by is the Date Dt of the operand a, as order by Dt. Both engines
                     raised an internal AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_14b", number_inputs=2, references_names=["1"])

    def test_GH_1355_14c(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_8 as a , DS_9 as b
                        calc Me_9 := count ( a#Me_1 over ( partition by a#Id_1
                            order by a#Me_2 ) ) );
        Description: Extra of test_GH_1355_14: in a join, a qualified Measure is the operand
                     of an analytic and the component of its order by. Both engines raised an
                     internal AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_14c", number_inputs=2, references_names=["1"])

    def test_GH_1355_14d(self):
        """
        Status: OK
        Expression: DS_r <- DS_8 [ calc Me_9 := count ( Me_1 over ( partition by Id_1
                        order by Me_2 ) ) ];
        Description: Extra of test_GH_1355_14: count of a Component ordered by a Measure gives
                     the count, not the Measure of the order by. The pandas engine raised an
                     internal KeyError, where the DuckDB engine gave the result.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_14d", number_inputs=1, references_names=["1"])

    def test_GH_1355_14f(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ rename Me_1 to 'Me#1' ];
        Description: Extra of test_GH_1355_14: a Component named with # out of a join would
                     read as the qualified name of a Component of a join, so the new name of a
                     rename cannot hold it. Both engines renamed Me_1 to Me#1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_14f", number_inputs=1, exception_code="1-1-6-19"
        )

    def test_GH_1355_15(self):
        """
        Status: OK
        Expression: DS_r <- DS_1 [ calc Me_2 := rank ( over ( order by DS_1#Me_1 ) ) ];
        Description: Out of a join, DS_1#Me_1 is a membership, which a clause does not allow
                     in order by, as in DS_1 [ calc Me_2 := DS_1#Me_1 ]. Both engines and the
                     semantic analysis raised an internal AttributeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_15",
            number_inputs=1,
            exception_code="1-1-6-6",
            exception_message="Membership is not allowed inside a clause, found DS_1#Me_1.",
        )

    def test_GH_1355_16(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_3 as B
                        aggr Me_3 := sum ( B#Me_2 ) group by A#Id_2 );
        Description: In a join, the qualified Identifier A#Id_2 in group by is Id_2 of the
                     operand A, as group by Id_2. Both engines and the semantic analysis raised
                     an internal TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_16", number_inputs=2, references_names=["1"])
        self.BaseTest(
            code="GH_1355_16", number_inputs=2, references_names=["1"], only_semantic=True
        )

    def test_GH_1355_16b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_3 as B
                        aggr Me_3 := sum ( B#Me_2 ) group except A#Id_2 );
        Description: Extra of test_GH_1355_16: the same in group except, as group except
                     Id_2. Both engines raised an internal TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_16b", number_inputs=2, references_names=["1"])

    def test_GH_1355_16c(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 as B , DS_1 as A using Id_1
                        aggr Me_3 := max ( A#Me_1 ) group by A#Id_2 );
        Description: Extra of test_GH_1355_16: with using and the operand that has Id_2 as a
                     Measure first, the result keeps the Identifier Id_2 of DS_1. The DuckDB
                     data had no Id_2 column.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_16c", number_inputs=2, references_names=["1"])

    def test_GH_1355_16d(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_2 as B , DS_1 as A using Id_1
                        aggr Me_3 := max ( A#Me_1 ) group except Id_1 );
        Description: Extra of test_GH_1355_16: the same with group except. The DuckDB data had
                     no Id_2 column.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_16d", number_inputs=2, references_names=["1"])

    def test_GH_1355_16e(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B aggr Me_9 := sum ( A#Me_1 )
                        group by Id_1 having sum ( A#Me_1 ) > 0 );
        Description: Extra of test_GH_1355_16: a qualified name in the having of an aggr in a
                     join is the Component of that operand, as in the aggr expression. Both
                     engines and the semantic analysis raised 1-1-1-10 for A#Me_1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Result.
        """
        self.BaseTest(code="GH_1355_16e", number_inputs=2, references_names=["1"])

    def test_GH_1355_17(self):
        """
        Status: OK
        Expression: DS_r <- sum ( DS_1 group by DS_1#Id_2 );
        Description: Out of a join, DS_1#Id_2 is a membership, not a component name, so the
                     grouping raises 1-1-6-6. Both engines and the semantic analysis raised an
                     internal TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_17",
            number_inputs=1,
            exception_code="1-1-6-6",
            exception_message="Membership is not allowed inside a clause, found DS_1#Id_2.",
        )

    def test_GH_1355_18(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_3 as B drop A#Id_2 );
        Description: The qualified Identifier A#Id_2 is Id_2, which drop cannot remove, as
                     drop Id_2. Both engines and the semantic analysis raised an internal
                     TypeError.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GH_1355_18", number_inputs=2, exception_code="1-1-6-2")

    def test_GH_1355_19(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_3 as B keep A#Id_2 );
        Description: As test_GH_1355_18, for keep, as keep Id_2. Both engines and the semantic
                     analysis raised 1-1-1-10 with the whole Component, data included, in the
                     message.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(code="GH_1355_19", number_inputs=2, exception_code="1-1-6-2")

    def test_GH_1355_19b(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_2 as B using Id_1 , Id_2 );
        Description: Extra of test_GH_1355_19: the using key Id_2 is an Attribute of DS_2,
                     an operand other than the left-most one, where it must be an Identifier
                     (the Join operators page of the VTL 2.2 manual: the using clause names
                     identifiers, and the keys are matched as in an SQL inner join). Both
                     engines accepted it and gave different data points.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_19b",
            number_inputs=2,
            exception_code="1-1-13-7",
            exception_message="At op inner_join: Invalid subcase B2, the using component 'Id_2' "
            "must be an Identifier of the Dataset 'B', as of every operand except the "
            "left-most one.",
        )

    def test_GH_1355_19c(self):
        """
        Status: OK
        Expression: DS_r <- left_join ( DS_1 as A , DS_2 as B using Id_1 , Id_2 );
        Description: Extra of test_GH_1355_19: as test_GH_1355_19b for left_join, whose using
                     clause names for each operand but the left-most one identifiers in
                     common with the left-most operand (the Join operators page of the VTL
                     2.2 manual).
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_19c",
            number_inputs=2,
            exception_code="1-1-13-7",
            exception_message="At op left_join: Invalid subcase B2, the using component 'Id_2' "
            "must be an Identifier of the Dataset 'B', as of every operand except the "
            "left-most one.",
        )

    def test_GH_1355_26j(self):
        """
        Status: OK
        Expression: SC_1 <- SI_1 + 1;
                    DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SC_1 preceding and current data point ) ) ];
                    (SI_1 an input Integer Scalar, null)
        Description: Extra of test_GH_1355_26: the bound of the window that a statement computes
                     is null, which raises 2-1-18-1 on both engines. The DuckDB engine raised an
                     internal InvalidInputException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26j",
            number_inputs=1,
            exception_code="2-1-18-1",
            exception_message="At op sum: Window bound parameter should be a non-negative Integer.",
            scalars={"SI_1": None},
        )

    def test_GH_1355_26k(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns dataset is
                        sum ( DS_3 over ( order by Id_1 data points
                        between current data point and x following ) ) end operator;
                    DS_r <- f ( null );
        Description: Extra of test_GH_1355_26: the null written as the argument of the parameter
                     x is the bound of the window, which is no Integer Scalar, so it raises
                     1-1-18-4 on both engines and in the semantic analysis, as the script alone
                     shows it. The DuckDB engine raised an internal BinderException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26k",
            number_inputs=1,
            exception_code="1-1-18-4",
            exception_message="At op sum: Window bound parameter should be an Integer Scalar.",
        )

    def test_GH_1355_26l(self):
        """
        Status: OK
        Expression: SC_1 <- SI_1 * 2;
                    DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SC_1 preceding and current data point ) ) ];
                    DS_s <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SI_1 preceding and current data point ) ) ];
                    (SI_1 an input Integer Scalar, -1)
        Description: Extra of test_GH_1355_26: a negative bound of the window that a Scalar
                     gives, computed by a statement or an input one, raises 2-1-18-1 on both
                     engines, as a null one does. The input Scalar gave an unbounded window, and
                     the computed one an internal ParserException on the pandas engine and an
                     internal InvalidInputException on the DuckDB engine.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26l",
            number_inputs=1,
            exception_code="2-1-18-1",
            exception_message="At op sum: Window bound parameter should be a non-negative Integer.",
            scalars={"SI_1": -1},
        )

    def test_GH_1355_26m(self):
        """
        Status: OK
        Expression: DS_a <- round ( DS_3 , 1 );
                    DS_r <- DS_3 [ calc Me_3 := sum ( Me_2 over ( order by Id_1 data points
                        between SF_2 preceding and current data point ) ) ];
                    (SF_2 an input Number Scalar, 10000000000000000000.0)
        Description: Extra of test_GH_1355_26: the bound of the window that a Scalar gives does
                     not depend on an earlier round, so a bound beyond the range of an Integer
                     raises 2-1-5-1 on both engines. After the round the pandas engine raised an
                     internal ConversionException.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26m",
            number_inputs=1,
            exception_code="2-1-5-1",
            scalars={"SF_2": 10000000000000000000.0},
        )

    def test_GH_1355_26n(self):
        """
        Status: OK
        Expression: define operator f ( x integer ) returns dataset is
                        sum ( DS_3 over ( order by Id_1 data points
                        between current data point and x following ) ) end operator;
                    DS_r <- f ( -1 );
        Description: Extra of test_GH_1355_26: the negative literal -1 given to the parameter x
                     is the bound of the window, which raises 1-1-3-5 on both engines and in the
                     semantic analysis, as the same literal written in the window does. Both
                     engines and the semantic analysis raised 2-1-18-1.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_26n",
            number_inputs=1,
            exception_code="1-1-3-5",
            exception_message="Invalid limit x = -1 of an analytic window: a limit cannot be "
            "negative.",
        )

    def test_GH_1355_case35(self):
        """
        Status: OK
        Expression: DS_r <- inner_join ( DS_1 as A , DS_3 as B calc Me_9 := C#Me_2 );
        Description: Case 35: C is no operand of the join, so C#Me_2 in a calc expression raises
                     1-1-1-10 on both engines and in the semantic analysis. Both engines and the
                     semantic analysis read it as the Me_2 of the join.
        Git Issue: https://github.com/Meaningful-Data/vtlengine/issues/1355
        Goal: Check Exception.
        """
        self.NewSemanticExceptionTest(
            code="GH_1355_case35",
            number_inputs=2,
            exception_code="1-1-1-10",
            exception_message="Component Me_2 not found in Dataset C.",
        )

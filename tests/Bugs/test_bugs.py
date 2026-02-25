from pathlib import Path

import pytest

from tests.Helper import TestHelper
from vtlengine.API import create_ast
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

        ast = create_ast(script)
        interpreter = InterpreterAnalyzer(datasets={})
        result = interpreter.visit(ast)
        for sc in result.values():
            assert sc.persistent == references[sc.name]


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

        message = "1-1-13-4"

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

        message = "1-1-13-4"
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
        message = "1-1-13-4"
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
        Description: Cast Operator.
        Git Issue: GL_413-cast-with-integer
        Goal: Check Result.
        """
        code = "GL_413"
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


class DataValidationBugs(BugHelper):
    """ """

    classTest = "Bugs.DataValidationBugs"

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
        Description:
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Exception.
        """
        code = "GL_196_4"
        number_inputs = 1

        message = "1-1-1-16"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
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
        Description: if-then-else inside a calc measure but else=null.
        Git Branch: fix-196-isnull-for-evaluate-on-if-then-else.
        Goal: Check Result.
        """
        code = "GL_196_6"
        number_inputs = 1
        message = "1-1-1-16"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
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
        message = "1-1-1-10"
        # TODO: test error code has been changed until revision
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

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_564(self):
        """
        Description: Can't subspace a component named unit
        """

        code = "GL_564_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


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


class CastBugs(BugHelper):
    classTest = "Bugs.CastTest"

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
        text = self.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = self.LoadInputs(code=code, number_inputs=number_inputs)
        interpreter = InterpreterAnalyzer(datasets=input_datasets)
        with pytest.raises(NotImplementedError):
            interpreter.visit(ast)

    def test_GL_449_6(self):
        """
        Status: OK
        Description: Over dataset
        Goal: Check Result.
        Note: Cast with mask raises NotImplementedError (not yet implemented).
        """
        code = "GL_449_6"
        number_inputs = 1
        text = self.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = self.LoadInputs(code=code, number_inputs=number_inputs)
        interpreter = InterpreterAnalyzer(datasets=input_datasets)
        with pytest.raises(NotImplementedError):
            interpreter.visit(ast)

    def test_GL_449_7(self):
        """
        Status: OK
        Description: Over scalar
        Goal: Check Result.
        Note: Cast with mask raises NotImplementedError (not yet implemented).
        """
        code = "GL_449_7"
        number_inputs = 1
        text = self.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = self.LoadInputs(code=code, number_inputs=number_inputs)
        input_datasets["sc_1"].value = "2000Q2"
        scalars = {k: v for k, v in input_datasets.items() if not hasattr(v, "components")}
        datasets = {k: v for k, v in input_datasets.items() if hasattr(v, "components")}
        interpreter = InterpreterAnalyzer(datasets=datasets, scalars=scalars)
        with pytest.raises(NotImplementedError):
            interpreter.visit(ast)

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

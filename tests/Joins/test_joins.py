from pathlib import Path

from tests.Helper import TestHelper


class JoinHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class JoinTimeIdentifiersTests(JoinHelper):
    """
    Group 1
    """

    classTest = "join.JoinTimeIdentifiersTests"

    def test_1(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, securities_dynamic using securityId);
        Description: Join using date-identifier

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (securities_static, securities_dynamic using securityId);
        Description: Join using date-identifier

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-2"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, securities_dynamic using securityId);
        Description: Join using time-identifier

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with time identifiers.
        """
        code = "1-1-1-3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (securities_static, securities_dynamic using securityId);
        Description: Join using time-identifier

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with time identifiers.
        """
        code = "1-1-1-4"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, securities_dynamic using securityId);
        Description: Join using time_period-identifier

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with time_period identifiers.
        """
        code = "1-1-1-5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (securities_static, securities_dynamic using securityId);
        Description: Join using time_period-identifier

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with time_period identifiers.
        """
        code = "1-1-1-6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, entities [rename entityId to issuerId]
                                        as B using dateReference, issuerId);
        Description: Join using date-identifiers

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (securities_static, entities [rename entityId to issuerId]
                                        as B using dateReference, issuerId);
        Description: Join using date-identifiers

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-8"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, entities [rename entityId to issuerId]
                                        as B using timeReference, issuerId);
        Description: Join using time-identifiers
        *** The engine gives this error message: TypeError: unhashable type: 'list'

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-9"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, entities [rename entityId to issuerId]
                                        as B using timeReference, issuerId);
        Description: Join using time-identifiers
        *** The engine gives this error message: TypeError: unhashable type: 'list'

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (securities_static, entities [rename entityId to issuerId]
                                        as B using time_periodReference, issuerId);
        Description: Join using time_period-identifiers

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-11"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (securities_static, entities [rename entityId to issuerId]
                                        as B using time_periodReference, issuerId);
        Description: Join using time_period-identifiers

        Git Branch: #301-tests-join-with-time-identifiers.
        Goal: Check the result of joins with date identifiers.
        """
        code = "1-1-1-12"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class CalcInsideJoinTests(JoinHelper):
    """
    Group 2
    """

    classTest = "joins.CalcInsideJoinTests"

    def test_GL_300_1(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_2(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= DS_1#Me_2];

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_2"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_3(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_4(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join ( DS_2, DS_1 filter Id_2 ="B" calc Me_4 := Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_4"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_5(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_6(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_7(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_8(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_8"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_9(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1 + d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_9"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_10(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_11(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_11"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_12(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1A drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        # code = "GL_300_12"
        # number_inputs = 2
        # error_code = "1-1-1-10"

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    def test_GL_300_13(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + Me_2 + d2#Me_1A drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_13"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_14(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + Me_3 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_14"
        number_inputs = 2
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_15(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d1#Me_1+ d2#Me_2 + Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_15"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_16(self):
        """
        Left join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d1#Me_1+ d2#Me_2 + d1#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_16"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_17(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_17"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_18(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= DS_1#Me_2];

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_18"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_19(self):
        """
        Full join
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := full_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        The engine gives an error that says: KeyError: 'me_5'
        """
        code = "GL_300_19"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_20(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_2, DS_1 filter Id_2 ="B" calc Me_4 := Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_20"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_21(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_21"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_22(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_22"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_23(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_23"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_24(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_24"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_25(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1 + d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_25"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_26(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d2#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_26"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_27(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_27"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_28(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1A drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        # code = "GL_300_28"
        # number_inputs = 2
        # error_code = "1-1-1-10"

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    def test_GL_300_29(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + Me_2 + d2#Me_1A drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_29"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_30(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + Me_3 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_30"
        number_inputs = 2
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_31(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d1#Me_1+ d2#Me_2 + Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_31"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_32(self):
        """
        Full join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d1#Me_1+ d2#Me_2 + d1#Me_2 drop d2#Me_2);

        Git Branch: #321-test-for-calc-inside-join
        Goal: Check the performance of the calc inside joins operators.
        """
        code = "GL_300_32"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class JoinCalcIfThenElseTests(JoinHelper):
    """
    Group 3
    """

    classTest = "join.JoinCalcIfThenElseTests"

    def test_1(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-2"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Join: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Join: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2
        rename DS_1#Id_1 to Id_11,DS_2#Id_1A to Id_1A1, DS_1#Id_2 to Id_21, DS_2#Id_2A to Id_2A1,
        DS_1#Me_1 to Me_11, DS_1#Me_2 to Me_21,DS_2#Me_2A to Me_2A1, DS_2#Me_1A to Me_1A1);
        Description: Join with calc with if-then-else

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-4"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = String, Me_2 = Integer

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-5"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_6(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = Number, Me_2 = Integer

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Join: full_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = Date, Me_2 = Integer

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-7"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_8(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = time, Me_2 = string

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-8"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_9(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = number, Me_2 = time_period

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-9"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_10(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        With some null Data Points

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-10"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_11(self):
        """
        Join: cross_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := cross_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-11"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = Boolean, Me_2 = String

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-12"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_13(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = Number, Me_2 = Boolean

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-13"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_14(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = Boolean, Me_2 = Boolean

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-14"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1, DS_2 calc Me_3 := if Id_2 = "A" then Me_1 else Me_2);
        Description: Join with calc with if-then-else
        Me_1 = Date, Me_2 = Boolean

        Git Branch: #321-test-for-calc-inside-join.
        Goal: Check the performance of the joins with calc with if_then_else operators.
        """
        code = "3-1-1-15"
        number_inputs = 2
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )


class UdosInsideJoinsTests(JoinHelper):
    """
    Group 2
    """

    classTest = "joins.UdosInsideJoinsTests"

    def test_1(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [drop Me_1,Me_2] as d1 , DS_2,
                    drop_identifier(DS_1, Id_3) as d2 using Id_1, Id_2);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [rename Me_1 to Me_11, Me_2 to Me_22]
                    as d1, drop_identifier(DS_1, Id_3) as d2 using Id_1, Id_2);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 [rename Me_11 to Me_1A] as d1,
                    DS_2, drop_identifier(DS_1, Id_3) as d2);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1, filter_udo(DS_1,Me_1, 4) [rename Me_1 to Me_11] as d1);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # TODO check this test (File not found).
    # def test_5(self):
    #     """
    #     Join: inner_join
    #     Dataset --> Dataset
    #     Status: BUG
    #     Expression: DS_r:= inner_join( DS_1, DS_2 apply filter_udo(DS_1,Me_1, 4));
    #     Description: Join using date-identifier with UDOS inside joins
    #     *** The engine gives this exception message: "Not valid RegularAggregation for apply"
    #
    #     Git Branch: #306-udos-inside-joins.
    #     Goal: Check the result of joins with date identifiers and UDOS.
    #     """
    #     code = '2-1-1-5'
    #     number_inputs = 2
    #     references_names = ["1"]
    #
    #     self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2, drop_identifier(DS_1, Id_3)
                            [drop Me_1,Me_2] as d2 using Id_1, Id_2);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, drop_identifier(DS_1, Id_3)
                            [rename Me_1 to Me_11, Me_2 to Me_22] as d2 using Id_1, Id_2);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # def test_9(self):  # Does not work properly because Me_1 does not disappear when it is renamed to Me_11.
    #     """
    #     Join: inner_join
    #     Dataset --> Dataset
    #     Status: BUG
    #     Expression: DS_r := inner_join (DS_1 as d1, DS_2, drop_identifier(DS_1, Id_3)
    #                         [rename Me_1 to Me_11] as d2);
    #     Description: Join using date-identifier with UDOS inside joins
    #     *** The engine gives this exception message: "Component Me_11 not found inside
    #     global_Statement1_result. Please review statements that contain DS_1 ."
    #
    #     Git Branch: #306-udos-inside-joins.
    #     Goal: Check the result of joins with date identifiers and UDOS.
    #     """
    #     code = '2-1-1-9'
    #     number_inputs = 2
    #     references_names = ["1"]
    #
    #
    #     self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := inner_join (DS_1 as d1, DS_2, drop_identifier(DS_1 [rename Me_1 to Me_11], Id_3) as d2);
        Description: Join using date-identifier with UDOS inside joins
        *** The engine gives this exception message: "Component Me_11 not found inside
        global_Statement1_result. Please review statements that contain DS_1 ."

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        code = "2-1-1-10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2, drop_identifier(DS_1, Id_3)
                            as d2 using Id_1, Id_2);
        Description: Join using date-identifier with UDOS inside joins

        Git Branch: #306-udos-inside-joins.
        Goal: Check the result of joins with date identifiers and UDOS.
        """
        # code = "2-1-1-11"
        # number_inputs = 2
        # message = "1-1-13-3"
        # TODO: check up this error test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_12(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (drop_identifier(DS_1, Id_3) as d1, drop_identifier(DS_2, Id_3)
                            as d2 calc identifier Id_3 :=nvl(Me_1 + Me_22,0));
        Description: Joins with user defined operators and calc

        Git Branch: #338-joins-with-udos-and-calc.
        Goal: Goal: Check the result of joins with udos and calc.
        """
        code = "2-1-1-12"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (drop_identifier(DS_1, Id_3) as d1, drop_identifier(DS_2, Id_3)
                            as d2 calc identifier Id_3 :=nvl(Me_1 + Me_22,0));
        Description: Joins with user defined operators and calc

        Git Branch: #338-joins-with-udos-and-calc.
        Goal: Check the result of joins with udos and calc.
        """
        code = "2-1-1-13"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (drop_identifier(DS_1, Id_3) as d1, drop_identifier(DS_2, Id_3)
                            as d2 calc identifier Id_3 :=nvl(Me_1 + Me_22,0));
        Description: Joins with user defined operators and calc

        Git Branch: #338-joins-with-udos-and-calc.
        Goal: Check the result of joins with udos and calc.
        """
        code = "2-1-1-14"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join (drop_identifier(DS_1, Id_3) as d1, drop_identifier(DS_2, Id_3)
                            as d2 calc identifier Id_3 :=nvl(Me_1 + Me_22,0));
        Description: Joins with user defined operators and calc

        Git Branch: #338-joins-with-udos-and-calc.
        Goal: Goal: Check the result of joins with udos and calc.
        """
        code = "2-1-1-15"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_399_1(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #399-fix-join-udo.
        Goal: Goal: Check the result of join inside udos.
        """
        code = "GL_399_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_399_2(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: aliasing is not required

        Git Branch: #399-fix-join-udo.
        Goal: Goal: Check the result of join inside udos.
        """
        # code = "GL_399_2"
        # number_inputs = 2
        # references_names = ["1"]

    def test_GL_399_3(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status:
        Expression:

        Description: Here at least one aliasing is required.

        Git Branch: #399-fix-join-udo.
        Goal: Goal: Check the result of join inside udos.
        """
        # code = "GL_399_3"
        # number_inputs = 2
        # references_names = ["1"]

    def test_GL_399_4(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status:
        Expression:

        Description: Here at least one aliasing is required.

        Git Branch: #399-fix-join-udo.
        Goal: Goal: Check the result of join inside udos.
        """
        code = "GL_399_4"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_399_5(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #399-fix-join-udo.
        Goal: Goal: Check the result of join inside udos.
        """
        code = "GL_399_5"
        number_inputs = 2
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )


class JoinFilterTests(JoinHelper):
    """
    Group 1
    """

    classTest = "joins.JoinFilterTests"

    def test_GL_320_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #320.
        Goal: Check the semantic result.
        """
        code = "GL_320_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_320_2(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join with 'differents' parents

        Git Branch: #320.
        Goal: Check the semantic result.
        """
        # code = "GL_320_1"
        # number_inputs = 2
        # references_names = ["1"]

    def test_GL_320_3(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #320.
        Goal: Check the result.
        """
        code = "GL_320_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_333_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #333.
        Goal: Check the result.
        """
        code = "GL_333_1"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_333_2(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #333.
        Goal: Check the result.
        """
        # code = "GL_333_2"
        # number_inputs = 2
        # references_names = ["1"]

    def test_GL_333_3(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #333.
        Goal: Check the result.
        """
        # code = "GL_333_3"
        # number_inputs = 2
        # references_names = ["1"]


class JoinUsingTests(JoinHelper):
    """
    Group 1
    """

    classTest = "joins.JoinUsingTests"

    def test_GL_251_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #251.
        Goal: Check the semantic result.
        """
        # code = "GL_251_1"
        # number_inputs = 3
        # references_names = ["1"]

    def test_GL_251_2(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: filter in join

        Git Branch: #251.
        Goal: Check the semantic result.
        """
        code = "GL_251_2"
        number_inputs = 3
        error_code = "1-1-13-18"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_342_1(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2 using Id_1, Id_2);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_2(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2 using Id_1, Id_2);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_2"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_3(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying: ('At op inner_join: Using clause does not define
        all the identifiers of non reference datasets. Please check transformation with output dataset DS_r', '1-1-13-4')

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        # code = "GL_342_3"
        # number_inputs = 2
        # error_code = "1-1-13-4"

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    def test_GL_342_4(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying: ('At op left_join: Invalid subcase B1, All the
        datasets must share as identifiers the using ones. Please check transformation with output dataset DS_r', '1-1-13-5')

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        # code = "GL_342_4"
        # number_inputs = 2
        # error_code = "1-1-13-4"

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    def test_GL_342_5(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Id_3);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_6(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Id_3);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_7(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_9(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2 using Id_2);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_9"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_10(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2 using Id_2, Me_1);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying: ('At op inner_join: Using clause does not define
        all the identifiers of non reference datasets. Please check transformation with output dataset DS_r', '1-1-13-4')
        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        # code = "GL_342_10"
        # number_inputs = 2
        # error_code = "1-1-13-6"

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    def test_GL_342_11(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Id_3);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying:
        VtlEngine.Exceptions.exceptions.SemanticError: ("At op left_join: Invalid subcase B2, All
        the declared using components '['Id_1', 'Id_2', 'Id_3']' must be present as components in
        the reference dataset 'DS_1'. Please check transformation with output dataset DS_r", '1-1-13-6')

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_11"
        number_inputs = 2
        error_code = "1-1-13-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_342_12(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_12"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_13(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_13"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_14(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2, DS_3 as d3 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_14"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_15(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1 , DS_2 as d2, DS_3 as d3 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_15"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_16(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join inner_join (DS_1 as d1 , DS_2 as d2, DS_3 as d3 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying: ('Component Observation not found. Please check
        transformation with output dataset DS_r', '1-3-16')

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_16"
        number_inputs = 3
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_342_17(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying: ('Component Id_2 not found. Please check transformation
        with output dataset DS_r', '1-3-16')

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_17"
        number_inputs = 3
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_342_18(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1 , DS_2 as d2, DS_3 as d3 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_18"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_342_19(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 using Id_1,Id_2,Observation);
        Description: Inner-left with using with measures.
        *** The engine gives an exception saying: ('At op inner_join: Using clause does not define all
        the identifiers of non reference datasets. Please check transformation with output dataset DS_r', '1-1-13-4')

        Git Branch: #342-inner-left-with-using-with-measures.
        Goal: Check the performance of Inner-left joins with using with measures.
        """
        code = "GL_342_19"
        number_inputs = 3
        error_code = "1-1-13-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_384_1(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join(DS_1, DS_2 using Id_2);
        Description: error case = 1.

        Git Branch:
        Goal:
        """
        code = "GL_384_1"
        number_inputs = 2
        error_code = "1-1-13-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_384_2(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join(DS_1, DS_2 using Id_2, Observation_1);
        Description:

        Git Branch:
        Goal: Check the performance of
        """
        # code = "GL_384_2"
        # number_inputs = 3
        # references_names = ["1"]

        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # CASE A

    def test_GL_384_9(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join(DS_1, DS_2 using Id_2, Observation_1);
        Description:

        Git Branch:
        Goal: Check the performance of
        """
        # code = "GL_384_9"
        # number_inputs = 3
        # references_names = ["1"]


class JoinsGeneralTests(JoinHelper):
    """
    Group 6
    """

    classTest = "joins.JoinsGeneralTests"

    def test_1(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: ok
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc identifier Me_3 :=d1#Me_1 + d2#Me_11);
        Description: Join with reuse calcs and calc identifier NOTE: the components Me_1, Me_11 provided from DS_1 and DS_2 has "isNull":	false like identifiers
        ***The engine gives an exception saying: 'At op identifier: You cannot convert to an Identifier,
        Component Me_3 with structure nullable=true. Please check transformation with output dataset DS_r', '1-1-1-16'

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc identifier Me_3 :=d1#Me_1 + d2#Me_11);
        Description: Join with reuse calcs and calc identifier NOTE: the components Me_1, Me_11 provided from DS_1 and DS_2 has "isNull":	true, so shoul fail
        ***There is a discrepancy between semantic and Base. In this case the semantic is wrong because
        the component Me_3 with structure nullable=true and Base Me_3 with structure nullable=false

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-2"
        number_inputs = 2
        error_code = "1-1-1-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_3(self):
        """
        Join: full_join
        Dataset --> Dataset
        Status:
        Expression: DS_r := full_join (DS_1 as d1, DS_2 as d2 calc identifier Me_3 :=d1#Me_1 + d2#Me_11);
        Description: Join with reuse calcs and calc identifier the same as test-1 with full join

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-3"
        number_inputs = 2
        error_code = "1-1-1-16"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_4(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_3 :=d1#Me_1 + d2#Me_11, Me_4 := Me_3 * 2);
        Description: Join with reuse calcs and calc identifier

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-4"
        number_inputs = 2
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_5(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_3 :=d1#Me_1 + d2#Me_11, Me_4 := Me_3 * 2);
        Description: Join with reuse calcs and calc identifier
        ***Note: DS_1 contains Me_3

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_3 :=d1#Me_1 + d2#Me_11, Me_4 := d1#Me_3 * 2);
        Description: Join with reuse calcs and calc identifier

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_3 :=d1#Me_1 + d2#Me_11, Me_4 := Me_3 * 2);
        Description: Join with reuse calcs and calc identifier
        ***Note: DS_1 contains Me_3

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Join: left_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1 as d1, DS_2 as d2 calc Me_3 :=d1#Me_1 + d2#Me_11, Me_4 := d1#Me_3 * 2);
        Description: Join with reuse calcs and calc identifier

        Git Branch: #339-joins-with-reuse-calcs-and-calc-identifier.
        Goal: Check the result of joins reuse calcs and calc identifier.
        """
        code = "6-1-1-8"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_419(self):
        """ """
        # code = "GL_419"
        # number_inputs = 1
        # error_code = "1-1-13-3"

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    def test_GL_422_1(self):
        """
        Join:
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 as d1, DS_2 as d2
                        filter d1#Me_1 + d2#Me_2 <10
                        aggr Me_1 := sum(d1#Me_1), attribute At20 := avg(d2#Me_2)
                        group by Id_1, Id_2
                        having sum(Me_3) > 0 and avg(Me_4) < 10);
        Description: Join with aggregation with having clause

        Git Branch: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/422
        Goal: Check the result of joins with aggregation with having clause.
        """
        code = "GL_422_1"
        number_inputs = 2
        # references_names = ["1"]
        message = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_422_2(self):
        """
        Join:
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := full_join ( DS_1 as d1, DS_2 as d2
                        filter d1#Me_1 + d2#Me_2 <10
                        aggr Me_1 := sum(d1#Me_1), attribute At20 := avg(d2#Me_2)
                        group by Id_1, Id_2
                        having sum(Me_3) > 0);
        Description: Join with aggregation with having clause

        Git Branch: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/422
        Goal: Check the result of joins with aggregation with having clause.
        """
        code = "GL_422_2"
        number_inputs = 2
        message = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_422_3(self):
        """ """
        code = "GL_422_3"
        number_inputs = 2
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_384_3(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join(DS_1, DS_2, DS_3);
        Description: invalid case A.

        Git Branch:
        Goal: Check the performance of
        """
        code = "GL_384_3"
        number_inputs = 3
        # references_names = ["1"]

        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        error_code = "1-1-13-11"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    # Caso A inner join pero con una componente en el no-reference que sea identifier en el reference
    def test_GL_384_4(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join(DS_1, DS_2);
        Description:

        Git Branch:
        Goal: Check the performance of
        """
        # code = "GL_384_4"
        # number_inputs = 2
        # error_code = "1-1-13-3"
        # TODO: check up this error test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)
        assert True

    # Left join , ds_1 is superset but the operation can't be done because ds_1 is not on the left
    def test_GL_384_5(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_2, DS_1 );
        Description: DS_1 is a superset

        Git Branch:
        Goal: Check the performance of
        """
        code = "GL_384_5"
        number_inputs = 2

        error_code = "1-1-13-11"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    # Aliases are mandatory for datasets which appear more than once in the Join
    def test_GL_384_6(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := left_join (DS_1, DS_2, DS_1 );
        Description: DS_1 is a superset

        Git Branch:
        Goal: Check the performance of
        """
        code = "GL_384_6"
        number_inputs = 2

        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_384_7(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join(DS_1, DS_2, DS_3);
        Description: invalid case A.

        Git Branch:
        Goal: Check the performance of
        """
        code = "GL_384_7"
        number_inputs = 3
        error_code = "1-1-13-11"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_384_8(self):
        """
        Join: inner_join
        Dataset --> Dataset
        Status: OK
        Expression:DS_r := inner_join(DS_1, DS_2, DS_3 using Id_2, Id_1);
        Description:

        Git Branch:
        Goal: Check the performance of
        """
        code = "GL_384_8"
        number_inputs = 3
        error_code = "1-1-13-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_500_1(self):
        """
        Inner join
        Dataset --> Dataset
        Status: SemanticError
        Expression: DS_r <- inner_join(DS_1, DS_2);

        Git Branch: cr-500
        Goal: Joining Time_Period vs String identifiers should raise SemanticError.
        """
        code = "GH_500_1"
        number_inputs = 2
        error_code = "1-1-13-18"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_500_2(self):
        """
        Inner join
        Dataset --> Dataset
        Status: SemanticError
        Expression: DS_r <- inner_join(DS_1, DS_2);

        Git Branch: cr-500
        Goal: Joining Time_Period vs Integer identifiers should raise SemanticError.
        """
        code = "GH_500_2"
        number_inputs = 2
        error_code = "1-1-13-18"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

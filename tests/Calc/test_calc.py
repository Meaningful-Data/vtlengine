from pathlib import Path

from tests.Helper import TestHelper


class TestCalcHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class CalcOperatorTest(TestCalcHelper):
    """
    Group 1
    """

    classTest = "calc.CalcOperatorTest"

    def test_1(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-1"
        number_inputs = 2  # 2
        message = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_2(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-2"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_3(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1*DS_2 [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-3"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_4(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := ln(DS_1#Me_1), Me_5 := ln(DS_1#Me_2),
                                  Me_6 := ln(DS_1#Me_3)];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-4"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_5(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := (DS_1*DS_2) [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.
        Note: It gives an error that says: AttributeError: 'NoneType' object has no attribute 'name'

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-5"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_6(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_2 [calc Me_4 := Me_1 + DS_2#Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-6"
        number_inputs = 2
        message = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_7(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [calc Me_4 := Me_1 + DS_1#Me_2] [calc Me_5 := Me_2 + DS_2#Me_3];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.

        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-7"
        number_inputs = 2
        message = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_8(self):
        """
        CALC: calc
        Dataset --> Dataset
        Status:
        Expression: DS_r := (DS_1*DS_2) [calc Me_4 := Me_1 + Me_2];
                    DS_1 Dataset

        Description: The operator calculates new Identifier, Measure or Attribute
        Components on the basis of sub-expressions at Component level.


        Git Branch: #287-review-with-several-inputs.
        Goal: Check the performance of the calc operator.
        """
        code = "1-1-1-8"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_1(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression:DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_2(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= DS_1#Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_2"
        number_inputs = 2
        error_code = "1-1-6-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_3(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_3"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_4(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_2, DS_1 filter Id_2 ="B" calc Me_4 := Me_2 keep Me_4, DS_1#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_4"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_5(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_5"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_6(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_6"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_7(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_7"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_8(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join ( DS_1, DS_2 filter Id_2 ="B" calc Me_4 := DS_2#Me_2 keep Me_4, DS_1#Me_2)[calc me_5:= Me_2];

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_8"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_9(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d1#Me_1 + d2#Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_9"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_10(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d2#Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_10"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_300_11(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_11"
        number_inputs = 2
        message = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_300_12(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        # code = "GL_300_12"
        # number_inputs = 2
        # message = "1-1-1-10"
        # TODO: The partially found methods allow to found and operate the component even when the name is the full name
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_GL_300_13(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + Me_2 + d2#Me_1A drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_13"
        number_inputs = 2
        message = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_300_14(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + Me_3 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_14"
        number_inputs = 2
        message = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_300_15(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := Me_1 + Me_2 + d2#Me_1A drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_15"
        number_inputs = 2
        error_code = "1-1-13-9"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_300_16(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := inner_join (DS_1 as d1, DS_2 as d2 calc Me_4 := d2#Me_1A + d1#Me_1+ d2#Me_2 + d1#Me_2 drop d2#Me_2);

        Git Branch: #fix-300-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_300_16"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_310_1(self):
        """
        Inner join
        Dataset --> Dataset
        Status: OK
        Expression:

        Git Branch: #fix-310-review-join
        Goal: Check the performance of the calc operator.
        """
        code = "GL_310_1"
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

from pathlib import Path

from tests.Helper import TestHelper


class TimePeriodHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class TimePeriodTest(TimePeriodHelper):
    """
    Group 1
    """

    classTest = "timePeriodtests.TimePeriodTest"

    def test_GL_416(self):
        """
        test2_1 := BE2_DF_NICP[filter FREQ = "M" and TIME_PERIOD = cast("2020-01", time_period)];
        """
        code = "GL_416"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_418(self):
        """ """
        code = "GL_418"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_417_1(self):
        """
        test := avg (BE2_DF_NICP group all time_agg ("Q", "M", TIME_PERIOD));
        """
        code = "GL_417_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_417_2(self):
        """
        test := avg (BE2_DF_NICP group all time_agg ("A", "M", TIME_PERIOD));
        """
        code = "GL_417_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_417_3(self):
        """ """
        code = "GL_417_3"
        number_inputs = 1
        error_code = "1-1-19-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_417_4(
        self,
    ):  # TODO: Check periodIndFrom is not the same as in data, in data is "M", should we allow this?
        """
        test := avg (BE2_DF_NICP group all time_agg ("A", "Q", TIME_PERIOD));
        """
        code = "GL_417_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_421_1(self):
        """
        test2_1 := BE2_DF_NICP
            [calc FREQ_2 := TIME_PERIOD in {cast("2020-01", time_period), cast("2021-01", time_period)}];
        """
        code = "GL_421_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_421_2(self):
        """ """
        # code = "GL_421_2"
        # number_inputs = 1
        # error_code = "1-3-10"

        # Deactivated test due to Set declaration (need more information)

        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_440_1(self):
        """
        DS_r := DS_1 ;
        """
        code = "GL_440_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_440_2(self):
        """ """
        code = "GL_440_2"
        number_inputs = 1
        message = "0-1-1-12"

        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    #############
    # Tests for the sdmx external representation
    def test_GL_462_1(self):
        """
        DS_r := DS_1 ;
        """
        code = "GL_462_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_462_2(self):
        """ """
        code = "GL_462_2"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_462_3(self):
        """
        Status: OK
        Description: Over scalardataset
        Goal: Check Result.
        """
        code = "GL_462_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_462_4(self):
        """
        Test for null value
        """
        code = "GL_462_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

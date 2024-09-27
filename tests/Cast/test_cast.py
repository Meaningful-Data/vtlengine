from pathlib import Path

from tests.Helper import TestHelper


class CastHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class CastExplicitWithoutMask(CastHelper):
    """

    """

    classTest = 'Cast.CastExplicitWithoutMask'

    def test_GL_461_1(self):
        code = 'GL_461_1'
        number_inputs = 1

        error_code = "1-1-5-5"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs,
                                      exception_code=error_code)

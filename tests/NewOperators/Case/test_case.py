from pathlib import Path

from tests.Helper import TestHelper


class TestCase(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class CaseTest(TestCase):
    """
    Group 1
    """

    classTest = "case.CaseTest"

    def test_1(self):
        """
        """
        code = "1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):

        code = "2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        """
        code = "3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        """
        code = "4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        """
        code = "5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


from pathlib import Path

from tests.Helper import TestHelper

"""
Add to vs code User settings:

    {"python.unitTest.unittestEnabled": true,
    "python.unitTest.pyTestEnabled": false,
    "python.unitTest.nosetestsEnabled": false,
    }

"""

"""
Improvements:

"""
"""
TODOS:
    [1]: Add test with Datasets and Componets with null values for Numeric, Comparison and Boolean Operators.
"""


# Path Selection.---------------------------------------------------------------


class ThreeValueHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"

    ds_input_prefix = "DS_"


class ThreeValueTests(ThreeValueHelper):
    """
    Group 1
    """

    classTest = "3VL.AndTest"

    maxDiff = None

    def test_1(self):
        """
        And logic Test
        """
        text = """DS_r := DS_1[calc Me_3 := Me_1 and Me_2];"""
        code = "1"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_2(self):
        """
        Component-Scalar Test true
        """
        text = """DS_r := DS_1[calc Me_3 := Me_1 and true];"""
        code = "2"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_3(self):
        """
        Component-Scalar Test false
        """
        text = """DS_r := DS_1[calc Me_3 := Me_1 and false];"""
        code = "3"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_4(self):
        """
        Dataset-scalar Test true
        """
        text = """DS_r := DS_1 and true;"""
        code = "4"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_5(self):
        """
        Dataset-scalar Test true
        """
        text = """DS_r := DS_1 and false;"""
        code = "5"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_6(self):
        """
        Or logic test
        """
        text = """DS_r := DS_1[calc Me_3 := Me_1 or Me_2];"""
        code = "6"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

    def test_7(self):
        """
        Xor logic test
        """
        text = """DS_r := DS_1[calc Me_3 := Me_1 xor Me_2];"""
        code = "7"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(
            text=text, code=code, number_inputs=number_inputs, references_names=references_names
        )

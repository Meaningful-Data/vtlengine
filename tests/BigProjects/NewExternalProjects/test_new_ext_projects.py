from pathlib import Path

from tests.Helper import TestHelper


class ExternalProjectsHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class AnaVal(ExternalProjectsHelper):
    """ """

    classTest = "NewExternalProjects.AnaVal"

    def test_Monthly_validations_only_semantic(self):
        """
        Description: EEAS_OA30
        Git Branch: feat-test-projects
        Goal: AnaValMonthly with empty data successful execution
        """
        code = "AnaVal_Monthly_validations_1"
        number_inputs = 16
        vd_names = ["EU_countries", "AnaCreditCountries"]
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_Monthly_validations_1(self):
        """
        Description: EEAS_OA30
        Git Branch: feat-test-projects
        Goal: AnaValMonthly with empty data successful execution
        """
        code = "AnaVal_Monthly_validations_1"
        number_inputs = 16
        vd_names = ["EU_countries", "AnaCreditCountries"]
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_Monthly_validations_2(self):
        """
        Description: EEAS_OA26
        Git Branch: feat-test-projects
        Goal: AnaValMonthly execution with data.
        """
        code = "AnaVal_Monthly_validations_2"
        number_inputs = 16
        vd_names = ["EU_countries", "AnaCreditCountries"]
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=message,
            vd_names=vd_names,
        )

    def test_Quarterly_validations_only_semantic(self):
        """
        Description: EEAS_OA30
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = "AnaVal_Quarterly_validations_1"
        number_inputs = 11
        vd_names = ["EU_countries", "AnaCreditCountries"]
        rn = [str(i) for i in range(1, 24)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
            only_semantic=True,
        )

    def test_Quarterly_validations_1(self):
        """
        Description: EEAS_OA30
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = "AnaVal_Quarterly_validations_1"
        number_inputs = 11
        vd_names = ["EU_countries", "AnaCreditCountries"]
        rn = [str(i) for i in range(1, 24)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )

    def test_Quarterly_validations_2(self):
        """
        Description: EEAS_OA26
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = "AnaVal_Quarterly_validations_2"
        number_inputs = 11
        vd_names = ["EU_countries", "AnaCreditCountries"]
        rn = [str(i) for i in range(1, 24)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
        )

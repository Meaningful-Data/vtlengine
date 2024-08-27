from pathlib import Path

from testSuite.Helper import TestHelper


class ExternalProjectsHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class BOP(ExternalProjectsHelper):
    """

    """

    classTest = 'ExternalProjects.BOP'

    def test_BOP_Q_Review_1(self):
        """
        Description:
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'BOP_Q_Review_1'
        number_inputs = 1
        rn = [str(i) for i in range(1, 30)]
        references_names = rn

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AnaVal(ExternalProjectsHelper):
    """

    """

    classTest = 'ExternalProjects.AnaVal'

    def test_Monthly_validations_1(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaVal_Monthly_validations_1'
        number_inputs = 36
        vd_names = ["EU_countries", "AnaCreditCountries_1"]
        rn = [str(i) for i in range(1, 288)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names
        )

    def test_Quarterly_validations_1(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaVal_Quarterly_validations_1'
        number_inputs = 12
        vd_names = ["EU_countries", "AnaCreditCountries_1"]
        rn = [str(i) for i in range(1, 38)]
        references_names = rn

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names
        )

    def test_GL_283_1(self):
        """
        USER DEFINED OPERATORS
        Status: OK
        Description:
        Git Branch: #283
        """
        code = 'GL_283_1'
        number_inputs = 36
        vd_names = ["EU_countries", "AnaCreditCountries_1"]
        rn = [str(i) for i in range(1, 129)]
        references_names = rn

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names, vd_names=vd_names)


class AnaMart(ExternalProjectsHelper):
    """

    """

    classTest = 'ExternalProjects.AnaMart'

    def test_AnaMart_AnaMart_1(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaMart_AnaMart_1'
        number_inputs = 30
        vd_names = ["anaCreditCountries_2"]
        # rn = [str(i) for i in range(1, 303)]
        rn = [str(i) for i in range(1, 30)]
        rn += [str(i) for i in range(72, 303)]
        references_names = rn
        sql_names = [
            "instDates",
            "instrFctJn",
            "instrFctJn2",
            "prtctnDts",
            "prtctnFctJn"
        ]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
            sql_names=sql_names
        )
        exception_code = "1-1-13-4"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_AnaMart_AnaMart_2(self):
        """
        Description: AT_Bank_201906
        Git Branch: feat-test-projects
        Goal: Check semantic result and interpreter results.
        """
        code = 'AnaMart_AnaMart_2'
        number_inputs = 30
        vd_names = ["anaCreditCountries_2"]
        # rn = [str(i) for i in range(1, 303)]
        rn = [str(i) for i in range(1, 30)]
        rn += [str(i) for i in range(72, 303)]
        references_names = rn
        sql_names = [
            "instDates",
            "instrFctJn",
            "instrFctJn2",
            "prtctnDts",
            "prtctnFctJn"
        ]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            vd_names=vd_names,
            sql_names=sql_names
        )

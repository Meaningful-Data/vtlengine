from pathlib import Path

from tests.Helper import TestHelper


class TestIdentifiersTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class IdentifiersTypeCheckingAdd(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = "Identifiers.IdentifiersTypeCheckingAdd"

    def test_1(self):
        """
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 + DS_2;
        Number: Number + Integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-3-1"
        # 4 For group numeric
        # 6 For group identifiers
        # 3 For add operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_2 + DS_1 ;
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-3-2"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
        string-integer
        Description: operations between identifiers, string and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-3-3"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        ADD OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 + DS_2 ;
        string-number
        Description: operations between identifiers, numbers and strings.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-3-4"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingSubstraction(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = "Identifiers.IdentifiersTypeCheckingSubstraction"

    # SUBSTRACTION OPERATOR

    def test_1(self):
        """
        SUBSTRACTION OPERATOR
        Status: OK
        Expression: DS_r := DS_1 - DS_2;
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-4-1"
        # 4 For group numeric
        # 6 For group identifiers
        # 4 For substraction operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingMultiplication(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = "Identifiers.IdentifiersTypeCheckingMultiplication"

    # MULTIPLICATION OPERATOR

    def test_1(self):
        """
        MULTIPLICATION OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 * DS_2;
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-5-1"
        # 4 For group numeric
        # 6 For group identifiers
        # 5 For multiplication operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingDivision(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = "Identifiers.IdentifiersTypeCheckingDivision"

    # DIVISION OPERATOR

    def test_1(self):
        """
        DIVISION OPERATOR
        Status: BUG
        Expression: DS_r := DS_1 / DS_2;
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-6-1"
        # 4 For group numeric
        # 6 For group identifiers
        # 6 For multiplication operator in numeric
        # 1 Number of test
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class IdentifiersTypeCheckingModule(TestIdentifiersTypeChecking):
    """
    Group 4
    """

    classTest = "Identifiers.IdentifiersTypeCheckingModule"

    # MOD OPERATOR

    def test_1(self):
        """
        MOD OPERATOR
        Status: BUG
        Expression: DS_r := mod ( DS_1, DS_2 );
        number - integer
        Description: operations between identifiers, numbers and integers.
        Jira issue: VTLEN 566.
        Git Branch: feat-VTLEN-566-Type-checking-for-identifiers-Numeric.
        Goal: Check Doubt.
        """
        code = "4-6-7-1"
        number_inputs = 2
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

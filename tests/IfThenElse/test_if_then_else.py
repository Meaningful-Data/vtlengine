from pathlib import Path

from tests.Helper import TestHelper


class TestIfThenElse(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class IfThenElseTest(TestIfThenElse):
    """
    Group 1
    """

    classTest = "if_then_else.IfThenElseTest"

    def test_1(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then amortisedCost else marketValue];
        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then amortisedCost else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then amortisedCost else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Note: There is a discrepancy between base and semantic with respect
        to isNull of the measure (carryingAmount).
        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then amortisedCost else null];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then null else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = 0
                                        then amortisedCost else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = null
                                        then amortisedCost else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then 0 else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then amortisedCost else 0];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then currentDate else marketValue];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-10"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_11(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if accountingClassification = "ac"
                                        then amortisedCost else currentDate];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-11"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_12(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc carryingAmount := if currentDate = "2021-08-25" then
                                        amortisedCost else currentDate];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-12"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_13(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if DsCond#Id_2 = "A" then DsThen else DsElse;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-13"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if DsCond#Id_2 = "A" then DsThen else DsElse;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Note: The Me_1 of DsThen is a string
        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-14"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if DsCond#Id_2 = "A" then null else DsElse;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-15"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if (DsCond#Id_2 = "A") = 0 then DsThen else DsElse;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-16"
        number_inputs = 3
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_17(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if (DsCond#Id_2 = "A") then DsThen else null;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-17"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if (DsCond#Id_2 = "A") then null else null;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        # code = "1-1-1-18"
        # number_inputs = 3
        # error_code = "1-1-9-12"
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=["DS_r"])

    def test_19(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if (DsCond#Id_2 = "A") then DsThen=0 else DsElse;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-19"
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if (DsCond#Id_2 = "A") then DsThen=0 else null;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-20"
        number_inputs = 3
        error_code = "1-1-9-13"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_21(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := if (DsCond#Id_2 = "A") then DsThen else DsElse=0;
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.

        Git Branch: #166-review-if-then-else-for-component-component.
        Goal: Check the result of if-then-else at Data Set level.
        """
        code = "1-1-1-21"
        number_inputs = 3
        error_code = "1-1-9-13"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_22(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := securities [calc Me :=
                    if accountingClassification = "ac" then
                        if securityId = 456 then 1 else marketValue
                    else if securityId = 123 then amortisedCost else 3 ];

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Git Branch: #404-if-then-else-refactor
        Goal: Check the result of if-then-else for component-component.
        """
        code = "1-1-1-22"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_424_1(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: INPUT_CHECK_REGEX :=
            if
                (match_characters(BIS_LOC_STATS # OBS_VALUE,r'[0-9]*[.,]?[0-9]*\\Z'))
            then
                length(BIS_LOC_STATS # OBS_VALUE) > 0 and length(BIS_LOC_STATS # OBS_VALUE) < 20
            else
                if(match_characters(BIS_LOC_STATS # OBS_VALUE,"/[^A-Za-z ]/g"))
                then
                    BIS_LOC_STATS # REP_COUNTRY in { "TR" }
                else
                BIS_LOC_STATS # REP_COUNTRY in { "US" }
            ;

        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Git Branch: #https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/424
        Goal: Check the result of if-then-else for dataset-dataset.
        """
        code = "GL_424_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_424_2(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: INPUT_CHECK_REGEX :=
            if(match_characters(BIS_LOC_STATS # OBS_VALUE,"/[^A-Za-z ]/g"))
            then
                BIS_LOC_STATS # REP_COUNTRY in { "TR" }
            else
            BIS_LOC_STATS # REP_COUNTRY in { "US" }
            ;
        Description: The if operator returns thenOperand if condition evaluates
                     to true, elseOperand otherwise.
        Git Branch: #https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/424
        Goal: Check the result of if-then-else for dataset-dataset.
        """
        code = "GL_424_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_436_1(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK

        Description: filter with if then for scalars inside a udo.
        Git Branch: #https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/436
        Goal: Check the result of if-then-else.
        """
        code = "GL_436_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_436_2(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK

        Description: calc with if then for scalars inside a udo.
        Git Branch: #https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/436
        Goal: Check the result of if-then-else.
        """
        code = "GL_436_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_436_3(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK

        Description: calc with if then for scalars
        Git Branch: #https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/436
        Goal: Check the result of if-then-else.
        """
        code = "GL_436_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

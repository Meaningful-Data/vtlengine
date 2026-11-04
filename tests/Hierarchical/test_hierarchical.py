import os
from pathlib import Path
from unittest import mock

from tests.Helper import TestHelper
from vtlengine.API import create_ast
from vtlengine.AST.ASTString import ASTString


class HierarchicalHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class HierarchicalRulsetOperatorsTest(HierarchicalHelper):
    """
    Group 1
    """

    classTest = "test_hierarchical.HierarchicalRulsetOperatorsTest"

    def test_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG /OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "1-1-1-1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "1-1-1-7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status:
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY partial_zero);
                                        BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-11"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-12"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # Other formulation
    def test_GL_240_1(self):
        pass

    def test_GL_265_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule COUNT_SECTOR) is
                        A = B + N + U errorcode "totalComparedToBanks" errorlevel 4;
                        A >= U errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(BIS_LOC_STATS, sectorsHierarchy) ;

        Description: check hierarchy without rule in the call.

        Git Branch: #265.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "GL_265_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_265_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule COUNT_SECTOR) is
                        A = B + N + U errorcode "totalComparedToBanks" errorlevel 4;
                        A >= U errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(BIS_LOC_STATS, sectorsHierarchy rule COUNT_SECTOR) ;

        Description: check hierarchy without rule in the call.

        Git Branch: #265.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "GL_265_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_265_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule COUNT_SECTOR_a) is
                        A = B + N + U errorcode "totalComparedToBanks" errorlevel 4;
                        A >= U errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(BIS_LOC_STATS, sectorsHierarchy) ;

        Description: check hierarchy without rule in the call.

        Git Branch: #265.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "GL_265_3"
        number_inputs = 1
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_265_4(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule COUNT_SECTOR_a) is
                        A = B + N + U errorcode "totalComparedToBanks" errorlevel 4;
                        A > U errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(BIS_LOC_STATS, sectorsHierarchy rule COUNT_SECTOR) ;

        Description: check hierarchy without rule in the call.

        Git Branch: #265.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "GL_265_4"
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_265_5(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule COUNT_SECTOR) is
                        A = B + N + U errorcode "totalComparedToBanks" errorlevel 4 ;
                        A >= U errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;
                    define hierarchical ruleset accountingEntryQ (variable rule BS_POSITION) is
                        L < C errorcode "Net (assets-liabilities) [Accounting_Entry rule 'N = A - L']" errorlevel 4
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(BIS_LOC_STATS, sectorsHierarchy) ;
                    sectors := check_hierarchy(BIS_LOC_STATS_2, accountingEntryQ) ;
                    sectors_hier_val_unf_1 := check_hierarchy(BIS_LOC_STATS_3, sectorsHierarchy always_zero) ;

        Description: check hierarchy without rule in the call.

        Git Branch: #265.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "GL_265_5"
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            non_null all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        **Note: The output does not return the variable Bool_Var (TRUE OR FALSE)

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "1-1-1-19"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-20"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            partial_null all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-21"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_zero);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-22"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            always_null invalid);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-23"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_zero);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-24"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            non_null all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "1-1-1-25"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            non_zero all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-26"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            partial_null all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-27"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            partial_zero invalid);
                                        BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-28"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            always_null all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-29"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            always_zero all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "1-1-1-30"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset hie1 (variable rule Id2) is
                        A=B+C   errorcode "error"   errorlevel 5
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(DS_1, hie1 rule Id2 partial_zero all);
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        """
        code = "1-1-1-31"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_275_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY
                            always_zero all_measures);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "GL_275_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    """
    - if variable rule in signature:
        - In invocation, rule not necessary
        - If rule, then name has to be equal to variable rule is siganture

    -if valuedomain rule in signature
        - Name of rule in inovcation can be equal or different
    """

    def test_GL_145_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule Id_1) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy rule Id_2 non_zero);
                                        BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check exception because If rule, then name has to be equal to variable rule is siganture
        """

        code = "GL_145_1"
        number_inputs = 1
        error_code = "1-1-10-3"  # wrong error code, 1-1-10-8?

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_145_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (valuedomain rule abstract) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy rule Id_2 non_zero);
                                        BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check Result
        """

        code = "GL_145_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_145_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (valuedomain rule abstract) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check exception, name in the call is mandatory for vd
        """

        code = "GL_145_3"
        number_inputs = 1
        error_code = "1-1-10-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_145_4(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (valuedomain rule Id_2) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy rule Id_2 non_zero);
                                        BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check Result, value domain and component have the same name.
        """

        code = "GL_145_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable condition Id_1 rule Id_2) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy condition Id_1 rule Id_2 non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:.
        """

        code = "GL_397_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Exception, condition signature and condition call are differents.
        """

        code = "GL_397_3"
        number_inputs = 1
        error_code = "1-1-10-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_397_5(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable condition Id_3, Id_1 rule Id_2) is
                        when Id_1 > 0 then B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        when Id_1 > 0 then N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := check_hierarchy(DS_1, sectorsHierarchy condition Id_3, Id_1 rule Id_2 non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:.
        """

        code = "GL_397_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_7(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Exception, condition signature and condition call are differents.
        """

        code = "GL_397_7"
        number_inputs = 1
        error_code = "1-1-10-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_397_9(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Neither condition nor rule is provided on a ruleset defined on variables.
        """

        code = "GL_397_9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_11(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Condition is provided but rule is not.
        """

        code = "GL_397_11"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_13(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_13"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_15(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Exception, the rule in the call is not the rule variable of the signature.
        """

        code = "GL_397_15"
        number_inputs = 1
        error_code = "1-1-10-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_397_17(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_17"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_18(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_18"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_23(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_23"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_24(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Condition and rule are not provided, so the ones of the signature are used.
        """

        code = "GL_397_24"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_25(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: value domain with alias.
        """

        code = "GL_397_25"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_27(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: several value domain with alias.
        """

        code = "GL_397_27"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_29(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: check the warning because different types in the rule.
        """

        code = "GL_397_29"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_397_30(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:
        """

        code = "GL_397_30"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_32(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:
        """

        code = "GL_397_32"
        number_inputs = 1
        references_names = ["1"]
        # with pytest.raises(Exception, match="cast .+? without providing a mask"):
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_34(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:
        """

        code = "GL_397_34"
        number_inputs = 1
        references_names = ["1"]

        # with pytest.raises(Exception, match="cast .+? without providing a mask"):
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_36(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:
        """

        code = "GL_397_36"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_566_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression: Check GL_566_1.vtl

        Git Branch: Cr-295.
        """

        code = "GL_566_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_598_1(self):
        """
        check_hierarchy with boolean errorcode and errorlevel constants.
        Dataset --> Dataset
        Status: OK

        define hierarchical ruleset hr_bool (variable rule Id_1) is
            B = C - D errorcode true errorlevel true;
            N = A - L errorcode false errorlevel false
        end hierarchical ruleset;

        DS_r <- check_hierarchy(DS_1, hr_bool rule Id_1);

        Git Branch: cr-596.
        Goal: Verify boolean constants work as errorcode/errorlevel in hierarchical
        rulesets, including AST round-trip.
        """
        code = "GH_598_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_598_1_roundtrip(self):
        """
        AST round-trip for check_hierarchy with boolean errorcode/errorlevel.

        Git Branch: cr-596.
        Goal: Verify that rendering the AST back to VTL and re-parsing produces the
        same result when boolean constants are used in errorcode/errorlevel.
        """
        code = "GH_598_1"
        number_inputs = 1
        references_names = ["1"]

        text = self.LoadVTL(code)
        rendered = ASTString().render(create_ast(text))
        self.BaseTest(
            code=code, number_inputs=number_inputs, references_names=references_names, text=rendered
        )

    def test_GH_1141_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_2 as C_2
                        rule VD_1) is
                        when C_2 = "X" then TOTAL = A + B
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_2 rule Id_1 all);

        Description: A rule whose condition is FALSE for Id_2 = "Y" gives no data point there.

        Git Branch: #1141.
        Goal: Check output all returns only the data points where the condition is TRUE.
        """
        code = "GH_1141_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1141_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_2 as C_2
                        rule VD_1) is
                        when C_2 = "X" then TOTAL = A + B
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_2 rule Id_1 all_measures);

        Description: A rule whose condition is FALSE for Id_2 = "Y" gives no data point there.

        Git Branch: #1141.
        Goal: Check output all_measures returns only the data points where the condition is TRUE.
        """
        code = "GH_1141_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class HierarchicalRollUpOperatorsTest(HierarchicalHelper):
    """
    Group 2
    """

    classTest = "test_hierarchical.HierarchicalRollUpOperatorsTest"

    def test_1(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: TODO: Question about the row that contains PT
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status:
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: BUG
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_null all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-13"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-14"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: TODO: Question should be an error?
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        N > A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        The result should be an empty datapoint

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-15"
        number_inputs = 1
        error_code = "1-1-10-5"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_16(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        One calculated component should be 0.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-16"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_null dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        One calculated component should be 0.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-17"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        One calculated component should be 0.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-18"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_null);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-19"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY partial_zero);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-20"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_null);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-21"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_zero);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-22"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N >= A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_null all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-23"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance(credit-debit)" errorlevel 4;
                        N >= A + L errorcode "Net(assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-24"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: TODO: Question should be an error?
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        N > A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY non_zero dataset);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        The result should be an empty datapoint

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-25"
        number_inputs = 1
        error_code = "1-1-10-5"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_26(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N > A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        One calculated component should be 0.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = "2-1-1-26"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_null all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        One calculated component should be 0.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-27"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.
        One calculated component should be 0.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-28"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY partial_null all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-29"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY partial_zero all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-30"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY always_null all);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-31"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A + L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule
                            ACCOUNTING_ENTRY always_zero);
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #186-test-for-hr.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = "2-1-1-32"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_145_5(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule Id_1) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy rule Id_2 non_zero);
                                        BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check exception because If rule, then name has to be equal to variable rule is siganture
        """

        code = "GL_145_5"
        number_inputs = 1
        error_code = "1-1-10-3"  # wrong error code 1-1-10-8?

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_145_6(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (valuedomain rule abstract) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy  rule Id_1 non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check Result when we use value domain
        """

        code = "GL_145_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_145_7(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (valuedomain rule abstract) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check exception, name in the call is mandatory for vd
        """

        code = "GL_145_7"
        number_inputs = 1
        error_code = "1-1-10-4"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_145_8(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule Id_2) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check Result
        """

        code = "GL_145_8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_145_9(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (valuedomain rule abstract) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy  rule Id_1 non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #145.
        Goal: Check Result when we use value domain( id_1 type: Date)
        """

        code = "GL_145_9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable condition Id_1 rule Id_2) is
                        B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy condition Id_1 rule Id_2 non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Result, value domain and component have the same name.
        """

        code = "GL_397_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_4(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Exception, condition signature and condition call are differents.
        """

        code = "GL_397_4"
        number_inputs = 1
        error_code = "1-1-10-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_397_6(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable condition Id_3, Id_1 rule Id_2) is
                        when Id_1 > 0 then B = C - D errorcode "totalComparedToBanks" errorlevel 4;
                        when Id_1 > 0 then N >  A + L errorcode "totalGeUnal" errorlevel 3
                    end hierarchical ruleset;

                    sectors_hier_val_unf := hierarchy(DS_1, sectorsHierarchy condition Id_3, Id_1 rule Id_2 non_zero);

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:.
        """

        code = "GL_397_6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_8(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Exception, condition signature and condition call are differents.
        """

        code = "GL_397_8"
        number_inputs = 1
        error_code = "1-1-10-6"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_397_10(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Neither condition nor rule is provided on a ruleset defined on variables.
        """

        code = "GL_397_10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_12(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Condition is provided but rule is not.
        """

        code = "GL_397_12"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_14(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_14"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(
            code=code,
            number_inputs=number_inputs,
            references_names=references_names,
            only_semantic=True,
        )

    def test_GL_397_16(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Check Exception, the rule in the call is not the rule variable of the signature.
        """

        code = "GL_397_16"
        number_inputs = 1
        error_code = "1-1-10-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_397_19(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_19"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_20(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_20"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_21(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.
        """

        code = "GL_397_21"
        number_inputs = 1
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_GL_397_22(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: Rule is provided but condition is not.  # TODO: Consider a warning for the second rule
        """

        code = "GL_397_22"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_26(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: value domain with alias.
        """

        code = "GL_397_26"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_28(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal: several value domain with alias.
        """

        code = "GL_397_28"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_31(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: EU is defined by four rules, one per period of a string condition.

        Description: A code item can be defined by several rules with different conditions
        (EuropeanUnionAreaCountries1 example of the VTL 2.2 Reference Manual).

        Git Branch: #397, #1142.
        Goal: Check each period takes EU from the rule whose condition holds.
        """

        code = "GL_397_31"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_33(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: EU is defined by four rules, one per period of a date condition.

        Description: A code item can be defined by several rules with different conditions
        (EuropeanUnionAreaCountries1 example of the VTL 2.2 Reference Manual).

        Git Branch: #397, #1142.
        Goal: Check each period takes EU from the rule whose condition holds.
        """

        code = "GL_397_33"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_35(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:
        """

        code = "GL_397_35"
        number_inputs = 1
        references_names = ["1"]

        # with pytest.raises(Exception, match="cast .+? without providing a mask"):
        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_37(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:
        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:
        """

        code = "GL_397_37"
        number_inputs = 1
        error_code = "1-1-1-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GL_463_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Uses SDMX-CSV 1.0

        Git Branch: #463.
        Goal:
        """

        code = "GL_463_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_494_5(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Test hyerarchical rule_priority

        Git Branch: #494.
        Goal:
        """

        code = "GL_494_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_567_1(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset hie1 (variable rule Id_2) is
                        E = A + F    errorcode "error"   errorlevel 5;
                        A = B + C    errorcode "error2"   errorlevel 5;
                        D = E + A   errorcode "error3"  errorlevel 5;
                        A >= B        errorcode "error4"  errorlevel 5
                    end hierarchical ruleset;

                    DS_r := hierarchy(DS_1, hie1 rule Id_2 all);

        Description: Hierarchy with mixed comparison operators (= and >=).
        Non-EQ rules are filtered out by the DAG during rule analysis.

        Git Branch: #567.
        Goal: Verify that hierarchy filters non-EQ rules and computes correctly.
        """

        code = "GH_567_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_567_2(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset hie1 (variable rule Id_2) is
                        A >= B        errorcode "error4"  errorlevel 5
                    end hierarchical ruleset;

                    DS_r := hierarchy(DS_1, hie1 rule Id_2 all);

        Description: Hierarchy with no EQ rules should raise SemanticError.

        Git Branch: #567.
        Goal: Verify that hierarchy raises error 1-1-10-5 when no rules have = operator.
        """

        code = "GH_567_2"
        number_inputs = 1
        error_code = "1-1-10-5"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_567_3(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset hie1 (variable rule Id_2) is
                        A >= B        errorcode "error4"  errorlevel 5
                    end hierarchical ruleset;

                    DS_r := hierarchy(DS_1, hie1 rule Id_2 all);

        Description: Hierarchy with no EQ rules should raise SemanticError.

        Git Branch: #567.
        Goal: Verify that hierarchy raises error 1-1-10-5 when no rules have = operator.
        """

        code = "GH_567_3"
        number_inputs = 1
        error_code = "1-1-10-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_567_4(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset hie1 (variable rule Id_2) is
                        A >= B        errorcode "error4"  errorlevel 5
                    end hierarchical ruleset;

                    DS_r := hierarchy(DS_1, hie1 rule Id_2 all);

        Description: Hierarchy with no EQ rules should raise SemanticError.

        Git Branch: #567.
        Goal: Verify that hierarchy raises error 1-1-10-5 when no rules have = operator.
        """

        code = "GH_567_4"
        number_inputs = 1
        error_code = "1-3-2-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_919_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset sectorsHierarchy (variable rule Id_2) is
                        A = B + N + U errorcode "totalComparedToBanks" errorlevel 4;
                        A >= U        errorcode "totalGeUnal"          errorlevel 3
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(DS_1, sectorsHierarchy rule Id_2 all);

        Description: Number comparisons in hierarchical rules honour the
        COMPARISON_ABSOLUTE_THRESHOLD tolerance, so residues below the
        significant-digits threshold are not reported as violations.

        Git Branch: #919.
        Goal: Verify that = and >= only flag imbalances above the tolerance.
        """

        code = "GH_919_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1140_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_1) is
                        C = A + B
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 partial_zero);

        Description: The pandas engine computes partial_zero rules when no right-side code item of C is found for Id_1 = 1.

        Git Branch: #1140.
        Goal: Check the result matches the DuckDB engine.
        """

        code = "GH_1140_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1140_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_1) is
                        C = A + B
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 partial_null);

        Description: The pandas engine computes partial_null rules when no right-side code item of C is found for Id_1 = 1.

        Git Branch: #1140.
        Goal: Check the result matches the DuckDB engine.
        """

        code = "GH_1140_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1140_3(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_1) is
                        C = A + B; E = C + D
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 partial_zero);

        Description: The pandas engine computes partial_zero rules when E takes C from the first rule, which gives no C for Id_1 = 3.

        Git Branch: #1140.
        Goal: Check the result matches the DuckDB engine.
        """

        code = "GH_1140_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1140_4(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_1) is
                        C = A + B; E = C + D
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 partial_null);

        Description: The pandas engine computes partial_null rules when E takes C from the first rule, which gives no C for Id_1 = 3.

        Git Branch: #1140.
        Goal: Check the result matches the DuckDB engine.
        """

        code = "GH_1140_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1151_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 > 1995 then EU = BE + FR errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 invalid);

        Description: On a ruleset defined on variables, the call can omit condition and rule.

        Git Branch: #1151.
        Goal: Check the call takes the condition and rule variables of the signature.
        """

        code = "GH_1151_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1151_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 > 1995 then EU = BE + FR
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1);

        Description: On a ruleset defined on variables, the call can omit condition and rule.

        Git Branch: #1151.
        Goal: Check the call takes the condition and rule variables of the signature.
        """

        code = "GH_1151_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1151_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 > 1995 then EU = BE + FR errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 invalid);

        Description: On a ruleset defined on variables, the call can omit condition only.

        Git Branch: #1151.
        Goal: Check the call takes the condition variables of the signature.
        """

        code = "GH_1151_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1151_4(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 rule VD_2) is
                        when VD_1 > 1995 then EU = BE + FR errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 invalid);

        Description: On a ruleset defined on value domains, the condition is mandatory.

        Git Branch: #1151.
        Goal: Check the call without condition still raises an error.
        """

        code = "GH_1151_4"
        number_inputs = 1
        error_code = "1-1-10-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_GH_1149_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1, Id_2 rule Id_3) is
                        when Id_1 > 1995 and Id_2 = "S1" then EU = BE + FR errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1, Id_2 rule Id_3 invalid);

        Description: A ruleset on variables accepts more than one condition variable.

        Git Branch: #1149.
        Goal: Check the rule is only validated where both conditions are true.
        """

        code = "GH_1149_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1149_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1, Id_2 rule Id_3) is
                        when Id_1 > 1995 and Id_2 = "S1" then EU = BE + FR
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1, Id_2 rule Id_3);

        Description: A ruleset on variables accepts more than one condition variable.

        Git Branch: #1149.
        Goal: Check the rule is only computed where both conditions are true.
        """

        code = "GH_1149_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when C_1 = "X" then TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: TOTAL is defined by two rules with different conditions.

        Git Branch: #1142.
        Goal: Check each rule computes TOTAL where its condition holds.
        """

        code = "GH_1142_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when C_1 = "X" then TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 non_null);
                    ... one statement per mode ...
                    DS_r6 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 always_zero);

        Description: B is missing on X and no condition holds on Z.

        Git Branch: #1142.
        Goal: Check every mode only computes TOTAL where a condition holds.
        """

        code = "GH_1142_2"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_3(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when C_1 = "X" then TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A;
                        G = TOTAL + B
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: G reads TOTAL, which two earlier rules define.

        Git Branch: #1142.
        Goal: Check G takes TOTAL from both rules.
        """

        code = "GH_1142_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_4(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        G = TOTAL + B;
                        when C_1 = "X" then TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: G reads TOTAL and is written before the two rules that define it.

        Git Branch: #1142.
        Goal: Check G is computed after both rules.
        """

        code = "GH_1142_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_5(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when C_1 in {"X", "Y"} then TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: Both conditions hold on Y, so two rules compute TOTAL for it: 30 and 10.

        Git Branch: #1142, #1193.
        Goal: Check rules that compute different values raise a runtime error.
        """

        code = "GH_1142_5"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1142_6(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: The rule without a condition always overlaps the other rule for TOTAL.

        Git Branch: #1142.
        Goal: Check the semantic error is raised.
        """

        code = "GH_1142_6"
        number_inputs = 1
        exception_code = "1-1-10-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1142_7(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_2) is
                        R1: TOTAL = A + B;
                        R1: G = A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2);

        Description: Two rules share the name R1.

        Git Branch: #1142.
        Goal: Check the semantic error is raised.
        """

        code = "GH_1142_7"
        number_inputs = 1
        exception_code = "1-3-1-5"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1142_8(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_2) is
                        R1: TOTAL = A + B;
                        G = A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2);

        Description: One rule is named and the other is not.

        Git Branch: #1142.
        Goal: Check the semantic error is raised.
        """

        code = "GH_1142_8"
        number_inputs = 1
        exception_code = "1-3-1-7"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1142_9(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when C_1 = "X" then TOTAL = A + B;
                        when C_1 = "Y" then TOTAL = A
                    end hierarchical ruleset;

                    define viral propagation At_1 (variable At_1) is
                        when "Dx" then "Dx";
                        when "Pz" then "Pz"
                    end viral propagation;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: At_1 is viral and TOTAL is defined by two rules with different conditions.

        Git Branch: #1142.
        Goal: Check At_1 of TOTAL combines the code items of the rule that applies.
        """

        code = "GH_1142_9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_10(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when C_1 = "Y" then TOTAL = A + B;
                        G = TOTAL + A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: TOTAL only applies to Y, which is not the first Id_1 of the data.

        Git Branch: #1142.
        Goal: Check G pairs TOTAL and A on the same Id_1.
        """

        code = "GH_1142_10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1142_11(self):
        """
        HIERARCHICAL RULSET: hierarchy, check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain rule VD_2) is
                        R1: TOTAL = A + B;
                        R2: TOTAL >= A
                    end hierarchical ruleset;

                    DS_h <- hierarchy(DS_1, HR_1 rule Id_2);
                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 all);

        Description: hierarchy only uses the = rules of the ruleset.

        Git Branch: #1142.
        Goal: Check a later check_hierarchy with the same ruleset still applies R2.
        """

        code = "GH_1142_11"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1145_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_1) is
                        G = B + C errorcode "G";
                        B = M + N errorcode "B";
                        C = P + Q errorcode "C"
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_1 all);

        Description: The unnamed rules are written in another order than their dependencies.

        Git Branch: #1145.
        Goal: Check each rule is numbered by its position in the ruleset.
        """

        code = "GH_1145_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1141_3(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_2 as C_2
                        rule VD_1) is
                        when C_2 = "X" then TOTAL = A + B
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_2 rule Id_1 always_zero);

        Description: A rule whose condition is FALSE for Id_2 = "Y" computes nothing there.

        Git Branch: #1141.
        Goal: Check no null TOTAL is computed where the condition is FALSE.
        """
        code = "GH_1141_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1141_4(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_2 as C_2
                        rule VD_1) is
                        when C_2 = "X" then TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_2 rule Id_1 always_zero rule computed);

        Description: With input rule, GRAND takes TOTAL from the first rule, which computes no
        TOTAL for Id_2 = "Y", so TOTAL counts as missing there.

        Git Branch: #1141.
        Goal: Check the next rule does not read the dataset TOTAL where the rule gives none.
        """
        code = "GH_1141_4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1155_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 rule VD_2) is
                        when Id_3 = "X" then TOTAL = A + B
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 all);

        Description: The condition of the rule uses Id_3, which is not in the signature.

        Git Branch: #1155.
        Goal: Check the semantic error is raised at the ruleset definition.
        """

        code = "GH_1155_1"
        number_inputs = 1
        exception_code = "1-3-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1155_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 rule VD_2) is
                        when VD_1 = "X" then TOTAL = A [Id_3 = "Y"] + B
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 all);

        Description: The condition of the code item A uses Id_3, which is not in the signature.

        Git Branch: #1155.
        Goal: Check the semantic error is raised at the ruleset definition.
        """

        code = "GH_1155_2"
        number_inputs = 1
        exception_code = "1-3-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1155_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1 rule VD_2) is
                        when VD_1 = "X" then TOTAL = A + B
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 all);

        Description: The condition uses VD_1, but the signature declares it with the alias C_1.

        Git Branch: #1155.
        Goal: Check the semantic error is raised at the ruleset definition.
        """

        code = "GH_1155_3"
        number_inputs = 1
        exception_code = "1-3-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1168_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        EU = BE [Id_1 > 1995] + FR
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 non_null);
                    ... the same for the other five modes (DS_r2 to DS_r6)

        Description: The condition of BE is FALSE in 1990, where the rule is EU = FR.

        Git Branch: #1168.
        Goal: Check BE is left out of the rule, not read as a missing data point, in every mode.
        """

        code = "GH_1168_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1168_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        EU = BE [Id_1 > 1995] + FR
                    end hierarchical ruleset;

                    DS_r1 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 non_null all);
                    ... the same for the other five modes (DS_r2 to DS_r6)

        Description: The condition of BE is FALSE in 1990, where the rule is EU = FR.

        Git Branch: #1168.
        Goal: Check BE is left out of the rule, not read as a missing data point, in every mode.
        """

        code = "GH_1168_2"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1168_3(self):
        """
        HIERARCHICAL RULSET: hierarchy, check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        EU = BE [Id_1 > 1995] + FR
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 non_null);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 partial_null all);
                    DS_r3 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 always_zero all);

        Description: BE is missing in 2000, where its condition is TRUE.

        Git Branch: #1168.
        Goal: Check BE is still read as a missing data point where its condition holds.
        """

        code = "GH_1168_3"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1168_4(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        EU = FR - BE [Id_1 > 1995];
                        WORLD = EU [Id_1 > 1995] + US
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 non_null);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 partial_null);

        Description: A subtracted code item and a computed code item have a condition.

        Git Branch: #1168.
        Goal: Check both are left out of their rules in 1990, where the conditions are FALSE.
        """

        code = "GH_1168_4"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1194_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as A rule VD_2) is
                        when (A = "X") then TOTAL = B + C
                    end hierarchical ruleset;
                    ... HR_2 to HR_5 with the conditions not (A = "Y"), substr(A, 1, 1) = "X",
                    if A = "X" then true else false and A in {"X", "Y"} and (A = "X" or A = "W")

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    ... the same for HR_2 to HR_5 (DS_r2 to DS_r5)

        Description: The conditions use the alias of the value domain inside parentheses,
        not, a function and if-then-else.

        Git Branch: #1194.
        Goal: Check each condition reads the component the call gives for the alias.
        """

        code = "GH_1194_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1194_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 rule VD_2) is
                        when (VD_1 = "X") then TOTAL = B + C
                    end hierarchical ruleset;
                    define hierarchical ruleset HR_2 (valuedomain condition VD_1 as A rule VD_2) is
                        TOTAL = B [substr(A, 1, 1) = "X"] + C
                    end hierarchical ruleset;

                    DS_r1 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 all);
                    DS_r2 <- check_hierarchy(DS_1, HR_2 condition Id_1 rule Id_2 all);

        Description: A condition names the value domain without an alias, and a right code
        item condition uses the alias in a function.

        Git Branch: #1194.
        Goal: Check both conditions read the component the call gives.
        """

        code = "GH_1194_2"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1169_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 dataset);

        Description: GRAND reads TOTAL, which the first rule computes.

        Git Branch: #1169.
        Goal: Check GRAND reads TOTAL from the operand, also where the first rule computes none.
        """

        code = "GH_1169_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1169_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2 rule);
                    DS_r2 <- hierarchy(DS_1, HR_1 rule Id_2 rule_priority);
                    DS_r3 <- hierarchy(DS_1, HR_1 rule Id_2 always_zero dataset);

        Description: GRAND reads TOTAL, which the first rule computes, in each input mode.

        Git Branch: #1169.
        Goal: Check only input dataset reads TOTAL from the operand where the rule computes it.
        """

        code = "GH_1169_2"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1169_3(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2 rule);
                    DS_r2 <- hierarchy(DS_1, HR_1 rule Id_2 rule_priority);
                    DS_r3 <- hierarchy(DS_1, HR_1 rule Id_2 dataset);

        Description: A and B are missing, so the first rule computes no TOTAL at all.

        Git Branch: #1169.
        Goal: Check input rule reads no TOTAL, and the other inputs read it from the operand.
        """

        code = "GH_1169_3"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1195_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        TOTAL_0 = B [Id_1 = "X"] + C;
                        TOTAL_1 = B [Id_1 in {"X"}] + C;
                        TOTAL_2 = B [(Id_1 = "X")] + C;
                        TOTAL_3 = B [Id_1 in {"X", "Y"} and Id_1 = "X"] + C;
                        TOTAL_4 = B [Id_1 = "X" or Id_1 = "W"] + C;
                        TOTAL_5 = B [not (Id_1 = "Y")] + C;
                        TOTAL_6 = B [between(Id_1, "W", "X")] + C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: The same right code item condition, written in several ways.

        Git Branch: #1195.
        Goal: Check every rule applies the condition, as TOTAL_0 does.
        """

        code = "GH_1195_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1195_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        TOTAL = B [Id_1 in {"X"}] + C
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 all);

        Description: The right code item condition is not a single comparison.

        Git Branch: #1195.
        Goal: Check B is left out of the rule for Id_1 = "Y".
        """

        code = "GH_1195_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1170_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 rule_priority);

        Description: The first rule computes no TOTAL for Id_1 = 2, where A and B are missing.

        Git Branch: #1170.
        Goal: Check GRAND takes TOTAL from the operand where the rule computes no TOTAL.
        """

        code = "GH_1170_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1170_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_2 as C_2
                        rule VD_1) is
                        when C_2 = "X" then TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_2 rule Id_1 always_zero rule_priority);

        Description: The condition of the first rule is FALSE for Id_2 = "Y", where it computes
        no TOTAL.

        Git Branch: #1170.
        Goal: Check GRAND takes TOTAL from the operand there, instead of reading it as 0.
        """

        code = "GH_1170_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1173_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 dataset_priority all);

        Description: TOTAL is missing from DS_1 on Id_1 = 2.

        Git Branch: #1173.
        Goal: Check the second rule takes TOTAL from the first rule there.
        """

        code = "GH_1173_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1173_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        R1: when Id_1 = 1 then TOTAL = A + B;
                        R2: when Id_1 = 2 then TOTAL = D + E;
                        R3: GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 dataset_priority all);

        Description: TOTAL is missing from DS_1, and two rules compute it under different
        conditions.

        Git Branch: #1173.
        Goal: Check R3 takes TOTAL from the rule that computes it on each Id_1.
        """

        code = "GH_1173_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1173_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        TOTAL = A + B;
                        GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 dataset_priority);

        Description: TOTAL is null in DS_1 on Id_1 = 1.

        Git Branch: #1173.
        Goal: Check the second rule takes TOTAL from the first rule there, while the first rule
        validates the TOTAL of DS_1.
        """

        code = "GH_1173_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1173_4(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: TOTAL = A + B;
                        R2: TOTAL = D + E;
                        R3: GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 dataset_priority all);

        Description: TOTAL is missing from DS_1 on Id_1 = 2, where R1 and R2 both compute it.

        Git Branch: #1173.
        Goal: Check the runtime error is raised.
        """

        code = "GH_1173_4"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1173_5(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: TOTAL = A + B;
                        R2: TOTAL = D + E;
                        R3: GRAND = TOTAL + C
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 dataset_priority invalid);

        Description: R1 and R2 compute different TOTAL values, but DS_1 has TOTAL on every
            Id_1, so R3 reads the TOTAL of DS_1 and no error is raised.

        Git Branch: #1173.
        Goal: Check Result.
        """

        code = "GH_1173_5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1190_1(self):
        """
        HIERARCHICAL RULSET: hierarchy, check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        C = - A + B
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 rule Id_2 all);

        Description: The right side of the rule starts with a minus sign.

        Git Branch: #1190.
        Goal: Check the rule computes C = B - A.
        """

        code = "GH_1190_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1190_2(self):
        """
        HIERARCHICAL RULSET: hierarchy, check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        C = + A + B
                    end hierarchical ruleset;
                    define hierarchical ruleset HR_2 (variable rule Id_2) is
                        C = - A
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_2 rule Id_2);
                    DS_r3 <- check_hierarchy(DS_1, HR_2 rule Id_2 all);

        Description: A plus sign before the first code item, and a signed code item alone.

        Git Branch: #1190.
        Goal: Check both rules compute their signed code items.
        """

        code = "GH_1190_2"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1190_3(self):
        """
        HIERARCHICAL RULSET: hierarchy, check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        C = - A [Id_1 > 1995] + B
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 all);

        Description: The signed code item has a condition, which is FALSE in 1990.

        Git Branch: #1190.
        Goal: Check A is left out of the rule in 1990 and subtracted in 2000.
        """

        code = "GH_1190_3"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1150_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 as A_1 rule Id_2) is
                        when A_1 > 1995 then EU = BE + FR errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 invalid);

        Description: The alias of a condition variable is used inside the ruleset, and the
        call uses the variable name.

        Git Branch: #1150.
        Goal: Check the rule is only validated where the aliased condition is true.
        """

        code = "GH_1150_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1150_1_roundtrip(self):
        """
        AST round-trip for check_hierarchy with an alias on a condition variable.

        Git Branch: #1150.
        Goal: Verify that rendering the AST back to VTL keeps the alias of the condition.
        """
        code = "GH_1150_1"
        number_inputs = 1
        references_names = ["1"]

        text = self.LoadVTL(code)
        rendered = ASTString().render(create_ast(text))
        self.BaseTest(
            code=code, number_inputs=number_inputs, references_names=references_names, text=rendered
        )

    def test_GH_1150_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 as A_1 rule Id_2) is
                        when A_1 > 1995 then EU = BE + FR
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: The alias of a condition variable is used inside the ruleset, and the
        call uses the variable name.

        Git Branch: #1150.
        Goal: Check the rule is only computed where the aliased condition is true.
        """

        code = "GH_1150_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"Y", "Z"} then TOTAL = B + C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    DS_n <- count(DS_r group by Id_2);

        Description: Both rules apply to Y and compute the same value for it. DS_n counts
        the TOTAL data points, so a repeated one shows there.

        Git Branch: #1193.
        Goal: Check TOTAL is computed once for Y.
        """

        code = "GH_1193_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"W", "X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"W", "Y", "Z"} then TOTAL = B + D
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 partial_null);

        Description: Both rules apply to W and Y. C and D are equal on Y and null on W, so
        the rules compute 30 for Y and, in partial_null, null for W.

        Git Branch: #1193.
        Goal: Check rules with different formulas that compute the same value, null
        included, give the data point once.
        """

        code = "GH_1193_2"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_3(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"W", "X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"W", "Y", "Z"} then TOTAL = B + D
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 partial_null);

        Description: Both rules apply to W: the first one computes null, as C is null, and
        the second one 12.

        Git Branch: #1193.
        Goal: Check a null and a number are different values, so the runtime error is raised.
        """

        code = "GH_1193_3"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1193_4(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"Y", "Z"} then TOTAL = B + C;
                        G = TOTAL + A
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    DS_n <- count(DS_r group by Id_2);

        Description: G reads TOTAL, which both rules compute for Y. DS_n counts the data
        points of each code item, so a repeated one shows there.

        Git Branch: #1193.
        Goal: Check G reads the TOTAL of Y once.
        """

        code = "GH_1193_4"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_5(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"Y", "Z"} then TOTAL = B + C;
                        G = TOTAL + A
                    end hierarchical ruleset;

                    define viral propagation VP_1 (variable At_1) is
                        aggregate sum
                    end viral propagation;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    DS_n <- count(DS_r group by Id_2);

        Description: At_1 is viral and both rules compute TOTAL for Y. DS_n counts the data
        points of each code item, so a repeated one shows there.

        Git Branch: #1193.
        Goal: Check At_1 of TOTAL is computed once for Y and At_1 of G adds it once.
        """

        code = "GH_1193_5"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_6(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"Y", "Z"} then TOTAL = B + D
                    end hierarchical ruleset;

                    define viral propagation VP_1 (variable At_1) is
                        aggregate sum
                    end viral propagation;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: Both rules compute 30 for TOTAL of Y, but At_1 is 3 for the first one
        and 6 for the second one.

        Git Branch: #1193.
        Goal: Check different viral attribute values raise the runtime error.
        """

        code = "GH_1193_6"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1193_7(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = A + B + C;
                        when Id_1 in {"Y", "Z"} then TOTAL = C + B + A;
                        G = TOTAL + D
                    end hierarchical ruleset;

                    define viral propagation VP_1 (variable At_1) is
                        aggregate sum
                    end viral propagation;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);
                    DS_n <- count(DS_r group by Id_2);

        Description: Both rules apply to Y, where Me_1 and At_1 of A, B and C are 0.1, 0.2
        and 0.3. Adding them in the order of the first rule gives 0.6000000000000001 and in
        the order of the second one 0.6, the same value within 15 significant digits. DS_n
        counts the data points of each code item, so a repeated one shows there.

        Git Branch: #1193.
        Goal: Check TOTAL is computed once for Y, with the values of the second rule, which
        G reads.
        """

        code = "GH_1193_7"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_8(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B;
                        when Id_1 in {"Y", "Z"} then TOTAL = C
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: Both rules apply to Y: the first one computes 1.0 and the second one
        1.00000000000001, which differ in the 15th significant digit.

        Git Branch: #1193.
        Goal: Check values that differ within 15 significant digits raise the runtime error.
        """

        code = "GH_1193_8"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1193_9(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B;
                        when Id_1 in {"Y", "Z"} then TOTAL = C;
                        G = TOTAL + D
                    end hierarchical ruleset;

                    define viral propagation VP_1 (variable At_1) is
                        aggregate sum
                    end viral propagation;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: With COMPARISON_ABSOLUTE_THRESHOLD set to 6, the 1.0 that the first
        rule computes for TOTAL of Y, in Me_1 and in At_1, and the 1.000001 of the second
        one are the same value.

        Git Branch: #1193.
        Goal: Check the data point keeps the values of the last rule, which G reads.
        """

        code = "GH_1193_9"
        number_inputs = 1
        references_names = ["1"]

        with mock.patch.dict(os.environ, {"COMPARISON_ABSOLUTE_THRESHOLD": "6"}):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1193_10(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        when Id_1 in {"X", "Y"} then TOTAL = B + C;
                        when Id_1 in {"Y", "Z"} then TOTAL = B + D;
                        C = E + F;
                        H = TOTAL + B
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2);

        Description: With COMPARISON_ABSOLUTE_THRESHOLD set to 6, the 1.0 that the first
        rule computes for TOTAL of Y and the 1.000001 of the second one are the same value.
        The first rule reads C, which a later rule computes.

        Git Branch: #1193.
        Goal: Check the data point keeps the value of the last rule written, which H reads,
        even when the first one depends on a later rule.
        """

        code = "GH_1193_10"
        number_inputs = 1
        references_names = ["1"]

        with mock.patch.dict(os.environ, {"COMPARISON_ABSOLUTE_THRESHOLD": "6"}):
            self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2)[calc attribute At_1 := "z"];

        Description: hierarchy drops the attribute At_1 of its operand, which the calc
        clause then creates again.

        Git Branch: #1218.
        Goal: Check a calc clause on the hierarchy result sets a dropped attribute.
        """

        code = "GH_1218_1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2 always_zero all)
                                [calc attribute At_1 := "z"];

        Description: With output all, the data points of the operand that no rule
        computes are kept without the attribute At_1, which the calc clause creates again.

        Git Branch: #1218.
        Goal: Check hierarchy with output all drops the attributes that are not viral.
        """

        code = "GH_1218_2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 rule Id_2 all)
                                [calc attribute At_1 := "z"];

        Description: check_hierarchy drops the attribute At_1 of its operand, which the
        calc clause then creates again.

        Git Branch: #1218.
        Goal: Check a calc clause on the check_hierarchy result sets a dropped attribute.
        """

        code = "GH_1218_3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_4(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2)[rename Me_1 to Me_2];
                    DS_r2 <- hierarchy(DS_1, HR_1 rule Id_2)[sub Id_1 = "X"];
                    DS_r3 <- hierarchy(DS_1, HR_1 rule Id_2)
                                [calc attribute At_1 := "z"][calc attribute At_2 := "w"];
                    DS_r4 <- hierarchy(DS_1, HR_1 rule Id_2)
                                [calc attribute At_1 := "z", attribute At_2 := "w"];

        Description: hierarchy drops the attributes At_1 (String) and At_2 (Date) of its
        operand. The clauses on the result do not refer to them, and the calc clauses create
        them again, At_2 with another type.

        Git Branch: #1218.
        Goal: Check clauses chained on a hierarchy result whose operand has attributes.
        """

        code = "GH_1218_4"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_5(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2 all);
                    DS_r2 <- hierarchy(DS_1, HR_1 rule Id_2 dataset all)
                                [calc attribute At_1 := "z"];
                    DS_r3 <- hierarchy(DS_1, HR_1 rule Id_2 always_zero rule_priority all)
                                [calc attribute At_1 := "z"];

        Description: With output all, the data points of the operand that no rule computes
        are kept without the attributes At_1 and At_2, with and without a clause on the
        result and with the dataset and rule_priority inputs.

        Git Branch: #1218.
        Goal: Check hierarchy with output all drops the attributes that are not viral.
        """

        code = "GH_1218_5"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_6(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Id_1 rule Id_2) is
                        R1: when Id_1 = "X" then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_2 condition Id_1 rule Id_2)
                                [calc attribute At_1 := "z"];
                    DS_r2 <- hierarchy(DS_1[filter At_1 <> "y"], HR_1 rule Id_2 partial_zero)
                                [calc attribute At_1 := "z"];
                    DS_r3 <- hierarchy(DS_1[calc attribute At_3 := "q"], HR_1 rule Id_2)
                                [calc attribute At_3 := "z"];

        Description: A conditioned rule, an operand filtered on one of its attributes and an
        operand with an attribute created by a calc clause, each followed by a calc clause
        that creates a dropped attribute again.

        Git Branch: #1218.
        Goal: Check other hierarchy operands and rules drop the attributes that are not viral.
        """

        code = "GH_1218_6"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_7(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(hierarchy(DS_1, HR_1 rule Id_2 all), HR_1 rule Id_2);
                    DS_r2 <- check_hierarchy(hierarchy(DS_1, HR_1 rule Id_2 all),
                                HR_1 rule Id_2 all);
                    DS_r3 <- inner_join(hierarchy(DS_1, HR_1 rule Id_2) as a, DS_2 as b
                                calc attribute At_1 := "z");

        Description: The result of hierarchy, whose operand has attributes, is the operand of
        another hierarchy, of a check_hierarchy and of a join with a calc clause.

        Git Branch: #1218.
        Goal: Check operators that take a hierarchy result as operand.
        """

        code = "GH_1218_7"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_8(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define operator f (ds dataset)
                        returns dataset is
                            hierarchy(ds, HR_1 rule Id_2)
                    end operator;

                    DS_r <- f(DS_1)[calc attribute At_1 := "z"];

        Description: A user defined operator returns the hierarchy result, and a calc clause
        on its call creates a dropped attribute again.

        Git Branch: #1218.
        Goal: Check a calc clause on a user defined operator that returns a hierarchy result.
        """

        code = "GH_1218_8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_9(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define viral propagation VP (variable VAt_1) is
                        aggregate max
                    end viral propagation;

                    define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2)[calc attribute At_1 := "z"];

        Description: The operand has the attribute At_1 and the viral attribute VAt_1.
        hierarchy keeps VAt_1, combined with max, and drops At_1, which the calc clause
        creates again.

        Git Branch: #1218.
        Goal: Check hierarchy keeps the viral attributes and drops the other ones.
        """

        code = "GH_1218_9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_10(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define datapoint ruleset DPR (variable Me_1) is
                        R1: Me_1 > 2 errorcode "E2" errorlevel 1
                    end datapoint ruleset;

                    DS_r1 <- check_datapoint(DS_1, DPR all)[calc attribute At_1 := "z"];
                    DS_r2 <- check(DS_1 > 2 errorcode "E3" errorlevel 1 imbalance DS_1 - 2 all)
                                [calc attribute At_1 := "z"];
                    DS_r3 <- check_hierarchy(DS_1, HR_1 rule Id_2 invalid)
                                [calc attribute At_1 := "z"];

        Description: The validation operators drop the attributes of their operand, which a
        calc clause on their result creates again.

        Git Branch: #1218.
        Goal: Check a calc clause on the result of the validation operators.
        """

        code = "GH_1218_10"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1218_11(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 rule Id_2)[filter At_1 = "x"];

        Description: hierarchy drops the attribute At_1 of its operand, so the filter clause
        on the result refers to a component that does not exist.

        Git Branch: #1218.
        Goal: Check a clause on a hierarchy result cannot use a dropped attribute.
        """

        code = "GH_1218_11"
        number_inputs = 1
        exception_code = "1-1-1-10"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1223_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: when At_1 = "x" then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_null);
                    DS_r3 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 always_zero all);

        Description: The condition of the rule is on the attribute At_1. A data point takes
        part in the rule only when its own At_1 satisfies it, a null At_1 does not. The
        result drops the attributes At_1 and At_2.

        Git Branch: #1223.
        Goal: Check hierarchy with an attribute as condition component.
        """

        code = "GH_1223_1"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: when At_1 = "x" then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2 all);
                    DS_r3 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2
                                partial_null all);
                    DS_r4 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2
                                always_zero all_measures);

        Description: The condition of the rule is on the attribute At_1. The validated and the
        right code items take part in the rule only when their own At_1 satisfies it.

        Git Branch: #1223.
        Goal: Check check_hierarchy with an attribute as condition component.
        """

        code = "GH_1223_2"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_3(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define viral propagation VP_1 (variable At_1) is
                        aggregate max
                    end viral propagation;

                    define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: when At_1 <> "z" then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero all);
                    DS_r3 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2
                                always_zero all);

        Description: The condition of the rule is on the viral attribute At_1, which still
        propagates to the result: the computed data points combine the At_1 of the right code
        items, the validated ones keep their own.

        Git Branch: #1223.
        Goal: Check a viral attribute as condition component.
        """

        code = "GH_1223_3"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_4(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1;
                        R2: when At_1 = "x" then D = A + E errorcode "E2" errorlevel 2
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero);
                    DS_r3 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero
                                rule_priority);
                    DS_r4 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero
                                dataset);
                    DS_r5 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero
                                dataset_priority all);

        Description: R2 reads the code item A that R1 computes. R2 applies to the groups
        with a data point whose At_1 is "x", also of a code item outside the ruleset (Z), and
        reads there the computed A.

        Git Branch: #1223.
        Goal: Check an attribute condition on a rule that reads a computed code item.
        """

        code = "GH_1223_4"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_5(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: A = B [At_1 = "x"] + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2 all);

        Description: The right code item B has a condition on the attribute At_1. B is left
        out of the rule only when no data point of the group satisfies it.

        Git Branch: #1223.
        Goal: Check an attribute in the condition of a right code item.
        """

        code = "GH_1223_5"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_6(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (valuedomain condition VD_1 as C_1
                        rule VD_2) is
                        R1: when C_1 = "x" then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Id_1, At_1
                        rule Id_2) is
                        R1: when Id_1 > 1 and At_1 = "x" then A = B + C errorcode "E1"
                            errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2 all);
                    DS_r3 <- hierarchy(DS_2, HR_2 condition Id_1, At_1 rule Id_2);

        Description: The alias C_1 of a value domain condition refers to the attribute At_1
        of DS_1, whose only identifier is the rule component. HR_2 conditions on an
        identifier and an attribute together.

        Git Branch: #1223.
        Goal: Check attribute conditions with an alias, without other identifiers and mixed
        with an identifier.
        """

        code = "GH_1223_6"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1224_1(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Me_1 rule Id_2) is
                        R1: when Me_1 > 0 then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Me_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition Me_1 rule Id_2 all);

        Description: The condition of the rule is on the measure Me_1, which every data point
        satisfies.

        Git Branch: #1224.
        Goal: Check a measure as condition component.
        """

        code = "GH_1224_1"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1224_2(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Me_1 rule Id_2) is
                        R1: when Me_1 < 50 then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Me_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition Me_1 rule Id_2 partial_null);
                    DS_r3 <- hierarchy(DS_1, HR_1 condition Me_1 rule Id_2 always_zero all);
                    DS_r4 <- check_hierarchy(DS_1, HR_1 condition Me_1 rule Id_2 all);
                    DS_r5 <- check_hierarchy(DS_1, HR_1 condition Me_1 rule Id_2
                                partial_null all);
                    DS_r6 <- check_hierarchy(DS_1, HR_1 condition Me_1 rule Id_2
                                always_zero all_measures);

        Description: The condition of the rule is on the measure Me_1. A data point takes part
        in the rule only when its own Me_1 satisfies it.

        Git Branch: #1224.
        Goal: Check a measure condition evaluated per data point.
        """

        code = "GH_1224_2"
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1224_3(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Me_1 rule Id_2) is
                        R1: A = B [Me_1 < 5] + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Me_1 rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1;
                        R2: when Me_1 > 5 then D = A + E errorcode "E2" errorlevel 2
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Me_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition Me_1 rule Id_2 all);
                    DS_r3 <- hierarchy(DS_2, HR_2 condition Me_1 rule Id_2);
                    DS_r4 <- hierarchy(DS_2, HR_2 condition Me_1 rule Id_2 partial_zero);

        Description: A measure in the condition of a right code item, and in the condition of
        a rule that reads a computed code item.

        Git Branch: #1224.
        Goal: Check measure conditions on right code items and on computed code items.
        """

        code = "GH_1224_3"
        number_inputs = 2
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_7(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define viral propagation VP_1 (variable At_1) is
                        aggregate max
                    end viral propagation;

                    define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: when At_1 = "w" then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 all);

        Description: No data point satisfies the condition on the viral attribute At_1, so
        the rule computes no data point. The result keeps At_1.

        Git Branch: #1223.
        Goal: Check a hierarchy that computes no data point of an operand with a viral
        attribute.
        """

        code = "GH_1223_7"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_8(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1;
                        R2: when At_1 = "x" then D = A + E errorcode "E2" errorlevel 2
                    end hierarchical ruleset;

                    DS_r1 <- check_hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_null
                                dataset_priority all);
                    DS_r2 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero);

        Description: The rule component is the only identifier of DS_1, and no data point
        satisfies the condition of R2: R2 does not apply, even if R1 computes A.

        Git Branch: #1223.
        Goal: Check a rule that reads a computed code item and whose condition no data point
        satisfies, without other identifiers.
        """

        code = "GH_1223_8"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_9(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1, Me_1
                        rule Id_2) is
                        R1: when At_1 = "x" then A = B + C errorcode "E1" errorlevel 1;
                        R2: when Me_1 > 0 then A = B - C errorcode "E2" errorlevel 2;
                        R3: when At_1 = "x" then D = A + E errorcode "E3" errorlevel 3
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition At_1, Me_1 rule Id_2
                                dataset_priority all);

        Description: R3 reads A, which R1 and R2 compute with different values. The A of
        the operand does not satisfy the condition of R3, so R3 does not read it, and the
        runtime error is raised.

        Git Branch: #1223.
        Goal: Check dataset_priority reads the operand as the condition of the reading rule
        sees it.
        """

        code = "GH_1223_9"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1223_10(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define viral propagation VP_1 (variable At_1) is
                        aggregate max
                    end viral propagation;

                    define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: when At_1 = "y" then A = B + F errorcode "E1" errorlevel 1;
                        R2: when At_1 = "x" then A = B + C errorcode "E2" errorlevel 2;
                        R3: when At_1 = "x" then D = A + E errorcode "E3" errorlevel 3
                    end hierarchical ruleset;

                    DS_r <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2 partial_zero);

        Description: The rule component is the only identifier of DS_1. R1 and R2 may both
        compute A, but only R1 does, so D combines the viral At_1 that R1 gives to A.

        Git Branch: #1223.
        Goal: Check the viral attributes of a code item that only some of the rules that may
        compute it compute, without other identifiers.
        """

        code = "GH_1223_10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_11(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: F = D - A errorcode "E1" errorlevel 1;
                        R2: G = F + B errorcode "E2" errorlevel 2
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2 non_zero rule_priority);
                    DS_r2 <- hierarchy(DS_1, HR_1 rule Id_2 non_zero);
                    DS_r3 <- check_hierarchy(DS_1, HR_1 rule Id_2 non_zero dataset_priority all);

        Description: R1 computes a zero F for Id_1 1, which the non_zero mode drops, so R2
        does not read it: with rule_priority it reads the F of the operand, with rule none.

        Git Branch: #1223.
        Goal: Check a rule does not read a computed data point that the mode drops.
        """

        code = "GH_1223_11"
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_12(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        R1: when Id_1 > 0 then A = B + C errorcode "E1" errorlevel 1;
                        R2: when Id_1 < 10 then A = C + B errorcode "E2" errorlevel 2;
                        R3: D = A + E errorcode "E3" errorlevel 3
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Me_1 rule Id_2) is
                        R1: when Me_1 > 0 then A = B + C errorcode "E1" errorlevel 1;
                        R2: when Me_1 < 10 then A = C + B errorcode "E2" errorlevel 2;
                        R3: D = A + E errorcode "E3" errorlevel 3
                    end hierarchical ruleset;

                    DS_r1 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 always_zero
                                dataset_priority all);
                    DS_r2 <- check_hierarchy(DS_2, HR_2 condition Me_1 rule Id_2 always_zero
                                dataset_priority all);

        Description: R1 and R2 compute the same A, which the operand lacks, so R3 reads it
        once. DS_2 has no identifier other than the rule component.

        Git Branch: #1223.
        Goal: Check dataset_priority with overlapping rules that compute the same value.
        """

        code = "GH_1223_12"
        number_inputs = 2
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_13(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        R1: when Id_1 > 0 then A = B + C errorcode "E1" errorlevel 1;
                        R2: when Id_1 < 10 then A = B - C errorcode "E2" errorlevel 2;
                        R3: when Id_1 > 5 then D = A + E errorcode "E3" errorlevel 3
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2
                                dataset_priority all);

        Description: R1 and R2 compute different values of A. R3, which reads A, does not
        apply to Id_1 1, so it does not read the A of the operand there.

        Git Branch: #1223.
        Goal: Check the runtime error of overlapping rules where the reading rule does not
        apply.
        """

        code = "GH_1223_13"
        number_inputs = 1
        exception_code = "2-1-10-1"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_GH_1223_14(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        R1: A = B [Id_1 = 1] + C + B [Id_1 = 2] errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition At_1 rule Id_2) is
                        R1: A = B [At_1 = "x"] + C + B [At_1 = "y"] errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 always_zero);
                    DS_r2 <- hierarchy(DS_1, HR_2 condition At_1 rule Id_2 always_zero);

        Description: B appears twice in the rule, each time with its own condition.

        Git Branch: #1223.
        Goal: Check a code item twice in a rule with different right conditions.
        """

        code = "GH_1223_14"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_15(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = b + B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 rule Id_2 always_zero all);

        Description: The code items b and B differ only in case.

        Git Branch: #1223.
        Goal: Check code items that differ only in case are different code items.
        """

        code = "GH_1223_15"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_16(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition At_1 rule Id_2) is
                        R1: when time_agg("A", _, At_1, first) = cast("2020A", time_period)
                            then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Id_1 rule Id_2) is
                        R1: when time_agg("A", _, Id_1, first) = cast("2020A", time_period)
                            then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition At_1 rule Id_2);
                    DS_r2 <- hierarchy(DS_1, HR_2 condition Id_1 rule Id_2);

        Description: The conditions aggregate the time period of an attribute and of an
        identifier.

        Git Branch: #1223.
        Goal: Check time_agg in the condition of a hierarchical rule.
        """

        code = "GH_1223_16"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_17(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R0: T0 = A0 + B0 errorcode "E0" errorlevel 1;
                        ...
                        R16: T16 = A16 + B16 errorcode "E16" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Id_1 rule Id_2) is
                        R0: when Id_1 = 1 then X = A0 + B0 errorcode "E0" errorlevel 1;
                        R1: when Id_1 = 2 then X = A1 + B1 errorcode "E1" errorlevel 1;
                        R2: Y = X + A2 errorcode "E2" errorlevel 1;
                        R3: T3 = A3 + B3 errorcode "E3" errorlevel 1;
                        ...
                        R16: T16 = A16 + B16 errorcode "E16" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- check_hierarchy(DS_1, HR_1 rule Id_2 always_zero
                                dataset_priority all);
                    DS_r2 <- check_hierarchy(DS_1, HR_2 condition Id_1 rule Id_2 always_zero
                                dataset_priority all);

        Description: Rulesets of more rules than one pivot holds, with and without overlapping
        rules that another rule reads.

        Git Branch: #1223.
        Goal: Check dataset_priority on large rulesets.
        """

        code = "GH_1223_17"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_18(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Me_1 rule Id_2) is
                        R1: when Me_1 = 0.3 then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_2 (variable condition Id_1 rule Id_2) is
                        R1: when getyear(Id_1) = 2020 then A = B + C errorcode "E1"
                            errorlevel 1
                    end hierarchical ruleset;

                    define hierarchical ruleset HR_3 (variable condition Id_1 rule Id_2) is
                        R1: when period_indicator(Id_1) = cast("Q", duration)
                            then A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Me_1 rule Id_2 always_zero);
                    DS_r2 <- hierarchy(DS_2, HR_2 condition Id_1 rule Id_2);
                    DS_r3 <- hierarchy(DS_2, HR_3 condition Id_1 rule Id_2);

        Description: The conditions compare a Number with the tolerance of the comparisons,
        and the year and the period indicator of a time period.

        Git Branch: #1223.
        Goal: Check conditions are evaluated as the same expressions elsewhere.
        """

        code = "GH_1223_18"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_19(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        R1: X = E [Id_1 > 1] + D [Id_1 = 2] + B errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1 rule Id_2 partial_null);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2
                                partial_null all);

        Description: For Id_1 1, E and D are left out of the rule, so X is B, which exists.

        Git Branch: #1223.
        Goal: Check partial_null with two code items next to each other left out of the rule.
        """

        code = "GH_1223_19"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_20(self):
        """
        HIERARCHICAL RULSET: hierarchy and check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1, At_1
                        rule Id_2) is
                        R1: when Id_1 > 1 then Y = A + D [At_1 = "x"] errorcode "E1"
                            errorlevel 1
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 condition Id_1, At_1 rule Id_2
                                partial_null);
                    DS_r2 <- check_hierarchy(DS_1, HR_1 condition Id_1, At_1 rule Id_2
                                partial_null all);

        Description: The data point of E, a code item outside the ruleset, satisfies the
        condition of D, so D is missing, not left out of the rule.

        Git Branch: #1223.
        Goal: Check a right condition on an attribute in a rule with a condition on an
        identifier.
        """

        code = "GH_1223_20"
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1225_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C
                    end hierarchical ruleset;

                    DS_r1 <- union(hierarchy(DS_1, HR_1 rule Id_2), DS_2);
                    DS_r2 <- intersect(hierarchy(DS_1, HR_1 rule Id_2), DS_2);
                    DS_r3 <- setdiff(hierarchy(DS_1, HR_1 rule Id_2), DS_2);
                    DS_r4 <- symdiff(hierarchy(DS_1, HR_1 rule Id_2), DS_2);

        Description: The first operand of each set operator is a hierarchy written inline.

        Git Branch: #1225.
        Goal: Check the set operators read an inline hierarchy as their first operand.
        """

        code = "GH_1225_1"
        number_inputs = 2
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1225_2(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C
                    end hierarchical ruleset;

                    DS_r1 <- union(DS_2, hierarchy(DS_1, HR_1 rule Id_2 all));
                    DS_r2 <- intersect(DS_2, hierarchy(DS_1, HR_1 rule Id_2 all));
                    DS_r3 <- setdiff(DS_2, hierarchy(DS_1, HR_1 rule Id_2 all));
                    DS_r4 <- symdiff(DS_2, hierarchy(DS_1, HR_1 rule Id_2 all));

        Description: The second operand of each set operator is a hierarchy written inline,
        in output mode all.

        Git Branch: #1225.
        Goal: Check the set operators read an inline hierarchy as their second operand.
        """

        code = "GH_1225_2"
        number_inputs = 2
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1225_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C errorcode "E1" errorlevel 1
                    end hierarchical ruleset;

                    DS_c <- check_hierarchy(DS_2, HR_1 rule Id_2 all);
                    DS_r1 <- union(check_hierarchy(DS_1, HR_1 rule Id_2 all), DS_c);
                    DS_r2 <- setdiff(check_hierarchy(DS_1, HR_1 rule Id_2 all), DS_c);

        Description: The first operand of union and setdiff is a check_hierarchy written
        inline.

        Git Branch: #1225.
        Goal: Check the set operators read an inline check_hierarchy as an operand.
        """

        code = "GH_1225_3"
        number_inputs = 2
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GH_1223_21(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable condition Id_1 rule Id_2) is
                        R1: when Id_1 > 0 then A = B + C errorcode "E1" errorlevel 1;
                        R2: when Id_1 < 10 then A = B - C errorcode "E2" errorlevel 2;
                        R3: D >= A + E errorcode "E3" errorlevel 3
                    end hierarchical ruleset;

                    DS_r <- check_hierarchy(DS_1, HR_1 condition Id_1 rule Id_2
                                dataset_priority all);

        Description: R1 and R2 compute different values of A, which the operand lacks and
        R3, a rule that only validates, reads.

        Git Branch: #1223.
        Goal: Check the runtime error names the ruleset when a validating rule reads the
        code item.
        """

        code = "GH_1223_21"
        number_inputs = 1
        exception_code = "2-1-10-1"
        exception_message = (
            "More than one rule of the Hierarchy Roll-up HR_1 computes code item A "
            "for the same data point, with different values."
        )

        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=exception_code,
            exception_message=exception_message,
        )

    def test_GH_1250_1(self):
        """
        HIERARCHICAL RULSET: hierarchy, check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset HR_1 (variable rule Id_2) is
                        R1: A = B + C;
                        R2: C = D + E;
                        R3: Y < Z
                    end hierarchical ruleset;

                    DS_r1 <- hierarchy(DS_1, HR_1 rule Id_2 always_null dataset all);
                    DS_r2 <- hierarchy(DS_1, HR_1 rule Id_2 always_zero dataset all);
                    DS_r3 <- check_hierarchy(DS_1, HR_1 rule Id_2 always_null all);
                    DS_r4 <- check_hierarchy(DS_1, HR_1 rule Id_2 always_zero all);

        Description: Id_1 3 has neither D nor E, Id_1 4 has no code item of the ruleset, and
        no Id_1 has Y or Z.

        Git Branch: #1250.
        Goal: Check the always modes produce the result of a rule whose code items are all
        missing.
        """

        code = "GH_1250_1"
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

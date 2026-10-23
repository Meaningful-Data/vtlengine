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

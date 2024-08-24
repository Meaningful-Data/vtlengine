import json
from pathlib import Path
from typing import Dict, List, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class HierarchicalHelper(TestCase):
    """

    """

    base_path = Path(__file__).parent
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_vtl = base_path / "data" / "vtl"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    # File extensions.--------------------------------------------------------------
    JSON = '.json'
    CSV = '.csv'
    VTL = '.vtl'

    @classmethod
    def LoadDataset(cls, ds_path, dp_path):
        with open(ds_path, 'r') as file:
            structures = json.load(file)

        for dataset_json in structures['datasets']:
            dataset_name = dataset_json['name']
            components = {
                component['name']: Component(name=component['name'],
                                             data_type=SCALAR_TYPES[component['type']],
                                             role=Role(component['role']),
                                             nullable=component['nullable'])
                for component in dataset_json['DataStructure']}
            data = pd.read_csv(dp_path, sep=',')

            return Dataset(name=dataset_name, components=components, data=data)

    @classmethod
    def LoadInputs(cls, code: str, number_inputs: int) -> Dict[str, Dataset]:
        '''

        '''
        datasets = {}
        for i in range(number_inputs):
            json_file_name = str(cls.filepath_json / f"{code}-{str(i + 1)}{cls.JSON}")
            csv_file_name = str(cls.filepath_csv / f"{code}-{str(i + 1)}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadOutputs(cls, code: str, references_names: List[str]) -> Dict[str, Dataset]:
        """

        """
        datasets = {}
        for name in references_names:
            json_file_name = str(cls.filepath_out_json / f"{code}-{name}{cls.JSON}")
            csv_file_name = str(cls.filepath_out_csv / f"{code}-{name}{cls.CSV}")
            dataset = cls.LoadDataset(json_file_name, csv_file_name)
            datasets[dataset.name] = dataset

        return datasets

    @classmethod
    def LoadVTL(cls, code: str) -> str:
        """

        """
        vtl_file_name = str(cls.filepath_vtl / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, text: Any, code: str, number_inputs: int, references_names: List[str]):
        '''

        '''
        if text is None:
            text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        interpreter = InterpreterAnalyzer(input_datasets)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str, vd_names=Any):
        assert True

    @classmethod
    def NewExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True


class HierarchicalRulsetOperatorsTest(HierarchicalHelper):
    """
    Group 1
    """

    classTest = 'test_hierarchical.HierarchicalRulsetOperatorsTest'

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
        code = '1-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_265_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_265_2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = 'GL_265_3'
        number_inputs = 1
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_GL_265_4(self):
        # este el de varios statements, no va bien del todo, revisar
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

        code = 'GL_265_4'
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_265_5(self):
        # El tercer resultado parece que esta mal
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

        code = 'GL_265_5'
        number_inputs = 3
        references_names = ["1", "2", "3"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-21'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-24'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-25'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-26'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-27'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-29'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '1-1-1-30'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-31'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_275_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_145_1'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-3"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_145_2'
        number_inputs = 1
        vd_names = []  # TODO: Now it's not necessary
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_145_3'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-4"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_145_4'
        number_inputs = 1
        vd_names = []  # TODO: Now it's not necessary
        references_names = ["1"]

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

        code = 'GL_397_1'
        number_inputs = 1
        references_names = ["1"]

        # self.BaseTest(text= None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_3'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_397_5'
        number_inputs = 1
        references_names = ["1"]

        # self.BaseTest(text= None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_7'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-7"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

    def test_GL_397_9(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:.
        """

        code = 'GL_397_9'
        number_inputs = 1
        references_names = ["1"]

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

        code = 'GL_397_11'
        number_inputs = 1
        references_names = ["1"]

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

        code = 'GL_397_13'
        number_inputs = 1
        references_names = ["1"]

    def test_GL_397_15(self):
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

        code = 'GL_397_15'
        number_inputs = 1
        vd_names = []
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_397_17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_24(self):
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

        code = 'GL_397_24'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_25'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_27'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_29'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_30'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_32'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_34'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_36'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-1-3"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)


class HierarchicalRollUpOperatorsTest(HierarchicalHelper):
    """
    Group 2
    """

    classTest = 'test_hierarchical.HierarchicalRollUpOperatorsTest'

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
        code = '2-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-15'
        number_inputs = 1
        error_code = "2-1-1-3"

        self.NewExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

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

        code = '2-1-1-16'
        number_inputs = 1
        code = '2-1-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-21'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-23'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-24'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = '2-1-1-25'
        number_inputs = 1
        error_code = "2-1-1-3"

        self.NewExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

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

        code = '2-1-1-26'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-27'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-29'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-30'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-31'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '2-1-1-32'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_145_5'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-3"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_145_6'
        number_inputs = 1
        vd_names = []  # TODO: Now it's not necessary
        references_names = ["1"]

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

        code = 'GL_145_7'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-4"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_145_8'
        number_inputs = 1
        vd_names = []  # TODO: Now it's not necessary
        references_names = ["1"]

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

        code = 'GL_145_9'
        number_inputs = 1
        vd_names = []  # TODO: Now it's not necessary
        references_names = ["1"]

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

        code = 'GL_397_2'
        number_inputs = 1
        references_names = ["1"]

        # self.BaseTest(text= None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_4'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-6"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_397_6'
        number_inputs = 1
        references_names = ["1"]

        # self.BaseTest(text= None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_8'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-10-7"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

    def test_GL_397_10(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: 

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #397.
        Goal:.
        """

        code = 'GL_397_10'
        number_inputs = 1
        references_names = ["1"]

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

        code = 'GL_397_12'
        number_inputs = 1
        references_names = ["1"]

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

        code = 'GL_397_14'
        number_inputs = 1
        references_names = ["1"]

    def test_GL_397_16(self):
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

        code = 'GL_397_16'
        number_inputs = 1
        vd_names = []
        error_code = "1-3-16"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code,
                                      vd_names=vd_names)

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

        code = 'GL_397_19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_20'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_21'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_22'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_26'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_31(self):
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

        code = 'GL_397_31'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_397_33(self):
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

        code = 'GL_397_33'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_35'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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

        code = 'GL_397_37'
        number_inputs = 1
        vd_names = []
        error_code = "1-1-1-3"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code, vd_names=vd_names
        )

    def test_GL_463_1(self):
        """
        HIERARCHICAL RULSET: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: #463.
        Goal:
        """

        code = 'GL_463_1'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

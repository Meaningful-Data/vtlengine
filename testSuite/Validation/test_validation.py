import json
import os.path
from pathlib import Path
from typing import List, Dict, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Dataset, Component, ExternalRoutine, Role, ValueDomain


class ValidationHelper(TestCase):
    """

    """
    # Path Selection.----------------------------------------------------------
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"

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
            if not os.path.exists(dp_path):
                data = pd.DataFrame(columns=list(components.keys()))
            else:
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
        vtl_file_name = str(cls.filepath_VTL / f"{code}{cls.VTL}")
        with open(vtl_file_name, 'r') as file:
            return file.read()

    @classmethod
    def BaseTest(cls, code: str, number_inputs: int, references_names: List[str], vd_names: List[str] = None,
                 sql_names: List[str] = None):
        '''

        '''
        text = cls.LoadVTL(code)
        ast = create_ast(text)
        input_datasets = cls.LoadInputs(code, number_inputs)
        reference_datasets = cls.LoadOutputs(code, references_names)
        value_domains = None
        if vd_names is not None:
            value_domains = cls.LoadValueDomains(vd_names)

        external_routines = None
        if sql_names is not None:
            external_routines = cls.LoadExternalRoutines(sql_names)
        interpreter = InterpreterAnalyzer(input_datasets,
                                          value_domains=value_domains,
                                          external_routines=external_routines)
        result = interpreter.visit(ast)
        assert result == reference_datasets

    @classmethod
    def NewSemanticExceptionTest(cls, code: str, number_inputs: int, exception_code: str):
        assert True

    @classmethod
    def LoadValueDomains(cls, vd_names):
        value_domains = {}
        for name in vd_names:
            vd_file_name = str(cls.filepath_valueDomain / f"{name}.json")
            with open(vd_file_name, 'r') as file:
                vd = ValueDomain.from_json(file.read())
                value_domains[vd.name] = vd
        return value_domains

    @classmethod
    def LoadExternalRoutines(cls, sql_names):
        external_routines = {}
        for name in sql_names:
            sql_file_name = str(cls.filepath_sql / f"{name}.sql")
            with open(sql_file_name, 'r') as file:
                external_routines[name] = ExternalRoutine.from_sql_query(name, file.read())
        return external_routines


class ValidationOperatorsTests(ValidationHelper):
    """
    Group 1
    """

    classTest = 'validation.ValidationOperatorsTests'

    def test_1(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset)
                            [calc OBS_VALUE_N := OBS_VALUE * 2];
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: 309-validation-operators-tests.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = '1-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY dataset)
                            [rename OBS_VALUE to OBS_VALUE_N];
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: 309-validation-operators-tests.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = '1-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        HIERARCHICAL RULSET: check_hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := check_hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero dataset all_measures)
                            [aggr OBS_VALUE_N := sum (OBS_VALUE) group by ACCOUNTING_ENTRY];
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: 309-validation-operators-tests.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """
        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero dataset)
                            [calc OBS_VALUE_N := OBS_VALUE + 1];
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: 309-validation-operators-tests.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = '1-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero dataset all)
                    [rename OBS_VALUE to OBS_VALUE_N];
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: 309-validation-operators-tests.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        HIERARCHICAL ROLL-UP: hierarchy
        Dataset --> Dataset
        Status: OK
        Expression: define hierarchical ruleset accountingEntry (variable rule ACCOUNTING_ENTRY) is
                        B = C - D errorcode "Balance (credit-debit)" errorlevel 4;
                        N = A - L errorcode "Net (assets-liabilities)" errorlevel 4
                    end hierarchical ruleset;

                    DS_r := hierarchy(BOP, accountingEntry rule ACCOUNTING_ENTRY non_zero dataset)
                            [aggr OBS_VALUE_N := sum (OBS_VALUE) group by ACCOUNTING_ENTRY];
                    BOP Dataset

        Description: Hierarchical Rulsets are Vertical validations apply to a
        component over a set of data point.

        Git Branch: 309-validation-operators-tests.
        Goal: Vertical validations shall be used to validate the consistency of
        one component between different data points. In other words, the consistency
        of multiple rows of data.
        """

        code = '1-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1;
        sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
        sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
        sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
        sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

        end datapoint ruleset;

        DS_r := check_datapoint (BOP, signValidation all) [calc TEST := substr(error_code, 4)];

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: 309-validation-operators-tests.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1;
        sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
        sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
        sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
        sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

        end datapoint ruleset;

        DS_r := check_datapoint (BOP, signValidation all) [rename error_level to level];

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: 309-validation-operators-tests.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
        FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
        sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
        sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1;
        sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
        sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
        sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
        sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

        end datapoint ruleset;

        DS_r := check_datapoint (BOP, signValidation all) [aggr errorlevel_N := sum (error_level) group by ACCOUNTING_ENTRY];

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: 309-validation-operators-tests.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        DS_r := check (BOP > 0 errorcode 111 errorlevel 1 imbalance - BOP all)[rename error_level to level];

        Git Branch: 309-validation-operators-tests.
        Goal: Check the result of check.
        """
        code = '1-1-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        DS_r := check (BOP > 0 errorcode 111 errorlevel 1 imbalance - BOP all)
                [calc attribute At_1:= "EP"];

        Git Branch: 309-validation-operators-tests.
        Goal: Check the result of check.
        """
        code = '1-1-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        DS_r := check (BOP > 0 errorcode 111 errorlevel 1 imbalance - BOP all)
                [aggr errorlevel_N := sum (error_level) group by ACCOUNTING_ENTRY];

        Git Branch: 309-validation-operators-tests.
        Goal: Check the result of check.
        """
        code = '1-1-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_426(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK
        define datapoint ruleset DMIDCheckRegex (variable OBS_VALUE, REP_COUNTRY, CURRENCY) is
            DF1:
                when
                REP_COUNTRY in{"CY"} and
                CURRENCY in {"TO1"}
                then
                match_characters (OBS_VALUE, "^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])")
                and length(OBS_VALUE) > 0
                and length(OBS_VALUE) < 20
                errorcode "DF1"
                errorlevel 1    
        end datapoint ruleset;

        INPUT_NULL_CHECK :=
            check(
                isnull(BIS_LOC_STATS # OBS_VALUE)
                errorcode "not_null"
                errorlevel 1
                invalid
                );
        DF1 := check_datapoint (BIS_LOC_STATS,DMIDCheckRegex);

        test_result := INPUT_NULL_CHECK
                    [drop bool_var, 'imbalance'];
        test_result_2 := DF1
                    [rename 'errorcode' to errc, 'errorlevel' to errl];

        Git Branch: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/426
        Goal: Check imabalnce errorcode and errorlevel after check.
        """
        code = 'GL_426'
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_446_1(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK
        """
        code = 'GL_446_1'
        number_inputs = 1
        references_names = ["1", "2"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_446_2(self):
        '''
        Description: Left_join review.
        Jira issue: VTLEN 540.
        Git Branch: feat-VTLEN-540-left_join-review.
        Goal: Check Semantic Exception.
        '''
        code = 'GL_446_2'
        number_inputs = 1
        error_code = "1-1-10-8"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=error_code
        )

    def test_GL_446_3(self):
        '''
        Description: Left_join review.
        Jira issue: VTLEN 540.
        Git Branch: feat-VTLEN-540-left_join-review.
        Goal: Check Semantic Exception.
        '''
        code = 'GL_446_3'
        number_inputs = 1
        error_code = "1-1-10-8"
        self.NewSemanticExceptionTest(
            code=code,
            number_inputs=number_inputs,
            exception_code=error_code
        )

    def test_GL_cs_22(self):
        """
        eschaped characters in the hierarchical ruleset have to be replaced by the corresponding character
        '_T' -> 'T'
        """
        code = 'GL_cs_22'
        number_inputs = 1
        references_names = ["1", "2", "3", "4", "5", "6", "7"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

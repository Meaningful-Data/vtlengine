from pathlib import Path

from testSuite.Helper import TestHelper


class TestDataPointRuleset(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class DatapointRulesetTests(TestDataPointRuleset):
    """
    Group 1
    """

    classTest = 'rulesets.DatapointRulesetTests'

    def test_1(self):
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

        DS_r := check_datapoint (BOP, signValidation);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
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

        DS_r := check_datapoint (BOP, signValidation all);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
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

        DS_r := check_datapoint (BOP, signValidation all_measures);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY, INT_ACC_ITEM,
    FUNCTIONAL_CAT, INSTR_ASSET, OBS_VALUE) is
    sign1c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "G" then OBS_VALUE > 0 errorcode "sign1c" errorlevel 1;
    sign2c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "GA" then OBS_VALUE > 0 errorcode "sign2c" errorlevel 1;
    sign3c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "S" then OBS_VALUE > 0 errorcode "sign3c" errorlevel 1;
    sign4c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "IN1" then OBS_VALUE > 0 errorcode "sign4c" errorlevel 2;
    sign9: when INT_ACC_ITEM = "D4Q" and FUNCTIONAL_CAT = "D" and INSTR_ASSET = "FL" then OBS_VALUE > 0 errorcode "sign9" errorlevel 3;
    sign10: when INT_ACC_ITEM = "D45" and FUNCTIONAL_CAT = "P" and INSTR_ASSET = "F" then OBS_VALUE > 0 errorcode "sign10" errorlevel 4

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
    FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
    sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4;
    sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
    sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
    sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
    sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
    sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation all);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
    FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
    sign1c: when AE = "C" and IAI = "G" then O > 0;
    sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1;
    sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c";
    sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c";
    sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
    sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorlevel 4

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation all);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
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
    sign1c: when if AE = "C" and IAI = "G" then false else true then O > 0 errorcode "sign1c" errorlevel 1;
    sign2c: when AE = "C" and IAI = "GA" then O > 0 errorcode "sign2c" errorlevel 1;
    sign3c: when if AE = "C" and IAI = "S" then true else false then O > 0 errorcode "sign3c" errorlevel 1;
    sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
    sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
    sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation all);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
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

    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as ACCOUNTING_ENTRY,
    INT_ACC_ITEM as INT_ACC_ITEM, FUNCTIONAL_CAT as FUNCTIONAL_CAT,INSTR_ASSET as INSTR_ASSET,
    OBS_VALUE as OBS_VALUE) is
    sign1c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "G" then OBS_VALUE > 0 errorcode "sign1c" errorlevel 1;
    sign2c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "GA" then OBS_VALUE > 0 errorcode "sign2c" errorlevel 1;
    sign3c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "S" then OBS_VALUE > 0 errorcode "sign3c" errorlevel 1;
    sign4c: when ACCOUNTING_ENTRY = "C" and INT_ACC_ITEM = "IN1" then OBS_VALUE > 0 errorcode "sign4c" errorlevel 2;
    sign9: when INT_ACC_ITEM = "D4Q" and FUNCTIONAL_CAT = "D" and INSTR_ASSET = "FL" then OBS_VALUE > 0 errorcode "sign9" errorlevel 3;
    sign10: when INT_ACC_ITEM = "D45" and FUNCTIONAL_CAT = "P" and INSTR_ASSET = "F" then OBS_VALUE > 0 errorcode "sign10" errorlevel 4

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation all);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
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
    sign2c: when AE = "C" and IAI = "GA" then nvl (O,0) errorcode "sign2c" errorlevel 1;
    sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
    sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
    sign9: when IAI = "D4Q" and FC = "D" and IA = "FL" then O > 0 errorcode "sign9" errorlevel 3;
    sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
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

    define datapoint ruleset signValidation (variable ACCOUNTING_ENTRY as AE, INT_ACC_ITEM as IAI,
    FUNCTIONAL_CAT as FC, INSTR_ASSET as IA, OBS_VALUE as O) is
    sign1c: when AE = "C" and IAI = "G" then O > 0 errorcode "sign1c" errorlevel 1;
    sign2c: when AE = "C" and IAI = "GA" then nvl (O,0) errorcode "sign2c" errorlevel 1;
    sign3c: when AE = "C" and IAI = "S" then O > 0 errorcode "sign3c" errorlevel 1;
    sign4c: when AE = "C" and IAI = "IN1" then O > 0 errorcode "sign4c" errorlevel 2;
    sign9: when IAI = "D" and FC = "D" and IA = "S" then nvl (O,0) errorcode "sign9" errorlevel 3;
    sign10: when IAI = "D45" and FC = "P" and IA = "F" then O > 0 errorcode "sign10" errorlevel 4

    end datapoint ruleset;

    DS_r := check_datapoint (BOP, signValidation);

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #291-tests-datapoint-rulesets-dpr.
        Goal: Check the result of datapoint rulesets.
        """
        code = '1-1-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # with value domains
    def test_11(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #144
        Goal: 
        """
        code = '1-1-1-11'
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

    # If rule, then name has to be equal to variable rule is siganture
    def test_12(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #144
        Goal: Has to be an exception.
        """
        code = '1-1-1-12'
        number_inputs = 1

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code="1-1-10-3")

    # Value domain
    def test_13(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #144
        Goal: Has to be an exception because types of variable and vd are differents.
        """
        code = '1-1-1-13'
        number_inputs = 1

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code="1-1-10-5")

    def test_14(self):
        """
        define datapoint ruleset
        Dataset --> Dataset
        Status: OK

        Description: This operator defines a persistent Data Point Ruleset named
                     rulesetName that can be used for validation purposes.

        Git Branch: #144
        Goal: Has to be an exception.
        """
        code = '1-1-1-14'
        number_inputs = 1

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code="1-1-10-3")

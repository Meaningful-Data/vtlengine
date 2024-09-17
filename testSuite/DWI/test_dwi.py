from pathlib import Path

from testSuite.Helper import TestHelper


class DWIHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class Membership(DWIHelper):
    """
        Membership tests for datasets without identifiers
    """

    classTest = 'DWIHelper.Membership'

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1#INSTTTNL_SCTR;';
        Description: Membership correct loading.
        Git Branch: feat-200-DWI-membership.
        Goal: Check Result.
        """
        code = 'GL_200-1'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1#BLNC_SHT_TTL_CRRNCY + DS_2#BLNC_SHT_TTL_CRRNCY;
        Description: Should raise a semantic exception
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_1'
        number_inputs = 2

        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ calc Me_AUX := BLNC_SHT_TTL_CRRNCY + 10];
        Description:
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1 + DWI_1
        Description: Should raise a semantic exception
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_3'
        number_inputs = 2

        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_4(self):
        """
        Status: OK
        Expression: DS_r := sqrt(DS_1);
        Description: Try for unary op is working
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class Aggregate(DWIHelper):
    """
        Aggregate tests for datasets without identifiers as result
    """

    classTest = 'DWIHelper.Aggregate'

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Status: OK
        Expression: DS_r := avg (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := min (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := max (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := median (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Status: OK
        Expression: DS_r := sum (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_7'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_8'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Status: OK
        Expression: DS_r := var_pop (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_9'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Status: OK
        Expression: DS_r := var_samp (DS_1);';
        Description: Perform the Aggregate operation correctly.
        Git Branch: feat-62-DWI-aggregate.
        Goal: Check Result.
        """
        code = 'GL_62_10'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # Aggregate tests for datasets without identifiers as input

    def test_GL_218_5(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(one record)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_6(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(without record)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_7(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(null record)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_8(self):
        """
        Status: OK
        Expression: DS_r := count (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.(nulls and record, the result should be 1)
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_9(self):
        """
        Status: OK
        Expression: DS_r := avg (DS_1);
        Description: Aggregate tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_10(self):
        """
        Status: OK
        Expression: DS_r := avg (DS_1);
        Description: Aggregate tests for datasets to check an error.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_10'
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_12(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ aggr Me_aux := max(BLNC_SHT_TTL_CRRNCY) group by Id_1];
        Description: Aggregate tests for datasets without identifiers as input. Id_1 is not in DS_1.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_12'
        number_inputs = 1
        message = "1-1-1-10"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_13(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ aggr Me_aux := max(BLNC_SHT_TTL_CRRNCY) group by NMBR_EMPLYS];
        Description: Aggregate tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_13'
        number_inputs = 1
        message = "1-1-1-10"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)


class Clause(DWIHelper):
    """
        Clause operator tests for datasets without identifiers as result
    """

    classTest = 'DWIHelper.Clause'

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ sub Id_1 = "AA"]';
        Description: Perform the Sub operation correctly.
        Git Branch: feat-201-DWI-sub.
        Goal: Check Result.
        """
        code = 'GL_201_1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ rename BLNC_SHT_TTL_CRRNCY to Me_1 ];
        Description: Perform the rename operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_1'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ drop BLNC_SHT_TTL_CRRNCY ]
        Description: Perform the drop operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_2'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ keep BLNC_SHT_TTL_CRRNCY ]
        Description: Perform the keep operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_3'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_1 := BLNC_SHT_TTL_CRRNCY + ANNL_TRNVR_CRRNCY + NMBR_EMPLYS ]
        Description: Perform the calc operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result.
        """
        code = 'GL_202_4'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ filter BLNC_SHT_TTL_CRRNCY + ANNL_TRNVR_CRRNCY <= 10000 ]
        Description: Perform the filter operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result (empty).
        """
        code = 'GL_202_5'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Status: OK
        Expression: DS_r := DS_1 [ filter BLNC_SHT_TTL_CRRNCY + ANNL_TRNVR_CRRNCY >= 10000 ]
        Description: Perform the filter operation correctly.
        Git Branch: feat-202-DWI-clause.
        Goal: Check Result (with one row).
        """
        code = 'GL_202_6'
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_14(self):
        """
        Status: OK. Commented as pivot is not implemented.
        Expression: DS_r := DS_1[ pivot BLNC_SHT_TTL_CRRNCY, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_14'
        number_inputs = 1

        message = "pivot not implemented"
        # self.SemanticExceptionTest(code=code, number_inputs=number_inputs, exception_message=message)

    def test_GL_218_15(self):
        """
        Status: OK. Commented as pivot is not implemented.
        Expression: DS_r := DS_1[ pivot Id_1, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.

        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_15'
        number_inputs = 1
        message = "pivot not implemented"
        # self.SemanticExceptionTest(code=code, number_inputs=number_inputs, exception_message=message)

    def test_GL_218_16(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ unpivot BLNC_SHT_TTL_CRRNCY, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_16'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_17(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ unpivot Id_1, NMBR_EMPLYS];
        Description: Pivot tests for datasets without identifiers as input.
                    Should raise semantic error, raise a error on the evaluate, this is wrong
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_17'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_18(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ sub NMBR_EMPLYS = 10.0];
        Description: subspace tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_18'
        number_inputs = 1
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_19(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ sub Id_1 = "C"];
        Description: subspace tests for datasets with identifiers as input. and the id doesn't match.
        Git Issue: #218.
        Goal: Check Result.
        """
        code = 'GL_218_19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_218_20(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ drop Id_1 ];
        Description: 
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_20'
        number_inputs = 1
        message = "1-1-6-2"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_21(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ keep BLNC_SHT_TTL_CRRNCY ][ drop BLNC_SHT_TTL_CRRNCY ];
        Description: keep + drop tests for datasets without identifiers as input.
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_21'
        number_inputs = 1
        message = "1-1-6-12"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_22(self):
        """
        Status: OK
        Expression: DS_r := DS_1[ calc identifier Id_1 := BLNC_SHT_TTL_CRRNCY + NMBR_EMPLYS];
        Description:
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_22'
        number_inputs = 1
        message = "1-1-1-16"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)


class Join(DWIHelper):
    """
        Clause operator tests for datasets without identifiers as result
    """

    classTest = 'DWIHelper.Join'

    # DS join DWI
    def test_GL_218_23(self):
        """
        Status: OK
        Expression:DS_r := inner_join (DS_1, DWI_1);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_23'
        number_inputs = 2
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_25(self):
        """
        Status: OK
        Expression:DS_r := left_join (DS_1, DWI_1);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_25'
        number_inputs = 2
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_26(self):
        """
        Status: OK
        Expression:DS_r := full_join (DS_1, DWI_1);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_26'
        number_inputs = 2
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_27(self):
        """
        Status: OK
        Expression: DS_r := cross_join (DS_1, DWI_1 as dw rename dw#BLNC_SHT_TTL_CRRNCY to col1, dw#ANNL_TRNVR_CRRNCY to col2, dw#NMBR_EMPLYS to col3);
        Description: 
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_27'
        number_inputs = 2
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_218_28(self):
        """
        Status: OK
        Expression: DS_r := cross_join (DS_1, DWI_1);
        Description: 
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_28'
        number_inputs = 2
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    # DWI join DWI

    def test_GL_218_29(self):
        """
        Status: OK
        Expression:DS_r := inner_join (DWI_1, DWI_2);
        Description: The error message should be the op is forbidden because has no sense or something else
        Git Issue: #218.
        Goal: Check Exception.
        """
        code = 'GL_218_29'
        number_inputs = 2
        message = "1-3-27"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

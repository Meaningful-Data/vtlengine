from pathlib import Path
from testSuite.Helper import TestHelper


class TestAggregateHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"

class AggregateOperatorsTest(TestAggregateHelper):
    """
    Group 1
    """

    classTest = 'aggregate.AggregateOperatorsTest'

    def test_1(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := min (DS_1 [rename Me_1 to Me_1A, Me_2 to Me_2A] group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min operator.
        """
        code = '1-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := min (DS_1 [filter Id_1 = "A"] group by Id_2);
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min operator.
        """
        code = '1-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:=min (DS_1#Me_2 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min operator.
        """
        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:=min(DS_1 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min operator.
        """
        code = '1-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := max (DS_1 [filter Id_1 = "B"] [rename Me_1 to Me_1A, Me_2 to Me_2A]
                                 group by Id_2);
                    DS_1 Dataset

        Description: The operator returns the maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := max (DS_1 [filter Id_2 = "XX"] [keep At_1] group by Id_1);
                    DS_1 Dataset

        Description: The operator returns the maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= max (Me_1) group by Id_3];
                    DS_1 Dataset

        Description: The operator returns the maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sum (DS_1 [drop Me_2] group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the sum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the sum operator.
        """
        code = '1-1-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := max (DS_1# Me_1 + DS_2# Me_1 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-9'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sum (DS_1# Me_1 - DS_2# Me_1 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the sum of the input values.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the sum operator.
        """
        code = '1-1-1-10'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Average value: avg
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := avg (DS_1# Me_1 * DS_2# Me_1 group by Id_3);
                    DS_1,DS_2 Dataset

        Description: The operator returns the average of the input values.
        Note: The engine does not give the expected result

        Git Branch: #273 aggregate-operators.

        Goal: Check the performance of the avg operator.
        """
        code = '1-1-1-11'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Average value: avg
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := avg (DS_1 [filter Id_3 = 2020] [rename Me_1 to Me_1A, Me_2 to Me_2A];
                                 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the average of the input values.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the avg operator.
        """
        code = '1-1-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := count ( DS_1 [filter Id_1 = "A"] group by Id_2 );
                    DS_1 Dataset

        Description: The operator returns the number of the input Data Points.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the count operator.
        """
        code = '1-1-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= count () group by Id_1];
                    DS_1 Dataset

        Description: The operator returns the number of the input Data Points.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the count operator.
        """
        code = '1-1-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := max (if DsCond#Id_2 = "A" then DsThen else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-15'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := min (if DsCond#Id_2 = "A" then DsThen else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-16'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := avg (if DsCond#Id_2 = "A" then DsThen else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-17'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sum (if DsCond#Id_2 = "A" then DsThen else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-18'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := count (if DsCond#Id_2 = "A" then DsThen else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-19'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := count (if DsCond#Id_2 = "A" then false else true group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-20'
        number_inputs = 3
        message = "1-1-9-12"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_21(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := count (if DsCond#Id_2 = "A" then false else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-21'
        number_inputs = 3
        message = "1-1-1-1"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_22(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := max (if DsCond#Id_2 = "YY" then true else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-22'
        number_inputs = 3
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        if-then-else: if
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sum (if DsCond#Id_2 = "A" then false else DsElse group by Id_3);
                    DS_1 Dataset

        Description: The if operator returns thenOperand if condition evaluates to true,
                    elseOperand otherwise.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the if-then-else operator.
        """
        code = '1-1-1-23'
        number_inputs = 3
        message = "1-1-1-1"

        # TODO: Review this test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_25(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [aggr Me_1:= count () group by Id_1] [rename Id_1 to Id_1A];
                    DS_1 Dataset

        Description: The operator returns the number of the input Data Points.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the count operator.
        """
        code = '1-1-1-25'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [rename Id_1 to Id_1A] [aggr Me_3:= count () group by Id_1A];
                    DS_1 Dataset

        Description: The operator returns the number of the input Data Points.
        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the count operator.
        """
        code = '1-1-1-26'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sum (DS_1 [filter Id_2 = "XX"] [rename Id_1 to Id_1A] [drop Me_2]
                                 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the sum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the sum operator.
        """
        code = '1-1-1-27'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:=max (DS_1#Me_2 group by Id_3);
                    DS_1 Dataset

        Description: The operator returns the maximum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-28'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_29(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3:= max (Me_1) group by Id_2] as d1,
                           DS_1 [aggr Me_4:= max (Me_2) group by Id_2] as d2,
                           DS_2 [aggr Me_5:= max (Me_11) group by Id_2] as d3,
                           DS_2 [aggr Me_6:= max (Me_22) group by Id_2] as d4);
                    DS_1 Dataset

        Description: The operator returns the maximum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-29'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3:= min (Me_1) group by Id_2] as d1,
                           DS_1 [aggr Me_4:= min (Me_2) group by Id_2] as d2,
                           DS_2 [aggr Me_5:= min (Me_11) group by Id_2] as d3,
                           DS_2 [aggr Me_6:= min (Me_22) group by Id_2] as d4);
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min operator.
        """
        code = '1-1-1-30'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3:= sum (Me_1) group by Id_2] as d1,
                           DS_1 [aggr Me_4:= sum (Me_2) group by Id_2] as d2,
                           DS_2 [aggr Me_5:= sum (Me_11) group by Id_2] as d3,
                           DS_2 [aggr Me_6:= sum (Me_22) group by Id_2] as d4);
                    DS_1 Dataset

        Description: The operator returns the sum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the sum operator.
        """
        code = '1-1-1-31'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_32(self):
        """
        Average value: avg
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3:= avg (Me_1) group by Id_2] as d1,
                           DS_1 [aggr Me_4:= avg (Me_2) group by Id_2] as d2,
                           DS_2 [aggr Me_5:= avg (Me_11) group by Id_2] as d3,
                           DS_2 [aggr Me_6:= avg (Me_22) group by Id_2] as d4);
                    DS_1 Dataset

        Description: The operator returns the average of the input values.
        Note: In the output data structure there is a type discrepancy between
        number and integer.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the avg operator.
        """
        code = '1-1-1-32'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_33(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3:= count () group by Id_2] as d1,
                    DS_2 [aggr Me_4:= count () group by Id_2] as d2);
                    DS_1,DS_2 Dataset

        Description: The operator returns the number of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the count operator.
        """
        code = '1-1-1-33'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        """
        min,max,sum,count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3:= min (Me_1) group by Id_2] as d1,
                            DS_1 [aggr Me_4:= max (Me_2) group by Id_2] as d2,
                            DS_2 [aggr Me_5:= sum (Me_11) group by Id_2] as d3,
                            DS_2 [aggr Me_6:= count () group by Id_2] as d4);
                    DS_1 Dataset

        Description: The operator returns the minimum, maximum, sum, number
                     of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max,sum,count operator.
        """
        code = '1-1-1-34'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_3:= max (Me_1) group by Id_2] as d1);
                    DS_1 Dataset

        Description: The operator returns the maximum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-35'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_3:= max (Me_1) group by Id_2] as d1,
                                      DS_1 [aggr Me_4:= max (Me_2) group by Id_2] as d2);
                    DS_1 Dataset

        Description: The operator returns the maximum value of the input values.
        Note: There is a discrepancy between semantic and base.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-36'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_37(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_3:= max (Me_1) group by Id_2] as d1,
                                      DS_2 [aggr Me_4:= max (Me_11) group by Id_2] as d2);
                    DS_1 Dataset

        Description: The operator returns the maximum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the max operator.
        """
        code = '1-1-1-37'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_3:= min (Me_1) group by Id_2] as d1,
                                      DS_2 [aggr Me_4:= min (Me_11) group by Id_2] as d2);
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min operator.
        """
        code = '1-1-1-38'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_3:= sum (Me_1) group by Id_2] as d1,
                                      DS_2 [aggr Me_4:= sum (Me_11) group by Id_2] as d2);
                    DS_1 Dataset

        Description: The operator returns the sum value of the input values.
        Note: The engine does not give the expected result

        Git Branch: #273 aggregate-operators.

        Goal: Check the performance of the sum operator.
        """
        code = '1-1-1-39'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        Average: avg
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_3:= avg (Me_1) group by Id_2] as d1,
                                      DS_2 [aggr Me_4:= avg (Me_11) group by Id_2] as d2);
                    DS_1 Dataset

        Description: The operator returns the sum value of the input values.
        Note: In the output data structure there is a type discrepancy
        between number and integer.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the sum operator.
        """
        code = '1-1-1-40'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        """
        min,max,sum,count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= full_join (DS_1 [aggr Me_1:= min (Me_1) group by Id_2] as d1,
                            DS_2 [aggr Me_2:= max (Me_1) group by Id_2] as d2,
                            DS_3 [aggr Me_3:= sum (Me_1) group by Id_2] as d3,
                            DS_4 [aggr Me_4:= count () group by Id_2] as d4);
                    DS_1 Dataset

        Description: The operator returns the minimum, maximum, sum, number
                     of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max,sum,count operator.
        """
        code = '1-1-1-41'
        number_inputs = 4
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        min,max,sum,count,avg
        Dataset --> Dataset
        Status: BUG
        Expression: DS_r:= full_join (DS_1 [aggr Me_1:= min (Me_1) group by Id_2] as d1,
                            DS_2 [aggr Me_2:= max (Me_1) group by Id_2] as d2,
                            DS_3 [aggr Me_3:= sum (Me_1) group by Id_2] as d3,
                            DS_4 [aggr Me_4:= count () group by Id_2] as d4,
                            DS_5 [aggr Me_5:= avg (Me_1) group by Id_2] as d5);
                    DS_1 Dataset

        Description: The operator returns the minimum, maximum, sum, number, average
                     of the input values.
        Note: In the output data structure there is a type discrepancy
        between number and integer.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max,sum,count,avg operator.
        """
        code = '1-1-1-42'
        number_inputs = 5
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= DS_1 [aggr Me_3:= min (Me_1), Me_4 := max (Me_1) group by Id_2];
                    DS_1 Dataset

        Description: The operator returns the minimum, maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max,sum,count,avg operator.
        """
        code = '1-1-1-43'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_44(self):
        """
        min,max,sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3 := min (Me_1) group by Id_2] as d1,
                           DS_2 [aggr Me_4 := max (Me_1) group by Id_2] as d2)
                           [aggr Me_5 := sum (Me_4) group by Id_2];
                    DS_1,DS_2  Dataset

        Description: The operator returns the minimum, maximum, sum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max,sum operator.
        """
        code = '1-1-1-44'
        number_inputs = 2
        error_code = "1-1-1-10"

        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=error_code)

    def test_45(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3 := min (Me_1) group by Id_2] as d1,
                                       DS_2 [aggr Me_4 := max (Me_1) group by Id_2] as d2)
                                       [rename Id_2 to Id_2A];
                    DS_1,DS_2  Dataset

        Description: The operator returns the minimum, maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max operator.
        """
        code = '1-1-1-45'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        """
        min,max,count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3 := min (Me_1) group by Id_2] as d1,
                    DS_2 [aggr Me_4 := max (Me_1) group by Id_2] as d2)
                    [rename Me_3 to Me_3A, Me_4 to Me_4A] [aggr Me_5:= count () group by Id_2];
                    DS_1,DS_2  Dataset

        Description: The operator returns the minimum, maximum, number of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max operator.
        """
        code = '1-1-1-46'
        number_inputs = 2
        message = "1-3-1"

        # TODO: the error code does not have any sense with this vtl code
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_47(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3 := min (Me_1) group by Id_2] as d1,
                    DS_2 [aggr Me_4 := max (Me_1) group by Id_2] as d2)
                    [rename Me_3 to Me_3A, Me_4 to Me_4A] [drop Me_4A];
                    DS_1,DS_2  Dataset

        Description: The operator returns the minimum, maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max operator.
        """
        code = '1-1-1-47'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_48(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3 := min (Me_1) group by Id_2] as d1,
                    DS_2 [rename Me_1 to Me_1A] as d2) [rename Me_3 to Me_3A];
                    DS_1,DS_2  Dataset

        Description: The operator returns the minimum, maximum of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max operator.
        """
        code = '1-1-1-48'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_49(self):
        """
        min,max,count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r:= inner_join (DS_1 [aggr Me_3 := min (Me_1) group by Id_2] as d1,
                    DS_2 [rename Me_1 to Me_1A] as d2) [rename Me_3 to Me_3A]
                    [aggr Me_4:= count () group by Id_2];
                    DS_1,DS_2  Dataset

        Description: The operator returns the minimum, maximum, number of the input values.

        Git Branch: #273 aggregate-operators.
        Goal: Check the performance of the min,max operator.
        """
        code = '1-1-1-49'
        number_inputs = 2
        message = "1-3-1"

        # TODO: the error code does not have any sense with this vtl code
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        assert True

    def test_GL_315_1(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_315_1'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_2(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression:
            RSDL_MRTY_WA := inner_join(
                DBTR_FCT_DYS,
                VRBLS_AGGR
                aggr RSDL_MTRTY_DYS_WGHTD_AVRG := sum (
                    RSDL_MTRTY*PST_D_DYS/MAX_RSDL_MTRTY_DYS) group by DT_RFRNC, ENTTY_RIAD_CD
                );

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_315_2'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_3(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_315_3'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_4(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_315_4'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_5(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_315_5'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_6(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_315_6'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_7(self):
        """
        min,max,count
        Dataset --> Dataset
        Status:
        Expression:

        Description: The operator returns the minimum, maximum, number of the input values.

        Git Branch: #fix-378-count.
        Goal: Check the performance of the min,max,count operator.
        """
        code = 'GL_315_7'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_8(self):
        """
        min,max
        Dataset --> Dataset
        Status: OK
        Expression:

        Description:

        Git Branch: #315.
        Goal: Check the performance of aggr inside join.
        """
        code = 'GL_315_8'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_9(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: Without using

        Git Branch: #315.
        Goal: Check the performance of aggr inside join.
        """
        code = 'GL_315_9'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_10(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: With redundant using

        Git Branch: #315.
        """
        code = 'GL_315_10'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_11(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: inner join with aggregate

        Git Branch: #315.
        """
        code = 'GL_315_11'
        number_inputs = 2
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_315_12(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: The operator returns the minimum, maximum, number of the input values.

        Git Branch: #315
        Goal: Show case b1 ambiguity.
        """
        code = 'GL_315_12'
        number_inputs = 2
        message = "1-1-13-4"

        # TODO: Review this test
        # self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_323_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: 315 with several statements

        Git Branch: #325.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_323_1'
        number_inputs = 1
        references_names = ["1", "2", "3", "4"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_411_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: 315 with several statements

        Git Branch: #325.
        Goal: Check the performance of aggr inside join
        """
        code = 'GL_411_1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_GL_466_1(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/466

        Git Branch: #466.
        Goal: aggr (count) with null values and fill_time_series
        """
        code = 'GL_466_1'
        number_inputs = 1
        # references_names = ["1", "2", "3"]

        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)
        message = "1-1-1-16"
        self.NewSemanticExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_GL_466_2(self):
        """
        Dataset --> Dataset
        Status: OK
        Expression:

        Description: https://gitlab.meaningfuldata.eu/vtl-suite/vtlengine/-/issues/466

        Git Branch: #466.
        Goal: aggr (count) with null values
        """
        code = 'GL_466_2'
        number_inputs = 1
        references_names = ["1", "2", "3"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

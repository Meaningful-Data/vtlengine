from pathlib import Path

from tests.Helper import TestHelper


class TestAggregateTypeChecking(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class AggregateOperatorsDatasetTypeChecking(TestAggregateTypeChecking):
    """
    Group 10
    """

    classTest = "AggregateOperators.AggregateOperatorsDatasetTypeChecking"

    # average operator

    def test_1(self):
        """
        Operation with int and number, me1 int me2 number.
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: All the measures are involved and the results should be type Number.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-1"
        # 10 For group aggregate operators
        # 1 For group dataset
        # 1 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Test 1 plus nulls.
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: The nulls are ignored in the average.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-2"
        # 10 For group aggregate operators
        # 1 For group dataset
        # 2 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with  measure String, and more measures, if one measure fails the exception is raised.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-3"
        # 10 For group aggregate operators
        # 1 For group dataset
        # 3 Number of test
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-4"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-5"
        number_inputs = 1

        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_6(self):
        """
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with time_period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-6"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-7"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_8(self):
        """
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-8"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_9(self):
        """
        Average with time again
        Status: OK
        Expression: DS_r := avg ( DS_1 group by Id_1);
        Description: Average with time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-9"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # count operator

    def test_10(self):
        """
        Count with integer and number
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: There are measures int and num without nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-10"
        # 10 For group aggregate operators.
        # 1 For group datasets
        # 10 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Count one measure
        Status: OK
        Expression: DS_r := count ( Me_1 group by Id_1);
        Description: Special case of count with a component, should ignore nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-11"
        # 10 For group aggregate operators.
        # 1 For group datasets
        # 11 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Count with string
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: There isnt fail because take the null as empty string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-12"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Count with time
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Time with null, counts the null
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-13"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Count with date
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Date with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-14"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Count with time period
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Time Period with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-15"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        Count with duration
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Duration with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-16"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Count with boolean
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Measure Boolean with null, doesn't count the null, we think that should.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-17"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        Count with number and integer
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: There are measures int and num with nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-18"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        Count with number and integer
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: Example that takes the most left measure.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-19"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        Status: OK
        Expression: DS_r := count ( DS_1 group by Id_1);
        Description: count with grouping by identifier string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-20"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # max operator
    def test_21(self):
        """
        Max for integers
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: All the measures Integers are involved and the results should be type Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-21"
        # 10 for group aggregate operators
        # 1 For group datasets
        # 21 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        Max for integers and numbers
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: All the measures Integers and Numbers are involved and the results should be the parent type.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-22"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        Max for integers and string
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for string is ok on a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-23"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        Max for integers and time
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for time takes the the mayor number but not the mayor time,
                    2008M1/2008M12 should be the result not 2010M1/2010M12.
                    In this test is not present but max fails for time with nulls .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-24"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_25(self):
        """
        Max for integers and date
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for date and nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-25"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        Max for integers and time period
        Status: OK.
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max doesnt work with nulls and diferent time_period in the same id (2012Q2,2012M12).
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        text="DS_r := max(DS_1 group by Id_1);"
        code = "10-1-26"
        number_inputs = 1
        exception_code = "2-1-19-20"

        self.NewSemanticExceptionTest(
            text=text, code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_27(self):
        """
        Max for integers and duration
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max doesnt work with nulls and take the max duration in a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-27"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        Max for integers and boolean
        Status: OK
        Expression: DS_r := max ( DS_1 group by Id_1);
        Description: Max for booleans takes True as max.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-28"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # median operator
    def test_29(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers  The nulls are ignored in the average and the result measures has the type number .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-29"
        # 10 For group aggregate operators
        # 1 For group dataset
        # 29 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_30(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and numbers, all the meaures are calculated.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-30"
        # 10 For group aggregate operators
        # 1 For group dataset
        # 30 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_31(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-31"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_32(self):
        """
        Status:
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-32"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_33(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-33"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_34(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Time_Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-34"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_35(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-35"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_36(self):
        """
        Status: OK
        Expression: DS_r := median ( DS_1 group by Id_1);
        Description: Median for integers and Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-36"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # min operator
    def test_37(self):
        """
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for integers. All the measures Integers are involved and the results should be type Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-37"
        # 10 For group aggregate operators.
        # 1 For group datasets
        # 37 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        Status: OK
        Expression: DS_r := DS_r := min ( DS_1 group by Id_1);
        Description: Min for integers and numbers with nulls and the results should be the parent type.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-38"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_39(self):
        """
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for string is ok on a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-39"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for time takes the the minor number but not the minor time,
                    2010M1/2010M12 should be the result not 2008M1/2008M12.
                    In this test is not present but max fails for time with nulls .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-40"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # references_names = ["DS_r"]
        #
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_41(self):
        """
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-41"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        Status: TO REVIEW
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min Time Period doesnt work with nulls and diferent time_period in the same id (2012Q2,2012M12)..
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        text = "DS_r := min(DS_1 group by Id_1);"
        code = "10-1-42"
        number_inputs = 1
        exception_code = "2-1-19-20"

        self.NewSemanticExceptionTest(
            text=text, code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_43(self):
        """
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min doesnt work with nulls and take the min duration in a lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-43"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_44(self):
        """
        Status: OK
        Expression: DS_r := min ( DS_1 group by Id_1);
        Description: Min for booleans takes False as min.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-44"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # stddev_pop operator

    def test_45(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: population standard deviation for integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-45"
        # 10 For group aggregation
        # 1 For group dataset
        # 45 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_46(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: Population standard deviation for integers and numbers
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-46"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_47(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for strings.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-47"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_48(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-48"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_49(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-49"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_50(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for time period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-50"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_51(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-51"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_52(self):
        """
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 group by Id_1);
        Description: stddev_pop for boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-52"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # stddev_samp operator

    def test_53(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-53"
        # 10 For group aggregate oerators
        # 1 For datasets
        # 53 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_54(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-54"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_55(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-55"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_56(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-56"
        # 2 For group join
        # 2 For group identifiers
        # 4 For clause- for the moment only op cross_join
        # 1 Number of test
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_57(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-57"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_58(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Time_period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-58"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_59(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-59"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_60(self):
        """
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 group by Id_1);
        Description: stddev_samp for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-60"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # sum operator

    def test_61(self):
        """
        Status: OK? TO REVIEW doubt about result type
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum inputs Integer and results type Number.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-61"
        # 10 For group aggregate operators
        # 1 For group datasets
        # 61 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_62(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: sum Numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-62"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_63(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-63"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_64(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-64"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_65(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-65"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_66(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-66"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_67(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-67"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_68(self):
        """
        Status: OK
        Expression: DS_r := sum ( DS_1 group by Id_1);
        Description: Sum for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-68"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # var_pop operator

    def test_69(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: Population standard deviation for integers .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-69"
        # 10 For group aggregate operators
        # 1 For group datasets
        # 69 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_70(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: Population standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-70"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_71(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-71"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_72(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-72"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_73(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-73"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_74(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-74"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_75(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Duration .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-75"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_76(self):
        """
        Status: OK
        Expression: DS_r := var_pop ( DS_1 group by Id_1);
        Description: var_pop for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-76"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # var_samp operator

    def test_77(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-77"
        # 10 For group aggregate operators
        # 1 For group dataset
        # 77 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_78(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: Sample standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-1-78"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_79(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-79"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_80(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-80"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_81(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-81"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_82(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-82"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_83(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-83"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_84(self):
        """
        Status: OK
        Expression: DS_r := var_samp ( DS_1 group by Id_1);
        Description: var_samp for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-1-84"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )


class AggregateOperatorsComponentTypeChecking(TestAggregateTypeChecking):
    """
    Group 2
    """

    classTest = "AggregateOperators.AggregateOperatorsComponentTypeChecking"

    # average operator

    def test_1(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: Avg between Integer, the result have to be type number.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-1"
        # 10 For group aggregate operators
        # 2 For group components
        # 1 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: average between Number with nulls.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-2"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: Average with String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-3"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_4(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: Average with Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-4"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_5(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: Average with Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-5"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_6(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: Average with time_period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-6"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_7(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: Average with Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-7"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_8(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := avg ( Me_1 ) , Me_4 := avg ( Me_2 ) group by Id_1];
        Description: average with Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-8"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # count operator
    def test_9(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: count with Integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-9"
        # 10 For group aggregate operators
        # 2 For group component
        # 9 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: count with Number.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-10"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description:  count with string
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-11"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: Count with Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-12"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: Count with Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-13"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: Count with Time_period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-14"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: Count with Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-15"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := count ( Me_1 ) , Me_4 := count ( Me_2 ) group by Id_1];
        Description: Count with boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-16"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # max operator
    def test_17(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: Max for Integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-17"
        # 10 For group aggregate operators
        # 2 For group component
        # 17 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: max for Numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-18"
        # 10 For group Aggregate operators
        # 2 For group Components
        # 18 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: max for integers and string Follows lexicographic order.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-19"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        Status: TO REVIEW
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: Max for time. Show the same problems that the dataset group
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-20"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # references_names = ["DS_r"]
        #
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: max for date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-21"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        Status: TO REVIEW
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: max for time period. Show the same problems that the dataset group
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        text = "DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];"
        code = "10-2-22"
        number_inputs = 1
        exception_code = "2-1-19-20"

        self.NewSemanticExceptionTest(
            text=text, code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_23(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: max for integers and duration. Seems that happens the same errors presents in dataset group
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-23"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := max ( Me_1 ) , Me_4 := max ( Me_2 ) group by Id_1];
        Description: max for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-24"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # median operator
    def test_25(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for integers. The result type has to be number not integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-25"
        # 10 For group aggregate operators
        # 2 For group component
        # 25 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for integers and numbers. The result has to be number not integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-26"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-27"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_28(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-28"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_29(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-29"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_30(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-30"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_31(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-31"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_32(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := median ( Me_1 ) , Me_4 := median ( Me_2 ) group by Id_1];
        Description: Median for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-32"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # min operator
    def test_33(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-33"
        # 10 For group aggregate operators
        # 2 For group component
        # 33 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_34(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for Integers and Numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-34"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_35(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: min for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-35"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_36(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for  time. The same problems that in dataset group
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-36"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)
        # references_names = ["DS_r"]
        #
        #
        # self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_37(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-37"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_38(self):
        """
        Status: TO REVIEW
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for time period.THe same problems that earlier
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        text = "DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];"
        code = "10-2-38"
        number_inputs = 1
        exception_code = "2-1-19-20"

        self.NewSemanticExceptionTest(
            text=text, code=code, number_inputs=number_inputs, exception_code=exception_code
        )

    def test_39(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for duration. should choose the minimun duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-39"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_40(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := min ( Me_1 ) , Me_4 := min ( Me_2 ) group by Id_1];
        Description: Min for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-40"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    # stddev_pop operator

    def test_41(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: population standard deviation for integers. The result's type should be number not integer .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-41"
        # 10 For group aggregate operators.
        # 2 For group Component
        # 41 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_42(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: population standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-42"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_43(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: stddev_pop for string.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-43"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_44(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: stddev_pop for time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-44"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_45(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: stddev_pop for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-45"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_46(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: stddev_pop for time period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-46"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_47(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: stddev_pop for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-47"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_48(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_pop ( Me_1 ) , Me_4 := stddev_pop ( Me_2 ) group by Id_1];
        Description: stddev_pop for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-48"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # stddev_samp operator

    def test_49(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: sample standard deviation for integers. Result should be type Number not Integer .
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-49"
        # 10 For group Aggregate
        # 2 For group Component
        # 49 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_50(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: population standard deviation for integers and numbers
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-50"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_51(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: stddev_samp for String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-51"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_52(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: stddev_samp for Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-52"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_53(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: stddev_samp for Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-53"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_54(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: stddev_samp for time period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-54"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_55(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: stddev_samp for Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-55"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_56(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := stddev_samp ( Me_1 ) , Me_4 := stddev_samp ( Me_2 ) group by Id_1];
        Description: stddev_samp for Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-56"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # sum operator

    def test_57(self):
        """
        Status: OK?doubt about type
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: Sum between integers, the result type is integer.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-57"
        # 10 For group aggregate
        # 2 For group component
        # 57 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_58(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: sum Number.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-58"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_59(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: sum String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-59"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_60(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: Sum Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-60"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_61(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: Sum Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-61"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_62(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: Sum Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-62"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_63(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: Sum Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-63"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_64(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := sum ( Me_1 ) , Me_4 := sum ( Me_2 ) group by Id_1];
        Description: Sum Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-64"

        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # var_pop operator

    def test_65(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: sample standard deviation for integers.The result should be type Number
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-65"
        # 10 For group aggregate
        # 2 For group components
        # 65 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_66(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: population standard deviation for integers and numbers. The result should be type Number
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-66"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_67(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: var_pop String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-67"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_68(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: var_pop Time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-68"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_69(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: var_pop Date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-69"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_70(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: var_pop Time Period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-70"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_71(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: var_pop Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-71"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_72(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_pop ( Me_1 ) , Me_4 := var_pop ( Me_2 ) group by Id_1];
        Description: var_pop Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-72"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    # var_samp operator

    def test_73(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: sample standard deviation for integers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-73"
        # 10 For group aggregate
        # 2 For group component
        # 73 Number of test
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_74(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: population standard deviation for integers and numbers.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Result.
        """
        code = "10-2-74"
        number_inputs = 1
        references_names = ["DS_r"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_75(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: var_samp String.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-75"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_76(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: var_samp time.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-76"
        number_inputs = 1
        message = "0-1-1-12"
        self.DataLoadExceptionTest(code=code, number_inputs=number_inputs, exception_code=message)

    def test_77(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: var_samp date.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-77"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_78(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: var_samp time period.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-78"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_79(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: var_samp Duration.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-79"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

    def test_80(self):
        """
        Status: OK
        Expression: DS_r := DS_1[aggr Me_3 := var_samp ( Me_1 ) , Me_4 := var_samp ( Me_2 ) group by Id_1];
        Description: var_samp Boolean.
        Git Branch: tests-21-aggregation-types-checking-tests.
        Goal: Check Exception.
        """
        code = "10-2-80"
        number_inputs = 1
        message = "1-1-1-1"
        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

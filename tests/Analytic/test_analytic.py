from pathlib import Path

from tests.Helper import TestHelper


class AnalyticHelper(TestHelper):
    base_path = Path(__file__).parent
    filepath_VTL = base_path / "data" / "vtl"
    filepath_valueDomain = base_path / "data" / "ValueDomain"
    filepath_json = base_path / "data" / "DataStructure" / "input"
    filepath_csv = base_path / "data" / "DataSet" / "input"
    filepath_out_json = base_path / "data" / "DataStructure" / "output"
    filepath_out_csv = base_path / "data" / "DataSet" / "output"
    filepath_sql = base_path / "data" / "sql"


class AnalyticOperatorsTest(AnalyticHelper):
    """
    Group 1
    """

    classTest = "analytic.AnalyticOperatorsTest"

    def test_1(self):
        """
        First value: first_value
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := first_value ( DS_1 over ( partition by Id_1, Id_2
                                         order by Id_3 asc) );
                    DS_1 Dataset

        Description: The operator returns the first value (in the value order)
        of the set of Data Points that belong to the same analytic window as
        the current Data Point.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the first_value operator.
        """
        code = "1-1-1-1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Last Value: last_value
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := last_value ( DS_1 over ( partition by Id_1, Id_2
                                        order by Id_3 data points between 1
                                        preceding and 1 following ) );
                    DS_1 Dataset

        Description: The operator returns the last value (in the value order)
        of the set of Data Points that belong to the same analytic window as
        the current Data Point.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the last_value operator.
        """
        code = "1-1-1-2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Lag: lag
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := lag ( DS_1 , 1 over ( partition by Id_1, Id_2 order by Id_3 ) );
                    DS_1 Dataset

        Description: In the ordered set of Data Points of the current partition,
        the operator returns the value(s) taken from the Data Point at the
        specified physical offset prior to the current Data Point.
        If defaultValue is not specified then the value returned when the offset
        goes outside the partition is NULL.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the lag operator.
        """
        code = "1-1-1-3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Lead: lead
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := lead ( DS_1 , 1 over ( partition by Id_1 , Id_2 order by Id_3 ) );
                    DS_1 Dataset

        Description: In the ordered set of Data Points of the current partition,
        the operator returns the value(s) taken from the Data Point at
        the specified physical offset beyond the current Data Point.
        If defaultValue is not specified, then the value returned when the offset
        goes outside the partition is NULL.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the lead operator.
        """
        code = "1-1-1-4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Rank: rank
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me2 := rank ( over ( partition by Id_1 ,
                                  Id_2 order by Me_1 ) ) ];
                    DS_1 Dataset

        Description: The operator returns an order number (rank) for each Data
        Point, starting from the number 1 and following the order specified in
        the orderClause. If some Data Points are in the same order according to
        the specified orderClause, the same order number (rank) is assigned and
        a gap appears in the sequence of the assigned ranks.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the rank operator.
        """
        code = "1-1-1-5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Ratio to report: ratio_to_report
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := ratio_to_report ( DS_1 over ( partition by Id_1, Id_2 ) );
                    DS_1 Dataset

        Description: The operator returns the ratio between the value of the
        current Data Point and the sum of the values of the partition which the
        current Data Point belongs to.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the ratio_to_report operator.
        """
        code = "1-1-1-6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := max ( DS_1 over ( partition by Id_1, Id_2 order by
                                 Id_3 data points between 1 preceding and 1 following) );
                    DS_1 Dataset

        Description: The operator returns the maximum of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the max operator.
        """
        code = "1-1-1-7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := min ( DS_1 over ( partition by Id_1, Id_2 order by
                                 Id_3 data points between 1 preceding and 1 following) );
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the min operator.
        """
        code = "1-1-1-8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Average value: avg
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := avg ( DS_1 over ( partition by Id_1 order by Id_2
                                 data points between 3 preceding and current data point) );
                    DS_1 Dataset

        Description: The operator returns the average of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the avg operator.
        """
        code = "1-1-1-9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := count ( DS_1 over ( partition by Id_1, Id_2 order by
                                   Id_3 data points between 2 preceding
                                   and current data point) );
                    DS_1 Dataset

        Description: The operator returns the number of the input Data Points.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the count operator.
        """
        code = "1-1-1-10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Median value: median
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := median ( DS_1 over ( partition by Id_1, Id_2 order by
                                    Id_3 data points between 2 preceding and 1 following) );
                    DS_1 Dataset

        Description: The operator returns the median value of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the median operator.
        """
        code = "1-1-1-11"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := sum ( DS_1 over ( partition by Id_1, Id_2 order by
                                 Id_3 data points between current data point
                                 and 2 following) );
                    DS_1 Dataset

        Description: The operator returns the sum of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the sum operator.
        """
        code = "1-1-1-12"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Population standard deviation: stddev_pop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := stddev_pop ( DS_1 over ( partition by Id_1, Id_2 order by
                                 Id_3 data points between current data point
                                 and 3 following) );
                    DS_1 Dataset

        Description: The operator returns the “population standard deviation”
                     of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the stddev_pop operator.
        """
        code = "1-1-1-13"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Sample standard deviation: stddev_samp
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := stddev_samp ( DS_1 over ( partition by Id_1, Id_2 order by
                                 Id_3 data points between 3 preceding and 2 following) );
                    DS_1 Dataset

        Description: The operator returns the “sample standard deviation”
                     of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the stddev_samp operator.
        """
        code = "1-1-1-14"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Population variance: var_pop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := var_pop ( DS_1 over ( partition by Id_1, Id_2 order
                                     by Id_3 data points between 1 preceding and
                                     current data point ) );
                    DS_1 Dataset

        Description: The operator returns the “population variance” of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the var_pop operator.
        """
        code = "1-1-1-15"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        Sample variance: var_samp
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := var_samp ( DS_1 over ( partition by Id_1, Id_2 order
                                     by Id_3 data points between current data
                                     point and 2 following) );
                    DS_1 Dataset

        Description: The operator returns the sample variance of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the var_samp operator.
        """
        code = "1-1-1-16"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Lag: lag
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := lag ( DS_1 , 1, -1 over ( partition by Id_1, Id_2 order by Id_3 ) );
                    DS_1 Dataset

        Description: In the ordered set of Data Points of the current partition,
        the operator returns the value(s) taken from the Data Point at the
        specified physical offset prior to the current Data Point.
        If defaultValue is not specified then the value returned when the offset
        goes outside the partition is -1.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the lag operator.
        """
        code = "1-1-1-17"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_18(self):
        """
        First value: first_value
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := first_value ( DS_1 over ( partition by Id_1, Id_2
                                         order by Id_3 data points between 3 following and 1 following) );
                    DS_1 Dataset

        Description: The operator returns the first value (in the value order)
        of the set of Data Points that belong to 3 preceding data point.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the first_value operator.
        """
        code = "1-1-1-18"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        First value: first_value
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := first_value ( DS_1 over ( partition by Id_1, Id_2
                                         order by Id_3 data points between 1 following and 3 following) );
                    DS_1 Dataset

        Description: The operator returns the first value (in the value order)
        of the set of Data Points that belong to 3 preceding data point.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the first_value operator.
        """
        code = "1-1-1-19"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)


class AnalyticOperatorsWithCalcTest(AnalyticHelper):
    """
    Group 2
    """

    classTest = "analytic.AnalyticOperatorsWithCalcTest"

    def test_1(self):
        """
        First value: first_value
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := first_value ( Me_1 over ( partition by Id_1, Id_2
                                  order by Id_3 data points between 1
                                  preceding and 1 following) ) ] [ calc Me_22 := first_value
                                  ( Me_2 over ( partition by Id_1, Id_2
                                  order by Id_3 data points between 1
                                  preceding and 1 following) ) ];
                    DS_1 Dataset

        Description: The operator returns the first value (in the value order)
        of the set of Data Points that belong to the same analytic window as
        the current Data Point.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the first_value with calc operator.
        """
        code = "2-1-1-1"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_2(self):
        """
        Last Value: last_value
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := last_value ( Me_1 over ( partition by Id_1, Id_2
                                  order by Id_3 data points between 1
                                  preceding and 1 following) ) ] [ calc Me_22 := last_value
                                  ( Me_2 over ( partition by Id_1, Id_2
                                  order by Id_3 data points between 1
                                  preceding and current data point) ) ];
                    DS_1 Dataset

        Description: The operator returns the last value (in the value order)
        of the set of Data Points that belong to the same analytic window as
        the current Data Point.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the last_value with calc operator.
        """
        code = "2-1-1-2"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_3(self):
        """
        Lag: lag
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := lag ( Me_1, 1 over (partition by
                                  Id_1, Id_2 order by Id_3 ) ) ] [ calc Me_22 := lag
                                  ( Me_2, 2 over (partition by
                                  Id_1, Id_2 order by Id_3 ) ) ];
                    DS_1 Dataset

        Description: In the ordered set of Data Points of the current partition,
        the operator returns the value(s) taken from the Data Point at the
        specified physical offset prior to the current Data Point.
        If defaultValue is not specified then the value returned when the offset
        goes outside the partition is NULL.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the lag with calc operator.
        """
        code = "2-1-1-3"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_4(self):
        """
        Lead: lead
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := lead ( Me_1, 2 over (partition by
                                  Id_1, Id_2 order by Id_3 ) ) ] [ calc Me_22 := lead
                                  ( Me_2, 1 over (partition by Id_1, Id_2 order by Id_3 ) ) ];
                    DS_1 Dataset

        Description: In the ordered set of Data Points of the current partition,
        the operator returns the value(s) taken from the Data Point at
        the specified physical offset beyond the current Data Point.
        If defaultValue is not specified, then the value returned when the offset
        goes outside the partition is NULL.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the lead with calc operator.
        """
        code = "2-1-1-4"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_5(self):
        """
        Rank: rank
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := rank ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Me_1 ) ) ] [ calc Me_22 := rank
                                  ( Me_2 over ( partition by Id_1, Id_2 order by Me_2 ) ) ];
                    DS_1 Dataset

        Description: The operator returns an order number (rank) for each Data
        Point, starting from the number 1 and following the order specified in
        the orderClause. If some Data Points are in the same order according to
        the specified orderClause, the same order number (rank) is assigned and
        a gap appears in the sequence of the assigned ranks.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the rank with calc operator.
        """
        code = "2-1-1-5"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_6(self):
        """
        Ratio to report: ratio_to_report
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := ratio_to_report ( Me_1 over ( partition
                                    by Id_1, Id_2 ) ) ] [ calc Me_22 := ratio_to_report
                                    (Me_2 over ( partition by Id_1, Id_2 ) ) ];
                    DS_1 Dataset

        Description: The operator returns the ratio between the value of the
        current Data Point and the sum of the values of the partition which the
        current Data Point belongs to.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the ratio_to_report with calc operator.
        """
        code = "2-1-1-6"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_7(self):
        """
        Maximum value: max
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := max (Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between 1
                                  preceding and 1 following) ) ] [ calc Me_22
                                  := max ( Me_2 over ( partition by Id_1, Id_2 order
                                  by Id_3 data points between current data point
                                  and 1 following) ) ];
                    DS_1 Dataset

        Description: The operator returns the maximum of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the max with calc operator.
        """
        code = "2-1-1-7"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_8(self):
        """
        Minimum value: min
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := min ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between current
                                  data point and 1 following) ) ] [ calc Me_22
                                  := min ( Me_2 over ( partition by Id_1, Id_2 order
                                  by Id_3 data points between 1 preceding and 1
                                  following) ) ];
                    DS_1 Dataset

        Description: The operator returns the minimum value of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the min with calc operator.
        """
        code = "2-1-1-8"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_9(self):
        """
        Average value: avg
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := avg ( Me_1 over ( partition by Id_1 order by Id_2
                                  data points between 3 preceding and current data point) ) ]
                                 [ calc Me_22 := avg ( Me_2 over ( partition by Id_1 order by Id_1 asc, Id_2 desc
                                  data points between 1 preceding and current data point) ) ];
                    DS_1 Dataset

        Description: The operator returns the average of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the avg with calc operator.
        """
        code = "2-1-1-9"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_10(self):
        """
        Counting the number of data points: count
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_1 := count ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between 3 preceding
                                  and current data point) )];
                    DS_1 Dataset

        Description: The operator returns the number of the input Data Points.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the count with calc operator.
        """
        code = "2-1-1-10"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_11(self):
        """
        Median value: median
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := median ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between 2 preceding
                                  and 1 following) )] [ calc Me_22 := median ( Me_2 over
                                  ( partition by Id_1, Id_2 order by Id_3 data points
                                  between 1 preceding and 1 following) )];
                    DS_1 Dataset

        Description: The operator returns the median value of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the median with calc operator.
        """
        code = "2-1-1-11"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_12(self):
        """
        Sum: sum
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := sum ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between current data point
                                  and 2 following) )] [ calc Me_22 := sum ( Me_2 over
                                  ( partition by Id_1, Id_2 order by Id_3 data points
                                  between current data point and 1 following) )];
                    DS_1 Dataset

        Description: The operator returns the sum of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the sum with calc operator.
        """
        code = "2-1-1-12"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_13(self):
        """
        Population standard deviation: stddev_pop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := stddev_pop ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between current data point
                                  and 3 following) )] [ calc Me_22 := stddev_pop ( Me_2 over
                                  ( partition by Id_1, Id_2 order by Id_3 data points
                                  between current data point and 3 following) )];
                    DS_1 Dataset

        Description: The operator returns the “population standard deviation”
                     of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the stddev_pop with calc operator.
        """
        code = "2-1-1-13"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_14(self):
        """
        Sample standard deviation: stddev_samp
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := stddev_samp ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between 3 preceding
                                  and 2 following) )] [ calc Me_22 := stddev_samp ( Me_2 over
                                  ( partition by Id_1, Id_2 order by Id_3 data points
                                  between 3 preceding and 2 following) )];
                    DS_1 Dataset

        Description: The operator returns the “sample standard deviation”
                     of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the stddev_samp with calc operator.
        """
        code = "2-1-1-14"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_15(self):
        """
        Population variance: var_pop
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := var_pop ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between 1 preceding and
                                  current data point ) )] [ calc Me_22 := var_pop ( Me_2 over
                                  ( partition by Id_1, Id_2 order by Id_3 data points
                                  between 1 preceding and current data point ) )];
                    DS_1 Dataset

        Description: The operator returns the “population variance” of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the var_pop with calc operator.
        """
        code = "2-1-1-15"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_16(self):
        """
        Sample variance: var_samp
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := var_samp ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Id_3 data points between current data
                                  point and 2 following) )] [ calc Me_22 := var_samp
                                  ( Me_2 over ( partition by Id_1, Id_2 order by Id_3
                                  data points between current data point and 2 following) )];
                    DS_1 Dataset

        Description: The operator returns the sample variance of the input values.

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the var_samp with calc operator.
        """
        code = "2-1-1-16"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_17(self):
        """
        Rank: rank
        Dataset --> Dataset
        Status: OK
        Expression: DS_r := DS_1 [ calc Me_11 := rank ( Me_1 over ( partition by Id_1,
                                  Id_2 order by Me_1 ) ) ];
                    DS_1 Dataset

        Description: Exception check if partition symbol is not a Identifier

        Git Branch: #246 test-for-analytic-operators.
        Goal: Check the performance of the rank with calc operator.
        """
        code = "2-1-1-17"
        number_inputs = 1
        error_code = "1-1-3-2"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=error_code
        )

    def test_18(self):
        """
        Status: OK
        Expression: dsr := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
        [calc
            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS data points between 1 preceding and unbounded preceding)),
            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following))
        ];
                    DS_1 Dataset

        Description: Window managing with 2 preceding and unbounded
        """
        code = "2-1-1-18"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_19(self):
        """
        Status: OK
        Expression: dsr := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
        [calc
            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS data points between unbounded preceding and 1 preceding)),
            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following))
        ];
                    DS_1 Dataset

        Description: Window managing with 2 preceding and unbounded (same result as test_18)
        """
        code = "2-1-1-19"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_20(self):
        """
        Status: OK
        Expression: dsr := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
        [calc
            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS data points between unbounded following and 1 following)),
            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following))
        ];
                    DS_1 Dataset

        Description: Window managing with 2 following and unbounded
        """
        code = "2-1-1-20"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_21(self):
        """
        Status: OK
        Expression: dsr := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
        [calc
            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS data points between 1 following and unbounded following)),
            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following))
        ];
                    DS_1 Dataset

        Description: Window managing with 2 preceding and unbounded (same result as test_20)
        """
        code = "2-1-1-21"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_22(self):
        """
        Status: OK
        Expression: dsr := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
        [calc
            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS range between 1 preceding and unbounded preceding)),
            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following))
        ];
                    DS_1 Dataset

        Description: Window managing on range mode with 2 preceding and unbounded
        """
        code = "2-1-1-22"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_23(self):
        """
        Status: OK
        Expression: dsr := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
        [calc
            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS range between 1 following and unbounded following)),
            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following))
        ];
                    DS_1 Dataset

        Description: Window managing with 2 preceding and unbounded (same result as test_22)
        """
        code = "2-1-1-23"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_24(self):
        """
        Status: OK
        Expression: INCRMNTL_3PPC_P := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
                                        [calc
                                            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS range between unbounded preceding and 0 preceding)),
                                            CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following)),
                                            NMBR_INSTRMNT_SCRD := count(PRTCTN_ID over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID data points between unbounded preceding and unbounded following))
                                        ];
                    DS_1 Dataset

        Description: Management of different types of results in Analytic (Series, Dataframe (with and without _y on result)
        """
        code = "2-1-1-24"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_25(self):
        """
        Status: OK
        Expression: INCRMNTL_3PPC_P := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
                                        [calc
                                            MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID
                                            order by THRD_PRTY_PRRTY_CLMS range between unbounded preceding and 1 preceding))];
                    DS_1 Dataset

        Description: Window creation with range and nan or float with decimals on column
        """
        code = "2-1-1-25"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_26(self):
        """
        Status: OK
        Expression: INCRMNTL_3PPC_P :=
                            ANCRDT_INSTRMNT_PRTCTN_RCVD_C
                                [calc
                                    MX_3PPC := max(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID order by THRD_PRTY_PRRTY_CLMS range between unbounded preceding and 1 preceding)),
                                    CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID range between 0 preceding and 0 following)),
                                    NMBR_INSTRMNT_SCRD := count(PRTCTN_ID over (partition by DT_RFRNC, OBSRVD_AGNT_CD, PRTCTN_ID data points between unbounded preceding and unbounded following))
                                ];
                    DS_1 Dataset

        Description: Window creation with range and nan or float with decimals on column
        """
        code = "2-1-1-26"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_27(self):
        """
        Status: OK
        Expression: DS_r := ANCRDT_INSTRMNT_PRTCTN_RCVD_C
                                [calc
                                    CNT_3PPC := count(THRD_PRTY_PRRTY_CLMS over(partition by DT_RFRNC order by DT_RFRNC range between 0 preceding and 0 following))
                                ];
                    DS_1 Dataset

        Description: Window creation with range and nan or float with decimals on column
        """
        code = "2-1-1-27"
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(code=code, number_inputs=number_inputs, references_names=references_names)

    def test_28(self):
        """
        Status: OK
        Expression: DS_r := DS_1[calc Me_2 := rank(over(order by Id_1))];
                    DS_1 Dataset
        Description: Error with no measures inside Analytic operator (see issue #34, on comments)
        """

        code = "2-1-1-28"
        number_inputs = 1
        message = "1-1-1-8"

        self.NewSemanticExceptionTest(
            code=code, number_inputs=number_inputs, exception_code=message
        )

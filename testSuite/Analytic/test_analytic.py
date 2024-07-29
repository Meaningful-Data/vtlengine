import json
from pathlib import Path
from typing import Dict, List, Any
from unittest import TestCase

import pandas as pd

from API import create_ast
from DataTypes import SCALAR_TYPES
from Interpreter import InterpreterAnalyzer
from Model import Component, Role, Dataset


class AnalyticHelper(TestCase):
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


class AnalyticOperatorsTest(AnalyticHelper):
    """
    Group 1
    """

    classTest = 'analytic.AnalyticOperatorsTest'

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
        code = '1-1-1-1'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-2'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-3'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-4'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-5'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-6'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-7'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-8'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-9'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-10'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-11'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-12'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-13'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-14'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-15'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-16'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-17'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-18'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)

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
        code = '1-1-1-19'
        number_inputs = 1
        references_names = ["1"]

        self.BaseTest(text=None, code=code, number_inputs=number_inputs, references_names=references_names)
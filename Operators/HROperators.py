import operator
from copy import copy
from typing import Dict

import pandas as pd
from pandas import DataFrame

import Operators
from AST.Grammar.tokens import HIERARCHY
from DataTypes import Boolean, Number
from Model import DataComponent, Dataset, Role, Component


def get_measure_from_dataset(dataset: Dataset, code_item: str) -> DataComponent:
    measure_name = dataset.get_measures_names()[0]
    return DataComponent(name=code_item, data=dataset.data[measure_name],
                         data_type=dataset.components[measure_name].data_type,
                         role=dataset.components[measure_name].role,
                         nullable=dataset.components[measure_name].nullable)


class HRComparison(Operators.Binary):

    @classmethod
    def imbalance_func(cls, x, y):
        if pd.isnull(x) or pd.isnull(y):
            return None
        return x - y

    @staticmethod
    def hr_func(x, y, hr_mode, func):
        # In comments, it is specified the condition for evaluating the rule,
        # so we delete the cases that does not satisfy the condition
        # (line 6509 of the reference manual)
        if hr_mode == 'non_null':
            # If all the involved Data Points are not NULL
            if pd.isnull(x) or pd.isnull(y):
                return "REMOVE_VALUE"
        elif hr_mode == 'non_zero':
            # If at least one of the involved Data Points is <> zero
            if not (pd.isnull(x) and pd.isnull(y)) and (x == 0 and y == 0):
                return "REMOVE_VALUE"
        elif hr_mode in ('partial_null', 'partial_zero'):
            if pd.isnull(x) and pd.isnull(y):
                return "REMOVE_VALUE"

        return func(x, y)

    @classmethod
    def apply_hr_func(cls, left_series, right_series, hr_mode, func):
        return left_series.combine(right_series, lambda x, y: cls.hr_func(x, y, hr_mode, func))

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: DataComponent, hr_mode: str) -> Dataset:
        result_components = {comp_name: copy(comp) for comp_name, comp in
                             left_operand.components.items() if comp.role == Role.IDENTIFIER}
        result_components['bool_var'] = Component(name='bool_var',
                                                  data_type=Boolean,
                                                  role=Role.MEASURE,
                                                  nullable=True)
        result_components['imbalance'] = Component(name='imbalance',
                                                   data_type=Number,
                                                   role=Role.MEASURE,
                                                   nullable=True)
        return Dataset(name=f"{left_operand.name}{cls.op}{right_operand.name}",
                       components=result_components,
                       data=None)

    @classmethod
    def evaluate(cls, left: Dataset, right: DataComponent, hr_mode: str) -> Dataset:
        result = cls.validate(left, right, hr_mode)
        result.data = left.data.copy()
        measure_name = left.get_measures_names()[0]
        result.data['bool_var'] = cls.apply_hr_func(left.data[measure_name], right.data,
                                                    hr_mode, cls.op_func)
        result.data['imbalance'] = cls.apply_hr_func(left.data[measure_name], right.data,
                                                     hr_mode, cls.imbalance_func)
        # Removing datapoints that should not be returned
        # (we do it below imbalance calculation
        # to avoid errors on different shape)
        result.data = result.data[result.data['bool_var'] != "REMOVE_VALUE"]
        result.data.drop(measure_name, axis=1, inplace=True)
        return result


class HREqual(HRComparison):
    op = '='
    py_op = operator.eq


class HRGreater(HRComparison):
    op = '>'
    py_op = operator.gt


class HRGreaterEqual(HRComparison):
    op = '>='
    py_op = operator.ge


class HRLess(HRComparison):
    op = '<'
    py_op = operator.lt


class HRLessEqual(HRComparison):
    op = '<='
    py_op = operator.le


class HRBinNumeric(Operators.Binary):

    @classmethod
    def evaluate(cls, left: DataComponent, right: DataComponent) -> DataComponent:
        result_data = cls.apply_operation_two_series(left.data, right.data)
        return DataComponent(name=f"{left.name}{cls.op}{right.name}", data=result_data,
                             data_type=left.data_type,
                             role=left.role, nullable=left.nullable)


class HRBinPlus(HRBinNumeric):
    op = '+'
    py_op = operator.add


class HRBinMinus(HRBinNumeric):
    op = '-'
    py_op = operator.sub


class HRUnNumeric(Operators.Unary):

    @classmethod
    def evaluate(cls, operand: DataComponent):
        result_data = cls.apply_operation_component(operand.data)
        return DataComponent(name=f"{cls.op}({operand.name})", data=result_data,
                             data_type=operand.data_type,
                             role=operand.role, nullable=operand.nullable)


class HRUnPlus(HRUnNumeric):
    op = '+'
    py_op = operator.pos


class HRUnMinus(HRUnNumeric):
    op = '-'
    py_op = operator.neg

class HAAssignment(Operators.Binary):

    @classmethod
    def validate(cls, left: Dataset, right: DataComponent) -> Dataset:
        result_components = {comp_name: copy(comp) for comp_name, comp in
                             left.components.items()}
        return Dataset(name=f"{left.name}",
                       components=result_components,
                       data=None)

    @classmethod
    def evaluate(cls, left: Dataset, right: DataComponent) -> Dataset:
        result = cls.validate(left, right)
        measure_name = left.get_measures_names()[0]
        result.data = left.data.copy()
        result.data[measure_name] = right.data
        return result

class Hierarchy(Operators.Operator):

    op = HIERARCHY

    @staticmethod
    def generate_computed_data(computed_dict: Dict[str, DataFrame]) -> DataFrame:
        list_data = list(computed_dict.values())
        df = pd.concat(list_data, axis=0)
        df.reset_index(drop=True, inplace=True)
        return df

    @classmethod
    def validate(cls, dataset: Dataset, computed_dict: Dict[str, DataFrame], output: str) -> Dataset:
        result_components = {comp_name: copy(comp) for comp_name, comp in
                             dataset.components.items()}
        return Dataset(name=dataset.name,
                       components=result_components,
                       data=None)

    @classmethod
    def evaluate(cls, dataset: Dataset, computed_dict: Dict[str, DataFrame], output: str) -> Dataset:
        result = cls.validate(dataset, computed_dict, output)
        computed_data = cls.generate_computed_data(computed_dict)
        if output == "computed":
            result.data = computed_data
            return result

        result.data = pd.concat([dataset.data, computed_data], axis=0, ignore_index=True)
        result.data.drop_duplicates(subset=dataset.get_identifiers_names(), keep='last', inplace=True)
        return result


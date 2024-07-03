from typing import List, Optional, Union

import pandas as pd

import Operators as Operator
from AST.Grammar.tokens import (AVG, COUNT, MAX, MEDIAN, MIN, STDDEV_POP, STDDEV_SAMP, SUM, VAR_POP,
                                VAR_SAMP)
from DataTypes import Integer, Number
from Model import DataComponent, Dataset, Role

ALL_ALLOWED_MODEL_DATA_TYPES = Union[DataComponent, Dataset]


def extract_grouping_identifiers(identifier_names: List[str],
                                 group_op: str,
                                 grouping_components: List[str]) -> List[str]:
    if group_op == 'group by':
        return grouping_components
    elif group_op == 'group except':
        return [comp for comp in identifier_names if comp not in grouping_components]
    else:
        return identifier_names


# noinspection PyMethodOverriding
class Aggregation(Operator.Unary):

    @classmethod
    def validate(cls, operand: Dataset,
                 group_op: Optional[str],
                 grouping_components: Optional[List[str]],
                 having_data: Optional[List[DataComponent]]) -> Dataset:
        result_components = operand.components.copy()
        if group_op is not None:
            for comp_name in grouping_components:
                if comp_name not in operand.components:
                    raise ValueError(f"Component {comp_name} not found in dataset")
                if operand.components[comp_name].role != Role.IDENTIFIER:
                    raise ValueError(f"Component {comp_name} is not an identifier")
            identifiers_to_keep = extract_grouping_identifiers(operand.get_identifiers_names(),
                                                               group_op,
                                                               grouping_components)
            for comp_name, comp in operand.components.items():
                if comp.role == Role.IDENTIFIER and comp_name not in identifiers_to_keep:
                    del result_components[comp_name]
        else:
            for comp_name, comp in operand.components.items():
                if comp.role == Role.IDENTIFIER:
                    del result_components[comp_name]
        # Remove Attributes
        for comp_name, comp in operand.components.items():
            if comp.role == Role.ATTRIBUTE:
                del result_components[comp_name]
        if len(operand.get_measures()) != 1:
            raise ValueError("Only one measure is allowed")
        # Change Measure data type
        for comp_name, comp in result_components.items():
            if comp.role == Role.MEASURE:
                # TODO: Type promotion
                if cls.return_type is not None:
                    comp.data_type = cls.return_type
        if cls.op == COUNT:
            measure_name = operand.get_measures_names()[0]
            new_comp = result_components.pop(measure_name)
            new_comp.name = "int_var"
            result_components["int_var"] = new_comp
        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def evaluate(cls,
                 operand: Dataset,
                 group_op: Optional[str],
                 grouping_columns: Optional[str],
                 having_data: Optional[List[DataComponent]]) -> Dataset:
        result = cls.validate(operand, group_op, grouping_columns, having_data)
        result.data = operand.data.copy()
        measure_name = operand.get_measures_names()[0]

        grouping_keys = result.get_identifiers_names()
        if len(grouping_keys) == 0:
            result_number = result.data[measure_name].agg(cls.py_op)
            result.data = pd.DataFrame(data=[result_number], columns=[measure_name])
            return result
        result.data = result.data[grouping_keys + [measure_name]]
        if cls.op == COUNT:
            result_df = result.data.groupby(grouping_keys).size().reset_index(name='int_var')
        else:
            comps_to_keep = grouping_keys + [measure_name]
            result_df = result.data.groupby(grouping_keys)[comps_to_keep].agg(cls.py_op)
            result_df = result_df[measure_name]
            result_df = result_df.reset_index()
        result.data = result_df
        return result


class Max(Aggregation):
    op = MAX
    py_op = pd.DataFrame.max


class Min(Aggregation):
    op = MIN
    py_op = pd.DataFrame.min


class Sum(Aggregation):
    op = SUM
    type_to_check = Number
    py_op = pd.DataFrame.sum


class Count(Aggregation):
    op = COUNT
    type_to_check = None
    return_type = Integer
    py_op = pd.DataFrame.count


class Avg(Aggregation):
    op = AVG
    type_to_check = Number
    return_type = Number
    py_op = pd.DataFrame.mean


class Median(Aggregation):
    op = MEDIAN
    type_to_check = Number
    return_type = Number
    py_op = pd.DataFrame.median


class PopulationStandardDeviation(Aggregation):
    op = STDDEV_POP
    type_to_check = Number
    return_type = Number

    @classmethod
    def py_op(cls, df):
        return df.std(ddof=0)


class SampleStandardDeviation(Aggregation):
    op = STDDEV_SAMP
    type_to_check = Number
    return_type = Number

    @classmethod
    def py_op(cls, df):
        return df.std(ddof=1)


class PopulationVariance(Aggregation):
    op = VAR_POP
    type_to_check = Number
    return_type = Number

    @classmethod
    def py_op(cls, df):
        return df.var(ddof=0)


class SampleVariance(Aggregation):
    op = VAR_SAMP
    type_to_check = Number
    return_type = Number
    py_op = pd.DataFrame.var

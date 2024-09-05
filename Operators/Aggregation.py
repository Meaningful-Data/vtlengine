import os
from copy import copy
from typing import List, Optional

from DataTypes.TimeHandling import DURATION_MAPPING, DURATION_MAPPING_REVERSED, TimePeriodHandler, \
    TimeIntervalHandler
from Exceptions import SemanticError

if os.getenv('SPARK', False):
    import pyspark.pandas as pd
else:
    import pandas as pd

import Operators as Operator
from AST.Grammar.tokens import (AVG, COUNT, MAX, MEDIAN, MIN, STDDEV_POP, STDDEV_SAMP, SUM, VAR_POP,
                                VAR_SAMP)
from DataTypes import Integer, Number, check_unary_implicit_promotion
from Model import Component, DataComponent, Dataset, Role


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
    def _handle_data_types(cls, data: pd.DataFrame, measures: List[Component], mode: str):
        if cls.op == COUNT:
            return
        if mode == 'input':
            to_replace = [None]
            new_value = ['']
        else:
            to_replace = ['']
            new_value = [None]

        for measure in measures:
            if measure.data_type.__name__ == 'Date':
                if cls.op == MIN:
                    if mode == 'input':
                        # Invalid date only for null values
                        new_value = ['9999-99-99']
                    else:
                        to_replace = ['9999-99-99']
                data[measure.name] = data[measure.name].replace(to_replace, new_value)
            elif measure.data_type.__name__ == 'TimePeriod':
                if mode == 'input':
                    data[measure.name] = data[measure.name].astype(object).map(
                        lambda x: TimePeriodHandler(x),
                        na_action='ignore')
                else:
                    data[measure.name] = data[measure.name].map(
                        lambda x: str(x), na_action='ignore')
            elif measure.data_type.__name__ == 'TimeInterval':
                if mode == 'input':
                    data[measure.name] = data[measure.name].astype(object).map(
                        lambda x: TimeIntervalHandler.from_iso_format(x),
                        na_action='ignore')
                else:
                    data[measure.name] = data[measure.name].map(
                        lambda x: str(x), na_action='ignore')
            elif measure.data_type.__name__ == 'String':
                data[measure.name] = data[measure.name].replace(to_replace, new_value)
            elif measure.data_type.__name__ == 'Duration':
                if mode == 'input':
                    data[measure.name] = data[measure.name].map(lambda x: DURATION_MAPPING[x],
                                                                na_action='ignore')
                else:
                    data[measure.name] = data[measure.name].map(
                        lambda x: DURATION_MAPPING_REVERSED[x], na_action='ignore')

    @classmethod
    def validate(cls, operand: Dataset,
                 group_op: Optional[str],
                 grouping_components: Optional[List[str]],
                 having_data: Optional[List[DataComponent]]) -> Dataset:
        result_components = {k: copy(v) for k, v in operand.components.items()}
        if cls.op not in [COUNT, MIN, MAX] and len(operand.get_measures_names()) == 0:
            raise SemanticError("1-1-2-1", op=cls.op)
        if group_op is not None:
            for comp_name in grouping_components:
                if comp_name not in operand.components:
                    raise SemanticError("1-1-2-4", op=cls.op, id_name=comp_name)
                if operand.components[comp_name].role != Role.IDENTIFIER:
                    raise SemanticError("1-1-2-2", op=cls.op,
                                        id_name=comp_name,
                                        id_type=operand.components[comp_name].role)

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
        # Change Measure data type
        for comp_name, comp in result_components.items():
            if comp.role == Role.MEASURE:
                check_unary_implicit_promotion(comp.data_type, cls.type_to_check)
                if cls.return_type is not None:
                    comp.data_type = cls.return_type
        if cls.op == COUNT:
            for measure_name in operand.get_measures_names():
                result_components.pop(measure_name)
            new_comp = Component(name="int_var", role=Role.MEASURE, data_type=Integer,
                                 nullable=True)
            result_components["int_var"] = new_comp
        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def evaluate(cls,
                 operand: Dataset,
                 group_op: Optional[str],
                 grouping_columns: Optional[str],
                 having_data: Optional[pd.DataFrame]) -> Dataset:
        result = cls.validate(operand, group_op, grouping_columns, having_data)

        grouping_keys = result.get_identifiers_names()
        result.data = operand.data.copy()
        if len(operand.get_measures_names()) == 0:
            if cls.op == COUNT:
                result.data = result.data[grouping_keys].groupby(grouping_keys).size().reset_index(name='int_var')
            else:
                result.data = result.data[grouping_keys].drop_duplicates(keep='first')
            return result
        if len(grouping_keys) == 0 and group_op is not None:
            grouping_keys = operand.get_identifiers_names()
        elif group_op is None:
            grouping_keys = []
        measure_names = operand.get_measures_names()
        result_df = result.data[grouping_keys + measure_names]
        if having_data is not None:
            result_df = result_df.merge(having_data, how='inner', on=grouping_keys)
        comps_to_keep = grouping_keys + measure_names
        if cls.op == COUNT:
            if len(grouping_keys) == 0:
                result_df = result_df.dropna(subset=measure_names, how='any')
                result_df = result_df.count().reset_index(name='int_var')
                result_df = result_df['int_var'].to_frame().drop_duplicates().reset_index(drop=True)
            else:
                # As Count does not include null values,
                # we remove them and merge using the grouping keys,
                # to ensure we do not lose any group that only has null values
                aux_df = result_df.dropna(subset=measure_names, how='any')
                aux_df = aux_df.groupby(grouping_keys).size().reset_index(name='int_var')
                result_df = result_df.drop_duplicates(subset=grouping_keys)[grouping_keys].reset_index(drop=True)
                result_df = result_df.merge(aux_df, how="left", on=grouping_keys)
        else:
            if os.getenv('SPARK', False) and cls.spark_op is not None:
                result_df = cls.spark_op(result_df, grouping_keys)
            else:
                cls._handle_data_types(result_df, operand.get_measures(), mode='input')
                if cls.op == SUM:
                    # Min_count is used to ensure we return null if all elements are null,
                    # instead of 0
                    agg_dict = {measure_name: lambda x: x.sum(min_count=1)
                                for measure_name in measure_names}
                elif cls.py_op.__name__ != 'py_op':
                    agg_dict = {measure_name: cls.py_op.__name__ for measure_name in measure_names}
                else:
                    agg_dict = {measure_name: cls.py_op for measure_name in measure_names}
                if len(grouping_keys) > 0:
                    result_df = result_df.groupby(grouping_keys)[comps_to_keep].agg(agg_dict).reset_index(
                            drop=False)
                else:
                    result_df = result_df[comps_to_keep].agg(agg_dict)
                    if isinstance(result_df, pd.Series):
                        result_df = result_df.to_frame().T
                cls._handle_data_types(result_df, operand.get_measures(), 'result')
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
    # TODO: Median has inconsistent behavior in spark
    #  test 144 has a median of 3, but the result is 2
    op = MEDIAN
    type_to_check = Number
    return_type = Number
    py_op = pd.DataFrame.median

    @classmethod
    def spark_op(cls, df, keys):
        return df.groupby(keys).median().reset_index(drop=False)

        # percentiles = [0.5]  # Median
        # return df.groupby(keys).approxQuantile(percentiles)[0].reset_index(drop=False)


class PopulationStandardDeviation(Aggregation):
    op = STDDEV_POP
    type_to_check = Number
    return_type = Number

    @classmethod
    def py_op(cls, df):
        return df.std(ddof=0)

    @classmethod
    def spark_op(cls, df, keys):
        return df.groupby(keys).std(ddof=0).reset_index(drop=False)


class SampleStandardDeviation(Aggregation):
    op = STDDEV_SAMP
    type_to_check = Number
    return_type = Number

    @classmethod
    def py_op(cls, df):
        return df.std(ddof=1)

    @classmethod
    def spark_op(cls, df, keys):
        return df.groupby(keys).std(ddof=1).reset_index(drop=False)


class PopulationVariance(Aggregation):
    op = VAR_POP
    type_to_check = Number
    return_type = Number

    @classmethod
    def py_op(cls, df):
        return df.var(ddof=0)

    @classmethod
    def spark_op(cls, df, keys):
        return df.groupby(keys).var(ddof=0).reset_index(drop=False)


class SampleVariance(Aggregation):
    op = VAR_SAMP
    type_to_check = Number
    return_type = Number
    py_op = pd.DataFrame.var

    @classmethod
    def spark_op(cls, df, keys):
        return df.groupby(keys).var().reset_index(drop=False)

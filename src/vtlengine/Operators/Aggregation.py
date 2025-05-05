from copy import copy
from typing import Any, List, Optional

import duckdb
import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    AVG,
    COUNT,
    MAX,
    MEDIAN,
    MIN,
    STDDEV_POP,
    STDDEV_SAMP,
    SUM,
    VAR_POP,
    VAR_SAMP,
)
from vtlengine.DataTypes import (
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    String,
    TimeInterval,
    TimePeriod,
    unary_implicit_promotion,
)
from vtlengine.DataTypes.TimeHandling import (
    PERIOD_IND_MAPPING,
    PERIOD_IND_MAPPING_REVERSE,
    TimeIntervalHandler,
    TimePeriodHandler,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role


def extract_grouping_identifiers(
    identifier_names: List[str], group_op: Optional[str], grouping_components: Any
) -> List[str]:
    if group_op == "group by":
        return grouping_components
    elif group_op == "group except":
        return [comp for comp in identifier_names if comp not in grouping_components]
    else:
        return identifier_names


# noinspection PyMethodOverriding
class Aggregation(Operator.Unary):
    @classmethod
    def _handle_data_types(cls, data: pd.DataFrame, measures: List[Component], mode: str) -> None:
        to_replace: List[Optional[str]]
        new_value: List[Optional[str]]
        if cls.op == COUNT:
            return
        if mode == "input":
            to_replace = [None]
            new_value = [""]
        else:
            to_replace = [""]
            new_value = [None]

        for measure in measures:
            if measure.data_type == Date:
                if cls.op == MIN:
                    if mode == "input":
                        # Invalid date only for null values
                        new_value = ["9999-99-99"]
                    else:
                        to_replace = ["9999-99-99"]
                data[measure.name] = data[measure.name].replace(to_replace, new_value)
            elif measure.data_type == TimePeriod:
                if mode == "input":
                    data[measure.name] = (
                        data[measure.name]
                        .astype(object)
                        .map(lambda x: TimePeriodHandler(str(x)), na_action="ignore")
                    )
                else:
                    data[measure.name] = data[measure.name].map(
                        lambda x: str(x), na_action="ignore"
                    )
            elif measure.data_type == TimeInterval:
                if mode == "input":
                    data[measure.name] = (
                        data[measure.name]
                        .astype(object)
                        .map(
                            lambda x: TimeIntervalHandler.from_iso_format(str(x)),
                            na_action="ignore",
                        )
                    )
                else:
                    data[measure.name] = data[measure.name].map(
                        lambda x: str(x), na_action="ignore"
                    )
            elif measure.data_type == String:
                data[measure.name] = data[measure.name].replace(to_replace, new_value)
            elif measure.data_type == Duration:
                if mode == "input":
                    data[measure.name] = data[measure.name].map(
                        lambda x: PERIOD_IND_MAPPING[x],  # type: ignore[index]
                        na_action="ignore",
                    )
                else:
                    data[measure.name] = data[measure.name].map(
                        lambda x: PERIOD_IND_MAPPING_REVERSE[x],  # type: ignore[index]
                        na_action="ignore",
                    )
            elif measure.data_type == Boolean:
                if mode == "result":
                    data[measure.name] = data[measure.name].map(
                        lambda x: Boolean().cast(x), na_action="ignore"
                    )
                    data[measure.name] = data[measure.name].astype(object)

    @classmethod
    def validate(  # type: ignore[override]
        cls,
        operand: Dataset,
        group_op: Optional[str],
        grouping_columns: Any,
        having_data: Any,
    ) -> Dataset:
        result_components = {k: copy(v) for k, v in operand.components.items()}
        if cls.op not in [COUNT, MIN, MAX] and len(operand.get_measures_names()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
        if group_op is not None:
            for comp_name in grouping_columns:
                if comp_name not in operand.components:
                    raise SemanticError(
                        "1-1-1-10",
                        op=cls.op,
                        comp_name=comp_name,
                        dataset_name=operand.name,
                    )
                if operand.components[comp_name].role != Role.IDENTIFIER:
                    raise SemanticError(
                        "1-1-2-2",
                        op=cls.op,
                        id_name=comp_name,
                        id_type=operand.components[comp_name].role,
                    )

            identifiers_to_keep = extract_grouping_identifiers(
                operand.get_identifiers_names(), group_op, grouping_columns
            )
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
        for _, comp in result_components.items():
            if comp.role == Role.MEASURE:
                unary_implicit_promotion(comp.data_type, cls.type_to_check)
                if cls.return_type is not None:
                    comp.data_type = cls.return_type
        if cls.op == COUNT:
            for measure_name in operand.get_measures_names():
                result_components.pop(measure_name)
            new_comp = Component(
                name="int_var", role=Role.MEASURE, data_type=Integer, nullable=True
            )
            result_components["int_var"] = new_comp
        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def _agg_func(
        cls,
        df: pd.DataFrame,
        grouping_keys: Optional[List[str]],
        measure_names: Optional[List[str]],
        having_expression: Optional[str],
    ) -> pd.DataFrame:
        grouping_names = (
            [f'"{name}"' for name in grouping_keys] if grouping_keys is not None else None
        )
        if grouping_names is not None and len(grouping_names) > 0:
            grouping = "GROUP BY " + ", ".join(grouping_names)
        else:
            grouping = ""

        if having_expression is None:
            having_expression = ""

        if measure_names is not None and len(measure_names) == 0 and cls.op == COUNT:
            if grouping_names is not None:
                query = (
                    f"SELECT {', '.join(grouping_names)}, COUNT() AS "
                    f"int_var from df {grouping} {having_expression}"
                )
            else:
                query = f"SELECT COUNT() AS int_var from df {grouping}"
            return duckdb.query(query).to_df()

        if measure_names is not None and len(measure_names) > 0:
            functions = ""
            for e in measure_names:
                e = f'"{e}"'
                if cls.type_to_check is not None and cls.op != COUNT:
                    functions += (
                        f"{cls.py_op}(CAST({e} AS DOUBLE)) AS {e}, "  # Count can only be one here
                    )
                elif cls.op == COUNT:
                    functions += f"{cls.py_op}({e}) AS int_var, "
                    break
                else:
                    functions += f"{cls.py_op}({e}) AS {e}, "
            if grouping_names is not None and len(grouping_names) > 0:
                query = (
                    f"SELECT {', '.join(grouping_names) + ', '}{functions[:-2]} "
                    f"from df {grouping} {having_expression}"
                )
            else:
                query = f"SELECT {functions[:-2]} from df"

        else:
            query = (
                f"SELECT {', '.join(grouping_names or [])} from df {grouping} {having_expression}"
            )

        try:
            return duckdb.query(query).to_df().astype(object)
        except RuntimeError as e:
            if "Conversion" in e.args[0]:
                raise SemanticError("2-3-8", op=cls.op, msg=e.args[0].split(":")[-1])
            else:
                raise SemanticError("2-1-1-1", op=cls.op)

    @classmethod
    def evaluate(  # type: ignore[override]
        cls,
        operand: Dataset,
        group_op: Optional[str],
        grouping_columns: Optional[List[str]],
        having_expr: Optional[str],
    ) -> Dataset:
        result = cls.validate(operand, group_op, grouping_columns, having_expr)

        grouping_keys = result.get_identifiers_names()
        result_df = operand.data.copy() if operand.data is not None else pd.DataFrame()
        measure_names = operand.get_measures_names()
        result_df = result_df[grouping_keys + measure_names]
        if cls.op == COUNT:
            result_df = result_df.dropna(subset=measure_names, how="any")
        cls._handle_data_types(result_df, operand.get_measures(), "input")
        result_df = cls._agg_func(result_df, grouping_keys, measure_names, having_expr)

        cls._handle_data_types(result_df, operand.get_measures(), "result")
        # Handle correct order on result
        aux_df = (
            operand.data[grouping_keys].drop_duplicates()
            if operand.data is not None
            else pd.DataFrame()
        )
        if len(grouping_keys) == 0:
            aux_df = result_df
            aux_df.dropna(subset=result.get_measures_names(), how="all", inplace=True)
            if cls.op == COUNT and len(result_df) == 0:
                aux_df["int_var"] = 0
        elif len(aux_df) == 0:
            aux_df = pd.DataFrame(columns=result.get_components_names())
        else:
            aux_df = pd.merge(aux_df, result_df, how="left", on=grouping_keys)
        if having_expr is not None:
            aux_df.dropna(subset=result.get_measures_names(), how="any", inplace=True)
        result.data = aux_df
        return result


class Max(Aggregation):
    op = MAX
    py_op = "max"


class Min(Aggregation):
    op = MIN
    py_op = "min"


class Sum(Aggregation):
    op = SUM
    type_to_check = Number
    py_op = "sum"


class Count(Aggregation):
    op = COUNT
    type_to_check = None
    return_type = Integer
    py_op = "count"


class Avg(Aggregation):
    op = AVG
    type_to_check = Number
    return_type = Number
    py_op = "avg"


class Median(Aggregation):
    op = MEDIAN
    type_to_check = Number
    return_type = Number
    py_op = "median"


class PopulationStandardDeviation(Aggregation):
    op = STDDEV_POP
    type_to_check = Number
    return_type = Number
    py_op = "stddev_pop"


class SampleStandardDeviation(Aggregation):
    op = STDDEV_SAMP
    type_to_check = Number
    return_type = Number
    py_op = "stddev_samp"


class PopulationVariance(Aggregation):
    op = VAR_POP
    type_to_check = Number
    return_type = Number
    py_op = "var_pop"


class SampleVariance(Aggregation):
    op = VAR_SAMP
    type_to_check = Number
    return_type = Number
    py_op = "var_samp"

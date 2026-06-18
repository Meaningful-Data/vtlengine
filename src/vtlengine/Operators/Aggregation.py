from copy import copy
from typing import Any, List, Optional

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
    Integer,
    Number,
    TimeInterval,
    unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Model._case_insensitive_dict import normalize_name


def extract_grouping_identifiers(
    identifier_names: List[str], group_op: Optional[str], grouping_components: Any
) -> List[str]:
    if group_op == "group by":
        return grouping_components
    elif group_op == "group except":
        # Regular names are case-insensitive.
        excluded = {normalize_name(comp) for comp in grouping_components}
        return [comp for comp in identifier_names if normalize_name(comp) not in excluded]
    elif group_op == "group all":
        return identifier_names if grouping_components else []
    else:
        return identifier_names


# noinspection PyMethodOverriding
class Aggregation(Operator.Unary):
    @classmethod
    def validate(  # type: ignore[override]
        cls,
        operand: Dataset,
        group_op: Optional[str],
        grouping_columns: Any,
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
            keep_norm = {normalize_name(name) for name in identifiers_to_keep}
            for comp_name, comp in operand.components.items():
                if comp.role == Role.IDENTIFIER and normalize_name(comp_name) not in keep_norm:
                    del result_components[comp_name]
        else:
            for comp_name, comp in operand.components.items():
                if comp.role == Role.IDENTIFIER:
                    del result_components[comp_name]
        # Remove Attributes
        for comp_name, comp in operand.components.items():
            if comp.role == Role.ATTRIBUTE:
                del result_components[comp_name]
        # TimeInterval is not supported as a measure in aggregate operations
        if any(
            comp.role == Role.MEASURE and comp.data_type is TimeInterval
            for comp in result_components.values()
        ):
            raise SemanticError(
                "1-1-19-12",
                op=cls.op,
                context="aggregate",
            )

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
                name="int_var",
                role=Role.MEASURE,
                data_type=Integer,
                nullable=True,
            )
            result_components["int_var"] = new_comp

        # VDS is handled in visit_Aggregation
        return Dataset(name="result", components=result_components, data=None)


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

from copy import copy
from typing import List, Optional

import vtlengine.Operators as Operator
from vtlengine.AST import OrderBy, Windowing
from vtlengine.AST.Grammar.tokens import (
    AVG,
    COUNT,
    FIRST_VALUE,
    LAG,
    LAST_VALUE,
    LEAD,
    MAX,
    MEDIAN,
    MIN,
    RANK,
    RATIO_TO_REPORT,
    STDDEV_POP,
    STDDEV_SAMP,
    SUM,
    VAR_POP,
    VAR_SAMP,
)
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    Duration,
    Integer,
    Number,
    String,
    TimeInterval,
    TimePeriod,
    unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.ViralPropagation import require_rules

return_integer_operators = [MAX, MIN, SUM]


# noinspection PyMethodOverriding
class Analytic(Operator.Unary):
    """
    Analytic class

    Class that inherits from Unary.

    Class methods:
        Validate: Validates the Dataset.
    """

    return_integer = None
    sql_op: Optional[str] = None

    @classmethod
    def validate(  # type: ignore[override]  # noqa: C901
        cls,
        operand: Dataset,
        partitioning: List[str],
        ordering: Optional[List[OrderBy]],
        window: Optional[Windowing],
        params: Optional[List[int]],
        component_name: Optional[str] = None,
    ) -> Dataset:
        order_components = [] if ordering is None else [o.component for o in ordering]
        identifier_names = operand.get_identifiers_names()
        result_components = {
            comp_name: comp
            for comp_name, comp in operand.components.items()
            if comp.role != Role.ATTRIBUTE
        }
        result_components = {
            comp_name: comp
            for comp_name, comp in operand.components.items()
            if comp.role != Role.ATTRIBUTE
        }

        for comp_name in partitioning:
            if comp_name not in operand.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=cls.op,
                    comp_name=comp_name,
                    dataset_name=operand.name,
                )
            if comp_name not in identifier_names:
                raise SemanticError(
                    "1-1-3-2",
                    op=cls.op,
                    id_name=comp_name,
                    id_type=operand.components[comp_name].role,
                )
        for comp_name in order_components:
            if comp_name not in operand.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=cls.op,
                    comp_name=comp_name,
                    dataset_name=operand.name,
                )
            # TimeInterval is not supported in ORDER BY
            if operand.components[comp_name].data_type is TimeInterval:
                raise SemanticError(
                    "1-1-19-12",
                    op=cls.op,
                    context="analytic",
                )
            # RANGE window is not supported for String, Duration, TimePeriod, TimeInterval
            range_unsupported_types = (String, Duration, TimePeriod, TimeInterval)
            if (
                window is not None
                and window.type_ != "data"
                and operand.components[comp_name].data_type in range_unsupported_types
            ):
                raise SemanticError(
                    "1-1-19-13",
                    op=cls.op,
                    data_type=operand.components[comp_name].data_type.__name__,
                    comp_name=comp_name,
                )

        # TimeInterval is not supported as a measure in analytic operations
        if component_name is not None:
            if operand.components[component_name].data_type is TimeInterval:
                raise SemanticError(
                    "1-1-19-12",
                    op=cls.op,
                    context="analytic",
                )
        else:
            if cls.op != RANK and any(
                me.data_type is TimeInterval for me in operand.get_measures()
            ):
                raise SemanticError(
                    "1-1-19-12",
                    op=cls.op,
                    context="analytic",
                )

        if component_name is not None:
            if cls.type_to_check is not None:
                unary_implicit_promotion(
                    operand.components[component_name].data_type, cls.type_to_check
                )

            if cls.op in return_integer_operators:
                cls.return_integer = isinstance(cls.return_type, Integer)

            elif cls.return_type is not None:
                result_components[component_name] = Component(
                    name=component_name,
                    data_type=cls.return_type,
                    role=operand.components[component_name].role,
                    nullable=operand.components[component_name].nullable,
                )
            if cls.op == COUNT:
                measure_name = COMP_NAME_MAPPING[cls.return_type]
                result_components[measure_name] = Component(
                    name=measure_name,
                    data_type=cls.return_type,
                    role=Role.MEASURE,
                    nullable=operand.components[component_name].nullable,
                )
                if component_name in result_components:
                    del result_components[component_name]
        else:
            measures = operand.get_measures()
            if len(measures) == 0:
                raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)

            if cls.op in return_integer_operators:
                isNumber = False
                has_non_numeric = False
                for measure in measures:
                    if isinstance(measure.data_type, (Integer, Number)):
                        isNumber |= isinstance(measure.data_type, Number)
                    else:
                        has_non_numeric = True
                cls.return_integer = not isNumber and not has_non_numeric

            if cls.type_to_check is not None:
                for measure in measures:
                    unary_implicit_promotion(measure.data_type, cls.type_to_check)

            if cls.op in return_integer_operators:
                for measure in measures:
                    new_measure = copy(measure)
                    if isinstance(measure.data_type, (Integer, Number)):
                        new_measure.data_type = Integer if cls.return_integer else Number
                    result_components[measure.name] = new_measure
            elif cls.return_type is not None:
                for measure in measures:
                    new_measure = copy(measure)
                    new_measure.data_type = cls.return_type
                    result_components[measure.name] = new_measure

            if cls.op == COUNT and len(measures) <= 1:
                measure_name = COMP_NAME_MAPPING[cls.return_type]
                nullable = False if len(measures) == 0 else measures[0].nullable
                if len(measures) == 1:
                    del result_components[measures[0].name]
                result_components[measure_name] = Component(
                    name=measure_name,
                    data_type=cls.return_type,
                    role=Role.MEASURE,
                    nullable=nullable,
                )
        dataset_name = VirtualCounter._new_ds_name()
        # Analytic combines the data points within each partition, so the surviving viral
        # attributes are combined and require a propagation rule (issue #906).
        require_rules(operand.get_viral_attributes())
        return Dataset(name=dataset_name, components=result_components, data=None)


class Max(Analytic):
    """
    Max operator
    """

    op = MAX
    sql_op = "MAX"
    return_integer = False


class Min(Analytic):
    """
    Min operator
    """

    op = MIN
    sql_op = "MIN"
    return_integer = False


class Sum(Analytic):
    """
    Sum operator
    """

    op = SUM
    sql_op = "SUM"
    return_integer = False


class Count(Analytic):
    """
    Count operator
    """

    op = COUNT
    type_to_check = None
    return_type = Integer
    sql_op = "COUNT"


class Avg(Analytic):
    """
    Average operator
    """

    op = AVG
    type_to_check = Number
    return_type = Number
    sql_op = "AVG"


class Median(Analytic):
    """
    Median operator
    """

    op = MEDIAN
    type_to_check = Number
    return_type = Number
    sql_op = "MEDIAN"


class PopulationStandardDeviation(Analytic):
    """
    Population deviation operator
    """

    op = STDDEV_POP
    type_to_check = Number
    return_type = Number
    sql_op = "STDDEV_POP"


class SampleStandardDeviation(Analytic):
    """
    Sample standard deviation operator.
    """

    op = STDDEV_SAMP
    type_to_check = Number
    return_type = Number
    sql_op = "STDDEV_SAMP"


class PopulationVariance(Analytic):
    """
    Variance operator
    """

    op = VAR_POP
    type_to_check = Number
    return_type = Number
    sql_op = "VAR_POP"


class SampleVariance(Analytic):
    """
    Sample variance operator
    """

    op = VAR_SAMP
    type_to_check = Number
    return_type = Number
    sql_op = "VAR_SAMP"


class FirstValue(Analytic):
    """
    First value operator
    """

    op = FIRST_VALUE
    sql_op = "FIRST"


class LastValue(Analytic):
    """
    Last value operator
    """

    op = LAST_VALUE
    sql_op = "LAST"


class Lag(Analytic):
    """
    Lag operator
    """

    op = LAG
    sql_op = "LAG"


class Lead(Analytic):
    """
    Lead operator
    """

    op = LEAD
    sql_op = "LEAD"


class Rank(Analytic):
    """
    Rank operator
    """

    op = RANK
    sql_op = "RANK"
    return_type = Integer


class RatioToReport(Analytic):
    """
    Ratio operator
    """

    op = RATIO_TO_REPORT
    type_to_check = Number
    return_type = Number

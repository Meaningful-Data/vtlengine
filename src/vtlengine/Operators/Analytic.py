from copy import copy
from typing import Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

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
    Date,
    Duration,
    Integer,
    Number,
    String,
    TimeInterval,
    TimePeriod,
    unary_implicit_promotion,
)
from vtlengine.DataTypes._time_checking import DATE_COMPARISON_DTYPE, _from_date_value
from vtlengine.Exceptions import RunTimeError, SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._dataframe import merge_frames
from vtlengine.Utils._duckdb import query_connection
from vtlengine.ViralPropagation import (
    apply_viral_return_types,
    get_current_registry,
    require_rules,
)

return_integer_operators = [MAX, MIN, SUM]


def _range_bound(value: Union[int, str], bound: str, by_days: bool) -> str:
    """A bound of a range window. Where a Date orders the window it is a timestamp, which
    an offset moves by days, as on the DuckDB engine: a date took the offset as days
    itself (issue #1284)."""
    if by_days and isinstance(value, int) and value > 0:
        return f"INTERVAL '{value}' DAY"
    return bound


# noinspection PyMethodOverriding
class Analytic(Operator.Unary):
    """
    Analytic class

    Class that inherits from Unary.

    Class methods:
        Validate: Validates the Dataset.
        analyticfunc: Specify class method that returns a dataframe using the duckdb library.
        Evaluate: Ensures the type of data is the correct one to perform the Analytic operators.
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
                # count never returns null, whatever the nullability of its operand: a
                # window with nothing to count gives 0 (issue #1277)
                measure_name = COMP_NAME_MAPPING[cls.return_type]
                result_components[measure_name] = Component(
                    name=measure_name,
                    data_type=cls.return_type,
                    role=Role.MEASURE,
                    nullable=False,
                )
                if component_name in result_components:
                    del result_components[component_name]
            if cls.op in (LAG, LEAD) and component_name in result_components:
                result_components[component_name] = cls._as_nullable(
                    result_components[component_name]
                )
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

            if cls.op == COUNT:
                # count gives the number of Data Points whatever the Measures of the operand,
                # so the result holds the single Measure int_var (issue #1249), never null
                # (issue #1277)
                measure_name = COMP_NAME_MAPPING[cls.return_type]
                for measure in measures:
                    del result_components[measure.name]
                result_components[measure_name] = Component(
                    name=measure_name,
                    data_type=cls.return_type,
                    role=Role.MEASURE,
                    nullable=False,
                )

            if cls.op in (LAG, LEAD):
                for measure in measures:
                    if measure.name in result_components:
                        result_components[measure.name] = cls._as_nullable(
                            result_components[measure.name]
                        )
        dataset_name = VirtualCounter._new_ds_name()
        # Analytic combines the data points within each partition, so the surviving viral
        # attributes are combined and require a propagation rule (issue #906).
        viral_attributes = operand.get_viral_attributes()
        require_rules(viral_attributes)
        apply_viral_return_types(viral_attributes, result_components)
        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def analyticfunc(
        cls,
        df: pd.DataFrame,
        partitioning: List[str],
        identifier_names: List[str],
        measure_names: List[str],
        ordering: List[OrderBy],
        window: Optional[Windowing],
        params: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Annotation class

        It is used to analyze the attributes specified bellow
        ensuring that the type of data is the correct one to perform
        the operation.

        Attributes:
            identifier_names: List with the id names.
            measure_names: List with the measures names.
            ordering: List with the ordering modes.
            window: ...
            params: No params are related to this class.
        """
        # Windowing: with no window clause, the default frame of SQL is the default window of
        # VTL 2.2, range between unbounded preceding and current data point with an order
        # clause and the whole partition without one (issue #1249)
        window_str = ""
        if window is not None:
            mode = "ROWS" if window.type_ == "data" else "RANGE"
            start_mode = (
                window.start_mode.upper()
                if (isinstance(window.start, int) and window.start != 0)
                or (isinstance(window.start, str) and window.start == "unbounded")
                else ""
            )
            stop_mode = (
                window.stop_mode.upper()
                if (isinstance(window.stop, int) and window.stop != 0)
                or (isinstance(window.stop, str) and window.stop == "unbounded")
                else ""
            )
            start = (
                "UNBOUNDED"
                if window.start == "unbounded" or window.start == -1
                else str(window.start)
            )
            stop = (
                "CURRENT ROW" if window.stop == "current" or window.stop == 0 else str(window.stop)
            )
            by_days = mode == "RANGE" and any(
                str(df[x.component].dtype) == DATE_COMPARISON_DTYPE for x in ordering
            )
            start = _range_bound(window.start, start, by_days)
            stop = _range_bound(window.stop, stop, by_days)
            window_str = f"{mode} BETWEEN {start} {start_mode} AND {stop} {stop_mode}"

        # Partitioning
        partition = (
            "PARTITION BY " + ", ".join(f'"{p}"' for p in partitioning)
            if len(partitioning) > 0
            else ""
        )

        # Ordering
        order_str = ""
        if len(ordering) > 0:
            for x in ordering:
                order_str += f'"{x.component}" {x.order}, '
            if len(order_str) > 0:
                order_str = "ORDER BY " + order_str[:-2]

        # Generating the complete analytic string
        analytic_str = f"OVER ( {partition} {order_str} {window_str})"

        measure_queries = []
        for measure in measure_names:
            if cls.op == RANK:
                measure_query = f"{cls.sql_op}()"
            elif cls.op == RATIO_TO_REPORT:
                measure_query = f'CAST("{measure}" AS DOUBLE) / SUM(CAST("{measure}" AS DOUBLE))'
            elif cls.op in [LAG, LEAD]:
                args = "".join(f", {param}" for param in params or [])
                measure_query = f'{cls.sql_op}("{measure}"{args})'
            else:
                measure_query = f'{cls.sql_op}("{measure}")'
            if cls.op in return_integer_operators and cls.return_integer:
                measure_query = f'CAST({measure_query} {analytic_str} AS INTEGER) as "{measure}"'
            else:
                measure_query += f' {analytic_str} as "{measure}"'
            measure_queries.append(measure_query)
        if cls.op == COUNT:
            # count gives the number of Data Points of the window as the single Measure
            # int_var, in place of one per Measure. A Data Point whose Measures are all null
            # is not counted, as in the Aggregate invocation (issue #1049)
            any_non_null = " OR ".join(f'"{name}" IS NOT NULL' for name in measure_names)
            measure_queries = [
                f"COUNT(CASE WHEN {any_non_null} THEN 1 END) {analytic_str} "
                f'as "{COMP_NAME_MAPPING[cls.return_type]}"'
            ]

        # ratio_to_report divides by the partition sum: select it too, so that a
        # zero denominator (including 0/0, which yields NaN, not inf) is detected
        denominator_prefix = "__vtl_rtr_denominator_"
        if cls.op == RATIO_TO_REPORT:
            for measure in measure_names:
                measure_queries.append(
                    f'SUM(CAST("{measure}" AS DOUBLE)) {analytic_str} '
                    f'as "{denominator_prefix}{measure}"'
                )

        measures_sql = ", ".join(measure_queries)
        identifiers_sql = ", ".join(f'"{name}"' for name in identifier_names)
        query = f"SELECT {identifiers_sql} , {measures_sql} FROM df"

        try:
            with query_connection() as conn:
                conn.register("df", df)
                result = conn.execute(query).fetchdf()
        except RuntimeError as e:
            if "Conversion" in e.args[0]:
                raise RunTimeError("2-3-8", op=cls.op, msg=e.args[0].split(":")[-1])
            else:
                raise RunTimeError("2-1-1-1", op=cls.op, error=e)
        if cls.op == RATIO_TO_REPORT:
            for col_name in measure_names:
                if (result[f"{denominator_prefix}{col_name}"] == 0).any():
                    raise RunTimeError("2-1-3-1", op=cls.op)
                arr = pa.array(result[col_name])
                if pa.types.is_floating(arr.type) and pc.any(pc.is_inf(arr)).as_py():
                    raise RunTimeError("2-1-3-1", op=cls.op)
            result = result.drop(columns=[f"{denominator_prefix}{name}" for name in measure_names])
        return result

    @classmethod
    def evaluate(  # type: ignore[override]
        cls,
        operand: Dataset,
        partitioning: List[str],
        ordering: Optional[List[OrderBy]],
        window: Optional[Windowing],
        params: Optional[List[int]],
        component_name: Optional[str] = None,
    ) -> Dataset:
        result = cls.validate(operand, partitioning, ordering, window, params, component_name)
        df = operand.data.copy() if operand.data is not None else pd.DataFrame()
        df = cls.normalize_dates(df, operand.components)
        identifier_names = operand.get_identifiers_names()

        if component_name is not None:
            measure_names = [component_name]
        else:
            measure_names = operand.get_measures_names()

        # Validate TimePeriod measures have same period indicator for MAX/MIN
        if cls.op in [MAX, MIN]:
            measures = (
                [operand.components[component_name]]
                if component_name is not None
                else operand.get_measures()
            )
            for measure in measures:
                if measure.data_type is TimePeriod:
                    indicators = df[measure.name].dropna().str.extract(r"^\d{4}-?([ASQMWD])")[0]
                    if indicators.nunique() > 1:
                        raise RunTimeError("2-1-19-20", op=cls.op)

        result.data = cls.analyticfunc(
            df=df,
            partitioning=partitioning,
            identifier_names=identifier_names,
            measure_names=measure_names,
            ordering=ordering or [],
            window=window,
            params=params,
        )

        # Propagate viral attributes over each partition (analyticfunc only
        # returns identifiers + measures, so they are re-attached here).
        viral_attr_names = operand.get_viral_attributes_names()
        if viral_attr_names and operand.data is not None and result.data is not None:
            registry = get_current_registry()
            result.data = cls.normalize_dates(result.data, result.components)
            viral_src = df[identifier_names + viral_attr_names].copy()
            for va_name in viral_attr_names:
                if registry.get_rule_for_variable(va_name) is None:
                    # No rule -> per-row passthrough (keep the original value).
                    continue
                if partitioning:
                    viral_src[va_name] = viral_src.groupby(partitioning, sort=False)[
                        va_name
                    ].transform(lambda vals, n=va_name: registry.resolve_group(n, list(vals)))
                else:
                    viral_src[va_name] = registry.resolve_group(va_name, list(viral_src[va_name]))
            result.data = merge_frames(result.data, viral_src, on=identifier_names, how="left")

        if result.data is not None:
            for comp_name, comp in result.components.items():
                if comp_name not in result.data.columns:
                    continue
                column = result.data[comp_name]
                if comp.data_type is Date:
                    # A Date is written back value by value, the bare date at midnight
                    column = column.map(_from_date_value, na_action="ignore")
                result.data[comp_name] = column.astype(comp.data_type.dtype())  # type: ignore[call-overload]

        return result

    @classmethod
    def _as_nullable(cls, component: Component) -> Component:
        """Copy of the Component marked as nullable.

        lag and lead have no value to shift in at the edges of the partition, so their
        result holds nulls whatever the operand declared.
        """
        if component.nullable:
            return component
        nullable_component = copy(component)
        nullable_component.nullable = True
        return nullable_component

    @classmethod
    def normalize_dates(
        cls, data: Optional[pd.DataFrame], components: Dict[str, Component]
    ) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame(columns=[comp.name for comp in components.values()])
        elif any(comp.data_type is Date for comp in components.values()):
            data = data.copy()
            for comp_name, comp in components.items():
                if comp.data_type is Date:
                    data[comp_name] = data[comp_name].astype(DATE_COMPARISON_DTYPE)
        return data


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
    type_to_check = Number
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

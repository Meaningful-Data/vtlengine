from copy import copy
from typing import List, Optional

import duckdb

# if os.environ.get("SPARK"):
#     import pyspark.pandas as pd
# else:
#     import pandas as pd
import pandas as pd

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
    Integer,
    Number,
    unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role

return_integer_operators = [MAX, MIN, SUM]


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
        result_components = operand.components.copy()

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
                for measure in measures:
                    isNumber |= isinstance(measure.data_type, Number)
                cls.return_integer = not isNumber

            if cls.type_to_check is not None:
                for measure in measures:
                    unary_implicit_promotion(measure.data_type, cls.type_to_check)

            if cls.op in return_integer_operators:
                for measure in measures:
                    new_measure = copy(measure)
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

        return Dataset(name="result", components=result_components, data=None)

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
        # Windowing
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
            window_str = f"{mode} BETWEEN {start} {start_mode} AND {stop} {stop_mode}"

        # Partitioning
        partition = "PARTITION BY " + ", ".join(partitioning) if len(partitioning) > 0 else ""

        # Ordering
        order_str = ""
        if len(ordering) > 0:
            for x in ordering:
                order_str += f"{x.component} {x.order}, "
            if len(order_str) > 0:
                order_str = "ORDER BY " + order_str[:-2]

        # Generating the complete analytic string
        analytic_str = f"OVER ( {partition} {order_str} {window_str})"

        measure_queries = []
        for measure in measure_names:
            if cls.op == RANK:
                measure_query = f"{cls.sql_op}()"
            elif cls.op == RATIO_TO_REPORT:
                measure_query = f"CAST({measure} AS DOUBLE) / SUM(CAST({measure} AS DOUBLE))"
            elif cls.op in [LAG, LEAD]:
                measure_query = f"{cls.sql_op}({measure}, {','.join(map(str, params or []))})"
            else:
                measure_query = f"{cls.sql_op}({measure})"
            if cls.op == COUNT and len(measure_names) == 1:
                measure_query += f" {analytic_str} as {COMP_NAME_MAPPING[cls.return_type]}"
            elif cls.op in return_integer_operators and cls.return_integer:
                measure_query = f"CAST({measure_query} {analytic_str} AS INTEGER) as {measure}"
            else:
                measure_query += f" {analytic_str} as {measure}"
            measure_queries.append(measure_query)
        if cls.op == COUNT and len(measure_names) == 0:
            measure_queries.append(
                f"COUNT(*) {analytic_str} as {COMP_NAME_MAPPING[cls.return_type]}"
            )

        measures_sql = ", ".join(measure_queries)
        identifiers_sql = ", ".join(identifier_names)
        query = f"SELECT {identifiers_sql} , {measures_sql} FROM df"

        if cls.op == COUNT:
            df[measure_names] = df[measure_names].fillna(-1)
        # if os.getenv("SPARK", False):
        #     df = df.to_pandas()
        return duckdb.query(query).to_df().astype(object)

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
        identifier_names = operand.get_identifiers_names()

        if component_name is not None:
            measure_names = [component_name]
        else:
            measure_names = operand.get_measures_names()

        result.data = cls.analyticfunc(
            df=df,
            partitioning=partitioning,
            identifier_names=identifier_names,
            measure_names=measure_names,
            ordering=ordering or [],
            window=window,
            params=params,
        )

        # if cls.return_type == Integer:
        #     result.data[measure_names] = result.data[measure_names].astype('Int64')

        return result


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

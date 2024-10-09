import os
from copy import copy
from typing import List, Optional

import duckdb

from vtlengine.Exceptions import SemanticError

# if os.environ.get("SPARK"):
#     import pyspark.pandas as pd
# else:
#     import pandas as pd
import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST import OrderBy, Windowing
from vtlengine.AST.Grammar.tokens import AVG, COUNT, FIRST_VALUE, LAG, LAST_VALUE, LEAD, MAX, \
    MEDIAN, MIN, \
    RANK, RATIO_TO_REPORT, STDDEV_POP, \
    STDDEV_SAMP, \
    SUM, VAR_POP, \
    VAR_SAMP
from vtlengine.DataTypes import COMP_NAME_MAPPING, Integer, Number, \
    unary_implicit_promotion
from vtlengine.Model import Component, Dataset, Role


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
    sql_op: Optional[str] = None

    @classmethod
    def validate(cls, operand: Dataset,
                 partitioning: List[str],
                 ordering: Optional[List[OrderBy]],
                 window: Optional[Windowing],
                 params: Optional[List[int]]) -> Dataset:
        if ordering is None:
            order_components = []
        else:
            order_components = [o.component for o in ordering]
        identifier_names = operand.get_identifiers_names()
        result_components = operand.components.copy()

        for comp_name in partitioning:
            if comp_name not in operand.components:
                raise SemanticError("1-1-1-10", op=cls.op, comp_name=comp_name,
                                    dataset_name=operand.name)
            if comp_name not in identifier_names:
                raise SemanticError("1-1-3-2", op=cls.op, id_name=comp_name,
                                    id_type=operand.components[comp_name].role)
        for comp_name in order_components:
            if comp_name not in operand.components:
                raise SemanticError("1-1-1-10", op=cls.op, comp_name=comp_name,
                                    dataset_name=operand.name)
        measures = operand.get_measures()
        if measures is None:
            raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
        if cls.type_to_check is not None:
            for measure in measures:
                unary_implicit_promotion(measure.data_type, cls.type_to_check)
        if cls.return_type is not None:
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
                nullable=nullable
            )

        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def analyticfunc(cls, df: pd.DataFrame, partitioning: List[str],
                     identifier_names: List[str],
                     measure_names: List[str],
                     ordering: List[OrderBy],
                     window: Optional[Windowing],
                     params: Optional[List[int]] = None) -> pd.DataFrame:
        """Annotation class

        It is used to analyze the attributes specified bellow ensuring that the type of data is the correct one to perform
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
            start_mode = window.start_mode if window.start_mode != 'current' and window.start != 'CURRENT ROW' else ''
            stop_mode = window.stop_mode if window.stop_mode != 'current' and window.stop != 'CURRENT ROW' else ''
            if window.start == -1:
                window.start = 'UNBOUNDED'

            if stop_mode == '' and window.stop == 0:
                window.stop = 'CURRENT ROW'
            window_str = f"{mode} BETWEEN {window.start} {start_mode} AND {window.stop} {stop_mode}"

        # Partitioning
        if len(partitioning) > 0:
            partition = "PARTITION BY " + ', '.join(partitioning)
        else:
            partition = ""

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
                measure_query = f"CAST({measure} AS REAL) / SUM(CAST({measure} AS REAL))"
            elif cls.op in [LAG, LEAD]:
                measure_query = f"{cls.sql_op}({measure}, {','.join(map(str, params))})"
            else:
                measure_query = f"{cls.sql_op}({measure})"
            if cls.op == COUNT and len(measure_names) == 1:
                measure_query += f" {analytic_str} as {COMP_NAME_MAPPING[cls.return_type]}"
            else:
                measure_query += f" {analytic_str} as {measure}"
            measure_queries.append(measure_query)
        if cls.op == COUNT and len(measure_names) == 0:
            measure_queries.append(
                f"COUNT(*) {analytic_str} as {COMP_NAME_MAPPING[cls.return_type]}")

        measures_sql = ', '.join(measure_queries)
        identifiers_sql = ', '.join(identifier_names)
        query = f"SELECT {identifiers_sql} , {measures_sql} FROM df"

        if cls.op == COUNT:
            df[measure_names] = df[measure_names].fillna(-1)
        if os.getenv("SPARK", False):
            df = df.to_pandas()
        return duckdb.query(query).to_df()

    @classmethod
    def evaluate(cls, operand: Dataset,
                 partitioning: List[str],
                 ordering: Optional[List[OrderBy]],
                 window: Optional[Windowing],
                 params: Optional[List[int]]) -> Dataset:
        result = cls.validate(operand, partitioning, ordering, window, params)
        df = operand.data.copy() if operand.data is not None else pd.DataFrame()
        measure_names = operand.get_measures_names()
        identifier_names = operand.get_identifiers_names()

        result.data = cls.analyticfunc(df=df, partitioning=partitioning,
                                       identifier_names=identifier_names,
                                       measure_names=measure_names,
                                       ordering=ordering, window=window, params=params)
        return result


class Max(Analytic):
    """
    Max operator
    """
    op = MAX
    sql_op = "MAX"


class Min(Analytic):
    """
    Min operator
    """
    op = MIN
    sql_op = "MIN"


class Sum(Analytic):
    """
    Sum operator
    """
    op = SUM
    type_to_check = Number
    return_type = Number
    sql_op = "SUM"


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

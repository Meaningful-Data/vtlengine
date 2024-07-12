import os
from typing import List, Optional

import duckdb
if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

import Operators as Operator
from AST import OrderBy, Windowing
from AST.Grammar.tokens import AVG, COUNT, FIRST_VALUE, LAG, LAST_VALUE, LEAD, MAX, MEDIAN, MIN, \
    RANK, RATIO_TO_REPORT, STDDEV_POP, \
    STDDEV_SAMP, \
    SUM, VAR_POP, \
    VAR_SAMP
from DataTypes import Integer, Number
from Model import Dataset, Role


# noinspection PyMethodOverriding
class Analytic(Operator.Unary):
    sql_op = None

    @classmethod
    def validate(cls, operand: Dataset,
                 partitioning: List[str],
                 ordering: Optional[List[OrderBy]],
                 window: Windowing) -> Dataset:
        if ordering is None:
            order_components = []
        else:
            order_components = [o.component for o in ordering]
        identifier_names = operand.get_identifiers_names()
        result_components = operand.components.copy()

        for comp_name in partitioning:
            if comp_name not in identifier_names:
                raise Exception(f"Component {comp_name} is not in the dataset {operand.name}")
        for comp_name in order_components:
            if comp_name not in operand.components:
                raise Exception(f"Component {comp_name} is not in the dataset {operand.name}")
        measures = operand.get_measures()

        # TODO: Type promotion
        if cls.type_to_check is not None:
            result_components[measures[0].name].data_type = cls.type_to_check

        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def analyticfunc(cls, df: pd.DataFrame, partitioning: List[str],
                     identifier_names: List[str],
                     measure_names: List[str],
                     ordering: List[OrderBy],
                     window: Optional[Windowing],
                     params: Optional[List[int]] = None):
        """

        """
        # Windowing
        window_str = ""
        if window is not None:
            mode = "ROWS" if window.type_ == "data" else "RANGE"
            start_mode = window.start_mode if window.start_mode != 'current' and window.start != 'CURRENT ROW' else ''
            stop_mode = window.stop_mode if window.stop_mode != 'current' and window.stop != 'CURRENT ROW' else ''
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
            measure_query += f" {analytic_str} as {measure}"
            measure_queries.append(measure_query)

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
        result = cls.validate(operand, partitioning, ordering, window)
        df = operand.data.copy()
        measure_names = operand.get_measures_names()
        identifier_names = operand.get_identifiers_names()

        result.data = cls.analyticfunc(df=df, partitioning=partitioning,
                                       identifier_names=identifier_names,
                                       measure_names=measure_names,
                                       ordering=ordering, window=window, params=params)
        return result


class Max(Analytic):
    op = MAX
    sql_op = "MAX"


class Min(Analytic):
    op = MIN
    sql_op = "MIN"


class Sum(Analytic):
    op = SUM
    type_to_check = Number
    sql_op = "SUM"


class Count(Analytic):
    op = COUNT
    type_to_check = None
    return_type = Integer
    sql_op = "COUNT"


class Avg(Analytic):
    op = AVG
    type_to_check = Number
    return_type = Number
    sql_op = "AVG"


class Median(Analytic):
    op = MEDIAN
    type_to_check = Number
    return_type = Number
    sql_op = "MEDIAN"


class PopulationStandardDeviation(Analytic):
    op = STDDEV_POP
    type_to_check = Number
    return_type = Number
    sql_op = "STDDEV_POP"


class SampleStandardDeviation(Analytic):
    op = STDDEV_SAMP
    type_to_check = Number
    return_type = Number
    sql_op = "STDDEV_SAMP"


class PopulationVariance(Analytic):
    op = VAR_POP
    type_to_check = Number
    return_type = Number
    sql_op = "VAR_POP"


class SampleVariance(Analytic):
    op = VAR_SAMP
    type_to_check = Number
    return_type = Number
    sql_op = "VAR_SAMP"


class FirstValue(Analytic):
    op = FIRST_VALUE
    sql_op = "FIRST"


class LastValue(Analytic):
    op = LAST_VALUE
    sql_op = "LAST"


class Lag(Analytic):
    op = LAG
    sql_op = "LAG"


class Lead(Analytic):
    op = LEAD
    sql_op = "LEAD"


class Rank(Analytic):
    op = RANK
    sql_op = "RANK"


class RatioToReport(Analytic):
    op = RATIO_TO_REPORT

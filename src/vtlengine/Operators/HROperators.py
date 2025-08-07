import operator
from copy import copy
from typing import Any, Dict

import pandas as pd
from duckdb.duckdb import DuckDBPyRelation
from pandas import DataFrame

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY
from vtlengine.DataTypes import Boolean, Number
from vtlengine.Model import Component, DataComponent, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.duckdb.duckdb_utils import duckdb_concat, empty_relation, duckdb_drop
from vtlengine.duckdb.to_sql_token import TO_SQL_TOKEN, MIDDLE, LEFT


def get_measure_from_dataset(dataset: Dataset, code_item: str) -> DataComponent:
    measure_name = dataset.get_measures_names()[0]
    data = None if dataset.data is None else dataset.data[measure_name]
    return DataComponent(
        name=code_item,
        data=data,
        data_type=dataset.components[measure_name].data_type,
        role=dataset.components[measure_name].role,
        nullable=dataset.components[measure_name].nullable,
    )


class HRComparison(Operators.Binary):

    @staticmethod
    def hr_func(left_rel: DuckDBPyRelation, right_rel: DuckDBPyRelation, hr_mode: str) -> DuckDBPyRelation:
        l_name = left_rel.columns[0]
        r_name = right_rel.columns[0]
        RM_val = "'REMOVE_VALUE'"
        expr = f'"{l_name}", "{r_name}"'
        
        if hr_mode == "partial_null":
            expr += f""",
                    CASE
                        WHEN "{r_name}" = {RM_val} AND "{r_name}" IS NOT NULL AND "{l_name}" IS NOT NULL THEN NULL
                        WHEN "{r_name}" = {RM_val} AND "{r_name}" IS NOT NULL THEN {RM_val}
                        ELSE 'true'
                    END AS hr_mask
                """

        elif hr_mode == "partial_zero":
            expr += f""",
                    CASE
                        WHEN "{r_name}" = {RM_val} AND "{r_name}" IS NOT NULL AND "{l_name}" != 0 THEN NULL
                        WHEN "{r_name}" = {RM_val} AND "{r_name}" IS NOT NULL THEN {RM_val}
                        ELSE 'true'
                    END AS hr_mask
                """

        elif hr_mode == "non_null":
            expr += f""",
                    CASE
                        WHEN "{l_name}" IS NULL OR "{r_name}" IS NULL THEN 'REMOVE_VALUE'
                        ELSE 'true'
                    END AS hr_mask
                """

        elif hr_mode == "non_zero":
            expr += f""",
                    CASE
                        WHEN "{l_name}" = 0 AND "{r_name}" = 0 THEN {RM_val}
                        ELSE 'true'
                    END AS hr_mask
                """
        else:
            expr += f", 'true' AS hr_mask"

        # Combine the relations and apply the masks
        combined_relation = duckdb_concat(left_rel, right_rel)
        return combined_relation.project(expr)

    @classmethod
    def apply_hr_func(cls, left_rel: DuckDBPyRelation, right_rel: DuckDBPyRelation, hr_mode: str, func: str, col_name: str) -> DuckDBPyRelation:
        # In order not to apply the function to the whole series, we align the series
        # and apply the function only to the valid values based on a validation mask.
        # The function is applied to the aligned series and the result is combined with the
        # original series.
        result = cls.hr_func(left_rel, right_rel, hr_mode)

        position = MIDDLE if func != "imbalance_func" else LEFT
        sql_token = TO_SQL_TOKEN.get(func, func)
        if isinstance(sql_token, tuple):
            sql_token, position = sql_token
        if position == MIDDLE:
            sql_func = f'("{left_rel.columns[0]}" {sql_token} "{right_rel.columns[0]}")'
        else:
            sql_func = f'{sql_token}("{left_rel.columns[0]}", "{right_rel.columns[0]}")'

        result = result.project(f"""
                    CASE 
                        WHEN hr_mask = 'true' 
                        THEN CAST({sql_func} AS VARCHAR)
                        ELSE hr_mask
                    END AS "{col_name}"
                """)
        return result

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: DataComponent, hr_mode: str) -> Dataset:
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in left_operand.components.items()
            if comp.role == Role.IDENTIFIER
        }
        result_components["bool_var"] = Component(
            name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
        )
        result_components["imbalance"] = Component(
            name="imbalance", data_type=Number, role=Role.MEASURE, nullable=True
        )
        return Dataset(
            name=f"{left_operand.name}{cls.op}{right_operand.name}",
            components=result_components,
            data=None,
        )

    @classmethod
    def evaluate(cls, left: Dataset, right: DataComponent, hr_mode: str) -> Dataset:  # type: ignore[override]
        result = cls.validate(left, right, hr_mode)
        result.data = left.data if left.data is not None else empty_relation()
        measure_name = left.get_measures_names()[0]

        if left.data is not None and right.data is not None:
            bool_var = cls.apply_hr_func(
                left.data[measure_name], right.data, hr_mode, cls.op, "bool_var"
            )
            imbalance = cls.apply_hr_func(
                left.data[measure_name], right.data, hr_mode, "imbalance_func", "imbalance"
            )
            result.data = duckdb_concat(result.data, duckdb_concat(bool_var, imbalance))

        # Removing datapoints that should not be returned
        # (we do it below imbalance calculation
        # to avoid errors on different shape)
        result.data = duckdb_drop(result.data.filter("bool_var != 'REMOVE_VALUE'"), measure_name)
        return result


class HREqual(HRComparison):
    op = "="
    py_op = operator.eq


class HRGreater(HRComparison):
    op = ">"
    py_op = operator.gt


class HRGreaterEqual(HRComparison):
    op = ">="
    py_op = operator.ge


class HRLess(HRComparison):
    op = "<"
    py_op = operator.lt


class HRLessEqual(HRComparison):
    op = "<="
    py_op = operator.le


class HRBinNumeric(Operators.Binary):
    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if not pd.isnull(x) and x == "REMOVE_VALUE":
            return "REMOVE_VALUE"
        return super().op_func(x, y)

    @classmethod
    def evaluate(cls, left: DataComponent, right: DataComponent) -> DataComponent:
        # TODO: remove type ignore on HROperators issue
        l_name = left.data.columns[0]
        r_name = right.data.columns[0]
        name = f"{l_name}{cls.op}{r_name}"
        result_data = duckdb_concat(left.data, right.data)
        result_data = result_data.project(Operators.apply_bin_op(cls, name, l_name, r_name))

        return DataComponent(
            name=f"{left.name}{cls.op}{right.name}",
            data=result_data,
            data_type=left.data_type,
            role=left.role,
            nullable=left.nullable,
        )


class HRBinPlus(HRBinNumeric):
    op = "+"
    py_op = operator.add


class HRBinMinus(HRBinNumeric):
    op = "-"
    py_op = operator.sub


class HRUnNumeric(Operators.Unary):
    @classmethod
    def evaluate(cls, operand: DataComponent) -> DataComponent:  # type: ignore[override]
        result_data = cls.apply_operation_component(operand.data)
        return DataComponent(
            name=f"{cls.op}({operand.name})",
            data=result_data,
            data_type=operand.data_type,
            role=operand.role,
            nullable=operand.nullable,
        )


class HRUnPlus(HRUnNumeric):
    op = "+"
    py_op = operator.pos


class HRUnMinus(HRUnNumeric):
    op = "-"
    py_op = operator.neg


class HAAssignment(Operators.Binary):
    @classmethod
    def validate(cls, left: Dataset, right: DataComponent, hr_mode: str) -> Dataset:
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        return Dataset(name=f"{left.name}", components=result_components, data=None)

    @classmethod
    def evaluate(  # type: ignore[override]
        cls, left: Dataset, right: DataComponent, hr_mode: str
    ) -> Dataset:
        result = cls.validate(left, right, hr_mode)
        measure_name = left.get_measures_names()[0]
        result.data = left.data if left.data is not None else empty_relation()
        if right.data is not None:
            result.data = duckdb_concat(
                result.data,
                right.data.project(
                    f'handle_mode("{right.data.columns[0]}", {repr(hr_mode)}) AS "{measure_name}"'
                )
            )
        result.data = result.data.filter(f'"{measure_name}" != \'REMOVE_VALUE\'')
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
    def validate(
        cls, dataset: Dataset, computed_dict: Dict[str, DataFrame], output: str
    ) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        result_components = {
            comp_name: copy(comp) for comp_name, comp in dataset.components.items()
        }
        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def evaluate(
        cls, dataset: Dataset, computed_dict: Dict[str, DataFrame], output: str
    ) -> Dataset:
        result = cls.validate(dataset, computed_dict, output)
        if len(computed_dict) == 0:
            computed_data = empty_relation(dataset.get_components_names())
        else:
            computed_data = cls.generate_computed_data(computed_dict)
        if output == "computed":
            result.data = computed_data
            return result

        # union(setdiff(op, R), R) where R is the computed data.
        # It is the same as union(op, R) and drop duplicates, selecting the last one available
        result.data = duckdb_concat(dataset.data, computed_data).distinct()
        return result


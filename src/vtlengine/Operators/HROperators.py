import operator
from copy import copy
from typing import Any, Dict, List

import pandas as pd
from duckdb.duckdb import DuckDBPyRelation
from pandas import DataFrame

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY
from vtlengine.DataTypes import Boolean, Number
from vtlengine.duckdb.custom_functions.HR import NINF
from vtlengine.duckdb.duckdb_utils import (
    duckdb_concat,
    duckdb_drop,
    duckdb_rename,
    empty_relation,
)
from vtlengine.duckdb.to_sql_token import LEFT, MIDDLE, TO_SQL_TOKEN
from vtlengine.Model import Component, DataComponent, Dataset, RelationProxy, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter

TRUE_VAL = 1
RM_VAL = NINF


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
    def hr_func(
        left_rel: DuckDBPyRelation, right_rel: DuckDBPyRelation, hr_mode: str
    ) -> DuckDBPyRelation:
        l_name = left_rel.columns[0]
        r_name = right_rel.columns[0]
        expr = f'"{l_name}", "{r_name}"'

        if hr_mode == "partial_null":
            expr += f""",
                    CASE
                        WHEN "{r_name}" = {RM_VAL} AND "{l_name}" IS NOT NULL
                        THEN NULL
                        WHEN "{r_name}" = {RM_VAL}
                        THEN {RM_VAL}
                        ELSE {TRUE_VAL}
                    END AS hr_mask
                """

        elif hr_mode == "partial_zero":
            expr += f""",
                    CASE
                        WHEN "{r_name}" = {RM_VAL} AND ("{l_name}" != 0 OR "{l_name}" IS NULL)
                        THEN NULL
                        WHEN "{r_name}" = {RM_VAL}
                        THEN {RM_VAL}
                        ELSE {TRUE_VAL}
                    END AS hr_mask
                """

        elif hr_mode == "non_null":
            expr += f""",
                    CASE
                        WHEN "{l_name}" IS NULL OR "{r_name}" IS NULL THEN {RM_VAL}
                        ELSE {TRUE_VAL}
                    END AS hr_mask
                """

        elif hr_mode == "non_zero":
            expr += f""",
                    CASE
                        WHEN "{l_name}" = 0 AND "{r_name}" = 0 THEN {RM_VAL}
                        ELSE {TRUE_VAL}
                    END AS hr_mask
                """
        else:
            expr += f", {TRUE_VAL} AS hr_mask"

        # Combine the relations and apply the masks
        combined_relation = duckdb_concat(left_rel, right_rel)
        return combined_relation.project(expr)

    @classmethod
    def apply_hr_func(
        cls,
        left_rel: DuckDBPyRelation,
        right_rel: DuckDBPyRelation,
        hr_mode: str,
        func: str,
        col_name: str,
    ) -> DuckDBPyRelation:
        l_name = left_rel.columns[0]
        r_name = right_rel.columns[0]
        if l_name == r_name:
            l_name = f"__l_{l_name}__"
            r_name = f"__r_{r_name}__"
            left_rel = duckdb_rename(left_rel, {left_rel.columns[0]: l_name})
            right_rel = duckdb_rename(right_rel, {right_rel.columns[0]: r_name})
        result = cls.hr_func(left_rel, right_rel, hr_mode)

        position = MIDDLE if func != "imbalance_func" else LEFT
        sql_token = TO_SQL_TOKEN.get(func, func)
        if isinstance(sql_token, tuple):
            sql_token, position = sql_token

        if position == MIDDLE:
            sql_func = f'("{l_name}" {sql_token} "{r_name}")'
        else:
            sql_func = f'{sql_token}("{l_name}", "{r_name}")'

        result = result.project(f"""
                    CASE
                        WHEN hr_mask = {TRUE_VAL} THEN CAST({sql_func} AS DOUBLE)
                        ELSE CAST(hr_mask AS DOUBLE)
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
            result.data["bool_var"] = cls.apply_hr_func(
                left.data[measure_name], right.data, hr_mode, cls.op, "bool_var"
            )
            result.data["imbalance"] = cls.apply_hr_func(
                left.data[measure_name], right.data, hr_mode, "imbalance_func", "imbalance"
            )

        # Removing datapoints that should not be returned
        # (we do it below imbalance calculation
        # to avoid errors on different shape)
        result.data = duckdb_drop(
            result.data.filter(f"bool_var IS DISTINCT FROM {NINF}"), measure_name
        )
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
    def op_func(cls, me_name: str, left: Any, right: Any) -> Any:
        left = left if left is not None else "NULL"
        right = right if right is not None else "NULL"
        return f"""
                    CASE
                        WHEN {left} = {NINF} THEN {NINF}
                        ELSE {left} {cls.op} {right}
                    END AS {me_name}
                """

    @classmethod
    def evaluate(cls, left: DataComponent, right: DataComponent) -> DataComponent:
        name = f"{left.name}{cls.op}{right.name}"
        left_rel = duckdb_rename(left.data.order_by_index(), {left.data.columns[0]: "left"})
        right_rel = duckdb_rename(right.data.order_by_index(), {right.data.columns[0]: "right"})
        expr = cls.op_func(f'"{name}"', '"left"', '"right"')
        result_data = duckdb_concat(left_rel, right_rel, how="inner")
        result_data = result_data.project(expr)

        return DataComponent(
            name=name,
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
            rcol = right.data.columns[0]
            result.data[measure_name] = right.data.project(
                f'handle_mode("{rcol}", \'{hr_mode}\') AS "{measure_name}"'
            )
        result.data = result.data[result.data[measure_name] != NINF]
        return result

    @classmethod
    def handle_mode(cls, x: Any, hr_mode: str) -> Any:
        if not pd.isnull(x) and x == NINF:
            return NINF
        if hr_mode == "non_null" and pd.isnull(x) or hr_mode == "non_zero" and x == 0:
            return NINF
        return x


class Hierarchy(Operators.Operator):
    op = HIERARCHY

    @staticmethod
    def generate_computed_data(
        computed_dict: Dict[str, RelationProxy], comp_names: List[str]
    ) -> RelationProxy:
        relations = list(computed_dict.values())
        if not relations:
            return empty_relation()
        combined_relation = relations[0]
        for rel in relations[1:]:
            combined_relation = duckdb_concat(combined_relation, rel, how="outer", on=comp_names)
        return RelationProxy(combined_relation).reset_index()

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
        comp_names = dataset.get_components_names()
        if len(computed_dict) == 0:
            computed_data = empty_relation(dataset.get_components_names())
        else:
            computed_data = RelationProxy(cls.generate_computed_data(computed_dict, comp_names))

        if output == "computed":
            result.data = computed_data
            return result

        # union(setdiff(op, R), R) where R is the computed data.
        # It is the same as union(op, R) and drop duplicates, selecting the last one available
        result.data = duckdb_concat(dataset.data, computed_data, how="outer", on=comp_names)
        result.data = result.data.distinct()
        result.data = result.data.reset_index()
        return result

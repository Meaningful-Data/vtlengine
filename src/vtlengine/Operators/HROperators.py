import operator
from copy import copy
from typing import Any, Dict

import pandas as pd
from pandas import DataFrame

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY
from vtlengine.DataTypes import Boolean, Number
from vtlengine.Model import Component, DataComponent, Dataset, Role


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
    @classmethod
    def imbalance_func(cls, x: Any, y: Any) -> Any:
        return None if pd.isnull(x) or pd.isnull(y) else x - y

    @staticmethod
    def hr_func(left_series: Any, right_series: Any, hr_mode: str) -> Any:
        result = pd.Series(True, index=left_series.index)

        if hr_mode in ("partial_null", "partial_zero"):
            mask_remove = (right_series == "REMOVE_VALUE") & (right_series.notnull())
            if hr_mode == "partial_null":
                mask_null = mask_remove & left_series.notnull()
            else:
                mask_null = mask_remove & (left_series != 0)
            result[mask_remove] = "REMOVE_VALUE"
            result[mask_null] = None
        elif hr_mode == "non_null":
            mask_remove = left_series.isnull() | right_series.isnull()
            result[mask_remove] = "REMOVE_VALUE"
        elif hr_mode == "non_zero":
            mask_remove = (left_series == 0) & (right_series == 0)
            result[mask_remove] = "REMOVE_VALUE"

        return result

    @classmethod
    def apply_hr_func(cls, left_series: Any, right_series: Any, hr_mode: str, func: Any) -> Any:
        # In order not to apply the function to the whole series, we align the series
        # and apply the function only to the valid values based on a validation mask.
        # The function is applied to the aligned series and the result is combined with the
        # original series.
        left_series, right_series = left_series.align(right_series)
        remove_result = cls.hr_func(left_series, right_series, hr_mode)
        mask_valid = remove_result == True
        result = pd.Series(remove_result, index=left_series.index)
        result.loc[mask_valid] = left_series[mask_valid].combine(right_series[mask_valid], func)
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
        result.data = left.data.copy() if left.data is not None else pd.DataFrame()
        measure_name = left.get_measures_names()[0]

        if left.data is not None and right.data is not None:
            result.data["bool_var"] = cls.apply_hr_func(
                left.data[measure_name], right.data, hr_mode, cls.op_func
            )
            result.data["imbalance"] = cls.apply_hr_func(
                left.data[measure_name], right.data, hr_mode, cls.imbalance_func
            )

        # Removing datapoints that should not be returned
        # (we do it below imbalance calculation
        # to avoid errors on different shape)
        result.data = result.data[result.data["bool_var"] != "REMOVE_VALUE"]
        result.data.drop(measure_name, axis=1, inplace=True)
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
        result_data = cls.apply_operation_two_series(left.data, right.data)
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
        result.data = left.data.copy() if left.data is not None else pd.DataFrame()
        if right.data is not None:
            result.data[measure_name] = right.data.map(lambda x: cls.handle_mode(x, hr_mode))
        result.data = result.data[result.data[measure_name] != "REMOVE_VALUE"]
        return result

    @classmethod
    def handle_mode(cls, x: Any, hr_mode: str) -> Any:
        if not pd.isnull(x) and x == "REMOVE_VALUE":
            return "REMOVE_VALUE"
        if hr_mode == "non_null" and pd.isnull(x) or hr_mode == "non_zero" and x == 0:
            return "REMOVE_VALUE"
        return x


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
        result_components = {
            comp_name: copy(comp) for comp_name, comp in dataset.components.items()
        }
        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(
        cls, dataset: Dataset, computed_dict: Dict[str, DataFrame], output: str
    ) -> Dataset:
        result = cls.validate(dataset, computed_dict, output)
        if len(computed_dict) == 0:
            computed_data = pd.DataFrame(columns=dataset.get_components_names())
        else:
            computed_data = cls.generate_computed_data(computed_dict)
        if output == "computed":
            result.data = computed_data
            return result

        # union(setdiff(op, R), R) where R is the computed data.
        # It is the same as union(op, R) and drop duplicates, selecting the last one available
        result.data = pd.concat([dataset.data, computed_data], axis=0, ignore_index=True)
        result.data.drop_duplicates(
            subset=dataset.get_identifiers_names(), keep="last", inplace=True
        )
        result.data.reset_index(drop=True, inplace=True)
        return result

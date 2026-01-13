import operator
from copy import copy
from typing import Any, Dict, Tuple, Optional

import pandas as pd
from pandas import DataFrame

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY, PARTIAL_NULL, PARTIAL_ZERO, NON_ZERO, NON_NULL
from vtlengine.DataTypes import Boolean, Number
from vtlengine.Model import Component, DataComponent, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


REMOVE = "REMOVE_VALUE"


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


class HRBinOp(Operators.Binary):
    @classmethod
    def apply_operation_two_series(cls, left: pd.Series, right: pd.Series, op: Optional[Any] = None) -> pd.Series:
        op = op if op is not None else cls.op_func
        result = list(map(op, left.values, right.values))
        return pd.Series(result, index=left.index, dtype=object)
    
    @classmethod
    def align_series(cls, left: pd.Series, right: pd.Series, mode: str) -> Tuple[pd.Series, pd.Series]:
        value = 0 if mode.endswith("zero") else None

        left_idx = left.index
        right_idx = right.index

        left_aligned, right_aligned = left.align(right, join="outer")

        left_new = ~left_aligned.index.isin(left_idx)
        right_new = ~right_aligned.index.isin(right_idx)

        left_aligned[left_new] = REMOVE
        right_aligned[right_new] = REMOVE

        left_remove = left_aligned == REMOVE
        right_remove = right_aligned == REMOVE

        left_aligned[left_remove] = value
        right_aligned[right_remove] = value

        if mode in (PARTIAL_ZERO, PARTIAL_NULL):
            mask_remove = left_remove & right_remove
        elif mode == NON_NULL:
            mask_remove = left_aligned.isna() | right_aligned.isna()
        elif mode == NON_ZERO:
            mask_remove = (left_aligned == 0) & (right_aligned == 0)
        else:
            return left_aligned, right_aligned
        return left_aligned[~mask_remove], right_aligned[~mask_remove]

    @classmethod
    def hr_op(cls, left_series: pd.Series, right_series: pd.Series, hr_mode: str) -> Any:
        left, right = cls.align_series(left_series, right_series, hr_mode)
        return cls.apply_operation_two_series(left, right)


class HRComparison(HRBinOp):
    @classmethod
    def imbalance_op(cls, x: Any, y: Any) -> Any:
        return None if pd.isnull(x) or pd.isnull(y) else x - y

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
    def evaluate(cls, left: Dataset, right: DataComponent, mode: str) -> Dataset:  # type: ignore[override]
        result = cls.validate(left, right, mode)
        result.data = left.data.copy() if left.data is not None else pd.DataFrame()
        measure_name = left.get_measures_names()[0]

        if left.data is not None and right.data is not None:
            left_data, right_data = cls.align_series(left.data[measure_name], right.data, mode)
            result.data = result.data.iloc[left_data.index]
            result.data["bool_var"] = cls.apply_operation_two_series(left_data, right_data)
            result.data["imbalance"] = cls.apply_operation_two_series(left_data, right_data, cls.imbalance_op)

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


class HRBinNumeric(HRBinOp):
    @classmethod
    def evaluate(cls, left: DataComponent, right: DataComponent, hr_mode: str) -> DataComponent:
        result_data = cls.hr_op(left.data, right.data, hr_mode)
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
            result.data = result.data.iloc[right.data.index[0:len(result.data)]]

        result.data = result.data[result.data[measure_name] != REMOVE]
        return result

    @classmethod
    def handle_mode(cls, x: Any, hr_mode: str) -> Any:
        if not pd.isnull(x) and x == REMOVE:
            return REMOVE
        if hr_mode == "non_null" and pd.isnull(x) or hr_mode == "non_zero" and x == 0:
            return REMOVE
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

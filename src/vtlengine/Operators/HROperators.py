import operator
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY, NON_NULL, NON_ZERO
from vtlengine.DataTypes import Boolean, Number
from vtlengine.Model import Component, DataComponent, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._number_config import (
    numbers_are_equal,
    numbers_are_greater_equal,
    numbers_are_less_equal,
)
from vtlengine.ViralPropagation import get_current_registry

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
    def apply_operation_two_series(cls, left: Any, right: Any, op: Any = None) -> Any:
        op = op if op is not None else cls.op_func
        result = list(map(op, left.values, right.values))
        return pd.Series(result, index=left.index, dtype=object)

    @classmethod
    def align_series(cls, left: Any, right: Any, mode: str) -> Tuple[Any, Any]:
        fill_value = 0 if mode.endswith("zero") else None
        # Convert to object dtype for sentinel-based alignment
        left_obj = left.astype(object)
        right_obj = right.astype(object) if isinstance(right, pd.Series) else right
        left_aligned, right_aligned = left_obj.align(right_obj, join="outer")

        left_aligned[left_aligned.index.difference(left.index, sort=False)] = REMOVE
        right_aligned[right_aligned.index.difference(right.index, sort=False)] = REMOVE
        mask_remove = (left_aligned == REMOVE) & (right_aligned == REMOVE)

        left_aligned = left_aligned.where(left_aligned != REMOVE, fill_value)
        right_aligned = right_aligned.where(right_aligned != REMOVE, fill_value)

        if mode == NON_NULL:
            mask_remove |= left_aligned.isna() | right_aligned.isna()
        elif mode == NON_ZERO:
            mask_remove |= (left_aligned == 0) & (right_aligned == 0)

        return left_aligned[~mask_remove], right_aligned[~mask_remove]

    @classmethod
    def hr_op(cls, left_series: Any, right_series: Any, hr_mode: str) -> Any:
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
    def evaluate(cls, left: Dataset, right: DataComponent, hr_mode: str) -> Dataset:  # type: ignore[override]
        result = cls.validate(left, right, hr_mode)
        result.data = left.data.copy() if left.data is not None else pd.DataFrame()
        measure_name = left.get_measures_names()[0]

        if left.data is not None and right.data is not None:
            left_data, right_data = cls.align_series(left.data[measure_name], right.data, hr_mode)
            result.data = result.data.loc[left_data.index]
            result.data[measure_name] = left_data
            result.data["bool_var"] = cls.apply_operation_two_series(left_data, right_data)
            result.data["imbalance"] = cls.apply_operation_two_series(
                left_data, right_data, cls.imbalance_op
            )

        return result


class HREqual(HRComparison):
    op = "="
    py_op = operator.eq

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return numbers_are_equal(x, y)
        return cls.py_op(x, y)


class HRGreater(HRComparison):
    op = ">"
    py_op = operator.gt


class HRGreaterEqual(HRComparison):
    op = ">="
    py_op = operator.ge

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return numbers_are_greater_equal(x, y)
        return cls.py_op(x, y)


class HRLess(HRComparison):
    op = "<"
    py_op = operator.lt


class HRLessEqual(HRComparison):
    op = "<="
    py_op = operator.le

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return numbers_are_less_equal(x, y)
        return cls.py_op(x, y)


class HRBinNumeric(HRBinOp):
    @classmethod
    def evaluate(cls, left: DataComponent, right: DataComponent, hr_mode: str) -> DataComponent:  # type: ignore[override]
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
            result.data = result.data.iloc[right.data.index[0 : len(result.data)]]

        result.data = result.data[result.data[measure_name] != REMOVE]
        return result

    @classmethod
    def handle_mode(cls, x: Any, hr_mode: str) -> Any:
        remove = (hr_mode == NON_NULL and pd.isnull(x)) or (hr_mode == NON_ZERO and x == 0)
        return REMOVE if remove else x


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
        cls,
        dataset: Dataset,
        computed_dict: Dict[str, DataFrame],
        output: str,
        viral_components: Optional[List[Component]] = None,
        viral_values: Optional[DataFrame] = None,
        node_children: Optional[Dict[str, List[str]]] = None,
        rule_component: Optional[str] = None,
    ) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        result_components = {
            comp_name: copy(comp) for comp_name, comp in dataset.components.items()
        }
        # Viral attributes propagate to the hierarchy result (issue #877).
        for viral_comp in viral_components or []:
            result_components[viral_comp.name] = copy(viral_comp)
        return Dataset(name=dataset_name, components=result_components, data=None)

    @staticmethod
    def _combine_node_viral(
        viral_values: DataFrame,
        viral_names: List[str],
        node_children: Dict[str, List[str]],
        rule_component: str,
        other_ids: List[str],
    ) -> Optional[DataFrame]:
        """Combine child viral values into each computed node via the propagation rule.

        Nodes are processed in topological order (``node_children`` is insertion-ordered by
        the rule sequence); a child that is itself a computed node reuses its combined value.
        """
        registry = get_current_registry()
        node_viral: Dict[str, DataFrame] = {}
        tagged_frames: List[DataFrame] = []
        for node, children in node_children.items():
            child_frames = []
            for child in children:
                if child in node_viral:
                    child_frames.append(node_viral[child])
                else:
                    child_frames.append(
                        viral_values.loc[
                            viral_values[rule_component] == child, other_ids + viral_names
                        ]
                    )
            if not child_frames:
                continue
            combined = pd.concat(child_frames, ignore_index=True)
            if other_ids:
                grouped = combined.groupby(other_ids, sort=False)
                agg = {
                    va: (lambda vals, name=va: registry.resolve_group(name, list(vals)))
                    for va in viral_names
                }
                nv = grouped.agg(agg).reset_index()
            else:
                nv = pd.DataFrame(
                    {va: [registry.resolve_group(va, list(combined[va]))] for va in viral_names}
                )
            node_viral[node] = nv
            tagged = nv.copy()
            tagged[rule_component] = node
            tagged_frames.append(tagged)
        if not tagged_frames:
            return None
        return pd.concat(tagged_frames, ignore_index=True)

    @classmethod
    def evaluate(
        cls,
        dataset: Dataset,
        computed_dict: Dict[str, DataFrame],
        output: str,
        viral_components: Optional[List[Component]] = None,
        viral_values: Optional[DataFrame] = None,
        node_children: Optional[Dict[str, List[str]]] = None,
        rule_component: Optional[str] = None,
    ) -> Dataset:
        result = cls.validate(dataset, computed_dict, output, viral_components=viral_components)
        if len(computed_dict) == 0:
            computed_data = pd.DataFrame(columns=result.get_components_names())
        else:
            computed_data = cls.generate_computed_data(computed_dict)
        # Convert computed id/measure columns to proper pyarrow dtypes (so merge keys match).
        for comp_name, comp in result.components.items():
            if comp_name in computed_data.columns:
                computed_data[comp_name] = computed_data[comp_name].astype(comp.data_type.dtype())  # type: ignore[call-overload]

        viral_names = [vc.name for vc in (viral_components or [])]
        # Combine child viral values for the computed nodes (issue #877).
        if (
            viral_names
            and viral_values is not None
            and node_children
            and rule_component is not None
            and len(computed_data) > 0
        ):
            other_ids = [i for i in dataset.get_identifiers_names() if i != rule_component]
            computed_viral = cls._combine_node_viral(
                viral_values, viral_names, node_children, rule_component, other_ids
            )
            if computed_viral is not None:
                computed_data = computed_data.merge(
                    computed_viral, on=other_ids + [rule_component], how="left"
                )

        if output == "computed":
            result.data = computed_data
        else:
            # union(setdiff(op, R), R): union then drop duplicates, keeping the computed row.
            original = dataset.data
            if viral_names and viral_values is not None and original is not None:
                original = original.merge(
                    viral_values, on=dataset.get_identifiers_names(), how="left"
                )
            result.data = pd.concat([original, computed_data], axis=0, ignore_index=True)
            result.data.drop_duplicates(
                subset=dataset.get_identifiers_names(), keep="last", inplace=True
            )
            result.data.reset_index(drop=True, inplace=True)

        result.data = result.data[result.get_components_names()]
        result.enforce_dtypes()
        return result

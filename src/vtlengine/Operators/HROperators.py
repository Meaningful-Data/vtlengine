import operator
from collections import Counter
from copy import copy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pandas import DataFrame

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY, NON_NULL, NON_ZERO
from vtlengine.DataTypes import Boolean, Number
from vtlengine.Exceptions import RunTimeError
from vtlengine.Model import Component, DataComponent, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._dataframe import merge_frames
from vtlengine.Utils._number_config import (
    numbers_are_equal,
    numbers_are_greater_equal,
    numbers_are_less_equal,
)
from vtlengine.ViralPropagation import (
    apply_viral_return_types,
    get_current_registry,
    require_rules,
)

REMOVE = "REMOVE_VALUE"
# A right code item whose condition is not TRUE is left out of the rule (issue #1168)
EXCLUDE = "EXCLUDE_VALUE"
# The rule that computed each data point of a hierarchy (issue #1193)
RULE = "__rule__"


def unique_data_points(
    data: DataFrame, components: Dict[str, Component], ruleset: str, rule_component: str
) -> DataFrame:
    """Keep once each data point that several rules compute (issue #1193).

    The conditions of the rules that compute a code item may hold on the same data point.
    When all of them give it the same values, it is kept once, with the values of the rule
    evaluated last, the ones later rules read: the data points come in evaluation order.
    Numbers are the same within the tolerance of the comparisons and a null is the same as
    a null; otherwise there would be two data points with the same identifiers.
    """
    identifiers = [name for name in data.columns if components[name].role == Role.IDENTIFIER]
    repeated = data[data.duplicated(subset=identifiers, keep=False)].copy()
    if len(repeated) == 0:
        return data
    values = [name for name in data.columns if name not in identifiers]
    numbers = [name for name in values if components[name].data_type == Number]
    for name in numbers:
        repeated[name] = pd.to_numeric(repeated[name].astype(object), errors="coerce")
    groups = repeated.groupby(identifiers, sort=False)
    filled = groups[values].count()
    # A null and a value are different
    differ = filled.gt(0) & filled.lt(groups.size(), axis=0)
    exact = [name for name in values if name not in numbers]
    if exact:
        differ[exact] |= groups[exact].nunique() > 1
    for name in numbers:
        low, high = groups[name].min(), groups[name].max()
        same = [pd.isna(a) or numbers_are_equal(a, b) for a, b in zip(low, high)]
        differ[name] |= ~pd.Series(same, index=low.index, dtype=bool)
    clashes = differ.index[differ.any(axis=1)]
    if len(clashes) > 0:
        key = clashes[0]
        code_item = key[identifiers.index(rule_component)] if isinstance(key, tuple) else key
        raise RunTimeError("2-1-10-1", ruleset=ruleset, code_item=code_item)
    return data.drop_duplicates(subset=identifiers, keep="last")


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
        # Convert to object dtype for sentinel-based alignment. A rule whose right code
        # items are all excluded reads them as missing.
        left_obj = left.astype(object).replace(EXCLUDE, REMOVE)
        right_obj = (
            right.astype(object).replace(EXCLUDE, REMOVE) if isinstance(right, pd.Series) else right
        )
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
        left_obj, right_obj = left_series.astype(object).align(
            right_series.astype(object), join="outer"
        )
        left_obj[left_obj.index.difference(left_series.index, sort=False)] = REMOVE
        right_obj[right_obj.index.difference(right_series.index, sort=False)] = REMOVE
        excluded = (left_obj == EXCLUDE) | (right_obj == EXCLUDE)
        left, right = cls.align_series(left_obj[~excluded], right_obj[~excluded], hr_mode)
        result = cls.apply_operation_two_series(left, right)
        if not excluded.any():
            return result
        kept = pd.Series(
            map(cls.without_excluded, left_obj[excluded], right_obj[excluded]),
            index=left_obj.index[excluded],
            dtype=object,
        )
        return pd.concat([result, kept]).sort_index()

    @classmethod
    def without_excluded(cls, left: Any, right: Any) -> Any:
        """Leave an excluded operand out of the operation (issue #1168): the other one is
        combined with 0, a missing or excluded one is kept as it is."""
        left_excluded = isinstance(left, str) and left == EXCLUDE
        other, zero_left = (right, True) if left_excluded else (left, False)
        if isinstance(other, str) or pd.isnull(other):
            return other
        return cls.op_func(0, other) if zero_left else cls.op_func(other, 0)


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
        # A missing or excluded code item is kept as it is
        result_data = (
            None
            if operand.data is None
            else operand.data.map(
                lambda x: x if isinstance(x, str) else cls.op_func(x), na_action="ignore"
            )
        )
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
            # The partial modes filter the operands, so their labels are not positions.
            result.data = result.data[result.data.index.isin(right.data.index)]

        result.data = result.data[result.data[measure_name] != REMOVE]
        return result

    @classmethod
    def handle_mode(cls, x: Any, hr_mode: str) -> Any:
        # A right side whose code items are all excluded reads as missing (issue #1168)
        remove = (
            isinstance(x, str)
            or (hr_mode == NON_NULL and pd.isnull(x))
            or (hr_mode == NON_ZERO and not pd.isnull(x) and x == 0)
        )
        return REMOVE if remove else x


class Hierarchy(Operators.Operator):
    op = HIERARCHY

    @staticmethod
    def generate_computed_data(computed_dict: Dict[str, DataFrame]) -> DataFrame:
        # Each data point keeps the rule that computed it (issue #1193)
        list_data = [data.assign(**{RULE: name}) for name, data in computed_dict.items()]
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
        node_children: Optional[Dict[str, Tuple[str, List[str]]]] = None,
        rule_component: Optional[str] = None,
        ruleset: Optional[str] = None,
    ) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        result_components = {
            comp_name: copy(comp) for comp_name, comp in dataset.components.items()
        }
        # Viral attributes propagate to the hierarchy result (issue #877).
        for viral_comp in viral_components or []:
            result_components[viral_comp.name] = copy(viral_comp)
        # The roll-up combines child nodes into each computed node, so the combined viral
        # attributes require a propagation rule (issue #906).
        require_rules(viral_components or [])
        apply_viral_return_types(viral_components or [], result_components)
        return Dataset(name=dataset_name, components=result_components, data=None)

    @staticmethod
    def _combine_node_viral(
        viral_values: DataFrame,
        viral_names: List[str],
        node_children: Dict[str, Tuple[str, List[str]]],
        rule_component: str,
        other_ids: List[str],
        computed_dict: Dict[str, DataFrame],
    ) -> Optional[DataFrame]:
        """Combine child viral values into each computed node via the propagation rule.

        ``node_children`` maps each rule to the node it computes and the children it
        combines. Rules are processed in topological order (the mapping is insertion-ordered
        by the rule sequence); a child that is itself a computed node reuses its combined
        value. When several rules compute the same node, each one only combines the groups
        it computed (issue #1142).
        """
        registry = get_current_registry()
        rules_per_node = Counter(node for node, _ in node_children.values())
        node_viral: Dict[str, DataFrame] = {}
        tagged_frames: List[DataFrame] = []
        for rule_name, (node, children) in node_children.items():
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
                # A missing child data point contributes a null attribute value to the
                # combination (issue #969): left-join every child onto the node's group
                # universe so the propagation rule sees one value per child and group.
                groups = combined[other_ids].drop_duplicates()
                padded = [merge_frames(groups, cf, on=other_ids, how="left") for cf in child_frames]
                combined = pd.concat(padded, ignore_index=True)
                grouped = combined.groupby(other_ids, sort=False)
                agg = {
                    va: (lambda vals, name=va: registry.resolve_group(name, list(vals)))
                    for va in viral_names
                }
                nv = grouped.agg(agg).reset_index()
            else:
                # One value per child; a missing child contributes a null (issue #969).
                nv = pd.DataFrame(
                    {
                        va: [
                            registry.resolve_group(
                                va,
                                [cf[va].iloc[0] if len(cf) else None for cf in child_frames],
                            )
                        ]
                        for va in viral_names
                    }
                )
            if rules_per_node[node] > 1:
                if rule_name not in computed_dict:
                    continue
                if other_ids:
                    computed_groups = computed_dict[rule_name][other_ids].drop_duplicates()
                    nv = merge_frames(nv, computed_groups, on=other_ids, how="inner")
            if node in node_viral:
                # A later rule combines the values of the last rule that computed the node
                # for a group, the rule whose measure it reads (issue #1193)
                both = pd.concat([node_viral[node], nv], ignore_index=True)
                node_viral[node] = (
                    both.drop_duplicates(subset=other_ids, keep="last")
                    if other_ids
                    else both.tail(1)
                )
            else:
                node_viral[node] = nv
            tagged = nv.copy()
            tagged[rule_component] = node
            tagged[RULE] = rule_name
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
        node_children: Optional[Dict[str, Tuple[str, List[str]]]] = None,
        rule_component: Optional[str] = None,
        ruleset: Optional[str] = None,
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
                viral_values, viral_names, node_children, rule_component, other_ids, computed_dict
            )
            if computed_viral is not None:
                # Each data point takes the viral values of the rule that computed it
                computed_data = merge_frames(
                    computed_data,
                    computed_viral,
                    on=[*other_ids, rule_component, RULE],
                    how="left",
                )
        computed_data = computed_data.drop(columns=RULE, errors="ignore")

        rules_per_node = Counter(node for node, _ in (node_children or {}).values())
        repeated_nodes = max(rules_per_node.values(), default=0) > 1
        if ruleset is not None and rule_component is not None and repeated_nodes:
            # Several rules may compute the same data point: it is kept once when they give
            # it the same measure and viral attribute values (issue #1193)
            computed_data = unique_data_points(
                computed_data, result.components, ruleset, rule_component
            ).reset_index(drop=True)

        if output == "computed":
            result.data = computed_data
        else:
            # union(setdiff(op, R), R): union then drop duplicates, keeping the computed row.
            original = dataset.data
            if viral_names and viral_values is not None and original is not None:
                original = merge_frames(
                    original, viral_values, on=dataset.get_identifiers_names(), how="left"
                )
            result.data = pd.concat([original, computed_data], axis=0, ignore_index=True)
            result.data.drop_duplicates(
                subset=dataset.get_identifiers_names(), keep="last", inplace=True
            )
            result.data.reset_index(drop=True, inplace=True)

        result.data = result.data[result.get_components_names()]
        result.enforce_dtypes()
        return result

from copy import copy
from typing import Any, Dict, List, Union, cast

import pandas as pd

from vtlengine.DataTypes import (
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    Null,
    binary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar
from vtlengine.Operators import Binary, Operator, _id_type_promotion_join_keys
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._dataframe import merge_frames
from vtlengine.ViralPropagation import (
    apply_viral_return_types,
    combined_viral_components,
    get_current_registry,
    require_rules,
)

COND_COL = "__cond__"


def component_assign(cond: Any, op: Union[DataComponent, Scalar]) -> Any:
    idx = cond.index[cond.fillna(False)]
    if isinstance(op, DataComponent):
        return pd.Series(dtype=op.data_type.dtype()) if op.data is None else op.data.reindex(idx)
    return pd.Series(op.value, index=idx)


def dataset_assign(
    cond: pd.DataFrame, op: Union[Dataset, Scalar], ids: List[str], measures: List[str]
) -> pd.DataFrame:
    if isinstance(op, Dataset):
        if op.data is None or cond.empty:
            # All the Components, or the result lacks the Attributes when no data point takes
            # this operand (issue #1304)
            return pd.DataFrame(columns=op.get_components_names() + [COND_COL])
        return merge_frames(cond, op.data, on=ids, how="inner")
    return cond.assign(**dict.fromkeys(measures, op.value))


def _promote_measures(components: Dict[str, Component], operands: List[Any]) -> None:
    """Give each Measure of ``components`` the common type of its own type and the type of
    every operand, a Scalar or a Data Set with the same Components. A retyped Measure is
    replaced with a copy: the Components may be the ones of an operand."""
    for name, comp in list(components.items()):
        if comp.role != Role.MEASURE:
            continue
        data_type = comp.data_type
        for op in operands:
            op_type = op.components[name].data_type if isinstance(op, Dataset) else op.data_type
            data_type = binary_implicit_promotion(data_type, op_type)
        if data_type != comp.data_type:
            components[name] = copy(comp)
            components[name].data_type = data_type


def _combine_viral_attributes(result: Any, operands: List[Any]) -> None:
    """Combine the viral attributes, as the viral attribute binary combination"""
    if not isinstance(result, Dataset) or result.data is None:
        return
    res_data = result.data
    ds_ops = [op for op in operands if isinstance(op, Dataset) and op.data is not None]
    if len(ds_ops) < 2:
        return
    registry = get_current_registry()
    ids = result.get_identifiers_names()
    n = len(res_data)
    for va in result.get_viral_attributes_names():
        carriers: List[pd.DataFrame] = [
            op.data
            for op in ds_ops
            if op.data is not None
            and va in op.get_viral_attributes_names()
            and va in op.data.columns
        ]
        if len(carriers) < 2:
            continue
        if ids:
            base = res_data[ids]
            cols = [
                list(
                    merge_frames(
                        base, d[ids + [va]].rename(columns={va: "__v__"}), on=ids, how="left"
                    )["__v__"]
                )
                for d in carriers
            ]
        else:
            cols = [list(d[va]) for d in carriers]
        res_data[va] = [registry.resolve_group(va, [col[r] for col in cols]) for r in range(n)]


class If(Operator):
    """
    If class:
        `If-then-else <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=225&zoom=100,72,142>`_ operator
        inherits from Operator, a superclass that contains general validate and evaluate class methods.
        It has the following class methods:
    Class methods:
        evaluate: Evaluates if the operation is well constructed, checking the actual condition and
        dropping a boolean result.
        The result will depend on the data class, such as datacomponent and dataset.

        component_level_evaluation: Returns a pandas dataframe with data to set the condition

        dataset_level_evaluation: Sets the dataset and evaluates its correct schema to be able to perform the condition.

        validate: Class method that has two branches so datacomponent and datasets can be validated. With datacomponent,
        the code reviews if it is actually a Measure and if it is a binary operation. Dataset branch reviews if the
        identifiers are the same in 'if', 'then' and 'else'.
        is_from_having: Boolean flag indicating if the evaluation is being performed in the context of a HAVING clause.

    """  # noqa E501

    @classmethod
    def evaluate(
        cls, condition: Any, true_branch: Any, false_branch: Any, is_from_having: bool = False
    ) -> Any:
        result = cls.validate(condition, true_branch, false_branch, is_from_having)
        if isinstance(result, DataComponent):
            result.data = cls.component_level_evaluation(condition, true_branch, false_branch)
        elif isinstance(result, Dataset):
            cls.dataset_level_evaluation(result, condition, true_branch, false_branch)
        return result

    @classmethod
    def component_level_evaluation(
        cls,
        condition: DataComponent,
        true_branch: Union[DataComponent, Scalar],
        false_branch: Union[DataComponent, Scalar],
    ) -> Any:
        if condition.data is None:
            return pd.Series()

        cond = condition.data.fillna(False).astype("bool[pyarrow]")
        t_base = component_assign(cond, true_branch)
        f_base = component_assign(~cond, false_branch)
        return pd.concat([t_base, f_base])

    @classmethod
    def dataset_level_evaluation(
        cls,
        result: Dataset,
        condition: Dataset,
        true_branch: Union[Dataset, Scalar],
        false_branch: Union[Dataset, Scalar],
    ) -> None:
        if condition.data is None:
            result.data = pd.DataFrame(columns=result.get_components_names())
            return

        ids = result.get_identifiers_names()
        measures = result.get_measures_names()

        cond_measure = condition.get_measures_names()[0]
        # condition.data is the DataFrame the stored condition variable holds, so it
        # must not be mutated: popping the measure out of it broke every statement
        # that read the condition dataset afterwards (issue #1024).
        cond = condition.data[ids].copy()
        cond[COND_COL] = condition.data[cond_measure].fillna(False).astype("bool[pyarrow]")

        t_base = dataset_assign(cond[cond[COND_COL]], true_branch, ids, measures)
        f_base = dataset_assign(cond[~cond[COND_COL]], false_branch, ids, measures)
        # The merge needs the same dtypes on both sides: the one of the result Component, the
        # common type of both operands. The dtype of the then operand cut the Number values of
        # the else operand under an Integer then operand, and failed on its nulls (issue #1314).
        for col in t_base.columns.intersection(f_base.columns):
            if col != COND_COL and str(t_base[col].dtype) != str(f_base[col].dtype):
                common_dtype = result.components[col].data_type.dtype()
                t_base[col] = t_base[col].astype(common_dtype)  # type: ignore[call-overload]
                f_base[col] = f_base[col].astype(common_dtype)  # type: ignore[call-overload]
        result.data = merge_frames(t_base, f_base, how="outer").drop(columns=COND_COL)
        _combine_viral_attributes(result, [true_branch, false_branch])

    @classmethod
    def validate(  # noqa: C901
        cls, condition: Any, true_branch: Any, false_branch: Any, is_from_having: bool = False
    ) -> Union[Scalar, DataComponent, Dataset]:
        nullable = False
        left = true_branch
        right = false_branch
        dataset_name = VirtualCounter._new_ds_name()
        if true_branch.__class__ != false_branch.__class__:
            if (isinstance(true_branch, DataComponent) and isinstance(false_branch, Dataset)) or (
                isinstance(true_branch, Dataset) and isinstance(false_branch, DataComponent)
            ):
                raise SemanticError(
                    "1-1-9-12",
                    op=cls.op,
                    then_symbol=true_branch.name,
                    else_symbol=false_branch.name,
                )
            if isinstance(true_branch, Scalar):
                left = false_branch
                right = true_branch

        # Datacomponent
        comp_name = VirtualCounter._new_dc_name()
        if isinstance(condition, DataComponent):
            if not condition.data_type == Boolean:
                raise SemanticError(
                    "1-1-9-11",
                    op=cls.op,
                    type=SCALAR_TYPES_CLASS_REVERSE[condition.data_type],
                )
            if left.data_type == Null or right.data_type == Null:
                nullable = True
            if isinstance(left, DataComponent):
                nullable |= left.nullable
            if isinstance(right, DataComponent):
                nullable |= right.nullable
            return DataComponent(
                name=comp_name,
                data=None,
                data_type=binary_implicit_promotion(left.data_type, right.data_type),
                role=Role.MEASURE,
                nullable=nullable,
            )

        # Dataset
        if isinstance(left, Scalar) and isinstance(right, Scalar) and not is_from_having:
            raise SemanticError(
                "1-1-9-12", op=cls.op, then_symbol=left.name, else_symbol=right.name
            )
        if isinstance(left, DataComponent):
            raise SemanticError(
                "1-1-9-12", op=cls.op, then_symbol=left.name, else_symbol=right.name
            )
        if isinstance(left, Scalar):
            left.data_type = right.data_type = binary_implicit_promotion(
                left.data_type, right.data_type
            )
            return Dataset(name=dataset_name, components=copy(condition.components), data=None)
        if left.get_identifiers() != condition.get_identifiers():
            raise SemanticError("1-1-9-10", op=cls.op, clause=left.name)
        if isinstance(right, Scalar):
            for component in left.get_measures():
                if component.data_type != right.data_type:
                    component.data_type = binary_implicit_promotion(
                        component.data_type, right.data_type
                    )
        if isinstance(right, Dataset):
            if left.get_identifiers() != condition.get_identifiers():
                raise SemanticError("1-1-9-10", op=cls.op, clause=right.name)
            # The branches must hold the same Components, in whatever order each one
            # happens to declare them (issue #1008).
            if set(left.get_components_names()) != set(right.get_components_names()):
                raise SemanticError("1-1-9-13", op=cls.op, then=left.name, else_clause=right.name)
            for component in left.get_measures():
                if component.data_type != right.components[component.name].data_type:
                    component.data_type = right.components[component.name].data_type = (
                        binary_implicit_promotion(
                            component.data_type,
                            right.components[component.name].data_type,
                        )
                    )
        if isinstance(condition, Dataset):
            if len(condition.get_measures()) != 1:
                raise SemanticError("1-1-9-4", op=cls.op, name=condition.name)
            if condition.get_measures()[0].data_type != Boolean:
                raise SemanticError(
                    "1-1-9-5",
                    op=cls.op,
                    type=SCALAR_TYPES_CLASS_REVERSE[condition.get_measures()[0].data_type],
                )
            if left.get_identifiers() != condition.get_identifiers():
                raise SemanticError("1-1-9-6", op=cls.op)
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        # if-then-else over two datasets combines their data points per row, so viral
        # attributes carried by both branches require a propagation rule (issue #906).
        combined = combined_viral_components([b for b in (left, right) if isinstance(b, Dataset)])
        require_rules(combined)
        apply_viral_return_types(combined, result_components)
        return Dataset(name=dataset_name, components=result_components, data=None)


class Nvl(Binary):
    """
    Null class:
        `Nvl <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=229&zoom=100,72,370>`_operator class.
        It has the following class methods:

    Class methods:
        Validate: Class method that validates if the operation at scalar,
        datacomponent or dataset level can be performed.
        Evaluate: Evaluates the actual operation, returning the result.
    """  # noqa E501

    @classmethod
    def evaluate(cls, left: Any, right: Any) -> Union[Scalar, DataComponent, Dataset]:
        result = cls.validate(left, right)

        if isinstance(left, Scalar) and isinstance(result, Scalar):
            if left.data_type is Null:
                result.data_type = right.data_type
                result.value = right.value
            elif right.data_type is Null:
                result.data_type = left.data_type
                result.value = left.value
            else:
                result.value = left.value
        else:
            if not isinstance(result, Scalar):
                if isinstance(result, Dataset):
                    result.data = left.data.copy()
                    for me in result.get_measures_names():
                        if me in result.data.columns:
                            result.data[me] = result.data[me].astype(  # type: ignore[call-overload]
                                result.components[me].data_type.dtype()
                            )
                else:
                    result.data = left.data.astype(result.data_type.dtype())
                if isinstance(right, Scalar):
                    # A null applicable value replaces nothing, so the operand is
                    # returned as it is instead of filling with it (issue #1008).
                    fill_null = right.value is None
                    if isinstance(result, Dataset):
                        for me in result.get_measures_names():
                            if me in result.data.columns and not fill_null:
                                result.data[me] = result.data[me].fillna(right.value)
                    elif not fill_null:
                        result.data = result.data.fillna(right.value)
                elif isinstance(result, Dataset):
                    join_keys = [
                        id_name
                        for id_name in left.get_identifiers_names()
                        if id_name in right.get_identifiers_names()
                    ]
                    measures = [
                        me for me in result.get_measures_names() if me in right.data.columns
                    ]
                    right_data = right.data[join_keys + measures].copy()
                    for join_key in join_keys:
                        _id_type_promotion_join_keys(
                            left.get_component(join_key),
                            right.get_component(join_key),
                            join_key,
                            result.data,
                            right_data,
                        )
                    result.data = merge_frames(
                        result.data, right_data, how="inner", on=join_keys, suffixes=("", "_r")
                    )
                    for me in measures:
                        result.data[me] = (
                            result.data[me]
                            .fillna(result.data[f"{me}_r"])
                            .astype(result.components[me].data_type.dtype())  # type: ignore[call-overload]
                        )
                else:
                    # nvl is single-operand: the primary operand's viral attributes are
                    # copied through unchanged; no rule is executed (issue #906).
                    result.data = result.data.fillna(right.data)
                if isinstance(result, Dataset):
                    result.data = result.data[result.get_components_names()]
        return result

    @classmethod
    def validate(cls, left: Any, right: Any) -> Union[Scalar, DataComponent, Dataset]:
        dataset_name = VirtualCounter._new_ds_name()
        comp_name = VirtualCounter._new_dc_name()
        result_components = {}
        keeps_nulls = isinstance(right, Scalar) and (right.data_type is Null or right.value is None)
        if isinstance(right, DataComponent):
            keeps_nulls = right.nullable
        if isinstance(left, Scalar):
            if not isinstance(right, Scalar):
                raise SemanticError(
                    "1-1-9-2", op=cls.op, left_type="Scalar", right_type=type(right).__name__
                )
            data_type = cls.type_validation(left.data_type, right.data_type)
            # Matching main (#762): a scalar nvl result is reported non-nullable. The flag
            # must not depend on the operand VALUES (keeps_nulls tests right.value), or
            # semantic_analysis and run() would disagree on the same script's structure.
            return Scalar(name="result", value=None, data_type=data_type, nullable=False)
        if isinstance(left, DataComponent):
            if isinstance(right, Dataset):
                raise SemanticError(
                    "1-1-9-2", op=cls.op, left_type="Component", right_type="Data Set"
                )
            data_type = cls.type_validation(left.data_type, right.data_type)
            return DataComponent(
                name=comp_name,
                data=pd.Series(dtype=data_type.dtype()),
                data_type=data_type,
                role=Role.MEASURE,
                nullable=keeps_nulls and left.nullable,
            )
        if isinstance(left, Dataset):
            if isinstance(right, DataComponent):
                raise SemanticError(
                    "1-1-9-2", op=cls.op, left_type="Data Set", right_type="Component"
                )
            promoted = {}
            if isinstance(right, Scalar):
                for component in left.get_measures():
                    promoted[component.name] = cls.type_validation(
                        component.data_type, right.data_type
                    )
            if isinstance(right, Dataset):
                for component in left.get_measures():
                    promoted[component.name] = cls.type_validation(
                        component.data_type, right.components[component.name].data_type
                    )
            result_components = {
                comp_name: copy(comp)
                for comp_name, comp in left.components.items()
                if comp.role != Role.ATTRIBUTE
            }
            for comp in result_components.values():
                replacement_keeps_nulls = keeps_nulls
                if isinstance(right, Dataset) and comp.name in right.components:
                    replacement_keeps_nulls = right.components[comp.name].nullable
                comp.nullable = replacement_keeps_nulls and comp.nullable
                if comp.name in promoted:
                    comp.data_type = promoted[comp.name]
        return Dataset(name=dataset_name, components=result_components, data=None)


class Case(Operator):
    @classmethod
    def evaluate(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any, is_from_having: bool = False
    ) -> Union[Scalar, DataComponent, Dataset]:
        result = cls.validate(conditions, thenOps, elseOp, is_from_having)
        if isinstance(result, Scalar):
            # Take the value of the same operand validate takes the data type from
            # (issue #1008).
            result.value = cls._selected_scalar(conditions, thenOps, elseOp).value
        else:
            operation_level = list({type(c) for c in conditions if not isinstance(c, Scalar)})
            if operation_level[0] == DataComponent:
                component = cast(DataComponent, result)
                component.data = cls.component_level_evaluation(
                    conditions, thenOps, elseOp, component.data_type.dtype()
                )
            else:
                cls.dataset_level_evaluation(result, conditions, thenOps, elseOp)
        return result

    @staticmethod
    def _selected_scalar(conditions: List[Scalar], thenOps: List[Scalar], elseOp: Scalar) -> Scalar:
        """The then operand of the first true condition, or the else operand if none is
        (issue #1251)."""
        return next(
            (thenOp for condition, thenOp in zip(conditions, thenOps) if condition.value),
            elseOp,
        )

    @classmethod
    def component_level_evaluation(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any, result_dtype: str
    ) -> Any:
        # Build the result on an object Series: the branches come in different dtypes
        # (numpy float, pyarrow double, string...) and pandas 3 no longer lets a .loc
        # assignment widen the target's dtype. The declared result dtype is applied
        # once every branch is in place.
        if isinstance(elseOp, DataComponent):
            result = pd.Series(dtype=object) if elseOp.data is None else elseOp.data.astype(object)
        else:
            result = pd.Series(elseOp.value, index=conditions[0].data.index, dtype=object)

        # The branches are written from the last to the first: where several conditions
        # are true, the then operand of the first one is the one that stays (issue #1251).
        for i in reversed(range(len(conditions))):
            case = conditions[i].data[conditions[i].data.fillna(False).astype("bool[pyarrow]")]
            case_result = component_assign(case, thenOps[i])
            result = result.reindex(result.index.union(case.index))
            result.loc[case.index] = case_result

        return result.astype(result_dtype)  # type: ignore[call-overload]

    @classmethod
    def dataset_level_evaluation(
        cls, result: Any, conditions: List[Any], thenOps: List[Any], elseOp: Any
    ) -> None:
        ids = result.get_identifiers_names()
        measures = result.get_measures_names()

        # The first condition gives the data points, and each one takes the operand of the
        # first condition that is true on it, or the else operand when none is (issue #1251):
        # the true data points of a condition that the first condition has and no earlier
        # condition took are inner joined with its then operand, the data points no condition
        # took with the else operand, and the result is the union (issue #1304).
        parts = []
        first = pd.MultiIndex.from_frame(conditions[0].data[ids])
        taken = first[:0]
        for condition, thenOp in zip(conditions, thenOps):
            case = condition.data.rename(columns={condition.get_measures_names()[0]: COND_COL})
            case = case[ids + [COND_COL]]
            case = case[case[COND_COL].fillna(False)]
            keys = pd.MultiIndex.from_frame(case[ids])
            parts.append(
                dataset_assign(case[keys.isin(first) & ~keys.isin(taken)], thenOp, ids, measures)
            )
            taken = taken.append(keys)

        else_cond = conditions[0].data[ids].copy()
        else_cond[COND_COL] = True
        else_cond = else_cond[~pd.MultiIndex.from_frame(else_cond[ids]).isin(taken)]
        parts.append(dataset_assign(else_cond, elseOp, ids, measures))

        result.data = pd.concat(parts, ignore_index=True).drop(columns=COND_COL)
        _combine_viral_attributes(result, [*thenOps, elseOp])

    @classmethod
    def validate(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any, is_from_having: bool = False
    ) -> Union[Scalar, DataComponent, Dataset]:
        dataset_name = VirtualCounter._new_ds_name()
        comp_name = VirtualCounter._new_dc_name()
        if len(set(map(type, conditions))) > 1:
            raise SemanticError("1-1-9-14", op=cls.op)

        ops = thenOps + [elseOp]
        then_else_types = set(map(type, ops))
        condition_type = type(conditions[0])

        if condition_type is Scalar:
            for condition in conditions:
                if condition.data_type != Boolean:
                    raise SemanticError("1-1-9-15", op=cls.op, name=condition.name)
            if list(then_else_types) != [Scalar]:
                raise SemanticError("1-1-9-16", op=cls.op)

            output_data_type = cls._selected_scalar(conditions, thenOps, elseOp).data_type

            nullable = any(op.nullable or op.data_type == Null for op in ops)
            return Scalar(
                name="result",
                value=None,
                data_type=output_data_type,
                nullable=nullable,
            )

        elif condition_type is DataComponent:
            for condition in conditions:
                if not condition.data_type == Boolean:
                    raise SemanticError("1-1-9-17", op=cls.op, name=condition.name)

            nullable = any(
                (thenOp.nullable if isinstance(thenOp, DataComponent) else thenOp.data_type == Null)
                for thenOp in ops
            )

            data_type = ops[0].data_type
            for op in ops[1:]:
                data_type = binary_implicit_promotion(data_type, op.data_type)

            return DataComponent(
                name=comp_name,
                data=None,
                data_type=data_type,
                role=Role.MEASURE,
                nullable=nullable,
            )

        # Dataset
        for condition in conditions:
            if len(condition.get_measures_names()) != 1:
                raise SemanticError("1-1-1-4", op=cls.op)
            if condition.get_measures()[0].data_type != Boolean:
                raise SemanticError("1-1-9-18", op=cls.op, name=condition.name)

        if Dataset not in then_else_types:
            if not is_from_having:
                raise SemanticError("1-1-9-19", op=cls.op)

            data_type = ops[0].data_type
            for branch in ops[1:]:
                data_type = binary_implicit_promotion(data_type, branch.data_type)
            measure_name = condition.get_measures_names()[0]
            components = {comp.name: comp for comp in conditions[0].get_identifiers()}
            components[measure_name] = Component(
                name=measure_name,
                data_type=data_type,
                role=Role.MEASURE,
                nullable=any(branch.nullable or branch.data_type == Null for branch in ops),
            )
            return Dataset(name=dataset_name, components=components, data=None)

        # Rebuild the dict: the result must never share it with the branch dataset, as
        # apply_viral_return_types may replace entries in it.
        components = dict(next(op for op in ops if isinstance(op, Dataset)).components)
        comp_names = {comp.name for comp in components.values()}
        for op in ops:
            # The operands must hold the same Components, in whatever order each one
            # happens to declare them (issue #1008).
            if isinstance(op, Dataset) and set(op.get_components_names()) != comp_names:
                raise SemanticError("1-1-9-20", op=cls.op)

        # The Measures take the common type of the operands, as in a Data Set if, and not the
        # type of the first Data Set operand (issue #1314)
        _promote_measures(components, ops)

        # case over two or more datasets combines their data points, so viral attributes
        # carried by two or more branches require a propagation rule (issue #906).
        combined = combined_viral_components([op for op in ops if isinstance(op, Dataset)])
        require_rules(combined)
        apply_viral_return_types(combined, components)
        return Dataset(name=dataset_name, components=components, data=None)

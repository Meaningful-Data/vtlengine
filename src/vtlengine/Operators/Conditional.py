from copy import copy
from typing import Any, Dict, List, Union, cast

import pandas as pd

from vtlengine.DataTypes import (
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    Null,
    String,
    binary_implicit_promotion,
    to_strings,
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


def _supertype(operands: List[Any]) -> Any:
    """The common type of the scalar or Component operands of an if or a case: the
    SemanticError 1-1-1-1 when they have none, and the other type for a null operand."""
    data_type = operands[0].data_type
    for op in operands[1:]:
        data_type = binary_implicit_promotion(data_type, op.data_type)
    return data_type


def _selected_value(operand: Scalar, data_type: Any) -> Any:
    """The value of the scalar operand that a scalar if or case selects, in the type of the
    result."""
    if operand.value is None or operand.data_type == data_type:
        return operand.value
    return data_type.implicit_cast(operand.value, operand.data_type)


def _as_type(data: Any, data_type: Any) -> Any:
    """The data in the dtype of ``data_type``. As a String, a Boolean is "true" or "false", as
    VTL casts it, where astype writes a numpy or Python Boolean as "True" (issue #1314)."""
    return to_strings(data) if data_type is String else data.astype(data_type.dtype())


def _identifiers(dataset: Dataset) -> Dict[str, Component]:
    """The Identifiers of a Data Set by name: an operand of an if or a case matches the
    condition in whatever order each one declares them, as their Components (issue #1008)."""
    return {comp.name: comp for comp in dataset.get_identifiers()}


def _promote_measures(components: Dict[str, Component], operands: List[Any]) -> None:
    """Give each Measure of ``components`` the common type of its own type and the type of
    every operand, a Scalar or a Data Set with the same Components, and make it nullable
    when an operand can be null in it (issue #1314). A changed Measure is replaced with a
    copy: the Components may be the ones of an operand."""
    for name, comp in list(components.items()):
        if comp.role != Role.MEASURE:
            continue
        data_type, nullable = comp.data_type, comp.nullable
        for op in operands:
            op_comp = op.components[name] if isinstance(op, Dataset) else op
            data_type = binary_implicit_promotion(data_type, op_comp.data_type)
            nullable = nullable or op_comp.nullable or op_comp.data_type == Null
        if data_type != comp.data_type or nullable != comp.nullable:
            components[name] = copy(comp)
            components[name].data_type = data_type
            components[name].nullable = nullable


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
        # A column of the type of the Attribute: a list of nulls made a column of Python None,
        # which the viral rule of an operator that takes this result directly cannot add to a
        # number (issue #1314)
        res_data[va] = pd.Series(
            [registry.resolve_group(va, [col[r] for col in cols]) for r in range(n)],
            index=res_data.index,
            dtype=result.components[va].data_type.dtype(),
        )


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
            if result.data_type is String:
                result.data = to_strings(result.data)
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
        # In the order of the Data Points, not the then part first: the operator that takes
        # the result pairs its values with another operand by position (issue #1314)
        return pd.concat([t_base, f_base]).reindex(cond.index)

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
            comp = result.components.get(col)
            if comp is not None and (
                comp.data_type is String or str(t_base[col].dtype) != str(f_base[col].dtype)
            ):
                t_base[col] = _as_type(t_base[col], comp.data_type)
                f_base[col] = _as_type(f_base[col], comp.data_type)
        # The parts hold different Data Points, so the result is their union, in the order of
        # the Identifiers. An outer merge on all the columns also compared the Measures, and
        # failed when one part held 0.0 and the other -0.0 (issue #1314)
        data = pd.concat([t_base, f_base], ignore_index=True).drop(columns=COND_COL)
        result.data = data.sort_values(ids, ignore_index=True)
        _combine_viral_attributes(result, [true_branch, false_branch])

    @classmethod
    def scalar_validation(
        cls, condition: Scalar, true_branch: Scalar, false_branch: Scalar
    ) -> Scalar:
        """A scalar if needs a Boolean condition, and takes the common type of its operands,
        as at Component and Data Set level, and the value of the operand that its condition
        selects (issue #1314). A null literal condition selects the else operand."""
        if condition.data_type not in (Boolean, Null):
            raise SemanticError(
                "1-1-9-1", op=cls.op, type=SCALAR_TYPES_CLASS_REVERSE[condition.data_type]
            )
        data_type = _supertype([true_branch, false_branch])
        selected = true_branch if condition.value else false_branch
        return Scalar(
            name="result",
            value=_selected_value(selected, data_type),
            data_type=data_type,
            # Null when either operand can be, as a case, whatever the operand selected
            nullable=any(op.nullable or op.data_type == Null for op in (true_branch, false_branch)),
            literal=selected.literal,
        )

    @classmethod
    def validate(  # noqa: C901
        cls, condition: Any, true_branch: Any, false_branch: Any, is_from_having: bool = False
    ) -> Union[Scalar, DataComponent, Dataset]:
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

        if isinstance(condition, Scalar):
            return cls.scalar_validation(condition, true_branch, false_branch)

        # Datacomponent
        comp_name = VirtualCounter._new_dc_name()
        if isinstance(condition, DataComponent):
            if not condition.data_type == Boolean:
                raise SemanticError(
                    "1-1-9-11",
                    op=cls.op,
                    type=SCALAR_TYPES_CLASS_REVERSE[condition.data_type],
                )
            # A scalar operand can be null too, as a nullable input scalar (issue #1314)
            nullable = any(op.nullable or op.data_type == Null for op in (left, right))
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
            # Two Scalars (having) only need a common type: an operand is never retyped
            binary_implicit_promotion(left.data_type, right.data_type)
            return Dataset(name=dataset_name, components=copy(condition.components), data=None)
        if _identifiers(left) != _identifiers(condition):
            raise SemanticError("1-1-9-10", op=cls.op, clause=left.name)
        if isinstance(right, Dataset):
            if _identifiers(right) != _identifiers(condition):
                raise SemanticError("1-1-9-10", op=cls.op, clause=right.name)
            # The branches must hold the same Components, in whatever order each one
            # happens to declare them (issue #1008).
            if set(left.get_components_names()) != set(right.get_components_names()):
                raise SemanticError("1-1-9-13", op=cls.op, then=left.name, else_clause=right.name)
        # The Measures take the common type of both operands on the Components of the result:
        # the ones of the operands keep their types for the statements after this one
        # (issue #1314)
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        _promote_measures(result_components, [right])
        if isinstance(condition, Dataset):
            if len(condition.get_measures()) != 1:
                raise SemanticError("1-1-9-4", op=cls.op, name=condition.name)
            if condition.get_measures()[0].data_type != Boolean:
                raise SemanticError(
                    "1-1-9-5",
                    op=cls.op,
                    type=SCALAR_TYPES_CLASS_REVERSE[condition.get_measures()[0].data_type],
                )
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
                # The value of the result type, the applicable one where the operand is null
                selected = right if left.value is None else left
                result.value = _selected_value(selected, result.data_type)
        else:
            if not isinstance(result, Scalar):
                if isinstance(result, Dataset):
                    result.data = left.data.copy()
                    for me in result.get_measures_names():
                        if me in result.data.columns:
                            result.data[me] = _as_type(
                                result.data[me], result.components[me].data_type
                            )
                else:
                    result.data = _as_type(left.data, result.data_type)
                if isinstance(right, Scalar):
                    # A null applicable value replaces nothing, so the operand is
                    # returned as it is instead of filling with it (issue #1008).
                    fill_null = right.value is None
                    if isinstance(result, Dataset):
                        for me in result.get_measures_names():
                            if me in result.data.columns and not fill_null:
                                value = _selected_value(right, result.components[me].data_type)
                                result.data[me] = result.data[me].fillna(value)
                    elif not fill_null:
                        value = _selected_value(right, result.data_type)
                        result.data = result.data.fillna(value)
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
                        data_type = result.components[me].data_type
                        fill = _as_type(result.data[f"{me}_r"], data_type)
                        result.data[me] = _as_type(result.data[me].fillna(fill), data_type)
                else:
                    # nvl is single-operand: the primary operand's viral attributes are
                    # copied through unchanged; no rule is executed (issue #906).
                    result.data = result.data.fillna(_as_type(right.data, result.data_type))
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
            # The value of the then operand of the first true condition (issues #1008 and
            # #1251), in the common type of the operands (issue #1314)
            selected = cls._selected_scalar(conditions, thenOps, elseOp)
            result.value = _selected_value(selected, result.data_type)
        else:
            operation_level = list({type(c) for c in conditions if not isinstance(c, Scalar)})
            if operation_level[0] == DataComponent:
                component = cast(DataComponent, result)
                component.data = cls.component_level_evaluation(
                    conditions, thenOps, elseOp, component.data_type
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
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any, data_type: Any
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
            if conditions[i].data is None:
                # No Data Point is left for this condition, so it was not computed (#1314)
                continue
            case = conditions[i].data[conditions[i].data.fillna(False).astype("bool[pyarrow]")]
            case_result = component_assign(case, thenOps[i])
            result = result.reindex(result.index.union(case.index))
            result.loc[case.index] = case_result

        return _as_type(result, data_type)

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
            # A Boolean mask, as in a Data Set if: indexing with the empty object column of an
            # if or a case without Data Points selected no column instead (issue #1314)
            case = case[case[COND_COL].fillna(False).astype("bool[pyarrow]")]
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
        for comp in result.get_measures():
            if comp.data_type is String:
                result.data[comp.name] = to_strings(result.data[comp.name])
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

            # The common type of the operands, as at Component and Data Set level, and not
            # the type of the operand selected by the values of the conditions (issue #1314)
            nullable = any(op.nullable or op.data_type == Null for op in ops)
            return Scalar(
                name="result",
                value=None,
                data_type=_supertype(ops),
                nullable=nullable,
            )

        elif condition_type is DataComponent:
            for condition in conditions:
                if not condition.data_type == Boolean:
                    raise SemanticError("1-1-9-17", op=cls.op, name=condition.name)

            # A scalar operand can be null too, as a nullable input scalar (issue #1314)
            nullable = any(op.nullable or op.data_type == Null for op in ops)

            return DataComponent(
                name=comp_name,
                data=None,
                data_type=_supertype(ops),
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

            measure_name = condition.get_measures_names()[0]
            components = {comp.name: comp for comp in conditions[0].get_identifiers()}
            components[measure_name] = Component(
                name=measure_name,
                data_type=_supertype(ops),
                role=Role.MEASURE,
                nullable=any(branch.nullable or branch.data_type == Null for branch in ops),
            )
            return Dataset(name=dataset_name, components=components, data=None)

        # Each later condition and each Data Set operand has the Identifiers of the first
        # condition, which gives the Data Points, as in an if (issue #1314)
        for item in [*conditions[1:], *ops]:
            if isinstance(item, Dataset) and _identifiers(item) != _identifiers(conditions[0]):
                raise SemanticError("1-1-9-10", op=cls.op, clause=item.name)

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

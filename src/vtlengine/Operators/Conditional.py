from copy import copy
from typing import Any, List, Union

import pandas as pd
from duckdb.duckdb import DuckDBPyRelation

from vtlengine.DataTypes import (
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    Null,
    binary_implicit_promotion,
)
from vtlengine.duckdb.duckdb_utils import (
    duckdb_concat,
    duckdb_fillna,
    duckdb_rename,
    duckdb_select,
    empty_relation,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Role, Scalar
from vtlengine.Operators import Binary, Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


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
    """  # noqa E501

    @classmethod
    def evaluate(cls, condition: Any, true_branch: Any, false_branch: Any) -> Any:
        result = cls.validate(condition, true_branch, false_branch)
        if not isinstance(result, Scalar):
            if isinstance(condition, DataComponent):
                result.data = cls.component_level_evaluation(condition, true_branch, false_branch)
            if isinstance(condition, Dataset):
                result = cls.dataset_level_evaluation(result, condition, true_branch, false_branch)
        return result

    @classmethod
    def component_level_evaluation(
        cls,
        condition: DataComponent,
        true_branch: Union[DataComponent, Scalar],
        false_branch: Union[DataComponent, Scalar],
    ) -> DuckDBPyRelation:
        def get_expr(branch: Any) -> str:
            return (
                f"{repr(branch.value)}"
                if isinstance(branch, Scalar) and branch.value is not None
                else ("NULL" if isinstance(branch, Scalar) else f'"{branch.data.columns[0]}"')
            )

        if condition.data is None:
            return empty_relation()

        cond_col = f'"{condition.data.columns[0]}"'
        true_expr = get_expr(true_branch)
        false_expr = get_expr(false_branch)

        base = duckdb_fillna(condition.data, "TRUE", cond_col)
        if not isinstance(true_branch, Scalar):
            base = duckdb_concat(base, true_branch.data)
        if not isinstance(false_branch, Scalar):
            base = duckdb_concat(base, false_branch.data)

        expr = f'CASE WHEN {cond_col} THEN {true_expr} ELSE {false_expr} END AS "{condition.name}"'
        return base.project(expr)

    @classmethod
    def dataset_level_evaluation(
        cls, result: Any, condition: Any, true_branch: Any, false_branch: Any
    ) -> Any:
        ids = condition.get_identifiers_names()
        cond_col = f"{condition.get_measures_names()[0]}"

        base = duckdb_fillna(condition.data, "FALSE", cond_col)

        if not isinstance(true_branch, Scalar):
            measures = true_branch.get_measures_names()
            true_branch.data = duckdb_rename(
                true_branch.data, {m: f"__{m}@true_branch__" for m in measures}
            )
            base = duckdb_concat(base, true_branch.data, ids)
        if not isinstance(false_branch, Scalar):
            measures = false_branch.get_measures_names()
            false_branch.data = duckdb_rename(
                false_branch.data, {m: f"__{m}@false_branch__" for m in measures}
            )
            base = duckdb_concat(base, false_branch.data, ids)

        exprs = []
        for comp_name in result.get_components_names():
            if comp_name in ids:
                expr = f'"{comp_name}"'
            else:
                true_expr = (
                    f'"__{comp_name}@true_branch__"'
                    if not isinstance(true_branch, Scalar)
                    else repr(true_branch.value)
                    if true_branch.value is not None
                    else "NULL"
                )
                false_expr = (
                    f'"__{comp_name}@false_branch__"'
                    if not isinstance(false_branch, Scalar)
                    else repr(false_branch.value)
                    if false_branch.value is not None
                    else "NULL"
                )
                expr = (
                    f'CASE WHEN {cond_col} THEN {true_expr} ELSE {false_expr} END AS "{comp_name}"'
                )
            exprs.append(expr)

        result.data = base.project(", ".join(exprs))
        return result

    @classmethod
    def validate(  # noqa: C901
        cls, condition: Any, true_branch: Any, false_branch: Any
    ) -> Union[Scalar, DataComponent, Dataset]:
        nullable = False
        left = true_branch
        right = false_branch
        dataset_name = VirtualCounter._new_ds_name()
        if true_branch.__class__ != false_branch.__class__:
            if (isinstance(true_branch, DataComponent) and isinstance(false_branch, Dataset)) or (
                isinstance(true_branch, Dataset) and isinstance(false_branch, DataComponent)
            ):
                raise ValueError(
                    "If then and else operands cannot be dataset and component respectively"
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
            if not isinstance(left, Scalar) or not isinstance(right, Scalar):
                nullable = condition.nullable
            else:
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
        if isinstance(left, Scalar) and isinstance(right, Scalar):
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
            if left.get_components_names() != right.get_components_names():
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
                result.value = right.value
            else:
                result.value = left.value
        else:
            if not isinstance(result, Scalar):
                if isinstance(right, Scalar):
                    result.data = duckdb_fillna(left.data, right.value)
                else:
                    result.data = left.data.fillna(right.data)
                if isinstance(result, Dataset):
                    result.data = duckdb_select(result.data, result.get_components_names())
        return result

    @classmethod
    def validate(cls, left: Any, right: Any) -> Union[Scalar, DataComponent, Dataset]:
        dataset_name = VirtualCounter._new_ds_name()
        comp_name = VirtualCounter._new_dc_name()
        result_components = {}
        if isinstance(left, Scalar):
            if not isinstance(right, Scalar):
                raise ValueError(
                    "Nvl operation at scalar level must have scalar "
                    "types on right (applicable) side"
                )
            cls.type_validation(left.data_type, right.data_type)
            return Scalar(name="result", value=None, data_type=left.data_type)
        if isinstance(left, DataComponent):
            if isinstance(right, Dataset):
                raise ValueError(
                    "Nvl operation at component level cannot have "
                    "dataset type on right (applicable) side"
                )
            cls.type_validation(left.data_type, right.data_type)
            return DataComponent(
                name=comp_name,
                data=pd.Series(dtype=object),
                data_type=left.data_type,
                role=Role.MEASURE,
                nullable=False,
            )
        if isinstance(left, Dataset):
            if isinstance(right, DataComponent):
                raise ValueError(
                    "Nvl operation at dataset level cannot have component "
                    "type on right (applicable) side"
                )
            if isinstance(right, Scalar):
                for component in left.get_measures():
                    cls.type_validation(component.data_type, right.data_type)
            if isinstance(right, Dataset):
                for component in left.get_measures():
                    cls.type_validation(
                        component.data_type, right.components[component.name].data_type
                    )
            result_components = {
                comp_name: copy(comp)
                for comp_name, comp in left.components.items()
                if comp.role != Role.ATTRIBUTE
            }
            for comp in result_components.values():
                comp.nullable = False
        return Dataset(name=dataset_name, components=result_components, data=None)


class Case(Operator):
    @classmethod
    def evaluate(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any
    ) -> Union[Scalar, DataComponent, Dataset]:
        result = cls.validate(conditions, thenOps, elseOp)
        for condition in conditions:
            if isinstance(condition, (Dataset, DataComponent)) and condition.data is not None:
                condition_measure = (
                    condition.get_measures_names()[0]
                    if isinstance(condition, Dataset)
                    else condition.name
                )
                condition.data = duckdb_fillna(condition.data, False, condition_measure)
            elif isinstance(condition, Scalar) and condition.value is None:
                condition.value = False

        if isinstance(result, Scalar):
            result.value = elseOp.value
            for i in range(len(conditions)):
                if conditions[i].value:
                    result.value = thenOps[i].value
                    return result

        relations = []
        for condition in conditions:
            if hasattr(condition, "data") and condition.data is not None:
                relations.append(condition.data)
        for thenOp in thenOps:
            if hasattr(thenOp, "data") and thenOp.data is not None:
                relations.append(thenOp.data)
        if hasattr(elseOp, "data") and elseOp.data is not None:
            relations.append(elseOp.data)

        if relations:
            base = relations[0]
            for rel in relations[1:]:
                base = duckdb_concat(base, rel)
        else:
            base = empty_relation()

        case_expr = "CASE "
        for i, condition in enumerate(conditions):
            if isinstance(condition, Scalar):
                cond = f"{repr(condition.value)}"
            else:
                cond = f'"{condition.get_measures_names()[0] if isinstance(condition, Dataset) else condition.name}"'
            if isinstance(thenOps[i], Scalar):
                then_val = f"{repr(thenOps[i].value)}"
            else:
                then_val = f'"{thenOps[i].name}"'
            case_expr += f"WHEN {cond} THEN {then_val} "

        if isinstance(elseOp, Scalar):
            else_val = f"{repr(elseOp.value)}"
        else:
            else_val = f'"{elseOp.name}"'
        case_expr += f'ELSE {else_val} END AS "{result.name}"'

        result.data = base.project(case_expr)
        return result

    @classmethod
    def validate(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any
    ) -> Union[Scalar, DataComponent, Dataset]:
        dataset_name = VirtualCounter._new_ds_name()
        comp_name = VirtualCounter._new_dc_name()
        if len(set(map(type, conditions))) > 1:
            raise SemanticError("2-1-9-1", op=cls.op)

        ops = thenOps + [elseOp]
        then_else_types = set(map(type, ops))
        condition_type = type(conditions[0])

        if condition_type is Scalar:
            for condition in conditions:
                if condition.data_type != Boolean:
                    raise SemanticError("2-1-9-2", op=cls.op, name=condition.name)
            if list(then_else_types) != [Scalar]:
                raise SemanticError("2-1-9-3", op=cls.op)

            # The output data type is the data type of the last then operation that has a true
            # condition, defaulting to the data type of the else operation if no condition is true
            output_data_type = elseOp.data_type
            for i in range(len(conditions)):
                if conditions[i].value:
                    output_data_type = thenOps[i].data_type

            return Scalar(
                name="result",
                value=None,
                data_type=output_data_type,
            )

        elif condition_type is DataComponent:
            for condition in conditions:
                if not condition.data_type == Boolean:
                    raise SemanticError("2-1-9-4", op=cls.op, name=condition.name)

            nullable = any(
                (thenOp.nullable if isinstance(thenOp, DataComponent) else thenOp.data_type == Null)
                for thenOp in ops
            )
            nullable |= any(condition.nullable for condition in conditions)

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
                raise SemanticError("2-1-9-5", op=cls.op, name=condition.name)

        if Dataset not in then_else_types:
            raise SemanticError("2-1-9-6", op=cls.op)

        components = next(op for op in ops if isinstance(op, Dataset)).components
        comp_names = [comp.name for comp in components.values()]
        for op in ops:
            if isinstance(op, Dataset) and op.get_components_names() != comp_names:
                raise SemanticError("2-1-9-7", op=cls.op)

        return Dataset(name=dataset_name, components=components, data=None)

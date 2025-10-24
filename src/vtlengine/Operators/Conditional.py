from copy import copy
from typing import Any, List, Union

from duckdb import DuckDBPyRelation  # type: ignore[import-untyped]

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
    empty_relation, duckdb_merge,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Role, Scalar, INDEX_COL, RelationProxy
from vtlengine.Operators import Binary, Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.duckdb.to_sql_token import to_sql_literal


COND_COL = "__cond__"


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
                result.data = cls.component_level_evaluation(
                    condition, true_branch, false_branch, result.name
                )
            if isinstance(condition, Dataset):
                result = cls.dataset_level_evaluation(result, condition, true_branch, false_branch)
        return result

    @classmethod
    def component_level_evaluation(
        cls, condition: DataComponent, true_branch: Any, false_branch: Any, result_name: str
    ) -> Any:
        if condition.data is None:
            return empty_relation()

        cond_col = condition.name
        base = duckdb_fillna(condition.data, False, cond_col)

        t_base = base[base[cond_col] == True]
        if isinstance(true_branch, DataComponent):
            t_base[result_name] = true_branch.data[true_branch.data.columns[0]]
        else:
            t_base[result_name] = true_branch.value

        f_base = base[base[cond_col] == False]
        if isinstance(false_branch, DataComponent):
            f_base[result_name] = false_branch.data[false_branch.data.columns[0]]
        else:
            f_base[result_name] = false_branch.value

        return duckdb_concat(t_base, f_base, on=[INDEX_COL]).project(f'"{result_name}"')

    @classmethod
    def dataset_level_evaluation(
        cls, result: Any, condition: Any, true_branch: Any, false_branch: Any
    ) -> Dataset:

        ids = condition.get_identifiers_names()
        cond_measure = condition.get_measures_names()[0]
        measures = result.get_measures_names()
        base = duckdb_fillna(condition.data, False, cond_measure)
        base = duckdb_rename(base, {cond_measure: COND_COL})

        t_mask = base[COND_COL] == True
        if isinstance(true_branch, Dataset) and true_branch.data is not None:
            t_base = duckdb_merge(base[t_mask], true_branch.data, on=ids, how="inner")
        else:
            t_base = base[t_mask]
            for m in measures:
                t_base[m] = true_branch.value

        f_mask = base[COND_COL] == False
        if isinstance(false_branch, Dataset) and false_branch.data is not None:
            f_base = duckdb_merge(base[f_mask], false_branch.data, on=ids, how="inner")
        else:
            f_base = base[f_mask]
            for m in measures:
                f_base[m] = false_branch.value

        result.data = duckdb_concat(t_base, f_base, on=ids).drop(columns=COND_COL)
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
                data=empty_relation(),
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
        def get_condition_measure(op: Union[DataComponent, Dataset, Scalar]) -> str:
            return op.get_measures_names()[0] if isinstance(op, Dataset) else op.name

        result = cls.validate(conditions, thenOps, elseOp)
        for op in conditions:
            if isinstance(op, (Dataset, DataComponent)) and op.data is not None:
                condition_measure = get_condition_measure(op)
                op.data = duckdb_fillna(op.data, False, condition_measure)
            elif isinstance(op, Scalar) and op.value is None:
                op.value = False

        if isinstance(result, Scalar):
            result.value = elseOp.value
            for i in range(len(conditions)):
                if conditions[i].value:
                    result.value = thenOps[i].value
                    break
            return result

        measure = get_condition_measure(conditions[0])
        base = duckdb_rename(conditions[0].data, {measure: f"cond_0.{measure}"})
        for i, cond in enumerate(conditions[1:]):
            measure = get_condition_measure(cond)
            cond_ = duckdb_rename(cond.data, {measure: f"cond_{i + 1}.{measure}"})
            base = duckdb_concat(base, cond_)
        else_condition_query = f"""
        *, NOT({
            " OR ".join(f'"{col}"' for col in base.columns if col.startswith("cond_"))
        }) AS "cond_else"
        """
        base = base.project(else_condition_query)

        operands = thenOps + [elseOp]
        for i, op in enumerate(operands):
            if hasattr(op, "data") and op.data is not None:
                op_data = duckdb_rename(op.data, {col: f"op_{i}.{col}" for col in op.data.columns})
                base = duckdb_concat(base, op_data)

        ids = next((op.get_identifiers_names() for op in operands if isinstance(op, Dataset)), [])
        exprs = [f'"{id_}"' for id_ in ids]
        columns = base.columns
        measures = next(
            (op.get_measures_names() for op in operands if isinstance(op, Dataset)),
            [VirtualCounter._new_dc_name()],
        )
        for col in measures:
            expr = "CASE "
            # CASE op ends whenever the first cond is matched, so in order to match the
            # VTL specification, we need to reverse the order of the operands
            for i, op in enumerate(reversed(operands)):
                i = len(operands) - 1 - i
                cond_col = next(
                    (col_ for col_ in columns if col_.startswith(f"cond_{i}")), "cond_else"
                )
                if isinstance(op, Scalar):
                    value = repr(op.value) if op.value is not None else "NULL"
                else:
                    col_ = col if isinstance(op, Dataset) else op.data.columns[0]
                    value = f'"op_{i}.{col_}"'
                expr += f'WHEN "{cond_col}" THEN {value} '
            expr += f'END AS "{col}"'
            exprs.append(expr)

        result.data = base.project(", ".join(exprs))
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

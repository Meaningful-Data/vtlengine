from copy import copy
from typing import Any, List, Union

from vtlengine.DataTypes import (
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    Null,
    binary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Role, Scalar
from vtlengine.Operators import Binary, Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.ViralPropagation import combined_viral_components, require_rules


class If(Operator):
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
        # if-then-else over two datasets combines their data points per row, so viral
        # attributes carried by both branches require a propagation rule (issue #906).
        require_rules(
            combined_viral_components([b for b in (left, right) if isinstance(b, Dataset)])
        )
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
            return Scalar(name="result", value=None, data_type=left.data_type, nullable=False)
        if isinstance(left, DataComponent):
            if isinstance(right, Dataset):
                raise ValueError(
                    "Nvl operation at component level cannot have "
                    "dataset type on right (applicable) side"
                )
            cls.type_validation(left.data_type, right.data_type)
            return DataComponent(
                name=comp_name,
                data=None,
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
                    raise SemanticError("2-1-9-4", op=cls.op, name=condition.name)

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
                raise SemanticError("2-1-9-5", op=cls.op, name=condition.name)

        if Dataset not in then_else_types:
            raise SemanticError("2-1-9-6", op=cls.op)

        components = next(op for op in ops if isinstance(op, Dataset)).components
        comp_names = [comp.name for comp in components.values()]
        for op in ops:
            if isinstance(op, Dataset) and op.get_components_names() != comp_names:
                raise SemanticError("2-1-9-7", op=cls.op)

        # case over two or more datasets combines their data points, so viral attributes
        # carried by two or more branches require a propagation rule (issue #906).
        require_rules(combined_viral_components([op for op in ops if isinstance(op, Dataset)]))
        return Dataset(name=dataset_name, components=components, data=None)

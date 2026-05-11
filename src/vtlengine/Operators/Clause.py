from copy import copy
from typing import List, Type, Union

from vtlengine.AST import RenameNode
from vtlengine.AST.Grammar.tokens import AGGREGATE, CALC, DROP, KEEP, RENAME, SUBSPACE
from vtlengine.DataTypes import (
    Boolean,
    ScalarType,
    String,
    check_unary_implicit_promotion,
    unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar
from vtlengine.Operators import Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


class Calc(Operator):
    op = CALC

    @classmethod
    def validate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset) -> Dataset:
        result_components = {name: copy(comp) for name, comp in dataset.components.items()}
        dataset_name = VirtualCounter._new_ds_name()
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)

        for operand in operands:
            if operand.name in result_dataset.components:
                if result_dataset.components[operand.name].role == Role.IDENTIFIER:
                    raise SemanticError("1-1-6-13", op=cls.op, comp_name=operand.name)
                # Override component with same name
                # TODO: Check this for version 2.1
                result_dataset.delete_component(operand.name)

            if isinstance(operand, Scalar):
                result_dataset.add_component(
                    Component(
                        name=operand.name,
                        data_type=operand.data_type,
                        role=Role.MEASURE,
                        nullable=True,
                    )
                )
            else:
                result_dataset.add_component(
                    Component(
                        name=operand.name,
                        data_type=operand.data_type,
                        role=operand.role,
                        nullable=operand.nullable,
                    )
                )
        return result_dataset


class Aggregate(Operator):
    op = AGGREGATE

    @classmethod
    def validate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        result_dataset = Dataset(name=dataset_name, components=dataset.components, data=None)

        for operand in operands:
            if operand.name in dataset.get_identifiers_names() or (
                isinstance(operand, DataComponent) and operand.role == Role.IDENTIFIER
            ):
                raise SemanticError("1-1-6-13", op=cls.op, comp_name=operand.name)

            elif operand.name in dataset.components:
                # Override component with same name
                dataset.delete_component(operand.name)

            if isinstance(operand, Scalar):
                result_dataset.add_component(
                    Component(
                        name=operand.name,
                        data_type=operand.data_type,
                        role=Role.MEASURE,
                        nullable=True,
                    )
                )
            else:
                result_dataset.add_component(
                    Component(
                        name=operand.name,
                        data_type=operand.data_type,
                        role=operand.role,
                        nullable=operand.nullable,
                    )
                )
        return result_dataset


class Filter(Operator):
    @classmethod
    def validate(cls, condition: DataComponent, dataset: Dataset) -> Dataset:
        if condition.data_type != Boolean:
            raise ValueError(f"Filter condition must be of type {Boolean}")
        dataset_name = VirtualCounter._new_ds_name()
        return Dataset(name=dataset_name, components=dataset.components, data=None)


class Keep(Operator):
    op = KEEP

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        for operand in operands:
            if operand not in dataset.get_components_names():
                raise SemanticError(
                    "1-1-1-10", op=cls.op, comp_name=operand, dataset_name=dataset_name
                )
            if dataset.get_component(operand).role == Role.IDENTIFIER:
                raise SemanticError("1-1-6-2", op=cls.op, name=operand, dataset=dataset_name)
        result_components = {
            name: comp
            for name, comp in dataset.components.items()
            if comp.name in operands or comp.role == Role.IDENTIFIER
        }
        return Dataset(name=dataset_name, components=result_components, data=None)


class Drop(Operator):
    op = DROP

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        for operand in operands:
            if operand not in dataset.components:
                raise SemanticError("1-1-1-10", comp_name=operand, dataset_name=dataset_name)
            if dataset.get_component(operand).role == Role.IDENTIFIER:
                raise SemanticError("1-1-6-2", op=cls.op, name=operand, dataset=dataset_name)
        if len(dataset.components) == len(operands):
            raise SemanticError("1-1-6-12", op=cls.op)
        result_components = {
            name: comp for name, comp in dataset.components.items() if comp.name not in operands
        }
        return Dataset(name=dataset_name, components=result_components, data=None)


class Rename(Operator):
    op = RENAME

    @classmethod
    def validate(cls, operands: List[RenameNode], dataset: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        from_names = [operand.old_name for operand in operands]
        if len(from_names) != len(set(from_names)):
            duplicates = set([name for name in from_names if from_names.count(name) > 1])
            raise SemanticError("1-1-6-9", op=cls.op, from_components=duplicates)

        to_names = [operand.new_name for operand in operands]
        if len(to_names) != len(set(to_names)):  # If duplicates
            duplicates = set([name for name in to_names if to_names.count(name) > 1])
            raise SemanticError("1-2-1", alias=duplicates)

        from_names_set = set(from_names)
        for operand in operands:
            if operand.old_name not in dataset.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=cls.op,
                    comp_name=operand.old_name,
                    dataset_name=dataset_name,
                )
            if operand.new_name in dataset.components and operand.new_name not in from_names_set:
                raise SemanticError(
                    "1-1-6-8",
                    op=cls.op,
                    comp_name=operand.new_name,
                    dataset_name=dataset_name,
                )

        rename_map = {op.old_name: op.new_name for op in operands}
        result_components = {}
        for comp in dataset.components.values():
            if comp.name in rename_map:
                new_name = rename_map[comp.name]
                result_components[new_name] = Component(
                    name=new_name,
                    data_type=comp.data_type,
                    role=comp.role,
                    nullable=comp.nullable,
                )
            else:
                result_components[comp.name] = comp
        return Dataset(name=dataset_name, components=result_components, data=None)


class Pivot(Operator):
    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        raise NotImplementedError


class Unpivot(Operator):
    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(operands) != 2:
            raise ValueError("Unpivot clause requires two operands")
        identifier, measure = operands

        if len(dataset.get_identifiers()) < 1:
            raise SemanticError("1-2-10", op=cls.op)
        if identifier in dataset.components:
            raise SemanticError("1-1-6-2", op=cls.op, name=identifier, dataset=dataset_name)

        result_components = {comp.name: comp for comp in dataset.get_identifiers()}
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        # noinspection PyTypeChecker
        result_dataset.add_component(
            Component(name=identifier, data_type=String, role=Role.IDENTIFIER, nullable=False)
        )
        base_type = None
        final_type: Type[ScalarType] = String
        for comp in dataset.get_measures():
            if base_type is None:
                base_type = comp.data_type
            else:
                if check_unary_implicit_promotion(base_type, comp.data_type) is None:
                    raise ValueError("All measures must have the same data type on unpivot clause")
            final_type = unary_implicit_promotion(base_type, comp.data_type)

        result_dataset.add_component(
            Component(name=measure, data_type=final_type, role=Role.MEASURE, nullable=True)
        )
        return result_dataset


class Sub(Operator):
    op = SUBSPACE

    @classmethod
    def validate(cls, operands: List[DataComponent], dataset: Dataset) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(dataset.get_identifiers()) < 1:
            raise SemanticError("1-2-10", op=cls.op)
        for operand in operands:
            if operand.name not in dataset.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=cls.op,
                    comp_name=operand.name,
                    dataset_name=dataset_name,
                )
            if operand.role != Role.IDENTIFIER:
                raise SemanticError(
                    "1-1-6-10",
                    op=cls.op,
                    operand=operand.name,
                    dataset_name=dataset_name,
                )
            if isinstance(operand, Scalar):
                raise SemanticError("1-1-6-5", op=cls.op, name=operand.name)

        result_components = {
            name: comp
            for name, comp in dataset.components.items()
            if comp.name not in [operand.name for operand in operands]
        }
        return Dataset(name=dataset_name, components=result_components, data=None)

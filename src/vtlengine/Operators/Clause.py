from copy import copy
from typing import List, Type, Union

import pandas as pd

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


class Calc(Operator):
    op = CALC

    @classmethod
    def validate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset) -> Dataset:
        result_components = {name: copy(comp) for name, comp in dataset.components.items()}
        result_dataset = Dataset(name=dataset.name, components=result_components, data=None)

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

    @classmethod
    def evaluate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data.copy() if dataset.data is not None else pd.DataFrame()
        for operand in operands:
            if isinstance(operand, Scalar):
                result_dataset.data[operand.name] = operand.value
            else:
                result_dataset.data[operand.name] = operand.data
        return result_dataset


class Aggregate(Operator):
    op = AGGREGATE

    @classmethod
    def validate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset) -> Dataset:
        result_dataset = Dataset(name=dataset.name, components=dataset.components, data=None)

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

    @classmethod
    def evaluate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = copy(dataset.data) if dataset.data is not None else pd.DataFrame()
        for operand in operands:
            if isinstance(operand, Scalar):
                result_dataset.data[operand.name] = operand.value
            else:
                if operand.data is not None and len(operand.data) > 0:
                    result_dataset.data[operand.name] = operand.data
                else:
                    result_dataset.data[operand.name] = None
        return result_dataset


class Filter(Operator):
    @classmethod
    def validate(cls, condition: DataComponent, dataset: Dataset) -> Dataset:
        if condition.data_type != Boolean:
            raise ValueError(f"Filter condition must be of type {Boolean}")
        return Dataset(name=dataset.name, components=dataset.components, data=None)

    @classmethod
    def evaluate(cls, condition: DataComponent, dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(condition, dataset)
        result_dataset.data = dataset.data.copy() if dataset.data is not None else pd.DataFrame()
        if condition.data is not None and len(condition.data) > 0 and dataset.data is not None:
            true_indexes = condition.data[condition.data == True].index
            result_dataset.data = dataset.data.iloc[true_indexes].reset_index(drop=True)
        return result_dataset


class Keep(Operator):
    op = KEEP

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        for operand in operands:
            if operand not in dataset.get_components_names():
                raise SemanticError(
                    "1-1-1-10", op=cls.op, comp_name=operand, dataset_name=dataset.name
                )
            if dataset.get_component(operand).role == Role.IDENTIFIER:
                raise SemanticError("1-1-6-2", op=cls.op, name=operand, dataset=dataset.name)
        result_components = {
            name: comp
            for name, comp in dataset.components.items()
            if comp.name in operands or comp.role == Role.IDENTIFIER
        }
        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        if len(operands) == 0:
            raise ValueError("Keep clause requires at least one operand")
        if dataset is None and sum(isinstance(operand, Dataset) for operand in operands) != 1:
            raise ValueError("Keep clause requires at most one dataset operand")
        result_dataset = cls.validate(operands, dataset)
        if dataset.data is not None:
            result_dataset.data = dataset.data[dataset.get_identifiers_names() + operands]
        return result_dataset


class Drop(Operator):
    op = DROP

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        for operand in operands:
            if operand not in dataset.components:
                raise SemanticError("1-1-1-10", comp_name=operand, dataset_name=dataset.name)
            if dataset.get_component(operand).role == Role.IDENTIFIER:
                raise SemanticError("1-1-6-2", op=cls.op, name=operand, dataset=dataset.name)
        if len(dataset.components) == len(operands):
            raise SemanticError("1-1-6-12", op=cls.op)
        result_components = {
            name: comp for name, comp in dataset.components.items() if comp.name not in operands
        }
        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(operands, dataset)
        if dataset.data is not None:
            result_dataset.data = dataset.data.drop(columns=operands, axis=1)
        return result_dataset


class Rename(Operator):
    op = RENAME

    @classmethod
    def validate(cls, operands: List[RenameNode], dataset: Dataset) -> Dataset:
        from_names = [operand.old_name for operand in operands]
        if len(from_names) != len(set(from_names)):
            duplicates = set([name for name in from_names if from_names.count(name) > 1])
            raise SemanticError("1-1-6-9", op=cls.op, from_components=duplicates)

        to_names = [operand.new_name for operand in operands]
        if len(to_names) != len(set(to_names)):  # If duplicates
            duplicates = set([name for name in to_names if to_names.count(name) > 1])
            raise SemanticError("1-3-1", alias=duplicates)

        for operand in operands:
            if operand.old_name not in dataset.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=cls.op,
                    comp_name=operand.old_name,
                    dataset_name=dataset.name,
                )
            if operand.new_name in dataset.components:
                raise SemanticError(
                    "1-1-6-8",
                    op=cls.op,
                    comp_name=operand.new_name,
                    dataset_name=dataset.name,
                )

        result_components = {comp.name: comp for comp in dataset.components.values()}
        for operand in operands:
            result_components[operand.new_name] = Component(
                name=operand.new_name,
                data_type=result_components[operand.old_name].data_type,
                role=result_components[operand.old_name].role,
                nullable=result_components[operand.old_name].nullable,
            )
            del result_components[operand.old_name]

        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[RenameNode], dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(operands, dataset)
        if dataset.data is not None:
            result_dataset.data = dataset.data.rename(
                columns={operand.old_name: operand.new_name for operand in operands}
            )
        return result_dataset


class Pivot(Operator):
    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        raise NotImplementedError

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        raise NotImplementedError


class Unpivot(Operator):
    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        if len(operands) != 2:
            raise ValueError("Unpivot clause requires two operands")
        identifier, measure = operands

        if len(dataset.get_identifiers()) < 1:
            raise SemanticError("1-3-27", op=cls.op)
        if identifier in dataset.components:
            raise SemanticError("1-1-6-2", op=cls.op, name=identifier, dataset=dataset.name)

        result_components = {comp.name: comp for comp in dataset.get_identifiers()}
        result_dataset = Dataset(name=dataset.name, components=result_components, data=None)
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

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(operands, dataset)
        if dataset.data is not None:
            result_dataset.data = dataset.data.melt(
                id_vars=dataset.get_identifiers_names(),
                value_vars=dataset.get_measures_names(),
                var_name=operands[0],
                value_name="NEW_COLUMN",
            )
            result_dataset.data.rename(columns={"NEW_COLUMN": operands[1]}, inplace=True)
            result_dataset.data = result_dataset.data.dropna().reset_index(drop=True)
        return result_dataset


class Sub(Operator):
    op = SUBSPACE

    @classmethod
    def validate(cls, operands: List[DataComponent], dataset: Dataset) -> Dataset:
        if len(dataset.get_identifiers()) < 1:
            raise SemanticError("1-3-27", op=cls.op)
        for operand in operands:
            if operand.name not in dataset.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=cls.op,
                    comp_name=operand.name,
                    dataset_name=dataset.name,
                )
            if operand.role != Role.IDENTIFIER:
                raise SemanticError(
                    "1-1-6-10",
                    op=cls.op,
                    operand=operand.name,
                    dataset_name=dataset.name,
                )
            if isinstance(operand, Scalar):
                raise SemanticError("1-1-6-5", op=cls.op, name=operand.name)

        result_components = {
            name: comp
            for name, comp in dataset.components.items()
            if comp.name not in [operand.name for operand in operands]
        }
        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[DataComponent], dataset: Dataset) -> Dataset:
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = copy(dataset.data) if dataset.data is not None else pd.DataFrame()
        operand_names = [operand.name for operand in operands]
        if dataset.data is not None and len(dataset.data) > 0:
            # Filter the Dataframe
            # by intersecting the indexes of the Data Component with True values
            true_indexes = set()
            is_first = True
            for operand in operands:
                if operand.data is not None:
                    if is_first:
                        true_indexes = set(operand.data[operand.data == True].index)
                        is_first = False
                    else:
                        true_indexes.intersection_update(
                            set(operand.data[operand.data == True].index)
                        )
            result_dataset.data = result_dataset.data.iloc[list(true_indexes)]
        result_dataset.data = result_dataset.data.drop(columns=operand_names, axis=1)
        result_dataset.data = result_dataset.data.reset_index(drop=True)
        return result_dataset

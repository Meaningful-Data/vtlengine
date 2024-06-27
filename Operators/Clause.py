from typing import List, Union

from AST import RenameNode
from DataTypes import Boolean, String
from Model import Component, DataComponent, Dataset, Role, Scalar


class Calc:

    @classmethod
    def validate(cls, operands: List[Union[DataComponent, Scalar]], dataset: Dataset):

        result_dataset = Dataset(name=dataset.name, components=dataset.components, data=None)

        for operand in operands:
            if operand.name in dataset.components:
                # Override component with same name
                dataset.delete_component(operand.name)

            if isinstance(operand, Scalar):
                result_dataset.add_component(Component(
                    name=operand.name,
                    data_type=operand.data_type,
                    role=Role.MEASURE,
                    nullable=True
                ))
            else:
                result_dataset.add_component(Component(
                    name=operand.name,
                    data_type=operand.data_type,
                    role=operand.role,
                    nullable=operand.nullable
                ))
        return result_dataset

    @classmethod
    def evaluate(cls, operands: List[DataComponent], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data.copy()
        for operand in operands:
            if isinstance(operand, Scalar):
                result_dataset.data[operand.name] = operand.value
            else:
                result_dataset.data[operand.name] = operand.data
        return result_dataset


class Filter:

    @classmethod
    def validate(cls, condition: DataComponent, dataset: Dataset):
        if condition.data_type == Boolean:
            raise ValueError(f"Filter condition must be of type {Boolean}")
        return Dataset(name=dataset.name, components=dataset.components, data=None)

    @classmethod
    def evaluate(cls, condition: DataComponent, dataset: Dataset):
        result_dataset = cls.validate(condition, dataset)
        result_dataset.data = dataset.data[condition.data].reset_index(drop=True)
        return result_dataset


class Keep:

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset):
        for operand in operands:
            if operand not in dataset.components:
                raise Exception(f"Component {operand} not found in dataset {dataset.name}")
            if dataset.get_component(operand).role == Role.IDENTIFIER:
                raise Exception(f"Component {operand} in dataset {dataset.name} is an "
                                f"{Role.IDENTIFIER} and cannot be used in keep clause")

        result_components = {name: comp for name, comp in dataset.components.items()
                             if comp.name in operands or comp.role == Role.IDENTIFIER}

        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data[dataset.get_identifiers_names() + operands]
        return result_dataset


class Drop:

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset):
        for operand in operands:
            if operand not in dataset.components:
                raise Exception(f"Component {operand} not found in dataset {dataset.name}")
            if dataset.get_component(operand).role == Role.IDENTIFIER:
                raise Exception(f"Component {operand} in dataset {dataset.name} is an "
                                f"{Role.IDENTIFIER} and cannot be used in drop clause")

        result_components = {name: comp for name, comp in dataset.components.items()
                             if comp.name not in operands}

        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data.drop(columns=operands, axis=1)
        return result_dataset


class Rename:

    @classmethod
    def validate(cls, operands: List[RenameNode], dataset: Dataset):
        for operand in operands:
            if operand.old_name not in dataset.components:
                raise Exception(f"Component {operand.old_name} not found in dataset {dataset.name}")
            if operand.new_name in dataset.components:
                raise Exception(
                    f"Component {operand.new_name} already exists in dataset {dataset.name}")

        result_components = {comp.name: comp for comp in dataset.components.values()}
        for operand in operands:
            result_components[operand.new_name] = result_components.pop(operand.old_name)
            result_components[operand.new_name].name = operand.new_name

        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[RenameNode], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data.rename(columns={operand.old_name: operand.new_name
                                                           for operand in operands})
        return result_dataset


class Pivot:

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset):
        raise NotImplementedError

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset):
        raise NotImplementedError


class Unpivot:

    @classmethod
    def validate(cls, operands: List[str], dataset: Dataset):
        if len(operands) != 2:
            raise ValueError("Unpivot clause requires two operands")
        identifier, measure = operands
        if identifier in dataset.components:
            raise ValueError(f"Component {identifier} already exists in dataset {dataset.name}")
        if measure in dataset.components:
            raise ValueError(f"Component {measure} already exists in dataset {dataset.name}")

        result_components = {comp.name: comp for comp in dataset.get_identifiers()}
        result_dataset = Dataset(name=dataset.name, components=result_components, data=None)
        # noinspection PyTypeChecker
        result_dataset.add_component(Component(name=identifier, data_type=String,
                                               role=Role.IDENTIFIER, nullable=False))
        # TODO: Add type promotion
        base_type = None
        for comp in dataset.get_measures():
            if base_type is None:
                base_type = comp.data_type
            else:
                if comp.data_type != base_type:
                    raise ValueError("All measures must have the same data type")
        result_dataset.add_component(Component(name=measure, data_type=base_type,
                                               role=Role.MEASURE, nullable=True))
        return result_dataset

    @classmethod
    def evaluate(cls, operands: List[str], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data.melt(id_vars=dataset.get_identifiers_names(),
                                                value_vars=dataset.get_measures_names(),
                                                var_name=operands[0], value_name=operands[1])
        result_dataset.data = result_dataset.data.reset_index(drop=True)
        return result_dataset


class Sub:

    @classmethod
    def validate(cls, operands: List[DataComponent], dataset: Dataset):
        for operand in operands:
            if operand.name not in dataset.components:
                raise Exception(f"Component {operand.name} not found in dataset {dataset.name}")
            if operand.role != Role.IDENTIFIER:
                raise Exception(f"Component {operand.name} in dataset {dataset.name} is not an "
                                f"{Role.IDENTIFIER}")

        result_components = {name: comp for name, comp in dataset.components.items()
                             if comp.name not in [operand.name for operand in operands]}
        return Dataset(name=dataset.name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, operands: List[DataComponent], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        result_dataset.data = dataset.data.copy()
        for operand in operands:
            result_dataset.data = result_dataset.data[operand.data]
            result_dataset.data = result_dataset.data.drop(columns=[operand.name], axis=1)
            result_dataset.data = result_dataset.data.reset_index(drop=True)
        return result_dataset

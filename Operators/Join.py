import os
from typing import List

from AST import BinOp

if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Dataset, Component, Role
from Operators import Operator


class Join(Operator):
    how = None

    @classmethod
    def get_components_union(cls, datasets: List[Dataset]) -> List[Component]:
        common = []
        common.extend(comp for dataset in datasets for comp in dataset.components.values() if comp not in common)
        return common

    @classmethod
    def get_components_intersection(cls, *operands: List[Component]):
        # if len(operands) < 2:
        #     return operands[0]
        # return list(set.intersection(*map(set, operands)))
        element_count = {}
        for operand in operands:
            operand_set = set(operand)
            for element in operand_set:
                element_count[element] = element_count.get(element, 0) + 1
        result = []
        for element, count in element_count.items():
            if count >= 2:
                result.append(element)
        return result

    @classmethod
    def merge_components(cls, dataset: Dataset, operands: List[Dataset]) -> Dataset:
        columns = dataset.data.columns.tolist()
        common = cls.get_components_union(operands)
        for component in common:
            if component.name in columns:
                dataset.components.update({component.name: component.copy()})
            else:
                for op_name in [op.name for op in operands]:
                    if op_name + '#' + component.name in columns:
                        dataset.components.update({op_name + '#' + component.name: component.copy()})
                        dataset.components[op_name + '#' + component.name].name = op_name + '#' + component.name
        dataset.components = {column: dataset.components[column] for column in columns}
        return dataset

    @classmethod
    def evaluate(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        result = cls.execute(operands, using)
        if result.get_components_names() != result.data.columns.tolist():
            raise Exception(f"Invalid components on result dataset")
        return result

    @classmethod
    def execute(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        result = cls.validate(operands, using)
        if result.name != "result":
            result.name = result
            return result
        if using:
            identifiers = using
        else:
            identifiers = result.get_identifiers_names()
        common = cls.get_components_intersection(*[op.get_measures_names() for op in operands])
        for op in operands:
            result.data = pd.merge(result.data, op.data, how=cls.how, on=identifiers)
            result.data = result.data.rename(
                columns={column: op.name + '#' + column for column in result.data.columns.tolist() if column in common})
        cls.merge_components(result, operands)
        result.data.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    def validate(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        if len(operands) < 1 or sum([isinstance(op, Dataset) for op in operands]) < 1:
            raise Exception("Join operator requires at least 1 dataset")
        if len(operands) == 1 and isinstance(operands[0], Dataset):
            return operands[0]
        cls.identifiers_validation(operands, using)

        components = {}
        for op in operands:
            if isinstance(op, Dataset):
                components.update({id: op.components[id] for id in op.get_identifiers_names()})
        data = next(op.data[components.keys()] for op in operands if isinstance(op, Dataset))
        return Dataset(name="result", components=components, data=pd.DataFrame(data=data, columns=components.keys()))

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        for op in operands:
            if op.get_identifiers() != operands[0].get_identifiers():
                raise Exception("All datasets must have the same identifiers")


class InnerJoin(Join):
    how = 'inner'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        is_superset = all(
            [set(operands[i].get_identifiers_names()).issubset(set(op2.get_identifiers_names())) for i, op1 in
             enumerate(operands) for op2 in operands])
        if not is_superset:
            raise Exception("At least one dataset must have all identifiers of the others")

        if using is not None:
            is_superset = all([set(using).issubset(set(op.get_identifiers_names())) for op in operands])
            if not is_superset:
                raise Exception("All identifiers from using must be on datasets identifiers")


class LeftJoin(Join):
    how = 'left'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        for op in operands:
            if op.get_identifiers() != operands[0].get_identifiers():
                raise Exception("All datasets must have the same identifiers")
        if using is not None:
            for identifier in using:
                if identifier not in operands[0].get_identifiers_names():
                    raise Exception(f"Identifier {identifier} from using not found on datasets identifiers")


class RightJoin(Join):
    how = 'right'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        for op in operands:
            if op.get_identifiers() != operands[0].get_identifiers():
                raise Exception("All datasets must have the same identifiers")
        if using is not None:
            for identifier in using:
                if identifier not in operands[0].get_identifiers_names():
                    raise Exception(f"Identifier {identifier} from using not found on datasets identifiers")


class FullJoin(Join):
    how = 'outer'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using=None) -> None:
        if using is not None:
            raise Exception("Cross join does not accept using clause")
        for op in operands:
            if op.get_identifiers() != operands[0].get_identifiers():
                raise Exception("All datasets must have the same identifiers")


class CrossJoin(Join):
    how = 'cross'

    @classmethod
    def execute(cls, operands: List[Dataset], using=None) -> Dataset:
        result = cls.validate(operands, using)
        if result.name != "result":
            result.name = result
            return result
        common = cls.get_components_intersection(*[op.get_components_names() for op in operands])
        for op in operands:
            if op is operands[0]:
                result.data = op.data
            else:
                result.data = pd.merge(result.data, op.data, how=cls.how)
            result.data = result.data.rename(
                columns={column: op.name + '#' + column for column in result.data.columns.tolist() if column in common})
        cls.merge_components(result, operands)
        result.data.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using=None) -> None:
        if using is not None:
            raise Exception("Cross join does not accept using clause")


class Apply(Operator):

    @classmethod
    def evaluate(cls, dataset: Dataset, expression, op_map: dict):
        for child in expression:
            dataset = cls.execute(dataset, op_map[child.op], child.left.value, child.right.value)
        return dataset

    @classmethod
    def execute(cls, dataset: Dataset, op, left: str, right: str) -> Dataset:
        left_dataset = cls.create_dataset("left", left, dataset)
        right_dataset = cls.create_dataset("right", right, dataset)
        left_dataset, right_dataset = cls.get_common_components(left_dataset, right_dataset)
        return op.evaluate(left_dataset, right_dataset)

    @classmethod
    def validate(cls, dataset: Dataset, child, op_map: dict) -> None:
        if not isinstance(child, BinOp):
            raise Exception(f"Invalid expression {child} on apply operator. Only BinOp are accepted")
        if child.op not in op_map:
            raise Exception(f"Operator {child.op} not implemented")
        left_components = [comp.name[len(child.left.value) + 1] for comp in dataset.components.values() if
                           comp.name.startswith(child.left.value)]
        right_components = [comp.name[len(child.right.value) + 1] for comp in dataset.components.values() if
                            comp.name.startswith(child.right.value)]
        if len(set(left_components) & set(right_components)) == 0:
            raise Exception(f"{child.left.value} and {child.right.value} has not any match on dataset components")

    @classmethod
    def create_dataset(cls, name: str, prefix: str, dataset: Dataset) -> Dataset:
        prefix += '#'
        components = {component.name: component for component in dataset.components.values() if
                      component.name.startswith(prefix) or component.role is Role.IDENTIFIER}
        data = dataset.data[list(components.keys())]

        for component in components.values():
            component.name = component.name[len(prefix):] if (
                    component.name.startswith(prefix) and component.role is not Role.IDENTIFIER) else component.name
        components = {component.name: component for component in components.values()}
        data.rename(columns={column: column[len(prefix):] for column in data.columns if column.startswith(prefix)},
                    inplace=True)
        return Dataset(name=name, components=components, data=data)

    @classmethod
    def get_common_components(cls, left: Dataset, right: Dataset) -> (Dataset, Dataset):
        common = set(left.get_components_names()) & set(right.get_components_names())
        left.components = {comp.name: comp for comp in left.components.values() if comp.name in common}
        right.components = {comp.name: comp for comp in right.components.values() if
                            comp.name in common}
        left.data = left.data[list(common)]
        right.data = right.data[list(common)]
        return left, right

import os
from typing import List

if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Dataset, Component
from Operators import Operator


class Join(Operator):
    how = None

    @classmethod
    def get_components_union(cls, datasets: List[Dataset]) -> List[Component]:
        common = []
        common.extend(comp for dataset in datasets for comp in dataset.components.values() if comp not in common)
        return common

    def get_components_intersection(*operands: List[Component]):
        if len(operands) < 2:
            return operands[0]
        return list(set.intersection(*map(set, operands)))

    @classmethod
    def merge_components(cls, dataset: Dataset, operands: List[Dataset]) -> Dataset:
        columns = dataset.data.columns.tolist()
        for comp in cls.get_components_union(operands):
            if comp.name in columns:
                dataset.components.update({comp.name: comp.copy()})
            else:
                for op_name in [op.name for op in operands]:
                    if op_name + '#' + comp.name in columns:
                        dataset.components.update({op_name + '#' + comp.name: comp.copy()})
                        dataset.components[op_name + '#' + comp.name].name = op_name + '#' + comp.name
        # TODO: check if is needed to order the components
        dataset.components = {col: dataset.components[col] for col in columns}
        return dataset

    @classmethod
    def evaluate(cls, operands: List[Dataset], using: str) -> Dataset:
        result = cls.execute(operands)
        if result.get_components_names() != result.data.columns.tolist():
            raise Exception(f"Invalid components on result dataset")
        return result

    @classmethod
    def execute(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        if result.name != "result":
            result.name = result
            return result
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
    def validate(cls, operands: List[Dataset]) -> Dataset:
        if len(operands) < 1:
            raise Exception("Join operator requires at least 1 operand")
        # TODO: check if only one operand is allowed on join operations
        if len(operands) == 1 and isinstance(operands[0], Dataset):
            return operands[0]
        if sum([isinstance(op, Dataset) for op in operands]) < 1:
            raise Exception("Join operator requires at least 1 dataset")
        components = {}
        for op in operands:
            if isinstance(op, Dataset):
                components.update({id: op.components[id] for id in op.get_identifiers_names()})
        data = next(op.data[components.keys()] for op in operands if isinstance(op, Dataset))
        return Dataset(name="result", components=components, data=pd.DataFrame(data=data, columns=components.keys()))


class InnerJoin(Join):
    how = 'inner'


class LeftJoin(Join):
    how = 'left'


class RightJoin(Join):
    how = 'right'


class FullJoin(Join):
    how = 'outer'


class CrossJoin(Join):
    how = 'cross'

    @classmethod
    def execute(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
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

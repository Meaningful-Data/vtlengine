from typing import List, Optional

from Model import Dataset, Role
from Operators import Operator


class Join(Operator):

    @classmethod
    def validate(cls, operands: List[Dataset], using: Optional[str] = None):
        if len(operands) < 2:
            raise ValueError("Join operation requires at least two datasets")

        dataset_names = [dataset.name for dataset in operands]
        if len(set(dataset_names)) < len(dataset_names):
            raise ValueError(f"Dataset names must be unique, found {dataset_names}")

        # TODO: Use constants for name of calculated datasets (or validate in the Analyzer)
        if 'result' in dataset_names:
            raise Exception("Alias is mandatory for calculated datasets")

        if using is not None:
            for dataset in operands:
                for used_component in using:
                    if used_component not in dataset.components:
                        raise ValueError(f"Component {used_component} "
                                         f"not found in dataset {dataset.name}")
                    if dataset.components[used_component].role != Role.IDENTIFIER:
                        raise ValueError(f"Component {used_component} in dataset {dataset.name} "
                                         f"must be an identifier")

    @classmethod
    def evaluate(cls, operands: List[Dataset], using: Optional[str] = None) -> Dataset:
        cls.validate(operands, using)


class InnerJoin(Join):
    pass


class LeftJoin(Join):
    pass


class FullJoin(Join):
    pass


class CrossJoin(Join):
    pass

from typing import List

from Model import DataComponent, Dataset, Component, Role


class Calc:

    @classmethod
    def evaluate(cls, operands: List[DataComponent], dataset: Dataset):
        for operand in operands:
            dataset.data[operand.name] = operand.data
            dataset.add_component(Component(
                name=operand.name,
                data_type=operand.data_type,
                role=operand.role,
                nullable=False if operand.role == Role.IDENTIFIER else True
            ))
        return dataset



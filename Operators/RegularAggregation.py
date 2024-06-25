from typing import List

from Model import Component, DataComponent, Dataset


class Calc:

    @classmethod
    def validate(cls, operands: List[DataComponent], dataset: Dataset):
        for operand in operands:
            dataset.add_component(Component(
                name=operand.name,
                data_type=operand.data_type,
                role=operand.role,
                nullable=operand.nullable
            ))
        return dataset

    @classmethod
    def evaluate(cls, operands: List[DataComponent], dataset: Dataset):
        result_dataset = cls.validate(operands, dataset)
        for operand in operands:
            result_dataset.data[operand.name] = operand.data
        return result_dataset

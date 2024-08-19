from typing import Union

from Model import DataComponent, Dataset
from Operators import Binary

ALL_MODEL_TYPES = Union[DataComponent, Dataset]


class Assignment(Binary):

    @classmethod
    def validate(cls, left: str, right: ALL_MODEL_TYPES) -> ALL_MODEL_TYPES:
        right.name = left
        return right

    @classmethod
    def evaluate(
        cls, left_operand: str, right_operand: ALL_MODEL_TYPES
    ) -> ALL_MODEL_TYPES:
        return cls.validate(left_operand, right_operand)

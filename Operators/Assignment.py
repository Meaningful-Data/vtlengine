from typing import Union

from Exceptions import SemanticError
from Model import DataComponent, Dataset
from Operators import Binary

ALL_MODEL_TYPES = Union[DataComponent, Dataset]


class Assignment(Binary):

    @classmethod
    def validate(cls, left_operand: str, right_operand: ALL_MODEL_TYPES) -> ALL_MODEL_TYPES:
        if isinstance(right_operand, DataComponent) and right_operand.role == "IDENTIFIER":
            raise SemanticError("1-1-6-13", op=cls.op, comp_name=right_operand.name)
        right_operand.name = left_operand
        return right_operand

    @classmethod
    def evaluate(cls, left_operand: str, right_operand: ALL_MODEL_TYPES) -> ALL_MODEL_TYPES:
        return cls.validate(left_operand, right_operand)

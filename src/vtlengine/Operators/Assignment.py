from typing import Any, Union

from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset
from vtlengine.Operators import Binary

ALL_MODEL_TYPES = Union[DataComponent, Dataset]


class Assignment(Binary):
    @classmethod
    def validate(cls, left_operand: Any, right_operand: Any) -> ALL_MODEL_TYPES:
        if (
            isinstance(right_operand, DataComponent)
            and right_operand.role.__str__() == "IDENTIFIER"
        ):
            raise SemanticError("1-1-6-13", op=cls.op, comp_name=right_operand.name)
        right_operand.name = left_operand
        return right_operand

    @classmethod
    def evaluate(cls, left_operand: Any, right_operand: Any) -> ALL_MODEL_TYPES:
        return cls.validate(left_operand, right_operand)

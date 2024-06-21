from dataclasses import dataclass
from typing import Union

from Model import Dataset, DataComponent


@dataclass
class Assignment:

    @classmethod
    def evaluate(cls, left_operand: str, right_operand: Union[Dataset, DataComponent]) -> Dataset:
        right_operand.name = left_operand
        return right_operand
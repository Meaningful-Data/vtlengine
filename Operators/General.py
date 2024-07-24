from DataTypes import CAST_MAPPING
from Model import Dataset, Role, Component
from Operators import Binary

class Membership(Binary):

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: str):
        if right_operand not in left_operand.components:
            raise ValueError(f"Component {right_operand} not found in dataset {left_operand.name}")

        component = left_operand.components[right_operand]
        if component.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            right_operand = f'{CAST_MAPPING[component.data_type.__name__].__name__}_var'
            left_operand.components[right_operand] = Component(name=right_operand, data_type=component.data_type,
                                                      role=Role.MEASURE, nullable=component.nullable)
            left_operand.data[right_operand] = left_operand.data[component.name]
        result_components = {name: comp for name, comp in left_operand.components.items()
                             if comp.role == Role.IDENTIFIER or comp.name == right_operand}
        result_dataset = Dataset(name="result", components=result_components, data=None)
        return result_dataset

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result_dataset = cls.validate(left_operand, right_operand)
        result_dataset.data = left_operand.data[list(result_dataset.components.keys())]
        return result_dataset


class Alias(Binary):

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: str):
        if left_operand.name == right_operand:
            raise ValueError("Alias operation requires different names")
        return Dataset(name=right_operand, components=left_operand.components, data=None)

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result = cls.validate(left_operand, right_operand)
        result.data = left_operand.data
        return result

from Model import Dataset, Role
from Operators import Binary


class Membership(Binary):

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: str):
        if right_operand not in left_operand.components:
            raise ValueError(f"Component {right_operand} not found in dataset {left_operand.name}")

        component = left_operand.components[right_operand]
        if component.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            raise ValueError(f"Component {right_operand} is an {component.role.value} and "
                             f"cannot be used in a membership operation")
        result_components = {name: comp for name, comp in left_operand.components.items()
                             if comp.role == Role.IDENTIFIER or comp.name == right_operand}
        result_dataset = Dataset(name="result", components=result_components, data=None)
        return result_dataset

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result_dataset = cls.validate(left_operand, right_operand)
        result_dataset.data = left_operand.data[list(result_dataset.components.keys())]
        return result_dataset

from Model import Dataset, Role


class Membership:
    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        components = {name: comp for name, comp in left_operand.components.items()
                      if comp.role == Role.IDENTIFIER or comp.name == right_operand}
        data = left_operand.data.loc[:, [name for name in components]]
        return Dataset(name="result", components=components, data=data)
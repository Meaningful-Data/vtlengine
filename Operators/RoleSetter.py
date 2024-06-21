from Model import Role, DataComponent


class Identifier:

    @classmethod
    def evaluate(cls, operand: DataComponent):
        operand.role = Role.IDENTIFIER
        return operand


class Attribute:

    @classmethod
    def evaluate(cls, operand: DataComponent):
        operand.role = Role.ATTRIBUTE
        return operand


class Measure:

    @classmethod
    def evaluate(cls, operand: DataComponent):
        operand.role = Role.MEASURE
        return operand

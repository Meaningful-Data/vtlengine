from Model import DataComponent, Role
from Operators import Unary


class Identifier(Unary):

    @classmethod
    def validate(cls, operand: DataComponent):
        operand.role = Role.IDENTIFIER
        if operand.nullable:
            raise Exception("An Identifier cannot be nullable")
        return operand

    @classmethod
    def evaluate(cls, operand: DataComponent):
        if any(operand.data.isnull()):
            raise Exception("An Identifier cannot have null values")
        return cls.validate(operand)


class Attribute(Unary):

    @classmethod
    def validate(cls, operand: DataComponent):
        operand.role = Role.ATTRIBUTE
        return operand

    @classmethod
    def evaluate(cls, operand: DataComponent):
        if operand.nullable and any(operand.data.isnull()):
            raise Exception("Found null values in an Attribute with nullable=True")
        return cls.validate(operand)


class Measure(Unary):

    @classmethod
    def validate(cls, operand: DataComponent):
        operand.role = Role.MEASURE
        return operand

    @classmethod
    def evaluate(cls, operand: DataComponent):
        if operand.nullable and any(operand.data.isnull()):
            raise Exception("Found null values in a Measure with nullable=True")
        return cls.validate(operand)

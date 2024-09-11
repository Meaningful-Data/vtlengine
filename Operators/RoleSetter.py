import os
from copy import copy

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import DataComponent, Role, Scalar
from Operators import Unary

ALLOWED_MODEL_TYPES = [DataComponent, Scalar]


class RoleSetter(Unary):
    role = None

    @classmethod
    def validate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0):
        if isinstance(operand, Scalar):

            nullable = True
            if cls.role == Role.IDENTIFIER or operand.value is not None:
                nullable = False

            return DataComponent(
                name=operand.name,
                data_type=operand.data_type,
                role=cls.role,
                nullable=nullable,
                data=None
            )
        operand.role = cls.role
        return copy(operand)

    @classmethod
    def evaluate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0):
        if isinstance(operand, DataComponent):
            if not operand.nullable and any(operand.data.isnull()):
                raise Exception(f"Found null values in {operand.name} with nullable=False")
        result = cls.validate(operand, data_size)
        if isinstance(operand, Scalar):
            result.data = pd.Series([operand.value] * data_size)
        else:
            result.data = operand.data
        return result


class Identifier(RoleSetter):
    role = Role.IDENTIFIER

    @classmethod
    def validate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0):
        result = super().validate(operand)
        if result.nullable and any(result.data.isnull()):
            raise Exception("An Identifier cannot be nullable")
        elif result.nullable:
            result.nullable = False
        return result

    @classmethod
    def evaluate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0):
        if isinstance(operand, Scalar):
            if operand.value is None:
                raise Exception("An Identifier cannot be nullable")
        return cls.validate(operand)


class Attribute(RoleSetter):
    role = Role.ATTRIBUTE


class Measure(RoleSetter):
    role = Role.MEASURE

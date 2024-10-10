import os
from copy import copy
from typing import Any

from vtlengine.Exceptions import SemanticError

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from vtlengine.Model import DataComponent, Role, Scalar
from vtlengine.Operators import Unary

ALLOWED_MODEL_TYPES = [DataComponent, Scalar]


class RoleSetter(Unary):
    role = None

    @classmethod
    def validate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0) -> DataComponent:
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
    def evaluate(cls, operand: Any, data_size: int = 0) -> DataComponent:
        if isinstance(operand, DataComponent):
            if not operand.nullable and any(operand.data.isnull()):
                raise SemanticError("1-1-1-16")
        result = cls.validate(operand, data_size)
        if isinstance(operand, Scalar):
            result.data = pd.Series([operand.value] * data_size, dtype=object)
        else:
            result.data = operand.data
        return result


class Identifier(RoleSetter):
    role = Role.IDENTIFIER

    @classmethod
    def validate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0) -> DataComponent:
        result = super().validate(operand)
        if result.nullable:
            raise SemanticError("1-1-1-16")
        return result

    @classmethod
    def evaluate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0) -> DataComponent:
        if isinstance(operand, Scalar):
            if operand.value is None:
                raise SemanticError("1-1-1-16")
        return super().evaluate(operand, data_size)


class Attribute(RoleSetter):
    role = Role.ATTRIBUTE


class Measure(RoleSetter):
    role = Role.MEASURE

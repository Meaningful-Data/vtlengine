from copy import copy
from typing import Union

from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Role, Scalar
from vtlengine.Operators import Unary

ALLOWED_MODEL_TYPES = Union[DataComponent, Scalar]


class RoleSetter(Unary):
    role: Role

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
                data=None,
            )
        operand.role = cls.role
        return copy(operand)


class Identifier(RoleSetter):
    role = Role.IDENTIFIER

    @classmethod
    def validate(cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0) -> DataComponent:
        result = super().validate(operand)
        if result.nullable:
            raise SemanticError("1-1-1-16")
        return result


class Attribute(RoleSetter):
    role = Role.ATTRIBUTE


class Measure(RoleSetter):
    role = Role.MEASURE


class ViralAttribute(RoleSetter):
    role = Role.VIRAL_ATTRIBUTE

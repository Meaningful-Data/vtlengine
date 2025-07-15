from copy import copy
from typing import Any, Union

from vtlengine.connection import con
from vtlengine.DataTypes import String
from vtlengine.Duckdb.duckdb_utils import null_counter
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

    @classmethod
    def evaluate(cls, operand: Any, data_size: int = 0) -> DataComponent:
        # TODO: I cant find another way to do it lazily,
        #  instead Im trying the lightweight way I found to do it.
        if isinstance(operand, DataComponent) and operand.data is not None and not operand.nullable:
            null_count = null_counter(operand.data, operand.name)
            if null_count > 0:
                raise SemanticError("1-1-1-16")

        result = cls.validate(operand, data_size)
        if isinstance(operand, Scalar):
            if operand.value is None:
                operand.value = "NULL"
            if operand.data_type == String:
                operand.value = f"'{operand.value}'"
            query = f"SELECT {operand.value} AS {result.name} FROM range({data_size})"
            result.data = con.query(query)
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
    def evaluate(  # type: ignore[override]
        cls, operand: ALLOWED_MODEL_TYPES, data_size: int = 0
    ) -> DataComponent:
        if isinstance(operand, Scalar) and operand.value is None:
            raise SemanticError("1-1-1-16")
        return super().evaluate(operand, data_size)


class Attribute(RoleSetter):
    role = Role.ATTRIBUTE


class Measure(RoleSetter):
    role = Role.MEASURE

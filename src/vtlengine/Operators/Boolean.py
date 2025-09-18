from typing import Optional

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import AND, NOT, OR, XOR
from vtlengine.DataTypes import Boolean


class Unary(Operator.Unary):
    type_to_check = Boolean
    return_type = Boolean

    @classmethod
    def apply_unary_op_scalar(cls, value: Optional[bool]) -> str:
        return cls.py_op(value)


class Binary(Operator.Binary):
    type_to_check = Boolean
    return_type = Boolean

    @classmethod
    def apply_bin_op(cls, me_name: Optional[str], left: str, right: str) -> str:
        if me_name is None:
            return f"{cls.duck_op(left, right)}"
        return f'{cls.duck_op(left, right)} AS {me_name}'

    @classmethod
    def apply_bin_op_scalar(cls, left: Optional[bool], right: Optional[bool]) -> str:
        return cls.py_op(left, right)

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        raise Exception("Method not allowed")


class And(Binary):
    op = AND

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        query = f"""
                CASE
                    WHEN {left} IS FALSE OR {right} IS FALSE THEN FALSE
                    WHEN {left} IS NULL OR {right} IS NULL THEN NULL
                    ELSE TRUE
                END
                """
        return query

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (x is None and y == False) or (x == False and y is None):
            return False
        elif x is None or y is None:
            return None
        return x and y


class Or(Binary):
    op = OR

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        return f"""
            CASE
                WHEN {left} IS TRUE OR {right} IS TRUE THEN TRUE
                WHEN {left} IS NULL OR {right} IS NULL THEN NULL
                ELSE FALSE
            END
            """

    @staticmethod
    def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if (x is None and y == True) or (x == True and y is None):
            return True
        elif x is None or y is None:
            return None
        return x or y


class Xor(Binary):
    op = XOR

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        return f"""
                CASE
                    WHEN {left} IS NULL OR {right} IS NULL THEN NULL
                    ELSE ({left} AND NOT {right}) OR (NOT {left} AND {right})
                END
                """

    @classmethod
    def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
        if x is None or y is None:
            return None
        return (x and not y) or (not x and y)


class Not(Unary):
    op = NOT

    @classmethod
    def apply_unary_op(cls, measure_name: Optional[str], operand: str) -> str:
        if measure_name is None:
            return f"{cls.duck_op(operand)}"
        else:
            return f'{cls.duck_op(operand)} AS "{measure_name}"'

    @classmethod
    def duck_op(cls, operand: str) -> str:
        query = f"""
                CASE
                    WHEN {operand} IS NULL THEN NULL
                    ELSE NOT {operand}
                END
                """
        return query

    @staticmethod
    def py_op(x: Optional[bool]) -> Optional[bool]:
        return None if x is None else not x

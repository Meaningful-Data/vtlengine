from typing import Any, Type, Optional

from pandas.io.stata import excessive_string_length_error

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import AND, NOT, OR, XOR
from vtlengine.DataTypes import Boolean

def value_handled(value: Any) -> str:
    if value is True or value == "True":
        return "TRUE"
    elif value is False or value == "False":
        return "FALSE"
    elif value is None or value == "None":
        return "NULL"
    elif isinstance(value, str):
        return f'"{value}"'
    return str(value)

class Unary(Operator.Unary):
    type_to_check = Boolean
    return_type = Boolean


class Binary(Operator.Binary):
    type_to_check = Boolean
    return_type = Boolean

    @classmethod
    def apply_bin_op(cls: Type["Binary"], me_name: str, left: Any, right: Any) -> str:
        # TODO: Change this or remove this to use duckdb functions in Boolean
        if me_name is None:
            return f"{cls.duck_op(left, right)}"
        return f"{cls.duck_op(left, right)} AS {me_name}"

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        raise Exception("Method not allowed")


class And(Binary):
    op = AND

    @classmethod
    def apply_bin_op(cls, me_name: Optional[str], left: str, right: str) -> str:

        if me_name is None:
            return f"{cls.duck_op(left, right)}"
        return f"{cls.duck_op(left, right)} AS \"{me_name}\""

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:

        query = f"""
                CASE
                    WHEN {left} IS FALSE OR {right} IS FALSE THEN FALSE
                    WHEN {left} IS NULL OR {right} IS NULL THEN NULL
                    ELSE TRUE
                END
                """
        print(query)
        return query

    # @staticmethod
    # def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
    #     if (x is None and y == False) or (x == False and y is None):
    #         return False
    #     elif x is None or y is None:
    #         return None
    #     return x and y


class Or(Binary):
    op = OR

    @classmethod
    def apply_bin_op(cls, me_name: str, left: str, right: str) -> str:
        if me_name is None:
            return f"{cls.duck_op(left, right)}"
        return f"{cls.duck_op(left, right)} AS \"{me_name}\""

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        return f"""
            CASE
                WHEN {left} IS TRUE OR {right} IS TRUE THEN TRUE
                WHEN {left} IS NULL OR {right} IS NULL THEN NULL
                ELSE FALSE
            END
            """

    # @staticmethod
    # def py_op(x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
    #     if (x is None and y == True) or (x == True and y is None):
    #         return True
    #     elif x is None or y is None:
    #         return None
    #     return x or y


class Xor(Binary):
    op = XOR

    @classmethod
    def apply_bin_op(cls, me_name: str, left: str, right: str) -> str:
        if me_name is None:
            return f"{cls.duck_op(left, right)}"
        return f"{cls.duck_op(left, right)} AS \"{me_name}\""

    @classmethod
    def duck_op(cls, left: str, right: str) -> str:
        return f"""
                CASE
                    WHEN {left} IS NULL OR {right} IS NULL THEN NULL
                    ELSE ({left} AND NOT {right}) OR (NOT {left} AND {right})
                END
                """

    # @classmethod
    # def py_op(cls, x: Optional[bool], y: Optional[bool]) -> Optional[bool]:
    #     if pd.isnull(x) or pd.isnull(y):
    #         return None
    #     return (x and not y) or (not x and y)


class Not(Unary):
    op = NOT

    @classmethod
    def apply_unary_op(cls, measure_name: Optional[str], operand: str) -> str:
        if measure_name is None:
            return f"{cls.duck_op(operand)}"
        else:
            return f"{cls.duck_op(operand)} AS \"{measure_name}\""


    @classmethod
    def duck_op(cls, operand: str) -> str:
        query = f"""
                CASE
                    WHEN {value_handled(operand)} IS NULL THEN NULL
                    ELSE NOT {value_handled(operand)}
                END
                """
        print(query)
        return query

    # @staticmethod
    # def py_op(x: Optional[bool]) -> Optional[bool]:
    #     return None if x is None else not x

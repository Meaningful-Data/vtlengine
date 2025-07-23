import operator
from typing import Any, Optional, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    CONCAT,
    INSTR,
    LCASE,
    LEN,
    LTRIM,
    REPLACE,
    RTRIM,
    SUBSTR,
    TRIM,
    UCASE,
)
from vtlengine.DataTypes import Integer, String, check_unary_implicit_promotion
from vtlengine.duckdb.custom_functions.String import (
    instr_duck,
    replace_duck,
    substr_duck,
)
from vtlengine.duckdb.duckdb_utils import duckdb_concat, empty_relation
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Scalar


class Unary(Operator.Unary):
    type_to_check = String

    @classmethod
    def op_func(cls, x: Any) -> Any:
        x = "" if pd.isnull(x) else str(x)
        return cls.py_op(x)

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        return series.map(lambda x: cls.py_op(str(x)), na_action="ignore")

    @classmethod
    def validate_dataset(cls, dataset: Dataset) -> None:
        """
        Validate that the dataset has exactly one measure.
        """
        measures = dataset.get_measures()

        if len(measures) != 1:
            raise SemanticError("1-1-18-1", op=cls.op, name=dataset.name)


class Length(Unary):
    op = LEN
    return_type = Integer
    py_op = len

    @classmethod
    def op_func(cls, x: Any) -> Any:
        result = super().op_func(x)
        if pd.isnull(result):
            return 0
        return result

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        return series.map(cls.op_func)


class Lower(Unary):
    op = LCASE
    py_op = str.lower
    return_type = String


class Upper(Unary):
    op = UCASE
    py_op = str.upper
    return_type = String


class Trim(Unary):
    op = TRIM
    py_op = str.strip
    return_type = String


class Ltrim(Unary):
    op = LTRIM
    py_op = str.lstrip
    return_type = String


class Rtrim(Unary):
    op = RTRIM
    py_op = str.rstrip
    return_type = String


class Binary(Operator.Binary):
    type_to_check = String

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        x = "" if pd.isnull(x) else str(x)
        y = "" if pd.isnull(y) else str(y)
        return cls.py_op(x, y)


class Concatenate(Binary):
    op = CONCAT
    py_op = operator.concat
    return_type = String


class Parameterized(Unary):
    sql_op: str = Unary.op

    @classmethod
    def validate(cls, *args: Any) -> Any:
        operand: Operator.ALL_MODEL_DATA_TYPES
        param1: Optional[Scalar]
        param2: Optional[Scalar]
        operand, param1, param2 = (args + (None, None))[:3]

        if param1 is not None:
            cls.check_param(param1, 1)
        if param2 is not None:
            cls.check_param(param2, 2)
        return super().validate(operand)

    @staticmethod
    # could be here the bug
    def handle_param_value(param: Optional[Union[DataComponent, Scalar]]) -> Optional[str]:
        if isinstance(param, DataComponent):
            print(param.name)
            return param.name
        elif isinstance(param, Scalar) and param.value is not None:
            print(param)
            return f"'{param.value}'"
        return "NULL"

    @classmethod
    def apply_parametrized_op(
        cls, me_name: str, param_name_1: str, param_name_2: str, output_column_name: Any
    ) -> str:
        return f'{cls.sql_op}({me_name}, {param_name_1}, {param_name_2}) AS "{output_column_name}"'

    @classmethod
    def dataset_evaluation(
        cls,
        *args: Any,
    ) -> Dataset:
        operand: Dataset
        param1: Optional[Scalar]
        param2: Optional[Scalar]
        operand, param1, param2 = args[:3]
        result_dataset = cls.validate(operand, param1, param2)
        result_data = operand.data if operand.data is not None else empty_relation()

        expr = [f"{d}" for d in operand.get_identifiers_names()]

        param1_value = "NULL" if param1 is None or param1.value is None else f"'{param1.value}'"
        param2_value = "NULL" if param2 is None or param2.value is None else f"'{param2.value}'"

        for measure_name in operand.get_measures_names():
            expr.append(
                cls.apply_parametrized_op(measure_name, param1_value, param2_value, measure_name)
            )

        result_dataset.data = result_data.project(", ".join(expr))

        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def component_evaluation(cls, *args: Any) -> DataComponent:
        operand: DataComponent
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        operand, param1, param2 = (args + (None, None))[:3]

        result = cls.validate(operand, param1, param2)

        param1_value = cls.handle_param_value(param1)
        param2_value = cls.handle_param_value(param2)

        # Combining data from operand and parameters into a single DuckdbPyRelation
        all_data = operand.data if operand.data is not None else empty_relation()

        for param in args[1:]:
            if isinstance(param, DataComponent):
                all_data = duckdb_concat(all_data, param.data)

        result.data = all_data.project(
            cls.apply_parametrized_op(operand.name, param1_value, param2_value, operand.name)
        )

        return result

    @classmethod
    def scalar_evaluation(cls, *args: Any) -> Scalar:
        operand: Scalar
        param1: Optional[Scalar]
        param2: Optional[Scalar]
        operand, param1, param2 = (args + (None, None))[:3]

        result = cls.validate(operand, param1, param2)
        param_value1 = None if param1 is None else param1.value
        param_value2 = None if param2 is None else param2.value
        result.value = (
            None if operand.value is None else cls.py_op(operand.value, param_value1, param_value2)
        )
        return result

    @classmethod
    def evaluate(cls, *args: Any) -> Union[Dataset, DataComponent, Scalar]:
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        operand, param1, param2 = (args + (None, None))[:3]

        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param1, param2)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param1, param2)
        return cls.scalar_evaluation(operand, param1, param2)

    @classmethod
    def check_param(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def check_param_value(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")


class Substr(Parameterized):
    op = SUBSTR
    return_type = String
    sql_op = "substr_duck"
    py_op = substr_duck

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2):
            raise SemanticError("1-1-18-3", op=cls.op, pos=position)
        data_type: Any = param.data_type

        if not check_unary_implicit_promotion(data_type, Integer):
            raise SemanticError("1-1-18-4", op=cls.op, param_type=cls.op, correct_type="Integer")

        if isinstance(param, Scalar):
            cls.check_param_value(param.value, position)

    @classmethod
    def check_param_value(cls, param: Optional[Any], position: int) -> None:
        if param is not None:
            if param is not None and not param >= 1 and position == 1:
                raise SemanticError("1-1-18-4", op=cls.op, param_type="Start", correct_type=">= 1")
            elif param is not None and not param >= 0 and position == 2:
                raise SemanticError("1-1-18-4", op=cls.op, param_type="Length", correct_type=">= 0")


class Replace(Parameterized):
    op = REPLACE
    return_type = String
    sql_op = "replace_duck"
    py_op = replace_duck

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2):
            raise SemanticError("1-1-18-3", op=cls.op, pos=position)
        data_type: Any = param.data_type

        if not check_unary_implicit_promotion(data_type, String):
            raise SemanticError("1-1-18-4", op=cls.op, param_type=cls.op, correct_type="String")

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)


class Instr(Parameterized):
    op = INSTR
    return_type = Integer
    sql_op = "instr_duck"
    py_op = instr_duck

    @classmethod
    def validate(
        cls,
        operand: Operator.ALL_MODEL_DATA_TYPES,
        param1: Optional[Operator.ALL_MODEL_DATA_TYPES] = None,
        param2: Optional[Operator.ALL_MODEL_DATA_TYPES] = None,
        param3: Optional[Operator.ALL_MODEL_DATA_TYPES] = None,
    ) -> Any:
        if (
            isinstance(param1, Dataset)
            or isinstance(param2, Dataset)
            or isinstance(param3, Dataset)
        ):
            raise SemanticError("1-1-18-10", op=cls.op)
        if param1 is not None:
            cls.check_param(param1, 1)
        if param2 is not None:
            cls.check_param(param2, 2)
        if param3 is not None:
            cls.check_param(param3, 3)

        return super().validate(operand)

    @classmethod
    def validate_params(cls, params: Any) -> None:
        if len(params) != 2:
            raise SemanticError("1-1-18-7", op=cls.op, number=len(params), expected=2)

    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position: int) -> None:
        if not param:
            return
        if position not in (1, 2, 3):
            raise SemanticError("1-1-18-9", op=cls.op)
        data_type: Any = param.data_type

        if position == 1:
            if not check_unary_implicit_promotion(data_type, String):
                raise SemanticError(
                    "1-1-18-4", op=cls.op, param_type="Pattern", correct_type="String"
                )
        elif position == 2:
            if not check_unary_implicit_promotion(data_type, Integer):
                raise SemanticError(
                    "1-1-18-4", op=cls.op, param_type="Start", correct_type="Integer"
                )
        else:
            if not check_unary_implicit_promotion(data_type, Integer):
                raise SemanticError(
                    "1-1-18-4",
                    op=cls.op,
                    param_type="Occurrence",
                    correct_type="Integer",
                )
        if position >= 2 and isinstance(param, Scalar) and param is not None:
            if position == 2 and param.value is not None and param.value < 1:
                raise SemanticError("1-1-18-4", op="instr", param_type="Start", correct_type=">= 1")
            elif position == 3 and param.value is not None and param.value < 1:
                raise SemanticError(
                    "1-1-18-4", op="instr", param_type="Occurrence", correct_type=">= 1"
                )
        return None

    @classmethod
    def apply_instr_op(
        cls,
        operand: str,
        param_1: str,
        param_2: Union[str, int],
        param_3: Union[str, int],
        output_column_name: Any,
    ) -> str:
        return f'{cls.sql_op}({operand}, {param_1}, {param_2}, {param_3}) AS "{output_column_name}"'

    @classmethod
    def dataset_evaluation(cls, *args: Any) -> Dataset:
        operand, param1, param2, param3 = args[:4]
        result_dataset = cls.validate(operand, param1, param2, param3)
        result_data = operand.data if operand.data is not None else empty_relation()

        expr = [f"{d}" for d in operand.get_identifiers_names()]

        param_value1 = "NULL" if param1 is None or param1.value is None else f"'{param1.value}'"
        param_value2 = "NULL" if param2 is None or param2.value is None else f"'{param2.value}'"
        param_value3 = "NULL" if param3 is None or param3.value is None else f"'{param3.value}'"

        for measure_name in operand.get_measures_names():
            expr.append(
                cls.apply_instr_op(
                    measure_name, param_value1, param_value2, param_value3, measure_name
                )
            )

        result_dataset.data = result_data.project(", ".join(expr))

        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def component_evaluation(cls, *args: Any) -> DataComponent:
        operand: DataComponent
        param1: Optional[Union[DataComponent, Scalar]]
        param2: Optional[Union[DataComponent, Scalar]]
        param3: Optional[Union[DataComponent, Scalar]]
        operand, param1, param2, param3 = args[:4]

        # Validation and handling parameters
        result = cls.validate(operand, param1, param2, param3)
        param1_value = cls.handle_param_value(param1)
        param2_value = cls.handle_param_value(param2)
        param3_value = cls.handle_param_value(param3)

        # Combining data from operand and parameters into a single DuckdbPyRelation
        all_data = operand.data if operand.data is not None else empty_relation()

        for param in args[1:]:
            if isinstance(param, DataComponent):
                all_data = duckdb_concat(all_data, param.data)

        result.data = all_data.project(
            cls.apply_instr_op(operand.name, param1_value, param2_value, param3_value, operand.name)
        )

        return result

    @classmethod
    def scalar_evaluation(cls, *args: Any) -> Scalar:
        operand: Scalar
        param1: Optional[Scalar]
        param2: Optional[Scalar]
        param3: Optional[Scalar]
        operand, param1, param2, param3 = args[:4]
        result = cls.validate(operand, param1, param2, param3)
        param_value1 = None if param1 is None else param1.value
        param_value2 = None if param2 is None else param2.value
        param_value3 = None if param3 is None else param3.value
        result.value = cls.py_op(operand.value, param_value1, param_value2, param_value3)
        return result

    @classmethod
    def evaluate(
        cls,
        operand: Operator.ALL_MODEL_DATA_TYPES,
        param1: Optional[Any] = None,
        param2: Optional[Any] = None,
        param3: Optional[Any] = None,
    ) -> Any:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param1, param2, param3)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param1, param2, param3)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, param1, param2, param3)

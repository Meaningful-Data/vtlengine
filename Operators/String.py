import operator
import os
import re

from Model import Scalar, Dataset, DataComponent

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from typing import Optional, Any, Union

from AST.Grammar.tokens import LEN, CONCAT, UCASE, LCASE, RTRIM, SUBSTR, LTRIM, TRIM, REPLACE, INSTR
from DataTypes import Integer, String, COMP_NAME_MAPPING, ScalarType
import Operators as Operator


class Unary(Operator.Unary):
    type_to_check = String


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
        x = "" if pd.isnull(x) else x
        y = "" if pd.isnull(y) else y
        return cls.py_op(x, y)


class Concatenate(Binary):
    op = CONCAT
    py_op = operator.concat
    return_type = String

class Parameterized(Unary):

    @classmethod
    def validate(cls, operand: Operator.ALL_MODEL_DATA_TYPES, param1: Optional[Scalar] = None,
                 param2: Optional[Scalar] = None):

        if param1 is not None:
            cls.check_param(param1, 2)
            if param2 is not None:
                cls.check_param(param2, 2)
                if isinstance(param1, Dataset) or isinstance(param2, Dataset):
                    raise Exception(f"{cls.op} cannot have a Dataset as parameter")
                if isinstance(param1, DataComponent) or isinstance(param2, DataComponent):
                    raise Exception(f"{cls.op} cannot have a DataComponent as parameter")


        return super().validate(operand)

    @classmethod
    def op_func(cls, x: Union[Dataset, String], param1: Optional[Any], param2: Optional[Any]) -> Any:
        x = "" if pd.isnull(x) else x
        return cls.py_op(x, param1, param2)

    @classmethod
    def apply_operation_two_series(cls, left_series: pd.Series, right_series: pd.Series) -> Any:
        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def apply_operation_series_scalar(cls, series: pd.Series, param1: Any, param2: Any) -> Any:
        return series.map(lambda x: cls.op_func(x, param1, param2))


    @classmethod
    def dataset_evaluation(cls, operand: Dataset,
                           param1: Optional[Union[DataComponent, Scalar]],
                           param2: Optional[Union[DataComponent, Scalar]]):
        result = cls.validate(operand, param1, param2)
        result.data = operand.data.copy()
        for measure_name in operand.get_measures_names():
            if isinstance(param1, DataComponent) or isinstance(param2, DataComponent):
                if isinstance(param1, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param1.data
                    )
                if isinstance(param2, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param2.data
                    )
            else:
                param_value1 = None if param1 is None else param1.value
                param_value2 = None if param2 is None else param2.value
                result.data[measure_name] = cls.apply_operation_series_scalar(
                    result.data[measure_name], param_value1, param_value2
                )
            cls.modify_measure_column(result)
        return result

    @classmethod
    def component_evaluation(cls, operand: DataComponent,
                             param1: Optional[Union[DataComponent, Scalar]],
                             param2: Optional[Union[DataComponent, Scalar]]):
        result = cls.validate(operand, param1, param2)
        result.data = operand.data.copy()
        if isinstance(param1, DataComponent) or isinstance(param2, DataComponent):
            if isinstance(param1, DataComponent):
                raise NotImplementedError
            if isinstance(param2, DataComponent):
                raise NotImplementedError
        else:
            param_value1 = None if param1 is None else param1.value
            param_value2 = None if param2 is None else param2.value
            result.data = cls.apply_operation_series_scalar(operand.data, param_value1, param_value2)
        return result

    @classmethod
    def scalar_evaluation(cls, operand: Scalar,
                          param1: Optional[Union[DataComponent, Scalar]],
                          param2: Optional[Union[DataComponent, Scalar]]):
        result = cls.validate(operand, param1, param2)
        param_value1 = None if param1 is None else param1.value
        param_value2 = None if param2 is None else param2.value
        result.value = cls.op_func(operand.value, param_value1, param_value2)
        return result

    @classmethod
    def evaluate(cls, operand: Operator.ALL_MODEL_DATA_TYPES,
                 param1: Optional[Union[DataComponent, Scalar]] = None,
                 param2: Optional[Union[DataComponent, Scalar]] = None) -> Operator.ALL_MODEL_DATA_TYPES:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param1, param2)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param1, param2)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, param1, param2)
        
    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position:int):
        raise Exception("Method should be implemented by inheritors")


class Substr(Parameterized):
    op = SUBSTR
    return_type = String

    @classmethod
    def py_op(cls, x: str, param1: Optional[Any], param2: Optional[Any]) -> Any:
        if param1 is None and param2 is None:
            return x
        if param1 is None:
            param1 = 0
        elif param1 is not 0:
            param1 -= 1
        elif param1 > (len(x)):
            return ""
        if param2 is None or (param1 + param2) > len(x):
            param2 = len(x)
        else:
            param2 = (param1 + param2)
        return x[param1:param2]
    
    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position:int):
        if param:
            if position not in (1,2):
                raise Exception("param position is not specified")
            data_type:ScalarType = param.data_type

            if not data_type == Integer:
                raise Exception("Substr params should be Integer")
            
            if isinstance(param, DataComponent):
                value = param.data[0]
            else:
                value = param.value
            if position == 1:
                if not value >= 1:
                    raise Exception("param start should be >= 1")
            else:
                if not value >= 0:
                    raise Exception("param length should be >= 0")


class Replace(Parameterized):
    op = REPLACE
    return_type = String

    @classmethod
    def py_op(cls, x: str, param1: Optional[Any], param2: Optional[Any]) -> Any:
        if param1 is None:
            return ""
        elif param2 is None:
            param2 = ''
        return x.replace(param1, param2)
    
    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position:int):
        if param:
            if position not in (1,2):
                raise Exception("param position is not specified")
            data_type:ScalarType = param.data_type

            if not data_type == String:
                raise Exception("Replace params should be String")


class Instr(Parameterized):
    op = INSTR
    return_type = Integer

    @classmethod
    def validate(cls, operand: Operator.ALL_MODEL_DATA_TYPES,
                 param1: Optional[Scalar] = None,
                 param2: Optional[Scalar] = None,
                 param3: Optional[Scalar] = None):

        if isinstance(param1, Dataset) or isinstance(param2, Dataset) or isinstance(param3, Dataset):
            raise Exception(f"{cls.op} cannot have a Dataset as parameter")
        if isinstance(param1, DataComponent) or isinstance(param2, DataComponent) or isinstance(param3, DataComponent):
            raise Exception(f"{cls.op} cannot have a DataComponent as parameter")

        if param1 is not None:
            cls.check_param(param1, 1)
        if param2 is not None:
            cls.check_param(param2, 2)
        if param3 is not None:
            cls.check_param(param3, 3)


        return super().validate(operand)
    
    @classmethod
    def check_param(cls, param: Optional[Union[DataComponent, Scalar]], position:int):
        if param:
            if position not in (1,2,3):
                raise Exception("param position is not specified")
            data_type:ScalarType = param.data_type
            
            if isinstance(param, DataComponent):
                value = param.data[0]
            else:
                value = param.value
            if position == 1:
                if not data_type == String:
                    raise Exception("Instr pattern param should be String")
            elif position ==2:
                if not data_type == Integer:
                    raise Exception("Instr start param should be Integer")
                if not value >= 1:
                    raise Exception("param start should be >= 1")
            else:
                if not data_type == Integer:
                    raise Exception("Instr occurrence param should be Integer")
                if not value >= 1:
                    raise Exception("param occurrence should be >= 1")

    @classmethod
    def apply_operation_two_series(cls, left_series: pd.Series, right_series: pd.Series) -> Any:
        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def apply_operation_series_scalar(cls, series: pd.Series, param1: Any, param2: Any, param3: Any) -> Any:
        return series.map(lambda x: cls.op_func(x, param1, param2, param3))

    @classmethod
    def dataset_evaluation(cls, operand: Dataset,
                           param1: Optional[Union[DataComponent, Scalar]],
                           param2: Optional[Union[DataComponent, Scalar]],
                           param3: Optional[Union[DataComponent, Scalar]]):
        result = cls.validate(operand, param1, param2, param3)
        result.data = operand.data.copy()
        for measure_name in operand.get_measures_names():
            if isinstance(param1, DataComponent) or isinstance(param2, DataComponent) or isinstance(param3,
                                                                                                    DataComponent):
                if isinstance(param1, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param1.data
                    )
                if isinstance(param2, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param2.data
                    )
                if isinstance(param3, DataComponent):
                    result.data[measure_name] = cls.apply_operation_two_series(
                        result.data[measure_name], param3.data
                    )
            else:
                param_value1 = None if param1 is None else param1.value
                param_value2 = None if param2 is None else param2.value
                param_value3 = None if param3 is None else param3.value
                result.data[measure_name] = cls.apply_operation_series_scalar(
                    result.data[measure_name], param_value1, param_value2, param_value3
                )
            cls.modify_measure_column(result)
        return result

    @classmethod
    def component_evaluation(cls, operand: DataComponent,
                             param1: Optional[Union[DataComponent, Scalar]],
                             param2: Optional[Union[DataComponent, Scalar]],
                             param3: Optional[Union[DataComponent, Scalar]]):
        result = cls.validate(operand, param1, param2)
        result.data = operand.data.copy()
        if isinstance(param1, DataComponent) or isinstance(param2, DataComponent) or isinstance(param3, DataComponent):
            if isinstance(param1, DataComponent):
                result.data = cls.apply_operation_two_series(operand.data, param1.data)
            if isinstance(param2, DataComponent):
                result.data = cls.apply_operation_two_series(operand.data, param2.data)
            if isinstance(param3, DataComponent):
                result.data = cls.apply_operation_two_series(operand.data, param3.data)
        else:
            param_value1 = None if param1 is None else param1.value
            param_value2 = None if param2 is None else param2.value
            param_value3 = None if param3 is None else param3.value
            result.data = cls.apply_operation_series_scalar(operand.data, param_value1, param_value2, param_value3)
        return result

    @classmethod
    def scalar_evaluation(cls, operand: Scalar,
                          param1: Optional[Union[DataComponent, Scalar]],
                          param2: Optional[Union[DataComponent, Scalar]],
                          param3: Optional[Union[DataComponent, Scalar]]):
        result = cls.validate(operand, param1, param2)
        param_value1 = None if param1 is None else param1.value
        param_value2 = None if param2 is None else param2.value
        param_value3 = None if param3 is None else param3.value
        result.value = cls.op_func(operand.value, param_value1, param_value2, param_value3)
        return result

    @classmethod
    def evaluate(cls, operand: Operator.ALL_MODEL_DATA_TYPES,
                 param1: Optional[Union[DataComponent, Scalar]] = None,
                 param2: Optional[Union[DataComponent, Scalar]] = None,
                 param3: Optional[Union[DataComponent, Scalar]] = None) -> Operator.ALL_MODEL_DATA_TYPES:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, param1, param2, param3)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, param1, param2, param3)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, param1, param2, param3)

    @classmethod
    def op_func(cls, x: Union[Dataset, String], param1: Optional[Any], param2: Optional[Any],
                param3: Optional[Any]) -> Any:
        x = "" if pd.isnull(x) else x
        return cls.py_op(x, param1, param2, param3)

    @classmethod
    def py_op(cls, str_value: str, str_to_find: Optional[str], start: Optional[int], occurrence: Optional[int]) -> Any:
        if start is not None:
            if isinstance(start, int) or start.is_integer():
                start = int(start - 1)
            else:
                # OPERATORS_STRINGOPERATORS.92
                raise Exception(f"At op {cls.op}: Start parameter value {start} should be integer.")
        else:
            start = 0

        if occurrence is not None:
            if isinstance(occurrence, int) or occurrence.is_integer():
                occurrence = int(occurrence - 1)
            else:
                # OPERATORS_STRINGOPERATORS.93
                raise Exception(f"At op {cls.op}: Occurrence parameter value {occurrence} should be integer.")
        else:
            occurrence = 0
        if str_to_find is None:
            return 0

        occurrences_list = [m.start() for m in re.finditer(str_to_find, str_value[start:])]

        length = len(occurrences_list)

        if occurrence > length - 1:
            position = 0
        else:
            position = int(start + occurrences_list[occurrence] + 1)

        return position

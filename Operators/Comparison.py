import operator
import os
import re
from typing import Any, Optional, Union

from Model import Component, DataComponent, Dataset, Role, Scalar

from AST.Grammar.tokens import EQ, GT, GTE, LT, LTE, NEQ
from Operators import Binary

if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

from AST.Grammar.tokens import CHARSET_MATCH, EQ, GT, GTE, IN, ISNULL, LT, LTE, NEQ
from DataTypes import Boolean, COMP_NAME_MAPPING, String
import Operators as Operator


class Unary(Operator.Unary):
    return_type = Boolean


class IsNull(Unary):
    op = ISNULL
    py_op = pd.isnull

    @classmethod
    def apply_operation_component(cls, series: pd.Series) -> Any:
        return series.isnull()

    @classmethod
    def op_func(cls, x: Any) -> Any:
        return pd.isnull(x)

    @classmethod
    def dataset_validation(cls, operand: Dataset):
        result = super().dataset_validation(operand)
        for measure in result.get_measures():
            measure.nullable = False
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent):
        result = super().component_validation(operand)
        result.nullable = False
        return result


class Binary(Operator.Binary):
    return_type = Boolean


class Equal(Binary):
    op = EQ
    py_op = operator.eq


class NotEqual(Binary):
    op = NEQ
    py_op = operator.ne


class Greater(Binary):
    op = GT
    py_op = operator.gt


class GreaterEqual(Binary):
    op = GTE
    py_op = operator.ge


class Less(Binary):
    op = LT
    py_op = operator.lt


class LessEqual(Binary):
    op = LTE
    py_op = operator.le


class In(Binary):
    op = IN

    @classmethod
    def apply_operation_two_series(cls,
                                   left_series: Any,
                                   right_series: Any) -> Any:
        return left_series.isin(right_series)

    @classmethod
    def py_op(cls, x, y):
        return operator.contains(y, x)


class Match(Binary):
    op = CHARSET_MATCH
    type_to_check = String

    @classmethod
    def py_op(cls, x, y):
        if isinstance(x, pd.Series):
            return x.str.fullmatch(y)
        return bool(re.fullmatch(y, x))


class Between(Operator.Operator):
    return_type = Boolean

    @classmethod
    def op_function(cls, x, y, z):
        return None if pd.isnull(x) or pd.isnull(y) or pd.isnull(z) else y <= x <= z

    @classmethod
    def apply_operation_component(cls, series: pd.Series,
                                  from_data: Any,
                                  to_data: Any) -> Any:
        return series.map(lambda x: cls.op_function(x, from_data, to_data))

    @classmethod
    def validate(cls, operand: Union[Dataset, DataComponent, Scalar],
                 from_: Union[DataComponent, Scalar],
                 to: Union[DataComponent, Scalar]) -> Any:
        if isinstance(operand, Dataset):
            cls.validate_dataset_type(operand)
            result = Dataset(name=operand.name, components=operand.components.copy(), data=None)
            cls.apply_return_type_dataset(result)
        elif isinstance(operand, DataComponent):
            cls.validate_component_type(operand)
            result = DataComponent(name=operand.name, data=None,
                                   data_type=operand.data_type, role=operand.role)
            cls.apply_return_type(result)
        else:
            cls.validate_scalar_type(operand)
            result = Scalar(name=operand.name, value=None, data_type=operand.data_type)
            cls.apply_return_type(result)

        if isinstance(from_, DataComponent):
            cls.validate_component_type(from_)
        if isinstance(from_, Scalar):
            cls.validate_scalar_type(from_)

        if isinstance(to, DataComponent):
            cls.validate_component_type(to)
        if isinstance(to, Scalar):
            cls.validate_scalar_type(to)

        if isinstance(operand, Dataset):
            for measure in operand.get_measures():
                cls.validate_type_compatibility(measure.data_type, from_.data_type)
                cls.validate_type_compatibility(from_.data_type, to.data_type)
        else:
            cls.validate_type_compatibility(operand.data_type, from_.data_type)
            cls.validate_type_compatibility(from_.data_type, to.data_type)

        return result

    @classmethod
    def evaluate(cls, operand: Union[DataComponent, Scalar],
                 from_: Union[DataComponent, Scalar],
                 to: Union[DataComponent, Scalar]) -> Any:
        result = cls.validate(operand, from_, to)

        from_data = from_.data if isinstance(from_, DataComponent) else from_.value
        to_data = to.data if isinstance(to, DataComponent) else to.value
        if isinstance(operand, Dataset):
            result.data = operand.data.copy()
            for measure_name in operand.get_measures_names():
                result.data[measure_name] = cls.apply_operation_component(
                    operand.data[measure_name],
                    from_data, to_data
                )
                if len(result.get_measures()) == 1:
                    result.data[COMP_NAME_MAPPING[cls.return_type]] = result.data[measure_name]
                    result.data = result.data.drop(columns=[measure_name])
        if isinstance(operand, DataComponent):
            result.data = cls.apply_operation_component(
                operand.data,
                from_data, to_data
            )
        if isinstance(operand, Scalar):
            result.value = from_data <= operand.value <= to_data

        return result


class ExistIn(Operator.Operator):
    op = IN

    # noinspection PyTypeChecker
    @classmethod
    def validate(cls, dataset_1: Dataset, dataset_2: Dataset,
                 retain_element: Optional[Boolean]) -> Any:
        left_identifiers = dataset_1.get_identifiers_names()
        right_identifiers = dataset_2.get_identifiers_names()

        is_subset_left = set(left_identifiers).issubset(right_identifiers)
        if not is_subset_left:
            raise ValueError("Datasets must have common identifiers")

        result_components = {comp.name: comp for comp in dataset_1.get_identifiers()}
        result_dataset = Dataset(name="result", components=result_components, data=None)
        result_dataset.add_component(Component(
            name='bool_var',
            data_type=Boolean,
            role=Role.MEASURE,
            nullable=False
        ))
        return result_dataset

    @classmethod
    def evaluate(cls, dataset_1: Dataset, dataset_2: Dataset,
                 retain_element: Optional[Boolean]) -> Any:
        result_dataset = cls.validate(dataset_1, dataset_2, retain_element)
        common = result_dataset.get_identifiers_names()
        df1: pd.DataFrame = dataset_1.data[common]
        df2: pd.DataFrame = dataset_2.data[common]
        compare_result = (df1 == df2).all(axis=1)
        result_dataset.data = df1
        result_dataset.data['bool_var'] = compare_result

        if retain_element is not None:
            result_dataset.data = result_dataset.data[
                result_dataset.data['bool_var'] == retain_element]
            result_dataset.data = result_dataset.data.reset_index(drop=True)

        return result_dataset

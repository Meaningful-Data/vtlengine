import operator
import os
import re
from copy import copy
from typing import Any, Optional, Union

from Model import Component, DataComponent, Dataset, Role, Scalar, ScalarSet

if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

from AST.Grammar.tokens import CHARSET_MATCH, EQ, GT, GTE, IN, ISNULL, LT, LTE, NEQ, NOT_IN
from DataTypes import Boolean, COMP_NAME_MAPPING, String, Number
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

    @classmethod
    def _cast_values(cls, x: Union[int, float, str, bool], y: Union[int, float, str, bool]) -> tuple:
        # Cast both values to the same data type
        # An integer can be considered a bool, we must check first boolean, then numbers
        if isinstance(x, str) and isinstance(y, bool):
            y = String.cast(y)
        elif isinstance(x, bool) and isinstance(y, str):
            x = String.cast(x)
        elif isinstance(x, str) and isinstance(y, (int, float)):
            x = Number.cast(x)
        elif isinstance(x, (int, float)) and isinstance(y, str):
            y = Number.cast(y)

        return x, y

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if pd.isnull(x) or pd.isnull(y):
            return None
        x, y = cls._cast_values(x, y)
        return cls.py_op(x, y)

    @classmethod
    def apply_operation_series_scalar(cls, series: pd.Series, scalar: Any,
                                      series_left: bool) -> Any:
        if scalar is None:
            return pd.Series(None, index=series.index)
        if series_left:
            return series.map(lambda x: cls.op_func(x, scalar), na_action='ignore')
        else:
            return series.map(lambda x: cls.op_func(scalar, x), na_action='ignore')

    @classmethod
    def apply_return_type_dataset(
            cls, result_dataset: Dataset, left_operand: Dataset,
            right_operand: Union[Dataset, Scalar, ScalarSet]
    ) -> None:
        super().apply_return_type_dataset(result_dataset, left_operand, right_operand)
        is_mono_measure = len(result_dataset.get_measures()) == 1
        if is_mono_measure:
            measure = result_dataset.get_measures()[0]
            component = Component(
                name=COMP_NAME_MAPPING[Boolean],
                data_type=Boolean,
                role=Role.MEASURE,
                nullable=measure.nullable
            )
            result_dataset.delete_component(measure.name)
            result_dataset.add_component(component)
            if result_dataset.data is not None:
                result_dataset.data.rename(columns={measure.name: component.name}, inplace=True)


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
                                   right_series: list) -> Any:
        right = pd.Series(right_series)
        if left_series.dtype != right.dtype:
            right = right.astype(left_series.dtype)
        return left_series.map(lambda x: x in right.values, na_action='ignore')

    @classmethod
    def py_op(cls, x, y):
        return operator.contains(y, x)


class NotIn(Binary):
    op = NOT_IN

    @classmethod
    def apply_operation_two_series(cls,
                                   left_series: Any,
                                   right_series: list) -> Any:
        series_result = In.apply_operation_two_series(left_series, right_series)
        return ~series_result

    @classmethod
    def py_op(cls, x, y):
        return not operator.contains(y, x)


class Match(Binary):
    op = CHARSET_MATCH
    type_to_check = String

    @classmethod
    def op_func(cls, x, y):
        if pd.isnull(x) or pd.isnull(y):
            return None
        if isinstance(x, pd.Series):
            return x.str.fullmatch(y)
        return bool(re.fullmatch(str(y), str(x)))


class Between(Operator.Operator):
    return_type = Boolean

    @classmethod
    def op_func(cls,
                x: Optional[Union[int, float, bool, str]],
                y: Optional[Union[int, float, bool, str]],
                z: Optional[Union[int, float, bool, str]]):
        return None if pd.isnull(x) or pd.isnull(y) or pd.isnull(z) else y <= x <= z

    @classmethod
    def apply_operation_component(cls, series: pd.Series,
                                  from_data: Optional[Union[pd.Series, int, float, bool, str]],
                                  to_data: Optional[
                                      Union[pd.Series, int, float, bool, str]]) -> Any:
        control_any_series_from_to = isinstance(from_data, pd.Series) or isinstance(to_data,
                                                                                    pd.Series)
        if control_any_series_from_to:
            if not isinstance(from_data, pd.Series):
                from_data = pd.Series(from_data, index=series.index)
            if not isinstance(to_data, pd.Series):
                to_data = pd.Series(to_data, index=series.index)
            df = pd.DataFrame({'operand': series, 'from_data': from_data, 'to_data': to_data})
            return df.apply(lambda x: cls.op_func(x['operand'], x['from_data'], x['to_data']),
                            axis=1)

        return series.map(lambda x: cls.op_func(x, from_data, to_data))

    @classmethod
    def apply_return_type_dataset(cls, result_dataset: Dataset, operand: Dataset) -> None:
        is_mono_measure = len(operand.get_measures()) == 1
        for measure in result_dataset.get_measures():
            operand_type = operand.get_component(measure.name).data_type

            result_data_type = cls.type_validation(operand_type)
            if is_mono_measure and operand_type.promotion_changed_type(result_data_type):
                component = Component(
                    name=COMP_NAME_MAPPING[result_data_type],
                    data_type=result_data_type,
                    role=Role.MEASURE,
                    nullable=measure.nullable
                )
                result_dataset.delete_component(measure.name)
                result_dataset.add_component(component)
                if result_dataset.data is not None:
                    result_dataset.data.rename(columns={measure.name: component.name}, inplace=True)
            elif is_mono_measure is False and operand_type.promotion_changed_type(result_data_type):
                raise Exception("Operation not allowed for multimeasure datsets")
            else:
                measure.data_type = result_data_type

    @classmethod
    def validate(cls, operand: Union[Dataset, DataComponent, Scalar],
                 from_: Union[DataComponent, Scalar],
                 to: Union[DataComponent, Scalar]) -> Any:
        if isinstance(operand, Dataset):
            result_components = {comp_name: copy(comp) for comp_name, comp in
                                 operand.components.items()
                                 if comp.role == Role.IDENTIFIER or comp.role == Role.MEASURE}
            result = Dataset(name=operand.name, components=result_components, data=None)
        elif isinstance(operand, DataComponent):
            result = DataComponent(name=operand.name, data=None,
                                   data_type=cls.return_type, role=operand.role)
        elif isinstance(operand, Scalar) and isinstance(from_, Scalar) and isinstance(to, Scalar):
            result = Scalar(name=operand.name, value=None, data_type=cls.return_type)
        else:  # From or To is a DataComponent, or both
            result = DataComponent(name=operand.name, data=None,
                                   data_type=cls.return_type, role=Role.MEASURE)

        if isinstance(operand, Dataset):
            for measure in operand.get_measures():
                cls.validate_type_compatibility(measure.data_type, from_.data_type)
                cls.validate_type_compatibility(measure.data_type, to.data_type)
                cls.apply_return_type_dataset(result, operand)
        else:
            cls.validate_type_compatibility(operand.data_type, from_.data_type)
            cls.validate_type_compatibility(operand.data_type, to.data_type)

        return result

    @classmethod
    def evaluate(cls, operand: Union[DataComponent, Scalar],
                 from_: Union[DataComponent, Scalar],
                 to: Union[DataComponent, Scalar]) -> Any:
        result = cls.validate(operand, from_, to)

        from_data = from_.data if isinstance(from_, DataComponent) else from_.value
        to_data = to.data if isinstance(to, DataComponent) else to.value

        if (
                isinstance(from_data, pd.Series) and
                isinstance(to_data, pd.Series) and
                len(from_data) != len(to_data)
        ):
            raise ValueError("From and To must have the same length")

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
            result.data = result.data[result.get_components_names()]
        if isinstance(operand, DataComponent):
            result.data = cls.apply_operation_component(
                operand.data,
                from_data, to_data
            )
        if isinstance(operand, Scalar) and isinstance(from_, Scalar) and isinstance(to, Scalar):
            if operand.value is None or from_data is None or to_data is None:
                result.value = None
            else:
                result.value = from_data <= operand.value <= to_data
        elif (
                isinstance(operand, Scalar) and
                (
                        isinstance(from_data, pd.Series) or
                        isinstance(to_data, pd.Series)
                )
        ):  # From or To is a DataComponent, or both
            if isinstance(from_data, pd.Series):
                series = pd.Series(operand.value, index=from_data.index)
            else:
                series = pd.Series(operand.value, index=to_data.index)
            result_series = cls.apply_operation_component(series, from_data, to_data)
            result = DataComponent(name=operand.name, data=result_series, data_type=cls.return_type,
                                   role=Role.MEASURE)

        return result


class ExistIn(Operator.Operator):
    op = IN

    # noinspection PyTypeChecker
    @classmethod
    def validate(cls, dataset_1: Dataset, dataset_2: Dataset,
                 retain_element: Optional[Boolean]) -> Any:
        left_identifiers = dataset_1.get_identifiers_names()
        right_identifiers = dataset_2.get_identifiers_names()

        is_subset_right = set(right_identifiers).issubset(left_identifiers)
        is_subset_left = set(left_identifiers).issubset(right_identifiers)
        if not (is_subset_left or is_subset_right):
            raise ValueError("Datasets must have common identifiers")

        result_components = {comp.name: copy(comp) for comp in dataset_1.get_identifiers()}
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

        # Checking the subset
        left_id_names = dataset_1.get_identifiers_names()
        right_id_names = dataset_2.get_identifiers_names()
        is_subset_left = set(left_id_names).issubset(right_id_names)

        # Identifiers for the result dataset
        reference_identifiers_names = left_id_names

        # Checking if the left dataset is a subset of the right dataset
        if is_subset_left:
            common_columns = left_id_names
        else:
            common_columns = right_id_names

        # Check if the common identifiers are equal between the two datasets
        true_results = pd.merge(dataset_1.data, dataset_2.data, how='inner',
                                left_on=common_columns,
                                right_on=common_columns, copy=False)
        true_results = true_results[reference_identifiers_names]

        # Check for empty values
        if true_results.empty:
            true_results['bool_var'] = None
        else:
            true_results['bool_var'] = True

        final_result = pd.merge(dataset_1.data, true_results, how='left',
                                left_on=reference_identifiers_names,
                                right_on=reference_identifiers_names, copy=False)
        final_result = final_result[reference_identifiers_names + ['bool_var']]

        # No null values are returned, only True or False
        final_result.fillna(False, axis=1, inplace=True)

        # Adding to the result dataset
        result_dataset.data = final_result

        # Retain only the elements that are specified (True or False)
        if retain_element is not None:
            result_dataset.data = result_dataset.data[
                result_dataset.data['bool_var'] == retain_element]
            result_dataset.data = result_dataset.data.reset_index(drop=True)

        return result_dataset

    @staticmethod
    def _check_all_columns(row):
        return all(col_value == True for col_value in row)

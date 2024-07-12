import os
from typing import Any, Union

import DataTypes
from AST.Grammar.tokens import AND, OR, XOR, NOT, CAST
from DataTypes import COMP_NAME_MAPPING, TYPE_MAPPING_POSITION, TYPE_PROMOTION_MATRIX, ScalarType
from DataTypes import Boolean as Boolean_type, Integer as Integer_type

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Component, Dataset, Role, Scalar, DataComponent, ScalarSet

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]


class Operator:
    """Superclass for all operators"""
    op = None
    py_op = None
    spark_op = None
    type_to_check = None
    return_type = None

    @classmethod
    def validate_component_type(cls, component: DataComponent) -> None:
        if (cls.type_to_check is not None and
                cls.validate_type_compatibility(component.data_type, cls.type_to_check)):
            raise Exception(f"{component.role} {component.name} "
                            f"is not a {cls.type_to_check.__name__}")

    @classmethod
    def validate_scalar_type(cls, scalar: Scalar) -> None:
        if (cls.type_to_check is not None and cls.validate_type_compatibility(scalar.data_type,
                                                                              cls.type_to_check)):
            raise Exception(f"{scalar.name} is not a {cls.type_to_check.__name__}")

    @classmethod
    def validate_dataset_type(cls, dataset: Dataset) -> None:
        if cls.type_to_check is not None:
            for measure in dataset.get_measures():
                if cls.validate_type_compatibility(measure.data_type, cls.type_to_check):
                    raise Exception(
                        f"{measure.role.value} {measure.name} "
                        f"is not a {cls.type_to_check.__name__}")

    @classmethod
    def validate_type_compatibility(cls, left_type: ScalarType, right_type: ScalarType) -> bool:
        # TODO: Implement this method (TypePromotion)
        return False

    @classmethod
    def apply_return_type_dataset(cls, dataset: Dataset) -> None:
        if cls.return_type is not None:
            for measure in dataset.get_measures():
                measure.data_type = cls.return_type
                if (len(dataset.get_measures()) == 1 and
                        cls.return_type in [Boolean_type, Integer_type] and
                        cls.op not in [AND, OR, XOR, NOT]):
                    component = Component(
                        name=COMP_NAME_MAPPING[cls.return_type],
                        data_type=cls.return_type,
                        role=Role.MEASURE,
                        nullable=measure.nullable
                    )
                    dataset.delete_component(measure.name)
                    dataset.add_component(component)
                    if dataset.data is not None:
                        dataset.data.rename(columns={measure.name: component.name}, inplace=True)

    @classmethod
    def apply_return_type(cls, result: Union[DataComponent, Scalar]):
        if cls.return_type is not None:
            result.data_type = cls.return_type


class Binary(Operator):

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        return None if pd.isnull(x) or pd.isnull(y) else cls.py_op(x, y)

    @classmethod
    def apply_operation_two_series(cls,
                                   left_series: Any,
                                   right_series: Any) -> Any:
        if os.getenv("SPARK", False):
            if cls.spark_op is None:
                cls.spark_op = cls.py_op

            nulls = left_series.isnull() | right_series.isnull()
            result = cls.spark_op(left_series, right_series)
            result.loc[nulls] = None
            return result
        return left_series.combine(right_series, cls.op_func)

    @classmethod
    def apply_operation_series_scalar(cls, series: pd.Series, scalar: Any,
                                      series_left: bool) -> Any:
        if series_left:
            return series.map(lambda x: cls.py_op(x, scalar), na_action='ignore')
        else:
            return series.map(lambda x: cls.py_op(scalar, x), na_action='ignore')

    @classmethod
    def dataset_validation(cls, left_operand: Dataset, right_operand: Dataset):
        left_identifiers = left_operand.get_identifiers_names()
        right_identifiers = right_operand.get_identifiers_names()

        if len(left_identifiers) < len(right_identifiers):
            use_right_components = True
        else:
            use_right_components = False

        cls.validate_dataset_type(left_operand)
        cls.validate_dataset_type(right_operand)

        left_measures = sorted(left_operand.get_measures(), key=lambda x: x.name)
        right_measures = sorted(right_operand.get_measures(), key=lambda x: x.name)

        if left_measures != right_measures:
            raise Exception("Measures do not match")

        for left_measure, right_measure in zip(left_measures, right_measures):
            cls.validate_type_compatibility(left_measure.data_type, right_measure.data_type)

        # We do not need anymore these variables
        del left_measures
        del right_measures

        join_keys = list(set(left_identifiers).intersection(right_identifiers))

        # Deleting extra identifiers that we do not need anymore

        base_operand = right_operand if use_right_components else left_operand
        result_components = {component_name: component for component_name, component in
                             base_operand.components.items()
                             if component.role == Role.MEASURE or component.name in join_keys}

        result_dataset = Dataset(name="result", components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset)
        return result_dataset

    @classmethod
    def dataset_scalar_validation(cls, dataset: Dataset, scalar: Scalar):
        cls.validate_dataset_type(dataset)
        cls.validate_scalar_type(scalar)

        result_dataset = Dataset(name="result", components=dataset.components.copy(),
                                 data=None)
        cls.apply_return_type_dataset(result_dataset)
        return result_dataset

    @classmethod
    def scalar_validation(cls, left_operand: Scalar, right_operand: Scalar):
        cls.validate_scalar_type(left_operand)
        cls.validate_scalar_type(right_operand)

        return Scalar(name="result",
                      data_type=cls.type_validation(left_operand.data_type, right_operand.data_type),
                      value=None)

    @classmethod
    def component_validation(cls, left_operand: DataComponent, right_operand: DataComponent):
        cls.validate_component_type(left_operand)
        cls.validate_component_type(right_operand)

        result = DataComponent(name="result",
                               data_type=cls.type_validation(left_operand.data_type, right_operand.data_type),
                               data=None,
                               role=left_operand.role,
                               nullable=(left_operand.nullable or right_operand.nullable))
        cls.apply_return_type(result)
        return result

    @classmethod
    def component_scalar_validation(cls, component: DataComponent, scalar: Scalar):
        cls.validate_component_type(component)
        cls.validate_scalar_type(scalar)

        result = DataComponent(name=component.name,
                               data_type=cls.type_validation(component.data_type, scalar.data_type),
                               data=None, role=component.role,
                               nullable=component.nullable or scalar is None)
        cls.apply_return_type(result)
        return result

    @classmethod
    def dataset_set_validation(cls, dataset: Dataset, scalar_set: ScalarSet):
        cls.validate_dataset_type(dataset)
        for measure in dataset.get_measures():
            cls.validate_type_compatibility(measure.data_type, scalar_set.data_type)

        result_dataset = Dataset(name="result", components=dataset.components.copy(),
                                 data=None)
        cls.apply_return_type_dataset(result_dataset)
        return result_dataset

    @classmethod
    def component_set_validation(cls, component: DataComponent, scalar_set: ScalarSet):
        cls.validate_component_type(component)
        cls.validate_type_compatibility(component.data_type, scalar_set.data_type)

        result = DataComponent(name="result", data_type=cls.type_validation(component.data_type, scalar_set.data_type),
                               data=None,
                               role=Role.MEASURE, nullable=component.nullable)
        cls.apply_return_type(result)
        return result

    @classmethod
    def scalar_set_validation(cls, scalar: Scalar, scalar_set: ScalarSet):
        cls.validate_scalar_type(scalar)
        cls.validate_type_compatibility(scalar.data_type, scalar_set.data_type)
        return Scalar(name="result", data_type=cls.type_validation(scalar.data_type, scalar_set.data_type), value=None)

    @classmethod
    def type_validation(cls, left_type: ScalarType, right_type: ScalarType) -> DataTypes:
        if cls.return_type is not None:
            return cls.return_type

        if left_type is None or right_type is None:
            return None

        left_position = TYPE_MAPPING_POSITION[left_type]
        right_position = TYPE_MAPPING_POSITION[right_type]
        conversion = TYPE_PROMOTION_MATRIX[left_position][right_position]
        if conversion is 'N':
            raise Exception(f"Cannot convert {left_type} to {right_type}")
        if conversion is 'E':
            if cls.op == CAST:
                return right_type
            conversion = TYPE_PROMOTION_MATRIX[right_position][left_position]
            if conversion is 'I':
                return left_type
            raise Exception(f"Cannot convert {left_type} to {right_type} without explicit cast")
        if conversion is 'I':
            return right_type
        return left_type

    @classmethod
    def validate(cls, left_operand, right_operand):
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Dataset):
            return cls.dataset_validation(left_operand, right_operand)
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Scalar):
            return cls.dataset_scalar_validation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Dataset):
            return cls.dataset_scalar_validation(right_operand, left_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Scalar):
            return cls.scalar_validation(left_operand, right_operand)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, DataComponent):
            return cls.component_validation(left_operand, right_operand)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, Scalar):
            return cls.component_scalar_validation(left_operand, right_operand)

        if isinstance(left_operand, Scalar) and isinstance(right_operand, DataComponent):
            return cls.component_scalar_validation(right_operand, left_operand)

        # In operator

        if isinstance(left_operand, Dataset) and isinstance(right_operand, ScalarSet):
            return cls.dataset_set_validation(left_operand, right_operand)

        if isinstance(left_operand, Scalar) and isinstance(right_operand, ScalarSet):
            return cls.scalar_set_validation(left_operand, right_operand)

    @classmethod
    def dataset_evaluation(cls, left_operand: Dataset, right_operand: Dataset):
        result_dataset = cls.dataset_validation(left_operand, right_operand)

        join_keys = result_dataset.get_identifiers_names()
        measure_names = left_operand.get_measures_names()

        # Deleting extra identifiers that we do not need anymore
        for column in left_operand.data.columns:
            if column not in join_keys + measure_names:
                del left_operand.data[column]

        for column in right_operand.data.columns:
            if column not in join_keys + measure_names:
                del right_operand.data[column]

        # Merge the data
        result_data: pd.DataFrame = pd.merge(
            left_operand.data, right_operand.data,
            how='inner', left_on=join_keys, right_on=join_keys)

        for measure_name in left_operand.get_measures_names():
            result_data[measure_name] = cls.apply_operation_two_series(
                result_data[measure_name + '_x'],
                result_data[measure_name + '_y'])
            result_data = result_data.drop([measure_name + '_x', measure_name + '_y'], axis=1)

            if cls.return_type in [Boolean_type, Integer_type] and len(
                    result_dataset.get_measures()) == 1 and cls.op not in [AND, OR, XOR, NOT]:
                result_data[COMP_NAME_MAPPING[cls.return_type]] = result_data[measure_name]
                result_data = result_data.drop(columns=[measure_name])

        result_dataset.data = result_data

        return result_dataset

    @classmethod
    def scalar_evaluation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        result_scalar = cls.scalar_validation(left_operand, right_operand)
        result_scalar.value = cls.op_func(left_operand.value, right_operand.value)
        return result_scalar

    @classmethod
    def dataset_scalar_evaluation(cls, dataset: Dataset, scalar: Scalar,
                                  dataset_left=True) -> Dataset:
        result_dataset = cls.dataset_scalar_validation(dataset, scalar)
        result_data = dataset.data.copy()
        result_dataset.data = result_data

        for measure_name in dataset.get_measures_names():
            result_dataset.data[measure_name] = cls.apply_operation_series_scalar(
                result_data[measure_name], scalar.value, dataset_left)

            if cls.return_type in [Boolean_type, Integer_type] and len(result_dataset.get_measures()) == 1:
                result_data[COMP_NAME_MAPPING[cls.return_type]] = result_data[measure_name]
                result_dataset.data = result_data.drop(columns=[measure_name])
        return result_dataset

    @classmethod
    def component_evaluation(cls, left_operand: DataComponent,
                             right_operand: DataComponent) -> DataComponent:
        result_component = cls.component_validation(left_operand, right_operand)
        result_component.data = cls.apply_operation_two_series(left_operand.data.copy(),
                                                               right_operand.data.copy())
        return result_component

    @classmethod
    def component_scalar_evaluation(cls, component: DataComponent, scalar: Scalar,
                                    component_left: bool) -> DataComponent:
        result_component = cls.component_scalar_validation(component, scalar)
        result_component.data = cls.apply_operation_series_scalar(component.data.copy(),
                                                                  scalar.value, component_left)
        return result_component

    @classmethod
    def dataset_set_evaluation(cls, dataset: Dataset, scalar_set: ScalarSet) -> Dataset:
        result_dataset = cls.dataset_set_validation(dataset, scalar_set)
        result_data = dataset.data.copy()

        for measure_name in dataset.get_measures_names():
            result_data[measure_name] = cls.apply_operation_two_series(dataset.data[measure_name],
                                                                       scalar_set.values)
            if cls.return_type and len(result_dataset.get_measures()) == 1:
                result_data[COMP_NAME_MAPPING[cls.return_type]] = result_data[measure_name]
                result_dataset.data = result_data.drop(columns=[measure_name],  axis=1)

        result_dataset.data = result_data
        return result_dataset

    @classmethod
    def component_set_evaluation(cls, component: DataComponent,
                                 scalar_set: ScalarSet) -> DataComponent:
        result_component = cls.component_set_validation(component, scalar_set)
        result_component.data = cls.apply_operation_two_series(component.data.copy(),
                                                               scalar_set.values)
        return result_component

    @classmethod
    def scalar_set_evaluation(cls, scalar: Scalar, scalar_set: ScalarSet) -> Scalar:
        result_scalar = cls.scalar_set_validation(scalar, scalar_set)
        result_scalar.value = cls.op_func(scalar.value, scalar_set.values)
        return result_scalar

    @classmethod
    def evaluate(cls,
                 left_operand: ALL_MODEL_DATA_TYPES,
                 right_operand: ALL_MODEL_DATA_TYPES) -> ALL_MODEL_DATA_TYPES:
        """
        Evaluate the operation (based on validation output)
        :param left_operand: The left operand
        :param right_operand: The right operand
        :return: The result of the operation
        """
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Dataset):
            return cls.dataset_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Scalar):
            return cls.scalar_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Scalar):
            return cls.dataset_scalar_evaluation(left_operand, right_operand, dataset_left=True)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Dataset):
            return cls.dataset_scalar_evaluation(right_operand, left_operand, dataset_left=False)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, DataComponent):
            return cls.component_evaluation(left_operand, right_operand)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, Scalar):
            return cls.component_scalar_evaluation(left_operand, right_operand, component_left=True)

        if isinstance(left_operand, Scalar) and isinstance(right_operand, DataComponent):
            return cls.component_scalar_evaluation(right_operand, left_operand,
                                                   component_left=False)

        if isinstance(left_operand, Dataset) and isinstance(right_operand, ScalarSet):
            return cls.dataset_set_evaluation(left_operand, right_operand)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, ScalarSet):
            return cls.component_set_evaluation(left_operand, right_operand)

        if isinstance(left_operand, Scalar) and isinstance(right_operand, ScalarSet):
            return cls.scalar_set_evaluation(left_operand, right_operand)


class Unary(Operator):

    @classmethod
    def op_func(cls, x: Any) -> Any:
        return None if pd.isnull(x) else cls.py_op(x)

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        return series.map(cls.op_func, na_action='ignore')

    @classmethod
    def dataset_validation(cls, operand: Dataset):
        cls.validate_dataset_type(operand)

        result_dataset = Dataset(name="result", components=operand.components.copy(), data=None)
        cls.apply_return_type_dataset(result_dataset)
        return result_dataset

    @classmethod
    def scalar_validation(cls, operand: Scalar):
        cls.validate_scalar_type(operand)
        result = Scalar(name="result", data_type=operand.data_type, value=None)
        cls.apply_return_type(result)
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent):
        cls.validate_component_type(operand)
        result = DataComponent(name="result", data_type=operand.data_type, data=None,
                               role=operand.role, nullable=operand.nullable)
        cls.apply_return_type(result)
        return result

    @classmethod
    def validate(cls, operand: ALL_MODEL_DATA_TYPES):
        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand)
        if isinstance(operand, Scalar):
            return cls.scalar_validation(operand)
        if isinstance(operand, DataComponent):
            return cls.component_validation(operand)

    @classmethod
    def evaluate(cls, operand: ALL_MODEL_DATA_TYPES) -> ALL_MODEL_DATA_TYPES:

        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand)

    @classmethod
    def dataset_evaluation(cls, operand: Dataset):
        result_dataset = cls.dataset_validation(operand)
        result_data = operand.data.copy()
        for measure_name in operand.get_measures_names():
            result_data[measure_name] = cls.apply_operation_component(result_data[measure_name])

            if (cls.return_type in [Boolean_type, Integer_type] and
                    len(result_dataset.get_measures()) == 1 and
                    cls.op not in [AND, OR, XOR, NOT]):
                result_data[COMP_NAME_MAPPING[cls.return_type]] = result_data[measure_name]
                result_data = result_data.drop(columns=[measure_name])

        result_dataset.data = result_data
        return result_dataset

    @classmethod
    def scalar_evaluation(cls, operand: Scalar):
        result_scalar = cls.scalar_validation(operand)
        result_scalar.value = cls.op_func(operand.value)
        return result_scalar

    @classmethod
    def component_evaluation(cls, operand: DataComponent):
        result_component = cls.component_validation(operand)
        result_component.data = cls.apply_operation_component(operand.data.copy())
        return result_component

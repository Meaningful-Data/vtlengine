import os
from typing import Any, Union

from DataTypes import ScalarType

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Dataset, Role, Scalar, DataComponent

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]


class Operator:
    """Superclass for all operators"""
    op = None
    py_op = None
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
    def validate_type_compatibility(cls, left_type: ScalarType, right_type: ScalarType):
        # TODO: Implement this method (TypePromotion)
        return False

    @classmethod
    def apply_return_type_dataset(cls, dataset: Dataset) -> None:
        if cls.return_type is not None:
            for measure in dataset.get_measures():
                measure.data_type = cls.return_type

    @classmethod
    def apply_return_type(cls, result: Union[DataComponent, Scalar]) -> ScalarType:
        if cls.return_type is not None:
            return cls.return_type
        return result.data_type


class Binary(Operator):

    @classmethod
    def apply_operation_component(cls,
                                  left_series: Any,
                                  right_series: Any) -> Any:
        return cls.py_op(left_series, right_series)

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

        result_dataset = Dataset(name="result", components=dataset.components, data=None)
        cls.apply_return_type_dataset(result_dataset)
        return result_dataset

    @classmethod
    def scalar_validation(cls, left_operand: Scalar, right_operand: Scalar):
        cls.validate_scalar_type(left_operand)
        cls.validate_scalar_type(right_operand)

        return Scalar(name="result", data_type=cls.return_type, value=None)

    @classmethod
    def component_validation(cls, left_operand: DataComponent, right_operand: DataComponent):
        cls.validate_component_type(left_operand)
        cls.validate_component_type(right_operand)

        return DataComponent(name="result", data_type=cls.apply_return_type(left_operand),
                             data=None, role=Role.MEASURE, nullable=(left_operand.nullable or
                                                                     right_operand.nullable))

    @classmethod
    def component_scalar_validation(cls, component: DataComponent, scalar: Scalar):
        cls.validate_component_type(component)
        cls.validate_scalar_type(scalar)

        return DataComponent(name=component.name, data_type=cls.apply_return_type(component),
                             data=None, role=component.role,
                             nullable=component.nullable or scalar is None)

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

    @classmethod
    def dataset_evaluation(cls, left_operand: Dataset, right_operand: Dataset):
        result_dataset = cls.dataset_validation(left_operand, right_operand)

        join_keys = result_dataset.get_identifiers_names()
        measure_names = result_dataset.get_measures_names()

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

        for measure_name in result_dataset.get_measures_names():
            result_data[measure_name] = cls.apply_operation_component(
                result_data[measure_name + '_x'],
                result_data[measure_name + '_y'])
            result_data = result_data.drop([measure_name + '_x', measure_name + '_y'], axis=1)

        result_dataset.data = result_data

        return result_dataset

    @classmethod
    def scalar_evaluation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        result_scalar = cls.scalar_validation(left_operand, right_operand)
        result_scalar.value = cls.py_op(left_operand.value, right_operand.value)
        return result_scalar

    @classmethod
    def dataset_scalar_evaluation(cls, dataset: Dataset, scalar: Scalar,
                                  dataset_left=True) -> Dataset:
        result_dataset = cls.dataset_scalar_validation(dataset, scalar)
        result_data = dataset.data.copy()
        result_dataset.data = result_data

        for measure_name in result_dataset.get_measures_names():
            if dataset_left:
                result_data[measure_name] = cls.py_op(dataset.data[measure_name], scalar.value)
            else:
                result_data[measure_name] = cls.py_op(scalar.value, dataset.data[measure_name])

        return result_dataset

    @classmethod
    def component_evaluation(cls, left_operand: DataComponent,
                             right_operand: DataComponent) -> DataComponent:
        result_component = cls.component_validation(left_operand, right_operand)
        result_component.data = cls.apply_operation_component(left_operand.data.copy(),
                                                              right_operand.data.copy())
        return result_component

    @classmethod
    def component_scalar_evaluation(cls, component: DataComponent, scalar: Scalar) -> DataComponent:
        result_component = cls.component_scalar_validation(component, scalar)
        result_component.data = cls.apply_operation_component(component.data.copy(), scalar.value)
        return result_component

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
            return cls.component_scalar_evaluation(left_operand, right_operand)

        if isinstance(left_operand, Scalar) and isinstance(right_operand, DataComponent):
            return cls.component_scalar_evaluation(right_operand, left_operand)


class Unary(Operator):

    @classmethod
    def apply_operation_component(cls, series: Any) -> Any:
        """Applies the operation to a component"""
        return cls.py_op(series)

    @classmethod
    def dataset_validation(cls, operand: Dataset):
        cls.validate_dataset_type(operand)
        result_components = operand.components

        result_dataset = Dataset(name="result", components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset)
        return result_dataset

    @classmethod
    def scalar_validation(cls, operand: Scalar):
        cls.validate_scalar_type(operand)
        return Scalar(name="result", data_type=cls.apply_return_type(operand), value=None)

    @classmethod
    def component_validation(cls, operand: DataComponent):
        cls.validate_component_type(operand)
        return DataComponent(name="result", data_type=cls.apply_return_type(operand), data=None)

    @classmethod
    def validate(cls, operand: ALL_MODEL_DATA_TYPES):
        if isinstance(operand, Dataset):
            cls.dataset_validation(operand)
        if isinstance(operand, Scalar):
            cls.scalar_validation(operand)
        if isinstance(operand, DataComponent):
            cls.component_validation(operand)

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
        for measure_name in result_dataset.get_measures_names():
            result_data[measure_name] = cls.apply_operation_component(result_data[measure_name])

        result_dataset.data = result_data
        return result_dataset

    @classmethod
    def scalar_evaluation(cls, operand: Scalar):
        result_scalar = cls.scalar_validation(operand)
        result_scalar.value = cls.py_op(operand.value)
        return result_scalar

    @classmethod
    def component_evaluation(cls, operand: DataComponent):
        result_component = cls.component_validation(operand)
        result_component.data = cls.apply_operation_component(operand.data.copy())
        return result_component

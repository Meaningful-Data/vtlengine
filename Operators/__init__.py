import os
from typing import Union

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Dataset, Role, Scalar, Component, DataComponent


class Binary:
    op = None
    py_op = None
    type_to_check = None
    return_type = None

    @classmethod
    def apply_operation_component(cls, left_series, right_series):
        return cls.py_op(left_series, right_series)

    @classmethod
    def dataset_evaluation(cls, left_operand: Dataset, right_operand: Dataset):
        left_identifiers = left_operand.get_identifiers_names()
        right_identifiers = right_operand.get_identifiers_names()

        if len(left_identifiers) < len(right_identifiers):
            use_right_components = True
        else:
            use_right_components = False

        left_measures = sorted(left_operand.get_measures_names())
        right_measures = sorted(right_operand.get_measures_names())

        if left_measures != right_measures:
            raise Exception("Measures do not match")

        measures = left_measures
        # We do not need anymore this variables
        del left_measures
        del right_measures

        join_keys = list(set(left_identifiers).intersection(right_identifiers))

        # Deleting extra identifiers that we do not need anymore
        for column in left_identifiers:
            if column not in join_keys:
                del left_operand.data[column]

        for column in right_identifiers:
            if column not in join_keys:
                del right_operand.data[column]

        # Merge the data
        result_data: pd.DataFrame = pd.merge(
            left_operand.data, right_operand.data,
            how='inner', left_on=join_keys, right_on=join_keys)

        for measure_name in measures:
            result_data[measure_name] = cls.apply_operation_component(result_data[measure_name + '_x'],
                                                                      result_data[measure_name + '_y'])
            result_data = result_data.drop([measure_name + '_x', measure_name + '_y'], axis=1)

        base_operand = right_operand if use_right_components else left_operand
        result_components = {component_name: component for component_name, component in base_operand.components.items()
                             if component.role == Role.MEASURE or component.name in join_keys}

        return Dataset(name="result", components=result_components, data=result_data)

    @classmethod
    def scalar_evaluation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        return Scalar(name="result", value=cls.py_op(left_operand.value, right_operand.value))

    @classmethod
    def dataset_scalar_evaluation(cls, dataset: Dataset, scalar: Scalar) -> Dataset:
        measures = dataset.get_measures_names()
        result_data = dataset.data.copy()

        for measure_name in measures:
            result_data[measure_name] = cls.py_op(result_data[measure_name], scalar.value)

        result_components = dataset.components

        return Dataset(name="result", components=result_components, data=result_data)

    @classmethod
    def component_evaluation(cls, left_operand: DataComponent, right_operand: DataComponent) -> DataComponent:
        return DataComponent(name="result", data=cls.apply_operation_component(left_operand.data.copy(),
                                                                               right_operand.data.copy()),
                             data_type=cls.return_type)

    @classmethod
    def component_scalar_evaluation(cls, component: DataComponent, scalar: Scalar) -> DataComponent:
        return DataComponent(name="result", data=cls.py_op(component.data.copy(), scalar.value),
                             data_type=cls.return_type)

    @classmethod
    def evaluate(cls, left_operand: Union[Dataset, Scalar, Component],
                 right_operand: [Dataset, Component, Scalar]) -> Union[Dataset, DataComponent, Scalar]:
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Dataset):
            return cls.dataset_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Scalar):
            return cls.scalar_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Dataset) and isinstance(right_operand, Scalar):
            return cls.dataset_scalar_evaluation(left_operand, right_operand)
        if isinstance(left_operand, Scalar) and isinstance(right_operand, Dataset):
            return cls.dataset_scalar_evaluation(right_operand, left_operand)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, DataComponent):
            return cls.component_evaluation(left_operand, right_operand)

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, Scalar):
            return cls.component_scalar_evaluation(left_operand, right_operand)

        if isinstance(left_operand, Scalar) and isinstance(right_operand, DataComponent):
            return cls.component_scalar_evaluation(right_operand, left_operand)


class Unary:
    op = None
    py_op = None
    type_to_check = None
    return_type = None

    @classmethod
    def apply_operation_component(cls, series):
        return cls.py_op(series)

    @classmethod
    def evaluate(cls, operand: Union[Dataset, Scalar, Component]) -> Union[Dataset, DataComponent, Scalar]:

        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand)

        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand)

        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand)

    @classmethod
    def dataset_evaluation(cls, operand: Dataset):
        measures = operand.get_measures_names()
        result_data = operand.data.copy()

        for measure_name in measures:
            result_data[measure_name] = cls.apply_operation_component(result_data[measure_name])

        result_components = operand.components

        return Dataset(name="result", components=result_components, data=result_data)

    @classmethod
    def scalar_evaluation(cls, operand: Scalar):
        return Scalar(name="result", value=cls.py_op(operand.value))

    @classmethod
    def component_evaluation(cls, operand: DataComponent):
        return DataComponent(name="result", data=cls.apply_operation_component(operand.data.copy()),
                             data_type=cls.return_type)

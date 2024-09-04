import os
from copy import copy
from typing import Any, Union

from AST.Grammar.tokens import CEIL, FLOOR, ROUND, EQ, NEQ, GT, GTE, LT, LTE
from DataTypes import COMP_NAME_MAPPING, ScalarType, \
    binary_implicit_promotion, check_binary_implicit_promotion, check_unary_implicit_promotion, \
    unary_implicit_promotion
from DataTypes.TimeHandling import TimeIntervalHandler, TimePeriodHandler, DURATION_MAPPING

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Component, Dataset, Role, Scalar, DataComponent, ScalarSet

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]

# This allows changing the data type of the Measure in the result Data Set
# when the operator is applied to mono-measure Data Sets.
# TODO: Check if there are more operators that allow this
MONOMEASURE_CHANGED_ALLOWED = [CEIL, FLOOR, ROUND]
BINARY_COMPARISON_OPERATORS = [EQ, NEQ, GT, GTE, LT, LTE]


class Operator:
    """Superclass for all operators"""
    op = None
    py_op = None
    spark_op = None
    type_to_check = None
    return_type = None

    @classmethod
    def cast_time_types(cls, data_type: ScalarType, series: pd.Series) -> pd.Series:
        if cls.op not in BINARY_COMPARISON_OPERATORS:
            return series
        if data_type.__name__ == "TimeInterval":
            series = series.map(lambda x: TimeIntervalHandler.from_iso_format(x),
                                na_action='ignore')
        elif data_type.__name__ == "TimePeriod":
            series = series.map(lambda x: TimePeriodHandler(x),
                                na_action='ignore')
        elif data_type.__name__ == "Duration":
            series = series.map(lambda x: DURATION_MAPPING[x],
                                na_action='ignore')
        return series

    @classmethod
    def modify_measure_column(cls, result: Dataset) -> None:
        """
        If an Operator change the data type of the Variable it is applied to (e.g., from string to number),
        the result Data Set cannot maintain this Variable as it happens in the previous cases,
        because a Variable cannot have different data types in different Data Sets.
        As a consequence, the converted variable cannot follow the same rules described in the sections above and must be replaced,
        in the result Data Set, by another Variable of the proper data type.
        For sake of simplicity, the operators changing the data type are allowed only on mono-measure operand Data Sets, so that the conversion happens on just one Measure.
        A default generic Measure is assigned by default to the result Data Set, depending on the data type of the result (the default Measure Variables are reported in the table below).

        Function used by the evaluate function when a dataset is involved
        """
        if len(result.get_measures()) == 1 and cls.return_type is not None:
            measure_name = result.get_measures_names()[0]
            components = list(result.components.keys())
            columns = list(result.data.columns)
            for column in columns:
                if column not in set(components):
                    result.data[measure_name] = result.data[column]
                    del result.data[column]

    @classmethod
    def validate_dataset_type(cls, dataset: Dataset) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_component_type(cls, component: DataComponent) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_scalar_type(cls, scalar: Scalar) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate(cls, *args):
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def dataset_validation(cls, *args) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def scalar_validation(cls, *args) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def component_validation(cls, *args) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_type_compatibility(cls, *args) -> bool:
        if len(args) == 1:
            operand = args[0]
            return check_unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)
        if len(args) == 2:
            left, right = args
            return check_binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def type_validation(cls, *args) -> ScalarType:
        if len(args) == 1:
            operand = args[0]
            return unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)
        if len(args) == 2:
            left, right = args
            return binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def apply_return_type_dataset(cls, *args) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def apply_return_type(cls, *args):
        raise Exception("Method should be implemented by inheritors")


def _id_type_promotion_join_keys(c_left: Component, c_right: Component, join_key: str,
                                 left_data: pd.DataFrame,
                                 right_data: pd.DataFrame) -> None:
    left_type_name = c_left.data_type.__name__
    right_type_name = c_right.data_type.__name__

    if left_type_name == right_type_name or len(left_data) == 0 or len(right_data) == 0:
        left_data[join_key] = left_data[join_key].astype(object)
        right_data[join_key] = right_data[join_key].astype(object)
        return
    if ((left_type_name == "Integer" and right_type_name == "Number") or
            (left_type_name == "Number" and right_type_name == "Integer")):
        left_data[join_key] = left_data[join_key].map(lambda x: int(float(x)))
        right_data[join_key] = right_data[join_key].map(lambda x: int(float(x)))
    elif left_type_name == "String" and right_type_name in ("Integer", "Number"):
        left_data[join_key] = left_data[join_key].map(lambda x: _handle_str_number(x))
    elif left_type_name in ("Integer", "Number") and right_type_name == "String":
        right_data[join_key] = right_data[join_key].map(lambda x: _handle_str_number(x))
    left_data[join_key] = left_data[join_key].astype(object)
    right_data[join_key] = right_data[join_key].astype(object)


def _handle_str_number(x: Union[str, int, float]) -> Union[int, float]:
    if isinstance(x, int):
        return x
    try:
        x = float(x)
        if x.is_integer():
            return int(x)
        return x
    except ValueError:  # Unable to get to string, return the same value that will not be matched
        return x


class Binary(Operator):

    @classmethod
    def op_func(cls, x: Any, y: Any) -> Any:
        if not pd.isnull(x) and x == "REMOVE_VALUE":
            return "REMOVE_VALUE"
        if pd.isnull(x) or pd.isnull(y):
            return None
        return cls.py_op(x, y)

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
        if scalar is None:
            return pd.Series(None, index=series.index)
        if series_left:
            return series.map(lambda x: cls.py_op(x, scalar), na_action='ignore')
        else:
            return series.map(lambda x: cls.py_op(scalar, x), na_action='ignore')

    @classmethod
    def validate(cls, left_operand: ALL_MODEL_DATA_TYPES, right_operand: ALL_MODEL_DATA_TYPES):
        """
        The main function for validate, applies the implicit promotion (or check it), and
        can do a semantic check too.
        Returns an operand.
        """
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

        if isinstance(left_operand, DataComponent) and isinstance(right_operand, ScalarSet):
            return cls.component_set_validation(left_operand, right_operand)

    @classmethod
    def dataset_validation(cls, left_operand: Dataset, right_operand: Dataset) -> Dataset:
        left_identifiers = left_operand.get_identifiers_names()
        right_identifiers = right_operand.get_identifiers_names()

        use_right_components = len(left_identifiers) < len(right_identifiers)

        left_measures = sorted(left_operand.get_measures(), key=lambda x: x.name)
        right_measures = sorted(right_operand.get_measures(), key=lambda x: x.name)
        left_measures_names = [measure.name for measure in left_measures]
        right_measures_names = [measure.name for measure in right_measures]

        if left_measures_names != right_measures_names:
            raise Exception("Measures do not match")

        for left_measure, right_measure in zip(left_measures, right_measures):
            if not cls.validate_type_compatibility(left_measure.data_type, right_measure.data_type):
                raise Exception(
                    f"{left_measure.name} with type {left_measure.data_type} "
                    f"and {right_measure.name} with type {right_measure.data_type} "
                    f"is not compatible with {cls.op} on datasets "
                    f"{left_operand.name} and {right_operand.name}"
                )

        # We do not need anymore these variables
        del left_measures
        del right_measures
        del left_measures_names
        del right_measures_names

        join_keys = list(set(left_identifiers).intersection(right_identifiers))

        # Deleting extra identifiers that we do not need anymore

        base_operand = right_operand if use_right_components else left_operand
        result_components = {component_name: copy(component) for component_name, component in
                             base_operand.components.items()
                             if component.role in [Role.IDENTIFIER, Role.MEASURE]}

        for comp in [x for x in result_components.values() if x.role == Role.MEASURE]:
            if comp.name in left_operand.components and comp.name in right_operand.components:
                left_comp = left_operand.components[comp.name]
                right_comp = right_operand.components[comp.name]
                comp.nullable = left_comp.nullable or right_comp.nullable

        result_dataset = Dataset(name="result", components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, left_operand, right_operand)
        return result_dataset

    @classmethod
    def dataset_scalar_validation(cls, dataset: Dataset, scalar: Scalar):

        result_components = {comp_name: copy(comp) for comp_name, comp in
                             dataset.components.items() if
                             comp.role in [Role.IDENTIFIER, Role.MEASURE]}

        result_dataset = Dataset(name="result", components=result_components,
                                 data=None)
        cls.apply_return_type_dataset(result_dataset, dataset, scalar)
        return result_dataset

    @classmethod
    def scalar_validation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        if not cls.validate_type_compatibility(left_operand.data_type, right_operand.data_type):
            raise Exception(
                f"{left_operand.name} with type {left_operand.data_type} "
                f"and {right_operand.name} with type {right_operand.data_type} is not compatible with {cls.op}"
            )

        return Scalar(name="result",
                      data_type=cls.type_validation(left_operand.data_type,
                                                    right_operand.data_type),
                      value=None)

    @classmethod
    def component_validation(cls, left_operand: DataComponent,
                             right_operand: DataComponent) -> DataComponent:
        """
        Validates the compatibility between the types of the components and the operator
        :param left_operand: The left component
        :param right_operand: The right component
        :return: The result data type of the validation
        """
        # We can ommite the first validation because we check again in the next line
        if not cls.validate_type_compatibility(left_operand.data_type, right_operand.data_type):
            raise Exception(
                f"{left_operand.name} with type {left_operand.data_type} "
                f"and {right_operand.name} with type {right_operand.data_type} is not compatible with {cls.op}"
            )
        result_data_type = cls.type_validation(left_operand.data_type, right_operand.data_type)

        result = DataComponent(name="result",
                               data_type=result_data_type,
                               data=None,
                               role=left_operand.role,
                               nullable=(left_operand.nullable or right_operand.nullable))

        return result

    @classmethod
    def component_scalar_validation(cls, component: DataComponent, scalar: Scalar):
        if not cls.validate_type_compatibility(component.data_type, scalar.data_type):
            raise Exception(
                f"{component.name} with type {component.data_type} "
                f"and {scalar.name} with type {scalar.data_type} is not compatible with {cls.op}"
            )

        result = DataComponent(name=component.name,
                               data_type=cls.type_validation(component.data_type, scalar.data_type),
                               data=None, role=component.role,
                               nullable=component.nullable or scalar is None)

        return result

    @classmethod
    def dataset_set_validation(cls, dataset: Dataset, scalar_set: ScalarSet) -> Dataset:

        for measure in dataset.get_measures():
            if not cls.validate_type_compatibility(measure.data_type, scalar_set.data_type):
                raise Exception(
                    f"{measure.name} with type {measure.data_type} "
                    f"and scalar_set with type {scalar_set.data_type} is not compatible with {cls.op}"
                )

        result_components = {comp_name: copy(comp) for comp_name, comp in
                             dataset.components.items() if
                             comp.role in [Role.IDENTIFIER, Role.MEASURE]}

        result_dataset = Dataset(name="result", components=result_components,
                                 data=None)
        cls.apply_return_type_dataset(result_dataset, dataset, scalar_set)
        return result_dataset

    @classmethod
    def component_set_validation(cls, component: DataComponent,
                                 scalar_set: ScalarSet) -> DataComponent:

        if not cls.validate_type_compatibility(component.data_type, scalar_set.data_type):
            raise Exception(
                f"{component.name} with type {component.data_type} "
                f"and scalar_set with type {scalar_set.data_type} is not compatible with {cls.op}"
            )

        result = DataComponent(name="result", data_type=cls.type_validation(component.data_type,
                                                                            scalar_set.data_type),
                               data=None,
                               role=Role.MEASURE, nullable=component.nullable)

        return result

    @classmethod
    def scalar_set_validation(cls, scalar: Scalar, scalar_set: ScalarSet):
        if not cls.validate_type_compatibility(scalar.data_type, scalar_set.data_type):
            raise Exception(
                f"{scalar.name} with type {scalar.data_type} "
                f"and scalar_set with type {scalar_set.data_type} is not compatible with {cls.op}"
            )
        return Scalar(name="result",
                      data_type=cls.type_validation(scalar.data_type, scalar_set.data_type),
                      value=None)

    # The following class method implements the type promotion
    @classmethod
    def type_validation(cls, left_type: ScalarType, right_type: ScalarType) -> ScalarType:
        """
        Validates the compatibility between the types of the operands and the operator
        and give us the result ScalarType of the promotion
        (implicit type promotion : binary_implicit_type_promotion)

        :param left_type: The left operand data type
        :param right_type: The right operand data type

        :return: result ScalarType or exception
        """
        return binary_implicit_promotion(left_type, right_type, cls.type_to_check, cls.return_type)

    # The following class method checks the type promotion
    @classmethod
    def validate_type_compatibility(cls, left: ScalarType, right: ScalarType) -> bool:
        """
        Validates the compatibility between the types of the operands and the operator
        (implicit type promotion : check_binary_implicit_type_promotion)

        :param left: The left operand
        :param right: The right operand

        :return: True if the types are compatible, False otherwise
        """
        return check_binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)

    @classmethod
    def apply_return_type_dataset(
            cls, result_dataset: Dataset, left_operand: Dataset,
            right_operand: Union[Dataset, Scalar, ScalarSet]
    ) -> None:
        """
        Used in dataset's validation.
        Changes the result dataset and give us his final form (#TODO: write this explanation in a better way)
        """
        changed_allowed = cls.op in MONOMEASURE_CHANGED_ALLOWED
        is_mono_measure = len(result_dataset.get_measures()) == 1
        for measure in result_dataset.get_measures():
            left_type = left_operand.get_component(measure.name).data_type
            if isinstance(right_operand, (ScalarSet, Scalar)):
                right_type = right_operand.data_type
            else:
                right_type = right_operand.get_component(measure.name).data_type

            result_data_type = cls.type_validation(left_type, right_type)
            if is_mono_measure and left_type.promotion_changed_type(result_data_type):
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
            elif (changed_allowed is False and
                  is_mono_measure is False and
                  left_type.promotion_changed_type(result_data_type)
            ):
                raise Exception("Operation not allowed for multimeasure datasets")
            else:
                measure.data_type = result_data_type

    @classmethod
    def dataset_evaluation(cls, left_operand: Dataset, right_operand: Dataset):
        result_dataset = cls.dataset_validation(left_operand, right_operand)

        use_right_as_base = False
        if len(left_operand.get_identifiers_names()) < len(right_operand.get_identifiers_names()):
            use_right_as_base = True
            base_operand_data = right_operand.data
            other_operand_data = left_operand.data
        else:
            base_operand_data = left_operand.data
            other_operand_data = right_operand.data

        join_keys = list(set(left_operand.get_identifiers_names()).intersection(
            right_operand.get_identifiers_names()))

        for join_key in join_keys:
            _id_type_promotion_join_keys(left_operand.get_component(join_key),
                                         right_operand.get_component(join_key),
                                         join_key, base_operand_data, other_operand_data)

        try:
            # Merge the data
            result_data: pd.DataFrame = pd.merge(
                base_operand_data, other_operand_data,
                how='inner', on=join_keys,
                suffixes=('_x', '_y'))
        except ValueError as e:
            raise Exception(f"Error merging datasets on Binary Operator: {str(e)}")

        # Measures are the same, using left operand measures names
        for measure in left_operand.get_measures():
            result_data[measure.name + '_x'] = cls.cast_time_types(measure.data_type,
                                                                   result_data[measure.name + '_x'])
            result_data[measure.name + '_y'] = cls.cast_time_types(measure.data_type,
                                                                   result_data[measure.name + '_y'])
            if use_right_as_base:
                result_data[measure.name] = cls.apply_operation_two_series(
                    result_data[measure.name + '_y'],
                    result_data[measure.name + '_x'])
            else:
                result_data[measure.name] = cls.apply_operation_two_series(
                    result_data[measure.name + '_x'],
                    result_data[measure.name + '_y'])
            result_data = result_data.drop([measure.name + '_x', measure.name + '_y'], axis=1)

        # Delete attributes from the result data
        attributes = list(
            set(left_operand.get_attributes_names()).union(right_operand.get_attributes_names()))
        for att in attributes:
            if att in result_data.columns:
                result_data = result_data.drop(att, axis=1)
            if att + '_x' in result_data.columns:
                result_data = result_data.drop(att + '_x', axis=1)
            if att + '_y' in result_data.columns:
                result_data = result_data.drop(att + '_y', axis=1)

        result_dataset.data = result_data
        cls.modify_measure_column(result_dataset)

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

        for measure in dataset.get_measures():
            measure_data = cls.cast_time_types(measure.data_type, result_data[measure.name].copy())
            result_dataset.data[measure.name] = cls.apply_operation_series_scalar(
                measure_data, scalar.value, dataset_left)

        result_dataset.data = result_data
        cols_to_keep = dataset.get_identifiers_names() + dataset.get_measures_names()
        result_dataset.data = result_dataset.data[cols_to_keep]
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def component_evaluation(cls, left_operand: DataComponent,
                             right_operand: DataComponent) -> DataComponent:
        result_component = cls.component_validation(left_operand, right_operand)
        left_data = cls.cast_time_types(left_operand.data_type, left_operand.data.copy())
        right_data = cls.cast_time_types(right_operand.data_type, right_operand.data.copy())
        result_component.data = cls.apply_operation_two_series(left_data, right_data)
        return result_component

    @classmethod
    def component_scalar_evaluation(cls, component: DataComponent, scalar: Scalar,
                                    component_left: bool) -> DataComponent:
        result_component = cls.component_scalar_validation(component, scalar)
        comp_data = cls.cast_time_types(component.data_type, component.data.copy())
        result_component.data = cls.apply_operation_series_scalar(comp_data,
                                                                  scalar.value, component_left)
        return result_component

    @classmethod
    def dataset_set_evaluation(cls, dataset: Dataset, scalar_set: ScalarSet) -> Dataset:
        result_dataset = cls.dataset_set_validation(dataset, scalar_set)
        result_data = dataset.data.copy()

        for measure_name in dataset.get_measures_names():
            result_data[measure_name] = cls.apply_operation_two_series(dataset.data[measure_name],
                                                                       scalar_set.values)

        cols_to_keep = dataset.get_identifiers_names() + dataset.get_measures_names()
        result_dataset.data = result_data[cols_to_keep]
        cls.modify_measure_column(result_dataset)

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
        return series.map(cls.py_op, na_action='ignore')

    @classmethod
    def validate(cls, operand: ALL_MODEL_DATA_TYPES):
        """
        The main function for validate, applies the implicit promotion (or check it), and
        can do a semantic check too.
        Returns an operand.
        """
        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand)
        elif isinstance(operand, Scalar):
            return cls.scalar_validation(operand)

    @classmethod
    def dataset_validation(cls, operand: Dataset) -> Dataset:
        cls.validate_dataset_type(operand)
        result_components = {comp_name: copy(comp) for comp_name, comp in
                             operand.components.items() if
                             comp.role in [Role.IDENTIFIER, Role.MEASURE]}

        result_dataset = Dataset(name="result", components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, operand)
        return result_dataset

    @classmethod
    def scalar_validation(cls, operand: Scalar) -> Scalar:
        if not cls.validate_type_compatibility(operand.data_type):
            raise Exception(
                f"{operand.name} with type {operand.data_type} is not compatible with {cls.op}")
        result_type = cls.type_validation(operand.data_type)
        result = Scalar(name="result", data_type=result_type, value=None)
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent) -> DataComponent:
        cls.validate_type_compatibility(operand.data_type)
        result_type = cls.type_validation(operand.data_type)
        result = DataComponent(name="result", data_type=result_type, data=None,
                               role=operand.role, nullable=operand.nullable)
        return result

    # The following class method implements the type promotion
    @classmethod
    def type_validation(cls, operand: ScalarType) -> ScalarType:
        return unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)

    # The following class method checks the type promotion
    @classmethod
    def validate_type_compatibility(cls, operand: ScalarType) -> bool:
        return check_unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)

    @classmethod
    def validate_dataset_type(cls, dataset: Dataset) -> None:
        if cls.type_to_check is not None:
            for measure in dataset.get_measures():
                if not cls.validate_type_compatibility(measure.data_type):
                    raise Exception(
                        f"{measure.role.value} {measure.name} can't be promoted to {cls.type_to_check}"
                    )

    @classmethod
    def validate_scalar_type(cls, scalar: Scalar) -> None:
        if (cls.type_to_check is not None and not cls.validate_type_compatibility(
                scalar.data_type)):
            raise Exception(f"{scalar.name} can't be promoted to {cls.type_to_check}")

    @classmethod
    def apply_return_type_dataset(cls, result_dataset: Dataset, operand: Dataset) -> None:
        changed_allowed = cls.op in MONOMEASURE_CHANGED_ALLOWED
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
            elif changed_allowed is False and is_mono_measure is False and operand_type.promotion_changed_type(
                    result_data_type):
                raise Exception("Operation not allowed for multimeasure datasets")
            else:
                measure.data_type = result_data_type

    @classmethod
    def evaluate(cls, operand: ALL_MODEL_DATA_TYPES) -> ALL_MODEL_DATA_TYPES:

        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand)

    @classmethod
    def dataset_evaluation(cls, operand: Dataset) -> Dataset:
        result_dataset = cls.dataset_validation(operand)
        result_data = operand.data.copy()
        for measure_name in operand.get_measures_names():
            result_data[measure_name] = cls.apply_operation_component(result_data[measure_name])

        cols_to_keep = operand.get_identifiers_names() + operand.get_measures_names()
        result_data = result_data[cols_to_keep]

        result_dataset.data = result_data
        cls.modify_measure_column(result_dataset)
        return result_dataset

    @classmethod
    def scalar_evaluation(cls, operand: Scalar) -> Scalar:
        result_scalar = cls.scalar_validation(operand)
        result_scalar.value = cls.op_func(operand.value)
        return result_scalar

    @classmethod
    def component_evaluation(cls, operand: DataComponent) -> DataComponent:
        result_component = cls.component_validation(operand)
        result_component.data = cls.apply_operation_component(operand.data.copy())
        return result_component

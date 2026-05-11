import math
import operator
from copy import copy
from typing import Any, Optional, Union

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import (
    CHARSET_MATCH,
    EQ,
    GT,
    GTE,
    IN,
    ISNULL,
    LT,
    LTE,
    NEQ,
    NOT_IN,
)
from vtlengine.DataTypes import COMP_NAME_MAPPING, Boolean, String
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar, ScalarSet
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


class Unary(Operator.Unary):
    """
    Unary comparison operator. It returns a boolean.
    """

    return_type = Boolean


class IsNull(Unary):
    """
    Class that allows to perform the isnull comparison operator.
    It has different class methods to allow performing the operation with different datatypes.
    """

    op = ISNULL

    @staticmethod
    def py_op(x: Any) -> bool:
        return x is None or (isinstance(x, float) and math.isnan(x))

    @classmethod
    def dataset_validation(cls, operand: Dataset) -> Dataset:
        result = super().dataset_validation(operand)
        for measure in result.get_measures():
            measure.nullable = False
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent) -> DataComponent:
        result = super().component_validation(operand)
        result.nullable = False
        return result


class Binary(Operator.Binary):
    """
    Binary comparison operator. It returns a boolean.
    """

    return_type = Boolean

    @classmethod
    def apply_return_type_dataset(
        cls,
        result_dataset: Dataset,
        left_operand: Dataset,
        right_operand: Union[Dataset, Scalar, ScalarSet],
    ) -> None:
        super().apply_return_type_dataset(result_dataset, left_operand, right_operand)
        is_mono_measure = len(result_dataset.get_measures()) == 1
        if is_mono_measure:
            measure = result_dataset.get_measures()[0]
            component = Component(
                name=COMP_NAME_MAPPING[Boolean],
                data_type=Boolean,
                role=Role.MEASURE,
                nullable=measure.nullable,
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


class NotIn(Binary):
    op = NOT_IN


class Match(Binary):
    op = CHARSET_MATCH
    type_to_check = String


class Between(Operator.Operator):
    return_type = Boolean
    """
    This comparison operator has the following class methods.

    Class methods:
        op_function: Sets the data to be manipulated.
        apply_operation_component: Returns a pandas dataframe with the operation,

        considering each component with the schema of op_function.

        apply_return_type_dataset: Because the result must be a boolean,
        this function evaluates if the measure is actually a boolean one.
    """

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
                    nullable=measure.nullable,
                )
                result_dataset.delete_component(measure.name)
                result_dataset.add_component(component)
                if result_dataset.data is not None:
                    result_dataset.data.rename(columns={measure.name: component.name}, inplace=True)
            elif is_mono_measure is False and operand_type.promotion_changed_type(result_data_type):
                raise SemanticError("1-1-1-4", op=cls.op)
            else:
                measure.data_type = result_data_type

    @classmethod
    def validate(
        cls,
        operand: Union[Dataset, DataComponent, Scalar],
        from_: Union[DataComponent, Scalar],
        to: Union[DataComponent, Scalar],
    ) -> Any:
        result: Union[Dataset, DataComponent, Scalar]
        if isinstance(operand, Dataset):
            if len(operand.get_measures()) == 0:
                raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
            result_components = {
                comp_name: copy(comp)
                for comp_name, comp in operand.components.items()
                if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
            }
            result = Dataset(name=operand.name, components=result_components, data=None)
        elif isinstance(operand, DataComponent):
            result = DataComponent(
                name=operand.name,
                data=None,
                data_type=cls.return_type,
                role=operand.role,
                nullable=operand.nullable,
            )
        elif isinstance(from_, Scalar) and isinstance(to, Scalar):
            result = Scalar(name=operand.name, value=None, data_type=cls.return_type)
        else:
            # From or To is a DataComponent, or both
            result = DataComponent(
                name=operand.name,
                data=None,
                data_type=cls.return_type,
                role=Role.MEASURE,
            )

        if isinstance(operand, Dataset):
            for measure in operand.get_measures():
                cls.validate_type_compatibility(measure.data_type, from_.data_type)
                cls.validate_type_compatibility(measure.data_type, to.data_type)
                if isinstance(result, Dataset):
                    cls.apply_return_type_dataset(result, operand)
        else:
            cls.validate_type_compatibility(operand.data_type, from_.data_type)
            cls.validate_type_compatibility(operand.data_type, to.data_type)

        return result


class ExistIn(Operator.Operator):
    """
    Class methods:
        validate: Sets the identifiers and check if the left one exists in the right one.
        evaluate: Evaluates if the result data type is actually a boolean.
    """

    op = IN

    # noinspection PyTypeChecker
    @classmethod
    def validate(
        cls, dataset_1: Dataset, dataset_2: Dataset, retain_element: Optional[Boolean]
    ) -> Any:
        dataset_name = VirtualCounter._new_ds_name()
        left_identifiers = dataset_1.get_identifiers_names()
        right_identifiers = dataset_2.get_identifiers_names()

        is_subset_right = set(right_identifiers).issubset(left_identifiers)
        is_subset_left = set(left_identifiers).issubset(right_identifiers)
        if not (is_subset_left or is_subset_right):
            raise ValueError("Datasets must have common identifiers")

        result_components = {comp.name: copy(comp) for comp in dataset_1.get_identifiers()}
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        result_dataset.add_component(
            Component(name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=False)
        )
        return result_dataset

    @staticmethod
    def _check_all_columns(row: Any) -> bool:
        return all(col_value == True for col_value in row)

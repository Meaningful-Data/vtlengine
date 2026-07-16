import re
from copy import copy
from typing import Any, Union

from vtlengine.AST.Grammar.tokens import (
    CEIL,
    FLOOR,
    ROUND,
)
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    binary_implicit_promotion,
    check_binary_implicit_promotion,
    check_unary_implicit_promotion,
    unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar, ScalarSet
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.ViralPropagation import combined_viral_components, require_rules

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]

# This allows changing the data type of the Measure in the result Data Set
# when the operator is applied to mono-measure Data Sets.
# TODO: Check if there are more operators that allow this
MONOMEASURE_CHANGED_ALLOWED = [CEIL, FLOOR, ROUND]


class Operator:
    """Superclass for all operators"""

    op: Any = None
    py_op: Any = None
    type_to_check: Any = None
    return_type: Any = None

    @classmethod
    def validate_dataset_type(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_scalar_type(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate(cls, *args: Any, **kwargs: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def scalar_validation(cls, *args: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def component_validation(cls, *args: Any) -> Any:
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def validate_type_compatibility(cls, *args: Any) -> bool:
        if len(args) == 1:
            operand = args[0]
            return check_unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)
        if len(args) == 2:
            left, right = args
            return check_binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def type_validation(cls, *args: Any) -> Any:
        if len(args) == 1:
            operand = args[0]
            return unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)
        if len(args) == 2:
            left, right = args
            return binary_implicit_promotion(left, right, cls.type_to_check, cls.return_type)
        raise Exception("Method should be implemented by inheritors")

    @classmethod
    def apply_return_type_dataset(cls, *args: Any) -> None:
        raise Exception("Method should be implemented by inheritors")


class Binary(Operator):
    @classmethod
    def validate(cls, *args: Any) -> Any:
        """
        The main function for validate, applies the implicit promotion (or check it), and
        can do a semantic check too.
        Returns an operand.
        """
        left_operand, right_operand = args[0], args[1]

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
        dataset_name = VirtualCounter._new_ds_name()
        left_identifiers = left_operand.get_identifiers_names()
        right_identifiers = right_operand.get_identifiers_names()

        use_right_components = len(left_identifiers) < len(right_identifiers)

        left_measures = sorted(left_operand.get_measures(), key=lambda x: x.name)
        right_measures = sorted(right_operand.get_measures(), key=lambda x: x.name)
        left_measures_names = [measure.name for measure in left_measures]
        right_measures_names = [measure.name for measure in right_measures]

        if left_measures_names != right_measures_names:
            raise SemanticError(
                "1-1-14-1",
                op=cls.op,
                left=left_measures_names,
                right=right_measures_names,
            )
        elif len(left_measures) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=left_operand.name)
        for left_measure, right_measure in zip(left_measures, right_measures):
            cls.type_validation(left_measure.data_type, right_measure.data_type)

        # We do not need anymore these variables
        del left_measures
        del right_measures
        del left_measures_names
        del right_measures_names

        left_ids_set = set(left_identifiers)
        right_ids_set = set(right_identifiers)
        if not left_ids_set or not right_ids_set:
            raise SemanticError("1-2-10", op=cls.op)
        if not (left_ids_set.issubset(right_ids_set) or right_ids_set.issubset(left_ids_set)):
            raise SemanticError(
                "1-2-15",
                op=cls.op,
                left_name=left_operand.name,
                left=sorted(left_ids_set),
                right_name=right_operand.name,
                right=sorted(right_ids_set),
            )

        # Deleting extra identifiers that we do not need anymore

        base_operand = right_operand if use_right_components else left_operand
        other_operand = left_operand if use_right_components else right_operand
        result_components = {
            component_name: copy(component)
            for component_name, component in base_operand.components.items()
            if component.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
        }
        # Also include viral attributes from the other operand
        for comp_name, comp in other_operand.components.items():
            if comp.role == Role.VIRAL_ATTRIBUTE and comp_name not in result_components:
                result_components[comp_name] = copy(comp)

        for comp in [x for x in result_components.values() if x.role == Role.MEASURE]:
            if comp.name in left_operand.components and comp.name in right_operand.components:
                left_comp = left_operand.components[comp.name]
                right_comp = right_operand.components[comp.name]
                comp.nullable = left_comp.nullable or right_comp.nullable

        # Viral attributes present in BOTH operands have their data points merged, so they
        # are combined and require a propagation rule; a viral attribute in a single operand
        # is copied through and needs none (issue #906).
        require_rules(combined_viral_components([left_operand, right_operand]))

        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, left_operand, right_operand)
        return result_dataset

    @classmethod
    def dataset_scalar_validation(cls, dataset: Dataset, scalar: Scalar) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(dataset.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=dataset.name)

        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in dataset.components.items()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
        }
        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, dataset, scalar)
        return result_dataset

    @classmethod
    def scalar_validation(cls, left_operand: Scalar, right_operand: Scalar) -> Scalar:
        if not cls.validate_type_compatibility(left_operand.data_type, right_operand.data_type):
            raise SemanticError(
                "1-1-1-2",
                type_1=left_operand.data_type,
                type_2=right_operand.data_type,
                type_check=cls.type_to_check,
            )
        return Scalar(
            name="result",
            data_type=cls.type_validation(left_operand.data_type, right_operand.data_type),
            value=None,
            nullable=left_operand.nullable or right_operand.nullable,
        )

    @classmethod
    def component_validation(
        cls, left_operand: DataComponent, right_operand: DataComponent
    ) -> DataComponent:
        """
        Validates the compatibility between the types of the components and the operator
        :param left_operand: The left component
        :param right_operand: The right component
        :return: The result data type of the validation
        """
        comp_name = VirtualCounter._new_dc_name()
        result_data_type = cls.type_validation(left_operand.data_type, right_operand.data_type)
        result = DataComponent(
            name=comp_name,
            data_type=result_data_type,
            data=None,
            role=left_operand.role,
            nullable=(left_operand.nullable or right_operand.nullable),
        )

        return result

    @classmethod
    def component_scalar_validation(cls, component: DataComponent, scalar: Scalar) -> DataComponent:
        cls.type_validation(component.data_type, scalar.data_type)
        result = DataComponent(
            name=component.name,
            data_type=cls.type_validation(component.data_type, scalar.data_type),
            data=None,
            role=component.role,
            nullable=component.nullable or scalar is None,
        )
        return result

    @classmethod
    def dataset_set_validation(cls, dataset: Dataset, scalar_set: ScalarSet) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(dataset.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=dataset.name)
        for measure in dataset.get_measures():
            cls.type_validation(measure.data_type, scalar_set.data_type)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in dataset.components.items()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
        }

        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, dataset, scalar_set)
        return result_dataset

    @classmethod
    def component_set_validation(
        cls, component: DataComponent, scalar_set: ScalarSet
    ) -> DataComponent:
        comp_name = VirtualCounter._new_dc_name()
        cls.type_validation(component.data_type, scalar_set.data_type)
        result = DataComponent(
            name=comp_name,
            data_type=cls.type_validation(component.data_type, scalar_set.data_type),
            data=None,
            role=Role.MEASURE,
            nullable=component.nullable,
        )
        return result

    @classmethod
    def scalar_set_validation(cls, scalar: Scalar, scalar_set: ScalarSet) -> Scalar:
        cls.type_validation(scalar.data_type, scalar_set.data_type)
        return Scalar(
            name="result",
            data_type=cls.type_validation(scalar.data_type, scalar_set.data_type),
            value=None,
            nullable=scalar.nullable,
        )

    # The following class method implements the type promotion
    @classmethod
    def type_validation(cls, left_type: Any, right_type: Any) -> Any:
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
    def validate_type_compatibility(cls, left: Any, right: Any) -> bool:
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
        cls, result_dataset: Dataset, left_operand: Any, right_operand: Any
    ) -> None:
        """
        Used in dataset's validation.
        Changes the result dataset and give us his final form
        (#TODO: write this explanation in a better way)
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
                    nullable=measure.nullable,
                )
                result_dataset.delete_component(measure.name)
                result_dataset.add_component(component)
            elif (
                changed_allowed is False
                and is_mono_measure is False
                and left_type.promotion_changed_type(result_data_type)
            ):
                raise SemanticError("1-1-1-4", op=cls.op)
            else:
                measure.data_type = result_data_type


class Unary(Operator):
    @classmethod
    def validate(cls, operand: Any) -> Any:
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
        dataset_name = VirtualCounter._new_ds_name()
        cls.validate_dataset_type(operand)
        if len(operand.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in operand.components.items()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
        }

        result_dataset = Dataset(name=dataset_name, components=result_components, data=None)
        cls.apply_return_type_dataset(result_dataset, operand)
        return result_dataset

    @classmethod
    def scalar_validation(cls, operand: Scalar) -> Scalar:
        result_type = cls.type_validation(operand.data_type)
        result = Scalar(name="result", data_type=result_type, value=None, nullable=operand.nullable)
        return result

    @classmethod
    def component_validation(cls, operand: DataComponent) -> DataComponent:
        comp_name = VirtualCounter._new_dc_name()
        result_type = cls.type_validation(operand.data_type)
        result = DataComponent(
            name=comp_name,
            data_type=result_type,
            data=None,
            role=operand.role,
            nullable=operand.nullable,
        )
        return result

    # The following class method implements the type promotion
    @classmethod
    def type_validation(cls, operand: Any) -> Any:
        return unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)

    # The following class method checks the type promotion
    @classmethod
    def validate_type_compatibility(cls, operand: Any) -> bool:
        return check_unary_implicit_promotion(operand, cls.type_to_check, cls.return_type)

    @classmethod
    def validate_dataset_type(cls, dataset: Dataset) -> None:
        if cls.type_to_check is not None:
            for measure in dataset.get_measures():
                if not cls.validate_type_compatibility(measure.data_type):
                    raise SemanticError(
                        "1-1-1-3",
                        op=cls.op,
                        entity=measure.role.value,
                        name=measure.name,
                        target_type=SCALAR_TYPES_CLASS_REVERSE[cls.type_to_check],
                    )

    @classmethod
    def validate_scalar_type(cls, scalar: Scalar) -> None:
        if cls.type_to_check is not None and not cls.validate_type_compatibility(scalar.data_type):
            raise SemanticError(
                "1-1-1-5",
                op=cls.op,
                name=scalar.name,
                type=SCALAR_TYPES_CLASS_REVERSE[scalar.data_type],
            )

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
                    nullable=measure.nullable,
                )
                result_dataset.delete_component(measure.name)
                result_dataset.add_component(component)
            elif (
                changed_allowed is False
                and is_mono_measure is False
                and operand_type.promotion_changed_type(result_data_type)
            ):
                raise SemanticError("1-1-1-4", op=cls.op)
            else:
                measure.data_type = result_data_type

    @classmethod
    def to_days(cls, value: str) -> int:
        iso8601_duration_pattern = r"^P((\d+Y)?(\d+M)?(\d+D)?)$"
        match = re.match(iso8601_duration_pattern, value)

        years = 0
        months = 0
        days = 0

        years_str = match.group(2)  # type: ignore[union-attr]
        months_str = match.group(3)  # type: ignore[union-attr]
        days_str = match.group(4)  # type: ignore[union-attr]
        if years_str:
            years = int(years_str[:-1])
        if months_str:
            months = int(months_str[:-1])
        if days_str:
            days = int(days_str[:-1])
        total_days = years * 365 + months * 30 + days
        return int(total_days)

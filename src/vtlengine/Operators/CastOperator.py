from copy import copy
from typing import Any, Optional, Type, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import CAST
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING,
    EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING,
    IMPLICIT_TYPE_PROMOTION_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    Date,
    Duration,
    Number,
    ScalarType,
    String,
    TimeInterval,
    TimePeriod,
)
from vtlengine.DataTypes.TimeHandling import str_period_to_date
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar

duration_mapping = {"A": 6, "S": 5, "Q": 4, "M": 3, "W": 2, "D": 1}

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]


class Cast(Operator.Unary):
    op = CAST

    # CASTS VALUES
    # Converts the value from one type to another in a way that is according to the mask
    @classmethod
    def cast_string_to_number(cls, value: Any, mask: str) -> Any:
        """
        This method casts a string to a number, according to the mask.

        """

        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def cast_string_to_date(cls, value: Any, mask: str) -> Any:
        """
        This method casts a string to a number, according to the mask.

        """

        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def cast_string_to_duration(cls, value: Any, mask: str) -> Any:
        """
        This method casts a string to a duration, according to the mask.

        """

        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def cast_string_to_time_period(cls, value: Any, mask: str) -> Any:
        """
        This method casts a string to a time period, according to the mask.

        """

        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def cast_string_to_time(cls, value: Any, mask: str) -> Any:
        """
        This method casts a string to a time, according to the mask.

        """

        raise NotImplementedError("How this cast should be implemented is not yet defined.")

    #
    # @classmethod
    # def cast_date_to_string(cls, value: Any, mask: str) -> Any:
    #     """ """
    #     return NotImplementedError("How this cast should be implemented is not yet defined.")
    #
    # @classmethod
    # def cast_duration_to_string(cls, value: Any, mask: str) -> Any:
    #     """ """
    #     return NotImplementedError("How this cast should be implemented is not yet defined.")
    #
    # @classmethod
    # def cast_time_to_string(cls, value: Any, mask: str) -> Any:
    #     """ """
    #     return NotImplementedError("How this cast should be implemented is not yet defined.")

    @classmethod
    def cast_time_period_to_date(cls, value: Any, mask_value: str) -> Any:
        """ """

        start = mask_value == "START"
        return str_period_to_date(value, start)

    invalid_mask_message = "At op {op}: Invalid mask to cast from type {type_1} to {type_2}."

    @classmethod
    def check_mask_value(
        cls, from_type: Type[ScalarType], to_type: Type[ScalarType], mask_value: str
    ) -> None:
        """
        This method checks if the mask value is valid for the cast operation.
        """

        if from_type == TimeInterval and to_type == String:
            return cls.check_mask_value_from_time_to_string(mask_value)
        # from = Date
        if from_type == Date and to_type == String:
            return cls.check_mask_value_from_date_to_string(mask_value)
        # from = Time_Period
        if from_type == TimePeriod and to_type == Date:
            return cls.check_mask_value_from_time_period_to_date(mask_value)
        # from = String
        if from_type == String and to_type == Number:
            return cls.check_mask_value_from_string_to_number(mask_value)
        if from_type == String and to_type == TimeInterval:
            return cls.check_mask_value_from_string_to_time(mask_value)
        if from_type == String and to_type == Date:
            return cls.check_mask_value_from_string_to_date(mask_value)
        if from_type == String and to_type == TimePeriod:
            return cls.check_mask_value_from_string_to_time_period(mask_value)
        if from_type == String and to_type == Duration:
            return cls.check_mask_value_from_string_to_duration(mask_value)
        # from = Duration
        if from_type == Duration and to_type == String:
            return cls.check_mask_value_from_duration_to_string(mask_value)
        raise SemanticError(
            "1-1-5-5",
            op=cls.op,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            mask_value=mask_value,
        )

    @classmethod
    def check_mask_value_from_time_period_to_date(cls, mask_value: str) -> None:
        if mask_value not in ["START", "END"]:
            raise SemanticError("1-1-5-4", op=cls.op, type_1="Time_Period", type_2="Date")

    @classmethod
    def check_mask_value_from_time_to_string(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_date_to_string(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_string_to_number(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_string_to_time(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_string_to_date(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_string_to_time_period(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_string_to_duration(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_mask_value_from_duration_to_string(cls, *args: Any) -> None:
        raise NotImplementedError("How this mask should be implemented is not yet defined.")

    @classmethod
    def check_cast(
        cls,
        from_type: Type[ScalarType],
        to_type: Type[ScalarType],
        mask_value: Optional[str],
    ) -> None:
        if mask_value is not None:
            cls.check_with_mask(from_type, to_type, mask_value)
        else:
            cls.check_without_mask(from_type, to_type)

    @classmethod
    def check_with_mask(
        cls, from_type: Type[ScalarType], to_type: Type[ScalarType], mask_value: str
    ) -> None:
        explicit_promotion = EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING[from_type]
        if to_type.is_included(explicit_promotion):
            return cls.check_mask_value(from_type, to_type, mask_value)

        raise SemanticError(
            "1-1-5-5",
            op=cls.op,
            type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            mask_value=mask_value,
        )

    @classmethod
    def check_without_mask(cls, from_type: Type[ScalarType], to_type: Type[ScalarType]) -> None:
        explicit_promotion = EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING[from_type]
        implicit_promotion = IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]
        if not (to_type.is_included(explicit_promotion) or to_type.is_included(implicit_promotion)):
            explicit_with_mask = EXPLICIT_WITH_MASK_TYPE_PROMOTION_MAPPING[from_type]
            if to_type.is_included(explicit_with_mask):
                raise SemanticError(
                    "1-1-5-3",
                    op=cls.op,
                    type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                    type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
                )
            raise SemanticError(
                "1-1-5-4",
                op=cls.op,
                type_1=SCALAR_TYPES_CLASS_REVERSE[from_type],
                type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
            )

    @classmethod
    def cast_component(
        cls, data: Any, from_type: Type[ScalarType], to_type: Type[ScalarType]
    ) -> Any:
        """
        Cast the component to the type to_type without mask
        """

        if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            result = data.map(lambda x: to_type.implicit_cast(x, from_type), na_action="ignore")
        else:
            result = data.map(lambda x: to_type.explicit_cast(x, from_type), na_action="ignore")
        return result

    @classmethod
    def cast_mask_component(cls, data: Any, from_type: Any, to_type: Any, mask: str) -> Any:
        result = data.map(lambda x: cls.cast_value(x, from_type, to_type, mask), na_action="ignore")
        return result

    @classmethod
    def cast_value(
        cls,
        value: Any,
        provided_type: Type[ScalarType],
        to_type: Type[ScalarType],
        mask_value: str,
    ) -> Any:
        if provided_type == String and to_type == Number:
            return cls.cast_string_to_number(value, mask_value)
        if provided_type == String and to_type == Date:
            return cls.cast_string_to_date(value, mask_value)
        if provided_type == String and to_type == Duration:
            return cls.cast_string_to_duration(value, mask_value)
        if provided_type == String and to_type == TimePeriod:
            return cls.cast_string_to_time_period(value, mask_value)
        if provided_type == String and to_type == TimeInterval:
            return cls.cast_string_to_time(value, mask_value)
        # if provided_type == Date and to_type == String:
        #     return cls.cast_date_to_string(value, mask_value)
        # if provided_type == Duration and to_type == String:
        #     return cls.cast_duration_to_string(value, mask_value)
        # if provided_type == TimeInterval and to_type == String:
        #     return cls.cast_time_to_string(value, mask_value)
        if provided_type == TimePeriod and to_type == Date:
            return cls.cast_time_period_to_date(value, mask_value)

        raise SemanticError(
            "2-1-5-1",
            op=cls.op,
            value=value,
            type_1=SCALAR_TYPES_CLASS_REVERSE[provided_type],
            type_2=SCALAR_TYPES_CLASS_REVERSE[to_type],
        )

    @classmethod
    def validate(  # type: ignore[override]
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        scalarType: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Any:
        if mask is not None and not isinstance(mask, str):
            raise Exception(f"{cls.op} mask must be a string")

        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand, scalarType, mask)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand, scalarType, mask)
        elif isinstance(operand, Scalar):
            return cls.scalar_validation(operand, scalarType, mask)

    @classmethod
    def dataset_validation(  # type: ignore[override]
        cls,
        operand: Dataset,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Dataset:
        """
        This method validates the operation when the operand is a Dataset.
        """

        # monomeasure
        if len(operand.get_measures()) != 1:
            raise Exception(f"{cls.op} can only be applied to a Dataset with one measure")
        measure = operand.get_measures()[0]
        from_type = measure.data_type

        cls.check_cast(from_type, to_type, mask)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in operand.components.items()
            if comp.role != Role.MEASURE
        }

        if not to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            measure_name = COMP_NAME_MAPPING[to_type]
        else:
            measure_name = measure.name
        result_components[measure_name] = Component(
            name=measure_name,
            data_type=to_type,
            role=Role.MEASURE,
            nullable=measure.nullable,
        )
        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def component_validation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> DataComponent:
        """
        This method validates the operation when the operand is a DataComponent.
        """

        from_type = operand.data_type
        cls.check_cast(from_type, to_type, mask)
        return DataComponent(name=operand.name, data=None, data_type=to_type, role=operand.role)

    @classmethod
    def scalar_validation(  # type: ignore[override]
        cls,
        operand: Scalar,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Scalar:
        """
        This method validates the operation when the operand is a DataComponent.
        """

        from_type = operand.data_type
        cls.check_cast(from_type, to_type, mask)
        return Scalar(name=operand.name, data_type=to_type, value=None)

    @classmethod
    def evaluate(  # type: ignore[override]
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        scalarType: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Any:
        if isinstance(operand, Dataset):
            return cls.dataset_evaluation(operand, scalarType, mask)
        if isinstance(operand, Scalar):
            return cls.scalar_evaluation(operand, scalarType, mask)
        if isinstance(operand, DataComponent):
            return cls.component_evaluation(operand, scalarType, mask)

    @classmethod
    def dataset_evaluation(  # type: ignore[override]
        cls,
        operand: Dataset,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Dataset:
        from_type = operand.get_measures()[0].data_type
        original_measure = operand.get_measures()[0]
        result_dataset = cls.dataset_validation(operand, to_type, mask)
        new_measure = result_dataset.get_measures()[0]
        result_dataset.data = operand.data.copy() if operand.data is not None else pd.DataFrame()

        if original_measure.name != new_measure.name:
            result_dataset.data.rename(
                columns={original_measure.name: new_measure.name}, inplace=True
            )
        measure_data = result_dataset.data[new_measure.name]
        if mask:
            result_dataset.data[new_measure.name] = cls.cast_mask_component(
                measure_data, from_type, to_type, mask
            )
        else:
            result_dataset.data[new_measure.name] = cls.cast_component(
                measure_data, from_type, to_type
            )
        return result_dataset

    @classmethod
    def scalar_evaluation(  # type: ignore[override]
        cls,
        operand: Scalar,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Scalar:
        from_type = operand.data_type
        result_scalar = cls.scalar_validation(operand, to_type, mask)
        if pd.isna(operand.value):
            return Scalar(name=result_scalar.name, data_type=to_type, value=None)
        if mask:
            casted_data = cls.cast_value(operand.value, operand.data_type, to_type, mask)
        else:
            if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
                casted_data = to_type.implicit_cast(operand.value, from_type)
            else:
                casted_data = to_type.explicit_cast(operand.value, from_type)
        return Scalar(name=result_scalar.name, data_type=to_type, value=casted_data)

    @classmethod
    def component_evaluation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> DataComponent:
        from_type = operand.data_type
        result_component = cls.component_validation(operand, to_type, mask)
        if mask:
            casted_data = cls.cast_mask_component(operand.data, from_type, to_type, mask)
        else:
            casted_data = cls.cast_component(operand.data, from_type, to_type)
        result_component.data = casted_data
        return result_component

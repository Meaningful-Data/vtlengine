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
    ScalarType,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, DataComponent, Dataset, Role, Scalar
from vtlengine.Utils.__Virtual_Assets import VirtualCounter

duration_mapping = {"A": 6, "S": 5, "Q": 4, "M": 3, "W": 2, "D": 1}

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]


class Cast(Operator.Unary):
    op = CAST

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
        raise NotImplementedError("Cast with mask is not yet implemented.")

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
        """Cast the component to the type to_type without mask."""
        if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            result = data.map(lambda x: to_type.implicit_cast(x, from_type), na_action="ignore")
        else:
            result = data.map(lambda x: to_type.explicit_cast(x, from_type), na_action="ignore")
        return result.astype(to_type.dtype())

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
        """This method validates the operation when the operand is a Dataset."""
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
        dataset_name = VirtualCounter._new_ds_name()
        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def component_validation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> DataComponent:
        """This method validates the operation when the operand is a DataComponent."""
        from_type = operand.data_type
        cls.check_cast(from_type, to_type, mask)
        comp_name = VirtualCounter._new_dc_name()
        return DataComponent(
            name=comp_name,
            data=None,
            data_type=to_type,
            role=operand.role,
            nullable=operand.nullable,
        )

    @classmethod
    def scalar_validation(  # type: ignore[override]
        cls,
        operand: Scalar,
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Scalar:
        """This method validates the operation when the operand is a Scalar."""
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
        result_dataset.data[new_measure.name] = cls.cast_component(measure_data, from_type, to_type)
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
        casted_data = cls.cast_component(operand.data, from_type, to_type)
        result_component.data = casted_data
        return result_component

from copy import copy
from typing import Any, Optional, Type, Union

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import CAST
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING,
    IMPLICIT_TYPE_PROMOTION_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    Date,
    ScalarType,
    String,
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
        if from_type == String and to_type == Date and operand.value is not None:
            Date.explicit_cast(operand.value, String)
        return Scalar(name=operand.name, data_type=to_type, value=None, nullable=operand.nullable)

    @classmethod
    def cast_scalar(
        cls,
        operand: Scalar,
        scalarType: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Scalar:
        """Cast a scalar operand to the given type and return the resulting Scalar."""
        from_type = operand.data_type
        cls.check_cast(from_type, scalarType, mask)
        if operand.value is None:
            return Scalar(
                name=operand.name, data_type=scalarType, value=None, nullable=operand.nullable
            )
        if scalarType.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            value = scalarType.implicit_cast(operand.value, from_type)
        else:
            value = scalarType.explicit_cast(operand.value, from_type)
        return Scalar(
            name=operand.name, data_type=scalarType, value=value, nullable=operand.nullable
        )

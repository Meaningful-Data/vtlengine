from copy import copy
from typing import Any, Optional, Type, Union

import pandas as pd

import vtlengine.Operators as Operator
from vtlengine.AST.Grammar.tokens import CAST
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    EXPLICIT_WITHOUT_MASK_TYPE_PROMOTION_MAPPING,
    IMPLICIT_TYPE_PROMOTION_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    Null,
    ScalarType,
)
from vtlengine.DataTypes.CastMasks import cast_with_mask, check_mask
from vtlengine.Exceptions import InputValidationException, RunTimeError, SemanticError
from vtlengine.Model import (
    Component,
    DataComponent,
    Dataset,
    Role,
    Scalar,
    ValueDomain,
    component_type_name,
)
from vtlengine.Utils.__Virtual_Assets import VirtualCounter

duration_mapping = {"A": 6, "S": 5, "Q": 4, "M": 3, "W": 2, "D": 1}

ALL_MODEL_DATA_TYPES = Union[Dataset, Scalar, DataComponent]
CAST_TARGET = Union[Type[ScalarType], ValueDomain]


class Cast(Operator.Unary):
    op = CAST

    @classmethod
    def check_cast(
        cls,
        source: Any,
        target: CAST_TARGET,
        mask_value: Optional[str],
    ) -> None:
        """A cast of the source, a component or a scalar, to the target. A value domain takes
        the values of its basic type, also the ones on another value domain."""
        from_type, to_type = source.data_type, cls.target_type(target)
        if mask_value is not None:
            cls.check_with_mask(from_type, to_type, mask_value)
        elif not isinstance(target, ValueDomain):
            cls.check_without_mask(from_type, to_type)
        elif from_type not in (to_type, Null):
            source_type = component_type_name(from_type, getattr(source, "value_domain", None))
            raise SemanticError("1-1-5-1", type_1=source_type, type_2=target.name)

    @staticmethod
    def target_type(target: CAST_TARGET) -> Type[ScalarType]:
        """The basic scalar type of the values a cast gives."""
        return target.type if isinstance(target, ValueDomain) else target

    @staticmethod
    def target_domain(target: CAST_TARGET) -> Optional[ValueDomain]:
        return target if isinstance(target, ValueDomain) else None

    @classmethod
    def check_with_mask(
        cls, from_type: Type[ScalarType], to_type: Type[ScalarType], mask_value: str
    ) -> None:
        check_mask(from_type, to_type, mask_value)

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
        cls,
        data: Any,
        from_type: Type[ScalarType],
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> Any:
        """Cast the component to the target, with the mask if there is one. The values
        cast to a value domain must be values of it."""
        to_type = cls.target_type(target)
        if mask is not None:
            result = data.map(
                lambda x: cast_with_mask(x, from_type, to_type, mask), na_action="ignore"
            )
        elif to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            result = data.map(lambda x: to_type.implicit_cast(x, from_type), na_action="ignore")
        else:
            result = data.map(lambda x: to_type.explicit_cast(x, from_type), na_action="ignore")
        result = result.astype(to_type.dtype())
        if isinstance(target, ValueDomain):
            Operator.check_values(result, target, cls.op)
        return result

    @classmethod
    def validate(  # type: ignore[override]
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        scalarType: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> Any:
        if mask is not None and not isinstance(mask, str):
            raise SemanticError("1-1-5-7", op=cls.op)

        if isinstance(operand, Dataset):
            return cls.dataset_validation(operand, scalarType, mask)
        elif isinstance(operand, DataComponent):
            return cls.component_validation(operand, scalarType, mask)
        elif isinstance(operand, Scalar):
            return cls.scalar_validation(operand, scalarType, mask)

    @staticmethod
    def result_measure_name(measure: Component, to_type: Type[ScalarType]) -> str:
        """A Measure keeps its name when it is implicitly promoted to the type, and takes the
        generic name of the type otherwise."""
        if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[measure.data_type]):
            return measure.name
        return COMP_NAME_MAPPING[to_type]

    @classmethod
    def dataset_validation(  # type: ignore[override]
        cls,
        operand: Dataset,
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> Dataset:
        """This method validates the operation when the operand is a Dataset."""
        if len(operand.get_measures()) == 0:
            raise SemanticError("1-1-1-8", op=cls.op, name=operand.name)
        if len(operand.get_measures()) > 1:
            raise SemanticError("1-1-1-4", op=cls.op)
        measure = operand.get_measures()[0]
        to_type = cls.target_type(target)

        cls.check_cast(measure, target, mask)
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in operand.components.items()
            if comp.role in (Role.IDENTIFIER, Role.VIRAL_ATTRIBUTE)
        }

        measure_name = cls.result_measure_name(measure, to_type)
        result_components[measure_name] = Component(
            name=measure_name,
            data_type=to_type,
            role=Role.MEASURE,
            nullable=measure.nullable,
            value_domain=cls.target_domain(target),
        )
        dataset_name = VirtualCounter._new_ds_name()
        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def component_validation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> DataComponent:
        """This method validates the operation when the operand is a DataComponent."""
        cls.check_cast(operand, target, mask)
        comp_name = VirtualCounter._new_dc_name()
        return DataComponent(
            name=comp_name,
            data=None,
            data_type=cls.target_type(target),
            role=operand.role,
            nullable=operand.nullable,
            value_domain=cls.target_domain(target),
        )

    @classmethod
    def scalar_validation(  # type: ignore[override]
        cls,
        operand: Scalar,
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> Scalar:
        """This method validates the operation when the operand is a Scalar. A scalar cast to
        a value domain keeps its basic type, as no scalar is defined on a value domain."""
        from_type, to_type = operand.data_type, cls.target_type(target)
        cls.check_cast(operand, target, mask)
        if operand.value is not None and not pd.isna(operand.value):
            try:
                value = cls.cast_value(operand.value, from_type, to_type, mask)
                value_domain = cls.target_domain(target)
                if value_domain is not None:
                    Operator.check_values(pd.Series([value]), value_domain, cls.op)
            except (RunTimeError, InputValidationException) as error:
                types = {
                    "type_1": SCALAR_TYPES_CLASS_REVERSE[from_type],
                    "type_2": component_type_name(to_type, cls.target_domain(target)),
                }
                # A literal is part of the script, a scalar value is data (issue #1192)
                if operand.literal:
                    raise SemanticError("1-1-5-6", value=operand.value, **types) from error
                if isinstance(error, RunTimeError):
                    raise
                raise RunTimeError("2-1-5-1", value=operand.value, **types) from error
        return Scalar(name=operand.name, data_type=to_type, value=None, nullable=operand.nullable)

    @classmethod
    def cast_value(
        cls,
        value: Any,
        from_type: Type[ScalarType],
        to_type: Type[ScalarType],
        mask: Optional[str] = None,
    ) -> Any:
        """Cast a single value to the type to_type, with the mask if there is one."""
        if mask is not None:
            return cast_with_mask(value, from_type, to_type, mask)
        if to_type.is_included(IMPLICIT_TYPE_PROMOTION_MAPPING[from_type]):
            return to_type.implicit_cast(value, from_type)
        return to_type.explicit_cast(value, from_type)

    @classmethod
    def evaluate(  # type: ignore[override]
        cls,
        operand: ALL_MODEL_DATA_TYPES,
        scalarType: CAST_TARGET,
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
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> Dataset:
        from_type = operand.get_measures()[0].data_type
        original_measure = operand.get_measures()[0]
        result_dataset = cls.dataset_validation(operand, target, mask)
        new_measure = result_dataset.get_measures()[0]
        attributes = [name for name, c in operand.components.items() if c.role == Role.ATTRIBUTE]
        result_dataset.data = (
            operand.data.drop(columns=attributes) if operand.data is not None else pd.DataFrame()
        )

        if original_measure.name != new_measure.name:
            result_dataset.data.rename(
                columns={original_measure.name: new_measure.name}, inplace=True
            )
        measure_data = result_dataset.data[new_measure.name]
        result_dataset.data[new_measure.name] = cls.cast_component(
            measure_data, from_type, target, mask
        )
        return result_dataset

    @classmethod
    def scalar_evaluation(  # type: ignore[override]
        cls,
        operand: Scalar,
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> Scalar:
        from_type, to_type = operand.data_type, cls.target_type(target)
        result_scalar = cls.scalar_validation(operand, target, mask)
        if pd.isna(operand.value):
            return Scalar(
                name=result_scalar.name, data_type=to_type, value=None, nullable=operand.nullable
            )
        casted_data = cls.cast_value(operand.value, from_type, to_type, mask)
        return Scalar(
            name=result_scalar.name, data_type=to_type, value=casted_data, nullable=operand.nullable
        )

    @classmethod
    def component_evaluation(  # type: ignore[override]
        cls,
        operand: DataComponent,
        target: CAST_TARGET,
        mask: Optional[str] = None,
    ) -> DataComponent:
        from_type = operand.data_type
        result_component = cls.component_validation(operand, target, mask)
        casted_data = cls.cast_component(operand.data, from_type, target, mask)
        result_component.data = casted_data
        return result_component

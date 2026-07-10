from copy import copy
from typing import Any, Dict, List, Optional, Type, Union

from vtlengine.AST.Grammar.tokens import CHECK, CHECK_HIERARCHY
from vtlengine.DataTypes import (
    Boolean,
    Integer,
    Number,
    ScalarType,
    String,
    check_unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Operators import Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter


# noinspection PyTypeChecker
class Check(Operator):
    op = CHECK

    @classmethod
    def validate(
        cls,
        validation_element: Dataset,
        imbalance_element: Optional[Dataset],
        error_code: Optional[Union[str, int, float, bool]],
        error_level: Optional[Union[str, int, float, bool]],
        invalid: bool,
    ) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        if len(validation_element.get_measures()) != 1:
            raise SemanticError("1-1-10-1", op=cls.op, op_type="validation", me_type="Boolean")
        measure = validation_element.get_measures()[0]
        if measure.data_type != Boolean:
            raise SemanticError("1-1-10-1", op=cls.op, op_type="validation", me_type="Boolean")
        error_level_type: Optional[Type[ScalarType]] = None
        if isinstance(error_level, bool):
            error_level_type = Boolean
        elif error_level is None or isinstance(error_level, int):
            error_level_type = Integer
        elif isinstance(error_level, str):
            error_level_type = String
        else:
            error_level_type = String

        imbalance_measure = None
        if imbalance_element is not None:
            operand_identifiers = validation_element.get_identifiers_names()
            imbalance_identifiers = imbalance_element.get_identifiers_names()
            if operand_identifiers != imbalance_identifiers:
                raise Exception(
                    "The validation and imbalance operands must have the same identifiers"
                )
            if len(imbalance_element.get_measures()) != 1:
                raise SemanticError("1-1-10-1", op=cls.op, op_type="imbalance", me_type="Numeric")

            imbalance_measure = imbalance_element.get_measures()[0]
            if imbalance_measure.data_type not in (Number, Integer):
                raise SemanticError("1-1-10-1", op=cls.op, op_type="imbalance", me_type="Numeric")

        # Generating the result dataset components
        result_components = {
            comp.name: comp
            for comp in validation_element.components.values()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE, Role.VIRAL_ATTRIBUTE]
        }
        if imbalance_measure is None:
            result_components["imbalance"] = Component(
                name="imbalance", data_type=Number, role=Role.MEASURE, nullable=True
            )
        else:
            result_components["imbalance"] = copy(imbalance_measure)
            result_components["imbalance"].name = "imbalance"

        result_components["errorcode"] = Component(
            name="errorcode", data_type=String, role=Role.MEASURE, nullable=True
        )

        result_components["errorlevel"] = Component(
            name="errorlevel",
            data_type=error_level_type,
            role=Role.MEASURE,
            nullable=True,
        )

        return Dataset(name=dataset_name, components=result_components, data=None)


# noinspection PyTypeChecker
class Validation(Operator):
    @classmethod
    def validate(
        cls,
        dataset_element: Dataset,
        rule_info: Dict[str, Any],
        output: str,
        viral_components: Optional[List[Component]] = None,
    ) -> Dataset:
        error_level_type: Optional[Type[ScalarType]] = None
        error_levels = [
            rule_data.get("errorlevel")
            for rule_data in rule_info.values()
            if "errorlevel" in rule_data
        ]
        non_null_levels = [el for el in error_levels if el is not None]

        if all(isinstance(el, bool) for el in non_null_levels) and len(non_null_levels) > 0:
            error_level_type = Boolean
        elif len(non_null_levels) == 0 or all(isinstance(el, int) for el in non_null_levels):
            error_level_type = Number
        elif all(isinstance(el, str) for el in non_null_levels):
            error_level_type = String
        else:
            error_level_type = String
        dataset_name = VirtualCounter._new_ds_name()
        result_components = {comp.name: comp for comp in dataset_element.get_identifiers()}
        result_components["ruleid"] = Component(
            name="ruleid", data_type=String, role=Role.IDENTIFIER, nullable=False
        )
        if output == "invalid":
            result_components = {
                **result_components,
                **{comp.name: copy(comp) for comp in dataset_element.get_measures()},
            }
        elif output == "all":
            result_components["bool_var"] = Component(
                name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
            )
        else:  # output == 'all_measures'
            result_components = {
                **result_components,
                **{comp.name: copy(comp) for comp in dataset_element.get_measures()},
                "bool_var": Component(
                    name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
                ),
            }
        result_components["errorcode"] = Component(
            name="errorcode", data_type=String, role=Role.MEASURE, nullable=True
        )
        result_components["errorlevel"] = Component(
            name="errorlevel",
            data_type=error_level_type,
            role=Role.MEASURE,
            nullable=True,
        )
        # Viral attributes propagate to the validation result (issue #877). check_datapoint
        # uses the operand's own viral; check_hierarchy passes captured viral (its dataset
        # is stripped by validate_hr_dataset).
        viral_comps = (
            viral_components
            if viral_components is not None
            else dataset_element.get_viral_attributes()
        )
        for viral_comp in viral_comps:
            result_components[viral_comp.name] = copy(viral_comp)

        return Dataset(name=dataset_name, components=result_components, data=None)


class Check_Datapoint(Validation):
    pass


class Check_Hierarchy(Validation):
    op = CHECK_HIERARCHY

    @classmethod
    def validate(
        cls,
        dataset_element: Dataset,
        rule_info: Dict[str, Any],
        output: str,
        viral_components: Optional[List[Component]] = None,
    ) -> Dataset:
        result = super().validate(
            dataset_element, rule_info, output, viral_components=viral_components
        )
        result.components["imbalance"] = Component(
            name="imbalance", data_type=Number, role=Role.MEASURE, nullable=True
        )
        return result

    @staticmethod
    def validate_hr_dataset(dataset: Dataset, component_name: str) -> None:
        if len(dataset.get_measures()) != 1:
            raise SemanticError(
                "1-1-10-1", op=Check_Hierarchy.op, op_type="hierarchy", me_type="Number"
            )
        measure = dataset.get_measures()[0]
        if not check_unary_implicit_promotion(measure.data_type, Number):
            raise SemanticError(
                "1-1-10-1", op=Check_Hierarchy.op, op_type="hierarchy", me_type="Number"
            )
        if component_name not in dataset.components:
            raise SemanticError(
                "1-1-1-10",
                op=Check_Hierarchy.op,
                comp_name=component_name,
                dataset_name=dataset.name,
            )
        if dataset.components[component_name].role != Role.IDENTIFIER:
            raise SemanticError(
                "1-2-7",
                name=component_name,
                role=dataset.components[component_name].role.value,
            )
        # Remove attributes from dataset
        if len(dataset.get_attributes()) > 0:
            for x in dataset.get_attributes():
                dataset.delete_component(x.name)

        if len(dataset.get_viral_attributes()) > 0:
            for x in dataset.get_viral_attributes():
                dataset.delete_component(x.name)

from copy import copy
from typing import Any, Dict, Optional

import pandas as pd

from vtlengine.AST.Grammar.tokens import CHECK, CHECK_HIERARCHY
from vtlengine.DataTypes import (
    Boolean,
    Integer,
    Number,
    String,
    check_unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Operators import Operator


# noinspection PyTypeChecker
class Check(Operator):
    op = CHECK

    @classmethod
    def validate(
        cls,
        validation_element: Dataset,
        imbalance_element: Optional[Dataset],
        error_code: Optional[str],
        error_level: Optional[int],
        invalid: bool,
    ) -> Dataset:
        if len(validation_element.get_measures()) != 1:
            raise SemanticError("1-1-10-1", op=cls.op, op_type="validation", me_type="Boolean")
        measure = validation_element.get_measures()[0]
        if measure.data_type != Boolean:
            raise SemanticError("1-1-10-1", op=cls.op, op_type="validation", me_type="Boolean")

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
            if imbalance_measure.data_type != Number:
                raise SemanticError("1-1-10-1", op=cls.op, op_type="imbalance", me_type="Numeric")

        # Generating the result dataset components
        result_components = {
            comp.name: comp
            for comp in validation_element.components.values()
            if comp.role in [Role.IDENTIFIER, Role.MEASURE]
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
            name="errorlevel", data_type=Integer, role=Role.MEASURE, nullable=True
        )

        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def evaluate(
        cls,
        validation_element: Dataset,
        imbalance_element: Optional[Dataset],
        error_code: Optional[str],
        error_level: Optional[int],
        invalid: bool,
    ) -> Dataset:
        result = cls.validate(
            validation_element, imbalance_element, error_code, error_level, invalid
        )
        if validation_element.data is None:
            validation_element.data = pd.DataFrame()
        columns_to_keep = (
            validation_element.get_identifiers_names() + validation_element.get_measures_names()
        )
        result.data = validation_element.data.loc[:, columns_to_keep]
        if imbalance_element is not None and imbalance_element.data is not None:
            imbalance_measure_name = imbalance_element.get_measures_names()[0]
            result.data["imbalance"] = imbalance_element.data[imbalance_measure_name]
        else:
            result.data["imbalance"] = None

        result.data["errorcode"] = error_code
        result.data["errorlevel"] = error_level
        if invalid:
            # TODO: Is this always bool_var?? In any case this does the trick for more use cases
            validation_measure_name = validation_element.get_measures_names()[0]
            result.data = result.data[result.data[validation_measure_name] == False]
            result.data.reset_index(drop=True, inplace=True)
        return result


# noinspection PyTypeChecker
class Validation(Operator):
    @classmethod
    def _generate_result_data(cls, rule_info: Dict[str, Any]) -> pd.DataFrame:
        rule_list_df = []
        for rule_name, rule_data in rule_info.items():
            rule_df = rule_data["output"]
            rule_df["ruleid"] = rule_name
            rule_df["errorcode"] = rule_df["bool_var"].map({False: rule_data["errorcode"]})
            rule_df["errorlevel"] = rule_df["bool_var"].map({False: rule_data["errorlevel"]})
            rule_list_df.append(rule_df)

        if len(rule_list_df) == 1:
            return rule_list_df[0]
        df = pd.concat(rule_list_df, ignore_index=True, copy=False)
        return df

    @classmethod
    def validate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
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
            name="errorlevel", data_type=Number, role=Role.MEASURE, nullable=True
        )

        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def evaluate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
        result = cls.validate(dataset_element, rule_info, output)
        result.data = cls._generate_result_data(rule_info)

        result.data = result.data.dropna(subset=result.get_identifiers_names(), how="any")
        result.data = result.data.drop_duplicates(
            subset=result.get_identifiers_names() + ["ruleid"]
        ).reset_index(drop=True)
        validation_measures = ["bool_var", "errorcode", "errorlevel"]
        # Only for check hierarchy
        if "imbalance" in result.components:
            validation_measures.append("imbalance")
        if output == "invalid":
            result.data = result.data[result.data["bool_var"] == False]
            result.data = result.data.drop(columns=["bool_var"])
            result.data.reset_index(drop=True, inplace=True)
        elif output == "all":
            result.data = result.data[result.get_identifiers_names() + validation_measures]
        else:  # output == 'all_measures'
            result.data = result.data[
                result.get_identifiers_names()
                + dataset_element.get_measures_names()
                + validation_measures
            ]

        result.data = result.data[result.get_components_names()]
        return result


class Check_Datapoint(Validation):
    pass


class Check_Hierarchy(Validation):
    op = CHECK_HIERARCHY

    @classmethod
    def _generate_result_data(cls, rule_info: Dict[str, Any]) -> pd.DataFrame:
        df = pd.DataFrame()
        for rule_name, rule_data in rule_info.items():
            rule_df = rule_data["output"]
            rule_df["ruleid"] = rule_name
            rule_df["errorcode"] = rule_data["errorcode"]
            rule_df["errorlevel"] = rule_data["errorlevel"]
            df = pd.concat([df, rule_df], ignore_index=True)
        if df is None:
            df = pd.DataFrame()
        return df

    @classmethod
    def validate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
        result = super().validate(dataset_element, rule_info, output)
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
                "1-3-20",
                name=component_name,
                role=dataset.components[component_name].role.value,
            )
        # Remove attributes from dataset
        if len(dataset.get_attributes()) > 0:
            for x in dataset.get_attributes():
                dataset.delete_component(x.name)

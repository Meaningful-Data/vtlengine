from copy import copy
from typing import Any, Dict, Optional

from duckdb.duckdb import DuckDBPyRelation  # type: ignore[import-untyped]

from vtlengine.AST.Grammar.tokens import CHECK, CHECK_HIERARCHY
from vtlengine.DataTypes import (
    Boolean,
    Integer,
    Number,
    String,
    check_unary_implicit_promotion,
)
from vtlengine.duckdb.duckdb_utils import (
    duckdb_concat,
    duckdb_select,
    empty_relation,
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
        error_code: Optional[str],
        error_level: Optional[int],
        invalid: bool,
    ) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
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

        return Dataset(name=dataset_name, components=result_components, data=None)

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
            validation_element.data = empty_relation()

        error_code_, error_level_ = (repr(error_code) or "NULL"), (repr(error_level) or "NULL")
        columns_to_keep = (
            validation_element.get_identifiers_names() + validation_element.get_measures_names()
        )
        result.data = validation_element.data.project(
            ", ".join(f'"{col}"' for col in columns_to_keep)
        )

        exprs = [f'"{col}"' for col in columns_to_keep]
        if imbalance_element and imbalance_element.data:
            measure = imbalance_element.get_measures_names()[0]
            result.data = duckdb_concat(result.data, duckdb_select(imbalance_element.data, measure))
            exprs.append(f'"{measure}" AS "imbalance"')
        else:
            exprs.append('NULL AS "imbalance"')
        exprs.extend([f'{error_code_} AS "errorcode"', f'{error_level_} AS "errorlevel"'])

        result.data = result.data.project(", ".join(exprs))
        if invalid:
            result.data = result.data.filter(
                f'"{validation_element.get_measures_names()[0]}" = False'
            )

        return result


# noinspection PyTypeChecker
class Validation(Operator):
    @classmethod
    def _generate_result_data(cls, rule_info: Dict[str, Any]) -> DuckDBPyRelation:
        rel_list = []
        for rule_name, rule_data in rule_info.items():
            rel = rule_data["output"]
            errorcode = repr(rule_data.get("errorcode", "NULL") or "NULL")
            errorlevel = repr(rule_data.get("errorlevel", "NULL") or "NULL")
            query = f"""
            *,
            {rule_name} AS ruleid,
            CASE WHEN bool_var = FALSE THEN {errorcode} ELSE NULL END AS errorcode,
            CASE WHEN bool_var = FALSE THEN {errorlevel} ELSE NULL END AS errorlevel
            """
            rel_list.append(rel.project(query))

        result = rel_list[0]
        for rel in rel_list[1:]:
            result = result.union(rel)
        return result

    @classmethod
    def validate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
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
            name="errorlevel", data_type=Number, role=Role.MEASURE, nullable=True
        )

        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def evaluate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
        result = cls.validate(dataset_element, rule_info, output)
        result.data = cls._generate_result_data(rule_info)

        identifiers = result.get_identifiers_names()
        result.data = result.data.filter(
            " AND ".join(f'"{id_}" IS NOT NULL' for id_ in identifiers)
        ).distinct()

        validation_measures = ["bool_var", "errorcode", "errorlevel"]
        if "imbalance" in result.components:
            validation_measures.append("imbalance")

        if output == "invalid":
            result.data = result.data.filter("bool_var = FALSE")
        elif output == "all":
            result.data = result.data.project(", ".join(identifiers + validation_measures))
        else:  # output == 'all_measures'
            result.data = result.data.project(
                ", ".join(identifiers + dataset_element.get_measures_names() + validation_measures)
            )

        result.data = result.data.project(", ".join(result.get_components_names()))
        return result


class Check_Datapoint(Validation):
    pass


class Check_Hierarchy(Validation):
    op = CHECK_HIERARCHY

    @classmethod
    def _generate_result_data(cls, rule_info: Dict[str, Any]) -> DuckDBPyRelation:
        result_data = None
        for rule_name, rule_data in rule_info.items():
            rule_output = rule_data["output"]
            rule_output = rule_output.project(
                f'*, "{rule_name}" AS ruleid, '
                f"{rule_data['errorcode']} AS errorcode, "
                f"{rule_data['errorlevel']} AS errorlevel"
            )
            result_data = rule_output if result_data is None else result_data.union_all(rule_output)
        return result_data

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

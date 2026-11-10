from copy import copy
from typing import Any, Collection, Dict, List, Optional, Tuple, Type, Union

import pandas as pd

from vtlengine.AST.Grammar.tokens import CHECK, CHECK_DATAPOINT, CHECK_HIERARCHY
from vtlengine.DataTypes import (
    Boolean,
    Integer,
    Number,
    ScalarType,
    String,
    check_unary_implicit_promotion,
)
from vtlengine.Exceptions import InputValidationException, SemanticError
from vtlengine.Model import Component, Dataset, Role, ValueDomain, component_type_name
from vtlengine.Operators import Operator
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.Utils._dataframe import merge_frames

ERROR_VALUE_DOMAINS = {"errorcode": "errorcode_vd", "errorlevel": "errorlevel_vd"}


def error_domains(
    value_domains: Optional[Dict[str, ValueDomain]],
) -> Dict[str, Optional[ValueDomain]]:
    """The value domains given for the errorcode and errorlevel Measures."""
    return {
        element: (value_domains or {}).get(name) for element, name in ERROR_VALUE_DOMAINS.items()
    }


def error_value(element: str, value: Any, value_domain: Optional[ValueDomain], where: str) -> Any:
    """The error code or level (element) that where writes, as a value of its value domain when
    it is given: one that is not raises."""
    if value is None or value_domain is None:
        return value
    try:
        cast_value = value_domain.type.cast(value)
    except (ValueError, TypeError, InputValidationException):
        cast_value = None
    if cast_value is None or len(value_domain.outside(pd.Series([cast_value]))):
        raise SemanticError(
            "1-1-10-13",
            element=element,
            value=value,
            where=where,
            value_domain=value_domain.name,
        )
    return cast_value


def error_values(
    code: Any, level: Any, value_domains: Optional[Dict[str, ValueDomain]], where: str
) -> Tuple[Any, Any]:
    """The error code and level that where writes, as values of errorcode_vd and
    errorlevel_vd when they are given."""
    domains = error_domains(value_domains)
    return (
        error_value("errorcode", code, domains["errorcode"], where),
        error_value("errorlevel", level, domains["errorlevel"], where),
    )


def _set_error_types(
    components: Dict[str, Component], value_domains: Optional[Dict[str, ValueDomain]]
) -> None:
    """The errorcode and errorlevel Measures take the basic types of the value domains given."""
    for element, value_domain in error_domains(value_domains).items():
        if value_domain is not None:
            components[element].data_type = value_domain.type


def _cast_error_columns(result: Dataset, value_domains: Optional[Dict[str, ValueDomain]]) -> None:
    if result.data is None:
        return
    for element, value_domain in error_domains(value_domains).items():
        if value_domain is not None and element in result.data.columns:
            result.data[element] = result.data[element].astype(value_domain.type.dtype())  # type: ignore[call-overload]


# noinspection PyTypeChecker
def bind_value_domains(
    ruleset: str,
    signature: Dict[str, str],
    called: List[str],
    components: Dict[str, Component],
    value_domains: Optional[Dict[str, ValueDomain]],
) -> Tuple[Dict[str, str], Dict[str, ValueDomain]]:
    """The components a datapoint ruleset on value domains is called with, bound in order to the
    value domains of its signature, or their aliases, and the ones whose
    values must be checked against their value domain.

    A component defined on the value domain is used as it is, one of its basic type takes its
    values (downcast), and one of another basic type raises. A value domain that is not given
    is not checked."""
    if all(name in components for name in signature.values()):
        # Backwards compatibility, to be removed: value domains named as components of the
        # Data Set are read as those components, whatever components the call lists
        return signature, {}
    if len(called) != len(signature):
        raise SemanticError("1-1-10-2", op=CHECK_DATAPOINT)
    checks: Dict[str, ValueDomain] = {}
    for name, comp_name in zip(signature.values(), called):
        value_domain = (value_domains or {}).get(name)
        component = components[comp_name]
        if value_domain is None or (
            component.value_domain is not None and component.value_domain.name == name
        ):
            continue
        if component.data_type != value_domain.type:
            raise SemanticError(
                "1-1-10-12",
                op=CHECK_DATAPOINT,
                comp_name=comp_name,
                type=component_type_name(component.data_type, component.value_domain),
                value_domain=name,
                ruleset=ruleset,
            )
        checks[comp_name] = value_domain
    return dict(zip(signature, called)), checks


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
        value_domains: Optional[Dict[str, ValueDomain]] = None,
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
                raise SemanticError("1-1-10-11", op=cls.op)
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
        _set_error_types(result_components, value_domains)

        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def evaluate(
        cls,
        validation_element: Dataset,
        imbalance_element: Optional[Dataset],
        error_code: Optional[Union[str, int, float, bool]],
        error_level: Optional[Union[str, int, float, bool]],
        invalid: bool,
        value_domains: Optional[Dict[str, ValueDomain]] = None,
    ) -> Dataset:
        result = cls.validate(
            validation_element, imbalance_element, error_code, error_level, invalid, value_domains
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

        # Set errorcode/errorlevel ONLY when validation explicitly fails (bool_var is False)
        # NULL bool_var means indeterminate - should NOT have errorcode/errorlevel
        validation_measure_name = validation_element.get_measures_names()[0]
        bool_col = result.data[validation_measure_name]
        is_false = bool_col.fillna(True) == False  # noqa: E712
        errorcode_type = result.components["errorcode"].data_type
        result.data["errorcode"] = pd.Series(
            None, index=result.data.index, dtype=errorcode_type.dtype()
        )
        if error_code is not None:
            result.data.loc[is_false, "errorcode"] = errorcode_type.cast(error_code)
        errorlevel_dtype = result.components["errorlevel"].data_type.dtype()
        result.data["errorlevel"] = pd.Series(None, index=result.data.index, dtype=errorlevel_dtype)
        if error_level is not None:
            result.data.loc[is_false, "errorlevel"] = error_level

        if invalid:
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
            bool_col = rule_df["bool_var"]
            if str(bool_col.dtype) != "bool[pyarrow]":
                bool_col = bool_col.astype("bool[pyarrow]")
            rule_df["errorcode"] = bool_col.map({False: rule_data["errorcode"]})
            rule_df["errorlevel"] = bool_col.map({False: rule_data["errorlevel"]})
            rule_list_df.append(rule_df)

        if len(rule_list_df) == 1:
            return rule_list_df[0]
        return pd.concat(rule_list_df, ignore_index=True)

    @classmethod
    def validate(
        cls,
        dataset_element: Dataset,
        rule_info: Dict[str, Any],
        output: str,
        viral_components: Optional[List[Component]] = None,
        viral_values: Optional[pd.DataFrame] = None,
        value_domains: Optional[Dict[str, ValueDomain]] = None,
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
        _set_error_types(result_components, value_domains)
        # Viral attributes propagate to the validation result; their values are
        # re-attached in evaluate (issue #877). check_datapoint uses the operand's own
        # viral; check_hierarchy passes captured viral (its dataset is stripped).
        viral_comps = (
            viral_components
            if viral_components is not None
            else dataset_element.get_viral_attributes()
        )
        for viral_comp in viral_comps:
            result_components[viral_comp.name] = copy(viral_comp)

        return Dataset(name=dataset_name, components=result_components, data=None)

    @classmethod
    def evaluate(
        cls,
        dataset_element: Dataset,
        rule_info: Dict[str, Any],
        output: str,
        viral_components: Optional[List[Component]] = None,
        viral_values: Optional[pd.DataFrame] = None,
        value_domains: Optional[Dict[str, ValueDomain]] = None,
    ) -> Dataset:
        result = cls.validate(
            dataset_element,
            rule_info,
            output,
            viral_components=viral_components,
            value_domains=value_domains,
        )
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

        # Re-attach viral attributes matched per datapoint on the identifiers (copied, not
        # combined: check_datapoint/check_hierarchy are row-preserving validations) (issue
        # #906). Source: the operand's own data (check_datapoint) or the captured pre-strip
        # viral (check_hierarchy).
        viral_comps = (
            viral_components
            if viral_components is not None
            else dataset_element.get_viral_attributes()
        )
        viral_names = [c.name for c in viral_comps]
        viral_src_data = viral_values if viral_values is not None else dataset_element.data
        # A viral attribute the ruleset signature names is already carried through the
        # rule output (issue #898); the "invalid" branch keeps its column, so only the
        # stripped ones are re-attached.
        if viral_names and viral_src_data is not None:
            missing_viral = [name for name in viral_names if name not in result.data.columns]
            if missing_viral:
                join_ids = [
                    name
                    for name in result.get_identifiers_names()
                    if name in viral_src_data.columns
                ]
                viral_src = viral_src_data[join_ids + missing_viral].drop_duplicates(
                    subset=join_ids
                )
                result.data = merge_frames(result.data, viral_src, on=join_ids, how="left")

        result.data = result.data[result.get_components_names()]
        _cast_error_columns(result, value_domains)
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
            # Set errorcode/errorlevel ONLY when validation explicitly fails (bool_var is False)
            # NULL bool_var means indeterminate - should NOT have errorcode/errorlevel
            bool_col = rule_df["bool_var"]
            if str(bool_col.dtype) != "bool[pyarrow]":
                bool_col = bool_col.astype("bool[pyarrow]")
            rule_df["errorcode"] = bool_col.map({False: rule_data["errorcode"]})
            rule_df["errorlevel"] = bool_col.map({False: rule_data["errorlevel"]})
            df = pd.concat([df, rule_df], ignore_index=True)
        if df is None:
            df = pd.DataFrame()
        return df

    @classmethod
    def validate(
        cls,
        dataset_element: Dataset,
        rule_info: Dict[str, Any],
        output: str,
        viral_components: Optional[List[Component]] = None,
        viral_values: Optional[pd.DataFrame] = None,
        value_domains: Optional[Dict[str, ValueDomain]] = None,
    ) -> Dataset:
        result = super().validate(
            dataset_element,
            rule_info,
            output,
            viral_components=viral_components,
            value_domains=value_domains,
        )
        result.components["imbalance"] = Component(
            name="imbalance", data_type=Number, role=Role.MEASURE, nullable=True
        )
        return result

    @staticmethod
    def validate_hr_dataset(
        dataset: Dataset, component_name: str, condition_components: Collection[str] = ()
    ) -> None:
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
        # The conditions of the rules read their attributes (issue #1223)
        Check_Hierarchy.remove_attributes(dataset, keep=condition_components)

    @staticmethod
    def remove_attributes(dataset: Dataset, keep: Collection[str] = ()) -> None:
        """Remove the attributes, viral or not, of the operand except the ones in ``keep``."""
        for attribute in dataset.get_attributes() + dataset.get_viral_attributes():
            if attribute.name not in keep:
                dataset.delete_component(attribute.name)

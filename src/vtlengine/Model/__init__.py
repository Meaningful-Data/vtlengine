import inspect
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from pandas import DataFrame as PandasDataFrame
from pandas._testing import assert_frame_equal
from sqlglot.dialects.duckdb import DuckDB

import vtlengine.DataTypes as DataTypes
from vtlengine.DataTypes import RESERVED_COMPONENT_NAMES, SCALAR_TYPES, ScalarType
from vtlengine.DataTypes._time_checking import check_date, check_time, check_time_period
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import InputValidationException, SemanticError

# The loaders write the values of the time types in one form, which the values of a value
# domain on one of them are compared in
_TIME_VALUE_CHECKS: Dict[Any, Any] = {
    DataTypes.Date: check_date,
    DataTypes.TimePeriod: check_time_period,
    DataTypes.TimeInterval: check_time,
}
TIME_TYPES = tuple(_TIME_VALUE_CHECKS)


def on_distinct(series: Any, function: Any) -> Any:
    """The Series the function gives for the series, computed on its distinct values that are
    not null, each once, and taken back to its rows, null where the series is null. A column of
    a time type holds few distinct values, and each costs Python calls."""
    codes, distinct = pd.factorize(series)
    result = function(pd.Series(distinct))
    return pd.Series(result.array.take(codes, allow_fill=True), index=series.index)


@lru_cache(maxsize=None)
def component_name_pattern() -> "re.Pattern[str]":
    """The pattern of a Component name in the JSON schema of the data structures."""
    schema_path = Path(__file__).parents[1] / "API" / "data" / "schema" / "json_schema_2.1.json"
    with open(schema_path, "r") as file:
        return re.compile(json.load(file)["$defs"]["component-name"]["pattern"])


# How a message names a character that cannot be printed
_CHARACTER_NAMES = {"\x00": "NUL", "\r": "carriage return"}


@lru_cache(maxsize=None)
def component_name_characters() -> str:
    """The characters a Component name cannot contain: the ones that the name pattern of the
    JSON schema of the data structures leaves out (issue #1355)."""
    pattern = component_name_pattern().pattern
    excluded = pattern[len("^[^") : pattern.index("]")]
    return excluded.encode().decode("unicode_escape")


def component_name_characters_text() -> str:
    """The characters a Component name cannot contain, as a message lists them."""
    names = [_CHARACTER_NAMES.get(char, char) for char in component_name_characters()]
    return f"{', '.join(names[:-1])} or {names[-1]}"


def join_homonyms(names: Iterable[str], name: str) -> List[str]:
    """The Components alias#name that a join names after the operands that all have name. A
    calc that computes name replaces them (Inner Join page of the VTL 2.2 manual), and no other
    Component name holds # (issue #1355)."""
    return [comp for comp in names if comp.partition("#")[2] == name]


def printable_name(name: str) -> str:
    """A name as a message writes it, with the characters that cannot be printed escaped."""
    return "".join(char if char.isprintable() else repr(char)[1:-1] for char in name)


def component_name_error(name: str) -> str:
    """Why an input structure cannot name a Component name (0-2-1-1)."""
    return (
        f"'{printable_name(name)}' is not a valid Component name: a Component name cannot be "
        f"empty or contain {component_name_characters_text()}, except quotes around the "
        "whole name"
    )


def check_input_component_name(name: str) -> None:
    """A Component name of an input structure follows the name pattern of the JSON schema of
    the data structures: no quotes, no # (issue #1355), no NUL and no carriage return."""
    if not component_name_pattern().search(name):
        raise InputValidationException(
            code="0-2-1-1",
            element=f"Component '{printable_name(name)}'",
            error=component_name_error(name),
        )


def value_of(data_type: Any, value: Any) -> Any:
    """The value written in the one form the loaders write the values of its type in, so that
    two forms of a value compare equal. A value that is not one of the type is kept as it is
    written, equal to no value of it."""
    check = _TIME_VALUE_CHECKS.get(data_type)
    if check is None:
        return value
    try:
        return check(str(value))
    except (InputValidationException, ValueError, TypeError):
        return value


def values_of(data_type: Any, values: Any) -> Any:
    """The values of the series, each written as value_of writes it."""
    if data_type not in _TIME_VALUE_CHECKS:
        return values
    return on_distinct(values, lambda distinct: distinct.map(lambda v: value_of(data_type, v)))


BASIC_TYPE_NAMES = frozenset(
    ("string", "integer", "number", "boolean", "date", "time", "time_period", "duration", "scalar")
)


@dataclass
class Scalar:
    """
    Class representing a scalar value
    """

    name: str
    data_type: Type[ScalarType]
    _value: Any
    nullable: bool = True
    persistent: bool = False
    # A literal written in the script, whose value the semantic analysis already knows
    literal: bool = field(default=False, repr=False)
    # The value domain the scalar is defined on, whose basic type is its data type
    value_domain: Optional["ValueDomain"] = None

    def __init__(
        self,
        name: str,
        data_type: Type[ScalarType],
        value: Any,
        nullable: bool = True,
        persistent: bool = False,
        literal: bool = False,
        value_domain: Optional["ValueDomain"] = None,
    ) -> None:
        self.name = name
        self.data_type = data_type
        self.value = value
        self.nullable = nullable
        self.persistent = persistent
        self.literal = literal
        self.value_domain = value_domain

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, new_value: Any) -> None:
        if self.data_type and not self.data_type.check(new_value):
            raise InputValidationException(
                code="0-1-2-7",
                value=new_value,
                type_=self.data_type.__name__,
                op_type=self.__class__.__name__,
                name=self.name,
            )
        self._value = new_value

    @classmethod
    def from_json(
        cls, json_str: str, value_domains: Optional[Dict[str, "ValueDomain"]] = None
    ) -> "Scalar":
        """The scalar to_json writes. Its type can name one of the value domains given, by
        name, on which the scalar is then defined."""
        data = json.loads(json_str)
        # Support both 'type' and 'data_type' for backward compatibility
        data_type, value_domain = named_type(
            data.get("type") or data.get("data_type"), value_domains, f"Scalar '{data['name']}'"
        )
        return cls(
            data["name"],
            data_type,
            data["value"],
            data.get("nullable", True),
            value_domain=value_domain,
        )

    def to_dict(self) -> Dict[str, Any]:
        data_type = self.data_type
        if not inspect.isclass(self.data_type):
            data_type = self.data_type.__class__
        return {
            "name": self.name,
            "type": component_type_name(data_type, self.value_domain),
            "value": self.value,
            "nullable": self.nullable,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __eq__(self, other: Any) -> bool:
        same_name = self.name == other.name
        same_type = self.data_type == other.data_type
        x = None if not pd.isnull(self.value) else self.value
        y = None if not pd.isnull(other.value) else other.value
        same_value = x == y
        same_domain = getattr(self.value_domain, "name", None) == getattr(
            other.value_domain, "name", None
        )
        return same_name and same_type and same_value and same_domain


Role_keys = [
    "Identifier",
    "Attribute",
    "Measure",
    "Viral Attribute",
]


class Role(Enum):
    """
    Enum class for the role of a component  (Identifier, Attribute, Measure, Viral Attribute)
    """

    IDENTIFIER = "Identifier"
    ATTRIBUTE = "Attribute"
    MEASURE = "Measure"
    VIRAL_ATTRIBUTE = "Viral Attribute"


def component_type_name(data_type: Any, value_domain: Optional["ValueDomain"]) -> str:
    """The type a component is written with: its value domain, if it is defined on one."""
    if value_domain is not None:
        return value_domain.name
    return DataTypes.SCALAR_TYPES_CLASS_REVERSE[data_type]


def named_type(
    type_name: Any, value_domains: Optional[Dict[str, "ValueDomain"]], element: str
) -> Tuple[Type[ScalarType], Optional["ValueDomain"]]:
    """The data type and the value domain of the type a Component or a scalar is written with,
    the inverse of component_type_name: a basic type, or one of the value domains given, by
    name, on its basic type. Any other name raises 0-2-1-1 for the element."""
    if value_domains is not None and not isinstance(value_domains, dict):
        raise InputValidationException(code="0-1-1-2", input=value_domains)
    if type_name in SCALAR_TYPES:
        return SCALAR_TYPES[type_name], None
    value_domain = (value_domains or {}).get(type_name)
    if value_domain is None:
        raise InputValidationException(
            code="0-2-1-1",
            element=element,
            error=f"'{type_name}' is neither a basic type nor the name of a value domain given",
        )
    return value_domain.type, value_domain


@dataclass
class DataComponent:
    """A component of a dataset with data"""

    name: str
    data: Optional[Any]
    data_type: Type[ScalarType]
    role: Role = Role.MEASURE
    nullable: bool = True
    value_domain: Optional["ValueDomain"] = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataComponent):
            return False
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_json(
        cls, json_str: Any, value_domains: Optional[Dict[str, "ValueDomain"]] = None
    ) -> "DataComponent":
        """The component to_dict writes. Its type can name one of the value domains given, by
        name, on which the component is then defined."""
        # Support both 'type' and 'data_type' for backward compatibility
        data_type, value_domain = named_type(
            json_str.get("type") or json_str.get("data_type"),
            value_domains,
            f"Component '{json_str['name']}'",
        )
        return cls(
            json_str["name"],
            None,
            data_type,
            Role(json_str["role"]),
            json_str["nullable"],
            value_domain,
        )

    def to_dict(self) -> Dict[str, Any]:
        data_type = self.data_type
        if not inspect.isclass(self.data_type):
            data_type = self.data_type.__class__
        return {
            "name": self.name,
            "data": self.data,
            "type": component_type_name(data_type, self.value_domain),
            "role": self.role.value if self.role is not None else None,  # type: ignore[redundant-expr]
            "nullable": self.nullable,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


@dataclass
class Component:
    """
    Class representing a component of a dataset
    """

    name: str
    data_type: Type[ScalarType]
    role: Role
    nullable: bool
    value_domain: Optional["ValueDomain"] = None

    def __post_init__(self) -> None:
        if self.role == Role.IDENTIFIER and self.nullable:
            raise InputValidationException(code="0-1-1-18", name=self.name)
        if self.role == Role.VIRAL_ATTRIBUTE and self.name in RESERVED_COMPONENT_NAMES:
            raise SemanticError(
                "1-3-3-7",
                name=self.name,
                reserved=", ".join(sorted(RESERVED_COMPONENT_NAMES)),
            )

    def __eq__(self, other: Any) -> bool:
        return self.to_dict() == other.to_dict()

    def __copy__(self) -> "Component":
        # The structure checks copy every Component of their operands, and the generic
        # reduce protocol costs several times more than the same field-for-field copy
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        return clone

    def copy(self) -> "Component":
        return Component(self.name, self.data_type, self.role, self.nullable, self.value_domain)

    @classmethod
    def from_json(
        cls, json_str: Any, value_domains: Optional[Dict[str, "ValueDomain"]] = None
    ) -> "Component":
        """The component to_dict writes. Its type can name one of the value domains given, by
        name, on which the component is then defined."""
        check_input_component_name(json_str["name"])
        # Support both 'type' and 'data_type' for backward compatibility
        data_type, value_domain = named_type(
            json_str.get("type") or json_str.get("data_type"),
            value_domains,
            f"Component '{json_str['name']}'",
        )
        return cls(
            json_str["name"], data_type, Role(json_str["role"]), json_str["nullable"], value_domain
        )

    def to_dict(self) -> Dict[str, Any]:
        data_type: Any = self.data_type
        if isinstance(data_type, ScalarType):
            data_type = data_type.__class__
        type_ = (
            component_type_name(data_type, self.value_domain)
            if inspect.isclass(data_type)
            else str(data_type)
        )
        return {
            "name": self.name,
            "type": type_,
            # Need to check here for NoneType as UDO argument has it
            "role": self.role.value if self.role is not None else None,  # type: ignore[redundant-expr]
            "nullable": self.nullable,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def rename(self, new_name: str) -> None:
        self.name = new_name

    def __str__(self) -> str:
        return self.to_json()

    __repr__ = __str__


@dataclass
class Dataset:
    name: str
    components: Dict[str, Component]
    data: Optional[PandasDataFrame] = None
    persistent: bool = False

    def __post_init__(self) -> None:
        if self.data is not None:
            if len(self.components) != len(self.data.columns):
                raise InputValidationException(code="0-1-1-19", name=self.name)
            for name, _ in self.components.items():
                if name not in self.data.columns:
                    raise InputValidationException(code="0-1-1-20", comp_name=name, name=self.name)

    def enforce_dtypes(self) -> None:
        """Ensure all DataFrame column dtypes match their component DataType."""
        if self.data is None:
            return
        for comp_name, comp in self.components.items():
            if comp_name in self.data.columns:
                col = self.data[comp_name]
                if isinstance(col, pd.DataFrame):
                    continue
                target_dtype = comp.data_type.dtype()
                if str(col.dtype) != target_dtype:
                    self.data[comp_name] = col.astype(target_dtype)  # type: ignore[call-overload]

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Dataset):
            return False

        same_name = self.name == other.name
        if not same_name:
            print("\nName mismatch")
            print("result:", self.name)
            print("reference:", other.name)
        same_components = self.components == other.components
        if not same_components:
            print("\nComponents mismatch")
            result_comps = self.to_dict()["components"]
            reference_comps = other.to_dict()["components"]
            if len(result_comps) != len(reference_comps):
                print(
                    f"Shape mismatch: result:{len(result_comps)} "
                    f"!= reference:{len(reference_comps)}"
                )
                if len(result_comps) < len(reference_comps):
                    print(
                        "Missing components in result:",
                        set(reference_comps.keys()) - set(result_comps.keys()),
                    )
                else:
                    print(
                        "Additional components in result:",
                        set(result_comps.keys()) - set(reference_comps.keys()),
                    )
                return False

            diff_comps = {
                k: v
                for k, v in result_comps.items()
                if (k in reference_comps and v != reference_comps[k]) or k not in reference_comps
            }
            ref_diff_comps = {k: v for k, v in reference_comps.items() if k in diff_comps}
            print(f"Differences in components {self.name}: ")
            print("result:", json.dumps(diff_comps, indent=4))
            print("reference:", json.dumps(ref_diff_comps, indent=4))
            return False

        if self.data is None and other.data is None:
            return True
        elif self.data is None or other.data is None:
            return False
        if len(self.data) == len(other.data) == 0 and self.data.shape != other.data.shape:
            raise InputValidationException(code="0-1-1-14", dataset1=self.name, dataset2=other.name)

        for comp in self.components.values():
            type_name: str = comp.data_type.__name__.__str__()
            if type_name in ["String", "Date", "TimeInterval", "Duration"]:
                self.data[comp.name] = self.data[comp.name].fillna("").astype(str)
                other.data[comp.name] = other.data[comp.name].fillna("").astype(str)
            elif type_name == "TimePeriod":
                self.data[comp.name] = self.data[comp.name].fillna("").astype(str)
                other.data[comp.name] = other.data[comp.name].fillna("").astype(str)
                self.data[comp.name] = self.data[comp.name].map(
                    lambda x: str(TimePeriodHandler(str(x))) if x != "" else "",
                )
                other.data[comp.name] = other.data[comp.name].map(
                    lambda x: str(TimePeriodHandler(str(x))) if x != "" else "",
                )
            elif type_name in ["Integer", "Number"]:
                type_ = "int64" if type_name == "Integer" else "float64"
                self.data[comp.name] = self.data[comp.name].fillna(-1234997).astype(type_)  # type: ignore[call-overload]
                other.data[comp.name] = other.data[comp.name].fillna(-1234997).astype(type_)  # type: ignore[call-overload]
        # Sort AFTER normalization so TimePeriod identifiers sort by their canonical
        # text on both sides ("2020-M6" vs "2020-M06" would otherwise misalign rows).
        sorted_identifiers = sorted(self.get_identifiers_names())
        self.data = self.data.sort_values(by=sorted_identifiers).reset_index(drop=True)
        other.data = other.data.sort_values(by=sorted_identifiers).reset_index(drop=True)
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)
        other.data = other.data.reindex(sorted(other.data.columns), axis=1)
        try:
            # Number is compared as float64 at rtol 1e-12: two orders of margin over
            # the engines' 15-significant-digit kernel (drift <= ~1e-14) and over the
            # %.15g reference parse-back error, while real precision bugs (the old
            # DECIMAL truncation class) sit at >= 1e-10 (issue #985).
            assert_frame_equal(
                self.data,
                other.data,
                check_dtype=False,
                check_index_type=False,
                check_exact=False,
                rtol=1e-12,
                atol=0.0,
            )
        except AssertionError as e:
            if "DataFrame shape" in str(e):
                print(f"\nDataFrame shape mismatch {self.name}:")
                print("result:", self.data.shape)
                print("reference:", other.data.shape)
            # Exact-equality rescue: with float64 columns this is strictly tighter
            # than the rtol comparison above, so it can only salvage frames where
            # assert_frame_equal tripped on representation rather than value.
            diff = pd.concat([self.data, other.data]).drop_duplicates(keep=False)
            if len(diff) == 0:
                return True
            # To display actual null values instead of -1234997
            for comp in self.components.values():
                if comp.data_type.__name__.__str__() in ["Integer", "Number"]:
                    diff[comp.name] = diff[comp.name].replace(-1234997, "")
            print("\n Differences between the dataframes in", self.name)
            print(diff)
            raise e
        return True

    def get_component(self, component_name: str) -> Component:
        return self.components[component_name]

    def add_component(self, component: Component) -> None:
        self.components[component.name] = component

    def delete_component(self, component_name: str) -> None:
        self.components.pop(component_name, None)
        if self.data is not None:
            self.data.drop(columns=[component_name], inplace=True)

    def get_components(self) -> List[Component]:
        return list(self.components.values())

    def get_identifiers(self) -> List[Component]:
        return [
            component for component in self.components.values() if component.role == Role.IDENTIFIER
        ]

    def get_attributes(self) -> List[Component]:
        return [
            component for component in self.components.values() if component.role == Role.ATTRIBUTE
        ]

    def get_measures(self) -> List[Component]:
        return [
            component for component in self.components.values() if component.role == Role.MEASURE
        ]

    def get_identifiers_names(self) -> List[str]:
        return [
            name for name, component in self.components.items() if component.role == Role.IDENTIFIER
        ]

    def get_attributes_names(self) -> List[str]:
        return [
            name for name, component in self.components.items() if component.role == Role.ATTRIBUTE
        ]

    def get_measures_names(self) -> List[str]:
        return [
            name for name, component in self.components.items() if component.role == Role.MEASURE
        ]

    def get_viral_attributes(self) -> List[Component]:
        return [
            component
            for component in self.components.values()
            if component.role == Role.VIRAL_ATTRIBUTE
        ]

    def get_viral_attributes_names(self) -> List[str]:
        return [
            name
            for name, component in self.components.items()
            if component.role == Role.VIRAL_ATTRIBUTE
        ]

    def get_components_names(self) -> List[str]:
        return list(self.components.keys())

    @classmethod
    def from_json(
        cls, json_str: Any, value_domains: Optional[Dict[str, "ValueDomain"]] = None
    ) -> "Dataset":
        """The Data Set to_dict writes. The type of a Component can name one of the value
        domains given, by name, on which the Component is then defined."""
        components = {
            k: Component.from_json(v, value_domains) for k, v in json_str["components"].items()
        }
        return cls(json_str["name"], components, pd.DataFrame(json_str["data"]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "data": (self.data.to_dict(orient="records") if self.data is not None else None),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_json_datastructure(self) -> str:
        dict_dataset = self.to_dict()["components"]
        order_keys = ["name", "role", "type", "nullable"]
        # Rename data_type to type
        for k in dict_dataset:
            dict_dataset[k] = {
                ik if ik != "data_type" else "type": v for ik, v in dict_dataset[k].items()
            }

        # Order keys
        for k in dict_dataset:
            dict_dataset[k] = {ik: dict_dataset[k][ik] for ik in order_keys}
        comp_values = list(dict_dataset.values())
        ds_info = {"name": self.name, "DataStructure": comp_values}
        result = {"datasets": [ds_info]}
        return json.dumps(result, indent=2)


@dataclass
class ScalarSet:
    """
    Class representing a set of scalar values
    """

    data_type: Type[ScalarType]
    _values: List[Union[int, float, str, bool]]

    def __init__(
        self, data_type: Type[ScalarType], values: List[Union[int, float, str, bool]]
    ) -> None:
        self.data_type = data_type
        self.values = values

    @property
    def values(self) -> List[Union[int, float, str, bool]]:
        return self._values

    @values.setter
    def values(self, new_values: List[Union[int, float, str, bool]]) -> None:
        for value in new_values:
            if self.data_type and not self.data_type.check(value):
                raise InputValidationException(
                    code="0-1-2-7",
                    value=value,
                    type_=self.data_type.__name__,
                    op_type=self.__class__.__name__,
                    name="",
                )
        self._values = new_values

    def __contains__(self, item: str) -> Optional[bool]:
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        if self.data_type == DataTypes.Null:
            return None
        value = self.data_type.cast(item)
        return value_of(self.data_type, value) in self.compared_values()

    def compared_values(self) -> List[Any]:
        """The values of the set, each written as the loaders write the values of its type: a
        value is compared with them as a value of its type, whatever form it is written in
        (issue #1349)."""
        return [value_of(self.data_type, value) for value in self.values]


@dataclass
class ValueDomain:
    """
    Class representing a value domain
    """

    name: str
    type: Type[ScalarType]
    setlist: List[Union[int, float, str, bool]]

    def __post_init__(self) -> None:
        """The name must not be a basic scalar type, and the setlist must hold valid values of
        the type, each once, also in another form of it."""
        if self.name in BASIC_TYPE_NAMES:
            raise InputValidationException(code="0-1-1-39", name=self.name)
        for value in self.setlist:
            if not self._is_valid(value):
                raise InputValidationException(
                    code="0-1-1-37",
                    value=value,
                    name=self.name,
                    type=DataTypes.SCALAR_TYPES_CLASS_REVERSE[self.type],
                )
        self.setlist = [self.type.cast(value) for value in self.setlist]
        forms: Dict[Any, List[Any]] = {}
        for item, value in zip(self.setlist, self.values()):
            forms.setdefault(value, []).append(item)
        duplicated = [
            item for items in forms.values() if len(items) > 1 for item in dict.fromkeys(items)
        ]
        if duplicated:
            raise InputValidationException(code="0-1-1-21", name=self.name, duplicated=duplicated)

    def _is_valid(self, value: Any) -> bool:
        try:
            return bool(self.type.check(value))
        except InputValidationException:
            return False

    @classmethod
    def from_json(cls, json_str: str) -> str:
        if len(json_str) == 0:
            raise InputValidationException(code="0-1-1-22")

        json_info = json.loads(json_str)
        return cls.from_dict(json_info)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> Any:
        for x in ("name", "type", "setlist"):
            if x not in value:
                raise InputValidationException(code="0-1-1-23")
        if value["type"] not in SCALAR_TYPES:
            raise InputValidationException(code="0-1-1-24", type=value["type"], name=value["name"])

        return cls(value["name"], SCALAR_TYPES[value["type"]], value["setlist"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": DataTypes.SCALAR_TYPES_CLASS_REVERSE[self.type],
            "setlist": self.setlist,
        }

    def values(self) -> List[Any]:
        """The values of the domain, written as the loaders write the values of its type."""
        return [value_of(self.type, value) for value in self.setlist]

    def outside(self, values: Any) -> Any:
        """The values of the series, other than the nulls, that are not values of the domain.
        A value is compared as a value of the type, whatever form it is written in: 2024Q1
        and 2024-Q1 are the same Time_Period (issue #1349)."""
        values = values[values.notna()]
        return values[~values_of(self.type, values).isin(self.values())]

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def __eq__(self, other: Any) -> bool:
        return self.to_dict() == other.to_dict()


@dataclass
class ExternalRoutine:
    """
    Class representing an external routine, used in Eval operator
    """

    dataset_names: List[str]
    query: str
    name: str

    @classmethod
    def from_sql_query(cls, name: str, query: str) -> "ExternalRoutine":
        try:
            dataset_names = cls._extract_dataset_names(query)
            return cls(dataset_names, query, name)
        except sqlglot.errors.ParseError as e:
            raise InputValidationException(code="0-1-1-25", name=name, error=e) from e

    @classmethod
    def _extract_dataset_names(cls, query: str) -> List[str]:
        """The tables the query reads, each once: a name its WITH clause defines is not one
        (issue #1356)."""
        expression = sqlglot.parse_one(query, dialect="duckdb")
        defined = {cte.alias for cte in expression.find_all(exp.CTE)}
        tables = [t.name for t in expression.find_all(exp.Table) if t.name not in defined]
        return list(dict.fromkeys(tables))

    def query_on(self, operands: Iterable["Dataset"]) -> str:
        """The query both engines run. A double-quoted name is an identifier when it names a
        Component or an operand, or the query defines it (an alias, a WITH name or column);
        otherwise it is a text, as between single quotes. DuckDB matches identifiers ignoring
        case, so does this (issue #1356)."""
        expression = sqlglot.parse_one(self.query, dialect="duckdb")
        names = [*self.dataset_names, *(a.alias for a in expression.find_all(exp.Alias))]
        for alias in expression.find_all(exp.TableAlias):
            names += [alias.name, *(column.name for column in alias.columns)]
        names += [name for operand in operands for name in operand.components]
        identifiers = {name.lower() for name in names}
        query = self.query
        for token in reversed(DuckDB().tokenize(self.query)):
            if self.query[token.start] == '"' and token.text.lower() not in identifiers:
                text = token.text.replace("'", "''")
                query = f"{query[: token.start]}'{text}'{query[token.end + 1 :]}"
        return query

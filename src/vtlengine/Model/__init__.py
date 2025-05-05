import inspect
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from pandas import DataFrame as PandasDataFrame
from pandas._testing import assert_frame_equal

import vtlengine.DataTypes as DataTypes
from vtlengine.DataTypes import SCALAR_TYPES, ScalarType
from vtlengine.DataTypes.TimeHandling import TimePeriodHandler
from vtlengine.Exceptions import SemanticError

# from pyspark.pandas import DataFrame as SparkDataFrame, Series as SparkSeries


@dataclass
class Scalar:
    """
    Class representing a scalar value
    """

    name: str
    data_type: Type[ScalarType]
    value: Any

    @classmethod
    def from_json(cls, json_str: str) -> "Scalar":
        data = json.loads(json_str)
        return cls(data["name"], SCALAR_TYPES[data["data_type"]], data["value"])

    def __eq__(self, other: Any) -> bool:
        same_name = self.name == other.name
        same_type = self.data_type == other.data_type
        x = None if not pd.isnull(self.value) else self.value
        y = None if not pd.isnull(other.value) else other.value
        same_value = x == y
        return same_name and same_type and same_value


Role_keys = [
    "Identifier",
    "Attribute",
    "Measure",
]


class Role(Enum):
    """
    Enum class for the role of a component  (Identifier, Attribute, Measure)
    """

    IDENTIFIER = "Identifier"
    ATTRIBUTE = "Attribute"
    MEASURE = "Measure"


@dataclass
class DataComponent:
    """A component of a dataset with data"""

    name: str
    # data: Optional[Union[PandasSeries, SparkSeries]]
    data: Optional[Any]
    data_type: Type[ScalarType]
    role: Role = Role.MEASURE
    nullable: bool = True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataComponent):
            return False
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_json(cls, json_str: Any) -> "DataComponent":
        return cls(
            json_str["name"],
            None,
            SCALAR_TYPES[json_str["data_type"]],
            Role(json_str["role"]),
            json_str["nullable"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "data": self.data,
            "data_type": self.data_type,
            "role": self.role,
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

    def __post_init__(self) -> None:
        if self.role == Role.IDENTIFIER and self.nullable:
            raise ValueError(f"Identifier {self.name} cannot be nullable")

    def __eq__(self, other: Any) -> bool:
        return self.to_dict() == other.to_dict()

    def copy(self) -> "Component":
        return Component(self.name, self.data_type, self.role, self.nullable)

    @classmethod
    def from_json(cls, json_str: Any) -> "Component":
        return cls(
            json_str["name"],
            SCALAR_TYPES[json_str["data_type"]],
            Role(json_str["role"]),
            json_str["nullable"],
        )

    def to_dict(self) -> Dict[str, Any]:
        data_type = self.data_type
        if not inspect.isclass(self.data_type):
            data_type = self.data_type.__class__  # type: ignore[assignment]
        return {
            "name": self.name,
            "data_type": DataTypes.SCALAR_TYPES_CLASS_REVERSE[data_type],
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
    # data: Optional[Union[SparkDataFrame, PandasDataFrame]]
    data: Optional[PandasDataFrame] = None

    def __post_init__(self) -> None:
        if self.data is not None:
            if len(self.components) != len(self.data.columns):
                raise ValueError(
                    "The number of components must match the number of columns in the data"
                )
            for name, _ in self.components.items():
                if name not in self.data.columns:
                    raise ValueError(f"Component {name} not found in the data")

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
            raise SemanticError("0-1-1-14", dataset1=self.name, dataset2=other.name)

        self.data.fillna("", inplace=True)
        other.data.fillna("", inplace=True)
        sorted_identifiers = sorted(self.get_identifiers_names())
        self.data = self.data.sort_values(by=sorted_identifiers).reset_index(drop=True)
        other.data = other.data.sort_values(by=sorted_identifiers).reset_index(drop=True)
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)
        other.data = other.data.reindex(sorted(other.data.columns), axis=1)
        for comp in self.components.values():
            type_name: str = comp.data_type.__name__.__str__()
            if type_name in ["String", "Date"]:
                self.data[comp.name] = self.data[comp.name].astype(str)
                other.data[comp.name] = other.data[comp.name].astype(str)
            elif type_name == "TimePeriod":
                self.data[comp.name] = self.data[comp.name].astype(str)
                other.data[comp.name] = other.data[comp.name].astype(str)
                self.data[comp.name] = self.data[comp.name].map(
                    lambda x: str(TimePeriodHandler(str(x))) if x != "" else "",
                    na_action="ignore",
                )
                other.data[comp.name] = other.data[comp.name].map(
                    lambda x: str(TimePeriodHandler(str(x))) if x != "" else "",
                    na_action="ignore",
                )
            elif type_name in ["Integer", "Number"]:
                type_ = "int64" if type_name == "Integer" else "float32"
                # We use here a number to avoid errors on equality on empty strings
                self.data[comp.name] = (
                    self.data[comp.name].replace("", -1234997).astype(type_)  # type: ignore[call-overload]
                )
                other.data[comp.name] = (
                    other.data[comp.name].replace("", -1234997).astype(type_)  # type: ignore[call-overload]
                )
        try:
            assert_frame_equal(
                self.data,
                other.data,
                check_dtype=False,
                check_index_type=False,
                check_datetimelike_compat=True,
                check_exact=False,
                rtol=0.01,
                atol=0.01,
            )
        except AssertionError as e:
            if "DataFrame shape" in str(e):
                print(f"\nDataFrame shape mismatch {self.name}:")
                print("result:", self.data.shape)
                print("reference:", other.data.shape)
            # Differences between the dataframes
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
        if component.name in self.components:
            raise ValueError(f"Component with name {component.name} already exists")
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

    def get_components_names(self) -> List[str]:
        return list(self.components.keys())

    @classmethod
    def from_json(cls, json_str: Any) -> "Dataset":
        components = {k: Component.from_json(v) for k, v in json_str["components"].items()}
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
    values: List[Union[int, float, str, bool]]

    def __contains__(self, item: str) -> Optional[bool]:
        if isinstance(item, float) and item.is_integer():
            item = int(item)
        if self.data_type == DataTypes.Null:
            return None
        value = self.data_type.cast(item)
        return value in self.values


@dataclass
class ValueDomain:
    """
    Class representing a value domain
    """

    name: str
    type: Type[ScalarType]
    setlist: List[Union[int, float, str, bool]]

    def __post_init__(self) -> None:
        if len(set(self.setlist)) != len(self.setlist):
            duplicated = [item for item, count in Counter(self.setlist).items() if count > 1]
            raise ValueError(
                f"The setlist must have unique values. Duplicated values: {duplicated}"
            )

        # Cast values to the correct type
        self.setlist = [self.type.cast(value) for value in self.setlist]

    @classmethod
    def from_json(cls, json_str: str) -> str:
        if len(json_str) == 0:
            raise ValueError("Empty JSON string for ValueDomain")

        json_info = json.loads(json_str)
        return cls.from_dict(json_info)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> Any:
        for x in ("name", "type", "setlist"):
            if x not in value:
                raise Exception("Invalid format for ValueDomain. Requires name, type and setlist.")
        if value["type"] not in SCALAR_TYPES:
            raise ValueError(f"Invalid data type {value['type']} for ValueDomain {value['name']}")

        return cls(value["name"], SCALAR_TYPES[value["type"]], value["setlist"])

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.type.__name__, "setlist": self.setlist}

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
        dataset_names = cls._extract_dataset_names(query)
        return cls(dataset_names, query, name)

    @classmethod
    def _extract_dataset_names(cls, query: str) -> List[str]:
        expression = sqlglot.parse_one(query, read="sqlite")
        tables_info = list(expression.find_all(exp.Table))
        dataset_names = [t.name for t in tables_info]
        return dataset_names

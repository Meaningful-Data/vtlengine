import inspect
import json
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import duckdb
import pandas as pd
import sqlglot
import sqlglot.expressions as exp
from duckdb.duckdb import DuckDBPyRelation  # type: ignore[import-untyped]
from pandas._testing import assert_frame_equal

import vtlengine.DataTypes as DataTypes
from vtlengine.connection import con
from vtlengine.DataTypes import SCALAR_TYPES, ScalarType
from vtlengine.duckdb.duckdb_utils import clean_execution_graph, normalize_data, quote_cols
from vtlengine.Exceptions import DataLoadError

from .relation_proxy import RelationProxy

INDEX_COL = "__index__"


def __duckdb_repr__(self: Any) -> str:
    """
    DuckDB internal repr based on pandas repr
    """
    return f"<DuckDBPyRelation: {self.df().__repr__()}>"


DuckDBPyRelation.__repr__ = __duckdb_repr__


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
            raise DataLoadError(code="0-1-1-4", name=self.name, null_identifier=self.name)

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
class DataComponent:
    """A component of a dataset with data"""

    name: str
    _data: Optional[Union[RelationProxy]]
    data_type: Type[ScalarType]
    role: Role = Role.MEASURE
    nullable: bool = True

    def __init__(
        self,
        name: str,
        data: Optional[Union[pd.DataFrame, DuckDBPyRelation, RelationProxy]],
        data_type: Type[ScalarType],
        role: Role = Role.MEASURE,
        nullable: bool = True,
    ) -> None:
        self.name = name
        self.data = data
        self.data_type = data_type
        self.role = role
        self.nullable = nullable
        self.__post_init__()

    def __post_init__(self) -> None:
        if isinstance(self.data, pd.Series):
            self.data = con.from_df(self.data.to_frame())
        if isinstance(self.data, DuckDBPyRelation):
            self.data = RelationProxy(self.data)

    @property
    def data(self) -> Optional[Union[RelationProxy]]:
        return self._data

    @data.setter
    def data(self, value: Optional[Union[pd.DataFrame, DuckDBPyRelation]]) -> None:
        if isinstance(value, pd.DataFrame):
            value = con.from_df(value)
        if isinstance(value, DuckDBPyRelation):
            self._data = RelationProxy(value)
        elif isinstance(value, RelationProxy):
            self._data = value
        elif value is None:
            self._data = None
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, DuckDBPyRelation, RelationProxy or None"
            )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataComponent):
            return False

        if self.name != other.name:
            return False

        if self.data_type != other.data_type:
            return False

        if self.role != other.role:
            return False

        if self.nullable != other.nullable:
            return False

        # Both data are None
        if self.data is None and other.data is None:
            return True

        # One of the data is None
        if self.data is None or other.data is None:
            return False

        cols_self = [c for c in self.data.columns if c != INDEX_COL]
        cols_other = [c for c in other.data.columns if c != INDEX_COL]
        if cols_self != cols_other:
            return False
        col = cols_self[0]
        sorted_self = self.data.order(col)
        sorted_other = other.data.order(col)
        diff = sorted_self.except_(sorted_other).union(sorted_other.except_(sorted_self))

        # Lazy comparison: check if any difference exists
        return not diff.limit(1).df().shape[0] > 0

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

    @property
    def df(self) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()
        if isinstance(self.data, RelationProxy):
            df = self.data.df(10)
        else:
            df = self.data.limit(10).df()
            if INDEX_COL in df.columns:
                df = df.set_index(INDEX_COL)
        return df


@dataclass
class Dataset:
    name: str
    components: Dict[str, Component]
    _data: Optional[RelationProxy] = None

    @property
    def data(self) -> Optional[RelationProxy]:
        return self._data

    @data.setter
    def data(self, value: Optional[Union[pd.DataFrame, DuckDBPyRelation]]) -> None:
        if isinstance(value, pd.DataFrame):
            value = con.from_df(value)
        if isinstance(value, DuckDBPyRelation):
            self._data = RelationProxy(value)
        elif isinstance(value, RelationProxy):
            self._data = value
        elif value is None:
            self._data = None
        else:
            raise ValueError(
                "Data must be a pandas DataFrame, DuckDBPyRelation, RelationProxy or None"
            )

    def __init__(
        self,
        name: str,
        components: Dict[str, Component],
        data: Optional[Union[pd.DataFrame, DuckDBPyRelation, RelationProxy]] = None,
    ) -> None:
        self.name = name
        self.components = components
        self.data = data
        self.__post_init__()

    def __post_init__(self) -> None:
        if isinstance(self.data, pd.DataFrame):
            self.data = con.from_df(self.data)
        if isinstance(self.data, DuckDBPyRelation):
            self.data = RelationProxy(self.data)

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

        # Check names
        if self.name != other.name:
            print(f"Name mismatch: {self.name} != {other.name}")
            return False

        # Check components
        if self.components != other.components:
            print("Components mismatch")
            diff_comps = {
                k: v
                for k, v in self.components.items()
                if k not in other.components or v != other.components[k]
            }
            print(f"Differences in components: {diff_comps}")
            return False

        # Check both data are None, they are equal
        if self.data is None and other.data is None:
            return True

        # If only one is empty they are not equal
        if self.data is None or other.data is None:
            return False

        self_cols_set = {c for c in self.data.columns if c != INDEX_COL}
        other_cols_set = {c for c in other.data.columns if c != INDEX_COL}
        if self_cols_set != other_cols_set:
            print("Column mismatch")
            return False

        if isinstance(self.data, pd.DataFrame) and isinstance(other.data, pd.DataFrame):
            # If both data are pandas DataFrames, compare them directly
            assert_frame_equal(self.data, other.data, check_dtype=False)
            return True
        elif isinstance(self.data, pd.DataFrame):
            self._to_duckdb()
        elif isinstance(other.data, pd.DataFrame):
            other._to_duckdb()

        # Ensuring the dataset is materialized
        self.data = clean_execution_graph(self.data)
        other.data = clean_execution_graph(other.data)
        # Round double values to avoid precision issues
        self.data, other.data = normalize_data(self.data, other.data)

        # Order by identifiers
        self_cols = quote_cols([c for c in self.data.columns if c != INDEX_COL])
        sorted_self = self.data.project(", ".join(self_cols), include_index=False)
        sorted_other = other.data.project(", ".join(self_cols), include_index=False)

        # Comparing data using DuckDB
        diff = sorted_self.except_(sorted_other).union(sorted_other.except_(sorted_self))
        # Loading only the first row to check if there are any internal structure differences
        # (avoiding memory overload)
        if diff.limit(1).execute().fetchone() is not None:
            print(f"\nData mismatch on dataset: {self.name} and {other.name}\n")
            print("\nSELF\n", self.data)
            print("\nOTHER\n", other.data)
            diff.show()
            return False
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
            self.data.drop(columns=component_name)

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
        data = None
        if "data" in json_str and json_str["data"] is not None:
            # Convert JSON data directly to DuckDB relation
            data = con.sql(f"SELECT * FROM json('{json.dumps(json_str['data'])}')")
        return cls(json_str["name"], components, data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "data": (
                [
                    dict(dict(zip([c for c in self.data.columns if c != INDEX_COL], row)).items())
                    for row in self.data.project(
                        ", ".join(quote_cols([c for c in self.data.columns if c != INDEX_COL]))
                    )
                    .execute()
                    .fetchall()
                ]
                if self.data is not None
                else None
            ),
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

    def __repr__(self) -> str:
        data = "None"
        if isinstance(self.data, pd.DataFrame):
            data = repr(self.data).replace("<NA>", "None")
        elif isinstance(self.data, RelationProxy):
            try:
                data = self.data.relation.limit(30).df()
            except duckdb.Error as e:
                raise DataLoadError.map_duckdb_error(e)
        elif isinstance(self.data, DuckDBPyRelation):
            try:
                tmp = self.data
                if INDEX_COL in tmp.columns:
                    tmp = tmp.set_index(INDEX_COL)
                data = tmp
            except duckdb.Error as e:
                raise DataLoadError.map_duckdb_error(e)
        return (
            f"Dataset("
            f"\nname={self.name},"
            f"\ncomponents={list(self.components.keys())},"
            f"\ndata=\n{data})"
        )

    @property
    def df(self) -> pd.DataFrame:
        if self.data is None:
            return pd.DataFrame()
        if isinstance(self.data, RelationProxy):
            return self.data.df(30)
        df = self.data.limit(30).df()
        if INDEX_COL in df.columns:
            df = df.set_index(INDEX_COL)
        return df

    def _to_duckdb(self) -> DuckDBPyRelation:
        """
        Convert underlying data to DuckDB relation if it is a pandas DataFrame.
        (RelationProxy already wraps a DuckDBPyRelation.)
        """
        if self.data is None or isinstance(self.data, (RelationProxy, DuckDBPyRelation)):
            return
        self.data = con.from_df(self.data)


@dataclass
class ScalarSet:
    """
    Class representing a set of scalar values
    """

    data_type: Type[ScalarType]
    values: Union[List[Union[int, float, str, bool]], DuckDBPyRelation]

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

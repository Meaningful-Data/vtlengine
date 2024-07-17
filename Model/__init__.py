import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame as PandasDataFrame, Series as PandasSeries
from pandas._testing import assert_frame_equal
from pyspark.pandas import DataFrame as SparkDataFrame, Series as SparkSeries

from DataTypes import SCALAR_TYPES, ScalarType


@dataclass
class Scalar:
    """
    Class representing a scalar value
    """
    name: str
    data_type: ScalarType
    value: Optional[Union[int, float, str, bool]]

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(data['name'], data['value'])


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
    data: Optional[Union[PandasSeries, SparkSeries]]
    data_type: ScalarType
    role: Role = Role.MEASURE
    nullable: bool = True

    def __eq__(self, other):
        if not isinstance(other, DataComponent):
            return False
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_json(cls, json_str):
        return cls(json_str['name'], None, SCALAR_TYPES[json_str['data_type']],
                   Role(json_str['role']), json_str['nullable'])

    def to_dict(self):
        return {
            'name': self.name,
            'data': self.data,
            'data_type': self.data_type,
            'role': self.role,
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)


@dataclass
class Component:
    """
    Class representing a component of a dataset
    """
    name: str
    data_type: ScalarType
    role: Role
    nullable: bool

    def __post_init__(self):
        if self.role == Role.IDENTIFIER:
            if self.nullable:
                raise ValueError("An Identifier cannot be nullable")

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def copy(self):
        return Component(self.name, self.data_type, self.role, self.nullable)

    @classmethod
    def from_json(cls, json_str):
        return cls(json_str['name'], SCALAR_TYPES[json_str['data_type']], Role(json_str['role']),
                   json_str['nullable'])

    def to_dict(self):
        return {
            'name': self.name,
            'data_type': self.data_type.__name__,
            'role': self.role.value,
            'nullable': self.nullable
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def rename(self, new_name: str):
        self.name = new_name


@dataclass
class Dataset:
    name: str
    components: Dict[str, Component]
    data: Optional[Union[SparkDataFrame, PandasDataFrame]]

    def __post_init__(self):
        if self.data is not None:
            if len(self.components) != len(self.data.columns):
                raise ValueError(
                    "The number of components must match the number of columns in the data")

    def __eq__(self, other):
        if not isinstance(other, Dataset):
            return False

        same_name = self.name == other.name
        same_components = self.components == other.components

        if isinstance(self.data, SparkDataFrame):
            self.data = self.data.to_pandas()
        if isinstance(other.data, SparkDataFrame):
            other.data = other.data.to_pandas()
        self.data.fillna("", inplace=True)
        other.data.fillna("", inplace=True)
        self.data = self.data.sort_values(by=list(self.data.columns)).reset_index(drop=True)
        other.data = other.data.sort_values(by=list(other.data.columns)).reset_index(drop=True)
        try:
            assert_frame_equal(self.data, other.data, check_dtype=False, check_like=True)
            same_data = True
        except AssertionError:
            same_data = False
        return same_name and same_components and same_data

    def get_component(self, component_name: str) -> Component:
        return self.components[component_name]

    def add_component(self, component: Component):
        if component.name in self.components:
            raise ValueError(f"Component with name {component.name} already exists")
        self.components[component.name] = component

    def delete_component(self, component_name: str):
        self.components.pop(component_name, None)

    def get_identifiers(self) -> List[Component]:
        return [component for component in self.components.values() if
                component.role == Role.IDENTIFIER]

    def get_attributes(self) -> List[Component]:
        return [component for component in self.components.values() if
                component.role == Role.ATTRIBUTE]

    def get_measures(self) -> List[Component]:
        return [component for component in self.components.values() if
                component.role == Role.MEASURE]

    def get_identifiers_names(self) -> List[str]:
        return [name for name, component in self.components.items() if
                component.role == Role.IDENTIFIER]

    def get_attributes_names(self) -> List[str]:
        return [name for name, component in self.components.items() if
                component.role == Role.ATTRIBUTE]

    def get_measures_names(self) -> List[str]:
        return [name for name, component in self.components.items() if
                component.role == Role.MEASURE]

    def get_components_names(self) -> List[str]:
        return list(self.components.keys())

    def rename_component(self, old_name: str, new_name: str):
        if old_name not in self.components:
            raise ValueError(f"Component with name {old_name} does not exist")
        if new_name in self.components:
            raise ValueError(f"Component with name {new_name} already exists")
        self.components[new_name] = self.components.pop(old_name)

    @classmethod
    def from_json(cls, json_str):
        components = {k: Component.from_json(v) for k, v in json_str['components'].items()}
        return cls(json_str['name'], components, pd.DataFrame(json_str['data']))

    def to_dict(self):
        return {
            'name': self.name,
            'components': {k: v.to_dict() for k, v in self.components.items()},
            'data': self.data.to_dict(orient='records')
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)


@dataclass
class ScalarSet:
    """
    Class representing a set of scalar values
    """
    data_type: ScalarType
    values: List[Union[int, float, str, bool]]

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union, Dict

from pandas import DataFrame as PandasDataFrame, Series as PandasSeries
from pyspark.pandas import DataFrame as SparkDataFrame, Series as SparkSeries

from DataTypes import ScalarType


@dataclass
class Scalar:
    """
    Class representing a scalar value
    """
    name: str
    data_type: ScalarType
    value: Optional[Union[int, float, str, bool]]


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

    def toJSON(self):
        return {
            "name": self.name,
            "data_type": self.data_type.__class__.__name__,
            "role": self.role,
            "nullable": self.nullable
        }


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

    def get_component(self, component_name: str) -> Component:
        return self.components[component_name]

    def add_component(self, component: Component):
        if component.name in self.components:
            raise ValueError(f"Component with name {component.name} already exists")
        self.components[component.name] = component

    def delete_component(self, component_name: str):
        self.components.pop(component_name, None)

    def get_identifiers(self) -> List[Component]:
        return [component for component in self.components.values() if component.role == Role.IDENTIFIER]

    def get_attributes(self) -> List[Component]:
        return [component for component in self.components.values() if component.role == Role.ATTRIBUTE]

    def get_measures(self) -> List[Component]:
        return [component for component in self.components.values() if component.role == Role.MEASURE]

    def get_identifiers_names(self) -> List[str]:
        return [name for name, component in self.components.items() if component.role == Role.IDENTIFIER]

    def get_attributes_names(self) -> List[str]:
        return [name for name, component in self.components.items() if component.role == Role.ATTRIBUTE]

    def get_measures_names(self) -> List[str]:
        return [name for name, component in self.components.items() if component.role == Role.MEASURE]

    def rename_component(self, old_name: str, new_name: str):
        if old_name not in self.components:
            raise ValueError(f"Component with name {old_name} does not exist")
        if new_name in self.components:
            raise ValueError(f"Component with name {new_name} already exists")
        self.components[new_name] = self.components.pop(old_name)

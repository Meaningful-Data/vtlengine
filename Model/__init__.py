from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from pandas import DataFrame as PandasDataFrame
from pyspark.pandas import DataFrame as SparkDataFrame

from DataTypes import ScalarType


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
    data_type: ScalarType
    role: Role
    nullable: bool

    def __post_init__(self):
        if self.role == Role.IDENTIFIER:
            if self.nullable:
                raise ValueError("An Identifier cannot be nullable")


@dataclass
class Dataset:
    components: List[Component]
    data: Optional[Union[SparkDataFrame, PandasDataFrame]]

    def __post_init__(self):
        if len(self.components) != len(set([component.name for component in self.components])):
            raise ValueError("The names of the components must be unique")
        if self.data is not None:
            if len(self.components) != len(self.data.columns):
                raise ValueError(
                    "The number of components must match the number of columns in the data")

import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import sqlglot
import sqlglot.expressions as exp
import sqlparse
from pandas import DataFrame as PandasDataFrame, Series as PandasSeries
from pandas._testing import assert_frame_equal
from pyspark.pandas import DataFrame as SparkDataFrame, Series as SparkSeries

import DataTypes
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

    def __eq__(self, other):
        same_name = self.name == other.name
        same_type = self.data_type == other.data_type
        x = None if not pd.isnull(self.value) else self.value
        y = None if not pd.isnull(other.value) else other.value
        same_value = x == y
        return same_name and same_type and same_value


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
        if self.role == Role.IDENTIFIER and self.nullable:
            raise ValueError(f"Identifier {self.name} cannot be nullable")

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
            for name, component in self.components.items():
                if name not in self.data.columns:
                    raise ValueError(f"Component {name} not found in the data")
                if component.data_type == DataTypes.TimePeriod or component.data_type == DataTypes.TimeInterval:
                    self.data[name] = self.data[name].map(self.refactor_time_period,
                                                          na_action="ignore")

    def __eq__(self, other):
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
            print("result:", json.dumps(self.to_dict()['components'], indent=4))
            print("reference:", json.dumps(other.to_dict()['components'], indent=4))
            return False

        if isinstance(self.data, SparkDataFrame):
            self.data = self.data.to_pandas()
        if isinstance(other.data, SparkDataFrame):
            other.data = other.data.to_pandas()
        if len(self.data) == len(other.data) == 0:
            assert self.data.shape == other.data.shape

        self.data.fillna("", inplace=True)
        other.data.fillna("", inplace=True)
        # self.data = self.data.sort_values(by=self.get_identifiers_names()).reset_index(drop=True)
        # other.data = other.data.sort_values(by=other.get_identifiers_names().sort()).reset_index(drop=True)
        sorted_identifiers = sorted(self.get_identifiers_names())
        self.data = self.data.sort_values(by=sorted_identifiers).reset_index(drop=True)
        other.data = other.data.sort_values(by=sorted_identifiers).reset_index(drop=True)
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)
        other.data = other.data.reindex(sorted(other.data.columns), axis=1)
        for comp in self.components.values():
            if comp.data_type.__name__ in ['String', 'Date', 'TimePeriod', 'TimeInterval']:
                self.data[comp.name] = self.data[comp.name].astype(str)
                other.data[comp.name] = other.data[comp.name].astype(str)
            elif comp.data_type.__name__ in ['Integer', 'Float']:
                if comp.data_type.__name__ == 'Integer':
                    type_ = "int64"
                else:
                    type_ = "float64"
                    # We use here a number to avoid errors on equality on empty strings
                self.data[comp.name] = self.data[comp.name].replace("", -1234997).astype(type_)
                other.data[comp.name] = other.data[comp.name].replace("", -1234997).astype(type_)
        try:
            assert_frame_equal(self.data, other.data, check_dtype=False, check_like=True,
                               check_index_type=False, check_datetimelike_compat=True)
        except AssertionError as e:
            if "DataFrame shape" in str(e):
                print("\nDataFrame shape mismatch")
                print("result:", self.data.shape)
                print("reference:", other.data.shape)
            # Differences between the dataframes
            diff = pd.concat([self.data, other.data]).drop_duplicates(keep=False)
            # To display actual null values instead of -1234997
            for comp in self.components.values():
                if comp.data_type.__name__ in ['Integer', 'Float']:
                    diff[comp.name] = diff[comp.name].replace(-1234997, "")
            print("\n Differences between the dataframes")
            print(diff)
            raise e
        return True

    def get_component(self, component_name: str) -> Component:
        return self.components[component_name]

    def add_component(self, component: Component):
        if component.name in self.components:
            raise ValueError(f"Component with name {component.name} already exists")
        self.components[component.name] = component

    def delete_component(self, component_name: str):
        self.components.pop(component_name, None)
        if self.data is not None:
            self.data.drop(columns=[component_name], inplace=True)

    def get_components(self) -> List[Component]:
        return list(self.components.values())

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

    def refactor_time_period(self, date: str):
        if not isinstance(date, str):
            return date
        if re.match(r"^\d{1,4}M(0?[1-9]|1[0-2])$", date):
            year, month = date.split("M")
            return "{}-M{}".format(year, month)
        if re.match(r"^\d{1,4}Q[1-4]$", date):
            year, quarter = date.split("Q")
            return "{}-Q{}".format(year, quarter)
        if re.match(r"^\d{1,4}S[1-2]$", date):
            year, semester = date.split("S")
            return "{}-S{}".format(year, semester)
        if re.match(r"^\d{1,4}M(0?[1-9]|1[0-2])/\d{1,4}M(0?[1-9]|1[0-2])$", date):
            date1, date2 = date.split("/")
            return "{}/{}".format(self.refactor_time_period(date1),
                                  self.refactor_time_period(date2))
        return date


@dataclass
class ScalarSet:
    """
    Class representing a set of scalar values
    """
    data_type: ScalarType
    values: List[Union[int, float, str, bool]]


@dataclass
class ValueDomain:
    """
    Class representing a value domain
    """
    name: str
    type: ScalarType
    setlist: List[Union[int, float, str, bool]]

    def __post_init__(self):
        if len(set(self.setlist)) != len(self.setlist):
            duplicated = [item for item, count in Counter(self.setlist).items() if count > 1]
            raise ValueError(
                f"The setlist must have unique values. Duplicated values: {duplicated}")

        # Cast values to the correct type
        self.setlist = [self.type.cast(value) for value in self.setlist]

    @classmethod
    def from_json(cls, json_str: str):
        if len(json_str) == 0:
            raise ValueError("Empty JSON string for ValueDomain")

        json_info = json.loads(json_str)

        if json_info['type'] not in SCALAR_TYPES:
            raise ValueError(
                f"Invalid data type {json_info['type']} for ValueDomain {json_info['name']}")

        return cls(json_info['name'], SCALAR_TYPES[json_info['type']], json_info['setlist'])

    def to_dict(self):
        return {
            'name': self.name,
            'type': self.type.__name__,
            'setlist': self.setlist
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def __eq__(self, other):
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
    def from_sql_query(cls, name: str, query: str):
        dataset_names = cls._extract_dataset_names(query)
        return cls(dataset_names, query, name)

    @classmethod
    def _get_tables(cls, d):
        """Using https://stackoverflow.com/questions/69684115/python-library-for-extracting-table-names-from-from-clause-in-sql-statetments"""
        f = False
        for i in getattr(d, 'tokens', []):
            if isinstance(i, sqlparse.sql.Token) and i.value.lower() == 'from':
                f = True
            elif isinstance(i, (sqlparse.sql.Identifier, sqlparse.sql.IdentifierList)) and f:
                f = False
                if not any(
                        isinstance(x, sqlparse.sql.Parenthesis) or 'select' in x.value.lower()
                        for x in getattr(i, 'tokens', [])):
                    fr = ''.join(str(j) for j in i if j.value not in {'as', '\n'})
                    for t in re.findall('(?:\w+\.\w+|\w+)\s+\w+|(?:\w+\.\w+|\w+)', fr):
                        yield {'table': (t1 := t.split())[0],
                               'alias': None if len(t1) < 2 else t1[-1]}
            yield from cls._get_tables(i)

    @classmethod
    def _extract_dataset_names(cls, query) -> List[str]:
        expression = sqlglot.parse_one(query, read="sqlite")
        tables_info = list(expression.find_all(exp.Table))
        dataset_names = [t.name for t in tables_info]
        return dataset_names

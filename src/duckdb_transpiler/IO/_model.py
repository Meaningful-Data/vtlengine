from typing import List, Union

from duckdb_transpiler.Model import Scalar
from vtlengine.Model import Dataset


class Query:
    def __init__(self, name: str, sql, inputs: List[str], structure: Union[Dataset, Scalar]):
        self.name = name
        self.sql = sql
        self.inputs = inputs
        self.structure = structure

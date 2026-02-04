from typing import List, Union

from vtlengine.Model import Dataset, Scalar


class Query:
    def __init__(
        self,
        name: str,
        sql: str,
        inputs: List[str],
        structure: Union[Dataset, Scalar],
        is_persistent: bool = False,
    ):
        self.name = name
        self.sql = sql
        self.inputs = inputs
        self.structure = structure
        self.is_persistent = is_persistent

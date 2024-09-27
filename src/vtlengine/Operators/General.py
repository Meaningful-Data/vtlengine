from typing import Dict, List

import pandas as pd
from vtlengine.DataTypes import COMP_NAME_MAPPING
from vtlengine.Operators import Binary, Unary
from pandasql import sqldf

from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Dataset, ExternalRoutine, Role, Component, DataComponent


class Membership(Binary):
    """Membership operator class.

    It inherits from Binary class and has the following class methods:

    Class methods:
        Validate: Checks if str right operand is actually within the Dataset.
        Evaluate: Checks validate operation and return the dataset to perform it.
    """

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: str):
        if right_operand not in left_operand.components:
            raise SemanticError("1-1-1-10", op=cls.op, comp_name=right_operand,
                                dataset_name=left_operand.name)

        component = left_operand.components[right_operand]
        if component.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            right_operand = COMP_NAME_MAPPING[component.data_type]
            left_operand.components[right_operand] = Component(name=right_operand,
                                                               data_type=component.data_type,
                                                               role=Role.MEASURE,
                                                               nullable=component.nullable)
            if left_operand.data is not None:
                left_operand.data[right_operand] = left_operand.data[component.name]
            left_operand.data[right_operand] = left_operand.data[component.name]

        result_components = {name: comp for name, comp in left_operand.components.items()
                             if comp.role == Role.IDENTIFIER or comp.name == right_operand}
        result_dataset = Dataset(name="result", components=result_components, data=None)
        return result_dataset

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str,
                 is_from_component_assignment=False) -> Dataset:
        result_dataset = cls.validate(left_operand, right_operand)
        if is_from_component_assignment:
            return DataComponent(name=right_operand,
                                 data_type=left_operand.components[right_operand].data_type,
                                 role=Role.MEASURE,
                                 nullable=left_operand.components[right_operand].nullable,
                                 data=left_operand.data[right_operand])
        result_dataset.data = left_operand.data[list(result_dataset.components.keys())]
        return result_dataset


class Alias(Binary):
    """Alias operator class
    It inherits from Binary class, and has the following class methods:

    Class methods:
        Validate: Ensures the name given in the right operand is different from the name of the Dataset.
        Evaluate: Checks if the data between both operators are the same.
    """

    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: str):
        new_name = right_operand if isinstance(right_operand, str) else right_operand.name
        if new_name != left_operand.name and new_name in left_operand.get_components_names():
            raise SemanticError("1-3-1", alias=new_name)
        return Dataset(name=new_name, components=left_operand.components, data=None)

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result = cls.validate(left_operand, right_operand)
        result.data = left_operand.data
        return result


class Eval(Unary):
    """Eval operator class
    It inherits from Unary class and has the following class methods

    Class methods:
        Validate: checks if the external routine name is the same as the operand name, which must be a Dataset.
        Evaluate: Checks if the operand and the output is actually a Dataset.

    """

    @staticmethod
    def _execute_query(query: str, dataset_names: List[str],
                       data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        for ds_name in dataset_names:
            locals()[ds_name] = data[ds_name]

        try:
            # df_result = duckdb.query(query).to_df()
            df_result = sqldf(query=query, env=locals(), db_uri='sqlite:///:memory:')
        except Exception as e:
            raise Exception(f"Error executing SQL query: {e}")
        for ds_name in dataset_names:
            del locals()[ds_name]

        return df_result

    @classmethod
    def validate(cls, operands: Dict[str, Dataset], external_routine: ExternalRoutine,
                 output: Dataset) -> Dataset:

        empty_data_dict = {}
        for ds_name in external_routine.dataset_names:
            if ds_name not in operands:
                raise ValueError(f"External Routine dataset {ds_name} "
                                 f"is not present in Eval operands")
            empty_data = pd.DataFrame(
                columns=[comp.name for comp in operands[ds_name].components.values()])
            empty_data_dict[ds_name] = empty_data

        df = cls._execute_query(external_routine.query, external_routine.dataset_names,
                                empty_data_dict)
        component_names = [name for name in df.columns]
        for comp_name in component_names:
            if comp_name not in output.components:
                raise SemanticError("1-1-1-10", op=cls.op, comp_name=comp_name,
                                    dataset_name=df.name)

        for comp_name in output.components:
            if comp_name not in component_names:
                raise ValueError(f"Component {comp_name} not found in External Routine result")

        output.name = external_routine.name

        return output

    @classmethod
    def evaluate(cls, operands: Dict[str, Dataset], external_routine: ExternalRoutine,
                 output: Dataset) -> Dataset:
        result = cls.validate(operands, external_routine, output)

        operands_data_dict = {ds_name: operands[ds_name].data
                              for ds_name in operands}

        result.data = cls._execute_query(external_routine.query,
                                         external_routine.dataset_names,
                                         operands_data_dict)

        return result

import duckdb
import pandas as pd

from DataTypes import CAST_MAPPING, COMP_NAME_MAPPING
from Model import Dataset, ExternalRoutine, Role, Component
from Operators import Binary, Unary


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
            raise ValueError(f"Component {right_operand} not found in dataset {left_operand.name}")

        component = left_operand.components[right_operand]
        if component.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            right_operand = COMP_NAME_MAPPING[component.data_type]
            left_operand.components[right_operand] = Component(name=right_operand,
                                                               data_type=component.data_type,
                                                               role=Role.MEASURE,
                                                               nullable=component.nullable)
            left_operand.data[right_operand] = left_operand.data[component.name]
        result_components = {name: comp for name, comp in left_operand.components.items()
                             if comp.role == Role.IDENTIFIER or comp.name == right_operand}
        result_dataset = Dataset(name="result", components=result_components, data=None)
        return result_dataset

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result_dataset = cls.validate(left_operand, right_operand)
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
        if left_operand.name == right_operand:
            raise ValueError("Alias operation requires different names")
        return Dataset(name=right_operand, components=left_operand.components, data=None)

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
    def _execute_query(query: str, dataset_name: str, data: pd.DataFrame) -> pd.DataFrame:
        locals()[dataset_name] = data

        try:
            df_result = duckdb.query(query).to_df()
        except Exception as e:
            raise Exception(f"Error validating SQL query with duckdb: {e}")
        del locals()[dataset_name]

        return df_result

    @classmethod
    def validate(cls,
                 operand: Dataset,
                 external_routine: ExternalRoutine,
                 output: Dataset) -> Dataset:
        if external_routine.dataset_name != operand.name:
            raise ValueError(f"Dataset name {external_routine.dataset_name} does not match "
                             f"operand name {operand.name}")

        empty_data = pd.DataFrame(columns=[comp.name for comp in operand.components.values()])

        df = cls._execute_query(external_routine.query, external_routine.dataset_name, empty_data)
        component_names = [name for name in df.columns]
        for comp_name in component_names:
            if comp_name not in output.components:
                raise ValueError(f"Component {comp_name} not found in output dataset")

        output.name = external_routine.name

        return output

    @classmethod
    def evaluate(cls,
                 operand: Dataset,
                 external_routine: ExternalRoutine,
                 output: Dataset) -> Dataset:
        result = cls.validate(operand, external_routine, output)

        result.data = cls._execute_query(external_routine.query,
                                         external_routine.dataset_name,
                                         operand.data)

        return result

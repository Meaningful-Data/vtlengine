from typing import Dict, List

import pandas as pd
from pandasql import sqldf

from DataTypes import COMP_NAME_MAPPING
from Model import Dataset, ExternalRoutine, Role, Component
from Operators import Binary, Unary


class Membership(Binary):

    @classmethod
    def validate(cls, left: Dataset, right: str):
        if right not in left.components:
            raise ValueError(f"Component {right} not found in dataset {left.name}")

        component = left.components[right]
        if component.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            right = COMP_NAME_MAPPING[component.data_type]
            left.components[right] = Component(
                name=right,
                data_type=component.data_type,
                role=Role.MEASURE,
                nullable=component.nullable,
            )
            left.data[right] = left.data[component.name]
        result_components = {
            name: comp
            for name, comp in left.components.items()
            if comp.role == Role.IDENTIFIER or comp.name == right
        }
        result_dataset = Dataset(name="result", components=result_components, data=None)
        return result_dataset

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result_dataset = cls.validate(left_operand, right_operand)
        result_dataset.data = left_operand.data[list(result_dataset.components.keys())]
        return result_dataset


class Alias(Binary):

    @classmethod
    def validate(cls, left: Dataset, right: str):
        if left.name == right:
            raise ValueError("Alias operation requires different names")
        return Dataset(name=right, components=left.components, data=None)

    @classmethod
    def evaluate(cls, left_operand: Dataset, right_operand: str) -> Dataset:
        result = cls.validate(left_operand, right_operand)
        result.data = left_operand.data
        return result


class Eval(Unary):

    @staticmethod
    def _execute_query(
        query: str, dataset_names: List[str], data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        for ds_name in dataset_names:
            locals()[ds_name] = data[ds_name]

        try:
            # df_result = duckdb.query(query).to_df()
            df_result = sqldf(query=query, env=locals(), db_uri="sqlite:///:memory:")
        except Exception as e:
            raise Exception(f"Error executing SQL query: {e}")
        for ds_name in dataset_names:
            del locals()[ds_name]

        return df_result

    @classmethod
    def validate(
        cls,
        operands: Dict[str, Dataset],
        external_routine: ExternalRoutine,
        output: Dataset,
    ) -> Dataset:

        empty_data_dict = {}
        for ds_name in external_routine.dataset_names:
            if ds_name not in operands:
                raise ValueError(
                    f"External Routine dataset {ds_name} "
                    f"is not present in Eval operands"
                )
            empty_data = pd.DataFrame(
                columns=[comp.name for comp in operands[ds_name].components.values()]
            )
            empty_data_dict[ds_name] = empty_data

        df = cls._execute_query(
            external_routine.query, external_routine.dataset_names, empty_data_dict
        )
        component_names = [name for name in df.columns]
        for comp_name in component_names:
            if comp_name not in output.components:
                raise ValueError(f"Component {comp_name} not found in output dataset")

        for comp_name in output.components:
            if comp_name not in component_names:
                raise ValueError(
                    f"Component {comp_name} not found in External Routine result"
                )

        output.name = external_routine.name

        return output

    @classmethod
    def evaluate(
        cls,
        operands: Dict[str, Dataset],
        external_routine: ExternalRoutine,
        output: Dataset,
    ) -> Dataset:
        result = cls.validate(operands, external_routine, output)

        operands_data_dict = {ds_name: operands[ds_name].data for ds_name in operands}

        result.data = cls._execute_query(
            external_routine.query, external_routine.dataset_names, operands_data_dict
        )

        return result

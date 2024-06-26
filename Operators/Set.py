import os
from typing import List

if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Dataset
from Operators import Operator


class Set(Operator):

    @classmethod
    def check_same_structure(cls, dataset_1: Dataset, dataset_2: Dataset) -> None:
        if len(dataset_1.components) != len(dataset_2.components):
            raise Exception(f"Datasets {dataset_1.name} and {dataset_2.name} "
                            f"have different number of components")

        for comp in dataset_1.components.values():
            if comp.name not in dataset_2.components:
                raise Exception(f"Component {comp.name} not found in dataset {dataset_2.name}")
            second_comp = dataset_2.components[comp.name]
            cls.validate_type_compatibility(comp.data_type, second_comp.data_type)
            if comp.role != second_comp.role:
                raise Exception(f"Component {comp.name} has different roles "
                                f"in datasets {dataset_1.name} and {dataset_2.name}")
            if comp.nullable != second_comp.nullable:
                raise Exception(f"Component {comp.name} has different nullability "
                                f"in datasets {dataset_1.name} and {dataset_2.name}")

    @classmethod
    def validate(cls, operands: List[Dataset]) -> Dataset:

        base_operand = operands[0]
        for operand in operands[1:]:
            cls.check_same_structure(base_operand, operand)

        result = Dataset(name="result", components=base_operand.components, data=None)
        return result


class Union(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        result.data = pd.concat(all_datapoints, sort=True,
                                ignore_index=True)
        identifiers_names = result.get_identifiers_names()
        result.data = result.data.drop_duplicates(subset=identifiers_names, keep='first')
        result.data.reset_index(drop=True, inplace=True)
        return result


class Intersection(Set):

    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        for data in all_datapoints:
            if result.data is None:
                result.data = data
            else:
                result.data = result.data.merge(data, how='inner',
                                                on=result.get_identifiers_names())

                not_identifiers = [col for col in result.get_measures_names() +
                                     result.get_attributes_names()]

                for col in not_identifiers:
                    result.data[col] = result.data[col + "_x"]
                result.data = result.data[result.get_identifiers_names() + not_identifiers]
        result.data.reset_index(drop=True, inplace=True)
        return result


class Symdiff(Set):

    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        for data in all_datapoints:
            if result.data is None:
                result.data = data
            else:
                result.data = (pd.merge(result.data, data, how='outer',
                                        on=result.get_identifiers_names(),
                                        indicator=True, suffixes=('_x', '_y'))
                               .query('_merge != "both"'))
                not_identifiers = [col for col in result.get_measures_names() +
                                   result.get_attributes_names()]
                for col in not_identifiers:
                    result.data[col] = result.data.apply(
                        lambda x, c=col: x[c + "_x"] if x["_merge"] == "left_only" else x[c + "_y"],
                        axis=1)
                result.data = result.data[result.get_identifiers_names() + not_identifiers]
        result.data.reset_index(drop=True, inplace=True)
        return result


class Setdiff(Set):

    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        for data in all_datapoints:
            if result.data is None:
                result.data = data
            else:
                result.data = (pd.merge(result.data, data, how='outer',
                                        on=result.get_identifiers_names(),
                                        indicator=True, suffixes=('_x', '_y'))
                               .query('_merge == "left_only"'))
                not_identifiers = [col for col in result.get_measures_names() +
                                   result.get_attributes_names()]
                for col in not_identifiers:
                    result.data[col] = result.data[col + "_x"]
                result.data = result.data[result.get_identifiers_names() + not_identifiers]
        result.data.reset_index(drop=True, inplace=True)
        return result

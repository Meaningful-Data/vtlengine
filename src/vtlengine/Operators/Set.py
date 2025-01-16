from typing import Any, Dict, List

# if os.environ.get("SPARK"):
#     import pyspark.pandas as pd
# else:
#     import pandas as pd
import pandas as pd

from vtlengine.DataTypes import binary_implicit_promotion
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Dataset
from vtlengine.Operators import Operator


class Set(Operator):
    @classmethod
    def check_same_structure(cls, dataset_1: Dataset, dataset_2: Dataset) -> None:
        if len(dataset_1.components) != len(dataset_2.components):
            raise SemanticError(
                "1-1-17-1",
                op=cls.op,
                dataset_1=dataset_1.name,
                dataset_2=dataset_2.name,
            )

        for comp in dataset_1.components.values():
            if comp.name not in dataset_2.components:
                raise Exception(f"Component {comp.name} not found in dataset {dataset_2.name}")
            second_comp = dataset_2.components[comp.name]
            binary_implicit_promotion(
                comp.data_type,
                second_comp.data_type,
                cls.type_to_check,
                cls.return_type,
            )
            if comp.role != second_comp.role:
                raise Exception(
                    f"Component {comp.name} has different roles "
                    f"in datasets {dataset_1.name} and {dataset_2.name}"
                )

    @classmethod
    def validate(cls, operands: List[Dataset]) -> Dataset:
        base_operand = operands[0]
        for operand in operands[1:]:
            cls.check_same_structure(base_operand, operand)

        result_components: Dict[str, Any] = {}
        for operand in operands:
            if len(result_components) == 0:
                result_components = operand.components
            else:
                for comp_name, comp in operand.components.items():
                    current_comp = result_components[comp_name]
                    result_components[comp_name].data_type = binary_implicit_promotion(
                        current_comp.data_type, comp.data_type
                    )
                    result_components[comp_name].nullable = current_comp.nullable or comp.nullable

        result = Dataset(name="result", components=result_components, data=None)
        return result


class Union(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        result.data = pd.concat(all_datapoints, sort=True, ignore_index=True)
        identifiers_names = result.get_identifiers_names()
        result.data = result.data.drop_duplicates(subset=identifiers_names, keep="first")
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
                if data is None:
                    result.data = pd.DataFrame(columns=result.get_identifiers_names())
                    break
                result.data = result.data.merge(
                    data, how="inner", on=result.get_identifiers_names()
                )

                not_identifiers = result.get_measures_names() + result.get_attributes_names()

                for col in not_identifiers:
                    result.data[col] = result.data[col + "_x"]
                result.data = result.data[result.get_identifiers_names() + not_identifiers]
        if result.data is not None:
            result.data.reset_index(drop=True, inplace=True)
        return result


class Symdiff(Set):
    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        for data in all_datapoints:
            if data is None:
                data = pd.DataFrame(columns=result.get_identifiers_names())
            if result.data is None:
                result.data = data
            else:
                # Realiza la operación equivalente en pyspark.pandas
                result.data = result.data.merge(
                    data,
                    how="outer",
                    on=result.get_identifiers_names(),
                    suffixes=("_x", "_y"),
                )

                for measure in result.get_measures_names():
                    result.data["_merge"] = result.data.apply(
                        lambda row: (
                            "left_only"
                            if pd.isnull(row[f"{measure}_y"])
                            else ("right_only" if pd.isnull(row[f"{measure}_x"]) else "both")
                        ),
                        axis=1,
                    )

                not_identifiers = result.get_measures_names() + result.get_attributes_names()
                for col in not_identifiers:
                    result.data[col] = result.data.apply(
                        lambda x, c=col: (
                            x[c + "_x"]
                            if x["_merge"] == "left_only"
                            else (x[c + "_y"] if x["_merge"] == "right_only" else None)
                        ),
                        axis=1,
                    )
                result.data = result.data[result.get_identifiers_names() + not_identifiers].dropna()
        if result.data is not None:
            result.data = result.data.reset_index(drop=True)
        return result


class Setdiff(Set):
    @staticmethod
    def has_null(row: Any) -> bool:
        return row.isnull().any()

    @classmethod
    def evaluate(cls, operands: List[Dataset]) -> Dataset:
        result = cls.validate(operands)
        all_datapoints = [ds.data for ds in operands]
        for data in all_datapoints:
            if result.data is None:
                result.data = data
            else:
                if data is None:
                    data = pd.DataFrame(columns=result.get_identifiers_names())
                result.data = result.data.merge(data, how="left", on=result.get_identifiers_names())
                if len(result.data) > 0:
                    result.data = result.data[result.data.apply(cls.has_null, axis=1)]

                not_identifiers = result.get_measures_names() + result.get_attributes_names()
                for col in not_identifiers:
                    if col + "_x" in result.data:
                        result.data[col] = result.data[col + "_x"]
                        del result.data[col + "_x"]
                    if col + "_y" in result.data:
                        del result.data[col + "_y"]
                result.data = result.data[result.get_identifiers_names() + not_identifiers]
        if result.data is not None:
            result.data.reset_index(drop=True, inplace=True)
        return result

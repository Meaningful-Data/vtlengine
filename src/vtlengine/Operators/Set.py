from typing import Any, Dict, List

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
    pass


class Intersection(Set):
    pass


class Symdiff(Set):
    pass


class Setdiff(Set):
    @staticmethod
    def has_null(row: Any) -> bool:
        return row.isnull().any()

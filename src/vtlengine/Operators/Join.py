from copy import copy
from functools import reduce
from typing import Any, Dict, List, Optional

# if os.environ.get("SPARK"):
#     import pyspark.pandas as pd
# else:
#     import pandas as pd
import pandas as pd

from vtlengine.AST import BinOp
from vtlengine.DataTypes import binary_implicit_promotion
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Role
from vtlengine.Operators import Operator, _id_type_promotion_join_keys


class Join(Operator):
    how: str
    reference_dataset: Dataset

    @classmethod
    def get_components_union(cls, datasets: List[Dataset]) -> List[Component]:
        common: List[Any] = []
        common.extend(
            copy(comp)
            for dataset in datasets
            for comp in dataset.components.values()
            if comp not in common
        )
        return common

    @classmethod
    def get_components_intersection(cls, operands: List[Any]) -> Any:
        element_count: Dict[str, Any] = {}
        for operand in operands:
            operand_set = set(operand)
            for element in operand_set:
                element_count[element] = element_count.get(element, 0) + 1
        result = []
        for element, count in element_count.items():
            if count >= 2:
                result.append(element)
        return result

    @classmethod
    def merge_components(
        cls, operands: Any, using: Optional[List[str]] = None
    ) -> Dict[str, Component]:
        nullability = {}
        merged_components = {}
        using = using or []
        common = cls.get_components_intersection([op.get_components_names() for op in operands])
        totally_common = list(
            reduce(
                lambda x, y: x & set(y.get_components_names()),  # type: ignore[operator]
                operands[1:],
                set(operands[0].get_components_names()),
            )
        )

        for op in operands:
            for comp in op.components.values():
                if comp.name in using:
                    is_identifier = all(
                        operand.components[comp.name].role == Role.IDENTIFIER
                        for operand in operands
                        if comp.name in operand.get_components_names()
                    )
                    comp.role = (
                        Role.IDENTIFIER
                        if is_identifier
                        else Role.MEASURE
                        if comp.role == Role.IDENTIFIER
                        else comp.role
                    )
                if comp.name not in nullability:
                    nullability[comp.name] = copy(comp.nullable)
                if comp.role == Role.IDENTIFIER:
                    nullability[comp.name] = False
                elif comp.name in totally_common:
                    nullability[comp.name] |= copy(comp.nullable)
                elif cls.how == "outer" or (
                    cls.how == "left"
                    and comp.name not in cls.reference_dataset.get_components_names()
                ):
                    nullability[comp.name] = True
                else:
                    nullability[comp.name] = copy(comp.nullable)

        for operand in operands:
            operand_name = operand.name
            components = {comp.name: copy(comp) for comp in operand.components.values()}

            for component_name, component in components.items():
                component.nullable = nullability[component_name]

                if component_name in common and component_name not in using:
                    if component.role != Role.IDENTIFIER or cls.how == "cross":
                        new_name = f"{operand_name}#{component_name}"
                        if new_name in merged_components:
                            raise SemanticError("1-1-13-9", comp_name=new_name)
                        while new_name in common:
                            new_name += "_dup"
                        merged_components[new_name] = component
                        merged_components[new_name].name = new_name
                    else:
                        merged_components[component_name] = component
                else:
                    if component_name in using and component_name in merged_components:
                        data_type = binary_implicit_promotion(
                            merged_components[component_name].data_type,
                            component.data_type,
                        )
                        component.data_type = data_type
                    merged_components[component_name] = component

        return merged_components

    @classmethod
    def generate_result_components(
        cls, operands: List[Dataset], using: Optional[List[str]] = None
    ) -> Dict[str, Component]:
        components = {}
        inter_identifiers = cls.get_components_intersection(
            [op.get_identifiers_names() for op in operands]
        )

        for op in operands:
            ids = op.get_identifiers_names()
            for id in inter_identifiers:
                components.update({id: copy(op.components[id])} if id in ids else {})
        return components

    @classmethod
    def evaluate(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        result = cls.execute([copy(operand) for operand in operands], using)
        if result.data is not None and sorted(result.get_components_names()) != sorted(
            result.data.columns.tolist()
        ):
            missing = list(set(result.get_components_names()) - set(result.data.columns.tolist()))
            if len(missing) == 0:
                missing.append("None")
            raise SemanticError("1-1-1-10", comp_name=missing[0], dataset_name=result.name)
        return result

    @classmethod
    def execute(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        result = cls.validate(operands, using)
        using = using if using else []
        if len(operands) == 1:
            result.data = operands[0].data
            return result

        common_measures = cls.get_components_intersection(
            [op.get_measures_names() + op.get_attributes_names() for op in operands]
        )
        for op in operands:
            if op.data is not None:
                for column in op.data.columns.tolist():
                    if column in common_measures and column not in using:
                        op.data = op.data.rename(columns={column: op.name + "#" + column})
        result.data = copy(cls.reference_dataset.data)

        join_keys = using if using else result.get_identifiers_names()

        for op in operands:
            if op is not cls.reference_dataset:
                merge_join_keys = (
                    [key for key in join_keys if key in op.data.columns.tolist()]
                    if (op.data is not None)
                    else []
                )
                if len(merge_join_keys) == 0:
                    raise SemanticError("1-1-13-14", name=op.name)
                for join_key in merge_join_keys:
                    _id_type_promotion_join_keys(
                        result.get_component(join_key),
                        op.get_component(join_key),
                        join_key,
                        result.data,
                        op.data,
                    )
                if op.data is not None and result.data is not None:
                    result.data = pd.merge(
                        result.data,
                        op.data,
                        how=cls.how,  # type: ignore[arg-type]
                        on=merge_join_keys,
                    )
                else:
                    result.data = pd.DataFrame()
        if result.data is not None:
            result.data.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    def validate(cls, operands: List[Dataset], using: Optional[List[str]]) -> Dataset:
        if len(operands) < 1 or sum([isinstance(op, Dataset) for op in operands]) < 1:
            raise Exception("Join operator requires at least 1 dataset")
        if not all(isinstance(op, Dataset) for op in operands):
            raise SemanticError("1-1-13-10")
        if len(operands) == 1 and isinstance(operands[0], Dataset):
            return Dataset(name="result", components=operands[0].components, data=None)
        for op in operands:
            if len(op.get_identifiers()) == 0:
                raise SemanticError("1-3-27", op=cls.op)
        cls.reference_dataset = (
            max(operands, key=lambda x: len(x.get_identifiers_names()))
            if cls.how not in ["cross", "left"]
            else operands[0]
        )
        cls.identifiers_validation(operands, using)
        components = cls.merge_components(operands, using)
        if len(set(components.keys())) != len(components):
            raise SemanticError("1-1-13-9", comp_name="")

        return Dataset(name="result", components=components, data=None)

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: Optional[List[str]]) -> None:
        # (Case A)
        info = {op.name: op.get_identifiers_names() for op in operands}
        for op_name, identifiers in info.items():
            if len(identifiers) == 0:
                raise SemanticError("1-1-13-14", op=cls.op, name=op_name)

        for op_name, identifiers in info.items():
            if (
                using is None
                and op_name != cls.reference_dataset.name
                and not set(identifiers).issubset(set(info[cls.reference_dataset.name]))
            ):
                missing_components = list(set(identifiers) - set(info[cls.reference_dataset.name]))
                raise SemanticError(
                    "1-1-13-11",
                    op=cls.op,
                    dataset_reference=cls.reference_dataset.name,
                    component=missing_components[0],
                )
        if using is None:
            return

        # (Case B1)
        if cls.reference_dataset is not None:
            for op_name, identifiers in info.items():
                if op_name != cls.reference_dataset.name and not set(identifiers).issubset(using):
                    raise SemanticError("1-1-13-4", op=cls.op, using_names=using, dataset=op_name)
            reference_components = cls.reference_dataset.get_components_names()
            if not set(using).issubset(reference_components):
                raise SemanticError(
                    "1-1-13-6",
                    op=cls.op,
                    using_components=using,
                    reference=cls.reference_dataset.name,
                )

            for _, identifiers in info.items():
                if not set(using).issubset(identifiers):
                    # (Case B2)
                    if not set(using).issubset(reference_components):
                        raise SemanticError("1-1-13-5", op=cls.op, using_names=using)
                else:
                    for op in operands:
                        if op is not cls.reference_dataset:
                            for component in using:
                                if component not in op.get_components_names():
                                    raise SemanticError(
                                        "1-1-1-10",
                                        op=cls.op,
                                        comp_name=component,
                                        dataset_name=op.name,
                                    )


class InnerJoin(Join):
    how = "inner"

    @classmethod
    def generate_result_components(
        cls, operands: List[Dataset], using: Optional[List[str]] = None
    ) -> Dict[str, Component]:
        if using is None:
            return super().generate_result_components(operands, using)

        components = {}
        for op in operands:
            components.update(
                {id: op.components[id] for id in using if id in op.get_measures_names()}
            )
        for op in operands:
            components.update({id: op.components[id] for id in op.get_identifiers_names()})
        return components


class LeftJoin(Join):
    how = "left"


class FullJoin(Join):
    how = "outer"

    @classmethod
    def identifiers_validation(
        cls, operands: List[Dataset], using: Optional[List[str]] = None
    ) -> None:
        if using is not None:
            raise SemanticError("1-1-13-8", op=cls.op)
        for op in operands:
            if op is cls.reference_dataset:
                continue
            if len(op.get_identifiers_names()) != len(
                cls.reference_dataset.get_identifiers_names()
            ):
                raise SemanticError("1-1-13-13", op=cls.op)
            if op.get_identifiers_names() != cls.reference_dataset.get_identifiers_names():
                raise SemanticError("1-1-13-12", op=cls.op)


class CrossJoin(Join):
    how = "cross"

    @classmethod
    def execute(cls, operands: List[Dataset], using: Optional[List[str]] = None) -> Dataset:
        result = cls.validate(operands, using)
        if len(operands) == 1:
            result.data = operands[0].data
            return result
        common = cls.get_components_intersection([op.get_components_names() for op in operands])

        for op in operands:
            if op.data is None:
                op.data = pd.DataFrame(columns=op.get_components_names())
            if op is operands[0]:
                result.data = op.data
            else:
                if result.data is not None:
                    result.data = pd.merge(
                        result.data,
                        op.data,
                        how=cls.how,  # type: ignore[arg-type]
                    )
            if result.data is not None:
                result.data = result.data.rename(
                    columns={
                        column: op.name + "#" + column
                        for column in result.data.columns.tolist()
                        if column in common
                    }
                )
        if result.data is not None:
            result.data.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    def identifiers_validation(
        cls, operands: List[Dataset], using: Optional[List[str]] = None
    ) -> None:
        if using is not None:
            raise SemanticError("1-1-13-8", op=cls.op)


class Apply(Operator):
    @classmethod
    def evaluate(cls, dataset: Dataset, expression: Any, op_map: Dict[str, Any]) -> Dataset:
        for child in expression:
            dataset = cls.execute(dataset, op_map[child.op], child.left.value, child.right.value)
        return dataset

    @classmethod
    def execute(cls, dataset: Dataset, op: Any, left: str, right: str) -> Dataset:
        left_dataset = cls.create_dataset("left", left, dataset)
        right_dataset = cls.create_dataset("right", right, dataset)
        left_dataset, right_dataset = cls.get_common_components(left_dataset, right_dataset)
        return op.evaluate(left_dataset, right_dataset)

    @classmethod
    def validate(cls, dataset: Dataset, child: Any, op_map: Dict[str, Any]) -> None:
        if not isinstance(child, BinOp):
            raise Exception(
                f"Invalid expression {child} on apply operator. Only BinOp are accepted"
            )
        if child.op not in op_map:
            raise Exception(f"Operator {child.op} not implemented")
        if hasattr(child.left, "value") and hasattr(child.right, "value"):
            left_components = [
                comp.name[len(child.left.value) + 1]
                for comp in dataset.components.values()
                if comp.name.startswith(child.left.value)
            ]
            right_components = [
                comp.name[len(child.right.value) + 1]
                for comp in dataset.components.values()
                if comp.name.startswith(child.right.value)
            ]
            if len(set(left_components) & set(right_components)) == 0:
                raise Exception(
                    f"{child.left.value} and {child.right.value} "
                    f"has not any match on dataset components"
                )

    @classmethod
    def create_dataset(cls, name: str, prefix: str, dataset: Dataset) -> Dataset:
        prefix += "#"
        components = {
            component.name: component
            for component in dataset.components.values()
            if component.name.startswith(prefix) or component.role is Role.IDENTIFIER
        }
        data = dataset.data[list(components.keys())] if dataset.data is not None else pd.DataFrame()

        for component in components.values():
            component.name = (
                component.name[len(prefix) :]
                if (component.name.startswith(prefix) and component.role is not Role.IDENTIFIER)
                else component.name
            )
        components = {component.name: component for component in components.values()}
        data.rename(
            columns={
                column: column[len(prefix) :]
                for column in data.columns
                if column.startswith(prefix)
            },
            inplace=True,
        )
        return Dataset(name=name, components=components, data=data)

    @classmethod
    def get_common_components(cls, left: Dataset, right: Dataset) -> (Dataset, Dataset):  # type: ignore[syntax]
        common = set(left.get_components_names()) & set(right.get_components_names())
        left.components = {
            comp.name: comp for comp in left.components.values() if comp.name in common
        }
        right.components = {
            comp.name: comp for comp in right.components.values() if comp.name in common
        }
        left.data = left.data[list(common)] if left.data is not None else pd.DataFrame()
        right.data = right.data[list(common)] if right.data is not None else pd.DataFrame()
        return left, right

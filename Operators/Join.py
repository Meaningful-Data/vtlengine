import os
from copy import copy
from functools import reduce
from typing import List, Dict

from AST import BinOp

if os.environ.get("SPARK"):
    import pyspark.pandas as pd
else:
    import pandas as pd

from Model import Dataset, Component, Role
from Operators import Operator


class Join(Operator):
    how = None
    reference_dataset = None

    @classmethod
    def get_components_union(cls, datasets: List[Dataset]) -> List[Component]:
        common = []
        common.extend(copy(comp) for dataset in datasets for comp in dataset.components.values() if comp not in common)
        return common

    @classmethod
    def get_components_intersection(cls, *operands: List[Component]):
        element_count = {}
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
    def merge_components(cls, operands, using=None):
        nullability = {}
        merged_components = {}
        using = using or []
        common = cls.get_components_intersection(*[op.get_components_names() for op in operands])
        totally_common = list(reduce(lambda x, y: x & set(y.get_components_names()), operands[1:], set(operands[0].get_components_names())))

        for op in operands:
            for comp in op.components.values():
                if comp.name in using:
                    is_identifier = all(operand.components[comp.name].role == Role.IDENTIFIER
                                        for operand in operands if comp.name in operand.get_components_names())
                    comp.role = Role.IDENTIFIER if is_identifier else Role.MEASURE if comp.role == Role.IDENTIFIER else comp.role
                    # role = Role.IDENTIFIER if is_identifier else Role.MEASURE if comp.role == Role.IDENTIFIER else comp.role
                    # comp = Component(name=comp.name, data_type=comp.data_type, role=role, nullable=comp.nullable)

                if comp.name not in nullability:
                    nullability[comp.name] = copy(comp.nullable)

                if comp.role == Role.IDENTIFIER:
                    nullability[comp.name] = False
                elif comp.name in totally_common:
                    nullability[comp.name] |= copy(comp.nullable)
                elif cls.how == 'outer' or (cls.how == 'left' and comp.name not in cls.reference_dataset.get_components_names()):
                    nullability[comp.name] = True
                else:
                    nullability[comp.name] = copy(comp.nullable)

        for operand in operands:
            operand_name = operand.name
            components = {comp.name: copy(comp) for comp in operand.components.values()}

            for component_name, component in components.items():
                component.nullable = nullability[component_name]

                if component_name in common and component_name not in using:
                    if component.role != Role.IDENTIFIER or cls.how == 'cross':
                        new_name = f'{operand_name}#{component_name}'
                        while new_name in common:
                            new_name += '_dup'
                        merged_components[new_name] = component
                        merged_components[new_name].name = new_name
                    else:
                        merged_components[component_name] = component
                else:
                    merged_components[component_name] = component

        return merged_components

    @classmethod
    def generate_result_components(cls, operands: List[Dataset], using=None) -> Dict[str, Component]:
        components = {}
        inter_identifiers = cls.get_components_intersection(*[op.get_identifiers_names() for op in operands])

        for op in operands:
            ids = op.get_identifiers_names()
            for id in inter_identifiers:
                components.update({id: copy(op.components[id])} if id in ids else {})
        return components

    @classmethod
    def evaluate(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        result = cls.execute([copy(operand) for operand in operands], using)
        if sorted(result.get_components_names()) != sorted(result.data.columns.tolist()):
            raise Exception(f"Invalid components on result dataset")
        return result

    @classmethod
    def execute(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        result = cls.validate(operands, using)
        using = using if using else []
        if len(operands) == 1:
            result.data = operands[0].data
            return result

        common_measures = cls.get_components_intersection(*[op.get_measures_names() for op in operands])
        for op in operands:
            for column in op.data.columns.tolist():
                if column in common_measures and column not in using:
                    op.data = op.data.rename(columns={column: op.name + '#' + column})
        result.data = copy(cls.reference_dataset.data)

        join_keys = using if using else result.get_identifiers_names()

        for op in operands:
            if op is not cls.reference_dataset:
                merge_join_keys = [key for key in join_keys if key in op.data.columns.tolist()]
                result.data = pd.merge(result.data, op.data, how=cls.how, on=merge_join_keys)

        result.data.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    def validate(cls, operands: List[Dataset], using: List[str]) -> Dataset:
        if len(operands) < 1 or sum([isinstance(op, Dataset) for op in operands]) < 1:
            raise Exception("Join operator requires at least 1 dataset")
        if not all([isinstance(op, Dataset) for op in operands]):
            raise Exception("All operands must be datasets")
        if len(operands) == 1 and isinstance(operands[0], Dataset):
            return Dataset(name="result", components=operands[0].components, data=None)
        cls.identifiers_validation(operands, using)

        cls.reference_dataset = max(operands, key=lambda x: len(x.get_identifiers_names())) if cls.how not in ['cross', 'left'] else operands[0]
        components = cls.generate_result_components(operands, using)
        if using is not None:
            for op in operands:
                components.update({id: op.components[id] for id in using if id in op.get_measures_names()})
        components = cls.merge_components(operands, using)

        return Dataset(name="result", components=components, data=None)

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        for op in operands:
            if op.get_identifiers_names() != operands[0].get_identifiers_names():
                raise Exception(f"All datasets must have the same identifiers on {cls.op}")


class InnerJoin(Join):
    how = 'inner'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        # (Case A)
        case_A = True
        info = {op.name: op.get_identifiers_names() for op in operands}
        most_identifiers = max(info, key=lambda x: len(info[x]))
        for op_name, identifiers in info.items():
            if op_name != most_identifiers and not set(identifiers).issubset(set(info[most_identifiers])):
                case_A = False

        if using is None:
            if not case_A:
                raise Exception("Case A: Identifiers from 'using' must be a subset of the common identifiers and respect Case A")
            return

        common_components = cls.get_components_intersection(*[op.get_components_names() for op in operands])
        common_identifiers = cls.get_components_intersection(*[op.get_identifiers_names() for op in operands])

        # (Case B)
        if set(using).issubset(common_components):
            # (Case B1)
            if case_A and set(using).issubset(common_identifiers):
                # reference_identifiers = set(operands[0].get_identifiers_names())
                # for op in operands:
                #     if not set(op.get_identifiers_names()).issubset(reference_identifiers):
                #         raise Exception(
                #             "Sub-case B1: Every non-reference dataset identifiers must be a subset of the reference dataset identifiers")
                return

            # (Case B2)
            # reference_identifiers = set(max(operands, key=lambda x: len(x.get_identifiers_names())).get_identifiers_names())
            # for identifier in using:
            #     if identifier not in reference_identifiers:
            #         raise Exception(
            #             f"Sub-case B2: Using clause must be a subset of the reference dataset identifiers {reference_identifiers}")
        else:
            raise Exception(
                "Case B: 'Using' clause identifiers must be a subset of the common identifiers across all datasets")

    @classmethod
    def generate_result_components(cls, operands: List[Dataset], using=None) -> Dict[str, Component]:

        if using is None:
            return super().generate_result_components(operands, using)

        components = {}
        for op in operands:
            components.update({id: op.components[id] for id in op.get_identifiers_names()})
        for op in operands:
            components.update({id: op.components[id] for id in using if id in op.get_measures_names()})
        return components


class LeftJoin(Join):

    how = 'left'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using: List[str]) -> None:
        # (Case A)
        case_A = True
        reference_identifiers = sorted(operands[0].get_identifiers_names())
        for op in operands:
            if sorted(op.get_identifiers_names()) != reference_identifiers:
                case_A = False

        if using is None:
            if not case_A:
                raise Exception("Case A: All datasets must have the same identifiers")
            return

        common_components = cls.get_components_intersection(*[op.get_components_names() for op in operands])
        common_identifiers = cls.get_components_intersection(*[op.get_identifiers_names() for op in operands])

        # (Case B)
        if set(using).issubset(common_components):
            # (Case B1)
            if case_A and set(using).issubset(common_identifiers):
                return
            # (Case B1)
            # if set(using).issubset(common_components):
            #     reference_identifiers = set(operands[0].get_identifiers_names())
            #     for op in operands:
            #         if not set(op.get_identifiers_names()).issubset(reference_identifiers):
            #             raise Exception(
            #                 "Sub-case B1: Every non-reference dataset identifiers must be a subset of the reference dataset identifiers")
            else:
                # (Case B2)
                # reference_identifiers = set(operands[0].get_identifiers_names())
                # for identifier in using:
                #     if identifier not in reference_identifiers:
                #         raise Exception(
                #             f"Sub-case B2: Using clause must be a subset of the reference dataset identifiers {reference_identifiers}")
                return
        else:
            raise Exception(
                "Case B: 'Using' clause identifiers must be a subset of the common identifiers across all datasets")


class FullJoin(Join):
    how = 'outer'

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using=None) -> None:
        if using is not None:
            raise Exception("Full join does not accept using clause")
        for op in operands:
            if op.get_identifiers_names() != operands[0].get_identifiers_names():
                raise Exception("All datasets must have the same identifiers")


class CrossJoin(Join):
    how = 'cross'

    # @classmethod
    # def execute(cls, operands: List[Dataset], using: List[str]) -> Dataset:
    #     result = cls.validate(operands, using)
    #     if len(operands) == 1:
    #         result.name = 'result'
    #         return result
    #
    #     common_measures = cls.get_components_intersection(*[op.get_measures_names() for op in operands])
    #     for op in operands:
    #         for column in op.data.columns.tolist():
    #             if column in common_measures:
    #                 op.data = op.data.rename(columns={column: op.name + '#' + column})
    #     result.data = copy(cls.reference_dataset.data)
    #
    #     # TODO: nullability = or between all comp nullability
    #     for op in operands:
    #         if op is operands[0]:
    #             result.data = op.data
    #         else:
    #             result.data = pd.merge(result.data, op.data, how=cls.how)
    #         result.data = result.data.rename(
    #             columns={column: op.name + '#' + column for column in result.data.columns.tolist()
    #                      if column in common_measures})
    #
    #     result.data.reset_index(drop=True, inplace=True)
    #     return result

    @classmethod
    def execute(cls, operands: List[Dataset], using=None) -> Dataset:
        result = cls.validate(operands, using)
        if len(operands) == 1:
            result.data = operands[0].data
            return result
        common = cls.get_components_intersection(*[op.get_components_names() for op in operands])

        for op in operands:
            if op is operands[0]:
                result.data = op.data
            else:
                result.data = pd.merge(result.data, op.data, how=cls.how)
            result.data = result.data.rename(
                columns={column: op.name + '#' + column for column in result.data.columns.tolist()
                         if column in common})
        result.data.reset_index(drop=True, inplace=True)
        return result

    @classmethod
    def identifiers_validation(cls, operands: List[Dataset], using=None) -> None:
        if using is not None:
            raise Exception("Cross join does not accept using clause")


class Apply(Operator):

    @classmethod
    def evaluate(cls, dataset: Dataset, expression, op_map: dict):
        for child in expression:
            dataset = cls.execute(dataset, op_map[child.op], child.left.value, child.right.value)
        return dataset

    @classmethod
    def execute(cls, dataset: Dataset, op, left: str, right: str) -> Dataset:
        left_dataset = cls.create_dataset("left", left, dataset)
        right_dataset = cls.create_dataset("right", right, dataset)
        left_dataset, right_dataset = cls.get_common_components(left_dataset, right_dataset)
        return op.evaluate(left_dataset, right_dataset)

    @classmethod
    def validate(cls, dataset: Dataset, child, op_map: dict) -> None:
        if not isinstance(child, BinOp):
            raise Exception(
                f"Invalid expression {child} on apply operator. Only BinOp are accepted")
        if child.op not in op_map:
            raise Exception(f"Operator {child.op} not implemented")
        left_components = [comp.name[len(child.left.value) + 1] for comp in
                           dataset.components.values() if
                           comp.name.startswith(child.left.value)]
        right_components = [comp.name[len(child.right.value) + 1] for comp in
                            dataset.components.values() if
                            comp.name.startswith(child.right.value)]
        if len(set(left_components) & set(right_components)) == 0:
            raise Exception(
                f"{child.left.value} and {child.right.value} has not any match on dataset components")

    @classmethod
    def create_dataset(cls, name: str, prefix: str, dataset: Dataset) -> Dataset:
        prefix += '#'
        components = {component.name: component for component in dataset.components.values() if
                      component.name.startswith(prefix) or component.role is Role.IDENTIFIER}
        data = dataset.data[list(components.keys())]

        for component in components.values():
            component.name = component.name[len(prefix):] if (
                    component.name.startswith(
                        prefix) and component.role is not Role.IDENTIFIER) else component.name
        components = {component.name: component for component in components.values()}
        data.rename(columns={column: column[len(prefix):] for column in data.columns if
                             column.startswith(prefix)},
                    inplace=True)
        return Dataset(name=name, components=components, data=data)

    @classmethod
    def get_common_components(cls, left: Dataset, right: Dataset) -> (Dataset, Dataset):
        common = set(left.get_components_names()) & set(right.get_components_names())
        left.components = {comp.name: comp for comp in left.components.values() if
                           comp.name in common}
        right.components = {comp.name: comp for comp in right.components.values() if
                            comp.name in common}
        left.data = left.data[list(common)]
        right.data = right.data[list(common)]
        return left, right

from dataclasses import dataclass
from typing import Any, Dict, Optional

import AST
from AST.ASTTemplate import ASTTemplate
from AST.Grammar.tokens import ALL, BETWEEN, EXISTS_IN, FILTER, ROUND, TRUNC
from DataTypes import BASIC_TYPES
from Model import DataComponent, Dataset, Scalar, ScalarSet
from Operators.Assignment import Assignment
from Operators.Comparison import Between, ExistIn
from Operators.Numeric import Round, Trunc
from Utils import BINARY_MAPPING, REGULAR_AGGREGATION_MAPPING, ROLE_SETTER_MAPPING, SET_MAPPING, \
    UNARY_MAPPING


@dataclass
class InterpreterAnalyzer(ASTTemplate):
    datasets: Dict[str, Dataset]
    is_from_assignment: bool = False
    is_from_regular_aggregation: bool = False
    regular_aggregation_dataset: Optional[Dataset] = None

    def visit_Start(self, node: AST.Start) -> Any:
        results = {}
        for child in node.children:
            result = self.visit(child)
            # TODO: Execute collected operations from Spark and add explain
            results[result.name] = result
        return results

    def visit_Assignment(self, node: AST.Assignment) -> Any:
        self.is_from_assignment = True
        left_operand: str = self.visit(node.left)
        self.is_from_assignment = False
        right_operand: Dataset = self.visit(node.right)
        return Assignment.evaluate(left_operand, right_operand)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Any:
        return self.visit_Assignment(node)  # type:ignore

    def visit_BinOp(self, node: AST.BinOp) -> None:
        left_operand = self.visit(node.left)
        right_operand = self.visit(node.right)
        if node.op not in BINARY_MAPPING:
            raise NotImplementedError
        return BINARY_MAPPING[node.op].evaluate(left_operand, right_operand)

    def visit_UnaryOp(self, node: AST.UnaryOp) -> None:
        operand = self.visit(node.operand)
        if node.op not in UNARY_MAPPING and node.op not in ROLE_SETTER_MAPPING:
            raise NotImplementedError
        if self.is_from_regular_aggregation and node.op in ROLE_SETTER_MAPPING:
            data_size = len(self.regular_aggregation_dataset.data)
            return ROLE_SETTER_MAPPING[node.op].evaluate(operand, data_size)
        return UNARY_MAPPING[node.op].evaluate(operand)

    def visit_MulOp(self, node: AST.MulOp):
        """
        MulOp: (op, children)

        op: BETWEEN : 'between'.

        Basic usage:

            for child in node.children:
                self.visit(child)
        """
        # Comparison Operators
        if node.op == BETWEEN:
            operand_element = self.visit(node.children[0])
            from_element = self.visit(node.children[1])
            to_element = self.visit(node.children[2])

            return Between.evaluate(operand_element, from_element, to_element)

        # Comparison Operators
        elif node.op == EXISTS_IN:
            dataset_1 = self.visit(node.children[0])
            if not isinstance(dataset_1, Dataset):
                raise Exception("First operand must be a dataset")
            dataset_2 = self.visit(node.children[1])
            if not isinstance(dataset_2, Dataset):
                raise Exception("Second operand must be a dataset")

            retain_element = None
            if len(node.children) == 3:
                retain_element = self.visit(node.children[2])
                if isinstance(retain_element, Scalar):
                    retain_element = retain_element.value
                if retain_element == ALL:
                    retain_element = None

            return ExistIn.evaluate(dataset_1, dataset_2, retain_element)

        raise NotImplementedError

    def visit_VarID(self, node: AST.VarID) -> Any:
        if self.is_from_assignment:
            return node.value

        if self.is_from_regular_aggregation:
            return DataComponent(name=node.value,
                                 data=self.regular_aggregation_dataset.data[node.value],
                                 data_type=
                                 self.regular_aggregation_dataset.components[
                                     node.value].data_type,
                                 role=self.regular_aggregation_dataset.components[
                                     node.value].role)

        if node.value not in self.datasets:
            raise Exception(f"Dataset {node.value} not found, please check input datastructures")
        return self.datasets[node.value]

    def visit_Collection(self, node: AST.Collection) -> Any:
        if node.kind == 'Set':
            elements = []
            for child in node.children:
                elements.append(self.visit(child).value)
            for element in elements:
                if type(element) != type(elements[0]):
                    raise Exception("All elements in a set must be of the same type")
            if len(elements) == 0:
                raise Exception("A set must contain at least one element")
            if len(elements) != len(set(elements)):
                raise Exception("A set must not contain duplicates")
            return ScalarSet(data_type=BASIC_TYPES[type(elements[0])], values=elements)
        raise NotImplementedError

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> None:
        if node.op not in REGULAR_AGGREGATION_MAPPING:
            raise NotImplementedError
        operands = []
        dataset = self.visit(node.dataset)
        if isinstance(dataset, Scalar):
            raise Exception(f"Scalar {dataset.name} cannot be used with clause operators")
        self.regular_aggregation_dataset = dataset
        for child in node.children:
            self.is_from_regular_aggregation = True
            operands.append(self.visit(child))
            self.is_from_regular_aggregation = False
        self.regular_aggregation_dataset = None
        if node.op == FILTER:
            return REGULAR_AGGREGATION_MAPPING[node.op].evaluate(operands[0], dataset)
        return REGULAR_AGGREGATION_MAPPING[node.op].evaluate(operands, dataset)

    def visit_MulOp(self, node: AST.MulOp) -> None:
        # Set Operators.
        if node.op in SET_MAPPING:
            datasets = []
            for child in node.children:
                datasets.append(self.visit(child))

            for ds in datasets:
                if not isinstance(ds, Dataset):
                    raise ValueError(f"Expected dataset, got {type(ds).__name__}")

            return SET_MAPPING[node.op].evaluate(datasets)

    def visit_RenameNode(self, node: AST.RenameNode) -> Any:
        return node

    def visit_Constant(self, node: AST.Constant) -> Any:
        return Scalar(name=str(node.value), value=node.value,
                      data_type=BASIC_TYPES[type(node.value)])

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        return node.value

    def visit_ParamOp(self, node: AST.ParamOp) -> None:
        if node.op == ROUND:
            op_element = self.visit(node.children[0])
            if len(node.params) != 0:
                param_element = self.visit(node.params[0])
            else:
                param_element = None

            return Round.evaluate(op_element, param_element)

        # Numeric Operator
        elif node.op == TRUNC:
            op_element = self.visit(node.children[0])
            param_element = None
            if len(node.params) != 0:
                param_element = self.visit(node.params[0])

            return Trunc.evaluate(op_element, param_element)

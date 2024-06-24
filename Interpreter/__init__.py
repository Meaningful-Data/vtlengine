from dataclasses import dataclass
from typing import Dict, Any, Optional

import AST
from AST.ASTTemplate import ASTTemplate
from Model import Dataset, Scalar, DataComponent
from Operators.Assignment import Assignment
from Utils import BINARY_MAPPING, UNARY_MAPPING, REGULAR_AGGREGATION_MAPPING


@dataclass
class InterpreterAnalyzer(ASTTemplate):
    datasets: Dict[str, Dataset]
    is_from_assignment: bool = False
    is_from_regular_aggregation: bool = False
    regular_aggregation_dataset: Optional[str] = None

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
        if node.op not in UNARY_MAPPING:
            raise NotImplementedError
        return UNARY_MAPPING[node.op].evaluate(operand)

    def visit_VarID(self, node: AST.VarID) -> Any:
        if self.is_from_assignment:
            return node.value


        if self.is_from_regular_aggregation:
            return DataComponent(name=node.value, data=self.datasets[self.regular_aggregation_dataset].data[node.value],
                                 data_type=self.datasets[self.regular_aggregation_dataset].components[node.value].data_type,
                                 role=self.datasets[self.regular_aggregation_dataset].components[node.value].role)

        if node.value not in self.datasets:
            raise ValueError(f"Dataset {node.value} not found, please check input datastructures")
        return self.datasets[node.value]

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> None:
        if node.op not in REGULAR_AGGREGATION_MAPPING:
            raise NotImplementedError
        operands = []
        dataset = self.visit(node.dataset)
        self.regular_aggregation_dataset = dataset.name
        for child in node.children:
            self.is_from_regular_aggregation = True
            operands.append(self.visit(child))
            self.is_from_regular_aggregation = False
        self.regular_aggregation_dataset = None
        return REGULAR_AGGREGATION_MAPPING[node.op].evaluate(operands, dataset)

    def visit_Constant(self, node: AST.Constant) -> Any:
        return Scalar(name=str(node.value), value=node.value)

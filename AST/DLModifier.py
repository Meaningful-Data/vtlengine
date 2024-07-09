from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

import AST
from AST import Assignment, Operator, ParamOp, Start, UDOCall
from AST.ASTTemplate import ASTTemplate
from Model import Dataset


@dataclass
class DLModifier(ASTTemplate):
    datasets: Dict[str, Dataset]
    is_from_param: bool = False
    is_from_call: bool = False

    current_udo_operator: List[str] = None
    current_udo_arguments: List[Dict[str, Any]] = None
    udo_mapping: Dict[str, Any] = None

    def visit_Start(self, node: Start) -> Any:
        for child in node.children:
            self.visit(child)
        children = [x for x in node.children if not isinstance(x, Assignment)]
        node.children = children
        return node

    def visit_Operator(self, node: Operator) -> None:
        if self.udo_mapping is None:
            self.udo_mapping = {}
        self.udo_mapping[node.op] = {
            'expression': node.expression,
            'arguments': node.parameters,
            'output': node.output_type
        }

    def visit_UDOCall(self, node: UDOCall):
        if node.op not in self.udo_mapping:
            raise Exception(f"User defined operator {node.op} not found")
        self.current_udo_operator.append(node.op)
        for param in node.params:
            self.is_from_param = True
            self.visit(node.params)
            self.is_from_param = False
        self.current_udo_operator.pop()
        node.expression = deepcopy(self.udo_mapping[node.op]['expression'])
        self.visit(node.expression)

"""
Interpreter DAG Analyzer.
===========================

Description
-----------
Direct Acyclic Graph.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import networkx as nx

from vtlengine.AST import (
    AST,
    Aggregation,
    Analytic,
    Assignment,
    BinOp,
    DefIdentifier,
    DPRuleset,
    HRuleset,
    Identifier,
    JoinOp,
    Operator,
    ParamOp,
    PersistentAssignment,
    RegularAggregation,
    Start,
    VarID,
)
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.DAG._words import DELETE, GLOBAL, INPUTS, INSERT, OUTPUTS, PERSISTENT
from vtlengine.AST.Grammar.tokens import AS, MEMBERSHIP, TO
from vtlengine.Exceptions import SemanticError


@dataclass
class DAGAnalyzer(ASTTemplate):
    udos: Optional[Dict[str, Any]] = None
    numberOfStatements: int = 1
    dependencies: Optional[Dict[str, Any]] = None
    vertex: Optional[Dict[str, Any]] = None
    extVertex: Optional[Dict[str, Any]] = None
    persVertex: Optional[list] = None
    edges: Optional[Dict[str, Any]] = None
    nov: int = 0
    dot: Optional[str] = None
    sorting: Optional[list] = None
    output: Optional[Dict[str, Any]] = None

    # Handlers
    isFirstAssignment: bool = False
    isFromRegularAggregation: bool = False
    isDataset: bool = False
    alias: Optional[list] = None

    # Statement Structure
    inputs: Optional[list] = None
    outputs: Optional[list] = None
    persistent: Optional[list] = None

    def __post_init__(self):
        self.dependencies = {}
        self.vertex = {}
        self.extVertex = {}
        self.persVertex = []
        self.edges = {}
        self.inputs = []
        self.outputs = []
        self.persistent = []
        self.alias = []

    @classmethod
    def ds_structure(cls, ast: AST):
        # Visit AST.
        dag = cls()
        dag.visit(ast)
        return dag._ds_usage_analysis()

    def _ds_usage_analysis(self):
        statements = {INSERT: {}, DELETE: {}}
        all_output = []
        global_inputs = []
        inserted = []
        persistent_datasets = []
        for key, statement in self.dependencies.items():
            outputs = statement[OUTPUTS]
            persistent = statement[PERSISTENT]
            reference = outputs + persistent
            if len(persistent) == 1 and persistent[0] not in persistent_datasets:
                persistent_datasets.append(persistent[0])
            deletion_key = key
            all_output.append(reference[0])
            for subKey, subStatement in self.dependencies.items():
                candidates = subStatement[INPUTS]
                if candidates and reference[0] in candidates:
                    deletion_key = subKey
            if deletion_key in statements[DELETE]:
                statements[DELETE][deletion_key].append(reference[0])
            else:
                statements[DELETE][deletion_key] = reference

        # Deletion of gloabl inputs
        for key, statement in self.dependencies.items():
            inputs = statement[INPUTS]
            for element in inputs:
                if element not in all_output and element not in global_inputs:
                    deletion_key = key
                    global_inputs.append(element)
                    for subKey, subStatement in self.dependencies.items():
                        candidates = subStatement[INPUTS]
                        if candidates and element in candidates:
                            deletion_key = subKey
                    if deletion_key in statements[DELETE]:
                        statements[DELETE][deletion_key].append(element)
                    else:
                        statements[DELETE][deletion_key] = [element]

        # Insertion of global inputs
        for key, statement in self.dependencies.items():
            for element in statement[INPUTS]:
                if element not in inserted and element in global_inputs:
                    inserted.append(element)
                    if key in statements[INSERT]:
                        statements[INSERT][key].append(element)
                    else:
                        statements[INSERT][key] = [element]

        statements[GLOBAL] = global_inputs
        statements[PERSISTENT] = persistent_datasets
        return statements

    @classmethod
    def createDAG(cls, ast: Start):
        """ """
        # Visit AST.
        dag = cls()
        dag.visit(ast)
        # Create graph.
        dag.loadVertex()
        dag.loadEdges()
        try:
            dag.nx_topologicalSort()
            # Create output dict.
            if len(dag.edges) != 0:
                dag.sortAST(ast)
            else:
                MLStatements: list = [
                    ML for ML in ast.children if not isinstance(ML, (HRuleset, DPRuleset, Operator))
                ]
                dag.check_overwriting(MLStatements)
            return dag

        except nx.NetworkXUnfeasible as error:
            error_keys = {}
            for v in dag.edges.values():
                aux_v0, aux_v1 = v[1], v[0]
                for iv in dag.edges.values():
                    if aux_v0 == iv[0] and aux_v1 == iv[1]:
                        error_keys[aux_v0] = dag.dependencies[aux_v0]
                        error_keys[aux_v1] = dag.dependencies[aux_v1]
                        break
            raise Exception(
                "Vtl Script contains Cycles, no DAG established.\nSuggestion {}, "
                "more_info:{}".format(error, error_keys)
            ) from None
        except SemanticError as error:
            raise error
        except Exception as error:
            raise Exception("Error creating DAG.") from error

    def loadVertex(self):
        """ """
        # For each vertex
        for key, statement in self.dependencies.items():
            output = statement[OUTPUTS] + statement[PERSISTENT]
            # If the statement has no := or -> symbol there is no vertex to add.
            if len(output) != 0:
                self.vertex[key] = output[0]

        # Set the number of vertex.
        self.nov = len(self.vertex)

    def loadEdges(self):
        """ """
        if len(self.vertex) != 0:
            countEdges = 0
            # For each vertex
            for key, statement in self.dependencies.items():
                outputs = statement[OUTPUTS]
                persistent = statement[PERSISTENT]
                reference = outputs + persistent
                for subKey, subStatement in self.dependencies.items():
                    subInputs = subStatement[INPUTS]
                    candidates = subInputs
                    if candidates and reference[0] in candidates:
                        self.edges[countEdges] = (key, subKey)
                        countEdges += 1

    def nx_topologicalSort(self):
        """ """
        edges = list(self.edges.values())
        DAG = nx.DiGraph()
        DAG.add_nodes_from(self.vertex)
        DAG.add_edges_from(edges)
        self.sorting = list(nx.topological_sort(DAG))

    def sort_elements(self, MLStatements):
        inter = []
        for x in self.sorting:
            for i in range(len(MLStatements)):
                if i == x - 1:
                    inter.append(MLStatements[i])
        return inter

    def check_overwriting(self, statements):
        non_repeated_outputs = []
        for statement in statements:
            if statement.left.value in non_repeated_outputs:
                raise SemanticError("1-3-3", varId_value=statement.left.value)
            else:
                non_repeated_outputs.append(statement.left.value)

    def sortAST(self, ast: AST):
        """ """
        statements_nodes = ast.children
        HRuleStatements: list = [HRule for HRule in statements_nodes if isinstance(HRule, HRuleset)]
        DPRuleStatement: list = [
            DPRule for DPRule in statements_nodes if isinstance(DPRule, DPRuleset)
        ]
        DOStatement: list = [DO for DO in statements_nodes if isinstance(DO, Operator)]
        MLStatements: list = [
            ML for ML in statements_nodes if not isinstance(ML, (HRuleset, DPRuleset, Operator))
        ]

        intermediate = self.sort_elements(MLStatements)
        self.check_overwriting(intermediate)
        ast.children = HRuleStatements + DPRuleStatement + DOStatement + intermediate

    def statementStructure(self) -> dict:
        """ """
        inputs = list(set(self.inputs))
        outputs = list(set(self.outputs))
        persistent = list(set(self.persistent))

        # Remove inputs that are outputs of some statement.
        inputsF = [inputf for inputf in inputs if inputf not in outputs]

        dict_ = {INPUTS: inputsF, OUTPUTS: outputs, PERSISTENT: persistent}

        return dict_

    """______________________________________________________________________________________


                                Start of visiting AST nodes.

    _______________________________________________________________________________________"""

    def visit_Start(self, node: Start) -> None:
        """
        Start: (children)

        Basic usage:

            for child in node.children:
                self.visit(child)
        """
        udos = {}
        for ast_element in node.children:
            if isinstance(ast_element, Operator):
                udos[ast_element.op] = ast_element
        self.udos = udos
        for child in node.children:
            if isinstance(child, (Assignment, PersistentAssignment)):
                self.isFirstAssignment = True
                self.visit(child)

                # Analyze inputs and outputs per each statement.
                self.dependencies[self.numberOfStatements] = copy.deepcopy(
                    self.statementStructure()
                )

                # Count the number of statements in order to name the scope symbol table for
                # each one.
                self.numberOfStatements += 1

                self.alias = []

                self.inputs = []
                self.outputs = []
                self.persistent = []

    def visit_Assignment(self, node: Assignment) -> None:
        if self.isFirstAssignment:
            self.outputs.append(node.left.value)
            self.isFirstAssignment = False

        self.visit(node.right)

    def visit_PersistentAssignment(self, node: PersistentAssignment) -> None:
        if self.isFirstAssignment:
            self.persistent.append(node.left.value)
            self.isFirstAssignment = False

        self.visit(node.right)

    def visit_RegularAggregation(self, node: RegularAggregation) -> None:
        self.visit(node.dataset)
        for child in node.children:
            self.isFromRegularAggregation = True
            self.visit(child)
            self.isFromRegularAggregation = False

    def visit_BinOp(self, node: BinOp) -> None:
        if node.op == MEMBERSHIP:
            self.isDataset = True
            self.visit(node.left)
            self.isDataset = False
            self.visit(node.right)
        elif node.op == AS or node.op == TO:
            self.visit(node.left)
            self.alias.append(node.right.value)
        else:
            self.visit(node.left)
            self.visit(node.right)

    def visit_VarID(self, node: VarID) -> None:
        if (not self.isFromRegularAggregation or self.isDataset) and node.value not in self.alias:
            self.inputs.append(node.value)

    def visit_Identifier(self, node: Identifier) -> None:
        if node.kind == "DatasetID" and node.value not in self.alias:
            self.inputs.append(node.value)

    def visit_ParamOp(self, node: ParamOp) -> None:
        if self.udos and node.op in self.udos:
            DO_AST: Operator = self.udos[node.op]

            for arg in node.params:
                index_arg = node.params.index(arg)
                if DO_AST.parameters[index_arg].type_.kind == "DataSet":
                    self.visit(arg)
        else:
            super(DAGAnalyzer, self).visit_ParamOp(node)

    def visit_Aggregation(self, node: Aggregation) -> None:
        if node.operand is not None:
            self.visit(node.operand)

    def visit_Analytic(self, node: Analytic) -> None:
        if node.operand is not None:
            self.visit(node.operand)

    def visit_JoinOp(self, node: JoinOp) -> None:
        for clause in node.clauses:
            self.visit(clause)


class HRDAGAnalyzer(DAGAnalyzer):
    @classmethod
    def createDAG(cls, ast: HRuleset):
        # Visit AST.
        dag = cls()
        dag.visit(ast)
        # Create graph.
        dag.loadVertex()
        dag.loadEdges()
        try:
            dag.nx_topologicalSort()
            # Create output dict.
            if len(dag.edges) != 0:
                dag.rules_ast = dag.sort_elements(dag.rules_ast)
                ast.rules = dag.rules_ast
            return dag

        except nx.NetworkXUnfeasible as error:
            error_keys = {}
            for v in dag.edges.values():
                aux_v0, aux_v1 = v[1], v[0]
                for iv in dag.edges.values():
                    if aux_v0 == iv[0] and aux_v1 == iv[1]:
                        error_keys[aux_v0] = dag.dependencies[aux_v0]
                        error_keys[aux_v1] = dag.dependencies[aux_v1]
                        break
            raise Exception(
                f"Vtl Script contains Cycles, no DAG established."
                f"\nSuggestion {error}, more_info:{error_keys}"
            )

    def visit_HRuleset(self, node: HRuleset) -> None:
        """
        HRuleset: (name, element, rules)

        Basic usage:

            self.visit(node.element)
            for rule in node.rules:
                self.visit(rule)
        """
        self.hierarchy_ruleset_name = node.name
        if isinstance(node.element, list):
            for element in node.element:
                self.visit(element)
        else:
            self.visit(node.element)
        # self.visit(node.element)
        self.rules_ast = node.rules
        for rule in node.rules:
            self.isFirstAssignment = True
            self.visit(rule)
            self.dependencies[self.numberOfStatements] = copy.deepcopy(self.statementStructure())

            # Count the number of statements in order to name the scope symbol table for each one.
            self.numberOfStatements += 1
            self.alias = []

            self.inputs = []
            self.outputs = []
            self.persistent = []

    def visit_DefIdentifier(self, node: DefIdentifier):
        """
        DefIdentifier: (value, kind)

        Basic usage:

            return node.value
        """
        # def visit_Identifier(self, node: Identifier) -> None:
        if node.kind == "CodeItemID":  # and node.value not in self.alias:
            if self.isFirstAssignment:
                self.isFirstAssignment = False
                self.outputs.append(node.value)
            else:
                self.inputs.append(node.value)

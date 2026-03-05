"""
Interpreter DAG Analyzer.
===========================

Description
-----------
Direct Acyclic Graph.
"""

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from vtlengine.AST import (
    AST,
    Aggregation,
    Analytic,
    Assignment,
    BinOp,
    Constant,
    DefIdentifier,
    DPRuleset,
    DPValidation,
    HROperation,
    HRuleset,
    Identifier,
    JoinOp,
    Operator,
    ParamOp,
    PersistentAssignment,
    RegularAggregation,
    Start,
    UDOCall,
    VarID,
)
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.DAG._models import Schedule, StatementDeps
from vtlengine.AST.Grammar.tokens import AS, DROP, KEEP, MEMBERSHIP, RENAME, TO
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset


@dataclass
class DAGAnalyzer(ASTTemplate):
    udos: Optional[Dict[str, Any]] = None
    number_of_statements: int = 1
    dependencies: Dict[int, StatementDeps] = field(default_factory=dict)
    vertex: Dict[int, str] = field(default_factory=dict)
    edges: Dict[int, tuple] = field(default_factory=dict)  # type: ignore[type-arg]
    sorting: Optional[List[int]] = None

    # Handlers
    is_first_assignment: bool = False
    is_from_regular_aggregation: bool = False
    is_dataset: bool = False
    alias: Set[str] = field(default_factory=set)

    # Per-statement accumulator (reset between statements)
    current_deps: StatementDeps = field(default_factory=StatementDeps)
    _current_has_dataset_op: bool = False
    # Cross-statement unknown variable tracking
    unknown_variables: Set[str] = field(default_factory=set)
    # Outputs that were consumed via unknown_variables (RegularAggregation context)
    _resolved_from_unknown: Set[str] = field(default_factory=set)

    @classmethod
    def ds_structure(cls, ast: AST) -> Schedule:
        dag = cls()
        dag.visit(ast)
        return dag._ds_usage_analysis()

    def _ds_usage_analysis(self) -> Schedule:
        """Analyze dataset dependencies to build insertion/deletion schedules."""
        deletion: Dict[int, List[str]] = defaultdict(list)
        insertion: Dict[int, List[str]] = defaultdict(list)
        all_outputs: Set[str] = set()
        persistent_datasets: List[str] = []

        # Reverse index: dataset_name -> last statement that uses it as input
        last_consumer: Dict[str, int] = {}
        for key, statement in self.dependencies.items():
            for input_name in statement.inputs:
                last_consumer[input_name] = key

        # Schedule deletion for statement outputs at their last consumer
        for key, statement in self.dependencies.items():
            reference = statement.outputs + statement.persistent
            if (
                len(statement.persistent) == 1
                and statement.persistent[0] not in persistent_datasets
            ):
                persistent_datasets.append(statement.persistent[0])
            ds_name = reference[0]
            all_outputs.add(ds_name)
            deletion[last_consumer.get(ds_name, key)].append(ds_name)

        # Schedule insertion (first use) and deletion (last use) for global inputs
        global_inputs: List[str] = []
        global_set: Set[str] = set()
        for key, statement in self.dependencies.items():
            for element in statement.inputs:
                if element not in all_outputs and element not in global_set:
                    global_set.add(element)
                    global_inputs.append(element)
                    deletion[last_consumer.get(element, key)].append(element)
                    insertion[key].append(element)

        classification = self._classify_global_inputs(all_outputs, global_inputs, global_set)

        return Schedule(
            insertion=dict(insertion),
            deletion=dict(deletion),
            global_inputs=classification["global_inputs"],
            global_input_datasets=classification["global_input_datasets"],
            global_input_scalars=classification["global_input_scalars"],
            global_input_dataset_or_scalar=classification["global_input_dataset_or_scalar"],
            global_input_component_or_scalar=classification["global_input_component_or_scalar"],
            persistent=persistent_datasets,
            all_outputs=sorted(all_outputs),
        )

    def _classify_global_inputs(
        self,
        all_outputs: Set[str],
        global_inputs: List[str],
        global_set: Set[str],
    ) -> Dict[str, List[str]]:
        """Classify global inputs into datasets, scalars, and ambiguous categories."""
        scalar_outputs, comp_or_scalar = self._compute_scalar_outputs(all_outputs)

        # Identify definite dataset inputs
        definite_dataset_inputs: Set[str] = set()
        for statement in self.dependencies.values():
            if statement.has_dataset_op:
                for inp in statement.inputs:
                    definite_dataset_inputs.add(inp)

        # Include component_or_scalar candidates in global_inputs
        for name in comp_or_scalar:
            if name not in global_set and name not in all_outputs:
                global_set.add(name)
                global_inputs.append(name)

        # Classify into four categories
        result: Dict[str, List[str]] = {
            "global_inputs": global_inputs,
            "global_input_datasets": [],
            "global_input_scalars": [],
            "global_input_dataset_or_scalar": [],
            "global_input_component_or_scalar": [],
        }

        for name in global_inputs:
            if name in comp_or_scalar:
                result["global_input_component_or_scalar"].append(name)
            elif name in definite_dataset_inputs:
                result["global_input_datasets"].append(name)
            elif self._feeds_only_scalar_chains(name, scalar_outputs):
                result["global_input_scalars"].append(name)
            else:
                result["global_input_dataset_or_scalar"].append(name)

        return result

    def _compute_scalar_outputs(self, all_outputs: Set[str]) -> tuple[Set[str], Set[str]]:
        """Compute scalar outputs via propagation and component/scalar candidates."""
        scalar_outputs: Set[str] = set()
        for statement in self.dependencies.values():
            reference = statement.outputs + statement.persistent
            if not reference:
                continue
            ds_name = reference[0]
            is_constant_assignment = not statement.has_dataset_op and not statement.inputs
            if is_constant_assignment or ds_name in self._resolved_from_unknown:
                scalar_outputs.add(ds_name)

        comp_or_scalar: Set[str] = set()
        for statement in self.dependencies.values():
            for uv in statement.unknown_variables:
                if uv not in all_outputs:
                    comp_or_scalar.add(uv)

        # Propagate: statements with no dataset ops where all inputs are known scalars
        changed = True
        while changed:
            changed = False
            for statement in self.dependencies.values():
                reference = statement.outputs + statement.persistent
                if not reference:
                    continue
                ds_name = reference[0]
                if ds_name in scalar_outputs or statement.has_dataset_op:
                    continue
                if statement.inputs and all(
                    inp in scalar_outputs or inp in comp_or_scalar for inp in statement.inputs
                ):
                    scalar_outputs.add(ds_name)
                    changed = True

        return scalar_outputs, comp_or_scalar

    def _feeds_only_scalar_chains(self, name: str, scalar_outputs: Set[str]) -> bool:
        """Check if a global input feeds only into scalar-output statements."""
        has_consumers = False
        for stmt in self.dependencies.values():
            if name not in stmt.inputs:
                continue
            has_consumers = True
            if stmt.has_dataset_op:
                return False
            reference = stmt.outputs + stmt.persistent
            if reference and reference[0] not in scalar_outputs:
                return False
        return has_consumers

    @classmethod
    def create_dag(cls, ast: Start) -> "DAGAnalyzer":
        dag = cls()
        dag.visit(ast)
        dag.load_vertex()
        dag.load_edges()
        try:
            dag._build_and_sort_graph("createDAG")
            if len(dag.edges) != 0:
                dag.sort_ast(ast)
            else:
                ml_statements: list = [
                    ml for ml in ast.children if not isinstance(ml, (HRuleset, DPRuleset, Operator))
                ]
                dag.check_overwriting(ml_statements)
            return dag
        except SemanticError:
            raise
        except Exception as error:
            raise SemanticError(code="1-3-2-0") from error

    def _build_and_sort_graph(self, error_op: str) -> None:
        """Build networkx DAG, perform topological sort, detect cycles."""
        edges = list(self.edges.values())
        graph = nx.DiGraph()
        graph.add_nodes_from(self.vertex)
        graph.add_edges_from(edges)

        try:
            result: list = []
            components = sorted(nx.weakly_connected_components(graph), key=min)
            for component in components:
                result.extend(nx.topological_sort(graph.subgraph(component)))
            self.sorting = result
        except nx.NetworkXUnfeasible:
            error_keys: Dict[int, Any] = {}
            for v in self.edges.values():
                aux_v0, aux_v1 = v[1], v[0]
                for iv in self.edges.values():
                    if aux_v0 == iv[0] and aux_v1 == iv[1]:
                        error_keys[aux_v0] = self.dependencies[aux_v0]
                        error_keys[aux_v1] = self.dependencies[aux_v1]
                        break
            raise SemanticError("1-3-2-3", op=error_op, nodes=error_keys) from None

    def load_vertex(self) -> None:
        for key, statement in self.dependencies.items():
            output = statement.outputs + statement.persistent + statement.unknown_variables
            if len(output) != 0:
                self.vertex[key] = output[0]

    def load_edges(self) -> None:
        if len(self.vertex) != 0:
            count_edges = 0
            ref_to_keys: Dict[str, int] = {}
            for key, statement in self.dependencies.items():
                reference = statement.outputs + statement.persistent
                if reference:
                    ref_to_keys[reference[0]] = key

            for sub_key, sub_statement in self.dependencies.items():
                for input_val in sub_statement.inputs:
                    if input_val in ref_to_keys:
                        key = ref_to_keys[input_val]
                        self.edges[count_edges] = (key, sub_key)
                        count_edges += 1

    def sort_elements(self, statements: list) -> list:
        return [statements[x - 1] for x in self.sorting]  # type: ignore[union-attr]

    def check_overwriting(self, statements: list) -> None:
        seen: Set[str] = set()
        for statement in statements:
            if statement.left.value in seen:
                raise SemanticError("1-2-2", varId_value=statement.left.value)
            seen.add(statement.left.value)

    def sort_ast(self, ast: AST) -> None:
        statements_nodes = ast.children
        hr_statements: list = [node for node in statements_nodes if isinstance(node, HRuleset)]
        dp_statements: list = [node for node in statements_nodes if isinstance(node, DPRuleset)]
        do_statements: list = [node for node in statements_nodes if isinstance(node, Operator)]
        ml_statements: list = [
            node
            for node in statements_nodes
            if not isinstance(node, (HRuleset, DPRuleset, Operator))
        ]

        intermediate = self.sort_elements(ml_statements)
        self.check_overwriting(intermediate)
        ast.children = hr_statements + dp_statements + do_statements + intermediate

    def statement_structure(self) -> StatementDeps:
        result = StatementDeps(
            inputs=[
                inp for inp in self.current_deps.inputs if inp not in self.current_deps.outputs
            ],
            outputs=list(self.current_deps.outputs),
            persistent=list(self.current_deps.persistent),
            unknown_variables=list(self.current_deps.unknown_variables),
            has_dataset_op=self._current_has_dataset_op,
        )
        self.unknown_variables.update(self.current_deps.unknown_variables)
        return result

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
                self.is_first_assignment = True
                self.visit(child)

                self.dependencies[self.number_of_statements] = self.statement_structure()
                self.number_of_statements += 1
                self.alias = set()
                self.current_deps = StatementDeps()
                self._current_has_dataset_op = False

        aux = copy.copy(self.unknown_variables)
        for variable in aux:
            for _number_of_statement, dependency in self.dependencies.items():
                if variable in dependency.outputs:
                    self.unknown_variables.discard(variable)
                    self._resolved_from_unknown.add(variable)
                    for _ns2, dep2 in self.dependencies.items():
                        if variable in dep2.unknown_variables:
                            dep2.unknown_variables.remove(variable)
                            dep2.inputs.append(variable)

    def visit_Assignment(self, node: Assignment) -> None:
        if self.is_first_assignment:
            self.current_deps.outputs.append(node.left.value)
            self.is_first_assignment = False

        self.visit(node.right)

    def visit_PersistentAssignment(self, node: PersistentAssignment) -> None:
        if self.is_first_assignment:
            self.current_deps.persistent.append(node.left.value)
            self.is_first_assignment = False

        self.visit(node.right)

    def visit_RegularAggregation(self, node: RegularAggregation) -> None:
        self._current_has_dataset_op = True
        self.visit(node.dataset)
        if node.op in [KEEP, DROP, RENAME]:
            return
        for child in node.children:
            self.is_from_regular_aggregation = True
            self.visit(child)
            self.is_from_regular_aggregation = False

    def visit_BinOp(self, node: BinOp) -> None:
        if node.op == MEMBERSHIP:
            self._current_has_dataset_op = True
            self.is_dataset = True
            self.visit(node.left)
            self.is_dataset = False
            self.visit(node.right)
        elif node.op == AS or node.op == TO:
            self.visit(node.left)
            self.alias.add(node.right.value)
        else:
            self.visit(node.left)
            self.visit(node.right)

    def visit_VarID(self, node: VarID) -> None:
        if (
            not self.is_from_regular_aggregation or self.is_dataset
        ) and node.value not in self.alias:
            if node.value not in self.current_deps.inputs:
                self.current_deps.inputs.append(node.value)
        elif (
            self.is_from_regular_aggregation
            and node.value not in self.alias
            and not self.is_dataset
            and node.value not in self.current_deps.unknown_variables
        ):
            self.current_deps.unknown_variables.append(node.value)

    def visit_Identifier(self, node: Identifier) -> None:
        if (
            node.kind == "DatasetID"
            and node.value not in self.alias
            and node.value not in self.current_deps.inputs
        ):
            self._current_has_dataset_op = True
            self.current_deps.inputs.append(node.value)

    def visit_ParamOp(self, node: ParamOp) -> None:
        if self.udos and node.op in self.udos:
            do_ast: Operator = self.udos[node.op]

            for arg in node.params:
                index_arg = node.params.index(arg)
                if do_ast.parameters[index_arg].type_.kind == "DataSet":
                    self._current_has_dataset_op = True
                    self.visit(arg)
        else:
            super(DAGAnalyzer, self).visit_ParamOp(node)

    def visit_Aggregation(self, node: Aggregation) -> None:
        self._current_has_dataset_op = True
        if node.operand is not None:
            self.visit(node.operand)

    def visit_Analytic(self, node: Analytic) -> None:
        self._current_has_dataset_op = True
        if node.operand is not None:
            self.visit(node.operand)

    def visit_JoinOp(self, node: JoinOp) -> None:
        self._current_has_dataset_op = True
        for clause in node.clauses:
            self.visit(clause)

    def visit_UDOCall(self, node: UDOCall) -> None:
        node_args = (self.udos or {}).get(node.op)
        if not node_args:
            super().visit_UDOCall(node)
        else:
            for p, param in zip(node_args.parameters, node.params):
                if isinstance(param, Constant):
                    continue
                if type(p.type_) is not Component:
                    if isinstance(p.type_, Dataset):
                        self._current_has_dataset_op = True
                    self.visit(param)

    def visit_HROperation(self, node: HROperation) -> None:
        """Visit HROperation node for dependency analysis."""
        self._current_has_dataset_op = True
        self.visit(node.dataset)

    def visit_DPValidation(self, node: DPValidation) -> None:
        """Visit DPValidation node for dependency analysis."""
        self._current_has_dataset_op = True
        self.visit(node.dataset)


class HRDAGAnalyzer(DAGAnalyzer):
    def visit_HRuleset(self, node: HRuleset) -> None:
        """
        HRuleset: (name, element, rules)

        Basic usage:

            self.visit(node.element)
            for rule in node.rules:
                self.visit(rule)
        """
        if isinstance(node.element, list):
            for element in node.element:
                self.visit(element)
        else:
            self.visit(node.element)
        self.rules_ast = node.rules
        for rule in node.rules:
            self.is_first_assignment = True
            self.visit(rule)
            self.dependencies[self.number_of_statements] = self.statement_structure()

            self.number_of_statements += 1
            self.alias = set()
            self.current_deps = StatementDeps()

    def visit_DefIdentifier(self, node: DefIdentifier) -> None:  # type: ignore[override]
        """
        DefIdentifier: (value, kind)

        Basic usage:

            return node.value
        """
        if node.kind == "CodeItemID":
            if self.is_first_assignment:
                self.is_first_assignment = False
                self.current_deps.outputs.append(node.value)
            elif node.value not in self.current_deps.inputs:
                self.current_deps.inputs.append(node.value)

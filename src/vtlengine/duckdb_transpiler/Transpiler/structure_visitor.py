"""
Structure Visitor for VTL AST.

This module provides a visitor that computes Dataset structures for AST nodes.
It follows the visitor pattern from ASTTemplate and is used by SQLTranspiler
to track structure transformations through expressions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.Model import Dataset


@dataclass
class StructureVisitor(ASTTemplate):
    """
    Visitor that computes Dataset structures for AST nodes.

    This visitor tracks how data structures transform through VTL operations.
    It maintains a context dict mapping AST node ids to their computed structures,
    which is cleared after each transformation (child of AST.Start).

    Attributes:
        available_tables: Dict of tables available for querying (inputs + intermediates).
        output_datasets: Dict of output Dataset structures from semantic analysis.
        _structure_context: Internal cache mapping AST node id -> computed Dataset.
        _udo_params: Stack of UDO parameter bindings for nested UDO calls.
    """

    available_tables: Dict[str, Dataset] = field(default_factory=dict)
    output_datasets: Dict[str, Dataset] = field(default_factory=dict)
    _structure_context: Dict[int, Dataset] = field(default_factory=dict)
    _udo_params: Optional[List[Dict[str, Any]]] = None

    def clear_context(self) -> None:
        """
        Clear the structure context cache.

        Call this after processing each transformation (child of AST.Start)
        to prevent stale cached structures from affecting subsequent transformations.
        """
        self._structure_context.clear()

    def get_structure(self, node: AST.AST) -> Optional[Dataset]:
        """
        Get computed structure for a node.

        Checks the cache first, then falls back to available_tables lookup
        for VarID nodes.

        Args:
            node: The AST node to get structure for.

        Returns:
            The Dataset structure if found, None otherwise.
        """
        if id(node) in self._structure_context:
            return self._structure_context[id(node)]
        if isinstance(node, AST.VarID):
            if node.value in self.available_tables:
                return self.available_tables[node.value]
            if node.value in self.output_datasets:
                return self.output_datasets[node.value]
        return None

    def set_structure(self, node: AST.AST, dataset: Dataset) -> None:
        """
        Store computed structure for a node in the cache.

        Args:
            node: The AST node to store structure for.
            dataset: The computed Dataset structure.
        """
        self._structure_context[id(node)] = dataset

    def get_udo_param(self, name: str) -> Optional[Any]:
        """
        Look up a UDO parameter by name from the current scope.

        Searches from innermost scope outward through the UDO parameter stack.

        Args:
            name: The parameter name to look up.

        Returns:
            The bound value if found, None otherwise.
        """
        if self._udo_params is None:
            return None
        for scope in reversed(self._udo_params):
            if name in scope:
                return scope[name]
        return None

    def push_udo_params(self, params: Dict[str, Any]) -> None:
        """
        Push a new UDO parameter scope onto the stack.

        Args:
            params: Dict mapping parameter names to their bound values.
        """
        if self._udo_params is None:
            self._udo_params = []
        self._udo_params.append(params)

    def pop_udo_params(self) -> None:
        """
        Pop the innermost UDO parameter scope from the stack.
        """
        if self._udo_params:
            self._udo_params.pop()
            if len(self._udo_params) == 0:
                self._udo_params = None

    def visit_VarID(self, node: AST.VarID) -> Optional[Dataset]:
        """
        Get structure for a VarID (dataset reference).

        Checks for UDO parameter bindings first, then looks up in
        available_tables and output_datasets.

        Args:
            node: The VarID node.

        Returns:
            The Dataset structure if found, None otherwise.
        """
        # Check for UDO parameter binding
        udo_value = self.get_udo_param(node.value)
        if udo_value is not None:
            if isinstance(udo_value, AST.AST):
                return self.visit(udo_value)
            if isinstance(udo_value, Dataset):
                return udo_value

        # Look up in available tables
        if node.value in self.available_tables:
            return self.available_tables[node.value]

        # Look up in output datasets (for intermediate results)
        if node.value in self.output_datasets:
            return self.output_datasets[node.value]

        return None

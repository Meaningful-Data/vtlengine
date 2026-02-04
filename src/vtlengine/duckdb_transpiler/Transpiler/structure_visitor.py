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

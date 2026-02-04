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
from vtlengine.Model import Component, Dataset, Role


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

    def visit_BinOp(self, node: AST.BinOp) -> Optional[Dataset]:
        """
        Get structure for a binary operation.

        Handles:
        - MEMBERSHIP (#): Returns structure with only extracted component
        - Alias (as): Returns same structure as left operand
        - Other ops: Returns left operand structure

        Args:
            node: The BinOp node.

        Returns:
            The Dataset structure if computable, None otherwise.
        """
        from vtlengine.AST.Grammar.tokens import MEMBERSHIP

        op_lower = str(node.op).lower()

        if op_lower == MEMBERSHIP:
            return self._visit_binop_membership(node)

        if op_lower == "as":
            # Alias: same structure as left operand
            return self.visit(node.left)

        # For other binary operations, return left operand structure
        return self.visit(node.left)

    def _visit_binop_membership(self, node: AST.BinOp) -> Optional[Dataset]:
        """
        Compute structure for membership (#) operator.

        Membership extracts a single component from a dataset, returning
        a structure with identifiers plus the extracted component as measure.

        Args:
            node: The BinOp node with MEMBERSHIP operator.

        Returns:
            Dataset with identifiers + extracted component, or None.
        """
        base_ds = self.visit(node.left)
        if base_ds is None:
            return None

        # Get component name and resolve through UDO params if needed
        comp_name = self._resolve_varid_value(node.right)

        # Build new dataset with only identifiers and the extracted component
        new_components: Dict[str, Component] = {}
        for name, comp in base_ds.components.items():
            if comp.role == Role.IDENTIFIER:
                new_components[name] = comp

        # Add the extracted component as a measure
        if comp_name in base_ds.components:
            orig_comp = base_ds.components[comp_name]
            new_components[comp_name] = Component(
                name=comp_name,
                data_type=orig_comp.data_type,
                role=Role.MEASURE,
                nullable=orig_comp.nullable,
            )

        return Dataset(name=base_ds.name, components=new_components, data=None)

    def _resolve_varid_value(self, node: AST.AST) -> str:
        """
        Resolve a VarID value, checking for UDO parameter bindings.

        Args:
            node: The AST node to resolve.

        Returns:
            The resolved string value.
        """
        if not isinstance(node, (AST.VarID, AST.Identifier)):
            return str(node)

        name = node.value
        udo_value = self.get_udo_param(name)
        if udo_value is not None:
            if isinstance(udo_value, (AST.VarID, AST.Identifier)):
                return self._resolve_varid_value(udo_value)
            if isinstance(udo_value, str):
                return udo_value
            return str(udo_value)
        return name

    def visit_UnaryOp(self, node: AST.UnaryOp) -> Optional[Dataset]:
        """
        Get structure for a unary operation.

        Handles:
        - ISNULL: Returns structure with bool_var as measure
        - Other ops: Returns operand structure unchanged

        Args:
            node: The UnaryOp node.

        Returns:
            The Dataset structure if computable, None otherwise.
        """
        from vtlengine.AST.Grammar.tokens import ISNULL
        from vtlengine.DataTypes import Boolean

        op = str(node.op).lower()
        base_ds = self.visit(node.operand)

        if base_ds is None:
            return None

        if op == ISNULL:
            # isnull produces bool_var as output measure
            new_components: Dict[str, Component] = {}
            for name, comp in base_ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    new_components[name] = comp
            # Add bool_var as the output measure
            new_components["bool_var"] = Component(
                name="bool_var",
                data_type=Boolean,
                role=Role.MEASURE,
                nullable=False,
            )
            return Dataset(name=base_ds.name, components=new_components, data=None)

        # For other unary ops, return the base structure
        return base_ds

    def visit_ParamOp(self, node: AST.ParamOp) -> Optional[Dataset]:
        """
        Get structure for a parameterized operation.

        Handles:
        - CAST: Returns structure with updated measure data types

        Args:
            node: The ParamOp node.

        Returns:
            The Dataset structure if computable, None otherwise.
        """
        from vtlengine.AST.Grammar.tokens import CAST
        from vtlengine.DataTypes import (
            Boolean,
            Date,
            Duration,
            Integer,
            Number,
            String,
            TimeInterval,
            TimePeriod,
        )

        op_lower = str(node.op).lower()

        if op_lower == CAST and node.children:
            base_ds = self.visit(node.children[0])
            if base_ds and len(node.children) >= 2:
                # Get target type from second child
                target_type_node = node.children[1]
                if hasattr(target_type_node, "value"):
                    target_type = target_type_node.value
                elif hasattr(target_type_node, "__name__"):
                    target_type = target_type_node.__name__
                else:
                    target_type = str(target_type_node)

                # Map VTL type name to DataType class
                type_map = {
                    "Integer": Integer,
                    "Number": Number,
                    "String": String,
                    "Boolean": Boolean,
                    "Date": Date,
                    "TimePeriod": TimePeriod,
                    "TimeInterval": TimeInterval,
                    "Duration": Duration,
                }
                new_data_type = type_map.get(target_type)

                if new_data_type:
                    # Build new structure with updated measure types
                    new_components: Dict[str, Component] = {}
                    for name, comp in base_ds.components.items():
                        if comp.role == Role.IDENTIFIER:
                            new_components[name] = comp
                        else:
                            # Update measure data type
                            new_components[name] = Component(
                                name=name,
                                data_type=new_data_type,
                                role=comp.role,
                                nullable=comp.nullable,
                            )
                    return Dataset(name=base_ds.name, components=new_components, data=None)
            return base_ds

        # For other ParamOps, return first child's structure if available
        if node.children:
            return self.visit(node.children[0])

        return None

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """
        Get structure for a clause operation (calc, filter, keep, drop, rename, etc.).

        Args:
            node: The RegularAggregation node.

        Returns:
            The transformed Dataset structure.
        """
        # Get base dataset structure
        base_ds = self.visit(node.dataset) if node.dataset else None
        if base_ds is None:
            return None

        return self._transform_dataset(base_ds, node)

    def _transform_dataset(self, base_ds: Dataset, clause_node: AST.AST) -> Dataset:
        """
        Compute transformed dataset structure after applying clause operations.

        Handles chained clauses by recursively transforming.

        Args:
            base_ds: The base Dataset structure.
            clause_node: The clause AST node.

        Returns:
            The transformed Dataset structure.
        """
        from vtlengine.AST.Grammar.tokens import (
            CALC,
            DROP,
            KEEP,
            RENAME,
            SUBSPACE,
        )

        if not isinstance(clause_node, AST.RegularAggregation):
            return base_ds

        # Handle nested clauses
        if clause_node.dataset:
            nested_structure = self.visit(clause_node.dataset)
            if nested_structure:
                base_ds = nested_structure

        op = str(clause_node.op).lower()

        if op == RENAME:
            return self._transform_rename(base_ds, clause_node.children)
        elif op == DROP:
            return self._transform_drop(base_ds, clause_node.children)
        elif op == KEEP:
            return self._transform_keep(base_ds, clause_node.children)
        elif op == SUBSPACE:
            return self._transform_subspace(base_ds, clause_node.children)
        elif op == CALC:
            return self._transform_calc(base_ds, clause_node.children)

        # For filter and other clauses, return as-is
        return base_ds

    def _transform_rename(self, base_ds: Dataset, children: List[AST.AST]) -> Dataset:
        """Transform structure for rename clause."""
        new_components: Dict[str, Component] = {}
        renames: Dict[str, str] = {}

        for child in children:
            if isinstance(child, AST.RenameNode):
                renames[child.old_name] = child.new_name

        for name, comp in base_ds.components.items():
            if name in renames:
                new_name = renames[name]
                new_components[new_name] = Component(
                    name=new_name,
                    data_type=comp.data_type,
                    role=comp.role,
                    nullable=comp.nullable,
                )
            else:
                new_components[name] = comp

        return Dataset(name=base_ds.name, components=new_components, data=None)

    def _transform_drop(self, base_ds: Dataset, children: List[AST.AST]) -> Dataset:
        """Transform structure for drop clause."""
        drop_cols: set[str] = set()
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                drop_cols.add(self._resolve_varid_value(child))

        new_components = {
            name: comp for name, comp in base_ds.components.items() if name not in drop_cols
        }
        return Dataset(name=base_ds.name, components=new_components, data=None)

    def _transform_keep(self, base_ds: Dataset, children: List[AST.AST]) -> Dataset:
        """Transform structure for keep clause."""
        # Identifiers are always kept
        keep_cols: set[str] = {
            name for name, comp in base_ds.components.items() if comp.role == Role.IDENTIFIER
        }
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                keep_cols.add(self._resolve_varid_value(child))

        new_components = {
            name: comp for name, comp in base_ds.components.items() if name in keep_cols
        }
        return Dataset(name=base_ds.name, components=new_components, data=None)

    def _transform_subspace(self, base_ds: Dataset, children: List[AST.AST]) -> Dataset:
        """Transform structure for subspace clause."""
        remove_cols: set[str] = set()
        for child in children:
            if isinstance(child, AST.BinOp):
                col_name = child.left.value if hasattr(child.left, "value") else str(child.left)
                remove_cols.add(col_name)

        new_components = {
            name: comp for name, comp in base_ds.components.items() if name not in remove_cols
        }
        return Dataset(name=base_ds.name, components=new_components, data=None)

    def _transform_calc(self, base_ds: Dataset, children: List[AST.AST]) -> Dataset:
        """Transform structure for calc clause."""
        from vtlengine.DataTypes import String

        new_components = dict(base_ds.components)

        for child in children:
            # Calc children are wrapped in UnaryOp with role
            if isinstance(child, AST.UnaryOp) and hasattr(child, "operand"):
                assignment = child.operand
                role_str = str(child.op).lower()
                if role_str == "measure":
                    role = Role.MEASURE
                elif role_str == "identifier":
                    role = Role.IDENTIFIER
                elif role_str == "attribute":
                    role = Role.ATTRIBUTE
                else:
                    role = Role.MEASURE
            elif isinstance(child, AST.Assignment):
                assignment = child
                role = Role.MEASURE
            else:
                continue

            if isinstance(assignment, AST.Assignment):
                if not isinstance(assignment.left, (AST.VarID, AST.Identifier)):
                    continue
                col_name = assignment.left.value
                if col_name not in new_components:
                    is_nullable = role != Role.IDENTIFIER
                    new_components[col_name] = Component(
                        name=col_name,
                        data_type=String,
                        role=role,
                        nullable=is_nullable,
                    )

        return Dataset(name=base_ds.name, components=new_components, data=None)

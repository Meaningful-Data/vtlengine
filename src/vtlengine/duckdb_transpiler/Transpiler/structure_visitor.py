"""
Structure visitor for the SQL Transpiler.

Resolves dataset structures, operand types, UDO parameters, component names,
SQL literals, and time/group columns from VTL AST nodes.

Can be used **standalone** (instantiated directly) to compute output dataset
structures from AST nodes, or as a **base class** for ``SQLTranspiler`` which
inherits these resolution methods while overriding the ``visit_*`` methods
with SQL-generating implementations.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar import tokens
from vtlengine.DataTypes import Boolean, Date, Integer, Number, TimePeriod
from vtlengine.DataTypes import String as StringType
from vtlengine.duckdb_transpiler.Transpiler.sql_builder import quote_identifier
from vtlengine.Model import Component, Dataset, Role

# Operand type constants
_DATASET = "Dataset"
_COMPONENT = "Component"
_SCALAR = "Scalar"

# VTL type name → Python DataType mapping (for cast structure resolution)
_VTL_TYPE_MAP: Dict[str, Any] = {
    "Integer": Integer,
    "Number": Number,
    "String": StringType,
    "Boolean": Boolean,
}


class StructureVisitor(ASTTemplate):
    """Visitor that resolves dataset structures from VTL AST nodes.

    When used standalone, the ``visit_*`` methods return ``Optional[Dataset]``.
    When inherited by ``SQLTranspiler``, the transpiler's own ``visit_*``
    methods (returning SQL strings) take precedence via normal MRO.
    """

    # -- Standalone constructor -----------------------------------------------
    # When used as a base class for the SQLTranspiler dataclass, this __init__
    # is NOT called — the dataclass-generated __init__ + __post_init__ set up
    # the same attributes.

    def __init__(
        self,
        available_tables: Optional[Dict[str, Dataset]] = None,
        output_datasets: Optional[Dict[str, Dataset]] = None,
        scalars: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_datasets: Dict[str, Dataset] = output_datasets or {}
        self.available_tables: Dict[str, Dataset] = {
            **(available_tables or {}),
            **self.output_datasets,
        }
        self.scalars: Dict[str, Any] = scalars or {}
        self.current_assignment: str = ""
        self._in_clause: bool = False
        self._current_dataset: Optional[Dataset] = None
        self._join_alias_map: Dict[str, str] = {}
        self._udo_params: Optional[List[Dict[str, Any]]] = None
        self._udos: Dict[str, Dict[str, Any]] = {}
        self._structure_context: Dict[int, Dataset] = {}

    # -- Public API for standalone usage --------------------------------------

    @property
    def udos(self) -> Dict[str, Dict[str, Any]]:
        """Public access to UDO definitions."""
        return self._udos

    @udos.setter
    def udos(self, value: Dict[str, Dict[str, Any]]) -> None:
        self._udos = value

    def get_udo_param(self, name: str) -> Any:
        """Public wrapper around :meth:`_get_udo_param`."""
        return self._get_udo_param(name)

    def push_udo_params(self, params: Dict[str, Any]) -> None:
        """Public wrapper around :meth:`_push_udo_params`."""
        self._push_udo_params(params)

    def pop_udo_params(self) -> None:
        """Public wrapper around :meth:`_pop_udo_params`."""
        self._pop_udo_params()

    def clear_context(self) -> None:
        """Clear the structure cache."""
        self._structure_context.clear()

    # =========================================================================
    # Standalone visit_* methods (return Optional[Dataset])
    #
    # These are overridden by SQLTranspiler's visit_* methods (returning str)
    # when the class is used as a base class.
    # =========================================================================

    def visit_VarID(self, node: AST.VarID) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a VarID."""
        return self._get_dataset_structure(node)

    def visit_BinOp(self, node: AST.BinOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a BinOp."""
        return self._get_dataset_structure(node)

    def visit_UnaryOp(self, node: AST.UnaryOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a UnaryOp.

        ``isnull`` replaces all measures with a single ``bool_var`` measure.
        """
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            return None
        op = str(node.op).lower()
        if op == tokens.ISNULL:
            comps: Dict[str, Component] = {
                n: c for n, c in ds.components.items() if c.role == Role.IDENTIFIER
            }
            comps["bool_var"] = Component(
                name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
            )
            return Dataset(name=ds.name, components=comps, data=None)
        return ds

    def visit_ParamOp(self, node: AST.ParamOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a ParamOp.

        ``cast`` updates measure data types to the target type.
        """
        op = str(node.op).lower()
        if op == tokens.CAST and len(node.children) >= 2:
            ds = self._get_dataset_structure(node.children[0])
            if ds is None:
                return None
            type_node = node.children[1]
            target_str = type_node.value if hasattr(type_node, "value") else str(type_node)
            target_type = _VTL_TYPE_MAP.get(target_str, Number)
            comps: Dict[str, Component] = {}
            for name, comp in ds.components.items():
                if comp.role == Role.MEASURE:
                    comps[name] = Component(
                        name=name, data_type=target_type, role=comp.role, nullable=comp.nullable
                    )
                else:
                    comps[name] = comp
            return Dataset(name=ds.name, components=comps, data=None)
        return self._get_dataset_structure(node)

    def visit_RegularAggregation(  # type: ignore[override]
        self, node: AST.RegularAggregation
    ) -> Optional[Dataset]:
        """Return dataset structure for a clause operation."""
        return self._get_dataset_structure(node)

    def visit_Aggregation(  # type: ignore[override]
        self, node: AST.Aggregation
    ) -> Optional[Dataset]:
        """Return dataset structure for an aggregation.

        Handles ``group by``, ``group except``, and scalar aggregation
        (no grouping → all identifiers removed).
        """
        if node.operand is None:
            return None
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            return None
        if node.grouping is not None or node.grouping_op is not None:
            all_ids = ds.get_identifiers_names()
            group_cols = set(self._resolve_group_cols(node, all_ids))
            comps: Dict[str, Component] = {}
            for name, comp in ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    if name in group_cols:
                        comps[name] = comp
                else:
                    comps[name] = comp
            return Dataset(name=ds.name, components=comps, data=None)
        # No grouping → scalar aggregation → remove all identifiers
        comps = {n: c for n, c in ds.components.items() if c.role != Role.IDENTIFIER}
        return Dataset(name=ds.name, components=comps, data=None)

    def visit_JoinOp(self, node: AST.JoinOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a join operation."""
        return self._get_dataset_structure(node)

    def visit_UDOCall(self, node: AST.UDOCall) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a UDO call."""
        return self._get_dataset_structure(node)

    def generic_visit(self, node: AST.AST) -> None:  # type: ignore[override]
        """Return None for any unhandled node type."""
        return None

    # =========================================================================
    # Operand type resolution
    # =========================================================================

    def _get_operand_type(self, node: AST.AST) -> str:  # noqa: C901
        """Determine the operand type of a node."""
        if isinstance(node, AST.VarID):
            return self._get_varid_type(node)
        if isinstance(node, (AST.Constant, AST.ParamConstant, AST.Collection)):
            return _SCALAR
        if isinstance(node, (AST.RegularAggregation, AST.JoinOp)):
            return _DATASET
        if isinstance(node, AST.Aggregation):
            if self._in_clause:
                return _SCALAR
            if node.operand:
                return self._get_operand_type(node.operand)
            return _SCALAR
        if isinstance(node, AST.Analytic):
            return _COMPONENT
        if isinstance(node, AST.BinOp):
            return self._get_binop_type(node)
        if isinstance(node, AST.UnaryOp):
            return self._get_operand_type(node.operand)
        if isinstance(node, AST.ParamOp):
            if node.children:
                return self._get_operand_type(node.children[0])
            return _SCALAR
        if isinstance(node, AST.MulOp):
            return self._get_mulop_type(node)
        if isinstance(node, AST.If):
            return self._get_operand_type(node.thenOp)
        if isinstance(node, AST.Case):
            if node.cases:
                return self._get_operand_type(node.cases[0].thenOp)
            return _SCALAR
        if isinstance(node, AST.UDOCall):
            if node.op in self._udos:
                return self._get_operand_type(self._udos[node.op]["expression"])
            return _SCALAR
        return _SCALAR

    def _get_binop_type(self, node: AST.BinOp) -> str:
        """Determine operand type for a BinOp."""
        left_t = self._get_operand_type(node.left)
        if left_t == _DATASET:
            return _DATASET
        right_t = self._get_operand_type(node.right)
        if right_t == _DATASET:
            return _DATASET
        return _SCALAR

    def _get_mulop_type(self, node: AST.MulOp) -> str:
        """Determine operand type for a MulOp."""
        op = str(node.op).lower()
        if op in (tokens.UNION, tokens.INTERSECT, tokens.SETDIFF, tokens.SYMDIFF):
            return _DATASET
        if op == tokens.EXISTS_IN:
            return _DATASET
        return _SCALAR

    def _get_varid_type(self, node: AST.VarID) -> str:
        """Determine operand type for a VarID."""
        name = node.value
        udo_val = self._get_udo_param(name)
        if udo_val is not None:
            # Check VarID specifically to avoid infinite recursion when
            # a UDO param name matches its argument name.
            if isinstance(udo_val, AST.VarID):
                if udo_val.value in self.available_tables:
                    return _DATASET
                if udo_val.value != name:
                    return self._get_operand_type(udo_val)
                return _SCALAR
            if isinstance(udo_val, AST.AST):
                return self._get_operand_type(udo_val)
            if isinstance(udo_val, str) and udo_val in self.available_tables:
                return _DATASET
            return _SCALAR
        if self._in_clause and self._current_dataset and name in self._current_dataset.components:
            return _COMPONENT
        if name in self.available_tables:
            return _DATASET
        if name in self.scalars:
            return _SCALAR
        return _SCALAR

    def _is_dataset(self, node: AST.AST) -> bool:
        """Check if a node represents a dataset-level operand."""
        return self._get_operand_type(node) == _DATASET

    # =========================================================================
    # Output dataset resolution
    # =========================================================================

    def _get_output_dataset(self) -> Optional[Dataset]:
        """Get the current assignment's output dataset."""
        return self.output_datasets.get(self.current_assignment)

    # =========================================================================
    # SQL literal conversion
    # =========================================================================

    def _to_sql_literal(self, value: Any, type_name: str = "") -> str:
        """Convert a Python value to a SQL literal string."""
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "TRUE" if value else "FALSE"
        if isinstance(value, str):
            if type_name == "Date":
                return f"DATE '{value}'"
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        if isinstance(value, (int, float)):
            return str(value)
        return str(value)

    def _constant_to_sql(self, node: AST.Constant) -> str:
        """Convert a Constant AST node to a SQL literal."""
        type_name = ""
        if node.type_:
            type_str = str(node.type_).upper()
            if "DATE" in type_str:
                type_name = "Date"
        return self._to_sql_literal(node.value, type_name)

    # =========================================================================
    # Dataset SQL source resolution
    # =========================================================================

    def _get_dataset_sql(self, node: AST.AST) -> str:
        """Get the SQL FROM source for a dataset node."""
        if isinstance(node, AST.VarID):
            name = node.value
            udo_val = self._get_udo_param(name)
            if udo_val is not None:
                if isinstance(udo_val, AST.VarID):
                    return quote_identifier(udo_val.value)
                if isinstance(udo_val, AST.AST):
                    inner_sql = self.visit(udo_val)
                    return f"({inner_sql})"
            return quote_identifier(name)
        inner_sql = self.visit(node)
        return f"({inner_sql})"

    def _resolve_dataset_name(self, node: AST.AST) -> str:
        """Resolve a VarID to its actual dataset name (handles UDO params)."""
        if isinstance(node, AST.VarID):
            udo_val = self._get_udo_param(node.value)
            if udo_val is not None:
                if isinstance(udo_val, AST.VarID):
                    return udo_val.value
                if isinstance(udo_val, AST.AST):
                    return self._resolve_dataset_name(udo_val)
                if isinstance(udo_val, str):
                    return udo_val
            return node.value
        if isinstance(node, AST.RegularAggregation) and node.dataset:
            return self._resolve_dataset_name(node.dataset)
        return ""

    # =========================================================================
    # UDO parameter handling
    # =========================================================================

    def _get_udo_param(self, name: str) -> Any:
        """Look up a UDO parameter by name from the current scope."""
        if self._udo_params is None:
            return None
        for scope in reversed(self._udo_params):
            if name in scope:
                return scope[name]
        return None

    def _push_udo_params(self, params: Dict[str, Any]) -> None:
        """Push a new UDO parameter scope onto the stack."""
        if self._udo_params is None:
            self._udo_params = []
        self._udo_params.append(params)

    def _pop_udo_params(self) -> None:
        """Pop the innermost UDO parameter scope from the stack."""
        if self._udo_params:
            self._udo_params.pop()
            if len(self._udo_params) == 0:
                self._udo_params = None

    # =========================================================================
    # Dataset structure resolution
    # =========================================================================

    def _get_dataset_structure(self, node: AST.AST) -> Optional[Dataset]:  # noqa: C901
        """Get dataset structure for a node, tracing to the source dataset."""
        if isinstance(node, AST.VarID):
            udo_val = self._get_udo_param(node.value)
            if udo_val is not None:
                # Check VarID specifically to avoid infinite recursion when
                # a UDO param name matches its argument name (e.g., DS → VarID('DS')).
                if isinstance(udo_val, AST.VarID):
                    if udo_val.value in self.available_tables:
                        return self.available_tables[udo_val.value]
                    # Avoid recursing with same name (would loop)
                    if udo_val.value != node.value:
                        return self._get_dataset_structure(udo_val)
                    return None
                if isinstance(udo_val, AST.AST):
                    return self._get_dataset_structure(udo_val)
                if isinstance(udo_val, str) and udo_val in self.available_tables:
                    return self.available_tables[udo_val]
            return self.available_tables.get(node.value)

        if isinstance(node, AST.RegularAggregation) and node.dataset:
            op = str(node.op).lower() if node.op else ""
            if op == tokens.UNPIVOT and len(node.children) >= 2:
                result = self._build_unpivot_structure(node)
                if result is not None:
                    return result
            if op == tokens.CALC:
                result = self._build_calc_structure(node)
                if result is not None:
                    return result
            if op == tokens.AGGREGATE:
                return self._build_aggregate_clause_structure(node)
            if op == tokens.RENAME:
                return self._build_rename_structure(node)
            if op == tokens.DROP:
                return self._build_drop_structure(node)
            if op == tokens.KEEP:
                return self._build_keep_structure(node)
            if op == tokens.SUBSPACE:
                return self._build_subspace_structure(node)
            return self._get_dataset_structure(node.dataset)

        if isinstance(node, AST.BinOp):
            op = str(node.op).lower()
            if op == tokens.MEMBERSHIP:
                return self._build_membership_structure(node)
            if op == "as":
                return self._get_dataset_structure(node.left)
            if self._get_operand_type(node.left) == _DATASET:
                return self._get_dataset_structure(node.left)
            if self._get_operand_type(node.right) == _DATASET:
                return self._get_dataset_structure(node.right)
            return None

        if isinstance(node, AST.UnaryOp):
            return self._get_dataset_structure(node.operand)

        if isinstance(node, AST.ParamOp):
            if node.children:
                return self._get_dataset_structure(node.children[0])
            return None

        if isinstance(node, AST.Aggregation) and node.operand:
            ds = self._get_dataset_structure(node.operand)
            if ds is not None and (node.grouping is not None or node.grouping_op is not None):
                all_ids = ds.get_identifiers_names()
                group_cols = set(self._resolve_group_cols(node, all_ids))
                comps = {}
                for name, comp in ds.components.items():
                    if comp.role == Role.IDENTIFIER:
                        if name in group_cols:
                            comps[name] = comp
                    else:
                        comps[name] = comp
                return Dataset(name=ds.name, components=comps, data=None)
            return ds

        if isinstance(node, AST.JoinOp):
            return self._build_join_structure(node)

        if isinstance(node, AST.UDOCall):
            if node.op in self._udos:
                udo_def = self._udos[node.op]
                expression = udo_def["expression"]
                bindings: Dict[str, Any] = {}
                for i, param_info in enumerate(udo_def["params"]):
                    param_name = param_info["name"]
                    if i < len(node.params):
                        bindings[param_name] = node.params[i]
                    elif param_info.get("default") is not None:
                        bindings[param_name] = param_info["default"]
                self._push_udo_params(bindings)
                try:
                    result = self._get_dataset_structure(expression)
                finally:
                    self._pop_udo_params()
                return result
            return self._get_output_dataset()

        if isinstance(node, AST.MulOp) and node.children:
            return self._get_dataset_structure(node.children[0])

        if isinstance(node, AST.If):
            return self._get_dataset_structure(node.thenOp)

        if isinstance(node, AST.Case) and node.cases:
            return self._get_dataset_structure(node.cases[0].thenOp)

        return None

    # =========================================================================
    # Structure builders for clause operations
    # =========================================================================

    def _build_unpivot_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output dataset structure for an unpivot clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        new_id = (
            node.children[0].value if hasattr(node.children[0], "value") else str(node.children[0])
        )
        new_measure = (
            node.children[1].value if hasattr(node.children[1], "value") else str(node.children[1])
        )
        comps = {
            name: comp for name, comp in input_ds.components.items() if comp.role == Role.IDENTIFIER
        }
        comps[new_id] = Component(
            name=new_id, data_type=StringType, role=Role.IDENTIFIER, nullable=False
        )
        measure_types = [
            c.data_type for c in input_ds.components.values() if c.role == Role.MEASURE
        ]
        m_type = measure_types[0] if measure_types else StringType
        comps[new_measure] = Component(
            name=new_measure, data_type=m_type, role=Role.MEASURE, nullable=True
        )
        return Dataset(name="_unpivot", components=comps, data=None)

    def _build_calc_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output dataset structure for a calc clause.

        The result contains all input columns plus any new columns defined
        by the calc assignments.  This is needed when a calc is used as an
        intermediate result (e.g. chained ``[calc A][calc B]``).
        """
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None

        output_ds = self._get_output_dataset()
        comps = dict(input_ds.components)
        for child in node.children:
            assignment = child
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                col_name = assignment.left.value if hasattr(assignment.left, "value") else ""
                # Resolve UDO component parameters for column names
                udo_val = self._get_udo_param(col_name)
                if udo_val is not None:
                    if isinstance(udo_val, (AST.VarID, AST.Identifier)):
                        col_name = udo_val.value
                    elif isinstance(udo_val, str):
                        col_name = udo_val
                if col_name not in comps and output_ds and col_name in output_ds.components:
                    comps[col_name] = output_ds.components[col_name]
                elif col_name not in comps:
                    from vtlengine.DataTypes import Number as NumberType

                    comps[col_name] = Component(
                        name=col_name, data_type=NumberType, role=Role.MEASURE, nullable=True
                    )
        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_aggregate_clause_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output dataset structure for an aggregate clause.

        After ``[aggr Me := func() group by Id]``, the result contains only
        the group-by identifiers and the computed measures.
        """
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None

        from vtlengine.DataTypes import Number as NumberType

        comps: Dict[str, Component] = {}

        # Determine group-by identifiers from children or default to all
        group_ids: set = set()
        for child in node.children:
            assignment = child
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                agg_node = assignment.right
                if (
                    isinstance(agg_node, AST.Aggregation)
                    and agg_node.grouping
                    and agg_node.grouping_op == "group by"
                ):
                    for g in agg_node.grouping:
                        if isinstance(g, (AST.VarID, AST.Identifier)):
                            group_ids.add(g.value)

        # Add group-by identifiers
        for name, comp in input_ds.components.items():
            if comp.role == Role.IDENTIFIER and name in group_ids:
                comps[name] = comp

        # Add computed measures
        for child in node.children:
            assignment = child
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                col_name = assignment.left.value if hasattr(assignment.left, "value") else ""
                comps[col_name] = Component(
                    name=col_name, data_type=NumberType, role=Role.MEASURE, nullable=True
                )

        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_membership_structure(self, node: AST.BinOp) -> Optional[Dataset]:
        """Build the output structure for a membership (#) operation.

        ``DS#comp`` returns identifiers + the single extracted component.
        """
        parent_ds = self._get_dataset_structure(node.left)
        if parent_ds is None:
            return None

        comp_name = node.right.value if hasattr(node.right, "value") else str(node.right)

        comps: Dict[str, Component] = {}
        for name, comp in parent_ds.components.items():
            if comp.role == Role.IDENTIFIER:
                comps[name] = comp

        # Add the extracted component as a measure
        if comp_name in parent_ds.components:
            orig = parent_ds.components[comp_name]
            comps[comp_name] = Component(
                name=comp_name, data_type=orig.data_type, role=Role.MEASURE, nullable=True
            )
        else:
            from vtlengine.DataTypes import Number as NumberType

            comps[comp_name] = Component(
                name=comp_name, data_type=NumberType, role=Role.MEASURE, nullable=True
            )
        return Dataset(name=parent_ds.name, components=comps, data=None)

    def _build_rename_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a rename clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None

        renames: Dict[str, str] = {}
        for child in node.children:
            if isinstance(child, AST.RenameNode):
                old = child.old_name
                # Check if alias-qualified name exists in input dataset
                if "#" in old and old in input_ds.components:
                    renames[old] = child.new_name
                elif "#" in old:
                    # Strip alias prefix from membership refs (e.g. d2#Me_2 -> Me_2)
                    old = old.split("#", 1)[1]
                    renames[old] = child.new_name
                else:
                    renames[old] = child.new_name

        comps: Dict[str, Component] = {}
        for name, comp in input_ds.components.items():
            if name in renames:
                new_name = renames[name]
                comps[new_name] = Component(
                    name=new_name,
                    data_type=comp.data_type,
                    role=comp.role,
                    nullable=comp.nullable,
                )
            else:
                comps[name] = comp

        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_drop_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a drop clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        drop_names = self._resolve_clause_component_names(node.children, input_ds)
        comps = {name: comp for name, comp in input_ds.components.items() if name not in drop_names}
        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_subspace_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a subspace clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        remove_ids: set = set()
        for child in node.children:
            if isinstance(child, AST.BinOp):
                col_name = child.left.value if hasattr(child.left, "value") else ""
                remove_ids.add(col_name)
        comps = {name: comp for name, comp in input_ds.components.items() if name not in remove_ids}
        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_keep_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a keep clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        # Identifiers are always kept
        keep_names = {
            name for name, comp in input_ds.components.items() if comp.role == Role.IDENTIFIER
        }
        keep_names |= self._resolve_clause_component_names(node.children, input_ds)
        comps = {name: comp for name, comp in input_ds.components.items() if name in keep_names}
        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_join_structure(self, node: AST.JoinOp) -> Optional[Dataset]:
        """Build the output structure for a join operation from its clauses.

        Merges all components from all joined datasets.  When multiple datasets
        share a non-identifier column name the duplicates are qualified with
        ``alias#comp`` – mirroring the VDS convention used by the interpreter.
        """
        # Determine the using identifiers for this join
        using_ids: Optional[List[str]] = None
        if node.using:
            using_ids = list(node.using)

        # Collect (alias, dataset) pairs
        clause_datasets: List[tuple] = []
        for i, clause in enumerate(node.clauses):
            actual_node = clause
            alias: Optional[str] = None
            if isinstance(clause, AST.BinOp) and str(clause.op).lower() == "as":
                actual_node = clause.left
                alias = clause.right.value if hasattr(clause.right, "value") else str(clause.right)
            ds = self._get_dataset_structure(actual_node)
            if alias is None:
                # Use the dataset name as alias (same convention as interpreter)
                alias = ds.name if ds else chr(ord("a") + i)
            if ds:
                clause_datasets.append((alias, ds))

        if not clause_datasets:
            return self._get_output_dataset()

        # Determine common identifiers if no USING specified
        # Use pairwise accumulation (same as visit_JoinOp) so that multi-
        # dataset joins where secondary datasets share different identifiers
        # work correctly.
        if using_ids is None:
            accumulated_ids = set(clause_datasets[0][1].get_identifiers_names())
            all_join_ids: Set[str] = set(accumulated_ids)
            for _, ds in clause_datasets[1:]:
                ds_ids = set(ds.get_identifiers_names())
                all_join_ids |= ds_ids
                accumulated_ids |= ds_ids
        else:
            all_join_ids = set(using_ids)

        # Find non-identifier component names that appear in more than one dataset
        comp_count: Dict[str, int] = {}
        for _, ds in clause_datasets:
            for comp_name in ds.components:
                if comp_name not in all_join_ids:
                    comp_count[comp_name] = comp_count.get(comp_name, 0) + 1

        duplicate_comps = {name for name, cnt in comp_count.items() if cnt >= 2}

        is_cross = str(node.op).lower() == tokens.CROSS_JOIN

        comps: Dict[str, Component] = {}
        for alias, ds in clause_datasets:
            for comp_name, comp in ds.components.items():
                is_join_id = comp.role == Role.IDENTIFIER or comp_name in all_join_ids
                if comp_name in duplicate_comps and (not is_join_id or is_cross):
                    qualified = f"{alias}#{comp_name}"
                    new_comp = Component(
                        name=qualified,
                        data_type=comp.data_type,
                        role=comp.role,
                        nullable=comp.nullable,
                    )
                    comps[qualified] = new_comp
                elif comp_name not in comps:
                    comps[comp_name] = comp
        if not comps:
            return self._get_output_dataset()
        return Dataset(name="_join", components=comps, data=None)

    # =========================================================================
    # Component name resolution helpers
    # =========================================================================

    def _resolve_clause_component_names(self, children: List[AST.AST], input_ds: Dataset) -> set:
        """Extract component names from clause children (keep/drop), resolving memberships."""
        names: set = set()
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                names.add(child.value)
            elif isinstance(child, AST.BinOp) and str(child.op).lower() == tokens.MEMBERSHIP:
                ds_alias = child.left.value if hasattr(child.left, "value") else str(child.left)
                comp = child.right.value if hasattr(child.right, "value") else str(child.right)
                qualified = f"{ds_alias}#{comp}"
                names.add(qualified if qualified in input_ds.components else comp)
        return names

    def _resolve_join_component_names(self, children: List[AST.AST]) -> List[str]:
        """Extract component names from clause children, resolving via join alias map."""
        names: List[str] = []
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                names.append(child.value)
            elif isinstance(child, AST.BinOp) and str(child.op).lower() == tokens.MEMBERSHIP:
                ds_alias = child.left.value if hasattr(child.left, "value") else str(child.left)
                comp = child.right.value if hasattr(child.right, "value") else str(child.right)
                qualified = f"{ds_alias}#{comp}"
                names.append(qualified if qualified in self._join_alias_map else comp)
        return names

    # =========================================================================
    # Time and group column helpers
    # =========================================================================

    def _split_time_identifier(self, ds: Dataset) -> Tuple[str, List[str]]:
        """Split identifiers into time identifier and other identifiers."""
        time_types = (Date, TimePeriod)
        time_id = ""
        other_ids: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                if comp.data_type in time_types:
                    time_id = name
                else:
                    other_ids.append(name)
        if not time_id and ds.get_identifiers_names():
            all_ids = ds.get_identifiers_names()
            time_id = all_ids[0]
            other_ids = all_ids[1:]
        return time_id, other_ids

    def _resolve_group_cols(
        self,
        node: AST.Aggregation,
        all_ids: List[str],
    ) -> List[str]:
        """Resolve group-by columns from an Aggregation node."""
        if node.grouping and node.grouping_op == "group by":
            group_cols: List[str] = []
            for g in node.grouping:
                if isinstance(g, (AST.VarID, AST.Identifier)):
                    resolved = g.value
                    udo_val = self._get_udo_param(resolved)
                    if udo_val is not None:
                        if isinstance(udo_val, (AST.VarID, AST.Identifier)):
                            resolved = udo_val.value
                        elif isinstance(udo_val, str):
                            resolved = udo_val
                    group_cols.append(resolved)
            return group_cols
        if node.grouping and node.grouping_op == "group except":
            except_cols: set = set()
            for g in node.grouping:
                if isinstance(g, (AST.VarID, AST.Identifier)):
                    resolved = g.value
                    udo_val = self._get_udo_param(resolved)
                    if udo_val is not None:
                        if isinstance(udo_val, (AST.VarID, AST.Identifier)):
                            resolved = udo_val.value
                        elif isinstance(udo_val, str):
                            resolved = udo_val
                    except_cols.add(resolved)
            return [id_ for id_ in all_ids if id_ not in except_cols]
        if node.grouping_op is None and not node.grouping:
            return []
        return list(all_ids)

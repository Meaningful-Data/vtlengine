"""Resolve VTL dataset structures for the DuckDB transpiler."""

from typing import Any, Dict, List, Optional, Set, Tuple

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar import tokens
from vtlengine.DataTypes import (
    _DUCKDB_TYPE_TO_VTL,
    COMP_NAME_MAPPING,
    SCALAR_TYPES,
    Boolean,
    Date,
    Integer,
    Number,
    TimeInterval,
    TimePeriod,
)
from vtlengine.DataTypes import String as StringType
from vtlengine.duckdb_transpiler.Transpiler.sql_builder import quote_name
from vtlengine.Model import Component, Dataset, Role

# Operand type tags
_DATASET = "Dataset"
_COMPONENT = "Component"
_SCALAR = "Scalar"

# Role encoded in UnaryOp.op for calc clauses.
_CALC_ROLE_BY_TOKEN: Dict[str, Role] = {
    tokens.IDENTIFIER: Role.IDENTIFIER,
    tokens.ATTRIBUTE: Role.ATTRIBUTE,
    tokens.MEASURE: Role.MEASURE,
}


class StructureVisitor(ASTTemplate):
    """Visitor that resolves dataset structures from VTL AST nodes."""

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

    # Dispatcher: two-level visit — first ``visit_{Class}_{op}``, then ``visit_{Class}``

    def visit(self, node: Any) -> Any:
        """Dispatch by node class and, if present, by ``node.op``."""
        op = getattr(node, "op", None)
        if isinstance(op, str) and op.isidentifier():
            handler = getattr(self, f"visit_{type(node).__name__}_{op}", None)
            if handler is not None:
                return handler(node)
        return super().visit(node)

    # Public API for standalone usage

    @property
    def udos(self) -> Dict[str, Dict[str, Any]]:
        """Public access to UDO definitions."""
        return self._udos

    @udos.setter
    def udos(self, value: Dict[str, Dict[str, Any]]) -> None:
        self._udos = value

    def get_udo_param(self, name: str) -> Any:
        """Return a UDO parameter from the current scope."""
        return self._get_udo_param(name)

    def push_udo_params(self, params: Dict[str, Any]) -> None:
        """Push a UDO parameter scope."""
        self._push_udo_params(params)

    def pop_udo_params(self) -> None:
        """Pop the innermost UDO parameter scope."""
        self._pop_udo_params()

    # Standalone visit_* methods (Optional[Dataset]).
    # SQLTranspiler overrides these with SQL-generating versions.

    def visit_VarID(self, node: AST.VarID) -> Optional[Dataset]:
        """Return dataset structure for a VarID."""
        return self._get_dataset_structure(node)

    def visit_BinOp(self, node: AST.BinOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a BinOp."""
        return self._get_dataset_structure(node)

    def visit_UnaryOp(self, node: AST.UnaryOp) -> Optional[Dataset]:
        """Return dataset structure for a unary op."""
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            return None
        if node.op == tokens.ISNULL:
            return self._build_boolean_result_structure(ds)
        return ds

    def visit_ParamOp(self, node: AST.ParamOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a parameterized op."""
        if node.op == tokens.CAST and len(node.children) >= 2:
            ds = self._get_dataset_structure(node.children[0])
            if ds is None:
                return None
            target_str = self._resolve_name(node.children[1])
            target_type = SCALAR_TYPES.get(
                target_str, _DUCKDB_TYPE_TO_VTL.get(target_str.upper(), Number)
            )
            comps: Dict[str, Component] = {}
            for name, comp in ds.components.items():
                if comp.role == Role.MEASURE:
                    comps[name] = self._make_comp(name, target_type, comp.role, comp.nullable)
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
        """Return dataset structure for an aggregation."""
        if node.operand is None:
            return None
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            return None
        if node.grouping is not None or node.grouping_op is not None:
            all_ids = ds.get_identifiers_names()
            group_cols = set(self._resolve_group_cols(node, all_ids))
            # Keep the time identifier when using time_agg with group by.
            if node.grouping:
                has_time_agg = any(isinstance(g, AST.TimeAggregation) for g in node.grouping)
                if has_time_agg and node.grouping_op != "group except":
                    for comp in ds.components.values():
                        if comp.data_type in (TimePeriod, Date) and comp.role == Role.IDENTIFIER:
                            group_cols.add(comp.name)
                            break
            comps: Dict[str, Component] = {}
            for name, comp in ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    if name in group_cols:
                        comps[name] = comp
                else:
                    comps[name] = comp
            return Dataset(name=ds.name, components=comps, data=None)
        # No grouping: remove identifiers.
        comps = {n: c for n, c in ds.components.items() if c.role != Role.IDENTIFIER}
        return Dataset(name=ds.name, components=comps, data=None)

    def visit_JoinOp(self, node: AST.JoinOp) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a join operation."""
        return self._get_dataset_structure(node)

    def visit_UDOCall(self, node: AST.UDOCall) -> Optional[Dataset]:  # type: ignore[override]
        """Return dataset structure for a UDO call."""
        return self._get_dataset_structure(node)

    def generic_visit(self, node: AST.AST) -> None:
        """Return None for any unhandled node type."""
        return None

    # Operand type resolution

    def _get_op_type(self, nodes: List[Optional[AST.AST]]) -> str:
        """Determine the operand type for a list of nodes (e.g. function args)."""
        result = _SCALAR
        for node in nodes:
            if node is None:
                continue
            operand_type = self._get_node_type(node)
            if operand_type == _DATASET:
                return _DATASET
            if operand_type == _COMPONENT:
                result = _COMPONENT
        return result

    def _get_node_type(self, node: AST.AST) -> str:  # noqa: C901
        """Determine the operand type of a node."""
        if isinstance(node, (AST.Analytic, AST.Identifier)) or (
            isinstance(node, AST.BinOp) and self._in_clause
        ):
            return _COMPONENT
        elif isinstance(
            node,
            (AST.RegularAggregation, AST.JoinOp, AST.Validation, AST.HROperation, AST.DPValidation),
        ):
            return _DATASET
        elif isinstance(node, AST.VarID):
            return self._get_varid_type(node)
        elif isinstance(node, (AST.ParFunction, AST.UnaryOp, AST.Aggregation)):
            children = [node.operand]
        elif isinstance(node, AST.BinOp):
            children = [node.left, node.right]
        elif isinstance(node, (AST.MulOp, AST.ParamOp)):
            children = list(node.children)
        elif isinstance(node, AST.If):
            children = [node.condition, node.thenOp, node.elseOp]
        elif isinstance(node, AST.Case):
            children = [c.thenOp for c in node.cases]
        elif isinstance(node, AST.UDOCall) and node.op in self._udos:
            children = [self._udos[node.op]["expression"]]
        else:
            return _SCALAR
        return self._get_op_type(children)

    def _get_varid_type(self, node: AST.VarID) -> str:
        """Determine operand type for a VarID."""
        name = node.value
        kind, val = self._resolve_udo_var(name)
        if kind == "varid":
            if val.value in self.available_tables:
                return _DATASET
            if val.value != name:
                return self._get_node_type(val)
            return _SCALAR
        if kind == "ast":
            return self._get_node_type(val)
        if kind == "str":
            return _DATASET if val in self.available_tables else _SCALAR
        if self._in_clause and self._current_dataset and name in self._current_dataset.components:
            return _COMPONENT
        if name in self.available_tables:
            return _DATASET
        return _SCALAR

    def _is_dataset(self, node: AST.AST) -> bool:
        """Check if a node represents a dataset-level operand."""
        return self._get_node_type(node) == _DATASET

    # Output dataset resolution

    def _get_output_dataset(self) -> Optional[Dataset]:
        """Get the current assignment's output dataset."""
        return self.output_datasets.get(self.current_assignment)

    # SQL literal conversion

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
            if type_name == "TimePeriod":
                return f"vtl_period_normalize('{escaped}')"
            return f"'{escaped}'"
        return str(value)

    def _constant_to_sql(self, node: AST.Constant) -> str:
        """Convert a Constant AST node to a SQL literal."""
        type_name = ""
        if node.type_:
            type_str = str(node.type_).upper()
            if "DATE" in type_str:
                type_name = "Date"
        return self._to_sql_literal(node.value, type_name)

    # Dataset SQL source resolution

    def _get_dataset_sql(self, node: AST.AST) -> str:
        """Get the SQL FROM source for a dataset node."""
        if isinstance(node, AST.VarID):
            kind, val = self._resolve_udo_var(node.value)
            if kind == "varid":
                return quote_name(val.value)
            if kind == "ast":
                return f"({self.visit(val)})"
            return quote_name(node.value)
        return f"({self.visit(node)})"

    def _resolve_dataset_name(self, node: AST.AST) -> str:
        """Resolve a VarID to its actual dataset name (handles UDO params)."""
        if isinstance(node, AST.VarID):
            kind, val = self._resolve_udo_var(node.value)
            if kind == "varid":
                return val.value
            if kind == "ast":
                return self._resolve_dataset_name(val)
            if kind == "str":
                return val
            return node.value
        if isinstance(node, AST.RegularAggregation) and node.dataset:
            return self._resolve_dataset_name(node.dataset)
        return ""

    # UDO parameter handling

    def _get_udo_param(self, name: str) -> Any:
        """Look up a UDO parameter by name from the current scope."""
        if self._udo_params:
            for scope in reversed(self._udo_params):
                if name in scope:
                    return scope[name]
        return None

    def _resolve_udo_var(self, name: str) -> Tuple[str, Any]:
        """Resolve a UDO parameter binding by name."""
        udo_val = self._get_udo_param(name)
        if isinstance(udo_val, AST.VarID):
            return "varid", udo_val
        if isinstance(udo_val, AST.AST):
            return "ast", udo_val
        if isinstance(udo_val, str):
            return "str", udo_val
        return "unbound", name

    def _resolve_udo_name(self, name: str) -> str:
        """Unwrap a UDO binding to a bare name (for rename/component contexts)."""
        udo_val = self._get_udo_param(name)
        if isinstance(udo_val, (AST.VarID, AST.Identifier)):
            return udo_val.value
        if isinstance(udo_val, str):
            return udo_val
        return name

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

    # Dataset structure resolution

    def _get_dataset_structure(self, node: Optional[AST.AST]) -> Optional[Dataset]:
        """Get dataset structure for a node, tracing to the source dataset."""
        if node is None:
            return None
        if isinstance(node, AST.VarID):
            return self._resolve_varid_structure(node)
        if isinstance(node, AST.RegularAggregation) and node.dataset:
            return self._resolve_regular_aggregation_structure(node)
        if isinstance(node, AST.BinOp):
            return self._resolve_binop_structure(node)
        if isinstance(node, AST.UnaryOp):
            return self._resolve_unaryop_structure(node)
        if isinstance(node, AST.ParFunction):
            return self._get_dataset_structure(node.operand)
        if isinstance(node, AST.ParamOp):
            return self._get_dataset_structure(node.children[0]) if node.children else None
        if isinstance(node, AST.Aggregation) and node.operand:
            return self._build_aggregation_structure(node)
        if isinstance(node, AST.JoinOp):
            return self._build_join_structure(node)
        if isinstance(node, AST.UDOCall):
            return self._resolve_udocall_structure(node)
        if isinstance(node, AST.MulOp) and node.children:
            if node.op == tokens.EXISTS_IN:
                return self._build_exists_in_structure(node)
            return self._get_dataset_structure(node.children[0])
        if isinstance(node, AST.Validation):
            return self._build_validation_structure(node)
        if isinstance(node, AST.HROperation):
            return self._build_hr_operation_structure(node)
        if isinstance(node, AST.DPValidation):
            return self._build_dp_validation_structure(node)
        if isinstance(node, AST.If):
            return self._get_dataset_structure(node.thenOp) or self._get_dataset_structure(
                node.elseOp
            )
        if isinstance(node, AST.Case) and node.cases:
            return self._get_dataset_structure(node.cases[0].thenOp)
        return None

    def _resolve_varid_structure(self, node: AST.VarID) -> Optional[Dataset]:
        """Resolve a VarID (including UDO bindings) to its dataset structure."""
        kind, val = self._resolve_udo_var(node.value)
        if kind == "varid":
            if val.value in self.available_tables:
                return self.available_tables[val.value]
            # Guard against recursion when param name matches argument name.
            if val.value != node.value:
                return self._get_dataset_structure(val)
            return None
        if kind == "ast":
            return self._get_dataset_structure(val)
        if kind == "str" and val in self.available_tables:
            return self.available_tables[val]
        return self.available_tables.get(node.value)

    _CLAUSE_BUILDER_ATTRS: Dict[str, str] = {
        tokens.AGGREGATE: "_build_aggregate_clause_structure",
        tokens.RENAME: "_build_rename_structure",
        tokens.DROP: "_build_drop_structure",
        tokens.KEEP: "_build_keep_structure",
        tokens.SUBSPACE: "_build_subspace_structure",
    }

    def _resolve_regular_aggregation_structure(
        self, node: AST.RegularAggregation
    ) -> Optional[Dataset]:
        """Resolve a clause-carrying RegularAggregation to its output structure."""
        op = node.op
        # unpivot/calc fall through to the source dataset when the builder returns None.
        if op == tokens.UNPIVOT and len(node.children) >= 2:
            result = self._build_unpivot_structure(node)
            if result is not None:
                return result
        elif op == tokens.CALC:
            result = self._build_calc_structure(node)
            if result is not None:
                return result
        builder_attr = self._CLAUSE_BUILDER_ATTRS.get(op)
        if builder_attr is not None:
            return getattr(self, builder_attr)(node)
        return self._get_dataset_structure(node.dataset)

    def _resolve_binop_structure(self, node: AST.BinOp) -> Optional[Dataset]:
        """Resolve a BinOp to its dataset structure."""
        op = node.op
        if op == tokens.MEMBERSHIP:
            return self._build_membership_structure(node)
        if op == tokens.AS:
            return self._get_dataset_structure(node.left)
        left_is_ds = self._get_node_type(node.left) == _DATASET
        right_is_ds = self._get_node_type(node.right) == _DATASET
        if left_is_ds and right_is_ds:
            return self._build_ds_ds_binop_structure(node)
        if left_is_ds:
            ds = self._get_dataset_structure(node.left)
            if ds is not None and op in (tokens.IN, tokens.NOT_IN):
                return self._build_boolean_result_structure(ds)
            return ds
        if right_is_ds:
            return self._get_dataset_structure(node.right)
        return None

    def _resolve_unaryop_structure(self, node: AST.UnaryOp) -> Optional[Dataset]:
        """Resolve a UnaryOp to its dataset structure."""
        ds = self._get_dataset_structure(node.operand)
        if ds is not None and node.op == tokens.ISNULL and len(ds.get_measures_names()) == 1:
            return self._build_boolean_result_structure(ds)
        return ds

    def _build_aggregation_structure(self, node: AST.Aggregation) -> Optional[Dataset]:
        """Resolve an Aggregation (count/sum/avg/…) to its output structure."""
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            return None
        group_cols = set(self._resolve_group_cols(node, ds.get_identifiers_names()))
        is_count = node.op == tokens.COUNT
        comps: Dict[str, Component] = {}
        for name, comp in ds.components.items():
            is_kept_id = comp.role == Role.IDENTIFIER and name in group_cols
            is_kept_measure = comp.role == Role.MEASURE and not is_count
            if is_kept_id or is_kept_measure:
                comps[name] = comp
        if is_count:
            comps["int_var"] = self._make_comp("int_var", Integer)
        return Dataset(name=ds.name, components=comps, data=None)

    def _resolve_udocall_structure(self, node: AST.UDOCall) -> Optional[Dataset]:
        """Resolve a UDO call by binding its parameters and visiting the body."""
        if node.op not in self._udos:
            return self._get_output_dataset()
        udo_def = self._udos[node.op]
        bindings: Dict[str, Any] = {}
        for i, param_info in enumerate(udo_def["params"]):
            param_name = param_info["name"]
            if i < len(node.params):
                bindings[param_name] = node.params[i]
            elif param_info.get("default") is not None:
                bindings[param_name] = param_info["default"]
        self._push_udo_params(bindings)
        try:
            return self._get_dataset_structure(udo_def["expression"])
        finally:
            self._pop_udo_params()

    def _build_validation_structure(self, node: AST.Validation) -> Optional[Dataset]:
        """Build the output structure for a Validation node."""
        inner_ds = self._get_dataset_structure(node.validation)
        if inner_ds is None:
            return None
        val_comps = self._identifiers_dict(inner_ds)
        self._add_error_measures(
            val_comps,
            errorlevel_type=Integer,
            with_ruleid=False,
            with_bool_var=True,
        )
        return Dataset(name="", components=val_comps, data=None)

    # =========================================================================
    # Component construction helpers
    # =========================================================================

    @staticmethod
    def _resolve_name(node: Any) -> str:
        """Return ``node.value`` if present, else ``str(node)``."""
        return node.value if hasattr(node, "value") else str(node)

    @staticmethod
    def _make_comp(
        name: str, dtype: Any, role: Role = Role.MEASURE, nullable: bool = True
    ) -> Component:
        """Build a ``Component`` with the common field ordering."""
        return Component(name=name, data_type=dtype, role=role, nullable=nullable)

    @staticmethod
    def _identifiers_dict(ds: Dataset) -> Dict[str, Component]:
        """Return a new dict containing only the identifier components of ``ds``."""
        return {n: c for n, c in ds.components.items() if c.role == Role.IDENTIFIER}

    def _add_error_measures(
        self,
        comps: Dict[str, Component],
        *,
        errorlevel_type: Any = Number,
        with_ruleid: bool = True,
        with_imbalance: bool = True,
        with_bool_var: bool = False,
    ) -> None:
        """Append the standard validation/hierarchy error-reporting measures."""
        if with_bool_var:
            comps["bool_var"] = self._make_comp("bool_var", Boolean)
        if with_imbalance:
            comps["imbalance"] = self._make_comp("imbalance", Number)
        if with_ruleid:
            comps["ruleid"] = self._make_comp("ruleid", StringType, Role.IDENTIFIER, False)
        comps["errorcode"] = self._make_comp("errorcode", StringType)
        comps["errorlevel"] = self._make_comp("errorlevel", errorlevel_type)

    # =========================================================================
    # Structure builders for validation/hierarchy operations
    # =========================================================================

    def _build_hr_operation_structure(self, node: AST.HROperation) -> Optional[Dataset]:
        """Build output dataset structure for hierarchy/check_hierarchy."""
        inner_ds = self._get_dataset_structure(node.dataset)
        if inner_ds is None:
            return None

        comps = self._identifiers_dict(inner_ds)
        measure_name = inner_ds.get_measures_names()[0] if inner_ds.get_measures_names() else ""
        if node.op == tokens.HIERARCHY:
            # hierarchy: same structure as input (identifiers + measures)
            for name, comp in inner_ds.components.items():
                if comp.role != Role.IDENTIFIER:
                    comps[name] = comp
        else:
            # check_hierarchy: output depends on output mode
            output_mode = node.output.value if node.output else "invalid"
            if output_mode == "all_measures" and measure_name:
                comps[measure_name] = inner_ds.components[measure_name]
            with_bool_var = output_mode in ("all", "all_measures")
            if output_mode == "invalid" and measure_name:
                comps[measure_name] = inner_ds.components[measure_name]
            self._add_error_measures(comps, with_bool_var=with_bool_var)
        return Dataset(name="", components=comps, data=None)

    def _build_dp_validation_structure(self, node: AST.DPValidation) -> Optional[Dataset]:
        """Build output dataset structure for check_datapoint."""
        inner_ds = self._get_dataset_structure(node.dataset)
        if inner_ds is None:
            return None

        comps = self._identifiers_dict(inner_ds)
        output_mode = node.output.value if node.output else "invalid"
        if output_mode in ("invalid", "all_measures"):
            for name, comp in inner_ds.components.items():
                if comp.role == Role.MEASURE:
                    comps[name] = comp

        self._add_error_measures(
            comps,
            with_imbalance=False,
            with_bool_var=output_mode in ("all", "all_measures"),
        )
        return Dataset(name="", components=comps, data=None)

    def _build_exists_in_structure(self, node: AST.MulOp) -> Optional[Dataset]:
        """Build output dataset structure for exists_in."""
        left_ds = self._get_dataset_structure(node.children[0])
        if left_ds is None:
            return None

        comps = self._identifiers_dict(left_ds)
        comps["bool_var"] = self._make_comp("bool_var", Boolean)
        return Dataset(name="", components=comps, data=None)

    # =========================================================================
    # Structure builders for clause operations
    # =========================================================================

    def _build_unpivot_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output dataset structure for an unpivot clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        new_id = self._resolve_name(node.children[0])
        new_measure = self._resolve_name(node.children[1])
        comps = self._identifiers_dict(input_ds)
        comps[new_id] = self._make_comp(new_id, StringType, role=Role.IDENTIFIER, nullable=False)
        measure_types = [
            c.data_type for c in input_ds.components.values() if c.role == Role.MEASURE
        ]
        m_type = measure_types[0] if measure_types else StringType
        comps[new_measure] = self._make_comp(new_measure, m_type)
        return Dataset(name="_unpivot", components=comps, data=None)

    def _build_calc_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output dataset structure for a calc clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None

        output_ds = self._get_output_dataset()
        comps = dict(input_ds.components)
        for child in node.children:
            assignment = child
            calc_role = Role.MEASURE
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                calc_role = _CALC_ROLE_BY_TOKEN.get(child.op, Role.MEASURE)
                assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                col = self._resolve_udo_name(self._resolve_name(assignment.left))
                if col in comps and comps[col].role != calc_role:
                    old = comps[col]
                    nullable = old.nullable if calc_role != Role.IDENTIFIER else False
                    comps[col] = self._make_comp(old.name, old.data_type, calc_role, nullable)
                elif col not in comps and output_ds and col in output_ds.components:
                    comps[col] = output_ds.components[col]
                elif col not in comps:
                    comps[col] = self._make_comp(col, Number)
        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_ds_ds_binop_structure(self, node: AST.BinOp) -> Optional[Dataset]:
        """Build structure for dataset-dataset binary ops."""
        left_ds = self._get_dataset_structure(node.left)
        right_ds = self._get_dataset_structure(node.right)
        if left_ds is None or right_ds is None:
            return left_ds or right_ds

        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        all_ids = left_ids | right_ids
        right_measures = set(right_ds.get_measures_names())

        comps: Dict[str, Component] = {}
        for name, comp in left_ds.components.items():
            is_common_id = comp.role == Role.IDENTIFIER and name in all_ids
            is_common_measure = comp.role == Role.MEASURE and name in right_measures
            if is_common_id or is_common_measure:
                comps[name] = comp
        # Add identifiers from right that aren't in left
        for name, comp in right_ds.components.items():
            if comp.role == Role.IDENTIFIER and name not in comps:
                comps[name] = comp

        return Dataset(name=left_ds.name, components=comps, data=None)

    @staticmethod
    def _iter_assignments(children: List[AST.AST]) -> List[AST.Assignment]:
        """Unwrap a clause's children into their ``Assignment`` nodes."""
        result: List[AST.Assignment] = []
        for child in children:
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                result.append(child.operand)
            elif isinstance(child, AST.Assignment):
                result.append(child)
        return result

    def _build_aggregate_clause_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output dataset structure for an aggregate clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None

        all_input_ids = {n for n, c in input_ds.components.items() if c.role == Role.IDENTIFIER}
        group_ids: Set[str] = set()
        grouping_op: str = ""
        measure_names: List[str] = []

        for assignment in self._iter_assignments(node.children):
            agg_node = assignment.right
            if isinstance(agg_node, AST.Aggregation) and agg_node.grouping:
                grouping_op = agg_node.grouping_op or ""
                for g in agg_node.grouping:
                    if isinstance(g, (AST.VarID, AST.Identifier)):
                        group_ids.add(g.value)
            measure_names.append(self._resolve_name(assignment.left))

        if grouping_op == tokens.GROUP_BY:
            kept_ids = group_ids
        elif grouping_op == tokens.GROUP_EXCEPT:
            kept_ids = all_input_ids - group_ids
        else:
            kept_ids = all_input_ids

        comps: Dict[str, Component] = {
            name: comp
            for name, comp in input_ds.components.items()
            if comp.role == Role.IDENTIFIER and name in kept_ids
        }
        for col_name in measure_names:
            comps[col_name] = self._make_comp(col_name, Number)

        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_membership_structure(self, node: AST.BinOp) -> Optional[Dataset]:
        """Build the output structure for a membership (#) operation."""
        parent_ds = self._get_dataset_structure(node.left)
        if parent_ds is None:
            return None

        name = self._resolve_udo_name(self._resolve_name(node.right))
        comps = self._identifiers_dict(parent_ds)
        orig = parent_ds.components.get(name)
        if orig is None:
            comps[name] = self._make_comp(name, Number)
        else:
            alias_name = COMP_NAME_MAPPING[orig.data_type] if orig.role != Role.MEASURE else name
            comps[alias_name] = self._make_comp(alias_name, orig.data_type)
        return Dataset(name=parent_ds.name, components=comps, data=None)

    def _build_boolean_result_structure(self, ds: Dataset) -> Dataset:
        """Replace all measures with a single ``bool_var`` Boolean measure."""
        comps = self._identifiers_dict(ds)
        comps["bool_var"] = self._make_comp("bool_var", Boolean)
        return Dataset(name=ds.name, components=comps, data=None)

    def _build_rename_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a rename clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None

        renames: Dict[str, str] = {}
        for child in node.children:
            if isinstance(child, AST.RenameNode):
                old = child.old_name
                # Strip alias prefix from membership refs.
                if "#" in old and old not in input_ds.components:
                    old = old.split("#", 1)[1]
                renames[old] = child.new_name

        unqualified_to_qualified: Dict[str, str] = {}
        for comp_name in input_ds.components:
            if "#" in comp_name:
                unqual = comp_name.split("#", 1)[1]
                unqualified_to_qualified[unqual] = comp_name

        comps: Dict[str, Component] = {}
        for name, comp in input_ds.components.items():
            # Check direct match first, then try matching via qualified name
            matched_new = renames.get(name)
            if matched_new is None and "#" in name:
                unqual = name.split("#", 1)[1]
                matched_new = renames.get(unqual)
            if matched_new is not None:
                comps[matched_new] = self._make_comp(
                    matched_new, comp.data_type, role=comp.role, nullable=comp.nullable
                )
            else:
                comps[name] = comp

        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_filtered_structure(self, input_ds: Dataset, keep: Set[str]) -> Dataset:
        """Return a Dataset containing only components whose names are in ``keep``."""
        comps = {name: comp for name, comp in input_ds.components.items() if name in keep}
        return Dataset(name=input_ds.name, components=comps, data=None)

    def _build_drop_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a drop clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        drop_names = set(self._extract_component_names(node.children, input_ds.components))
        keep = {name for name in input_ds.components if name not in drop_names}
        return self._build_filtered_structure(input_ds, keep)

    def _build_subspace_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a subspace clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        remove_ids = {
            self._resolve_name(child.left)
            for child in node.children
            if isinstance(child, AST.BinOp)
        }
        keep = {name for name in input_ds.components if name not in remove_ids}
        return self._build_filtered_structure(input_ds, keep)

    def _build_keep_structure(self, node: AST.RegularAggregation) -> Optional[Dataset]:
        """Build the output structure for a keep clause."""
        input_ds = self._get_dataset_structure(node.dataset)
        if input_ds is None:
            return None
        keep = {name for name, comp in input_ds.components.items() if comp.role == Role.IDENTIFIER}
        keep |= set(self._extract_component_names(node.children, input_ds.components))
        return self._build_filtered_structure(input_ds, keep)

    def _build_join_structure(self, node: AST.JoinOp) -> Optional[Dataset]:
        """Build the output structure for a join operation from its clauses."""
        # Determine the using identifiers for this join
        using_ids: Optional[List[str]] = None
        if node.using:
            using_ids = list(node.using)

        # Collect (alias, dataset) pairs
        clause_datasets: List[tuple[Optional[str], Dataset]] = []
        for i, clause in enumerate(node.clauses):
            actual_node = clause
            alias: Optional[str] = None
            if isinstance(clause, AST.BinOp) and clause.op == tokens.AS:
                actual_node = clause.left
                alias = self._resolve_name(clause.right)
            ds = self._get_dataset_structure(actual_node)
            if alias is None:
                # Use the dataset name as alias (same convention as interpreter)
                alias = ds.name if ds else chr(ord("a") + i)
            if ds:
                clause_datasets.append((alias, ds))

        if not clause_datasets:
            return self._get_output_dataset()

        # Determine common identifiers if no USING specified
        is_cross = node.op == tokens.CROSS_JOIN
        if using_ids is None:
            if is_cross:
                all_join_ids: Set[str] = set()
            else:
                accumulated_ids = set(clause_datasets[0][1].get_identifiers_names())
                all_join_ids = set(accumulated_ids)
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

        comps: Dict[str, Component] = {}
        duplicate_comps = {name for name, cnt in comp_count.items() if cnt >= 2}
        for alias, ds in clause_datasets:
            for comp_name, comp in ds.components.items():
                is_join_id = comp.role == Role.IDENTIFIER or comp_name in all_join_ids
                if comp_name in duplicate_comps and (not is_join_id or is_cross):
                    qualified = f"{alias}#{comp_name}"
                    comps[qualified] = self._make_comp(
                        qualified, comp.data_type, role=comp.role, nullable=comp.nullable
                    )
                elif comp_name not in comps:
                    comps[comp_name] = comp
        if not comps:
            return self._get_output_dataset()
        return Dataset(name="_join", components=comps, data=None)

    # =========================================================================
    # Component name resolution helpers
    # =========================================================================

    def _extract_component_names(
        self, children: List[AST.AST], lookup: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Extract component names from clause children, resolving memberships."""
        ctx = lookup or {}
        names: List[str] = []
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                names.append(child.value)
            elif isinstance(child, AST.BinOp) and child.op == tokens.MEMBERSHIP:
                ds_alias = self._resolve_name(child.left)
                comp = self._resolve_name(child.right)
                qualified = f"{ds_alias}#{comp}"
                names.append(qualified if qualified in ctx else comp)
        return names

    # =========================================================================
    # Time and group column helpers
    # =========================================================================

    def _get_time_id(self, ds: Dataset) -> Tuple[str, List[str]]:
        """Split identifiers into time identifier and other identifiers."""
        for name, comp in ds.get_identifiers():
            if comp.data_type in (Date, TimeInterval, TimePeriod):
                time_id = name
                break
        other_ids = [name for name, comp in ds.get_identifiers() if name != time_id]
        return time_id, other_ids

    def _resolve_grouping_names(self, grouping: List[AST.AST]) -> List[str]:
        """Resolve grouping node names with UDO parameter lookup."""
        grouping_nodes = (AST.VarID, AST.Identifier)
        return [self._resolve_udo_name(g.value) for g in grouping if isinstance(g, grouping_nodes)]

    def _resolve_group_cols(self, node: AST.Aggregation, all_ids: List[str]) -> List[str]:
        """Resolve group-by columns from an Aggregation node."""
        if node.grouping and node.grouping_op == "group by":
            return self._resolve_grouping_names(node.grouping)
        if node.grouping and node.grouping_op == "group except":
            except_cols = set(self._resolve_grouping_names(node.grouping))
            return [id_ for id_ in all_ids if id_ not in except_cols]
        if node.grouping_op is None and not node.grouping:
            return []
        return list(all_ids)

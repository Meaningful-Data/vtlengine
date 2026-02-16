"""
SQL Transpiler for VTL AST.

Converts VTL AST nodes into DuckDB SQL queries using the visitor pattern.
Each top-level Assignment produces one SQL SELECT query. Queries are executed
sequentially, with results registered as tables for subsequent queries.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar import tokens
from vtlengine.DataTypes import Date, TimePeriod
from vtlengine.duckdb_transpiler.Transpiler.operators import (
    get_duckdb_type,
    registry,
)
from vtlengine.duckdb_transpiler.Transpiler.sql_builder import SQLBuilder, quote_identifier
from vtlengine.Model import Dataset, ExternalRoutine, Role, Scalar, ValueDomain

# Operand type constants (replaces StructureVisitor.OperandType)
_DATASET = "Dataset"
_COMPONENT = "Component"
_SCALAR = "Scalar"


@dataclass
class SQLTranspiler(ASTTemplate):
    """
    Transpiler that converts VTL AST to SQL queries.

    Generates one SQL query per top-level Assignment. Each query can be
    executed sequentially, with results registered as tables for subsequent queries.
    """

    # Input structures from data_structures
    input_datasets: Dict[str, Dataset] = field(default_factory=dict)
    input_scalars: Dict[str, Scalar] = field(default_factory=dict)

    # Output structures from semantic analysis
    output_datasets: Dict[str, Dataset] = field(default_factory=dict)
    output_scalars: Dict[str, Scalar] = field(default_factory=dict)

    value_domains: Dict[str, ValueDomain] = field(default_factory=dict)
    external_routines: Dict[str, ExternalRoutine] = field(default_factory=dict)

    # DAG of dataset dependencies for execution order
    dag: Any = field(default=None)

    # RunTime context
    current_assignment: str = ""
    inputs: List[str] = field(default_factory=list)
    clause_context: List[str] = field(default_factory=list)

    # Merged lookup tables (populated in __post_init__)
    datasets: Dict[str, Dataset] = field(default_factory=dict, init=False)
    scalars: Dict[str, Scalar] = field(default_factory=dict, init=False)
    available_tables: Dict[str, Dataset] = field(default_factory=dict, init=False)

    # Clause context (replaces structure_visitor.in_clause / current_dataset)
    _in_clause: bool = field(default=False, init=False)
    _current_dataset: Optional[Dataset] = field(default=None, init=False)

    # UDO definitions: name -> Operator node info
    _udos: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # UDO parameter stack
    _udo_params: Optional[List[Dict[str, Any]]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize available tables."""
        self.datasets = {**self.input_datasets, **self.output_datasets}
        self.scalars = {**self.input_scalars, **self.output_scalars}
        self.available_tables = dict(self.datasets)

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _get_assignment_inputs(self, name: str) -> List[str]:
        if self.dag is None:
            return []
        if hasattr(self.dag, "dependencies"):
            for deps in self.dag.dependencies.values():
                if name in deps.get("outputs", []) or name in deps.get("persistent", []):
                    return deps.get("inputs", [])
        return []

    def _get_operand_type(self, node: AST.AST) -> str:  # noqa: C901
        """Determine the operand type of a node without StructureVisitor."""
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

    def _get_output_dataset(self) -> Optional[Dataset]:
        """Get the current assignment's output dataset."""
        return self.output_datasets.get(self.current_assignment)

    def _is_dataset(self, node: AST.AST) -> bool:
        """Check if a node represents a dataset-level operand."""
        return self._get_operand_type(node) == _DATASET

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
        if node.value is None:
            return "NULL"
        type_str = str(node.type_).upper() if node.type_ else ""
        if "BOOLEAN" in type_str:
            return "TRUE" if node.value else "FALSE"
        if "STRING" in type_str:
            escaped = str(node.value).replace("'", "''")
            return f"'{escaped}'"
        if "INTEGER" in type_str or "FLOAT" in type_str or "NUMBER" in type_str:
            return str(node.value)
        # Fallback
        if isinstance(node.value, bool):
            return "TRUE" if node.value else "FALSE"
        if isinstance(node.value, str):
            escaped = node.value.replace("'", "''")
            return f"'{escaped}'"
        return str(node.value)

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

    def _get_dataset_structure(self, node: AST.AST) -> Optional[Dataset]:
        """Get dataset structure for a node, tracing to the source dataset."""
        if isinstance(node, AST.VarID):
            udo_val = self._get_udo_param(node.value)
            if udo_val is not None:
                if isinstance(udo_val, AST.AST):
                    return self._get_dataset_structure(udo_val)
                if isinstance(udo_val, str) and udo_val in self.available_tables:
                    return self.available_tables[udo_val]
            return self.available_tables.get(node.value)

        if isinstance(node, AST.RegularAggregation) and node.dataset:
            return self._get_dataset_structure(node.dataset)

        if isinstance(node, AST.BinOp):
            op = str(node.op).lower()
            if op in (tokens.MEMBERSHIP, "as"):
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
            return self._get_dataset_structure(node.operand)

        if isinstance(node, AST.JoinOp):
            return self._get_output_dataset()

        if isinstance(node, AST.MulOp) and node.children:
            return self._get_dataset_structure(node.children[0])

        if isinstance(node, AST.UDOCall):
            return self._get_output_dataset()

        if isinstance(node, AST.If):
            return self._get_dataset_structure(node.thenOp)

        if isinstance(node, AST.Case) and node.cases:
            return self._get_dataset_structure(node.cases[0].thenOp)

        return None

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
    # Top-level visitors
    # =========================================================================

    def transpile(self, node: AST.Start) -> List[Tuple[str, str, bool]]:
        """Transpile the AST to a list of (name, SQL query, is_persistent) tuples."""
        return self.visit(node)

    def visit_Start(self, node: AST.Start) -> List[Tuple[str, str, bool]]:
        """Process the entire script, generating SQL for each top-level assignment."""
        queries: List[Tuple[str, str, bool]] = []

        for child in node.children:
            if isinstance(child, AST.Operator):
                self.visit(child)
            elif isinstance(child, AST.DPRuleset):
                pass
            elif isinstance(child, AST.Assignment):
                name = child.left.value
                self.current_assignment = name
                self.inputs = self._get_assignment_inputs(name)

                is_persistent = isinstance(child, AST.PersistentAssignment)
                query = self.visit(child)
                queries.append((name, query, is_persistent))

        return queries

    def visit_Assignment(self, node: AST.Assignment) -> str:
        """Visit an assignment and return the SQL for its right-hand side."""
        return self.visit(node.right)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> str:
        """Visit a persistent assignment (same as regular for SQL generation)."""
        return self.visit(node.right)

    # =========================================================================
    # UDO definition and call
    # =========================================================================

    def visit_Operator(self, node: AST.Operator) -> None:
        """Register a UDO definition."""
        params_list: List[Dict[str, Any]] = []
        for p in node.parameters:
            params_list.append({"name": p.name, "type": p.type_, "default": p.default})

        self._udos[node.op] = {
            "params": params_list,
            "output": node.output_type,
            "expression": node.expression,
        }

    def visit_UDOCall(self, node: AST.UDOCall) -> str:
        """Visit a UDO call by expanding its definition with parameter bindings."""
        if node.op not in self._udos:
            raise ValueError(f"Unknown UDO: {node.op}")

        udo_def = self._udos[node.op]
        params = udo_def["params"]
        expression = deepcopy(udo_def["expression"])

        bindings: Dict[str, Any] = {}
        for i, param_info in enumerate(params):
            param_name = param_info["name"]
            if i < len(node.params):
                bindings[param_name] = node.params[i]

        self._push_udo_params(bindings)
        try:
            result = self.visit(expression)
        finally:
            self._pop_udo_params()

        return result

    # =========================================================================
    # Leaf visitors
    # =========================================================================

    def visit_VarID(self, node: AST.VarID) -> str:
        """Visit a variable identifier."""
        name = node.value
        udo_val = self._get_udo_param(name)
        if udo_val is not None:
            if isinstance(udo_val, AST.AST):
                return self.visit(udo_val)
            if isinstance(udo_val, str):
                return quote_identifier(udo_val)

        if name in self.scalars:
            sc = self.scalars[name]
            return self._to_sql_literal(sc.value, type(sc.data_type).__name__)

        if self._in_clause and self._current_dataset and name in self._current_dataset.components:
            return quote_identifier(name)

        if name in self.available_tables:
            return f"SELECT * FROM {quote_identifier(name)}"

        return quote_identifier(name)

    def visit_Constant(self, node: AST.Constant) -> str:
        """Visit a constant literal."""
        return self._constant_to_sql(node)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        """Visit a parameter constant."""
        return str(node.value)

    def visit_Identifier(self, node: AST.Identifier) -> str:
        """Visit an identifier node."""
        return quote_identifier(node.value)

    def visit_ID(self, node: AST.ID) -> str:
        """Visit an ID node (used for type names, placeholders like '_', etc.)."""
        if node.value == "_":
            # VTL underscore means "use default" - return None marker
            return ""
        return node.value

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        """Visit a parenthesized function/expression."""
        return self.visit(node.operand)

    def visit_Collection(self, node: AST.Collection) -> str:
        """Visit a Collection (Set or ValueDomain reference)."""
        if node.kind == "ValueDomain":
            return self._visit_value_domain(node)
        values = [self.visit(child) for child in node.children]
        return f"({', '.join(values)})"

    def _visit_value_domain(self, node: AST.Collection) -> str:
        """Resolve a ValueDomain reference to SQL literal list."""
        if not self.value_domains:
            raise ValueError(
                f"Value domain '{node.name}' referenced but no value domains provided."
            )
        if node.name not in self.value_domains:
            raise ValueError(f"Value domain '{node.name}' not found in provided value domains.")
        vd = self.value_domains[node.name]
        type_name = vd.type.__name__ if hasattr(vd.type, "__name__") else str(vd.type)
        literals = [self._to_sql_literal(v, type_name) for v in vd.setlist]
        return f"({', '.join(literals)})"

    # =========================================================================
    # Dataset-level binary operation helpers
    # =========================================================================

    def _build_ds_ds_binary(
        self,
        left_node: AST.AST,
        right_node: AST.AST,
        op: str,
    ) -> str:
        """Build SQL for dataset-dataset binary operation (requires JOIN)."""
        left_ds = self._get_dataset_structure(left_node)
        right_ds = self._get_dataset_structure(right_node)
        output_ds = self._get_output_dataset()

        if left_ds is None or right_ds is None:
            raise ValueError("Cannot resolve dataset structures for binary operation")

        left_src = self._get_dataset_sql(left_node)
        right_src = self._get_dataset_sql(right_node)

        alias_a = "a"
        alias_b = "b"

        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        common_ids = sorted(left_ids & right_ids)
        all_ids = sorted(left_ids | right_ids)

        output_measure_names = list(output_ds.get_measures_names()) if output_ds else []
        left_measures = left_ds.get_measures_names()
        right_measures = right_ds.get_measures_names()
        common_measures = [m for m in left_measures if m in right_measures]

        cols: List[str] = []
        for id_name in all_ids:
            if id_name in left_ids:
                cols.append(f"{alias_a}.{quote_identifier(id_name)}")
            else:
                cols.append(f"{alias_b}.{quote_identifier(id_name)}")

        for measure in common_measures:
            left_ref = f"{alias_a}.{quote_identifier(measure)}"
            right_ref = f"{alias_b}.{quote_identifier(measure)}"
            expr = registry.binary.generate(op, left_ref, right_ref)

            out_name = measure
            if (
                output_measure_names
                and len(common_measures) == 1
                and len(output_measure_names) == 1
            ):
                out_name = output_measure_names[0]
            cols.append(f"{expr} AS {quote_identifier(out_name)}")

        on_parts = [
            f"{alias_a}.{quote_identifier(id_)} = {alias_b}.{quote_identifier(id_)}"
            for id_ in common_ids
        ]
        on_clause = " AND ".join(on_parts)

        builder = SQLBuilder().select(*cols).from_table(left_src, alias_a)
        if on_clause:
            builder.join(right_src, alias_b, on=on_clause, join_type="INNER")
        else:
            builder.cross_join(right_src, alias_b)

        return builder.build()

    def _build_ds_scalar_binary(
        self,
        ds_node: AST.AST,
        scalar_node: AST.AST,
        op: str,
        ds_on_left: bool = True,
    ) -> str:
        """Build SQL for dataset-scalar binary operation."""
        ds = self._get_dataset_structure(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve dataset structure")

        scalar_sql = self.visit(scalar_node)
        table_src = self._get_dataset_sql(ds_node)
        output_ds = self._get_output_dataset()
        output_measure_names = list(output_ds.get_measures_names()) if output_ds else []

        cols: List[str] = []
        measures = ds.get_measures_names()

        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                col_ref = quote_identifier(name)
                if ds_on_left:
                    expr = registry.binary.generate(op, col_ref, scalar_sql)
                else:
                    expr = registry.binary.generate(op, scalar_sql, col_ref)

                out_name = name
                if output_measure_names and len(measures) == 1 and len(output_measure_names) == 1:
                    out_name = output_measure_names[0]
                cols.append(f"{expr} AS {quote_identifier(out_name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    # =========================================================================
    # Expression visitors
    # =========================================================================

    def visit_BinOp(self, node: AST.BinOp) -> str:
        """Visit a binary operation."""
        op = str(node.op).lower() if node.op else ""

        # Normalize 'not in' to 'not_in'
        if op == "not in":
            op = tokens.NOT_IN

        if op == tokens.MEMBERSHIP:
            return self._visit_membership(node)

        if op == tokens.EXISTS_IN:
            return self._build_exists_in_sql(node.left, node.right)

        if op == tokens.CHARSET_MATCH:
            return self._visit_match_characters(node)

        if op == tokens.TIMESHIFT:
            return self._visit_timeshift(node)

        if op == tokens.RANDOM:
            return self._visit_random_binop(node)

        # Check operand types for dataset-level routing
        left_type = self._get_operand_type(node.left)
        right_type = self._get_operand_type(node.right)
        has_dataset = left_type == _DATASET or right_type == _DATASET

        if has_dataset:
            return self._visit_dataset_binary(node.left, node.right, op)

        # Scalar-scalar: use registry
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)
        if registry.binary.is_registered(op):
            return registry.binary.generate(op, left_sql, right_sql)
        # Fallback for unregistered ops
        return f"{op.upper()}({left_sql}, {right_sql})"

    def _visit_dataset_binary(self, left: AST.AST, right: AST.AST, op: str) -> str:
        """Route to the correct dataset binary handler."""
        left_type = self._get_operand_type(left)
        right_type = self._get_operand_type(right)

        if left_type == _DATASET and right_type == _DATASET:
            return self._build_ds_ds_binary(left, right, op)
        elif left_type == _DATASET:
            return self._build_ds_scalar_binary(left, right, op, ds_on_left=True)
        else:
            return self._build_ds_scalar_binary(right, left, op, ds_on_left=False)

    def _visit_membership(self, node: AST.BinOp) -> str:
        """Visit MEMBERSHIP (#): DS#comp -> SELECT ids, comp FROM DS."""
        comp_name = node.right.value if hasattr(node.right, "value") else str(node.right)
        udo_val = self._get_udo_param(comp_name)
        if udo_val is not None:
            if isinstance(udo_val, (AST.VarID, AST.Identifier)):
                comp_name = udo_val.value
            elif isinstance(udo_val, str):
                comp_name = udo_val

        ds = self._get_dataset_structure(node.left)
        table_src = self._get_dataset_sql(node.left)

        if ds is None:
            ds_name = self._resolve_dataset_name(node.left)
            return f"SELECT {quote_identifier(comp_name)} FROM {quote_identifier(ds_name)}"

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
        cols.append(quote_identifier(comp_name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_match_characters(self, node: AST.BinOp) -> str:
        """Visit match_characters operator using registry."""
        left_type = self._get_operand_type(node.left)
        pattern_sql = self.visit(node.right)

        if left_type == _DATASET:
            ds = self._get_dataset_structure(node.left)
            if ds is None:
                raise ValueError("Cannot resolve dataset for match_characters")

            table_src = self._get_dataset_sql(node.left)
            cols: List[str] = []
            for name, comp in ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    cols.append(quote_identifier(name))
                elif comp.role == Role.MEASURE:
                    expr = registry.binary.generate(
                        tokens.CHARSET_MATCH, quote_identifier(name), pattern_sql
                    )
                    cols.append(f"{expr} AS {quote_identifier(name)}")
            return SQLBuilder().select(*cols).from_table(table_src).build()
        else:
            left_sql = self.visit(node.left)
            return registry.binary.generate(tokens.CHARSET_MATCH, left_sql, pattern_sql)

    def _build_exists_in_sql(
        self,
        left_node: AST.AST,
        right_node: AST.AST,
    ) -> str:
        """Build SQL for exists_in operation."""
        left_ds = self._get_dataset_structure(left_node)
        right_ds = self._get_dataset_structure(right_node)

        if left_ds is None or right_ds is None:
            raise ValueError("Cannot resolve structures for exists_in")

        left_src = self._get_dataset_sql(left_node)
        right_src = self._get_dataset_sql(right_node)

        left_ids = left_ds.get_identifiers_names()
        right_ids = right_ds.get_identifiers_names()
        common_ids = [id_ for id_ in left_ids if id_ in right_ids]

        where_parts = [
            f"l.{quote_identifier(id_)} = r.{quote_identifier(id_)}" for id_ in common_ids
        ]
        where_clause = " AND ".join(where_parts)

        id_cols = ", ".join([f"l.{quote_identifier(id_)}" for id_ in left_ids])

        # Use subquery for right side, wrapping in SELECT * FROM if needed
        right_subq = right_src
        if not right_src.strip().upper().startswith("("):
            right_subq = f"(SELECT * FROM {right_src})"

        exists_subq = f"EXISTS(SELECT 1 FROM {right_subq} AS r WHERE {where_clause})"

        # Wrap left side similarly
        left_subq = left_src
        if not left_src.strip().upper().startswith("("):
            left_subq = f"(SELECT * FROM {left_src})"

        return f'SELECT {id_cols}, {exists_subq} AS "bool_var" FROM {left_subq} AS l'

    def _visit_timeshift(self, node: AST.BinOp) -> str:
        """Visit TIMESHIFT: shift time identifiers by n periods."""
        if not self._is_dataset(node.left):
            left_sql = self.visit(node.left)
            right_sql = self.visit(node.right)
            return f"vtl_period_shift({left_sql}, {right_sql})"

        ds = self._get_dataset_structure(node.left)
        if ds is None:
            raise ValueError("Cannot resolve dataset for timeshift")

        table_src = self._get_dataset_sql(node.left)
        shift_sql = self.visit(node.right)
        time_id, _ = self._split_time_identifier(ds)

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                if name == time_id:
                    cols.append(
                        f"vtl_period_shift({quote_identifier(name)}, {shift_sql})"
                        f" AS {quote_identifier(name)}"
                    )
                else:
                    cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                cols.append(quote_identifier(name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Visit a unary operation."""
        op = str(node.op).lower()

        if op == tokens.ISNULL:
            return self._visit_isnull(node)

        if op == tokens.PERIOD_INDICATOR:
            operand_sql = self.visit(node.operand)
            return f"vtl_period_indicator(vtl_period_parse({operand_sql}))"

        if op == tokens.FLOW_TO_STOCK:
            return self._visit_flow_to_stock(node)

        if op == tokens.STOCK_TO_FLOW:
            return self._visit_stock_to_flow(node)

        if op in (tokens.DAYTOYEAR, tokens.DAYTOMONTH, tokens.YEARTODAY, tokens.MONTHTODAY):
            return self._visit_duration_conversion(node, op)

        if op == tokens.FILL_TIME_SERIES:
            return self._visit_fill_time_series(node)

        operand_type = self._get_operand_type(node.operand)

        if operand_type == _DATASET:
            return self._visit_unary_dataset(node, op)
        else:
            operand_sql = self.visit(node.operand)
            if registry.unary.is_registered(op):
                return registry.unary.generate(op, operand_sql)
            if op == tokens.PLUS:
                return f"+{operand_sql}"
            if op == tokens.MINUS:
                return f"-{operand_sql}"
            return f"{op.upper()}({operand_sql})"

    def _visit_unary_dataset(self, node: AST.UnaryOp, op: str) -> str:
        """Visit a dataset-level unary operation."""
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            raise ValueError(f"Cannot resolve dataset for unary op '{op}'")

        table_src = self._get_dataset_sql(node.operand)
        output_ds = self._get_output_dataset()
        output_measure_names = list(output_ds.get_measures_names()) if output_ds else []

        if op == tokens.NOT:
            cols: List[str] = []
            for name, comp in ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    cols.append(quote_identifier(name))
                elif comp.role == Role.MEASURE:
                    expr = registry.unary.generate(tokens.NOT, quote_identifier(name))
                    cols.append(f"{expr} AS {quote_identifier(name)}")
            return SQLBuilder().select(*cols).from_table(table_src).build()

        cols = []
        measures = ds.get_measures_names()
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                if registry.unary.is_registered(op):
                    expr = registry.unary.generate(op, quote_identifier(name))
                else:
                    expr = f"{op.upper()}({quote_identifier(name)})"
                out_name = name
                if output_measure_names and len(measures) == 1 and len(output_measure_names) == 1:
                    out_name = output_measure_names[0]
                cols.append(f"{expr} AS {quote_identifier(out_name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_isnull(self, node: AST.UnaryOp) -> str:
        """Visit ISNULL operation."""
        operand_type = self._get_operand_type(node.operand)

        if operand_type == _DATASET:
            ds = self._get_dataset_structure(node.operand)
            if ds is None:
                raise ValueError("Cannot resolve dataset for isnull")

            table_src = self._get_dataset_sql(node.operand)
            measures = ds.get_measures_names()

            # isnull on mono-measure dataset always produces bool_var
            output_measure_names: List[str] = []
            if len(measures) == 1:
                output_measure_names = ["bool_var"]

            cols: List[str] = []
            for name, comp in ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    cols.append(quote_identifier(name))
                elif comp.role == Role.MEASURE:
                    out_name = name
                    if (
                        output_measure_names
                        and len(measures) == 1
                        and len(output_measure_names) == 1
                    ):
                        out_name = output_measure_names[0]
                    cols.append(
                        f"{registry.unary.generate(tokens.ISNULL, quote_identifier(name))}"
                        f" AS {quote_identifier(out_name)}"
                    )
            return SQLBuilder().select(*cols).from_table(table_src).build()
        else:
            operand_sql = self.visit(node.operand)
            return registry.unary.generate(tokens.ISNULL, operand_sql)

    def _visit_flow_to_stock(self, node: AST.UnaryOp) -> str:
        """Visit flow_to_stock: cumulative sum over time."""
        if not self._is_dataset(node.operand):
            raise ValueError("flow_to_stock requires a dataset operand")

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            raise ValueError("Cannot resolve dataset for flow_to_stock")

        table_src = self._get_dataset_sql(node.operand)
        time_id, other_ids = self._split_time_identifier(ds)

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                partition = ", ".join(quote_identifier(i) for i in other_ids)
                partition_clause = f"PARTITION BY {partition} " if partition else ""
                order_clause = f"ORDER BY {quote_identifier(time_id)}"
                expr = f"SUM({quote_identifier(name)}) OVER ({partition_clause}{order_clause})"
                cols.append(f"{expr} AS {quote_identifier(name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_stock_to_flow(self, node: AST.UnaryOp) -> str:
        """Visit stock_to_flow: current - lag(current) over time."""
        if not self._is_dataset(node.operand):
            raise ValueError("stock_to_flow requires a dataset operand")

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            raise ValueError("Cannot resolve dataset for stock_to_flow")

        table_src = self._get_dataset_sql(node.operand)
        time_id, other_ids = self._split_time_identifier(ds)

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                partition = ", ".join(quote_identifier(i) for i in other_ids)
                partition_clause = f"PARTITION BY {partition} " if partition else ""
                order_clause = f"ORDER BY {quote_identifier(time_id)}"
                lag_expr = f"LAG({quote_identifier(name)}) OVER ({partition_clause}{order_clause})"
                expr = f"COALESCE({quote_identifier(name)} - {lag_expr}, {quote_identifier(name)})"
                cols.append(f"{expr} AS {quote_identifier(name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

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

    def _visit_duration_conversion(self, node: AST.UnaryOp, op: str) -> str:
        """Visit duration conversion operators."""
        operand_sql = self.visit(node.operand)

        if op == tokens.DAYTOYEAR:
            return (
                f"'P' || CAST(FLOOR({operand_sql} / 365) AS VARCHAR) || 'Y' || "
                f"CAST({operand_sql} % 365 AS VARCHAR) || 'D'"
            )
        elif op == tokens.DAYTOMONTH:
            return (
                f"'P' || CAST(FLOOR({operand_sql} / 30) AS VARCHAR) || 'M' || "
                f"CAST({operand_sql} % 30 AS VARCHAR) || 'D'"
            )
        elif op == tokens.YEARTODAY:
            return (
                f"( CAST(REGEXP_EXTRACT({operand_sql}, 'P(\\d+)Y', 1) AS INTEGER) * 365"
                f" + CAST(REGEXP_EXTRACT({operand_sql}, '(\\d+)D', 1) AS INTEGER) )"
            )
        elif op == tokens.MONTHTODAY:
            return (
                f"( CAST(REGEXP_EXTRACT({operand_sql}, 'P(\\d+)M', 1) AS INTEGER) * 30"
                f" + CAST(REGEXP_EXTRACT({operand_sql}, '(\\d+)D', 1) AS INTEGER) )"
            )
        else:
            raise ValueError(f"Unknown duration conversion: {op}")

    def _visit_fill_time_series(self, node: AST.UnaryOp) -> str:
        """Visit fill_time_series operation."""
        if not self._is_dataset(node.operand):
            operand_sql = self.visit(node.operand)
            return f"fill_time_series({operand_sql})"

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            raise ValueError("Cannot resolve dataset for fill_time_series")

        table_src = self._get_dataset_sql(node.operand)
        return f"SELECT * FROM fill_time_series({table_src})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        """Visit a parameterized operation."""
        op = str(node.op).lower()

        if op == tokens.CAST:
            return self._visit_cast(node)

        if op == tokens.RANDOM:
            return self._visit_random(node)

        operand_type = self._get_operand_type(node.children[0]) if node.children else _SCALAR

        if operand_type == _DATASET:
            return self._visit_paramop_dataset(node, op)
        else:
            children_sql = [self.visit(c) for c in node.children]
            params_sql = self._visit_params(node.params)
            # Default precision for ROUND/TRUNC when no parameter given
            if op in (tokens.ROUND, tokens.TRUNC) and not params_sql:
                params_sql = ["0"]
            all_args = children_sql + params_sql
            if registry.parameterized.is_registered(op):
                return registry.parameterized.generate(op, *all_args)
            non_none = [a for a in all_args if a is not None]
            return f"{op.upper()}({', '.join(non_none)})"

    def _visit_params(self, params: List[Any]) -> List[Optional[str]]:
        """Visit param nodes, converting VTL defaults ('_', null) to None."""
        result: List[Optional[str]] = []
        for p in params:
            if (
                p is None
                or (isinstance(p, AST.ID) and p.value == "_")
                or (isinstance(p, AST.Constant) and p.value is None)
            ):
                result.append(None)
            else:
                result.append(self.visit(p))
        return result

    def _visit_paramop_dataset(self, node: AST.ParamOp, op: str) -> str:
        """Visit a dataset-level parameterized operation."""
        ds_node = node.children[0]
        ds = self._get_dataset_structure(ds_node)
        if ds is None:
            raise ValueError(f"Cannot resolve dataset for parameterized op '{op}'")

        table_src = self._get_dataset_sql(ds_node)
        params_sql = self._visit_params(node.params)

        # Default precision for ROUND/TRUNC when no parameter given
        if op in (tokens.ROUND, tokens.TRUNC) and not params_sql:
            params_sql = ["0"]

        cols: List[str] = []
        output_ds = self._get_output_dataset()
        output_measure_names = list(output_ds.get_measures_names()) if output_ds else []
        input_measures = ds.get_measures_names()

        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                col_ref = quote_identifier(name)
                if registry.parameterized.is_registered(op):
                    expr = registry.parameterized.generate(op, col_ref, *params_sql)
                else:
                    all_args = [col_ref] + [a for a in params_sql if a is not None]
                    expr = f"{op.upper()}({', '.join(all_args)})"
                out_name = name
                if (
                    output_measure_names
                    and len(input_measures) == 1
                    and len(output_measure_names) == 1
                ):
                    out_name = output_measure_names[0]
                cols.append(f"{expr} AS {quote_identifier(out_name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_cast(self, node: AST.ParamOp) -> str:
        """Visit CAST operation."""
        if not node.children:
            raise ValueError("CAST requires at least one operand")

        operand = node.children[0]
        target_type_str = ""
        if len(node.children) >= 2:
            type_node = node.children[1]
            target_type_str = type_node.value if hasattr(type_node, "value") else str(type_node)

        duckdb_type = get_duckdb_type(target_type_str)

        mask: Optional[str] = None
        if node.params:
            mask_node = node.params[0]
            if hasattr(mask_node, "value"):
                mask = mask_node.value

        operand_type = self._get_operand_type(operand)

        if operand_type == _DATASET:
            return self._visit_cast_dataset(operand, duckdb_type, target_type_str, mask)
        else:
            operand_sql = self.visit(operand)
            return self._cast_expr(operand_sql, duckdb_type, target_type_str, mask)

    def _cast_expr(
        self, expr: str, duckdb_type: str, target_type_str: str, mask: Optional[str]
    ) -> str:
        """Generate a CAST expression for a single value."""
        if mask and target_type_str == "Date":
            return f"STRPTIME({expr}, '{mask}')::DATE"
        return f"CAST({expr} AS {duckdb_type})"

    def _visit_cast_dataset(
        self,
        ds_node: AST.AST,
        duckdb_type: str,
        target_type_str: str,
        mask: Optional[str],
    ) -> str:
        """Visit dataset-level CAST."""
        ds = self._get_dataset_structure(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve dataset for CAST")

        table_src = self._get_dataset_sql(ds_node)
        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                expr = self._cast_expr(quote_identifier(name), duckdb_type, target_type_str, mask)
                cols.append(f"{expr} AS {quote_identifier(name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_random(self, node: AST.ParamOp) -> str:
        """Visit RANDOM operator (ParamOp form): deterministic hash-based random."""
        seed_node = node.children[0] if node.children else None
        index_node = node.params[0] if node.params else None

        seed_type = self._get_operand_type(seed_node) if seed_node else _SCALAR

        if seed_type == _DATASET:
            return self._visit_random_dataset(node)

        seed_sql = self.visit(seed_node) if seed_node else "0"
        index_sql = self.visit(index_node) if index_node else "0"

        return self._random_hash_expr(seed_sql, index_sql)

    def _visit_random_binop(self, node: AST.BinOp) -> str:
        """Visit RANDOM operator (BinOp form, e.g. inside calc)."""
        seed_node = node.left
        index_node = node.right

        seed_type = self._get_operand_type(seed_node) if seed_node else _SCALAR

        if seed_type == _DATASET:
            # Convert to dataset-level random; reuse the ParamOp handler logic
            ds = self._get_dataset_structure(seed_node)
            if ds is None:
                raise ValueError("Cannot resolve dataset for RANDOM")

            table_src = self._get_dataset_sql(seed_node)
            index_sql = self.visit(index_node) if index_node else "0"

            cols: List[str] = []
            for name, comp in ds.components.items():
                if comp.role == Role.IDENTIFIER:
                    cols.append(quote_identifier(name))
                elif comp.role == Role.MEASURE:
                    expr = self._random_hash_expr(quote_identifier(name), index_sql)
                    cols.append(f"{expr} AS {quote_identifier(name)}")

            return SQLBuilder().select(*cols).from_table(table_src).build()

        seed_sql = self.visit(seed_node) if seed_node else "0"
        index_sql = self.visit(index_node) if index_node else "0"

        return self._random_hash_expr(seed_sql, index_sql)

    @staticmethod
    def _random_hash_expr(seed_sql: str, index_sql: str) -> str:
        """Build a deterministic hash-based random expression in [0, 1)."""
        return (
            f"(ABS(hash(CAST({seed_sql} AS VARCHAR) || '_' || "
            f"CAST({index_sql} AS VARCHAR))) % 1000000) / 1000000.0"
        )

    def _visit_random_dataset(self, node: AST.ParamOp) -> str:
        """Visit dataset-level RANDOM."""
        ds_node = node.children[0]
        ds = self._get_dataset_structure(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve dataset for RANDOM")

        table_src = self._get_dataset_sql(ds_node)
        index_sql = self.visit(node.params[0]) if node.params else "0"

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                expr = (
                    f"(ABS(hash(CAST({quote_identifier(name)} AS VARCHAR) || '_' || "
                    f"CAST({index_sql} AS VARCHAR))) % 1000000) / 1000000.0"
                )
                cols.append(f"{expr} AS {quote_identifier(name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    # =========================================================================
    # Clause visitor (RegularAggregation)
    # =========================================================================

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:
        """Visit clause operations: filter, calc, keep, drop, rename, subspace, aggr."""
        op = str(node.op).lower()

        if op == tokens.FILTER:
            return self._visit_filter(node)
        elif op == tokens.CALC:
            return self._visit_calc(node)
        elif op == tokens.KEEP:
            return self._visit_keep(node)
        elif op == tokens.DROP:
            return self._visit_drop(node)
        elif op == tokens.RENAME:
            return self._visit_rename(node)
        elif op == tokens.SUBSPACE:
            return self._visit_subspace(node)
        elif op == tokens.AGGREGATE:
            return self._visit_clause_aggregate(node)
        elif op == tokens.APPLY:
            return self._visit_apply(node)
        else:
            if node.dataset:
                return self.visit(node.dataset)
            return ""

    def _visit_filter(self, node: AST.RegularAggregation) -> str:
        """Visit filter clause: DS[filter condition]."""
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds:
            self._in_clause = True
            self._current_dataset = ds

        conditions = []
        for child in node.children:
            cond_sql = self.visit(child)
            conditions.append(cond_sql)

        if ds:
            self._in_clause = False
            self._current_dataset = None

        where_clause = " AND ".join(conditions) if conditions else ""

        builder = SQLBuilder().select_all().from_table(table_src)
        if where_clause:
            builder.where(where_clause)
        return builder.build()

    def _visit_calc(self, node: AST.RegularAggregation) -> str:
        """Visit calc clause: DS[calc new_col := expr, ...]."""
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            return f"SELECT * FROM {table_src}"

        self._in_clause = True
        self._current_dataset = ds

        original_cols: List[str] = []
        for name in ds.components:
            original_cols.append(quote_identifier(name))

        calc_exprs: Dict[str, str] = {}
        for child in node.children:
            assignment = child
            if (
                isinstance(child, AST.UnaryOp)
                and hasattr(child, "operand")
                and isinstance(child.operand, AST.Assignment)
            ):
                assignment = child.operand

            if isinstance(assignment, AST.Assignment):
                col_name = assignment.left.value if hasattr(assignment.left, "value") else ""
                expr_sql = self.visit(assignment.right)
                calc_exprs[col_name] = expr_sql

        self._in_clause = False
        self._current_dataset = None

        # Build SELECT: keep original columns that are NOT being overwritten,
        # then add the calc expressions (possibly replacing originals).
        select_cols: List[str] = []
        for name in ds.components:
            if name in calc_exprs:
                # Overwrite: use calc expression instead of the original column
                select_cols.append(f"{calc_exprs[name]} AS {quote_identifier(name)}")
            else:
                select_cols.append(quote_identifier(name))

        # Add any new columns (not in original dataset)
        for col_name, expr_sql in calc_exprs.items():
            if col_name not in ds.components:
                select_cols.append(f"{expr_sql} AS {quote_identifier(col_name)}")

        # Wrap inner query as subquery: if it's already a SELECT, wrap in parens;
        # if it's a table name, use SELECT * FROM name
        if table_src.strip().upper().startswith("SELECT"):
            inner_src = f"({table_src})"
        else:
            inner_src = f"(SELECT * FROM {table_src})"

        return SQLBuilder().select(*select_cols).from_table(inner_src, "t").build()

    def _visit_keep(self, node: AST.RegularAggregation) -> str:
        """Visit keep clause."""
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            return f"SELECT * FROM {table_src}"

        # Identifiers are always kept (use output dataset if available for
        # correct names after preceding renames in a chain).
        output_ds = self._get_output_dataset()
        if output_ds is not None:
            id_names: List[str] = list(output_ds.get_identifiers_names())
        else:
            id_names = [
                name for name, comp in ds.components.items() if comp.role == Role.IDENTIFIER
            ]

        keep_names: List[str] = list(id_names)
        for child in node.children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                keep_names.append(child.value)

        cols = [quote_identifier(name) for name in keep_names]

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_drop(self, node: AST.RegularAggregation) -> str:
        """Visit drop clause.

        Uses DuckDB's ``SELECT * EXCLUDE (...)`` to avoid relying on column
        names that may have been changed by preceding clauses in a chain.
        """
        if not node.dataset:
            return ""

        table_src = self._get_dataset_sql(node.dataset)

        drop_names: List[str] = []
        for child in node.children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                drop_names.append(quote_identifier(child.value))

        if not drop_names:
            return f"SELECT * FROM {table_src}"

        exclude = ", ".join(drop_names)
        builder = SQLBuilder()
        builder.select(f"* EXCLUDE ({exclude})")
        builder.from_table(table_src)
        return builder.build()

    def _visit_rename(self, node: AST.RegularAggregation) -> str:
        """Visit rename clause."""
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            return f"SELECT * FROM {table_src}"

        renames: Dict[str, str] = {}
        for child in node.children:
            if isinstance(child, AST.RenameNode):
                renames[child.old_name] = child.new_name

        cols: List[str] = []
        for name in ds.components:
            if name in renames:
                cols.append(f"{quote_identifier(name)} AS {quote_identifier(renames[name])}")
            else:
                cols.append(quote_identifier(name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_subspace(self, node: AST.RegularAggregation) -> str:
        """Visit subspace clause."""
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            return f"SELECT * FROM {table_src}"

        where_parts: List[str] = []
        remove_ids: set = set()
        for child in node.children:
            if isinstance(child, AST.BinOp):
                col_name = child.left.value if hasattr(child.left, "value") else ""
                remove_ids.add(col_name)
                val_sql = self.visit(child.right)
                where_parts.append(f"{quote_identifier(col_name)} = {val_sql}")

        cols = [quote_identifier(name) for name in ds.components if name not in remove_ids]

        builder = SQLBuilder().select(*cols).from_table(table_src)
        for wp in where_parts:
            builder.where(wp)
        return builder.build()

    def _visit_clause_aggregate(self, node: AST.RegularAggregation) -> str:
        """Visit aggregate clause: DS[aggr Me := sum(Me) group by Id, ... having ...]."""
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            return f"SELECT * FROM {table_src}"

        self._in_clause = True
        self._current_dataset = ds

        calc_exprs: Dict[str, str] = {}
        having_sql: Optional[str] = None

        for child in node.children:
            assignment = child
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                col_name = assignment.left.value if hasattr(assignment.left, "value") else ""
                # Check for having clause on the Aggregation node
                agg_node = assignment.right
                if isinstance(agg_node, AST.Aggregation) and agg_node.having_clause is not None:
                    hc = agg_node.having_clause
                    # having_clause is a ParamOp(op=having) with params = condition BinOp
                    if isinstance(hc, AST.ParamOp) and hc.params is not None:
                        having_sql = self.visit(hc.params)

                expr_sql = self.visit(agg_node)
                calc_exprs[col_name] = expr_sql

        self._in_clause = False
        self._current_dataset = None

        output_ds = self._get_output_dataset()
        group_ids = output_ds.get_identifiers_names() if output_ds else ds.get_identifiers_names()

        cols: List[str] = [quote_identifier(id_) for id_ in group_ids]
        for col_name, expr_sql in calc_exprs.items():
            cols.append(f"{expr_sql} AS {quote_identifier(col_name)}")

        builder = SQLBuilder().select(*cols).from_table(table_src)
        if group_ids:
            builder.group_by(*[quote_identifier(id_) for id_ in group_ids])

        if having_sql:
            builder.having(having_sql)

        return builder.build()

    def _visit_apply(self, node: AST.RegularAggregation) -> str:
        """Visit apply clause."""
        if node.dataset:
            return self.visit(node.dataset)
        return ""

    # =========================================================================
    # Aggregation visitor
    # =========================================================================

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

    def visit_Aggregation(self, node: AST.Aggregation) -> str:
        """Visit a standalone aggregation: sum(DS group by Id)."""
        op = str(node.op).lower()

        # Component-level aggregation in clause context
        if self._in_clause and node.operand:
            operand_type = self._get_operand_type(node.operand)
            if operand_type in (_COMPONENT, _SCALAR):
                operand_sql = self.visit(node.operand)
                if registry.aggregate.is_registered(op):
                    return registry.aggregate.generate(op, operand_sql)
                return f"{op.upper()}({operand_sql})"

        if node.operand is None:
            return ""

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            operand_sql = self.visit(node.operand)
            if registry.aggregate.is_registered(op):
                return registry.aggregate.generate(op, operand_sql)
            return f"{op.upper()}({operand_sql})"

        table_src = self._get_dataset_sql(node.operand)

        # Use the output dataset structure when available, as it reflects
        # renames and other clause transformations applied to the operand.
        output_ds = self._get_output_dataset()
        effective_ds = output_ds if output_ds is not None else ds

        all_ids = effective_ds.get_identifiers_names()
        group_cols = self._resolve_group_cols(node, all_ids)

        cols: List[str] = [quote_identifier(g) for g in group_cols]

        # count replaces all measures with a single int_var column.
        # Use COUNT(*) since we don't need specific column references.
        if op == tokens.COUNT:
            output_measures = effective_ds.get_measures_names()
            alias = output_measures[0] if output_measures else "int_var"
            cols.append(f"COUNT(*) AS {quote_identifier(alias)}")
        else:
            measures = effective_ds.get_measures_names()
            for measure in measures:
                if registry.aggregate.is_registered(op):
                    expr = registry.aggregate.generate(op, quote_identifier(measure))
                else:
                    expr = f"{op.upper()}({quote_identifier(measure)})"
                cols.append(f"{expr} AS {quote_identifier(measure)}")

        builder = SQLBuilder().select(*cols).from_table(table_src)

        if group_cols:
            builder.group_by(*[quote_identifier(g) for g in group_cols])

        if node.having_clause:
            self._in_clause = True
            self._current_dataset = ds
            having_sql = self.visit(node.having_clause)
            self._in_clause = False
            self._current_dataset = None
            builder.having(having_sql)

        return builder.build()

    # =========================================================================
    # Analytic visitor
    # =========================================================================

    def _build_over_clause(self, node: AST.Analytic) -> str:
        """Build the OVER (...) clause for an analytic function."""
        over_parts: List[str] = []
        if node.partition_by:
            partition_cols = ", ".join(quote_identifier(p) for p in node.partition_by)
            over_parts.append(f"PARTITION BY {partition_cols}")
        if node.order_by:
            order_cols = ", ".join(
                f"{quote_identifier(o.component)} {o.order}" for o in node.order_by
            )
            over_parts.append(f"ORDER BY {order_cols}")
        if node.window:
            window_sql = self.visit_Windowing(node.window)
            over_parts.append(window_sql)
        return " ".join(over_parts)

    def _build_analytic_expr(self, op: str, operand_sql: str, node: AST.Analytic) -> str:
        """Build the analytic function expression (without OVER)."""
        if op == tokens.RANK:
            return "RANK()"
        if op in (tokens.LAG, tokens.LEAD) and node.params:
            offset = node.params[0] if node.params else 1
            default_val = node.params[1] if len(node.params) > 1 else None
            func_sql = f"{op.upper()}({operand_sql}, {offset}"
            if default_val is not None:
                func_sql += f", {default_val}"
            return func_sql + ")"
        if registry.analytic.is_registered(op):
            return registry.analytic.generate(op, operand_sql)
        return f"{op.upper()}({operand_sql})"

    def visit_Analytic(self, node: AST.Analytic) -> str:
        """Visit an analytic (window) function."""
        op = str(node.op).lower()

        # Check if operand is a dataset — needs dataset-level handling
        if node.operand and self._get_operand_type(node.operand) == _DATASET:
            return self._visit_analytic_dataset(node, op)

        # Component-level: single expression with OVER
        operand_sql = self.visit(node.operand) if node.operand else ""
        func_sql = self._build_analytic_expr(op, operand_sql, node)
        over_clause = self._build_over_clause(node)
        return f"{func_sql} OVER ({over_clause})"

    def _visit_analytic_dataset(self, node: AST.Analytic, op: str) -> str:
        """Visit a dataset-level analytic: applies the window function to each measure."""
        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            raise ValueError(f"Cannot resolve dataset for analytic '{op}'")

        table_src = self._get_dataset_sql(node.operand)
        over_clause = self._build_over_clause(node)

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                func_sql = self._build_analytic_expr(op, quote_identifier(name), node)
                cols.append(f"{func_sql} OVER ({over_clause}) AS {quote_identifier(name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def visit_Windowing(self, node: AST.Windowing) -> str:
        """Visit a windowing specification."""
        type_str = str(node.type_).upper() if node.type_ else "ROWS"
        # Map VTL types to SQL: DATA POINTS → ROWS
        if "DATA" in type_str:
            type_str = "ROWS"
        elif "RANGE" in type_str:
            type_str = "RANGE"

        def bound_str(value: Union[int, str], mode: str) -> str:
            if mode.upper() == "CURRENT ROW" or str(value).upper() == "CURRENT ROW":
                return "CURRENT ROW"
            if str(value).upper() == "UNBOUNDED":
                return f"UNBOUNDED {mode.upper()}"
            return f"{value} {mode.upper()}"

        start = bound_str(node.start, node.start_mode)
        stop = bound_str(node.stop, node.stop_mode)

        return f"{type_str} BETWEEN {start} AND {stop}"

    # =========================================================================
    # MulOp visitor (set ops, between, exists_in, current_date)
    # =========================================================================

    def visit_MulOp(self, node: AST.MulOp) -> str:
        """Visit a multi-operand operation."""
        op = str(node.op).lower()

        if op == tokens.CURRENT_DATE:
            return "CURRENT_DATE"

        if op == tokens.BETWEEN:
            return self._visit_between(node)

        if op == tokens.EXISTS_IN:
            return self._visit_exists_in_mul(node)

        if op in (tokens.UNION, tokens.INTERSECT, tokens.SETDIFF, tokens.SYMDIFF):
            return self._visit_set_operation(node, op)

        child_sqls = [self.visit(c) for c in node.children]
        return ", ".join(child_sqls)

    @staticmethod
    def _between_expr(operand: str, low: str, high: str) -> str:
        """Build a VTL-compliant BETWEEN expression with NULL propagation.

        VTL requires that if ANY operand of between is NULL, the result is NULL.
        SQL's three-valued logic differs: FALSE AND NULL = FALSE.  To match VTL
        semantics we wrap the expression with an explicit NULL check.
        """
        return (
            f"CASE WHEN {operand} IS NULL OR {low} IS NULL OR {high} IS NULL "
            f"THEN NULL ELSE ({operand} BETWEEN {low} AND {high}) END"
        )

    def _visit_between(self, node: AST.MulOp) -> str:
        """Visit BETWEEN: expr BETWEEN low AND high. Handles dataset operand."""
        if len(node.children) < 3:
            raise ValueError("BETWEEN requires 3 operands")

        operand_type = self._get_operand_type(node.children[0])

        if operand_type == _DATASET:
            return self._visit_between_dataset(node)

        operand_sql = self.visit(node.children[0])
        low_sql = self.visit(node.children[1])
        high_sql = self.visit(node.children[2])
        return self._between_expr(operand_sql, low_sql, high_sql)

    def _visit_between_dataset(self, node: AST.MulOp) -> str:
        """Visit dataset-level BETWEEN: applies BETWEEN to each measure."""
        ds_node = node.children[0]
        ds = self._get_dataset_structure(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve dataset for BETWEEN")

        table_src = self._get_dataset_sql(ds_node)
        low_sql = self.visit(node.children[1])
        high_sql = self.visit(node.children[2])

        output_ds = self._get_output_dataset()
        output_measure_names = list(output_ds.get_measures_names()) if output_ds else []

        cols: List[str] = []
        measures = ds.get_measures_names()
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                expr = self._between_expr(quote_identifier(name), low_sql, high_sql)
                out_name = name
                if output_measure_names and len(measures) == 1 and len(output_measure_names) == 1:
                    out_name = output_measure_names[0]
                cols.append(f"{expr} AS {quote_identifier(out_name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_exists_in_mul(self, node: AST.MulOp) -> str:
        """Visit EXISTS_IN in MulOp form, handling the optional retain parameter."""
        if len(node.children) < 2:
            raise ValueError("exists_in requires at least 2 operands")

        base_sql = self._build_exists_in_sql(node.children[0], node.children[1])

        # Check for retain parameter (true / false / all)
        if len(node.children) >= 3:
            retain_node = node.children[2]
            if isinstance(retain_node, AST.Constant) and retain_node.value is True:
                return (
                    f'SELECT * FROM ({base_sql}) AS _ei WHERE "bool_var" = TRUE'
                )
            if isinstance(retain_node, AST.Constant) and retain_node.value is False:
                return (
                    f'SELECT * FROM ({base_sql}) AS _ei WHERE "bool_var" = FALSE'
                )
            # "all" or any other value → return all rows (default behaviour)

        return base_sql

    def _visit_set_operation(self, node: AST.MulOp, op: str) -> str:
        """Visit set operations: UNION, INTERSECT, SETDIFF, SYMDIFF."""
        child_sqls = []
        for child in node.children:
            child_sql = self.visit(child)
            if not child_sql.strip().upper().startswith("SELECT"):
                child_sql = (
                    f"SELECT * FROM "
                    f"{quote_identifier(child.value if hasattr(child, 'value') else child_sql)}"
                )
            child_sqls.append(child_sql)

        if op == tokens.UNION:
            first_child = node.children[0]
            ds = self._get_dataset_structure(first_child)
            if ds:
                id_names = ds.get_identifiers_names()
                if id_names:
                    inner_sql = registry.set_ops.generate(op, *child_sqls)
                    id_cols = ", ".join(quote_identifier(i) for i in id_names)
                    return f"SELECT DISTINCT ON ({id_cols}) * FROM ({inner_sql}) AS _union_t"
            return registry.set_ops.generate(op, *child_sqls)

        if op == tokens.SYMDIFF and len(child_sqls) >= 2:
            a = child_sqls[0]
            b = child_sqls[1]
            return f"(({a}) EXCEPT ({b})) UNION ALL (({b}) EXCEPT ({a}))"

        return registry.set_ops.generate(op, *child_sqls)

    # =========================================================================
    # Conditional visitors (If, Case)
    # =========================================================================

    def visit_If(self, node: AST.If) -> str:
        """Visit IF-THEN-ELSE."""
        cond_sql = self.visit(node.condition)
        then_sql = self.visit(node.thenOp)
        else_sql = self.visit(node.elseOp)
        return f"CASE WHEN {cond_sql} THEN {then_sql} ELSE {else_sql} END"

    def visit_Case(self, node: AST.Case) -> str:
        """Visit CASE expression."""
        parts = ["CASE"]
        for case_obj in node.cases:
            cond_sql = self.visit(case_obj.condition)
            then_sql = self.visit(case_obj.thenOp)
            parts.append(f"WHEN {cond_sql} THEN {then_sql}")
        else_sql = self.visit(node.elseOp)
        parts.append(f"ELSE {else_sql} END")
        return " ".join(parts)

    # =========================================================================
    # Validation visitor
    # =========================================================================

    def visit_Validation(self, node: AST.Validation) -> str:
        """Visit CHECK validation operator.

        Produces the standard CHECK output structure:
          identifiers, bool_var, imbalance, errorcode, errorlevel

        The inner validation expression (a comparison) produces a boolean
        measure that must be renamed to ``bool_var``.
        """
        validation_sql = self.visit(node.validation)

        error_code = f"'{node.error_code}'" if node.error_code else "NULL"
        error_level = str(node.error_level) if node.error_level is not None else "NULL"

        # Discover the measure name produced by the inner comparison.
        ds = self._get_dataset_structure(node.validation)
        if ds is None:
            # Fallback: cannot determine structure – wrap as before.
            return (
                f'SELECT t.*, NULL AS "imbalance", '
                f'{error_code} AS "errorcode", '
                f'{error_level} AS "errorlevel" '
                f"FROM ({validation_sql}) AS t"
            )

        id_names = ds.get_identifiers_names()
        measure_names = ds.get_measures_names()
        bool_measure = measure_names[0] if measure_names else "Me_1"

        # Build explicit SELECT list with proper renaming.
        cols: List[str] = []
        for id_name in id_names:
            cols.append(f"t.{quote_identifier(id_name)}")

        # Rename the comparison measure to bool_var.
        cols.append(f't.{quote_identifier(bool_measure)} AS "bool_var"')

        # Handle imbalance.
        if node.imbalance is not None:
            imbalance_sql = self.visit(node.imbalance)
            imb_ds = self._get_dataset_structure(node.imbalance)
            if imb_ds is not None:
                imb_measure = imb_ds.get_measures_names()[0]
                # Join with the imbalance source on identifiers.
                join_cond = " AND ".join(
                    f"t.{quote_identifier(n)} = i.{quote_identifier(n)}" for n in id_names
                )
                cols.append(f'i.{quote_identifier(imb_measure)} AS "imbalance"')
            else:
                join_cond = None
                cols.append('NULL AS "imbalance"')
        else:
            imbalance_sql = None
            join_cond = None
            cols.append('NULL AS "imbalance"')

        # errorcode / errorlevel – set only when bool_var is explicitly FALSE.
        bool_ref = f"t.{quote_identifier(bool_measure)}"
        cols.append(f'CASE WHEN {bool_ref} IS FALSE THEN {error_code} ELSE NULL END AS "errorcode"')
        cols.append(
            f'CASE WHEN {bool_ref} IS FALSE THEN {error_level} ELSE NULL END AS "errorlevel"'
        )

        select_clause = ", ".join(cols)
        sql = f"SELECT {select_clause} FROM ({validation_sql}) AS t"

        # Join with imbalance source if present.
        if imbalance_sql is not None and join_cond is not None:
            sql += f" JOIN ({imbalance_sql}) AS i ON {join_cond}"

        # invalid mode: keep only rows where the condition is FALSE.
        if node.invalid:
            sql += f" WHERE {bool_ref} IS FALSE"

        return sql

    # =========================================================================
    # Join visitor
    # =========================================================================

    def visit_JoinOp(self, node: AST.JoinOp) -> str:
        """Visit a join operation."""
        op = str(node.op).lower()
        join_type_map = {
            tokens.INNER_JOIN: "INNER",
            tokens.LEFT_JOIN: "LEFT",
            tokens.FULL_JOIN: "FULL",
            tokens.CROSS_JOIN: "CROSS",
        }
        join_type = join_type_map.get(op, "INNER")

        clause_info: List[Dict[str, Any]] = []
        for i, clause in enumerate(node.clauses):
            alias: Optional[str] = None
            actual_node = clause

            if isinstance(clause, AST.BinOp) and str(clause.op).lower() == "as":
                actual_node = clause.left
                alias = clause.right.value if hasattr(clause.right, "value") else str(clause.right)

            ds = self._get_dataset_structure(actual_node)
            table_src = self._get_dataset_sql(actual_node)

            if alias is None:
                alias = chr(ord("a") + i)

            clause_info.append(
                {
                    "node": actual_node,
                    "ds": ds,
                    "table_src": table_src,
                    "alias": alias,
                }
            )

        if not clause_info:
            return ""

        first_ds = clause_info[0]["ds"]
        if first_ds is None:
            return ""

        first_ids = set(first_ds.get_identifiers_names())
        self._get_output_dataset()

        using_ids: List[str] = []
        if node.using:
            using_ids = list(node.using)
        else:
            common = set(first_ids)
            for info in clause_info[1:]:
                if info["ds"]:
                    common &= set(info["ds"].get_identifiers_names())
            using_ids = sorted(common)

        first_alias = clause_info[0]["alias"]
        builder = SQLBuilder()

        # Build columns from the actual joined tables (NOT from the output
        # dataset, which may include columns added by subsequent clauses like
        # calc that wrap the join result).
        all_comps: Dict[str, Any] = {}
        for info in clause_info:
            if info["ds"]:
                for comp_name, comp in info["ds"].components.items():
                    if comp_name not in all_comps:
                        all_comps[comp_name] = comp

        cols: List[str] = []
        # Build a map of component_name → alias for correct qualification
        comp_alias_map: Dict[str, str] = {}
        for info in clause_info:
            if info["ds"]:
                for comp_name in info["ds"].components:
                    if comp_name not in comp_alias_map:
                        comp_alias_map[comp_name] = info["alias"]

        for comp_name in all_comps:
            alias_for_comp = comp_alias_map.get(comp_name, first_alias)
            cols.append(f"{alias_for_comp}.{quote_identifier(comp_name)}")

        if not cols:
            builder.select_all()
        else:
            builder.select(*cols)

        builder.from_table(clause_info[0]["table_src"], first_alias)

        for info in clause_info[1:]:
            on_parts = [
                f"{first_alias}.{quote_identifier(id_)} = {info['alias']}.{quote_identifier(id_)}"
                for id_ in using_ids
            ]
            on_clause = " AND ".join(on_parts) if on_parts else "1=1"
            builder.join(
                info["table_src"],
                info["alias"],
                on=on_clause,
                join_type=join_type,
            )

        return builder.build()

    # =========================================================================
    # Time aggregation visitor
    # =========================================================================

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> str:
        """Visit TIME_AGG operation."""
        period = node.period_to
        operand_sql = self.visit(node.operand) if node.operand else ""

        cast_date = f"CAST({operand_sql} AS DATE)"

        period_formats = {
            "Y": f"STRFTIME({cast_date}, '%Y')",
            "Q": (f"(STRFTIME({cast_date}, '%Y') || 'Q' || CAST(QUARTER({cast_date}) AS VARCHAR))"),
            "M": (
                f"(STRFTIME({cast_date}, '%Y') || 'M' || "
                f"LPAD(CAST(MONTH({cast_date}) AS VARCHAR), 2, '0'))"
            ),
            "S": (
                f"(STRFTIME({cast_date}, '%Y') || 'S' || "
                f"CAST(CEIL(MONTH({cast_date}) / 6.0) AS INTEGER))"
            ),
            "D": f"STRFTIME({cast_date}, '%Y-%m-%d')",
        }
        return period_formats.get(period, f"STRFTIME({cast_date}, '%Y')")

    # =========================================================================
    # Eval operator visitor
    # =========================================================================

    def visit_EvalOp(self, node: AST.EvalOp) -> str:
        """Visit EVAL operator (external routine execution)."""
        if not self.external_routines:
            raise ValueError(
                f"External routine '{node.name}' referenced but no external routines provided."
            )
        if node.name not in self.external_routines:
            raise ValueError(
                f"External routine '{node.name}' not found in provided external routines."
            )

        routine = self.external_routines[node.name]
        return routine.query

"""
SQL Transpiler for VTL AST.

Converts VTL AST nodes into DuckDB SQL queries using the visitor pattern.
Each top-level Assignment produces one SQL SELECT query. Queries are executed
sequentially, with results registered as tables for subsequent queries.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar import tokens
from vtlengine.DataTypes import COMP_NAME_MAPPING
from vtlengine.duckdb_transpiler.Transpiler.operators import (
    get_duckdb_type,
    registry,
)
from vtlengine.duckdb_transpiler.Transpiler.sql_builder import SQLBuilder, quote_identifier
from vtlengine.duckdb_transpiler.Transpiler.structure_visitor import (
    _COMPONENT,
    _DATASET,
    _SCALAR,
    StructureVisitor,
)
from vtlengine.Model import Dataset, ExternalRoutine, Role, Scalar, ValueDomain

# Datapoint rule operator mappings (module-level to avoid dataclass mutable default)
_DP_OP_MAP: Dict[str, str] = {
    "=": "=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "<>": "!=",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "and": "AND",
    "or": "OR",
}


@dataclass
class SQLTranspiler(StructureVisitor, ASTTemplate):
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

    # Clause context for component-level resolution
    _in_clause: bool = field(default=False, init=False)
    _current_dataset: Optional[Dataset] = field(default=None, init=False)

    # Join context: maps "alias#comp" -> aliased column name in SQL output
    # e.g. {"d2#Me_2": "d2#Me_2"} for duplicate non-identifier columns
    _join_alias_map: Dict[str, str] = field(default_factory=dict, init=False)

    # Set of qualified names consumed (renamed/removed) by join body clauses
    _consumed_join_aliases: Set[str] = field(default_factory=set, init=False)

    # UDO definitions: name -> Operator node info
    _udos: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # UDO parameter stack
    _udo_params: Optional[List[Dict[str, Any]]] = field(default=None, init=False)

    # Datapoint ruleset definitions
    _dprs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # Datapoint ruleset context
    _dp_signature: Optional[Dict[str, str]] = field(default=None, init=False)

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
                if name in deps.outputs or name in deps.persistent:
                    return deps.inputs
        return []

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
                self.visit_DPRuleset(child)
            elif isinstance(child, AST.Assignment):
                name = child.left.value  # type: ignore[attr-defined]
                self.current_assignment = name
                self.inputs = self._get_assignment_inputs(name)

                # Check if this is a scalar assignment
                if name in self.output_scalars:
                    # Scalar assignments produce a literal value, wrap in SELECT
                    is_persistent = isinstance(child, AST.PersistentAssignment)
                    value_sql = self.visit(child)
                    # Ensure it's a valid SQL query
                    if not value_sql.strip().upper().startswith("SELECT"):
                        value_sql = f"SELECT {value_sql} AS value"
                    queries.append((name, value_sql, is_persistent))
                else:
                    is_persistent = isinstance(child, AST.PersistentAssignment)
                    query = self.visit(child)
                    # Post-process: unqualify any remaining "alias#comp" column
                    # names back to plain "comp" to match the expected output
                    # structure from semantic analysis.
                    query = self._unqualify_join_columns(name, query)
                    queries.append((name, query, is_persistent))

                # Reset join alias map after each assignment
                self._join_alias_map = {}
                self._consumed_join_aliases = set()

        return queries

    def _unqualify_join_columns(self, ds_name: str, query: str) -> str:
        """Wrap the query to rename any remaining alias#comp columns to comp.

        After join clauses (calc/drop/keep/rename) are applied, some columns
        may still have qualified names like ``d1#Me_2``.  The output dataset
        (from semantic analysis) expects plain names like ``Me_2``.  This
        method adds a wrapping SELECT to rename them.
        """
        if not self._join_alias_map:
            return query

        output_ds = self.output_datasets.get(ds_name)
        if output_ds is None:
            return query

        # Build a mapping from unqualified name -> list of qualified candidates,
        # excluding any that were consumed (renamed/removed) by join body clauses
        output_comp_names = set(output_ds.components.keys())
        candidates: Dict[str, List[str]] = {}

        for qualified in self._join_alias_map:
            if qualified in self._consumed_join_aliases:
                continue
            if qualified not in output_comp_names and "#" in qualified:
                unqualified = qualified.split("#", 1)[1]
                if unqualified in output_comp_names:
                    candidates.setdefault(unqualified, []).append(qualified)

        if not candidates:
            return query

        # For each unqualified name, pick the surviving qualified name
        renames: Dict[str, str] = {}
        for unqualified, quals in candidates.items():
            # Use the first (and typically only) surviving candidate
            renames[quals[0]] = unqualified

        if not renames:
            return query

        # Build a wrapping SELECT with renames
        cols: List[str] = []
        for comp_name in output_ds.components:
            # Check if this component comes from a qualified name
            reverse_found = False
            for qual, unqual in renames.items():
                if unqual == comp_name:
                    cols.append(f"{quote_identifier(qual)} AS {quote_identifier(comp_name)}")
                    reverse_found = True
                    break
            if not reverse_found:
                cols.append(quote_identifier(comp_name))

        select_clause = ", ".join(cols)
        return f"SELECT {select_clause} FROM ({query})"

    def visit_Assignment(self, node: AST.Assignment) -> str:
        """Visit an assignment and return the SQL for its right-hand side."""
        return self.visit(node.right)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> str:
        """Visit a persistent assignment (same as regular for SQL generation)."""
        return self.visit(node.right)

    # =========================================================================
    # Datapoint Ruleset definition and validation
    # =========================================================================

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        """Register a datapoint ruleset definition."""
        # Build signature: alias -> actual column name
        signature: Dict[str, str] = {}
        if not isinstance(node.params, AST.DefIdentifier):
            for param in node.params:
                alias = param.alias if param.alias is not None else param.value
                signature[alias] = param.value

        # Auto-number unnamed rules
        rule_names = [r.name for r in node.rules if r.name is not None]
        if len(rule_names) == 0:
            for i, rule in enumerate(node.rules):
                rule.name = str(i + 1)

        self._dprs[node.name] = {
            "rules": node.rules,
            "signature": signature,
            "signature_type": node.signature_type,
        }

    def visit_DPValidation(self, node: AST.DPValidation) -> str:  # type: ignore[override]
        """Generate SQL for check_datapoint operator."""
        dpr_name = node.ruleset_name
        dpr_info = self._dprs[dpr_name]
        signature = dpr_info["signature"]

        # Get input dataset SQL and structure
        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            raise ValueError("Cannot resolve dataset for check_datapoint")

        self._get_output_dataset()
        output_mode = node.output.value if node.output else "invalid"

        id_cols = ds.get_identifiers_names()
        measure_cols = ds.get_measures_names()

        # Build SQL for each rule and UNION ALL
        rule_queries: List[str] = []
        for rule in dpr_info["rules"]:
            rule_sql = self._build_dp_rule_sql(
                rule=rule,
                table_src=table_src,
                signature=signature,
                id_cols=id_cols,
                measure_cols=measure_cols,
                output_mode=output_mode,
            )
            rule_queries.append(rule_sql)

        if not rule_queries:
            # Empty ruleset — return empty select
            cols = [quote_identifier(c) for c in id_cols]
            return f"SELECT {', '.join(cols)} FROM {table_src} WHERE 1=0"

        combined = " UNION ALL ".join(rule_queries)
        return combined

    def _build_dp_rule_sql(
        self,
        rule: AST.DPRule,
        table_src: str,
        signature: Dict[str, str],
        id_cols: List[str],
        measure_cols: List[str],
        output_mode: str,
    ) -> str:
        """Build SQL for a single datapoint rule."""
        rule_name = rule.name or ""

        # Store the signature for DefIdentifier resolution
        self._dp_signature = signature

        has_when = rule.rule.op == "when"
        if has_when:
            when_cond_sql = self._visit_dp_expr(rule.rule.left, signature)
            then_expr_sql = self._visit_dp_expr(rule.rule.right, signature)
        else:
            when_cond_sql = None
            then_expr_sql = self._visit_dp_expr(rule.rule, signature)

        self._dp_signature = None

        # Common parts
        ec_sql = f"'{rule.erCode}'" if rule.erCode else "NULL"
        el_sql = str(float(rule.erLevel)) if rule.erLevel is not None else "NULL"
        fail_cond = (
            f"({when_cond_sql}) AND NOT ({then_expr_sql})"
            if when_cond_sql
            else f"NOT ({then_expr_sql})"
        )

        select_parts: List[str] = [quote_identifier(c) for c in id_cols]

        if output_mode == "invalid":
            # Include measures, filter to failing rows only
            select_parts.extend(quote_identifier(m) for m in measure_cols)
            select_parts.append(f"'{rule_name}' AS {quote_identifier('ruleid')}")
            select_parts.append(f"{ec_sql} AS {quote_identifier('errorcode')}")
            select_parts.append(f"{el_sql} AS {quote_identifier('errorlevel')}")
            return f"SELECT {', '.join(select_parts)} FROM {table_src} WHERE {fail_cond}"

        # "all" and "all_measures" share the same structure
        if output_mode == "all_measures":
            select_parts.extend(quote_identifier(m) for m in measure_cols)

        bool_expr = (
            f"CASE WHEN ({when_cond_sql}) THEN ({then_expr_sql}) ELSE TRUE END"
            if when_cond_sql
            else f"({then_expr_sql})"
        )
        select_parts.append(f"{bool_expr} AS {quote_identifier('bool_var')}")
        select_parts.append(f"'{rule_name}' AS {quote_identifier('ruleid')}")
        select_parts.append(
            f"CASE WHEN {fail_cond} THEN {ec_sql} ELSE NULL END AS {quote_identifier('errorcode')}"
        )
        select_parts.append(
            f"CASE WHEN {fail_cond} THEN {el_sql} ELSE NULL END AS {quote_identifier('errorlevel')}"
        )
        return f"SELECT {', '.join(select_parts)} FROM {table_src}"

    def _visit_dp_expr(self, node: AST.AST, signature: Dict[str, str]) -> str:
        """Visit an expression node in the context of a datapoint rule.

        Resolves DefIdentifier/VarID aliases via the signature mapping and
        delegates to the regular visitor for other node types.
        """
        if isinstance(node, AST.HRBinOp):
            return self._visit_dp_hr_binop(node, signature)
        if isinstance(node, AST.HRUnOp):
            return self._visit_dp_hr_unop(node, signature)
        if isinstance(node, (AST.DefIdentifier, AST.VarID)):
            col_name = signature.get(node.value, node.value)
            return quote_identifier(col_name)
        if isinstance(node, AST.Constant):
            return self._to_sql_literal(node.value)
        if isinstance(node, AST.BinOp):
            return self._visit_dp_binop(node, signature)
        if isinstance(node, AST.UnaryOp):
            return self._visit_dp_unop(node, signature)
        if isinstance(node, AST.If):
            cond_sql = self._visit_dp_expr(node.condition, signature)
            then_sql = self._visit_dp_expr(node.thenOp, signature)
            else_sql = self._visit_dp_expr(node.elseOp, signature)
            return f"CASE WHEN ({cond_sql}) THEN ({then_sql}) ELSE ({else_sql}) END"
        # Fallback: use the regular transpiler visitor, saving/restoring DP context
        saved_sig = self._dp_signature
        self._dp_signature = signature
        result = self.visit(node)
        self._dp_signature = saved_sig
        return result

    def _visit_dp_hr_binop(self, node: AST.HRBinOp, signature: Dict[str, str]) -> str:
        """Visit an HRBinOp in a datapoint rule context."""
        left_sql = self._visit_dp_expr(node.left, signature)
        right_sql = self._visit_dp_expr(node.right, signature)
        op = node.op
        if op == "when":
            return f"CASE WHEN ({left_sql}) THEN ({right_sql}) ELSE TRUE END"
        return self._dp_binary_sql(op, left_sql, right_sql)

    def _visit_dp_hr_unop(self, node: AST.HRUnOp, signature: Dict[str, str]) -> str:
        """Visit an HRUnOp in a datapoint rule context."""
        operand_sql = self._visit_dp_expr(node.operand, signature)
        return self._dp_unary_sql(node.op, operand_sql)

    def _visit_dp_binop(self, node: AST.BinOp, signature: Dict[str, str]) -> str:
        """Visit a BinOp in a datapoint rule context."""
        left_sql = self._visit_dp_expr(node.left, signature)
        right_sql = self._visit_dp_expr(node.right, signature)
        return self._dp_binary_sql(node.op, left_sql, right_sql)

    def _visit_dp_unop(self, node: AST.UnaryOp, signature: Dict[str, str]) -> str:
        """Visit a UnaryOp in a datapoint rule context."""
        operand_sql = self._visit_dp_expr(node.operand, signature)
        return self._dp_unary_sql(node.op, operand_sql)

    def _dp_binary_sql(self, op: str, left_sql: str, right_sql: str) -> str:
        """Generate SQL for a binary operation in datapoint rule context."""
        if op == "nvl":
            return f"COALESCE({left_sql}, {right_sql})"
        if registry.binary.is_registered(op):
            return registry.binary.generate(op, left_sql, right_sql)
        sql_op = _DP_OP_MAP.get(op, op)
        return f"({left_sql} {sql_op} {right_sql})"

    def _dp_unary_sql(self, op: str, operand_sql: str) -> str:
        """Generate SQL for a unary operation in datapoint rule context."""
        if op == "not":
            return f"NOT ({operand_sql})"
        if op == "-":
            return f"-({operand_sql})"
        if op == tokens.ISNULL:
            return f"({operand_sql} IS NULL)"
        if registry.unary.is_registered(op):
            return registry.unary.generate(op, operand_sql)
        return f"{op}({operand_sql})"

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

    def visit_UDOCall(self, node: AST.UDOCall) -> str:  # type: ignore[override]
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
            elif param_info.get("default") is not None:
                # Use the default value AST node when argument is not provided
                bindings[param_name] = param_info["default"]

        self._push_udo_params(bindings)
        try:
            result = self.visit(expression)
        finally:
            self._pop_udo_params()

        return result

    # =========================================================================
    # Leaf visitors
    # =========================================================================

    def visit_VarID(self, node: AST.VarID) -> str:  # type: ignore[override]
        """Visit a variable identifier."""
        name = node.value
        udo_val = self._get_udo_param(name)
        if udo_val is not None:
            # Handle VarID specifically to avoid infinite recursion when
            # a UDO param name matches its argument name (e.g., DS → VarID('DS')).
            if isinstance(udo_val, AST.VarID):
                resolved_name = udo_val.value
                if resolved_name in self.available_tables:
                    return f"SELECT * FROM {quote_identifier(resolved_name)}"
                if resolved_name in self.scalars:
                    sc = self.scalars[resolved_name]
                    return self._to_sql_literal(sc.value, type(sc.data_type).__name__)
                if resolved_name != name:
                    return self.visit(udo_val)
                return quote_identifier(resolved_name)
            if isinstance(udo_val, AST.AST):
                return self.visit(udo_val)
            if isinstance(udo_val, str):
                return quote_identifier(udo_val)

        if name in self.scalars:
            sc = self.scalars[name]
            return self._to_sql_literal(sc.value, type(sc.data_type).__name__)

        if self._in_clause and self._current_dataset and name in self._current_dataset.components:
            return quote_identifier(name)

        # In clause context, check if the variable matches a qualified column
        # (e.g., "Me_2" → "d1#Me_2" when datasets share that column name).
        if (
            self._in_clause
            and self._current_dataset
            and name not in self._current_dataset.components
        ):
            matches = [
                comp_name
                for comp_name in self._current_dataset.components
                if "#" in comp_name and comp_name.split("#", 1)[1] == name
            ]
            if len(matches) == 1:
                return quote_identifier(matches[0])

        if name in self.available_tables:
            return f"SELECT * FROM {quote_identifier(name)}"

        return quote_identifier(name)

    def visit_Constant(self, node: AST.Constant) -> str:  # type: ignore[override]
        """Visit a constant literal."""
        return self._constant_to_sql(node)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        """Visit a parameter constant."""
        return str(node.value)

    def visit_Identifier(self, node: AST.Identifier) -> str:
        """Visit an identifier node."""
        return quote_identifier(node.value)

    def visit_ID(self, node: AST.ID) -> str:  # type: ignore[override]
        """Visit an ID node (used for type names, placeholders like '_', etc.)."""
        if node.value == "_":
            # VTL underscore means "use default" - return None marker
            return ""
        return node.value

    def visit_ParFunction(self, node: AST.ParFunction) -> str:  # type: ignore[override]
        """Visit a parenthesized function/expression."""
        return self.visit(node.operand)

    def visit_Collection(self, node: AST.Collection) -> str:  # type: ignore[override]
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
    # Generic dataset-level helpers
    # =========================================================================

    def _apply_to_measures(
        self,
        ds_node: AST.AST,
        expr_fn: "Callable[[str], str]",
        output_name_override: Optional[str] = None,
    ) -> str:
        """Apply a SQL expression to each measure of a dataset, passing identifiers through.

        This factors out the very common pattern of:
          SELECT id1, id2, f(Me_1) AS Me_1, f(Me_2) AS Me_2 FROM ...

        Args:
            ds_node: The AST node for the dataset operand.
            expr_fn: A callable that receives a quoted column reference
                     (e.g. ``'"Me_1"'``) and returns the SQL expression
                     to use for that measure.
            output_name_override: When set, forces all measures to use this
                                  name (used for mono-measure → bool_var etc.).
                                  When ``None``, the output dataset from semantic
                                  analysis is consulted to remap single-measure
                                  names automatically.

        Returns:
            A complete ``SELECT … FROM …`` SQL string.
        """
        ds = self._get_dataset_structure(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve dataset structure for dataset-level operation")

        table_src = self._get_dataset_sql(ds_node)
        output_ds = self._get_output_dataset()
        output_measure_names = list(output_ds.get_measures_names()) if output_ds else []
        input_measures = ds.get_measures_names()

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
            elif comp.role == Role.MEASURE:
                expr = expr_fn(quote_identifier(name))
                if output_name_override is not None:
                    out_name = output_name_override
                elif (
                    output_measure_names
                    and len(input_measures) == 1
                    and len(output_measure_names) == 1
                ):
                    out_name = output_measure_names[0]
                else:
                    out_name = name
                cols.append(f"{expr} AS {quote_identifier(out_name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

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
        if ds is None or not isinstance(ds, Dataset):
            # Fallback: both sides are scalar-like (e.g. filter with scalar variables)
            left_sql = self.visit(ds_node)
            right_sql = self.visit(scalar_node)
            if ds_on_left:
                return registry.binary.generate(op, left_sql, right_sql)
            else:
                return registry.binary.generate(op, right_sql, left_sql)

        scalar_sql = self.visit(scalar_node)

        def _bin_expr(col_ref: str) -> str:
            if ds_on_left:
                return registry.binary.generate(op, col_ref, scalar_sql)
            return registry.binary.generate(op, scalar_sql, col_ref)

        return self._apply_to_measures(ds_node, _bin_expr)

    # =========================================================================
    # Expression visitors
    # =========================================================================

    def visit_BinOp(self, node: AST.BinOp) -> str:  # type: ignore[override]
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

        # Inside a clause context (e.g., join body calc/filter/keep/drop/rename),
        # membership just references a column name — but when there are duplicate
        # columns across joined datasets, use the qualified "alias#comp" name.
        if self._in_clause:
            ds_name = node.left.value if hasattr(node.left, "value") else str(node.left)
            qualified = f"{ds_name}#{comp_name}"
            if qualified in self._join_alias_map:
                return quote_identifier(qualified)
            # Check if the component exists without qualification in the dataset
            # (i.e. it's not duplicated across datasets)
            return quote_identifier(comp_name)

        ds = self._get_dataset_structure(node.left)
        table_src = self._get_dataset_sql(node.left)

        if ds is None:
            ds_name = self._resolve_dataset_name(node.left)
            return f"SELECT {quote_identifier(comp_name)} FROM {quote_identifier(ds_name)}"

        # Determine if the component needs renaming (identifiers/attributes become measures)
        target_comp = ds.components.get(comp_name)
        alias_name = comp_name
        if target_comp and target_comp.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            alias_name = COMP_NAME_MAPPING.get(target_comp.data_type, comp_name)

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_identifier(name))
        # Add the target component, with rename if needed
        if alias_name != comp_name:
            cols.append(f"{quote_identifier(comp_name)} AS {quote_identifier(alias_name)}")
        else:
            # For measures, just select the component (avoid duplicates with identifiers)
            if comp_name not in [n for n, c in ds.components.items() if c.role == Role.IDENTIFIER]:
                cols.append(quote_identifier(comp_name))
            else:
                # Component is an identifier but no mapping found, still select it aliased
                cols.append(quote_identifier(comp_name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_match_characters(self, node: AST.BinOp) -> str:
        """Visit match_characters operator using registry."""
        left_type = self._get_operand_type(node.left)
        pattern_sql = self.visit(node.right)

        if left_type == _DATASET:
            return self._apply_to_measures(
                node.left,
                lambda col: registry.binary.generate(tokens.CHARSET_MATCH, col, pattern_sql),
            )
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

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:  # type: ignore[override]
        """Visit a unary operation."""
        op = str(node.op).lower()

        # --- Special-case operators that need dedicated logic ---
        if op == tokens.PERIOD_INDICATOR:
            operand_sql = self.visit(node.operand)
            return f"vtl_period_indicator(vtl_period_parse({operand_sql}))"

        if op in (tokens.FLOW_TO_STOCK, tokens.STOCK_TO_FLOW):
            return self._visit_time_window_op(node, op)

        if op in (tokens.DAYTOYEAR, tokens.DAYTOMONTH, tokens.YEARTODAY, tokens.MONTHTODAY):
            return self._visit_duration_conversion(node, op)

        if op == tokens.FILL_TIME_SERIES:
            return self._visit_fill_time_series(node)

        # --- Generic path: registry-based unary ---
        operand_type = self._get_operand_type(node.operand)

        if operand_type == _DATASET:
            # isnull on mono-measure dataset produces "bool_var"
            name_override: Optional[str] = None
            if op == tokens.ISNULL:
                ds = self._get_dataset_structure(node.operand)
                if ds and len(ds.get_measures_names()) == 1:
                    name_override = "bool_var"

            def _unary_expr(col_ref: str) -> str:
                if registry.unary.is_registered(op):
                    return registry.unary.generate(op, col_ref)
                return f"{op.upper()}({col_ref})"

            return self._apply_to_measures(node.operand, _unary_expr, name_override)
        else:
            operand_sql = self.visit(node.operand)
            if registry.unary.is_registered(op):
                return registry.unary.generate(op, operand_sql)
            return f"{op.upper()}({operand_sql})"

    def _visit_time_window_op(self, node: AST.UnaryOp, op_name: str) -> str:
        """Visit a time-based window operation (flow_to_stock or stock_to_flow).

        Both operations share the same pattern: iterate dataset components,
        pass identifiers through, and apply a window function over the time
        identifier to each measure.
        """
        if not self._is_dataset(node.operand):
            raise ValueError(f"{op_name} requires a dataset operand")

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            raise ValueError(f"Cannot resolve dataset for {op_name}")

        time_id, other_ids = self._split_time_identifier(ds)

        partition = ", ".join(quote_identifier(i) for i in other_ids)
        partition_clause = f"PARTITION BY {partition} " if partition else ""
        order_clause = f"ORDER BY {quote_identifier(time_id)}"
        window = f"{partition_clause}{order_clause}"

        def _measure_expr(col_ref: str) -> str:
            if op_name == "flow_to_stock":
                return f"SUM({col_ref}) OVER ({window})"
            lag = f"LAG({col_ref}) OVER ({window})"
            return f"COALESCE({col_ref} - {lag}, {col_ref})"

        return self._apply_to_measures(node.operand, _measure_expr)

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

    def visit_ParamOp(self, node: AST.ParamOp) -> str:  # type: ignore[override]
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
        params_sql = self._visit_params(node.params)

        # Default precision for ROUND/TRUNC when no parameter given
        if op in (tokens.ROUND, tokens.TRUNC) and not params_sql:
            params_sql = ["0"]

        def _param_expr(col_ref: str) -> str:
            if registry.parameterized.is_registered(op):
                return registry.parameterized.generate(
                    op, col_ref, *[p for p in params_sql if p is not None]
                )
            all_args = [col_ref] + [a for a in params_sql if a is not None]
            return f"{op.upper()}({', '.join(all_args)})"

        return self._apply_to_measures(ds_node, _param_expr)

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
            return self._apply_to_measures(
                operand,
                lambda col: self._cast_expr(col, duckdb_type, target_type_str, mask),
            )
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

    def _visit_random(self, node: AST.ParamOp) -> str:
        """Visit RANDOM operator (ParamOp form): deterministic hash-based random."""
        seed_node = node.children[0] if node.children else None
        index_node = node.params[0] if node.params else None
        seed_type = self._get_operand_type(seed_node) if seed_node else _SCALAR

        if seed_type == _DATASET and seed_node is not None:
            index_sql = self.visit(index_node) if index_node else "0"
            return self._apply_to_measures(
                seed_node,
                lambda col: self._random_hash_expr(col, index_sql),
            )

        seed_sql = self.visit(seed_node) if seed_node else "0"
        index_sql = self.visit(index_node) if index_node else "0"
        return self._random_hash_expr(seed_sql, index_sql)

    def _visit_random_binop(self, node: AST.BinOp) -> str:
        """Visit RANDOM operator (BinOp form, e.g. inside calc)."""
        seed_node = node.left
        index_node = node.right

        seed_type = self._get_operand_type(seed_node)

        if seed_type == _DATASET:
            index_sql = self.visit(index_node)
            return self._apply_to_measures(
                seed_node,
                lambda col: self._random_hash_expr(col, index_sql),
            )

        seed_sql = self.visit(seed_node)
        index_sql = self.visit(index_node)

        return self._random_hash_expr(seed_sql, index_sql)

    @staticmethod
    def _random_hash_expr(seed_sql: str, index_sql: str) -> str:
        """Build a deterministic hash-based random expression in [0, 1)."""
        return (
            f"(ABS(hash(CAST({seed_sql} AS VARCHAR) || '_' || "
            f"CAST({index_sql} AS VARCHAR))) % 1000000) / 1000000.0"
        )

    # =========================================================================
    # Clause visitor (RegularAggregation)
    # =========================================================================

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:  # type: ignore[override]
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
        elif op == tokens.UNPIVOT:
            return self._visit_unpivot(node)
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
                # Resolve UDO component parameters for column names
                udo_val = self._get_udo_param(col_name)
                if udo_val is not None:
                    if isinstance(udo_val, (AST.VarID, AST.Identifier)):
                        col_name = udo_val.value
                    elif isinstance(udo_val, str):
                        col_name = udo_val
                expr_sql = self.visit(assignment.right)
                calc_exprs[col_name] = expr_sql

        self._in_clause = False
        self._current_dataset = None

        # Build SELECT: keep original columns that are NOT being overwritten,
        # then add the calc expressions (possibly replacing originals).
        select_cols: List[str] = []
        for name in ds.components:
            if name in calc_exprs:
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

        # Identifiers are always kept
        keep_names: List[str] = [
            name for name, comp in ds.components.items() if comp.role == Role.IDENTIFIER
        ]
        keep_names.extend(self._resolve_join_component_names(node.children))

        # Track qualified names that are NOT kept (consumed by this clause)
        keep_set = set(keep_names)
        for qualified in self._join_alias_map:
            if qualified not in keep_set:
                self._consumed_join_aliases.add(qualified)

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
        drop_names = self._resolve_join_component_names(node.children)

        # Track consumed qualified names
        for name in drop_names:
            if name in self._join_alias_map:
                self._consumed_join_aliases.add(name)

        if not drop_names:
            return f"SELECT * FROM {table_src}"

        exclude = ", ".join(quote_identifier(n) for n in drop_names)
        return SQLBuilder().select(f"* EXCLUDE ({exclude})").from_table(table_src).build()

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
                old = child.old_name
                # Check if alias-qualified name is in the join alias map
                if "#" in old and old in self._join_alias_map:
                    renames[old] = child.new_name
                    # Track renamed qualified name as consumed
                    self._consumed_join_aliases.add(old)
                elif "#" in old:
                    # Strip alias prefix from membership refs (e.g. d2#Me_2 -> Me_2)
                    old = old.split("#", 1)[1]
                    renames[old] = child.new_name
                else:
                    renames[old] = child.new_name

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
        remove_ids: set[str] = set()
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

        # Extract group-by identifiers from AST nodes to avoid using the
        # overall output dataset (which may represent a join result).
        group_ids: List[str] = []
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
                        if isinstance(g, (AST.VarID, AST.Identifier)) and g.value not in group_ids:
                            group_ids.append(g.value)

        # Fall back to output/input dataset identifiers when no explicit grouping
        if not group_ids:
            output_ds = self._get_output_dataset()
            group_ids = list(
                output_ds.get_identifiers_names() if output_ds else ds.get_identifiers_names()
            )

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

    def _visit_unpivot(self, node: AST.RegularAggregation) -> str:
        """Visit unpivot clause: DS[unpivot new_id, new_measure].

        Transforms measures into rows.  For each measure column, produces one
        row per data point with the measure *name* as the new identifier value
        and the measure *value* as the new measure value.  Rows where the
        measure value is NULL are dropped (VTL 2.1 RM line 7200).
        """
        if not node.dataset:
            return ""

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        if ds is None:
            return f"SELECT * FROM {table_src}"

        if len(node.children) < 2:
            raise ValueError("Unpivot clause requires two operands")

        new_id_name = (
            node.children[0].value if hasattr(node.children[0], "value") else str(node.children[0])
        )
        new_measure_name = (
            node.children[1].value if hasattr(node.children[1], "value") else str(node.children[1])
        )

        id_names = ds.get_identifiers_names()
        measure_names = ds.get_measures_names()

        if not measure_names:
            return f"SELECT * FROM {table_src}"

        # Build one SELECT per measure, filtering NULLs, then UNION ALL
        parts: List[str] = []
        for measure in measure_names:
            cols: List[str] = [quote_identifier(i) for i in id_names]
            cols.append(f"'{measure}' AS {quote_identifier(new_id_name)}")
            cols.append(f"{quote_identifier(measure)} AS {quote_identifier(new_measure_name)}")
            select_clause = ", ".join(cols)
            part = (
                f"SELECT {select_clause} FROM {table_src} "
                f"WHERE {quote_identifier(measure)} IS NOT NULL"
            )
            parts.append(part)

        return " UNION ALL ".join(parts)

    # =========================================================================
    # Aggregation visitor
    # =========================================================================

    def visit_Aggregation(self, node: AST.Aggregation) -> str:  # type: ignore[override]
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

        # count() with no operand -> COUNT excluding all-null measure rows
        if node.operand is None:
            if op == tokens.COUNT:
                # VTL count() without operand counts data points where at least
                # one measure is not null.  Build a CASE expression to skip rows
                # where all measures are null.
                if self._in_clause and self._current_dataset:
                    measures = self._current_dataset.get_measures_names()
                    if measures:
                        or_parts = " OR ".join(
                            f"{quote_identifier(m)} IS NOT NULL" for m in measures
                        )
                        return f"COUNT(CASE WHEN {or_parts} THEN 1 END)"
                return "COUNT(*)"
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
        if self._udo_params:
            effective_ds = ds
        else:
            output_ds = self._get_output_dataset()
            effective_ds = output_ds if output_ds is not None else ds

        all_ids = effective_ds.get_identifiers_names()
        group_cols = self._resolve_group_cols(node, all_ids)

        cols: List[str] = [quote_identifier(g) for g in group_cols]

        # count replaces all measures with a single int_var column.
        # VTL count() excludes rows where all measures are null.
        if op == tokens.COUNT:
            # VTL spec: count() always produces a single measure "int_var"
            alias = "int_var"
            # Build conditional count excluding all-null measure rows
            # VTL count returns NULL when no data points have any non-null measure
            source_measures = ds.get_measures_names()
            if source_measures:
                and_parts = " AND ".join(
                    f"{quote_identifier(m)} IS NOT NULL" for m in source_measures
                )
                cols.append(
                    f"NULLIF(COUNT(CASE WHEN {and_parts} THEN 1 END), 0)"
                    f" AS {quote_identifier(alias)}"
                )
            else:
                # No measures: count should be NULL (no non-null measures to count)
                cols.append(f"NULL AS {quote_identifier(alias)}")
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
        """Build the analytic function expression (without OVER).

        For ratio_to_report, returns the complete expression including OVER clause.
        Callers must check _is_self_contained_analytic() to avoid adding OVER again.
        """
        if op == tokens.RATIO_TO_REPORT:
            over_clause = self._build_over_clause(node)
            return f"CAST({operand_sql} AS DOUBLE) / SUM({operand_sql}) OVER ({over_clause})"
        if op == tokens.RANK:
            return "RANK()"
        if op in (tokens.LAG, tokens.LEAD) and node.params:
            offset = node.params[0] if node.params else 1
            default_val = node.params[1] if len(node.params) > 1 else None
            func_sql = f"{op.upper()}({operand_sql}, {offset}"
            if default_val is not None:
                if isinstance(default_val, AST.AST):
                    default_sql = self.visit(default_val)
                else:
                    default_sql = str(default_val)
                func_sql += f", {default_sql}"
            return func_sql + ")"
        if registry.analytic.is_registered(op):
            return registry.analytic.generate(op, operand_sql)
        return f"{op.upper()}({operand_sql})"

    def visit_Analytic(self, node: AST.Analytic) -> str:  # type: ignore[override]
        """Visit an analytic (window) function."""
        op = str(node.op).lower()

        # Check if operand is a dataset — needs dataset-level handling
        if node.operand and self._get_operand_type(node.operand) == _DATASET:
            return self._visit_analytic_dataset(node, op)

        # Component-level: single expression with OVER
        operand_sql = self.visit(node.operand) if node.operand else ""
        func_sql = self._build_analytic_expr(op, operand_sql, node)
        # ratio_to_report already includes its own OVER clause
        if op == tokens.RATIO_TO_REPORT:
            return func_sql
        over_clause = self._build_over_clause(node)
        return f"{func_sql} OVER ({over_clause})"

    def _visit_analytic_dataset(self, node: AST.Analytic, op: str) -> str:
        """Visit a dataset-level analytic: applies the window function to each measure."""
        over_clause = self._build_over_clause(node)

        def _analytic_expr(col_ref: str) -> str:
            func_sql = self._build_analytic_expr(op, col_ref, node)
            if op == tokens.RATIO_TO_REPORT:
                return func_sql
            return f"{func_sql} OVER ({over_clause})"

        # VTL count always produces a single "int_var" measure
        name_override = "int_var" if op == tokens.COUNT else None
        if node.operand is None:
            raise ValueError("Analytic node must have an operand")
        return self._apply_to_measures(node.operand, _analytic_expr, name_override)

    def visit_Windowing(self, node: AST.Windowing) -> str:  # type: ignore[override]
        """Visit a windowing specification."""
        type_str = str(node.type_).upper() if node.type_ else "ROWS"
        # Map VTL types to SQL: DATA POINTS → ROWS
        if "DATA" in type_str:
            type_str = "ROWS"
        elif "RANGE" in type_str:
            type_str = "RANGE"

        def bound_str(value: Union[int, str], mode: str) -> str:
            mode_up = mode.upper()
            val_str = str(value).upper()
            if "CURRENT" in mode_up or val_str == "CURRENT ROW":
                return "CURRENT ROW"
            if val_str == "UNBOUNDED" or (isinstance(value, int) and value < 0):
                return f"UNBOUNDED {mode_up}"
            return f"{value} {mode_up}"

        start = bound_str(node.start, node.start_mode)
        stop = bound_str(node.stop, node.stop_mode)

        return f"{type_str} BETWEEN {start} AND {stop}"

    # =========================================================================
    # MulOp visitor (set ops, between, exists_in, current_date)
    # =========================================================================

    def visit_MulOp(self, node: AST.MulOp) -> str:  # type: ignore[override]
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

        low_sql = self.visit(node.children[1])
        high_sql = self.visit(node.children[2])

        if operand_type == _DATASET:
            return self._apply_to_measures(
                node.children[0],
                lambda col: self._between_expr(col, low_sql, high_sql),
            )

        operand_sql = self.visit(node.children[0])
        return self._between_expr(operand_sql, low_sql, high_sql)

    def _visit_exists_in_mul(self, node: AST.MulOp) -> str:
        """Visit EXISTS_IN in MulOp form, handling the optional retain parameter."""
        if len(node.children) < 2:
            raise ValueError("exists_in requires at least 2 operands")

        base_sql = self._build_exists_in_sql(node.children[0], node.children[1])

        # Check for retain parameter (true / false / all)
        if len(node.children) >= 3:
            retain_node = node.children[2]
            if isinstance(retain_node, AST.Constant) and retain_node.value is True:
                return f'SELECT * FROM ({base_sql}) AS _ei WHERE "bool_var" = TRUE'
            if isinstance(retain_node, AST.Constant) and retain_node.value is False:
                return f'SELECT * FROM ({base_sql}) AS _ei WHERE "bool_var" = FALSE'
            # "all" or any other value → return all rows (default behaviour)

        return base_sql

    def _visit_set_operation(self, node: AST.MulOp, op: str) -> str:
        """Visit set operations: UNION, INTERSECT, SETDIFF, SYMDIFF.

        VTL set operations match data points by **identifiers only**, keeping
        the measure values from the first (or relevant) dataset.  This differs
        from SQL INTERSECT/EXCEPT which compare all columns.
        """
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

        if len(child_sqls) < 2:
            return child_sqls[0] if child_sqls else ""

        first_ds = self._get_dataset_structure(node.children[0])
        if first_ds is None:
            return registry.set_ops.generate(op, *child_sqls)

        id_names = first_ds.get_identifiers_names()
        a_sql = child_sqls[0]
        b_sql = child_sqls[1]

        on_parts = [f"a.{quote_identifier(id_)} = b.{quote_identifier(id_)}" for id_ in id_names]
        on_clause = " AND ".join(on_parts) if on_parts else "1=1"

        if op == tokens.INTERSECT:
            return (
                f"SELECT a.* FROM ({a_sql}) AS a "
                f"WHERE EXISTS (SELECT 1 FROM ({b_sql}) AS b WHERE {on_clause})"
            )

        if op == tokens.SETDIFF:
            return (
                f"SELECT a.* FROM ({a_sql}) AS a "
                f"WHERE NOT EXISTS (SELECT 1 FROM ({b_sql}) AS b WHERE {on_clause})"
            )

        if op == tokens.SYMDIFF:
            second_ds = self._get_dataset_structure(node.children[1])
            second_ids = second_ds.get_identifiers_names() if second_ds else id_names
            on_parts_rev = [
                f"c.{quote_identifier(id_)} = d.{quote_identifier(id_)}" for id_ in second_ids
            ]
            on_clause_rev = " AND ".join(on_parts_rev) if on_parts_rev else "1=1"
            return (
                f"(SELECT a.* FROM ({a_sql}) AS a "
                f"WHERE NOT EXISTS (SELECT 1 FROM ({b_sql}) AS b WHERE {on_clause})) "
                f"UNION ALL "
                f"(SELECT c.* FROM ({b_sql}) AS c "
                f"WHERE NOT EXISTS (SELECT 1 FROM ({a_sql}) AS d WHERE {on_clause_rev}))"
            )

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

    def visit_JoinOp(self, node: AST.JoinOp) -> str:  # type: ignore[override]  # noqa: C901
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
                # Use dataset name as alias (mirrors interpreter convention)
                alias = ds.name if ds else chr(ord("a") + i)

            # Quote alias for SQL if it contains special characters
            sql_alias = quote_identifier(alias) if ("." in alias or " " in alias) else alias

            clause_info.append(
                {
                    "node": actual_node,
                    "ds": ds,
                    "table_src": table_src,
                    "alias": alias,
                    "sql_alias": sql_alias,
                }
            )

        if not clause_info:
            return ""

        first_ds = clause_info[0]["ds"]
        if first_ds is None:
            return ""

        first_ids = set(first_ds.get_identifiers_names())
        self._get_output_dataset()

        explicit_using: Optional[List[str]] = None
        if node.using:
            explicit_using = list(node.using)

        # Compute pairwise join keys for each secondary dataset.
        # When explicit using is given, all secondary datasets use the same
        # keys.  Otherwise, each secondary dataset is joined on the identifiers
        # it shares with the accumulated result (mirroring the interpreter).
        accumulated_ids = set(first_ids)
        pairwise_keys: List[List[str]] = []
        for info in clause_info[1:]:
            if explicit_using is not None:
                pairwise_keys.append(list(explicit_using))
            else:
                ds_ids = set(info["ds"].get_identifiers_names()) if info["ds"] else set()
                common = sorted(accumulated_ids & ds_ids)
                pairwise_keys.append(common)
                # Accumulate identifiers from this dataset for the next pairwise join
                accumulated_ids |= ds_ids

        # Flatten all join keys for the purpose of determining which components
        # are treated as identifiers (not aliased as duplicates)
        all_join_ids: Set[str] = set()
        for keys in pairwise_keys:
            all_join_ids.update(keys)
        # Also include all identifiers from all datasets (they won't be aliased)
        for info in clause_info:
            if info["ds"]:
                for comp_name, comp in info["ds"].components.items():
                    if comp.role == Role.IDENTIFIER:
                        all_join_ids.add(comp_name)

        # Detect duplicate non-identifier component names across datasets
        comp_count: Dict[str, int] = {}
        for info in clause_info:
            if info["ds"]:
                for comp_name, _comp in info["ds"].components.items():
                    if comp_name not in all_join_ids:
                        comp_count[comp_name] = comp_count.get(comp_name, 0) + 1

        duplicate_comps = {name for name, cnt in comp_count.items() if cnt >= 2}
        is_cross = join_type == "CROSS"
        is_full = join_type == "FULL"

        first_sql_alias = clause_info[0]["sql_alias"]
        builder = SQLBuilder()

        # Build columns, aliasing duplicates with "alias#comp" convention
        cols: List[str] = []
        self._join_alias_map = {}
        seen_identifiers: set[str] = set()

        for info in clause_info:
            if not info["ds"]:
                continue
            sa = info["sql_alias"]
            for comp_name, comp in info["ds"].components.items():
                is_join_id = (
                    comp.role == Role.IDENTIFIER and not is_cross
                ) or comp_name in all_join_ids
                if is_join_id:
                    if comp_name not in seen_identifiers:
                        seen_identifiers.add(comp_name)
                        if is_full and comp_name in all_join_ids:
                            # For FULL JOIN identifiers, use COALESCE to pick
                            # the non-NULL value from either side.
                            coalesce_parts = [
                                f"{ci['sql_alias']}.{quote_identifier(comp_name)}"
                                for ci in clause_info
                                if ci["ds"] and comp_name in ci["ds"].components
                            ]
                            cols.append(
                                f"COALESCE({', '.join(coalesce_parts)})"
                                f" AS {quote_identifier(comp_name)}"
                            )
                        else:
                            cols.append(f"{sa}.{quote_identifier(comp_name)}")
                elif comp_name in duplicate_comps:
                    # Duplicate non-identifier: alias with "alias#comp" convention
                    qualified_name = f"{info['alias']}#{comp_name}"
                    cols.append(
                        f"{sa}.{quote_identifier(comp_name)} AS {quote_identifier(qualified_name)}"
                    )
                    self._join_alias_map[qualified_name] = qualified_name
                else:
                    cols.append(f"{sa}.{quote_identifier(comp_name)}")

        if not cols:
            builder.select_all()
        else:
            builder.select(*cols)

        builder.from_table(clause_info[0]["table_src"], first_sql_alias)

        for idx, info in enumerate(clause_info[1:]):
            join_keys = pairwise_keys[idx]
            if is_cross:
                builder.cross_join(info["table_src"], info["sql_alias"])
            else:
                on_parts = []
                for id_ in join_keys:
                    if id_ not in (info["ds"].components if info["ds"] else {}):
                        continue
                    # Find which preceding dataset alias has this identifier
                    # (for multi-dataset joins where identifiers come from
                    # different source datasets)
                    left_alias = first_sql_alias
                    for prev_info in clause_info[: idx + 1]:
                        if prev_info["ds"] and id_ in prev_info["ds"].components:
                            left_alias = prev_info["sql_alias"]
                            break
                    on_parts.append(
                        f"{left_alias}.{quote_identifier(id_)} = "
                        f"{info['sql_alias']}.{quote_identifier(id_)}"
                    )
                on_clause = " AND ".join(on_parts) if on_parts else "1=1"
                builder.join(
                    info["table_src"],
                    info["sql_alias"],
                    on=on_clause,
                    join_type=join_type,
                )

        return builder.build()

    # =========================================================================
    # Time aggregation visitor
    # =========================================================================

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> str:  # type: ignore[override]
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

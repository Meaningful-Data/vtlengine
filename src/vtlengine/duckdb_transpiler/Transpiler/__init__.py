"""
SQL Transpiler for VTL AST.

Converts VTL AST nodes into DuckDB SQL queries using the visitor pattern.
Each top-level Assignment produces one SQL SELECT query. Queries are executed
sequentially, with results registered as tables for subsequent queries.
"""

import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar import tokens
from vtlengine.DataTypes import COMP_NAME_MAPPING, Boolean, Date, Duration, TimePeriod
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
from vtlengine.Model import Component, Dataset, ExternalRoutine, Role, Scalar, ValueDomain

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

# TimePeriod-specific SQL for extraction operators (struct-based)
_TP_EXTRACTION_MAP: Dict[str, str] = {
    tokens.YEAR: "CAST(vtl_period_parse({0}).year AS BIGINT)",
    tokens.MONTH: "vtl_tp_getmonth(vtl_period_parse({0}))",
    tokens.DAYOFMONTH: "vtl_tp_dayofmonth(vtl_period_parse({0}))",
    tokens.DAYOFYEAR: "vtl_tp_dayofyear(vtl_period_parse({0}))",
}

# Mapping from VTL ordering operators to vtl_period_* comparison macros.
# Equality (=, <>) operates on VARCHAR directly — no macros needed.
_PERIOD_COMPARISON_MACROS: Dict[str, str] = {
    tokens.GT: "vtl_period_gt",
    tokens.GTE: "vtl_period_ge",
    tokens.LT: "vtl_period_lt",
    tokens.LTE: "vtl_period_le",
}

# Duration comparison operators that need vtl_duration_to_int for magnitude ordering.
_DURATION_COMPARISON_OPS: frozenset[str] = frozenset(
    {tokens.GT, tokens.GTE, tokens.LT, tokens.LTE, tokens.EQ, tokens.NEQ}
)

def _is_date_timeperiod_pair(left_comp: Component, right_comp: Component) -> bool:
    """Check if two components form a Date↔TimePeriod cross-type pair."""
    types = {left_comp.data_type, right_comp.data_type}
    return types == {Date, TimePeriod}


def _date_tp_compare_expr(
    left_ref: str,
    right_ref: str,
    left_comp: Component,
    right_comp: Component,
    op: str,
) -> str:
    """Build SQL expression for Date vs TimePeriod comparison via TimeInterval promotion."""
    # Convert each side to vtl_time_interval struct
    if left_comp.data_type == Date:
        left_interval = (
            f"{{'date1': CAST({left_ref} AS DATE),"
            f" 'date2': CAST({left_ref} AS DATE)}}::vtl_time_interval"
        )
        parsed = f"vtl_period_parse({right_ref})"
        right_interval = (
            f"{{'date1': vtl_tp_start_date({parsed}),"
            f" 'date2': vtl_tp_end_date({parsed})}}::vtl_time_interval"
        )
    else:
        parsed = f"vtl_period_parse({left_ref})"
        left_interval = (
            f"{{'date1': vtl_tp_start_date({parsed}),"
            f" 'date2': vtl_tp_end_date({parsed})}}::vtl_time_interval"
        )
        right_interval = (
            f"{{'date1': CAST({right_ref} AS DATE),"
            f" 'date2': CAST({right_ref} AS DATE)}}::vtl_time_interval"
        )
    return registry.binary.generate(op, left_interval, right_interval)


# String operators that require VARCHAR input — Boolean measures must be cast first.
_STRING_UNARY_OPS: frozenset[str] = frozenset(
    {
        tokens.UCASE,
        tokens.LCASE,
        tokens.LEN,
        tokens.TRIM,
        tokens.LTRIM,
        tokens.RTRIM,
    }
)
_STRING_PARAM_OPS: frozenset[str] = frozenset(
    {
        tokens.SUBSTR,
        tokens.REPLACE,
        tokens.INSTR,
    }
)


def _bool_to_str(col_ref: str) -> str:
    """Wrap a Boolean column reference with a cast that matches Python's str(bool)."""
    return f"CASE WHEN {col_ref} IS NULL THEN NULL WHEN {col_ref} THEN 'True' ELSE 'False' END"


@dataclass
class _ParsedHRRule:
    """Parsed components of a hierarchical rule (shared across check/hierarchy methods)."""

    has_when: bool
    when_node: Any  # AST node for the WHEN condition, or None
    comparison_node: Any  # AST node for the comparison (left = right)
    left_code_item: str  # Left-side code item name
    right_expr_node: AST.AST  # Right-side expression AST
    right_code_items: List[str]  # All code item names in the right-side expression


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
    _column_prefix: Optional[str] = field(default=None, init=False)

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

    # Hierarchical ruleset definitions
    _hrs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize available tables."""
        self.datasets = {**self.input_datasets, **self.output_datasets}
        self.scalars = {**self.input_scalars, **self.output_scalars}
        self.available_tables = dict(self.datasets)

    # =========================================================================
    # Helper methods
    # =========================================================================

    @contextmanager
    def _clause_scope(
        self,
        ds: Optional[Dataset] = None,
        prefix: Optional[str] = None,
    ) -> Generator[None, None, None]:
        """Save/restore clause state (_in_clause, _current_dataset, _column_prefix).

        Usage::

            with self._clause_scope(ds):
                expr_sql = self.visit(node)
        """
        old_in_clause = self._in_clause
        old_current_ds = self._current_dataset
        old_prefix = self._column_prefix
        self._in_clause = True
        self._current_dataset = ds
        self._column_prefix = prefix
        try:
            yield
        finally:
            self._in_clause = old_in_clause
            self._current_dataset = old_current_ds
            self._column_prefix = old_prefix

    def _resolve_clause_dataset(
        self, node: AST.RegularAggregation
    ) -> Optional[Tuple[Dataset, str]]:
        """Resolve the dataset and SQL source for a clause node.

        Returns ``(dataset, table_src)`` or ``None`` when the clause has
        no dataset or the dataset structure cannot be resolved.
        """
        if not node.dataset:
            return None
        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)
        if ds is None:
            return None
        return ds, table_src

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
            elif isinstance(child, AST.HRuleset):
                self._visit_HRuleset(child)
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

    visit_PersistentAssignment = visit_Assignment

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
        if not rule_names:
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

        rule_node = rule.rule
        has_when = (
            isinstance(  # type: ignore[redundant-expr]
                rule_node, AST.HRBinOp
            )
            and rule_node.op == "when"
        )
        if has_when:
            when_cond_sql: Optional[str] = self._visit_dp_expr(rule_node.left, signature)
            then_expr_sql = self._visit_dp_expr(rule_node.right, signature)
        else:
            when_cond_sql = None
            then_expr_sql = self._visit_dp_expr(rule_node, signature)

        self._dp_signature = None

        # Common parts — use typed NULLs for DuckDB type inference
        if rule.erCode:
            escaped_ec = rule.erCode.replace("'", "''")
            ec_sql = f"'{escaped_ec}'"
        else:
            ec_sql = "CAST(NULL AS VARCHAR)"
        el_sql = self._error_level_sql(rule.erLevel)
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
            f"CASE WHEN ({when_cond_sql}) THEN ({then_expr_sql})"
            f" WHEN NOT ({when_cond_sql}) THEN TRUE ELSE NULL END"
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
        # Binary nodes (HRBinOp and BinOp share left/right structure)
        if isinstance(node, (AST.HRBinOp, AST.BinOp)):
            left_sql = self._visit_dp_expr(node.left, signature)
            right_sql = self._visit_dp_expr(node.right, signature)
            if isinstance(node, AST.HRBinOp) and node.op == "when":
                return f"CASE WHEN ({left_sql}) THEN ({right_sql}) ELSE TRUE END"
            return self._dp_binary_sql(node.op, left_sql, right_sql)
        # Unary nodes (HRUnOp and UnaryOp share operand structure)
        if isinstance(node, (AST.HRUnOp, AST.UnaryOp)):
            operand_sql = self._visit_dp_expr(node.operand, signature)
            return self._dp_unary_sql(node.op, operand_sql)
        if isinstance(node, (AST.DefIdentifier, AST.VarID)):
            col_name = signature.get(node.value, node.value)
            return quote_identifier(col_name)
        if isinstance(node, AST.Constant):
            return self._to_sql_literal(node.value)
        if isinstance(node, AST.If):
            cond_sql = self._visit_dp_expr(node.condition, signature)
            then_sql = self._visit_dp_expr(node.thenOp, signature)
            else_sql = self._visit_dp_expr(node.elseOp, signature)
            return (
                f"CASE WHEN ({cond_sql}) THEN CAST(({then_sql}) AS BOOLEAN)"
                f" ELSE CAST(({else_sql}) AS BOOLEAN) END"
            )
        # Fallback: use the regular transpiler visitor, saving/restoring DP context
        saved_sig = self._dp_signature
        self._dp_signature = signature
        result = self.visit(node)
        self._dp_signature = saved_sig
        return result

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
    # Hierarchical Ruleset definition and check_hierarchy
    # =========================================================================

    def _visit_HRuleset(self, node: AST.HRuleset) -> None:
        """Register a hierarchical ruleset definition."""
        # Auto-number unnamed rules (same logic as interpreter)
        rule_names = [r.name for r in node.rules if r.name is not None]
        if not rule_names:
            for i, rule in enumerate(node.rules):
                rule.name = str(i + 1)

        # Extract condition components and signature
        cond_comp: List[str] = []
        signature_value: str
        if isinstance(node.element, list):
            cond_comp = [x.value for x in node.element[:-1]]
            signature_value = node.element[-1].value
        else:
            signature_value = node.element.value

        self._hrs[node.name] = {
            "rules": node.rules,
            "signature": signature_value,
            "condition": cond_comp,
            "signature_type": node.signature_type,
            "node": node,
        }

    def visit_HROperation(self, node: AST.HROperation) -> str:  # type: ignore[override]
        """Generate SQL for hierarchy or check_hierarchy operator."""
        hr_name = node.ruleset_name
        hr_info = self._hrs[hr_name]

        # Resolve dataset
        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)
        if ds is None:
            raise ValueError("Cannot resolve dataset for hierarchy operation")

        self._get_output_dataset()

        # Get rule component name: for valuedomain rulesets, use the actual column
        # from the invocation (node.rule_component), not the valuedomain name
        if hr_info["signature_type"] == "valuedomain" and node.rule_component is not None:
            component: str = node.rule_component.value  # type: ignore[attr-defined]
        else:
            component = hr_info["signature"]

        # Condition mapping: ruleset param -> dataset column (raw names, not quoted)
        cond_mapping: Dict[str, str] = {}
        if node.conditions and hr_info["condition"]:
            for i, cond_node in enumerate(node.conditions):
                param_name = hr_info["condition"][i]
                actual_col = cond_node.value  # type: ignore[attr-defined]
                cond_mapping[param_name] = actual_col

        if node.op == tokens.HIERARCHY:
            mode = node.validation_mode.value if node.validation_mode else "non_null"
            input_mode = node.input_mode.value if node.input_mode else "rule"
            output = node.output.value if node.output else "computed"
            # Filter to EQ/WHEN-EQ rules only
            rules = [r for r in hr_info["rules"] if self._is_hr_eq_rule(r)]
            return self._build_hierarchy_sql(
                table_src=table_src,
                ds=ds,
                rules=rules,
                rule_comp=component,
                mode=mode,
                input_mode=input_mode,
                output=output,
                cond_mapping=cond_mapping,
            )
        else:  # check_hierarchy
            mode = node.validation_mode.value if node.validation_mode else "non_null"
            output = node.output.value if node.output else "invalid"
            return self._build_check_hierarchy_sql(
                table_src=table_src,
                ds=ds,
                rules=hr_info["rules"],
                rule_comp=component,
                mode=mode,
                output=output,
                cond_mapping=cond_mapping,
            )

    @staticmethod
    def _error_level_sql(er_level: Any) -> str:
        """Convert an errorlevel value to a SQL literal (numeric or string)."""
        if er_level is None:
            return "CAST(NULL AS VARCHAR)"
        try:
            return str(float(er_level))
        except (ValueError, TypeError):
            escaped = str(er_level).replace("'", "''")
            return f"'{escaped}'"

    @staticmethod
    def _is_hr_eq_rule(rule: AST.HRule) -> bool:
        """Check if a hierarchical rule is an EQ rule (or WHEN-EQ)."""
        rule_node = rule.rule
        if not isinstance(rule_node, AST.HRBinOp):
            return False
        if rule_node.op == "when":
            right = rule_node.right
            return isinstance(right, AST.HRBinOp) and right.op == "="
        return rule_node.op == "="

    def _parse_hr_rule(self, rule: AST.HRule) -> _ParsedHRRule:
        """Parse a hierarchical rule into its constituent parts."""
        rule_node: Any = rule.rule
        has_when = isinstance(rule_node, AST.HRBinOp) and rule_node.op == "when"
        if has_when:
            when_node = rule_node.left
            comparison_node = rule_node.right
        else:
            when_node = None
            comparison_node = rule_node
        return _ParsedHRRule(
            has_when=has_when,
            when_node=when_node,
            comparison_node=comparison_node,
            left_code_item=comparison_node.left.value,
            right_expr_node=comparison_node.right,
            right_code_items=self._collect_hr_code_items(comparison_node.right)[0],
        )

    def _collect_all_hr_items(
        self,
        rules: list,  # type: ignore[type-arg]
        cond_mapping: Dict[str, str],
    ) -> Tuple[List[str], Dict[str, str]]:
        """Collect and deduplicate all code items and their conditions across rules.

        Returns (unique_items, code_item_conditions).
        """
        all_items: List[str] = []
        all_conds: Dict[str, str] = {}
        for rule in rules:
            parsed = self._parse_hr_rule(rule)
            all_items.append(parsed.left_code_item)
            all_items.extend(parsed.right_code_items)
            # Collect right-side conditions (e.g. [Time >= cast("1958-01-01", date)])
            rc = getattr(parsed.comparison_node.left, "_right_condition", None)
            if rc is not None:
                all_conds[parsed.left_code_item] = self._build_hr_when_sql(rc, cond_mapping)
            _, right_conds = self._collect_hr_code_items(parsed.right_expr_node, cond_mapping)
            all_conds.update(right_conds)
        # Deduplicate preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for ci in all_items:
            if ci not in seen:
                seen.add(ci)
                unique.append(ci)
        return unique, all_conds

    def _prepare_hr_pivot(
        self,
        table_src: str,
        ds: Dataset,
        rules: list,  # type: ignore[type-arg]
        rule_comp: str,
        cond_mapping: Dict[str, str],
    ) -> Tuple[str, str, List[str], List[str], Dict[str, str]]:
        """Shared setup for hierarchy / check_hierarchy: returns
        (pivot_cte, measure_name, other_ids, unique_items, item_conds).
        """
        measure_name = ds.get_measures_names()[0]
        other_ids = [n for n in ds.get_identifiers_names() if n != rule_comp]
        unique_items, item_conds = self._collect_all_hr_items(rules, cond_mapping)

        pivot_cte = self._build_hr_pivot_cte(
            table_src=table_src,
            code_items=unique_items,
            rule_comp=rule_comp,
            measure=measure_name,
            other_ids=other_ids,
            cond_mapping=cond_mapping,
            code_item_conditions=item_conds,
        )
        return pivot_cte, measure_name, other_ids, unique_items, item_conds

    def _build_check_hierarchy_sql(
        self,
        table_src: str,
        ds: Dataset,
        rules: list,  # type: ignore[type-arg]
        rule_comp: str,
        mode: str,
        output: str,
        cond_mapping: Dict[str, str],
    ) -> str:
        """Generate SQL for check_hierarchy using pivot CTE."""
        if not rules:
            out_ds = self._get_output_dataset()
            cols = [quote_identifier(c) for c in (out_ds.components if out_ds else ds.components)]
            return f"SELECT {', '.join(cols)} FROM {table_src} WHERE 1=0"

        pivot_cte, measure_name, other_ids, _, _ = self._prepare_hr_pivot(
            table_src, ds, rules, rule_comp, cond_mapping
        )

        rule_queries = [
            self._build_check_hr_rule_select(
                rule=rule,
                other_ids=other_ids,
                rule_comp=rule_comp,
                measure=measure_name,
                mode=mode,
                output=output,
                cond_mapping=cond_mapping,
            )
            for rule in rules
        ]
        return f"WITH {pivot_cte}\n" + " UNION ALL ".join(rule_queries)

    def _collect_hr_code_items(
        self,
        node: AST.AST,
        cond_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Extract all code item names and their right-side conditions from an HR expression.

        When *cond_mapping* is provided, also resolves ``_right_condition``
        attributes on DefIdentifier nodes into SQL WHERE fragments.
        """
        if isinstance(node, AST.DefIdentifier):
            conds: Dict[str, str] = {}
            if cond_mapping is not None:
                rc = getattr(node, "_right_condition", None)
                if rc is not None:
                    conds[node.value] = self._build_hr_when_sql(rc, cond_mapping)
            return [node.value], conds
        if isinstance(node, AST.HRBinOp):
            li, lc = self._collect_hr_code_items(node.left, cond_mapping)
            ri, rc = self._collect_hr_code_items(node.right, cond_mapping)
            lc.update(rc)
            return li + ri, lc
        if isinstance(node, AST.HRUnOp):
            return self._collect_hr_code_items(node.operand, cond_mapping)
        return [], {}

    def _build_hr_value_expr(self, code_item: str, mode: str) -> str:
        """Generate the value expression for a code item from pivot columns, per mode."""
        val_col = f"_val_{code_item}"
        has_col = f"_has_{code_item}"
        if mode in ("always_zero", "non_zero", "partial_zero"):
            return f"CASE WHEN {has_col} = 0 THEN 0 ELSE {val_col} END"
        return val_col

    def _build_hr_expr_sql(self, node: AST.AST, mode: str) -> str:
        """Generate SQL for a hierarchical rule arithmetic expression using pivot columns."""
        if isinstance(node, AST.DefIdentifier):
            return self._build_hr_value_expr(node.value, mode)
        if isinstance(node, AST.HRBinOp):
            left_sql = self._build_hr_expr_sql(node.left, mode)
            right_sql = self._build_hr_expr_sql(node.right, mode)
            return f"({left_sql} {node.op} {right_sql})"
        if isinstance(node, AST.HRUnOp):
            operand_sql = self._build_hr_expr_sql(node.operand, mode)
            return f"({node.op}{operand_sql})"
        raise ValueError(f"Unexpected node type in HR expression: {type(node).__name__}")

    def _build_hr_pivot_cte(
        self,
        table_src: str,
        code_items: List[str],
        rule_comp: str,
        measure: str,
        other_ids: List[str],
        cond_mapping: Dict[str, str],
        code_item_conditions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate the shared pivot CTE for hierarchy operations."""
        qrc = quote_identifier(rule_comp)
        qm = quote_identifier(measure)

        group_cols = [quote_identifier(c) for c in other_ids]
        group_cols.extend(quote_identifier(v) for v in cond_mapping.values())

        select_parts = list(group_cols)
        for ci in code_items:
            ci_cond = ""
            if code_item_conditions and ci in code_item_conditions:
                ci_cond = f" AND {code_item_conditions[ci]}"
            select_parts.append(
                f"MAX(CASE WHEN {qrc} = '{ci}'{ci_cond} THEN {qm} END) AS _val_{ci}"
            )
            select_parts.append(
                f"MAX(CASE WHEN {qrc} = '{ci}'{ci_cond} THEN 1 ELSE 0 END) AS _has_{ci}"
            )

        in_list = ", ".join(f"'{ci}'" for ci in code_items)
        group_by = f"GROUP BY {', '.join(group_cols)}" if group_cols else ""

        return (
            f"_pivot AS (\n"
            f"  SELECT {', '.join(select_parts)}\n"
            f"  FROM {table_src}\n"
            f"  WHERE {qrc} IN ({in_list})\n"
            f"  {group_by}\n"
            f")"
        )

    def _build_check_hr_rule_select(
        self,
        rule: AST.HRule,
        other_ids: List[str],
        rule_comp: str,
        measure: str,
        mode: str,
        output: str,
        cond_mapping: Dict[str, str],
    ) -> str:
        """Generate a SELECT for a single check_hierarchy rule from the pivot CTE."""
        parsed = self._parse_hr_rule(rule)
        rule_name = rule.name or ""

        # Build value expressions from pivot columns
        l_val = self._build_hr_value_expr(parsed.left_code_item, mode)
        r_val = self._build_hr_expr_sql(parsed.right_expr_node, mode)

        # Comparison and imbalance expressions
        comp_op: str = parsed.comparison_node.op
        bool_expr = f"({l_val} {comp_op} {r_val})"
        imbalance_expr = f"({l_val} - {r_val})"

        when_sql: Optional[str] = None
        if parsed.has_when:
            when_sql = self._build_hr_when_sql(parsed.when_node, cond_mapping)
            bool_expr = f"CASE WHEN NOT ({when_sql}) THEN TRUE ELSE {bool_expr} END"
            imbalance_expr = (
                f"CASE WHEN NOT ({when_sql}) THEN CAST(NULL AS DOUBLE) ELSE {imbalance_expr} END"
            )

        # Errorcode / errorlevel
        if rule.erCode:
            ec_sql = f"'{rule.erCode.replace(chr(39), chr(39) * 2)}'"
        else:
            ec_sql = "CAST(NULL AS VARCHAR)"
        el_sql = self._error_level_sql(rule.erLevel)
        el_null = self._error_level_null_sql(rule.erLevel)

        # SELECT columns
        q_rc = quote_identifier(rule_comp)
        q_m = quote_identifier(measure)
        select_parts: List[str] = [quote_identifier(c) for c in other_ids]
        select_parts.append(f"'{parsed.left_code_item}' AS {q_rc}")

        if output != "all":
            select_parts.append(f"{l_val} AS {q_m}")
        if output != "invalid":
            select_parts.append(f"{bool_expr} AS {quote_identifier('bool_var')}")

        select_parts.append(f"{imbalance_expr} AS {quote_identifier('imbalance')}")
        select_parts.append(f"'{rule_name}' AS {quote_identifier('ruleid')}")

        if output == "invalid":
            select_parts.append(f"{ec_sql} AS {quote_identifier('errorcode')}")
            select_parts.append(f"{el_sql} AS {quote_identifier('errorlevel')}")
        else:
            select_parts.append(
                f"CASE WHEN {bool_expr} IS NOT FALSE THEN CAST(NULL AS VARCHAR) "
                f"ELSE {ec_sql} END AS {quote_identifier('errorcode')}"
            )
            select_parts.append(
                f"CASE WHEN {bool_expr} IS NOT FALSE THEN {el_null} "
                f"ELSE {el_sql} END AS {quote_identifier('errorlevel')}"
            )

        # WHERE clause
        where_parts: List[str] = []
        if output == "invalid":
            if when_sql is not None:
                where_parts.append(f"({when_sql})")
            where_parts.append(f"({bool_expr}) = FALSE")

        where_parts.extend(
            self._build_hr_mode_filter(
                mode=mode,
                left_code_item=parsed.left_code_item,
                right_code_items=parsed.right_code_items,
                left_val_expr=l_val,
                right_val_expr=r_val,
                is_hierarchy=False,
            )
        )

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        return f"SELECT {', '.join(select_parts)} FROM _pivot{where_clause}"

    @staticmethod
    def _error_level_null_sql(er_level: Any) -> str:
        """Return the appropriate typed NULL for errorlevel columns."""
        if er_level is not None:
            try:
                float(er_level)
            except (ValueError, TypeError):
                return "CAST(NULL AS VARCHAR)"
        return "CAST(NULL AS DOUBLE)"

    def _build_hierarchy_sql(
        self,
        table_src: str,
        ds: Dataset,
        rules: list,  # type: ignore[type-arg]
        rule_comp: str,
        mode: str,
        input_mode: str,
        output: str,
        cond_mapping: Dict[str, str],
    ) -> str:
        """Generate SQL for hierarchy operator using pivot CTE."""
        if not rules:
            cols = [quote_identifier(c) for c in ds.get_components_names()]
            return f"SELECT {', '.join(cols)} FROM {table_src}"

        pivot_cte, measure_name, other_ids, unique_items, _ = self._prepare_hr_pivot(
            table_src, ds, rules, rule_comp, cond_mapping
        )

        return self._build_hierarchy_cte_chain(
            pivot_cte=pivot_cte,
            table_src=table_src,
            rules=rules,
            rule_comp=rule_comp,
            measure=measure_name,
            other_ids=other_ids,
            mode=mode,
            input_mode=input_mode,
            output=output,
            cond_mapping=cond_mapping,
            ds=ds,
            unique_items=unique_items,
        )

    def _build_hierarchy_cte_chain(
        self,
        pivot_cte: str,
        table_src: str,
        rules: list,  # type: ignore[type-arg]
        rule_comp: str,
        measure: str,
        other_ids: List[str],
        mode: str,
        input_mode: str,
        output: str,
        cond_mapping: Dict[str, str],
        ds: Dataset,
        unique_items: List[str],
    ) -> str:
        """Hierarchy SQL using CTE chain (rule/rule_priority/dataset modes)."""
        cte_parts: List[str] = [pivot_cte]
        rule_result_refs: List[Tuple[str, str]] = []
        current_pivot = "_pivot"

        join_keys = [quote_identifier(c) for c in other_ids]
        join_keys.extend(quote_identifier(v) for v in cond_mapping.values())

        for i, rule in enumerate(rules):
            parsed = self._parse_hr_rule(rule)

            rule_cte_name = f"_rule_{i}"
            rule_select = self._build_hierarchy_rule_cte(
                parsed=parsed,
                pivot_ref=current_pivot,
                other_ids=other_ids,
                mode=mode,
                cond_mapping=cond_mapping,
            )
            cte_parts.append(f"{rule_cte_name} AS (\n{rule_select}\n)")
            rule_result_refs.append((rule_cte_name, parsed.left_code_item))

            next_pivot = f"_pivot_{i}"
            pivot_update = self._build_hierarchy_pivot_update(
                prev_pivot=current_pivot,
                rule_cte=rule_cte_name,
                left_code_item=parsed.left_code_item,
                join_keys=join_keys,
                input_mode=input_mode,
                unique_items=unique_items,
            )
            cte_parts.append(f"{next_pivot} AS (\n{pivot_update}\n)")
            current_pivot = next_pivot

        # Final SELECT: collect all computed results
        final_selects: List[str] = []
        q_rc = quote_identifier(rule_comp)
        q_m = quote_identifier(measure)
        for rule_cte, left_ci in rule_result_refs:
            cols = [quote_identifier(c) for c in other_ids]
            cols.append(f"'{left_ci}' AS {q_rc}")
            cols.append(f"_computed AS {q_m}")

            result_filter: List[str] = []
            if mode == "non_null":
                result_filter.append("_computed IS NOT NULL")
            elif mode == "non_zero":
                result_filter.append("(_computed IS NULL OR _computed != 0)")

            where = f" WHERE {' AND '.join(result_filter)}" if result_filter else ""
            final_selects.append(f"SELECT {', '.join(cols)} FROM {rule_cte}{where}")

        computed_sql = " UNION ALL ".join(final_selects)

        if output == "computed":
            return f"WITH {','.join(cte_parts)}\n{computed_sql}"

        # output == "all"
        id_cols = [quote_identifier(c) for c in ds.get_identifiers_names()]
        all_cols = [quote_identifier(c) for c in ds.get_components_names()]
        cte_parts.append(f"_computed AS (\n{computed_sql}\n)")
        return (
            f"WITH {','.join(cte_parts)},\n"
            f"_combined AS (\n"
            f"  SELECT {', '.join(all_cols)}, 0 AS _src FROM {table_src}\n"
            f"  UNION ALL\n"
            f"  SELECT {', '.join(all_cols)}, 1 AS _src FROM _computed\n"
            f")\n"
            f"SELECT {', '.join(all_cols)} FROM (\n"
            f"  SELECT *, ROW_NUMBER() OVER ("
            f"PARTITION BY {', '.join(id_cols)} ORDER BY _src DESC) AS _rn\n"
            f"  FROM _combined\n"
            f") WHERE _rn = 1"
        )

    def _build_hierarchy_rule_cte(
        self,
        parsed: "_ParsedHRRule",
        pivot_ref: str,
        other_ids: List[str],
        mode: str,
        cond_mapping: Dict[str, str],
    ) -> str:
        """Generate SELECT for _rule_N CTE in hierarchy CTE chain."""
        r_val = self._build_hr_expr_sql(parsed.right_expr_node, mode)
        computed_expr = r_val
        if parsed.has_when:
            when_sql = self._build_hr_when_sql(parsed.when_node, cond_mapping)
            computed_expr = f"CASE WHEN {when_sql} THEN {computed_expr} ELSE NULL END"

        select_parts = [quote_identifier(c) for c in other_ids]
        select_parts.extend(quote_identifier(v) for v in cond_mapping.values())
        select_parts.append(f"{computed_expr} AS _computed")

        where_parts = self._build_hr_mode_filter(
            mode=mode,
            left_code_item=parsed.left_code_item,
            right_code_items=parsed.right_code_items,
            left_val_expr=self._build_hr_value_expr(parsed.left_code_item, mode),
            right_val_expr=r_val,
            is_hierarchy=True,
        )
        right_presence = [f"_has_{ci} = 1" for ci in parsed.right_code_items]
        if right_presence:
            where_parts.append(f"({' OR '.join(right_presence)})")

        where_clause = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        return f"  SELECT {', '.join(select_parts)} FROM {pivot_ref}{where_clause}"

    def _build_hierarchy_pivot_update(
        self,
        prev_pivot: str,
        rule_cte: str,
        left_code_item: str,
        join_keys: List[str],
        input_mode: str,
        unique_items: List[str],
    ) -> str:
        """Generate _pivot_N CTE that updates pivot with a rule's computed value."""
        val_col = f"_val_{left_code_item}"
        has_col = f"_has_{left_code_item}"

        other_val_has = []
        for ci in unique_items:
            if ci != left_code_item:
                other_val_has.append(f"p._val_{ci}")
                other_val_has.append(f"p._has_{ci}")

        key_cols = [f"p.{k}" for k in join_keys]
        first_key = join_keys[0] if join_keys else "_computed"

        if input_mode == "rule_priority":
            val_expr = (
                f"CASE WHEN r._computed IS NOT NULL THEN r._computed "
                f"ELSE p.{val_col} END AS {val_col}"
            )
        else:
            val_expr = (
                f"CASE WHEN r.{first_key} IS NOT NULL "
                f"THEN r._computed ELSE p.{val_col} END AS {val_col}"
            )
        has_expr = f"CASE WHEN r.{first_key} IS NOT NULL THEN 1 ELSE p.{has_col} END AS {has_col}"

        all_select = key_cols + other_val_has + [val_expr, has_expr]
        using_clause = ", ".join(join_keys) if join_keys else "1=1"

        return (
            f"  SELECT {', '.join(all_select)}\n"
            f"  FROM {prev_pivot} p\n"
            f"  LEFT JOIN {rule_cte} r USING ({using_clause})"
        )

    def _build_hr_mode_filter(
        self,
        mode: str,
        left_code_item: str,
        right_code_items: List[str],
        left_val_expr: str,
        right_val_expr: str,
        is_hierarchy: bool,
    ) -> List[str]:
        """Generate WHERE filter clauses for the validation mode using pivot columns."""
        all_items = [left_code_item] + right_code_items
        filters: List[str] = []

        if mode == "non_null":
            items = right_code_items if is_hierarchy else all_items
            for ci in items:
                filters.append(f"_val_{ci} IS NOT NULL")

        elif mode == "non_zero":
            if is_hierarchy:
                zero_checks = []
                for ci in right_code_items:
                    val = self._build_hr_value_expr(ci, mode)
                    zero_checks.append(f"({val} IS NOT NULL AND {val} = 0)")
                if zero_checks:
                    filters.append(f"NOT ({' AND '.join(zero_checks)})")
            else:
                filters.append(
                    f"NOT ("
                    f"({left_val_expr} IS NOT NULL AND {left_val_expr} = 0) AND "
                    f"({right_val_expr} IS NOT NULL AND {right_val_expr} = 0))"
                )

        elif mode in ("partial_null", "partial_zero"):
            items = right_code_items if is_hierarchy else all_items
            checks = [f"(_has_{ci} = 1 AND _val_{ci} IS NOT NULL)" for ci in items]
            if checks:
                filters.append(f"({' OR '.join(checks)})")

        elif mode in ("always_null", "always_zero"):
            presence = [f"_has_{ci} = 1" for ci in all_items]
            filters.append(f"({' OR '.join(presence)})")

        return filters

    def _build_hr_when_sql(self, node: AST.AST, cond_mapping: Dict[str, str]) -> str:
        """Generate SQL for a WHEN condition in a hierarchical rule."""
        if isinstance(node, (AST.HRBinOp, AST.BinOp)):
            left_sql = self._build_hr_when_sql(node.left, cond_mapping)
            right_sql = self._build_hr_when_sql(node.right, cond_mapping)
            sql_op = _DP_OP_MAP.get(node.op, node.op)
            return f"({left_sql} {sql_op} {right_sql})"
        if isinstance(node, (AST.DefIdentifier, AST.VarID)):
            col_name = cond_mapping.get(node.value, node.value)
            return quote_identifier(col_name)
        if isinstance(node, AST.Constant):
            return self._to_sql_literal(node.value)
        if isinstance(node, AST.HRUnOp):
            operand_sql = self._build_hr_when_sql(node.operand, cond_mapping)
            return f"({node.op}{operand_sql})"
        if isinstance(node, AST.UnaryOp):
            operand_sql = self._build_hr_when_sql(node.operand, cond_mapping)
            return f"({node.op}({operand_sql}))"
        if isinstance(node, AST.MulOp):
            children_sql = [self._build_hr_when_sql(c, cond_mapping) for c in node.children]
            if node.op.lower() == "between":
                return f"({children_sql[0]} BETWEEN {children_sql[1]} AND {children_sql[2]})"
            return f"{node.op}({', '.join(children_sql)})"
        # Fallback: delegate to the general visitor (handles ParamOp/cast, etc.)
        return self.visit(node)

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
        cast_bool_to_str: bool = False,
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
            cast_bool_to_str: When ``True``, Boolean measures are cast to
                              VARCHAR before being passed to *expr_fn* so that
                              DuckDB string functions receive the correct type.

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
                col_ref = quote_identifier(name)
                if cast_bool_to_str and comp.data_type == Boolean:
                    col_ref = _bool_to_str(col_ref)
                expr = expr_fn(col_ref)
                if output_name_override is not None:
                    out_name = output_name_override
                elif (
                    output_measure_names
                    and len(input_measures) == 1
                    and len(output_measure_names) == 1
                    and name == input_measures[0]
                    and name != output_measure_names[0]
                    and (
                        ds.name not in self.input_datasets
                        or name in self.input_datasets[ds.name].get_measures_names()
                    )
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
        left_sql_override: Optional[str] = None,
        left_ds_override: Optional[Dataset] = None,
    ) -> str:
        """Build SQL for dataset-dataset binary operation (requires JOIN).

        When ``left_sql_override`` / ``left_ds_override`` are provided, they
        are used instead of resolving the left node.  This allows iterative
        chaining without recursion.
        """
        left_ds = left_ds_override or self._get_dataset_structure(left_node)
        right_ds = self._get_dataset_structure(right_node)
        output_ds = self._get_output_dataset()

        if left_ds is None or right_ds is None:
            raise ValueError("Cannot resolve dataset structures for binary operation")

        left_src = left_sql_override or self._get_dataset_sql(left_node)
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

        # VTL: mono-measure datasets pair by position even if names differ
        paired_measures: List[Tuple[str, str]] = []
        if common_measures:
            paired_measures = [(m, m) for m in common_measures]
        elif len(left_measures) == 1 and len(right_measures) == 1:
            # When the output dataset has a single measure, inner visits rename
            # both sides to match the output name in the generated SQL.
            if output_measure_names and len(output_measure_names) == 1:
                out_m = output_measure_names[0]
                paired_measures = [(out_m, out_m)]
            else:
                paired_measures = [(left_measures[0], right_measures[0])]

        cols: List[str] = []
        for id_name in all_ids:
            if id_name in left_ids:
                cols.append(f"{alias_a}.{quote_identifier(id_name)}")
            else:
                cols.append(f"{alias_b}.{quote_identifier(id_name)}")

        for left_m, right_m in paired_measures:
            left_ref = f"{alias_a}.{quote_identifier(left_m)}"
            right_ref = f"{alias_b}.{quote_identifier(right_m)}"

            # Boolean→String promotion for concat
            if op == tokens.CONCAT:
                left_comp_c = left_ds.components.get(left_m)
                right_comp_c = right_ds.components.get(right_m)
                if left_comp_c and left_comp_c.data_type == Boolean:
                    left_ref = _bool_to_str(left_ref)
                if right_comp_c and right_comp_c.data_type == Boolean:
                    right_ref = _bool_to_str(right_ref)

            # TimePeriod ordering: use vtl_period_* macros with STRUCT comparison
            left_comp = left_ds.components.get(left_m)
            right_comp = right_ds.components.get(right_m)
            period_macro = _PERIOD_COMPARISON_MACROS.get(op)
            if left_comp and left_comp.data_type == TimePeriod and period_macro:
                expr = (
                    f"{period_macro}(vtl_period_parse({left_ref}), vtl_period_parse({right_ref}))"
                )
            # Duration ordering: use vtl_duration_to_int for magnitude ordering
            elif (
                left_comp
                and left_comp.data_type == Duration
                and op in _DURATION_COMPARISON_OPS
            ):
                left_int = f"vtl_duration_to_int({left_ref})"
                right_int = f"vtl_duration_to_int({right_ref})"
                expr = registry.binary.generate(op, left_int, right_int)
            # Date vs TimePeriod cross-type: promote both to vtl_time_interval
            elif left_comp and right_comp and _is_date_timeperiod_pair(left_comp, right_comp):
                expr = _date_tp_compare_expr(left_ref, right_ref, left_comp, right_comp, op)
            else:
                expr = registry.binary.generate(op, left_ref, right_ref)

            out_name = left_m
            if (
                output_measure_names
                and len(paired_measures) == 1
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
        period_macro = _PERIOD_COMPARISON_MACROS.get(op)

        # Check if any measure is TimePeriod for ordering comparisons
        has_time_period_measure = period_macro is not None and any(
            c.data_type == TimePeriod for c in ds.components.values() if c.role == Role.MEASURE
        )

        # Check if any measure is Duration for magnitude ordering
        has_duration_measure = op in _DURATION_COMPARISON_OPS and any(
            c.data_type == Duration for c in ds.components.values() if c.role == Role.MEASURE
        )

        def _bin_expr(col_ref: str) -> str:
            if has_time_period_measure:
                left = f"vtl_period_parse({col_ref})"
                right = f"vtl_period_parse({scalar_sql})"
                if ds_on_left:
                    return f"{period_macro}({left}, {right})"
                return f"{period_macro}({right}, {left})"
            if has_duration_measure:
                left = f"vtl_duration_to_int({col_ref})"
                right = f"vtl_duration_to_int({scalar_sql})"
                if ds_on_left:
                    return registry.binary.generate(op, left, right)
                return registry.binary.generate(op, right, left)
            if ds_on_left:
                return registry.binary.generate(op, col_ref, scalar_sql)
            return registry.binary.generate(op, scalar_sql, col_ref)

        return self._apply_to_measures(
            ds_node,
            _bin_expr,
            cast_bool_to_str=op == tokens.CONCAT,
        )

    # =========================================================================
    # Expression visitors
    # =========================================================================

    # Arithmetic ops that can form long left-associative chains.
    _ARITHMETIC_OPS = frozenset({"+", "-", "*", "/", "||"})

    def _is_chainable_ds_binop(self, node: AST.AST) -> bool:
        """Check if a node is a BinOp with an arithmetic op involving datasets."""
        if not isinstance(node, AST.BinOp):
            return False
        op = str(node.op).lower() if node.op else ""
        return op in self._ARITHMETIC_OPS and self._get_operand_type(node) == _DATASET

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

        if op == tokens.RANDOM:
            return self._visit_random_binop(node)

        if op == tokens.TIMESHIFT:
            return self._visit_timeshift(node)

        # Check operand types for dataset-level routing
        left_type = self._get_operand_type(node.left)
        right_type = self._get_operand_type(node.right)
        has_dataset = left_type == _DATASET or right_type == _DATASET

        if has_dataset:
            # in/not_in at dataset level: produce bool_var measure
            if op in (tokens.IN, tokens.NOT_IN) and left_type == _DATASET:
                collection_sql = self.visit(node.right)

                def _in_expr(col_ref: str) -> str:
                    if op == tokens.NOT_IN:
                        return f"({col_ref} NOT IN {collection_sql})"
                    return f"({col_ref} IN {collection_sql})"

                return self._apply_to_measures(node.left, _in_expr, output_name_override="bool_var")
            if op in self._ARITHMETIC_OPS and self._is_chainable_ds_binop(node.left):
                return self._visit_dataset_binary_chain(node)
            return self._visit_dataset_binary(node.left, node.right, op)

        # Scalar-scalar: use registry
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)

        # TimePeriod dispatch for datediff
        if op == tokens.DATEDIFF and (
            self._is_time_period_operand(node.left) or self._is_time_period_operand(node.right)
        ):
            return f"vtl_tp_datediff(vtl_period_parse({left_sql}), vtl_period_parse({right_sql}))"

        # Duration comparisons: use vtl_duration_to_int for magnitude ordering
        if op in _DURATION_COMPARISON_OPS and (
            self._is_duration_operand(node.left) or self._is_duration_operand(node.right)
        ):
            left_int = f"vtl_duration_to_int({left_sql})"
            right_int = f"vtl_duration_to_int({right_sql})"
            return registry.binary.generate(op, left_int, right_int)

        if registry.binary.is_registered(op):
            return registry.binary.generate(op, left_sql, right_sql)
        # Fallback for unregistered ops
        return f"{op.upper()}({left_sql}, {right_sql})"

    def _visit_dataset_binary_chain(self, node: AST.BinOp) -> str:
        """Iteratively fold a left-recursive chain of dataset binary operations."""
        # Flatten the left spine: collect (op, right_node) pairs.
        parts: list[tuple[str, AST.AST]] = []
        current: AST.AST = node
        while isinstance(current, AST.BinOp):
            bin_op = str(current.op).lower() if current.op else ""
            if bin_op not in self._ARITHMETIC_OPS:
                break
            if self._get_operand_type(current) != _DATASET:
                break
            parts.append((bin_op, current.right))
            current = current.left

        # ``current`` is the leftmost operand; ``parts`` is in reverse order.
        parts.reverse()

        # Resolve the leftmost operand's SQL and structure.
        result_sql = self._get_dataset_sql(current)
        result_ds = self._get_dataset_structure(current)

        # Track whether result_sql is a subquery (needs wrapping) or a table name.
        is_subquery = False

        # Fold: start with the leftmost operand and apply each (op, right) pair.
        for step_op, right_node in parts:
            right_type = self._get_operand_type(right_node)
            if right_type == _DATASET:
                left_src = f"({result_sql})" if is_subquery else result_sql
                result_sql = self._build_ds_ds_binary(
                    right_node,  # unused for left when overrides given
                    right_node,
                    step_op,
                    left_sql_override=left_src,
                    left_ds_override=result_ds,
                )
                # After each step, the result structure is the output dataset.
                result_ds = self._get_output_dataset() or result_ds
                is_subquery = True
            else:
                # ds-scalar: visit scalar and wrap the accumulated SQL.
                scalar_sql = self.visit(right_node)
                measure_names = result_ds.get_measures_names() if result_ds else []
                cols: list[str] = []
                if result_ds:
                    for id_name in result_ds.get_identifiers_names():
                        cols.append(quote_identifier(id_name))
                for m_name in measure_names:
                    m_ref = quote_identifier(m_name)
                    expr = registry.binary.generate(step_op, m_ref, scalar_sql)
                    cols.append(f"{expr} AS {m_ref}")
                if result_ds:
                    for attr_name in result_ds.get_attributes_names():
                        cols.append(quote_identifier(attr_name))
                left_src = f"({result_sql})" if is_subquery else result_sql
                select_clause = ", ".join(cols)
                result_sql = f"SELECT {select_clause} FROM {left_src}"
                is_subquery = True

        return result_sql

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
            col = quote_identifier(comp_name)
            if self._column_prefix:
                col = f"{self._column_prefix}.{col}"
            return col

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

    def _is_time_period_operand(self, node: AST.AST) -> bool:
        """Check if an operand resolves to a TimePeriod type."""
        # Column reference in a clause context
        if isinstance(node, AST.VarID) and self._in_clause and self._current_dataset:
            comp = self._current_dataset.components.get(node.value)
            if comp and comp.data_type == TimePeriod:
                return True
        # Named scalar
        if isinstance(node, AST.VarID) and node.value in self.scalars:
            sc = self.scalars[node.value]
            if sc.data_type == TimePeriod:
                return True
        # CAST to time_period: ParamOp with op=cast and target type = time_period
        if (
            isinstance(node, AST.ParamOp)
            and str(getattr(node, "op", "")).lower() == tokens.CAST
            and len(node.children) >= 2
        ):
            type_node = node.children[1]
            type_str = type_node.value if hasattr(type_node, "value") else str(type_node)
            if type_str.lower() in ("time_period", "timeperiod"):
                return True
        return False

    def _is_duration_operand(self, node: AST.AST) -> bool:
        """Check if an operand resolves to a Duration type."""
        if isinstance(node, AST.VarID) and self._in_clause and self._current_dataset:
            comp = self._current_dataset.components.get(node.value)
            if comp and comp.data_type == Duration:
                return True
        if isinstance(node, AST.VarID) and node.value in self.scalars:
            sc = self.scalars[node.value]
            if sc.data_type == Duration:
                return True
        if (
            isinstance(node, AST.ParamOp)
            and str(getattr(node, "op", "")).lower() == tokens.CAST
            and len(node.children) >= 2
        ):
            type_node = node.children[1]
            type_str = type_node.value if hasattr(type_node, "value") else str(type_node)
            if type_str.lower() == "duration":
                return True
        return False

    def _visit_period_indicator(self, node: AST.UnaryOp) -> str:
        """Visit PERIOD_INDICATOR: extract period indicator from TimePeriod."""
        operand_type = self._get_operand_type(node.operand)

        if operand_type == _DATASET:
            ds = self._get_dataset_structure(node.operand)
            src = self._get_dataset_sql(node.operand)
            if ds is None:
                raise ValueError("Cannot resolve structure for period_indicator")

            # Find time identifier
            time_id = None
            for comp in ds.components.values():
                if comp.data_type == TimePeriod and comp.role == Role.IDENTIFIER:
                    time_id = comp.name
                    break
            if time_id is None:
                raise ValueError("No TimePeriod identifier found for period_indicator")

            id_cols = [quote_identifier(c.name) for c in ds.get_identifiers()]
            extract_expr = (
                f'vtl_period_parse({quote_identifier(time_id)}).period_indicator AS "duration_var"'
            )
            cols_sql = ", ".join(id_cols) + ", " + extract_expr
            return f"SELECT {cols_sql} FROM {src}"
        else:
            operand_sql = self.visit(node.operand)
            return f"vtl_period_parse({operand_sql}).period_indicator"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:  # type: ignore[override]
        """Visit a unary operation."""
        op = str(node.op).lower()

        # Special-case operators
        if op == tokens.PERIOD_INDICATOR:
            return self._visit_period_indicator(node)

        if op in (tokens.FLOW_TO_STOCK, tokens.STOCK_TO_FLOW):
            return self._visit_flow_stock(node, op)

        # --- Generic path: registry-based unary ---
        operand_type = self._get_operand_type(node.operand)

        if operand_type == _DATASET:
            # isnull on mono-measure dataset produces "bool_var"
            name_override: Optional[str] = None
            if op == tokens.ISNULL:
                ds = self._get_dataset_structure(node.operand)
                if ds and len(ds.get_measures_names()) == 1:
                    name_override = "bool_var"

            # Check if dataset has TimePeriod measures for extraction dispatch
            ds_for_tp = self._get_dataset_structure(node.operand)
            has_tp_measures = ds_for_tp is not None and any(
                c.data_type == TimePeriod
                for c in ds_for_tp.components.values()
                if c.role == Role.MEASURE
            )

            def _unary_expr(col_ref: str) -> str:
                if op in _TP_EXTRACTION_MAP and has_tp_measures:
                    return _TP_EXTRACTION_MAP[op].format(col_ref)
                if registry.unary.is_registered(op):
                    return registry.unary.generate(op, col_ref)
                return f"{op.upper()}({col_ref})"

            return self._apply_to_measures(
                node.operand,
                _unary_expr,
                name_override,
                cast_bool_to_str=op in _STRING_UNARY_OPS,
            )
        else:
            # TimePeriod dispatch for extraction operators
            if op in _TP_EXTRACTION_MAP and self._is_time_period_operand(node.operand):
                operand_sql = self.visit(node.operand)
                return _TP_EXTRACTION_MAP[op].format(operand_sql)

            operand_sql = self.visit(node.operand)
            if registry.unary.is_registered(op):
                return registry.unary.generate(op, operand_sql)
            return f"{op.upper()}({operand_sql})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:  # type: ignore[override]
        """Visit a parameterized operation."""
        op = str(node.op).lower()

        if op == tokens.CAST:
            return self._visit_cast(node)

        if op == tokens.RANDOM:
            return self._visit_random(node)

        if op == tokens.DATE_ADD:
            return self._visit_dateadd(node)

        if op == tokens.FILL_TIME_SERIES:
            return self._visit_fill_time_series(node)

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
        """Visit param nodes, converting VTL '_' to None and VTL null to 'NULL'."""
        result: List[Optional[str]] = []
        for p in params:
            if p is None or (isinstance(p, AST.ID) and p.value == "_"):
                result.append(None)
            elif isinstance(p, AST.Constant) and p.value is None:
                result.append("NULL")
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
                return registry.parameterized.generate(op, col_ref, *params_sql)  # type:ignore[arg-type]
            all_args = [col_ref] + [a for a in params_sql if a is not None]
            return f"{op.upper()}({', '.join(all_args)})"

        return self._apply_to_measures(
            ds_node,
            _param_expr,
            cast_bool_to_str=op in _STRING_PARAM_OPS,
        )

    def _visit_fill_time_series(self, node: AST.ParamOp) -> str:
        """Visit FILL_TIME_SERIES: fill missing time periods with NULL rows.

        TimePeriod only. Uses recursive CTE to generate expected periods.
        Carries max_tp through the recursion (DuckDB can't reference other CTEs
        in recursive part).
        """
        ds_node = node.children[0]
        fill_mode = "all"
        if node.params:
            mode_val = self.visit(node.params[0])
            if isinstance(mode_val, str):
                fill_mode = mode_val.strip("'\"").lower()

        ds = self._get_dataset_structure(ds_node)
        src = self._get_dataset_sql(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve structure for fill_time_series")

        # Find time identifier
        time_id = None
        time_type = None
        for comp in ds.components.values():
            if comp.data_type in (TimePeriod, Date) and comp.role == Role.IDENTIFIER:
                time_id = comp.name
                time_type = comp.data_type
                break
        if time_id is None:
            raise ValueError("No time identifier found for fill_time_series")

        # Dispatch by type
        if time_type == Date:
            return self._fill_time_series_date(ds, src, time_id, fill_mode)

        time_col = quote_identifier(time_id)
        other_ids = [c.name for c in ds.get_identifiers() if c.name != time_id]
        other_id_cols = [quote_identifier(n) for n in other_ids]
        measure_names = [c.name for c in ds.components.values() if c.role != Role.IDENTIFIER]
        measure_cols = [quote_identifier(n) for n in measure_names]

        # Build JOIN conditions
        join_conds = [f"g.{time_col} = s.{time_col}"]
        for oc in other_id_cols:
            join_conds.append(f"g.{oc} = s.{oc}")
        join_on = " AND ".join(join_conds)

        # SELECT columns for final output
        g_cols = [f"g.{oc}" for oc in other_id_cols] + [f"g.{time_col}"]
        s_cols = [f"s.{mc}" for mc in measure_cols]
        final_select = ", ".join(g_cols + s_cols)
        order_by = ", ".join(g_cols)

        if fill_mode == "single" and other_ids:
            # Single mode: per-group bounds, carry max_tp + group keys through recursion
            oid_select = ", ".join(other_id_cols)
            oid_ep_refs = ", ".join(f"ep.{oc}" for oc in other_id_cols)

            cte = f"""
WITH RECURSIVE source AS (SELECT * FROM {src}),
parsed AS (
    SELECT *, vtl_period_parse({time_col}) AS tp FROM source
),
bounds AS (
    SELECT {oid_select},
        MIN(tp) AS min_tp,
        MAX(tp) AS max_tp
    FROM parsed
    GROUP BY {oid_select}, tp.period_indicator
),
expected_periods(tp, max_tp, {oid_select}) AS (
    SELECT min_tp, max_tp, {oid_select} FROM bounds
    UNION ALL
    SELECT CASE
        WHEN ep.tp.period_number + 1 > vtl_period_limit(ep.tp.period_indicator)
        THEN {{'year': ep.tp.year + 1, 'period_indicator': ep.tp.period_indicator,
              'period_number': 1}}::vtl_time_period
        ELSE {{'year': ep.tp.year, 'period_indicator': ep.tp.period_indicator,
              'period_number': ep.tp.period_number + 1}}::vtl_time_period
    END,
    ep.max_tp,
    {oid_ep_refs}
    FROM expected_periods ep
    WHERE ep.tp < ep.max_tp
),
full_grid AS (
    SELECT {oid_select}, vtl_period_to_string(tp) AS {time_col}
    FROM expected_periods
)
SELECT {final_select}
FROM full_grid g
LEFT JOIN source s ON {join_on}
ORDER BY {order_by}"""
        else:
            # All mode: global bounds, carry max_tp through recursion
            if other_ids:
                oid_join = ", ".join(other_id_cols)
                other_combos = f"""
group_freq AS (
    SELECT DISTINCT {oid_join},
        vtl_period_parse({time_col}).period_indicator AS ind
    FROM source
),"""
                grid_sql = (
                    f"SELECT gf.{', gf.'.join(other_id_cols)}, ps.{time_col} "
                    f"FROM group_freq gf "
                    f"JOIN period_strings ps "
                    f"ON vtl_period_parse(ps.{time_col}).period_indicator = gf.ind"
                )
            else:
                other_combos = ""
                grid_sql = f"SELECT {time_col} FROM period_strings"

            cte = f"""
WITH RECURSIVE source AS (SELECT * FROM {src}),
parsed AS (
    SELECT *, vtl_period_parse({time_col}) AS tp FROM source
),
year_range AS (
    SELECT MIN(tp.year) AS min_year, MAX(tp.year) AS max_year FROM parsed
),
freq_list AS (
    SELECT DISTINCT tp.period_indicator AS ind FROM parsed
),
bounds AS (
    SELECT ind,
        {{'year': min_year, 'period_indicator': ind,
          'period_number': 1}}::vtl_time_period AS min_tp,
        {{'year': max_year, 'period_indicator': ind,
          'period_number': vtl_period_limit(ind)}}::vtl_time_period AS max_tp
    FROM freq_list, year_range
),
expected_periods(tp, max_tp) AS (
    SELECT min_tp, max_tp FROM bounds
    UNION ALL
    SELECT CASE
        WHEN ep.tp.period_number + 1 > vtl_period_limit(ep.tp.period_indicator)
        THEN {{'year': ep.tp.year + 1, 'period_indicator': ep.tp.period_indicator,
              'period_number': 1}}::vtl_time_period
        ELSE {{'year': ep.tp.year, 'period_indicator': ep.tp.period_indicator,
              'period_number': ep.tp.period_number + 1}}::vtl_time_period
    END,
    ep.max_tp
    FROM expected_periods ep
    WHERE ep.tp < ep.max_tp
),
period_strings AS (
    SELECT vtl_period_to_string(tp) AS {time_col} FROM expected_periods
),{other_combos}
full_grid AS (
    {grid_sql}
)
SELECT {final_select}
FROM full_grid g
LEFT JOIN source s ON {join_on}
ORDER BY {order_by}"""

        return cte.strip()

    def _fill_time_series_date(self, ds: Dataset, src: str, time_id: str, fill_mode: str) -> str:
        """Fill time series for Date identifiers using frequency inference."""
        time_col = quote_identifier(time_id)
        other_ids = [c.name for c in ds.get_identifiers() if c.name != time_id]
        other_id_cols = [quote_identifier(n) for n in other_ids]
        measure_names = [c.name for c in ds.components.values() if c.role != Role.IDENTIFIER]
        measure_cols = [quote_identifier(n) for n in measure_names]

        join_conds = [f"g.{time_col} = s.{time_col}"]
        for oc in other_id_cols:
            join_conds.append(f"g.{oc} = s.{oc}")
        join_on = " AND ".join(join_conds)

        g_cols = [f"g.{oc}" for oc in other_id_cols] + [f"g.{time_col}"]
        s_cols = [f"s.{mc}" for mc in measure_cols]
        final_select = ", ".join(g_cols + s_cols)
        order_by = ", ".join(g_cols)

        partition = f"PARTITION BY {', '.join(other_id_cols)}" if other_id_cols else ""

        if fill_mode == "single" and other_ids:
            bounds_group = f"GROUP BY {', '.join(other_id_cols)}"
            bounds_select = f"{', '.join(other_id_cols)},"
        else:
            bounds_group = ""
            bounds_select = ""

        freq_step = "(SELECT step FROM freq)"
        if other_ids:
            if fill_mode == "single":
                grid_sql = f"""
SELECT b.{", b.".join(other_id_cols)},
    CAST(d AS TIMESTAMP) AS {time_col}
FROM bounds b, generate_series(b.min_d, b.max_d, {freq_step}) AS t(d)"""
            else:
                grid_sql = f"""
SELECT gf.{", gf.".join(other_id_cols)},
    CAST(d AS TIMESTAMP) AS {time_col}
FROM group_freq gf, generate_series(
    (SELECT min_d FROM bounds), (SELECT max_d FROM bounds), {freq_step}
) AS t(d)"""
        else:
            grid_sql = f"""
SELECT CAST(d AS TIMESTAMP) AS {time_col}
FROM generate_series(
    (SELECT min_d FROM bounds), (SELECT max_d FROM bounds), {freq_step}
) AS t(d)"""

        if fill_mode == "single" and other_ids:
            extra_ctes = ""
        elif other_ids:
            extra_ctes = f"""
group_freq AS (
    SELECT DISTINCT {", ".join(other_id_cols)} FROM source
),"""
        else:
            extra_ctes = ""

        return f"""
WITH source AS (SELECT * FROM {src}),
freq AS (
    SELECT CASE
        WHEN MIN(diff_days) BETWEEN 1 AND 6 THEN INTERVAL 1 DAY
        WHEN MIN(diff_days) BETWEEN 7 AND 27 THEN INTERVAL 7 DAY
        WHEN MIN(diff_days) BETWEEN 28 AND 89 THEN INTERVAL 1 MONTH
        WHEN MIN(diff_days) BETWEEN 90 AND 180 THEN INTERVAL 3 MONTH
        WHEN MIN(diff_days) BETWEEN 181 AND 364 THEN INTERVAL 6 MONTH
        ELSE INTERVAL 1 YEAR
    END AS step
    FROM (
        SELECT ABS(DATE_DIFF('day',
            LAG({time_col}) OVER ({partition} ORDER BY {time_col}),
            {time_col})) AS diff_days
        FROM source
    ) WHERE diff_days IS NOT NULL AND diff_days > 0
),
bounds AS (
    SELECT {bounds_select} MIN({time_col}) AS min_d, MAX({time_col}) AS max_d
    FROM source
    {bounds_group}
),{extra_ctes}
full_grid AS ({grid_sql}
)
SELECT {final_select}
FROM full_grid g
LEFT JOIN source s ON {join_on}
ORDER BY {order_by}""".strip()

    def _visit_flow_stock(self, node: AST.UnaryOp, op: str) -> str:
        """Visit FLOW_TO_STOCK or STOCK_TO_FLOW: window functions over time series."""
        ds = self._get_dataset_structure(node.operand)
        src = self._get_dataset_sql(node.operand)
        if ds is None:
            raise ValueError(f"Cannot resolve structure for {op}")

        # Find time identifier
        time_id = None
        time_type = None
        for comp in ds.components.values():
            if comp.data_type in (TimePeriod, Date) and comp.role == Role.IDENTIFIER:
                time_id = comp.name
                time_type = comp.data_type
                break
        if time_id is None:
            raise ValueError(f"No time identifier found for {op}")

        # Other identifiers for PARTITION BY
        other_ids = [quote_identifier(c.name) for c in ds.get_identifiers() if c.name != time_id]

        # For TimePeriod, also partition by period_indicator
        partition_parts = list(other_ids)
        if time_type == TimePeriod:
            partition_parts.append(
                f"vtl_period_parse({quote_identifier(time_id)}).period_indicator"
            )

        partition_clause = f"PARTITION BY {', '.join(partition_parts)}" if partition_parts else ""
        order_clause = f"ORDER BY {quote_identifier(time_id)}"
        window = f"({partition_clause} {order_clause})"

        # Build SELECT
        cols = []
        for comp in ds.components.values():
            col = quote_identifier(comp.name)
            if comp.role == Role.IDENTIFIER:
                cols.append(col)
            else:
                # Apply window function to measures
                if op == tokens.FLOW_TO_STOCK:
                    cols.append(
                        f"CASE WHEN {col} IS NULL THEN NULL ELSE "
                        f"SUM({col}) OVER ({partition_clause} {order_clause} "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) END AS {col}"
                    )
                else:  # STOCK_TO_FLOW
                    cols.append(f"COALESCE({col} - LAG({col}) OVER {window}, {col}) AS {col}")

        return f"SELECT {', '.join(cols)} FROM {src}"

    def _visit_timeshift(self, node: AST.BinOp) -> str:
        """Visit TIMESHIFT: shift time identifier by N periods."""
        ds_node = node.left
        shift_sql = self.visit(node.right)

        ds = self._get_dataset_structure(ds_node)
        src = self._get_dataset_sql(ds_node)
        if ds is None:
            raise ValueError("Cannot resolve structure for timeshift")

        # Find time identifier and its type
        time_id = None
        time_type = None
        for comp in ds.components.values():
            if comp.data_type in (TimePeriod, Date) and comp.role == Role.IDENTIFIER:
                time_id = comp.name
                time_type = comp.data_type
                break
        if time_id is None:
            raise ValueError("No time identifier found for timeshift")

        time_col = quote_identifier(time_id)

        if time_type == TimePeriod:
            shifted = f"vtl_tp_shift(vtl_period_parse({time_col}), {shift_sql}) AS {time_col}"
            cols = []
            for comp in ds.components.values():
                col = quote_identifier(comp.name)
                cols.append(shifted if comp.name == time_id else col)
            return f"SELECT {', '.join(cols)} FROM {src}"
        else:
            # Date: infer frequency from date diffs, then shift by freq * N
            other_ids = [
                quote_identifier(c.name) for c in ds.get_identifiers() if c.name != time_id
            ]
            partition = f"PARTITION BY {', '.join(other_ids)}" if other_ids else ""

            cols = []
            for comp in ds.components.values():
                col = quote_identifier(comp.name)
                if comp.name == time_id:
                    cols.append(f"vtl_dateadd({col}, {shift_sql}, freq.period_ind) AS {col}")
                else:
                    cols.append(col)

            return f"""SELECT {", ".join(cols)}
FROM {src}, (
    SELECT CASE
        WHEN MIN(diff_days) BETWEEN 1 AND 6 THEN 'D'
        WHEN MIN(diff_days) BETWEEN 7 AND 27 THEN 'W'
        WHEN MIN(diff_days) BETWEEN 28 AND 89 THEN 'M'
        WHEN MIN(diff_days) BETWEEN 90 AND 180 THEN 'Q'
        WHEN MIN(diff_days) BETWEEN 181 AND 364 THEN 'S'
        ELSE 'A'
    END AS period_ind
    FROM (
        SELECT ABS(DATE_DIFF('day',
            LAG({time_col}) OVER ({partition} ORDER BY {time_col}),
            {time_col})) AS diff_days
        FROM {src}
    ) WHERE diff_days IS NOT NULL AND diff_days > 0
) AS freq"""

    def _visit_dateadd(self, node: AST.ParamOp) -> str:
        """Visit DATEADD operation: dateadd(op, shiftNumber, periodInd)."""
        operand_node = node.children[0]
        operand_type = self._get_operand_type(operand_node)

        shift_sql = self.visit(node.params[0]) if node.params else "0"
        period_sql = self.visit(node.params[1]) if len(node.params) > 1 else "'D'"

        is_tp = self._is_time_period_operand(operand_node)

        if operand_type == _DATASET:
            ds_node = operand_node
            ds = self._get_dataset_structure(ds_node)
            has_tp = ds is not None and any(
                c.data_type == TimePeriod for c in ds.components.values() if c.role == Role.MEASURE
            )

            if has_tp and self.current_assignment:
                out_ds = self.output_datasets.get(self.current_assignment)
                if out_ds is not None:
                    for comp in out_ds.components.values():
                        if comp.data_type == TimePeriod:
                            comp.data_type = Date

            def _dateadd_expr(col_ref: str) -> str:
                if has_tp:
                    return f"vtl_tp_dateadd(vtl_period_parse({col_ref}), {shift_sql}, {period_sql})"
                return f"vtl_dateadd({col_ref}, {shift_sql}, {period_sql})"

            return self._apply_to_measures(ds_node, _dateadd_expr)
        else:
            operand_sql = self.visit(operand_node)
            if is_tp:
                return f"vtl_tp_dateadd(vtl_period_parse({operand_sql}), {shift_sql}, {period_sql})"
            return f"vtl_dateadd({operand_sql}, {shift_sql}, {period_sql})"

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
            return f"STRFTIME(STRPTIME({expr}, '{mask}'), '%Y-%m-%d %H:%M:%S')"
        # Normalize TimePeriod values on cast to ensure canonical format
        if target_type_str.lower() in ("time_period", "timeperiod"):
            return f"vtl_period_normalize(CAST({expr} AS VARCHAR))"
        # VTL cast to Integer truncates toward zero; DuckDB CAST rounds.
        if target_type_str == "Integer":
            return f"CAST(TRUNC({expr}) AS {duckdb_type})"
        return f"CAST({expr} AS {duckdb_type})"

    @staticmethod
    def _check_random_negative_index(index_node: Optional[AST.AST]) -> None:
        """Raise SemanticError if the index is a negative literal."""
        if (
            isinstance(index_node, AST.UnaryOp)
            and index_node.op == "-"
            and isinstance(index_node.operand, AST.Constant)
        ):
            from vtlengine.Exceptions import SemanticError

            raise SemanticError("2-1-15-2", op="random", value=index_node.operand.value)

    def _visit_random_impl(
        self,
        seed_node: Optional[AST.AST],
        index_node: Optional[AST.AST],
    ) -> str:
        """Generate SQL for RANDOM (shared by ParamOp and BinOp forms)."""
        self._check_random_negative_index(index_node)
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

    def _visit_random(self, node: AST.ParamOp) -> str:
        """Visit RANDOM operator (ParamOp form)."""
        seed_node = node.children[0] if node.children else None
        index_node = node.params[0] if node.params else None
        return self._visit_random_impl(seed_node, index_node)

    def _visit_random_binop(self, node: AST.BinOp) -> str:
        """Visit RANDOM operator (BinOp form, e.g. inside calc)."""
        return self._visit_random_impl(node.left, node.right)

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
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        with self._clause_scope(ds):
            conditions = [self.visit(child) for child in node.children]

        builder = SQLBuilder().select_all().from_table(table_src)
        if conditions:
            builder.where(" AND ".join(conditions))
        return builder.build()

    def _visit_calc(self, node: AST.RegularAggregation) -> str:
        """Visit calc clause: DS[calc new_col := expr, ...]."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        calc_exprs: Dict[str, str] = {}
        with self._clause_scope(ds):
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
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        # Identifiers are always kept
        keep_names: List[str] = [
            name for name, comp in ds.components.items() if comp.role == Role.IDENTIFIER
        ]
        keep_names.extend(self._extract_component_names(node.children, self._join_alias_map))

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

        table_src = self._get_dataset_sql(node.dataset)  # ds not needed for drop
        drop_names = self._extract_component_names(node.children, self._join_alias_map)

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
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        renames: Dict[str, str] = {}
        for child in node.children:
            if isinstance(child, AST.RenameNode):
                old = child.old_name
                new = child.new_name
                # Resolve UDO component parameters
                udo_old = self._get_udo_param(old)
                if udo_old is not None:
                    if isinstance(udo_old, (AST.VarID, AST.Identifier)):
                        old = udo_old.value
                    elif isinstance(udo_old, str):
                        old = udo_old
                udo_new = self._get_udo_param(new)
                if udo_new is not None:
                    if isinstance(udo_new, (AST.VarID, AST.Identifier)):
                        new = udo_new.value
                    elif isinstance(udo_new, str):
                        new = udo_new
                # Check if alias-qualified name is in the join alias map
                if "#" in old and old in self._join_alias_map:
                    renames[old] = new
                    # Track renamed qualified name as consumed
                    self._consumed_join_aliases.add(old)
                elif "#" in old:
                    # Strip alias prefix from membership refs (e.g. d2#Me_2 -> Me_2)
                    old = old.split("#", 1)[1]
                    renames[old] = new
                else:
                    renames[old] = new

        cols: List[str] = []
        for name in ds.components:
            # Check direct match first, then try matching via qualified name
            matched_new = renames.get(name)
            if matched_new is None and "#" in name:
                unqual = name.split("#", 1)[1]
                matched_new = renames.get(unqual)
            if matched_new is not None:
                cols.append(f"{quote_identifier(name)} AS {quote_identifier(matched_new)}")
            else:
                cols.append(quote_identifier(name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_subspace(self, node: AST.RegularAggregation) -> str:
        """Visit subspace clause."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

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
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        calc_exprs: Dict[str, str] = {}
        having_sql: Optional[str] = None

        with self._clause_scope(ds):
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
                        if isinstance(hc, AST.ParamOp) and hc.params is not None:
                            having_sql = self.visit(hc.params)

                    expr_sql = self.visit(agg_node)
                    calc_exprs[col_name] = expr_sql

        # Extract group-by identifiers from AST nodes to avoid using the
        # overall output dataset (which may represent a join result).
        group_ids: List[str] = []
        grouping_op: str = ""
        grouping_names: List[str] = []
        for child in node.children:
            assignment = child
            if isinstance(child, AST.UnaryOp) and isinstance(child.operand, AST.Assignment):
                assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                agg_node = assignment.right
                if isinstance(agg_node, AST.Aggregation) and agg_node.grouping:
                    grouping_op = agg_node.grouping_op or ""
                    for g in agg_node.grouping:
                        if (
                            isinstance(g, (AST.VarID, AST.Identifier))
                            and g.value not in grouping_names
                        ):
                            grouping_names.append(g.value)

        all_input_ids = list(ds.get_identifiers_names())
        if grouping_op == "group by":
            group_ids = grouping_names
        elif grouping_op == "group except":
            except_set = set(grouping_names)
            group_ids = [n for n in all_input_ids if n not in except_set]
        elif not grouping_names:
            # No explicit grouping → fall back to output/input dataset identifiers
            output_ds = self._get_output_dataset()
            group_ids = list(output_ds.get_identifiers_names() if output_ds else all_input_ids)

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
        """Visit apply clause: inner_join(... apply d1 op d2).

        For each BinOp child, applies the operator to common measures
        between the left and right aliases, producing a single output
        column per common measure.
        """
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        # Get the output structure (post-apply)
        output_ds = self.output_datasets.get(self.current_assignment)

        # Collect identifier columns
        id_names = ds.get_identifiers_names()

        # Build computed measure expressions from BinOp children
        computed: Dict[str, str] = {}
        for child in node.children:
            if not isinstance(child, AST.BinOp):
                continue
            left_alias = child.left.value if hasattr(child.left, "value") else str(child.left)
            right_alias = child.right.value if hasattr(child.right, "value") else str(child.right)
            op = str(child.op).lower() if child.op else ""

            # Find common measures: components that exist as both alias#comp in the join
            left_measures: Dict[str, str] = {}
            right_measures: Dict[str, str] = {}
            for qualified in self._join_alias_map:
                if "#" in qualified:
                    alias, comp = qualified.split("#", 1)
                    if alias == left_alias:
                        left_measures[comp] = qualified
                    elif alias == right_alias:
                        right_measures[comp] = qualified

            common_measures = set(left_measures.keys()) & set(right_measures.keys())
            for measure in common_measures:
                left_col = quote_identifier(left_measures[measure])
                right_col = quote_identifier(right_measures[measure])
                if registry.binary.is_registered(op):
                    expr = registry.binary.generate(op, left_col, right_col)
                else:
                    expr = f"{left_col} {op} {right_col}"
                computed[measure] = expr
                # Mark both qualified names as consumed
                self._consumed_join_aliases.add(left_measures[measure])
                self._consumed_join_aliases.add(right_measures[measure])

        # Build SELECT: identifiers + computed measures
        cols: List[str] = [quote_identifier(id_) for id_ in id_names]
        if output_ds:
            for comp_name in output_ds.get_measures_names():
                if comp_name in computed:
                    cols.append(f"{computed[comp_name]} AS {quote_identifier(comp_name)}")
                else:
                    cols.append(quote_identifier(comp_name))
        else:
            for measure, expr in computed.items():
                cols.append(f"{expr} AS {quote_identifier(measure)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_unpivot(self, node: AST.RegularAggregation) -> str:
        """Visit unpivot clause: DS[unpivot new_id, new_measure].

        Transforms measures into rows.  For each measure column, produces one
        row per data point with the measure *name* as the new identifier value
        and the measure *value* as the new measure value.  Rows where the
        measure value is NULL are dropped (VTL 2.1 RM line 7200).
        """
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            if node.dataset:
                return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
            return ""
        ds, table_src = resolved

        if len(node.children) < 2:
            raise ValueError("Unpivot clause requires two operands")

        raw_id = (
            node.children[0].value if hasattr(node.children[0], "value") else str(node.children[0])
        )
        raw_measure = (
            node.children[1].value if hasattr(node.children[1], "value") else str(node.children[1])
        )
        # Resolve UDO component parameters
        udo_id = self._get_udo_param(raw_id)
        if udo_id is not None:
            if isinstance(udo_id, (AST.VarID, AST.Identifier)):
                raw_id = udo_id.value
            elif isinstance(udo_id, str):
                raw_id = udo_id
        udo_measure = self._get_udo_param(raw_measure)
        if udo_measure is not None:
            if isinstance(udo_measure, (AST.VarID, AST.Identifier)):
                raw_measure = udo_measure.value
            elif isinstance(udo_measure, str):
                raw_measure = udo_measure
        new_id_name = raw_id
        new_measure_name = raw_measure

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

    def _build_agg_group_cols(
        self,
        node: AST.Aggregation,
        ds: Dataset,
        group_cols: List[str],
    ) -> Tuple[List[str], List[str]]:
        """Build SELECT and GROUP BY column lists, handling time_agg."""
        time_agg_expr: Optional[str] = None
        time_agg_id: Optional[str] = None
        if node.grouping:
            for g in node.grouping:
                if isinstance(g, AST.TimeAggregation):
                    with self._clause_scope(ds):
                        time_agg_expr = self.visit_TimeAggregation(g)
                    for comp in ds.components.values():
                        if comp.data_type in (TimePeriod, Date) and comp.role == Role.IDENTIFIER:
                            time_agg_id = comp.name
                            break

        # For group by/group all with time_agg, ensure the time identifier
        # is included in group_cols (it may not be listed explicitly).
        if (
            time_agg_id
            and time_agg_expr
            and node.grouping_op != "group except"
            and time_agg_id not in group_cols
        ):
            group_cols = list(group_cols) + [time_agg_id]

        cols: List[str] = []
        group_by_cols: List[str] = []
        for col_name in group_cols:
            if col_name == time_agg_id and time_agg_expr:
                cols.append(f"{time_agg_expr} AS {quote_identifier(col_name)}")
                group_by_cols.append(time_agg_expr)
            else:
                cols.append(quote_identifier(col_name))
                group_by_cols.append(quote_identifier(col_name))
        return cols, group_by_cols

    def visit_Aggregation(self, node: AST.Aggregation) -> str:  # type: ignore[override]
        """Visit a standalone aggregation: sum(DS group by Id)."""
        op = str(node.op).lower()

        # Component-level aggregation in clause context
        if self._in_clause and node.operand:
            operand_type = self._get_operand_type(node.operand)
            if operand_type in (_COMPONENT, _SCALAR):
                operand_sql = self.visit(node.operand)
                # Duration MIN/MAX: convert to int, aggregate, convert back
                if (
                    op in (tokens.MIN, tokens.MAX)
                    and self._current_dataset
                    and hasattr(node.operand, "value")
                ):
                    comp = self._current_dataset.components.get(node.operand.value)
                    if comp is not None and comp.data_type == Duration:
                        return (
                            f"vtl_int_to_duration({op.upper()}(vtl_duration_to_int({operand_sql})))"
                        )
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
                        return f"NULLIF(COUNT(CASE WHEN {or_parts} THEN 1 END), 0)"
                return "NULLIF(COUNT(*), 0)"
            return ""

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            operand_sql = self.visit(node.operand)
            if registry.aggregate.is_registered(op):
                return registry.aggregate.generate(op, operand_sql)
            return f"{op.upper()}({operand_sql})"

        table_src = self._get_dataset_sql(node.operand)

        # Resolve group columns from the input dataset's identifiers.
        # The input dataset (ds) reflects the actual columns available in
        # the source table.  The output dataset may include transformations
        # (calc, keep) applied after this aggregation and should NOT be
        # used for column references.
        all_ids = ds.get_identifiers_names()
        group_cols = self._resolve_group_cols(node, all_ids)

        cols, group_by_cols = self._build_agg_group_cols(node, ds, group_cols)

        # count replaces all measures with a single int_var column.
        # VTL count() excludes rows where all measures are null.
        if op == tokens.COUNT:
            # VTL spec: count() always produces a single measure "int_var"
            alias = "int_var"
            # Build conditional count excluding all-null measure rows
            source_measures = ds.get_measures_names()
            if source_measures:
                and_parts = " AND ".join(
                    f"{quote_identifier(m)} IS NOT NULL" for m in source_measures
                )
                count_expr = f"COUNT(CASE WHEN {and_parts} THEN 1 END)"
                # When there are group columns, return NULL for groups with zero
                # matching rows; for DWI (no group cols), return 0 directly.
                if group_cols:
                    count_expr = f"NULLIF({count_expr}, 0)"
                cols.append(f"{count_expr} AS {quote_identifier(alias)}")
            else:
                # No measures: count data points (rows)
                cols.append(f"COUNT(*) AS {quote_identifier(alias)}")
        else:
            measures = ds.get_measures_names()
            for measure in measures:
                comp = ds.components.get(measure)
                is_time_period = comp is not None and comp.data_type == TimePeriod
                qm = quote_identifier(measure)

                is_duration = comp is not None and comp.data_type == Duration
                if is_time_period and op in (tokens.MIN, tokens.MAX):
                    # TimePeriod MIN/MAX: parse to STRUCT, aggregate, format back
                    expr = f"vtl_period_to_string({op.upper()}(vtl_period_parse({qm})))"
                elif is_duration and op in (tokens.MIN, tokens.MAX):
                    # Duration MIN/MAX: convert to int, aggregate, convert back
                    expr = f"vtl_int_to_duration({op.upper()}(vtl_duration_to_int({qm})))"
                elif registry.aggregate.is_registered(op):
                    expr = registry.aggregate.generate(op, qm)
                else:
                    expr = f"{op.upper()}({qm})"
                cols.append(f"{expr} AS {qm}")

        builder = SQLBuilder().select(*cols).from_table(table_src)

        if group_cols:
            builder.group_by(*group_by_cols)
        elif all_ids:
            builder.having("COUNT(*) > 0")

        if node.having_clause:
            with self._clause_scope(ds):
                hc = node.having_clause
                if isinstance(hc, AST.ParamOp) and hc.params is not None:
                    having_sql = self.visit(hc.params)
                else:
                    having_sql = self.visit(hc)
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
                # Normalize column order across all branches to prevent
                # positional type mismatches in UNION ALL.
                output_ds = self._get_output_dataset()
                order_ds = output_ds if output_ds else ds
                col_order = list(order_ds.components.keys())
                ordered_cols = ", ".join(quote_identifier(c) for c in col_order)
                ordered_sqls = [f"SELECT {ordered_cols} FROM ({sql}) AS _ord" for sql in child_sqls]

                id_names = order_ds.get_identifiers_names()
                if id_names:
                    inner_sql = registry.set_ops.generate(op, *ordered_sqls)
                    id_cols = ", ".join(quote_identifier(i) for i in id_names)
                    return f"SELECT DISTINCT ON ({id_cols}) * FROM ({inner_sql}) AS _union_t"
                return registry.set_ops.generate(op, *ordered_sqls)
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

    def _scalar_if_sql(self, node: AST.If) -> str:
        """Build a simple CASE WHEN for scalar IF-THEN-ELSE."""
        cond_sql = self.visit(node.condition)
        then_sql = self.visit(node.thenOp)
        else_sql = self.visit(node.elseOp)
        return f"CASE WHEN {cond_sql} THEN {then_sql} ELSE {else_sql} END"

    def visit_If(self, node: AST.If) -> str:
        """Visit IF-THEN-ELSE."""
        if self._get_operand_type(node.condition) != _DATASET:
            return self._scalar_if_sql(node)
        return self._build_dataset_if(node)

    def _find_condition_source(self, node: AST.AST) -> Optional[AST.AST]:
        """Find the source dataset AST node from a condition expression."""
        if isinstance(node, AST.BinOp):
            op = str(node.op).lower() if node.op else ""
            if op == tokens.MEMBERSHIP:
                return node.left
            left = self._find_condition_source(node.left)
            if left is not None:
                return left
            return self._find_condition_source(node.right)
        if isinstance(node, (AST.UnaryOp, AST.ParFunction)):
            return self._find_condition_source(node.operand)
        if isinstance(node, AST.VarID) and self._get_operand_type(node) == _DATASET:
            return node
        return None

    def _build_dataset_if(self, node: AST.If) -> str:
        """Build SQL for dataset-level IF-THEN-ELSE with JOINs."""
        # Find the source dataset that the condition references
        source_node = self._find_condition_source(node.condition)
        if source_node is None:
            return self._scalar_if_sql(node)

        source_ds = self._get_dataset_structure(source_node)
        if source_ds is None:
            return self._scalar_if_sql(node)

        alias_cond = "cond"

        # When the condition is a binary op between two datasets (e.g. DS_1 > DS_2),
        # it cannot be evaluated as a simple column expression — evaluate it as a
        # subquery and reference its boolean measure column instead.
        cond_is_ds_vs_ds = (
            isinstance(node.condition, AST.BinOp)
            and self._get_operand_type(node.condition.left) == _DATASET
            and self._get_operand_type(node.condition.right) == _DATASET
        )
        cond_ds = self._get_dataset_structure(node.condition) if cond_is_ds_vs_ds else None
        if cond_ds is not None:
            source_sql = self.visit(node.condition)
            source_ids = list(cond_ds.get_identifiers_names())
            bool_measures = list(cond_ds.get_measures_names())
            cond_expr = (
                f"{alias_cond}.{quote_identifier(bool_measures[0])}" if bool_measures else "TRUE"
            )
        else:
            source_sql = self._get_dataset_sql(source_node)
            source_ids = list(source_ds.get_identifiers_names())
            # Evaluate condition as a column expression (not a full SELECT)
            with self._clause_scope(source_ds, prefix=alias_cond):
                cond_expr = self.visit(node.condition)

        then_type = self._get_operand_type(node.thenOp)
        else_type = self._get_operand_type(node.elseOp)

        # Determine output measures and attributes.
        def _is_plain_dataset(n: AST.AST) -> bool:
            return isinstance(n, AST.VarID) and self._get_operand_type(n) == _DATASET

        if then_type == _DATASET and _is_plain_dataset(node.thenOp):
            ref_ds = self._get_dataset_structure(node.thenOp)
            output_measures = list(ref_ds.get_measures_names()) if ref_ds else []
            output_attributes = list(ref_ds.get_attributes_names()) if ref_ds else []
        elif else_type == _DATASET and _is_plain_dataset(node.elseOp):
            ref_ds = self._get_dataset_structure(node.elseOp)
            output_measures = list(ref_ds.get_measures_names()) if ref_ds else []
            output_attributes = list(ref_ds.get_attributes_names()) if ref_ds else []
        else:
            output_ds = self._get_output_dataset()
            if output_ds is not None:
                output_measures = list(output_ds.get_measures_names())
                output_attributes = list(output_ds.get_attributes_names())
            else:
                output_measures = list(source_ds.get_measures_names())
                output_attributes = list(source_ds.get_attributes_names())

        # Build SELECT columns
        cols: List[str] = [f"{alias_cond}.{quote_identifier(id_)}" for id_ in source_ids]

        for col_name in output_measures + output_attributes:
            if then_type == _DATASET:
                then_ref = f"t.{quote_identifier(col_name)}"
            else:
                then_ref = self.visit(node.thenOp)

            if else_type == _DATASET:
                else_ref = f"e.{quote_identifier(col_name)}"
            else:
                else_ref = self.visit(node.elseOp)

            cols.append(
                f"CASE WHEN {cond_expr} THEN {then_ref} "
                f"ELSE {else_ref} END AS {quote_identifier(col_name)}"
            )

        # Use from_subquery when the source is a SELECT (e.g., dataset-level condition)
        if source_sql.lstrip().upper().startswith("SELECT"):
            builder = SQLBuilder().select(*cols).from_subquery(source_sql, alias_cond)
        else:
            builder = SQLBuilder().select(*cols).from_table(source_sql, alias_cond)

        # Use LEFT JOINs so empty datasets don't eliminate all rows
        then_join_id: Optional[str] = None
        if then_type == _DATASET:
            then_sql = self._get_dataset_sql(node.thenOp)
            then_ds = self._get_dataset_structure(node.thenOp)
            then_ids = set(then_ds.get_identifiers_names()) if then_ds else set()
            common = [id_ for id_ in source_ids if id_ in then_ids]
            on_parts = [
                f"{alias_cond}.{quote_identifier(id_)} = t.{quote_identifier(id_)}"
                for id_ in common
            ]
            if on_parts:
                builder.join(then_sql, "t", on=" AND ".join(on_parts), join_type="LEFT")
                then_join_id = f"t.{quote_identifier(common[0])}"

        else_join_id: Optional[str] = None
        if else_type == _DATASET:
            else_sql = self._get_dataset_sql(node.elseOp)
            else_ds = self._get_dataset_structure(node.elseOp)
            else_ids = set(else_ds.get_identifiers_names()) if else_ds else set()
            common = [id_ for id_ in source_ids if id_ in else_ids]
            on_parts = [
                f"{alias_cond}.{quote_identifier(id_)} = e.{quote_identifier(id_)}"
                for id_ in common
            ]
            if on_parts:
                builder.join(else_sql, "e", on=" AND ".join(on_parts), join_type="LEFT")
                else_join_id = f"e.{quote_identifier(common[0])}"

        # Filter: only keep rows where the selected side has a match.
        # Scalar sides always match; dataset sides need a LEFT JOIN hit.
        if then_join_id and else_join_id:
            builder.where(
                f"CASE WHEN {cond_expr} THEN {then_join_id} IS NOT NULL "
                f"ELSE {else_join_id} IS NOT NULL END"
            )
        elif then_join_id:
            # then=dataset, else=scalar: filter when condition is true
            builder.where(f"NOT ({cond_expr}) OR {then_join_id} IS NOT NULL")
        elif else_join_id:
            # then=scalar, else=dataset: filter when condition is false
            builder.where(f"({cond_expr}) OR {else_join_id} IS NOT NULL")

        return builder.build()

    def _build_case_when_sql(
        self,
        cases: List[Any],
        else_op: AST.AST,
    ) -> str:
        """Build a scalar CASE WHEN SQL with reversed order (VTL last-match-wins)."""
        parts = ["CASE"]
        for case_obj in reversed(cases):
            cond_sql = self.visit(case_obj.condition)
            then_sql = self.visit(case_obj.thenOp)
            parts.append(f"WHEN {cond_sql} THEN {then_sql}")
        parts.append(f"ELSE {self.visit(else_op)} END")
        return " ".join(parts)

    def visit_Case(self, node: AST.Case) -> str:
        """Visit CASE expression.

        VTL CASE uses last-match-wins semantics (later conditions override earlier
        ones), while SQL CASE uses first-match-wins.  We reverse the WHEN order so
        the SQL engine evaluates conditions with the same priority as VTL.

        For dataset-level CASE (where conditions are boolean datasets), we build
        JOINs similar to ``_build_dataset_if``.
        """
        cond_types = [self._get_operand_type(c.condition) for c in node.cases]
        if any(t == _DATASET for t in cond_types):
            return self._build_dataset_case(node)

        return self._build_case_when_sql(node.cases, node.elseOp)

    def _build_case_condition(
        self,
        case_obj: AST.CaseObj,
        alias: str,
        source_ids: List[str],
        alias_src: str,
        builder: SQLBuilder,
    ) -> str:
        """Join a CASE condition dataset and return the SQL condition expression."""
        cond_source = self._find_condition_source(case_obj.condition)
        cond_ds = self._get_dataset_structure(cond_source) if cond_source else None

        # JOIN condition dataset
        if cond_source is not None and cond_ds is not None:
            cond_sql = self._get_dataset_sql(cond_source)
            cond_ids = set(cond_ds.get_identifiers_names())
            common = [id_ for id_ in source_ids if id_ in cond_ids]
            on_parts = [
                f"{alias_src}.{quote_identifier(id_)} = {alias}.{quote_identifier(id_)}"
                for id_ in common
            ]
            if on_parts:
                builder.join(cond_sql, alias, on=" AND ".join(on_parts), join_type="LEFT")

        # Build condition expression
        if isinstance(case_obj.condition, AST.VarID) and cond_ds is not None:
            # Bare dataset VarID: reference its boolean measure column
            bool_measure = list(cond_ds.get_measures_names())[0]
            return f"{alias}.{quote_identifier(bool_measure)}"

        with self._clause_scope(cond_ds, prefix=alias):
            return self.visit(case_obj.condition)

    def _join_dataset_operand(
        self,
        operand: AST.AST,
        alias: str,
        source_ids: List[str],
        alias_src: str,
        builder: SQLBuilder,
    ) -> None:
        """LEFT JOIN a dataset operand (then or else branch)."""
        ds = self._get_dataset_structure(operand)
        if ds is None:
            return
        sql = self._get_dataset_sql(operand)
        ds_ids = set(ds.get_identifiers_names())
        common = [id_ for id_ in source_ids if id_ in ds_ids]
        on_parts = [
            f"{alias_src}.{quote_identifier(id_)} = {alias}.{quote_identifier(id_)}"
            for id_ in common
        ]
        if on_parts:
            builder.join(sql, alias, on=" AND ".join(on_parts), join_type="LEFT")

    def _build_dataset_case(self, node: AST.Case) -> str:
        """Build SQL for dataset-level CASE with JOINs."""
        source_node = self._find_condition_source(node.cases[0].condition)
        if source_node is None:
            return self._build_case_when_sql(node.cases, node.elseOp)
        source_ds = self._get_dataset_structure(source_node)
        source_sql = self._get_dataset_sql(source_node)
        if source_ds is None:
            return self._build_case_when_sql(node.cases, node.elseOp)

        source_ids = list(source_ds.get_identifiers_names())
        alias_src = "src"

        output_ds = self._get_output_dataset()
        output_measures = (
            list(output_ds.get_measures_names())
            if output_ds is not None
            else list(source_ds.get_measures_names())
        )

        builder = SQLBuilder().from_table(source_sql, alias_src)

        # Process each WHEN branch
        cond_exprs: List[str] = []
        then_aliases: List[Optional[str]] = []
        then_types: List[str] = []

        for i, case_obj in enumerate(node.cases):
            cond_expr = self._build_case_condition(
                case_obj, f"c{i}", source_ids, alias_src, builder
            )
            cond_exprs.append(cond_expr)

            t_type = self._get_operand_type(case_obj.thenOp)
            then_types.append(t_type)
            if t_type == _DATASET:
                t_alias = f"t{i}"
                self._join_dataset_operand(case_obj.thenOp, t_alias, source_ids, alias_src, builder)
                then_aliases.append(t_alias)
            else:
                then_aliases.append(None)

        # Handle else-operand
        else_type = self._get_operand_type(node.elseOp)
        else_alias: Optional[str] = None
        if else_type == _DATASET:
            else_alias = "e"
            self._join_dataset_operand(node.elseOp, else_alias, source_ids, alias_src, builder)

        # Build SELECT: identifiers + CASE WHEN per measure (reversed for last-match-wins)
        cols: List[str] = [f"{alias_src}.{quote_identifier(id_)}" for id_ in source_ids]
        for measure in output_measures:
            case_parts = ["CASE"]
            for i in reversed(range(len(node.cases))):
                then_ref = (
                    f"{then_aliases[i]}.{quote_identifier(measure)}"
                    if then_types[i] == _DATASET
                    else self.visit(node.cases[i].thenOp)
                )
                case_parts.append(f"WHEN {cond_exprs[i]} THEN {then_ref}")
            else_ref = (
                f"{else_alias}.{quote_identifier(measure)}"
                if else_type == _DATASET
                else self.visit(node.elseOp)
            )
            case_parts.append(f"ELSE {else_ref} END")
            cols.append(f"{' '.join(case_parts)} AS {quote_identifier(measure)}")

        builder.select(*cols)

        # Filter: only keep rows where the selected branch has a matching row.
        # Scalar/null branches always match; dataset branches need a LEFT JOIN hit.
        has_ds_branch = any(t == _DATASET for t in then_types) or else_type == _DATASET
        if has_ds_branch:
            id_col = quote_identifier(source_ids[0])
            filter_parts: List[str] = []
            for i in range(len(node.cases)):
                if then_types[i] == _DATASET:
                    match_check = f"{then_aliases[i]}.{id_col} IS NOT NULL"
                else:
                    match_check = "TRUE"
                filter_parts.append(f"({cond_exprs[i]} AND {match_check})")
            # Else branch: applies when no condition is true
            neg = " AND ".join(f"(NOT {c} OR {c} IS NULL)" for c in cond_exprs)
            if else_type == _DATASET:
                filter_parts.append(f"(({neg}) AND {else_alias}.{id_col} IS NOT NULL)")
            else:
                filter_parts.append(f"({neg})")
            builder.where(" OR ".join(filter_parts))

        return builder.build()

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
        # Temporarily clear output dataset to prevent _build_ds_ds_binary
        # from renaming measures to match the outer assignment.
        saved_assignment = self.current_assignment
        self.current_assignment = ""
        try:
            validation_sql = self.visit(node.validation)
        finally:
            self.current_assignment = saved_assignment

        error_code = f"'{node.error_code}'" if node.error_code else "CAST(NULL AS VARCHAR)"
        error_level = (
            str(node.error_level) if node.error_level is not None else "CAST(NULL AS BIGINT)"
        )

        # Discover the measure name produced by the inner comparison.
        ds = self._get_dataset_structure(node.validation)
        if ds is None:
            # Fallback: cannot determine structure – wrap as before.
            return (
                f'SELECT t.*, CAST(NULL AS DOUBLE) AS "imbalance", '
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

        # Handle imbalance (also with cleared output to prevent renaming).
        if node.imbalance is not None:
            self.current_assignment = ""
            try:
                imbalance_sql = self.visit(node.imbalance)
            finally:
                self.current_assignment = saved_assignment
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
                cols.append('CAST(NULL AS DOUBLE) AS "imbalance"')
        else:
            imbalance_sql = None
            join_cond = None
            cols.append('CAST(NULL AS DOUBLE) AS "imbalance"')

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
        # are treated as identifiers (not aliased as duplicates).
        # For cross joins, identifiers from different datasets must be qualified
        # (e.g. d1#Id_1, d2#Id_1), so we skip all identifier deduplication.
        all_join_ids: Set[str] = set()
        if join_type != "CROSS":
            for keys in pairwise_keys:
                all_join_ids.update(keys)
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
        target = node.period_to
        conf = node.conf  # "first", "last", or None

        if node.operand is not None:
            operand_type = self._get_operand_type(node.operand)

            # Dataset-level time_agg: apply to the time measure
            if operand_type == _DATASET:
                return self._visit_time_agg_dataset(node, target, conf)

            is_tp = self._is_time_period_operand(node.operand)
            operand_sql = self.visit(node.operand)

            if is_tp:
                return f"vtl_time_agg_tp(vtl_period_parse({operand_sql}), '{target}')"
            else:
                agg_expr = f"vtl_time_agg_date({operand_sql}, '{target}')"
                # For Date + conf, return start/end date of the computed period
                if conf == "first":
                    return f"vtl_tp_start_date(vtl_period_parse({agg_expr}))"
                elif conf == "last":
                    return f"vtl_tp_end_date(vtl_period_parse({agg_expr}))"
                return agg_expr
        else:
            # Without-operand case: inside group all, applies to time identifier
            if self._in_clause and self._current_dataset:
                for comp in self._current_dataset.components.values():
                    if comp.data_type == TimePeriod and comp.role == Role.IDENTIFIER:
                        col = quote_identifier(comp.name)
                        return f"vtl_time_agg_tp(vtl_period_parse({col}), '{target}')"
                for comp in self._current_dataset.components.values():
                    if comp.data_type == Date and comp.role == Role.IDENTIFIER:
                        col = quote_identifier(comp.name)
                        agg = f"vtl_time_agg_date({col}, '{target}')"
                        if conf == "first":
                            return f"vtl_tp_start_date(vtl_period_parse({agg}))"
                        elif conf == "last":
                            return f"vtl_tp_end_date(vtl_period_parse({agg}))"
                        return agg
            return f"vtl_time_agg_date(CURRENT_DATE, '{target}')"

    def _visit_time_agg_dataset(
        self, node: AST.TimeAggregation, target: str, conf: Optional[str]
    ) -> str:
        """Visit TIME_AGG at dataset level: apply to time measure."""
        if node.operand is None:
            raise ValueError("Cannot resolve structure for time_agg dataset")
        ds = self._get_dataset_structure(node.operand)
        src = self._get_dataset_sql(node.operand)
        if ds is None:
            raise ValueError("Cannot resolve structure for time_agg dataset")

        # Find time measures to transform
        cols = []
        for comp in ds.components.values():
            col = quote_identifier(comp.name)
            if comp.role == Role.IDENTIFIER:
                cols.append(col)
            elif comp.data_type == TimePeriod:
                cols.append(f"vtl_time_agg_tp(vtl_period_parse({col}), '{target}') AS {col}")
            elif comp.data_type == Date:
                agg = f"vtl_time_agg_date({col}, '{target}')"
                if conf == "first":
                    expr = f"vtl_tp_start_date(vtl_period_parse({agg}))"
                elif conf == "last":
                    expr = f"vtl_tp_end_date(vtl_period_parse({agg}))"
                else:
                    expr = agg
                cols.append(f"{expr} AS {col}")
            else:
                cols.append(col)

        return f"SELECT {', '.join(cols)} FROM {src}"

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
        query = routine.query

        # Convert double-quoted strings to single-quoted strings.
        # In standard SQL (and DuckDB), double quotes delimit identifiers,
        # but external routines written for SQLite use them for string literals.
        query = re.sub(r'"([^"]*)"', r"'\1'", query)

        # Map SQL table names to actual DuckDB table names.
        # Operands may have module prefixes (e.g. C07.MSMTCH_BL_DS) while
        # the SQL query references the short name (MSMTCH_BL_DS).
        operand_names: List[str] = []
        for operand in node.operands:
            if isinstance(operand, (AST.Identifier, AST.VarID)):
                operand_names.append(operand.value)
            else:
                operand_names.append(str(self.visit(operand)))

        for sql_table_name in routine.dataset_names:
            for op_name in operand_names:
                short_name = op_name.split(".")[-1] if "." in op_name else op_name
                if short_name == sql_table_name:
                    query = re.sub(
                        rf"\b{re.escape(sql_table_name)}\b",
                        quote_identifier(op_name),
                        query,
                    )
                    break

        return query

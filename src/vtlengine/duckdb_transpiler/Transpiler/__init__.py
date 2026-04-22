"""Transpile VTL AST nodes into DuckDB SQL."""

import re
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar import tokens
from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    TimeInterval,
    TimePeriod,
)
from vtlengine.duckdb_transpiler.Transpiler.operators import (
    _ORDERING_OPS,
    _STRING_PARAM_OPS,
    _STRING_UNARY_OPS,
    get_duckdb_type,
    registry,
)
from vtlengine.duckdb_transpiler.Transpiler.sql_builder import (
    CTEBuilder,
    SQLBuilder,
    quote_name,
)
from vtlengine.duckdb_transpiler.Transpiler.structure_visitor import (
    _COMPONENT,
    _DATASET,
    _SCALAR,
    StructureVisitor,
)
from vtlengine.Exceptions import RunTimeError
from vtlengine.Model import Component, Dataset, ExternalRoutine, Role, Scalar, ValueDomain


def _datediff_to_date(ref: str, dt: Optional[type]) -> str:
    """Convert a datediff operand to a DATE expression based on its VTL type."""
    if dt == TimePeriod:
        return f"vtl_tp_end_date(vtl_period_parse({ref}))"
    if dt == Date:
        return f"CAST({ref} AS DATE)"
    # TimeInterval or unknown: pass through (NULL propagates, non-null errors at runtime)
    return f"CAST({ref} AS DATE)"


def _add_tp_indicator_check(sql: str, table_src: str, tp_cols: List[tuple[str, str]]) -> str:
    """Add a TimePeriod indicator consistency check to an aggregate query."""
    checks: List[str] = []
    for col_name, agg_op in tp_cols:
        qc = quote_name(col_name)
        indicator = f"vtl_period_parse({qc}).period_indicator"
        err = (
            f"'VTL Error 2-1-19-20: Time Period operands with "
            f"different period indicators do not support < and > "
            f"Comparison operations, unable to get the {agg_op}'"
        )
        checks.append(
            f"CASE WHEN COUNT(DISTINCT {indicator}) "
            f"FILTER (WHERE {qc} IS NOT NULL) > 1 "
            f"THEN error({err}) ELSE 1 END"
        )
    check_cols = ", ".join(f"{c} AS _ok{i}" for i, c in enumerate(checks))
    subquery = f"(SELECT {check_cols} FROM {table_src}) AS _vtl_tp_check"
    where_conds = " AND ".join(f"_vtl_tp_check._ok{i} = 1" for i in range(len(checks)))
    from_pattern = f"FROM {table_src}"
    return sql.replace(from_pattern, f"FROM {table_src}, {subquery} WHERE {where_conds}", 1)


def _is_date_timeperiod_pair(left_type: Optional[type], right_type: Optional[type]) -> bool:
    """Return True when types are a Date and a TimePeriod."""
    return {left_type, right_type} == {Date, TimePeriod}


def _date_tp_compare_expr(
    left_ref: str,
    right_ref: str,
    left_type: type,
    right_type: type,
    op: str,
) -> str:
    """Build SQL for Date vs TimePeriod comparison using TimeInterval promotion."""
    if left_type == Date:
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
    return registry.generate(op, left_interval, right_interval)


def _bool_to_str(col_ref: str) -> str:
    """Cast a Boolean expression to Python-style string values."""
    return f"CASE WHEN {col_ref} IS NULL THEN NULL WHEN {col_ref} THEN 'True' ELSE 'False' END"


@dataclass
class _ParsedHRRule:
    """Parsed pieces of a hierarchical rule."""

    has_when: bool
    when_node: Any  # AST node for the WHEN condition, or None
    comparison_node: Any  # AST node for the comparison (left = right)
    left_code_item: str  # Left-side code item name
    right_expr_node: AST.AST  # Right-side expression AST
    right_code_items: List[str]  # All code item names in the right-side expression


@dataclass
class SQLTranspiler(StructureVisitor, ASTTemplate):
    """Transpiler that converts VTL AST nodes to SQL queries."""

    # Input structures
    input_datasets: Dict[str, Dataset] = field(default_factory=dict)
    input_scalars: Dict[str, Scalar] = field(default_factory=dict)

    # Output structures
    output_datasets: Dict[str, Dataset] = field(default_factory=dict)
    output_scalars: Dict[str, Scalar] = field(default_factory=dict)

    value_domains: Dict[str, ValueDomain] = field(default_factory=dict)
    external_routines: Dict[str, ExternalRoutine] = field(default_factory=dict)

    # Dependency graph
    dag: Any = field(default=None)

    # cast(time_period, string) format
    time_period_output_format: str = field(default="vtl")

    # Runtime context
    current_assignment: str = ""
    inputs: List[str] = field(default_factory=list)
    clause_context: List[str] = field(default_factory=list)

    # Merged lookups
    datasets: Dict[str, Dataset] = field(default_factory=dict, init=False)
    scalars: Dict[str, Scalar] = field(default_factory=dict, init=False)
    available_tables: Dict[str, Dataset] = field(default_factory=dict, init=False)

    # Clause context
    _in_clause: bool = field(default=False, init=False)
    _current_dataset: Optional[Dataset] = field(default=None, init=False)
    _column_prefix: Optional[str] = field(default=None, init=False)

    # Join context: "alias#comp" -> SQL column name
    _join_alias_map: Dict[str, str] = field(default_factory=dict, init=False)

    # Qualified names consumed by join clauses
    _consumed_join_aliases: Set[str] = field(default_factory=set, init=False)

    # UDO definitions
    _udos: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # UDO parameter stack
    _udo_params: Optional[List[Dict[str, Any]]] = field(default=None, init=False)

    # Datapoint rulesets
    _dprs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    # Datapoint ruleset context
    _dp_signature: Optional[Dict[str, str]] = field(default=None, init=False)

    # Hierarchical rulesets
    _hrs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize available tables."""
        self.datasets = {**self.input_datasets, **self.output_datasets}
        self.scalars = {**self.input_scalars, **self.output_scalars}
        self.available_tables = dict(self.datasets)

    # Helper methods

    @contextmanager
    def _clause_scope(
        self,
        ds: Optional[Dataset] = None,
        prefix: Optional[str] = None,
    ) -> Generator[None, None, None]:
        """Temporarily set clause state and restore it on exit."""
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

    @contextmanager
    def _stash_assignment(self) -> Generator[None, None, None]:
        """Temporarily stash the ``current_assignment`` and restore it on exit."""
        saved = self.current_assignment
        self.current_assignment = ""
        try:
            yield
        finally:
            self.current_assignment = saved

    def _resolve_clause_dataset(
        self, node: AST.RegularAggregation
    ) -> Optional[Tuple[Dataset, str]]:
        """Resolve and return (dataset, table_src) for a clause node."""
        if not node.dataset:
            return None
        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)
        if ds is None:
            return None
        return ds, table_src

    def _clause_fallback_sql(self, node: AST.RegularAggregation) -> str:
        """Return ``SELECT * FROM <dataset>`` or empty when the clause cannot resolve."""
        if node.dataset:
            return f"SELECT * FROM {self._get_dataset_sql(node.dataset)}"
        return ""

    def _get_assignment_inputs(self, name: str) -> List[str]:
        if self.dag is None:
            return []
        if hasattr(self.dag, "dependencies"):
            for deps in self.dag.dependencies.values():
                if name in deps.outputs or name in deps.persistent:
                    return deps.inputs
        return []

    # Top-level visitors

    def transpile(self, node: AST.Start) -> List[Tuple[str, str, bool]]:
        """Return (name, sql, is_persistent) tuples for the script."""
        return self.visit(node)

    def visit_Start(self, node: AST.Start) -> List[Tuple[str, str, bool]]:
        """Generate SQL for top-level nodes."""
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

                is_persistent = isinstance(child, AST.PersistentAssignment)
                if name in self.output_scalars:
                    value_sql = self.visit(child)
                    if not value_sql.strip().upper().startswith("SELECT"):
                        value_sql = f"SELECT {value_sql} AS value"
                    queries.append((name, value_sql, is_persistent))
                else:
                    query = self.visit(child)
                    query = self._unqualify_join_columns(name, query)
                    queries.append((name, query, is_persistent))

                self._join_alias_map = {}
                self._consumed_join_aliases = set()

        return queries

    def _unqualify_join_columns(self, ds_name: str, query: str) -> str:
        """Rename remaining alias#comp columns to plain component names."""
        if not self._join_alias_map:
            return query

        output_ds = self.output_datasets.get(ds_name)
        if output_ds is None:
            return query

        output_comp_names = set(output_ds.components.keys())
        unqual_to_qual: Dict[str, str] = {}
        for qualified in self._join_alias_map:
            if qualified in self._consumed_join_aliases or qualified in output_comp_names:
                continue
            if "#" in qualified:
                unqualified = qualified.split("#", 1)[1]
                if unqualified in output_comp_names and unqualified not in unqual_to_qual:
                    unqual_to_qual[unqualified] = qualified

        if not unqual_to_qual:
            return query

        cols: List[str] = []
        for comp_name in output_ds.components:
            qual = unqual_to_qual.get(comp_name)
            if qual is not None:
                cols.append(f"{quote_name(qual)} AS {quote_name(comp_name)}")
            else:
                cols.append(quote_name(comp_name))

        return f"SELECT {', '.join(cols)} FROM ({query})"

    def visit_Assignment(self, node: AST.Assignment) -> str:
        """Visit an assignment and return the SQL for its right-hand side."""
        return self.visit(node.right)

    visit_PersistentAssignment = visit_Assignment

    # Datapoint ruleset and validation

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        """Register a datapoint ruleset definition."""
        signature: Dict[str, str] = {}
        if not isinstance(node.params, AST.DefIdentifier):
            for param in node.params:
                alias = param.alias if param.alias is not None else param.value
                signature[alias] = param.value

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

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        self._get_output_dataset()
        output_mode = node.output.value if node.output else "invalid"

        id_cols = ds.get_identifiers_names()
        measure_cols = ds.get_measures_names()

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
            cols = [quote_name(c) for c in id_cols]
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
        self._dp_signature = signature
        rule_name = rule.name or ""
        rule_node = rule.rule
        ec_sql = self._error_code_sql(rule.erCode)
        el_sql = self._error_code_sql(rule.erLevel)
        if isinstance(rule_node, AST.HRBinOp) and rule_node.op == tokens.WHEN:
            when_cond_sql = self._visit_dp_expr(rule_node.left, signature)
            then_expr_sql = self._visit_dp_expr(rule_node.right, signature)
            fail_cond = f"({when_cond_sql}) AND NOT ({then_expr_sql})"
            bool_expr = f"""
                CASE WHEN ({when_cond_sql}) THEN ({then_expr_sql})
                 WHEN NOT ({when_cond_sql}) THEN TRUE ELSE NULL END
            """
        else:
            then_expr_sql = self._visit_dp_expr(rule_node, signature)
            fail_cond = f"NOT ({then_expr_sql})"
            bool_expr = f"({then_expr_sql})"

        self._dp_signature = None
        select_parts = [quote_name(c) for c in id_cols + measure_cols]
        if output_mode == "invalid":
            select_parts.append(f"'{rule_name}' AS {quote_name('ruleid')}")
            select_parts.append(f"{ec_sql} AS {quote_name('errorcode')}")
            select_parts.append(f"{el_sql} AS {quote_name('errorlevel')}")
            return f"SELECT {', '.join(select_parts)} FROM {table_src} WHERE {fail_cond}"

        select_parts.append(f"{bool_expr} AS {quote_name('bool_var')}")
        select_parts.append(f"'{rule_name}' AS {quote_name('ruleid')}")
        for val, col in [(ec_sql, quote_name("errorcode")), (el_sql, quote_name("errorlevel"))]:
            select_parts.append(f"CASE WHEN {fail_cond} THEN {val} ELSE NULL END AS {col}")
        return f"SELECT {', '.join(select_parts)} FROM {table_src}"

    def _visit_dp_expr(self, node: AST.AST, signature: Dict[str, str]) -> str:
        """Visit an expression in datapoint-rule context."""
        if isinstance(node, (AST.HRBinOp, AST.BinOp)):
            left_sql = self._visit_dp_expr(node.left, signature)
            right_sql = self._visit_dp_expr(node.right, signature)
            if isinstance(node, AST.HRBinOp) and node.op == tokens.WHEN:
                return f"CASE WHEN ({left_sql}) THEN ({right_sql}) ELSE TRUE END"
            return registry.generate(node.op, left_sql, right_sql)
        if isinstance(node, (AST.HRUnOp, AST.UnaryOp)):
            operand_sql = self._visit_dp_expr(node.operand, signature)
            return registry.generate(node.op, operand_sql)
        if isinstance(node, (AST.DefIdentifier, AST.VarID)):
            col_name = signature.get(node.value, node.value)
            return quote_name(col_name)
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
        saved_sig = self._dp_signature
        self._dp_signature = signature
        result = self.visit(node)
        self._dp_signature = saved_sig
        return result

    # Hierarchical ruleset and check_hierarchy

    def _visit_HRuleset(self, node: AST.HRuleset) -> None:
        """Register a hierarchical ruleset definition."""
        rule_names = [r.name for r in node.rules if r.name is not None]
        if not rule_names:
            for i, rule in enumerate(node.rules):
                rule.name = str(i + 1)

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

        ds = self._get_dataset_structure(node.dataset)
        table_src = self._get_dataset_sql(node.dataset)

        self._get_output_dataset()

        if hr_info["signature_type"] == "valuedomain" and node.rule_component is not None:
            component: str = node.rule_component.value  # type: ignore[attr-defined]
        else:
            component = hr_info["signature"]

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

    def _error_code_sql(self, value: Any) -> str:
        """Convert an errorcode value to a SQL literal."""
        return "CAST(NULL AS VARCHAR)" if value is None else self._to_sql_literal(value=value)

    def _get_node_value(self, node: Any) -> str:
        """Extract ``.value`` from an AST node, falling back to ``str(node)``."""
        return node.value if hasattr(node, "value") else str(node)

    def _unwrap_assignment(self, child: AST.AST) -> AST.AST:
        """Return the inner ``Assignment`` from ``UnaryOp(Assignment)`` wrappers."""
        return child.operand if isinstance(child, AST.UnaryOp) else child

    def _is_numeric(self, value: Any) -> bool:
        """Return True if ``value`` is ``None`` or coerces to ``float`` without error."""
        try:
            float(value)
        except (ValueError, TypeError):
            return False
        return True

    def _as_subquery(self, src: str) -> str:
        """Wrap *src* as a parenthesized subquery, adding ``SELECT *`` if needed."""
        stripped = src.strip().upper()
        if stripped.startswith("("):
            return src
        if stripped.startswith("SELECT"):
            return f"({src})"
        return f"(SELECT * FROM {src})"

    def _resolve_udo_name(self, raw_name: str) -> str:
        """Resolve a potential UDO parameter to its actual name."""
        udo_val = self._get_udo_param(raw_name)
        if isinstance(udo_val, (AST.VarID, AST.Identifier)):
            return udo_val.value
        return udo_val if udo_val is not None else raw_name

    def _is_hr_eq_rule(self, rule: AST.HRule) -> bool:
        """Check if a hierarchical rule is an EQ rule (or WHEN-EQ)."""
        node = rule.rule
        if not isinstance(node, AST.HRBinOp):
            return False
        if node.op == tokens.WHEN and isinstance(node.right, AST.HRBinOp):
            return node.right.op == tokens.EQ
        return node.op == tokens.EQ

    def _parse_hr_rule(self, rule: AST.HRule) -> _ParsedHRRule:
        """Parse a hierarchical rule into its constituent parts."""
        rule_node: Any = rule.rule
        has_when = isinstance(rule_node, AST.HRBinOp) and rule_node.op == tokens.WHEN
        when_node = rule_node.left if has_when else None
        comparison_node = rule_node.right if has_when else rule_node
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
        """Collect deduplicated code items and conditions across rules."""
        all_items: List[str] = []
        all_conds: Dict[str, str] = {}
        for rule in rules:
            parsed = self._parse_hr_rule(rule)
            all_items.append(parsed.left_code_item)
            all_items.extend(parsed.right_code_items)
            rc = getattr(parsed.comparison_node.left, "_right_condition", None)
            if rc is not None:
                all_conds[parsed.left_code_item] = self._build_hr_when_sql(rc, cond_mapping)
            _, right_conds = self._collect_hr_code_items(parsed.right_expr_node, cond_mapping)
            all_conds.update(right_conds)
        return list(dict.fromkeys(all_items)), all_conds

    def _build_hr_pivot(
        self,
        table_src: str,
        ds: Dataset,
        rules: list,  # type: ignore[type-arg]
        rule_comp: str,
        cond_mapping: Dict[str, str],
    ) -> Tuple[str, str, List[str], List[str], Dict[str, str]]:
        """Build the pivot SELECT SQL and metadata for hierarchy operations.

        Returns (pivot_sql, measure_name, other_ids, unique_items, item_conds).
        The pivot_sql is a plain SELECT (not wrapped in CTE syntax).
        """
        measure_name = ds.get_measures_names()[0]
        other_ids = [n for n in ds.get_identifiers_names() if n != rule_comp]
        unique_items, item_conds = self._collect_all_hr_items(rules, cond_mapping)

        qrc = quote_name(rule_comp)
        qm = quote_name(measure_name)

        group_cols = [quote_name(c) for c in (*other_ids, *cond_mapping.values())]

        select_parts = list(group_cols)
        for ci in unique_items:
            ci_cond = ""
            if item_conds and ci in item_conds:
                ci_cond = f" AND {item_conds[ci]}"
            select_parts.append(
                f"MAX(CASE WHEN {qrc} = '{ci}'{ci_cond} THEN {qm} END) AS _val_{ci}"
            )
            select_parts.append(
                f"MAX(CASE WHEN {qrc} = '{ci}'{ci_cond} THEN 1 ELSE 0 END) AS _has_{ci}"
            )

        in_list = ", ".join(f"'{ci}'" for ci in unique_items)
        group_by = f" GROUP BY {', '.join(group_cols)}" if group_cols else ""

        pivot_sql = (
            f"SELECT {', '.join(select_parts)} "
            f"FROM {table_src} WHERE {qrc} IN ({in_list}){group_by}"
        )
        return pivot_sql, measure_name, other_ids, unique_items, item_conds

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
            cols = [quote_name(c) for c in (out_ds.components if out_ds else ds.components)]
            return f"SELECT {', '.join(cols)} FROM {table_src} WHERE 1=0"

        pivot_sql, measure_name, other_ids, _, _ = self._build_hr_pivot(
            table_src, ds, rules, rule_comp, cond_mapping
        )
        cte = CTEBuilder()
        cte.cte("_pivot", pivot_sql)
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
        return cte.select(" UNION ALL ".join(rule_queries))

    def _collect_hr_code_items(
        self,
        node: AST.AST,
        cond_mapping: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[str], Dict[str, str]]:
        """Extract code-item names and right-side conditions from an HR expression."""
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
        if mode in ("always_zero", "non_zero", "partial_zero"):
            has_col = f"_has_{code_item}"
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
        # HRUnOp
        operand_sql = self._build_hr_expr_sql(node.operand, mode)
        return f"({node.op}{operand_sql})"

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

        l_val = self._build_hr_value_expr(parsed.left_code_item, mode)
        r_val = self._build_hr_expr_sql(parsed.right_expr_node, mode)

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

        ec_sql = self._error_code_sql(rule.erCode)
        el_sql = self._error_code_sql(rule.erLevel)
        el_null = (
            "CAST(NULL AS DOUBLE)" if self._is_numeric(rule.erLevel) else "CAST(NULL AS VARCHAR)"
        )

        q_rc = quote_name(rule_comp)
        q_m = quote_name(measure)
        select_parts: List[str] = [quote_name(c) for c in other_ids]
        select_parts.append(f"'{parsed.left_code_item}' AS {q_rc}")

        if output != "all":
            select_parts.append(f"{l_val} AS {q_m}")
        if output != "invalid":
            select_parts.append(f"{bool_expr} AS {quote_name('bool_var')}")

        select_parts.append(f"{imbalance_expr} AS {quote_name('imbalance')}")
        select_parts.append(f"'{rule_name}' AS {quote_name('ruleid')}")

        if output == "invalid":
            select_parts.append(f"{ec_sql} AS {quote_name('errorcode')}")
            select_parts.append(f"{el_sql} AS {quote_name('errorlevel')}")
        else:
            select_parts.append(
                f"CASE WHEN {bool_expr} IS NOT FALSE THEN CAST(NULL AS VARCHAR) "
                f"ELSE {ec_sql} END AS {quote_name('errorcode')}"
            )
            select_parts.append(
                f"CASE WHEN {bool_expr} IS NOT FALSE THEN {el_null} "
                f"ELSE {el_sql} END AS {quote_name('errorlevel')}"
            )

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
        """Generate SQL for hierarchy operator using CTE chain."""
        if not rules:
            cols = [quote_name(c) for c in ds.get_components_names()]
            return f"SELECT {', '.join(cols)} FROM {table_src}"

        pivot_sql, measure, other_ids, unique_items, _ = self._build_hr_pivot(
            table_src, ds, rules, rule_comp, cond_mapping
        )
        cte = CTEBuilder()
        cte.cte("_pivot", pivot_sql)
        rule_result_refs: List[Tuple[str, str]] = []
        current_pivot = "_pivot"

        join_keys = [quote_name(c) for c in (*other_ids, *cond_mapping.values())]

        for i, rule in enumerate(rules):
            parsed = self._parse_hr_rule(rule)

            rule_cte_name = f"_rule_{i}"
            cte.cte(
                rule_cte_name,
                self._build_hierarchy_rule_cte(
                    parsed=parsed,
                    pivot_ref=current_pivot,
                    other_ids=other_ids,
                    mode=mode,
                    cond_mapping=cond_mapping,
                ),
            )
            rule_result_refs.append((rule_cte_name, parsed.left_code_item))

            next_pivot = f"_pivot_{i}"
            cte.cte(
                next_pivot,
                self._build_hierarchy_pivot_update(
                    prev_pivot=current_pivot,
                    rule_cte=rule_cte_name,
                    left_code_item=parsed.left_code_item,
                    join_keys=join_keys,
                    input_mode=input_mode,
                    unique_items=unique_items,
                ),
            )
            current_pivot = next_pivot

        # Build final SELECT per rule
        final_selects: List[str] = []
        q_rc = quote_name(rule_comp)
        q_m = quote_name(measure)
        for rule_cte, left_ci in rule_result_refs:
            cols = [quote_name(c) for c in other_ids]
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
            return cte.select(computed_sql)

        # output == "all": union(setdiff(op, computed), computed)
        id_cols = [quote_name(c) for c in ds.get_identifiers_names()]
        all_cols = [quote_name(c) for c in ds.get_components_names()]
        all_cols_csv = ", ".join(all_cols)
        id_cols_csv = ", ".join(id_cols)
        cte.cte("_computed", computed_sql)
        cte.cte(
            "_combined",
            f"SELECT {all_cols_csv}, 0 AS _src FROM {table_src} "
            f"UNION ALL SELECT {all_cols_csv}, 1 AS _src FROM _computed",
        )
        return cte.select(
            f"SELECT {all_cols_csv} FROM ("
            f"SELECT *, ROW_NUMBER() OVER ("
            f"PARTITION BY {id_cols_csv} ORDER BY _src DESC) AS _rn "
            f"FROM _combined) WHERE _rn = 1"
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

        select_parts = [quote_name(c) for c in (*other_ids, *cond_mapping.values())]
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
        for i in unique_items:
            if i != left_code_item:
                other_val_has.append(f"p._val_{i}")
                other_val_has.append(f"p._has_{i}")

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
            for i in items:
                filters.append(f"_val_{i} IS NOT NULL")

        elif mode == "non_zero":
            if is_hierarchy:
                zero_checks = []
                for i in right_code_items:
                    val = self._build_hr_value_expr(i, mode)
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
            checks = [f"(_has_{i} = 1 AND _val_{i} IS NOT NULL)" for i in items]
            if checks:
                filters.append(f"({' OR '.join(checks)})")

        elif mode in ("always_null", "always_zero"):
            presence = [f"_has_{i} = 1" for i in all_items]
            filters.append(f"({' OR '.join(presence)})")

        return filters

    def _build_hr_when_sql(self, node: AST.AST, cond_mapping: Dict[str, str]) -> str:
        """Generate SQL for a WHEN condition in a hierarchical rule."""
        if isinstance(node, (AST.DefIdentifier, AST.VarID)):
            col_name = cond_mapping.get(node.value, node.value)
            return quote_name(col_name)
        if isinstance(node, AST.Constant):
            return self._to_sql_literal(node.value)
        if isinstance(node, (AST.HRUnOp, AST.UnaryOp)):
            operand_sql = self._build_hr_when_sql(node.operand, cond_mapping)
            return registry.generate(node.op, operand_sql)
        if isinstance(node, (AST.HRBinOp, AST.BinOp)):
            left_sql = self._build_hr_when_sql(node.left, cond_mapping)
            right_sql = self._build_hr_when_sql(node.right, cond_mapping)
            return registry.generate(node.op, left_sql, right_sql)
        if isinstance(node, AST.MulOp):
            children_sql = [self._build_hr_when_sql(c, cond_mapping) for c in node.children]
            return registry.generate(node.op, *children_sql)
        # Fallback to general visitor.
        return self.visit(node)

    # UDO definition and call

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
        udo_def = self._udos[node.op]
        params = udo_def["params"]
        expression = deepcopy(udo_def["expression"])

        bindings: Dict[str, Any] = {}
        for i, param_info in enumerate(params):
            param_name = param_info["name"]
            if i < len(node.params):
                bindings[param_name] = node.params[i]
            elif param_info.get("default") is not None:
                bindings[param_name] = param_info["default"]
            bindings[f"__type__{param_name}"] = param_info.get("type")

        self._push_udo_params(bindings)
        try:
            result = self.visit(expression)
        finally:
            self._pop_udo_params()

        return result

    # Leaf visitors

    def _scalar_literal(self, name: str) -> str:
        sc = self.scalars[name]
        return self._to_sql_literal(sc.value, getattr(sc.data_type, "__name__", ""))

    def _resolve_udo_param(self, name: str, udo_val: Any) -> str:
        if not isinstance(udo_val, AST.VarID):
            return self.visit(udo_val) if isinstance(udo_val, AST.AST) else quote_name(udo_val)
        resolved = udo_val.value
        is_component = isinstance(self._get_udo_param(f"__type__{name}"), Component)
        if resolved in self.available_tables and not is_component:
            return f"SELECT * FROM {quote_name(resolved)}"
        if resolved in self.scalars:
            return self._scalar_literal(resolved)
        if resolved != name:
            return self.visit(udo_val)
        return quote_name(resolved)

    def _resolve_clause_component(self, name: str) -> Optional[str]:
        if not (self._in_clause and self._current_dataset):
            return None
        if name in self._current_dataset.components:
            return quote_name(name)
        matches = [
            c for c in self._current_dataset.components if "#" in c and c.split("#", 1)[1] == name
        ]
        return quote_name(matches[0]) if len(matches) == 1 else None

    def visit_VarID(self, node: AST.VarID) -> str:  # type: ignore[override]
        """Visit a variable identifier."""
        name = node.value

        udo_val = self._get_udo_param(name)
        if udo_val is not None:
            return self._resolve_udo_param(name, udo_val)

        if name in self.scalars:
            return self._scalar_literal(name)

        clause_match = self._resolve_clause_component(name)
        if clause_match is not None:
            return clause_match

        if name in self.available_tables:
            return f"SELECT * FROM {quote_name(name)}"

        return quote_name(name)

    def visit_Constant(self, node: AST.Constant) -> str:  # type: ignore[override]
        """Visit a constant literal."""
        return self._constant_to_sql(node)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        """Visit a parameter constant."""
        return str(node.value)

    def visit_Identifier(self, node: AST.Identifier) -> str:
        """Visit an identifier node."""
        return quote_name(node.value)

    def visit_ID(self, node: AST.ID) -> str:  # type: ignore[override]
        """Visit an ID node."""
        return node.value

    def visit_ParFunction(self, node: AST.ParFunction) -> str:  # type: ignore[override]
        """Visit a parenthesized function/expression."""
        return self.visit(node.operand)

    def visit_Collection(self, node: AST.Collection) -> str:  # type: ignore[override]
        """Visit a Collection (Set or ValueDomain reference)."""
        if node.kind == "ValueDomain":
            return self._visit_value_domain(node)
        values = [self._visit_collection_element(child) for child in node.children]
        return f"({', '.join(values)})"

    def _visit_collection_element(self, child: AST.AST) -> str:
        """Visit a set element, preserving raw CAST behavior for time_period literals."""
        if isinstance(child, AST.ParamOp) and child.op == tokens.CAST:
            type_node = child.children[1]
            if type_node == TimePeriod:
                source_type = self._get_source_vtl_type(child.children[0])
                if source_type not in (Date, TimeInterval):
                    operand_sql = self.visit(child.children[0])
                    return f"CAST({operand_sql} AS VARCHAR)"
        return self.visit(child)

    def _visit_value_domain(self, node: AST.Collection) -> str:
        """Resolve a ValueDomain reference to SQL literal list."""
        vd = self.value_domains[node.name]
        type_name = vd.type.__name__ if hasattr(vd.type, "__name__") else str(vd.type)
        literals = [self._to_sql_literal(v, type_name) for v in vd.setlist]
        return f"({', '.join(literals)})"

    # Generic dataset-level helpers

    def _apply_measures(
        self,
        ds_node: AST.AST,
        expr_fn: "Callable[[str], str]",
        output_name_override: Optional[str] = None,
        cast_bool_to_str: bool = False,
    ) -> str:
        """Apply an expression to each dataset measure and pass identifiers through."""
        ds = self._get_dataset_structure(ds_node)
        table_src = self._get_dataset_sql(ds_node)
        output_ds = self._get_output_dataset()
        output_measures = list(output_ds.get_measures_names()) if output_ds else []

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_name(name))
            elif comp.role == Role.MEASURE:
                col_ref = quote_name(name)
                if cast_bool_to_str and comp.data_type == Boolean:
                    col_ref = _bool_to_str(col_ref)
                expr = expr_fn(col_ref)

                out_name = name
                if output_name_override is not None:
                    out_name = output_name_override
                elif len(output_measures) == 1 and (
                    ds.name not in self.input_datasets
                    or name in self.input_datasets[ds.name].get_measures_names()
                ):
                    out_name = output_measures[0]
                cols.append(f"{expr} AS {quote_name(out_name)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    # Dataset-level binary helpers

    @staticmethod
    def _build_agg_expr(
        op: str, col_ref: str, data_type: Optional[type], *, dataset_level: bool = False
    ) -> Optional[str]:
        """Build a type-aware aggregate expression for MIN/MAX on Duration/TimePeriod.

        Returns None when the standard ``registry.generate`` path should be used.

        Args:
            op: Aggregate operator token (e.g. ``tokens.MIN``).
            col_ref: Quoted column reference.
            data_type: Component data type, or None.
            dataset_level: True for dataset-level aggregation (normalizes TimePeriod
                and wraps with ``vtl_period_to_string``); False for clause-context
                aggregation (uses ``ARG_MIN``/``ARG_MAX``).
        """
        if op not in (tokens.MIN, tokens.MAX) or data_type is None:
            return None
        if data_type == Duration:
            return f"vtl_int_to_duration({op.upper()}(vtl_duration_to_int({col_ref})))"
        if data_type == TimePeriod:
            parsed = f"vtl_period_parse({col_ref})"
            if dataset_level:
                return f"vtl_period_to_string({op.upper()}({parsed}))"
            return f"ARG_{op.upper()}({col_ref}, {parsed})"
        return None

    @staticmethod
    def _join_on_clause(common_ids: List[str], left_alias: str, right_alias: str) -> str:
        """Build ``a."Id" = b."Id" AND ...`` for a JOIN ON or WHERE clause."""
        if not common_ids:
            return "1=1"
        return " AND ".join(
            f"{left_alias}.{quote_name(i)} = {right_alias}.{quote_name(i)}" for i in common_ids
        )

    def _left_join_dataset(
        self,
        operand: AST.AST,
        operand_type: str,
        alias: str,
        source_ids: List[str],
        source_alias: str,
        builder: "SQLBuilder",
    ) -> Optional[str]:
        """LEFT JOIN a dataset operand and return a ref to its first ID (for filtering)."""
        if operand_type != _DATASET:
            return None
        sql = self._get_dataset_sql(operand)
        ds = self._get_dataset_structure(operand)
        ds_ids = set(ds.get_identifiers_names()) if ds else set()
        common = [id_ for id_ in source_ids if id_ in ds_ids]
        if not common:
            return None
        on = self._join_on_clause(common, source_alias, alias)
        builder.join(sql, alias, on=on, join_type="LEFT")
        return f"{alias}.{quote_name(common[0])}"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:  # type: ignore[override]
        """Visit a unary operation."""
        op = node.op
        if op == tokens.PERIOD_INDICATOR:
            return self._visit_period_indicator(node)
        if op in (tokens.FLOW_TO_STOCK, tokens.STOCK_TO_FLOW):
            return self._visit_flow_stock(node, op)

        operand_type = self._get_node_type(node.operand)
        if operand_type == _DATASET:
            ds = self._get_dataset_structure(node.operand)
            name_override: Optional[str] = None
            if op == tokens.ISNULL and ds and len(ds.get_measures_names()) == 1:
                name_override = "bool_var"

            def _unary_expr(col_ref: str) -> str:
                comp = ds.components.get(col_ref.strip('"')) if ds else None
                dt = comp.data_type if comp else None
                return registry.generate(op, col_ref, data_type=dt)

            bool_to_str = op in _STRING_UNARY_OPS
            return self._apply_measures(node.operand, _unary_expr, name_override, bool_to_str)
        else:
            dt = self._detect_scalar_type(node.operand)
            operand_sql = self.visit(node.operand)
            return registry.generate(op, operand_sql, data_type=dt)

    def visit_BinOp(self, node: AST.BinOp) -> str:  # type: ignore[override]
        """Visit a binary operation."""
        op = node.op
        if op == tokens.MEMBERSHIP:
            return self._visit_membership(node)
        if op == tokens.EXISTS_IN:
            return self._visit_exists_in(node.left, node.right)
        if op == tokens.CHARSET_MATCH:
            return self._visit_match_characters(node)
        if op == tokens.RANDOM:
            return self._visit_random_binop(node)
        if op == tokens.TIMESHIFT:
            return self._visit_timeshift(node)

        left_type = self._get_node_type(node.left)
        right_type = self._get_node_type(node.right)
        if left_type == _DATASET or right_type == _DATASET:
            if op in (tokens.IN, tokens.NOT_IN) and left_type == _DATASET:
                collection = self.visit(node.right)

                def _in_expr(col_ref: str) -> str:
                    return f"({col_ref} {'IN' if op == tokens.IN else 'NOT IN'} {collection})"

                return self._apply_measures(node.left, _in_expr, output_name_override="bool_var")
            if left_type == _DATASET and right_type == _DATASET:
                return self._build_ds_ds_binary(node.left, node.right, op)
            if left_type == _DATASET:
                return self._build_ds_scalar_binary(node.left, node.right, op, ds_on_left=True)
            return self._build_ds_scalar_binary(node.right, node.left, op, ds_on_left=False)

        # Scalar-scalar binary: detect types and delegate to _make_binary_expr
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)
        left_dt = self._detect_scalar_type(node.left)
        right_dt = self._detect_scalar_type(node.right)
        return self._make_binary_expr(left_sql, right_sql, op, left_dt, right_dt)

    def _make_binary_expr(
        self,
        left_ref: str,
        right_ref: str,
        op: str,
        left_type: Optional[type] = None,
        right_type: Optional[type] = None,
    ) -> str:
        """Build a binary SQL expression with type-aware registry dispatch."""
        dt = left_type or right_type
        # TimeInterval: ordering not supported
        if op in _ORDERING_OPS and dt == TimeInterval:
            raise RunTimeError("2-1-19-17", op=op)
        # datediff: convert each operand to DATE individually based on its type
        if op == tokens.DATEDIFF and dt in (TimePeriod, TimeInterval, Date):
            left_ref = _datediff_to_date(left_ref, left_type)
            right_ref = _datediff_to_date(right_ref, right_type)
            return f"ABS(DATE_DIFF('day', {left_ref}, {right_ref}))"
        # Date↔TimePeriod cross-type promotion
        if left_type and right_type and _is_date_timeperiod_pair(left_type, right_type):
            return _date_tp_compare_expr(left_ref, right_ref, left_type, right_type, op)
        # Typed or generic registry lookup, with function-call fallback
        return registry.generate(op, left_ref, right_ref, data_type=dt)

    def _build_ds_ds_binary(
        self,
        left_node: AST.AST,
        right_node: AST.AST,
        op: str,
    ) -> str:
        """Build SQL for dataset-dataset binary operations using JOIN."""
        left_ds = self._get_dataset_structure(left_node)
        right_ds = self._get_dataset_structure(right_node)
        output_ds = self._get_output_dataset()

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

        paired_measures: List[Tuple[str, str]] = []
        if common_measures:
            paired_measures = [(m, m) for m in common_measures]
        elif len(left_measures) == 1 and len(right_measures) == 1:
            if output_measure_names and len(output_measure_names) == 1:
                out_m = output_measure_names[0]
                paired_measures = [(out_m, out_m)]
            else:
                paired_measures = [(left_measures[0], right_measures[0])]

        cols: List[str] = []
        for id_name in all_ids:
            if id_name in left_ids:
                cols.append(f"{alias_a}.{quote_name(id_name)}")
            else:
                cols.append(f"{alias_b}.{quote_name(id_name)}")

        for left_m, right_m in paired_measures:
            left_ref = f"{alias_a}.{quote_name(left_m)}"
            right_ref = f"{alias_b}.{quote_name(right_m)}"

            # Boolean→String promotion for concat
            if op == tokens.CONCAT:
                left_comp_c = left_ds.components.get(left_m)
                right_comp_c = right_ds.components.get(right_m)
                if left_comp_c and left_comp_c.data_type == Boolean:
                    left_ref = _bool_to_str(left_ref)
                if right_comp_c and right_comp_c.data_type == Boolean:
                    right_ref = _bool_to_str(right_ref)

            left_comp = left_ds.components.get(left_m)
            right_comp = right_ds.components.get(right_m)
            left_dt = left_comp.data_type if left_comp else None
            right_dt = right_comp.data_type if right_comp else None
            expr = self._make_binary_expr(left_ref, right_ref, op, left_dt, right_dt)

            out_name = left_m
            if (
                output_measure_names
                and len(paired_measures) == 1
                and len(output_measure_names) == 1
            ):
                out_name = output_measure_names[0]
            cols.append(f"{expr} AS {quote_name(out_name)}")

        on_clause = self._join_on_clause(common_ids, alias_a, alias_b)

        builder = SQLBuilder().select(*cols).from_table(left_src, alias_a)
        if on_clause != "1=1":
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
            left_sql = self.visit(ds_node)
            right_sql = self.visit(scalar_node)
            if ds_on_left:
                return registry.generate(op, left_sql, right_sql)
            return registry.generate(op, right_sql, left_sql)

        scalar_sql = self.visit(scalar_node)

        def _bin_expr(col_ref: str) -> str:
            comp = ds.components.get(col_ref.strip('"'))
            dt = comp.data_type if comp else None
            if ds_on_left:
                return self._make_binary_expr(col_ref, scalar_sql, op, dt, None)
            return self._make_binary_expr(scalar_sql, col_ref, op, None, dt)

        return self._apply_measures(
            ds_node,
            _bin_expr,
            cast_bool_to_str=op == tokens.CONCAT,
        )

    def _visit_membership(self, node: AST.BinOp) -> str:
        """Visit MEMBERSHIP (#): DS#comp -> SELECT ids, comp FROM DS."""
        comp_name = self._resolve_udo_name(self._get_node_value(node.right))

        if self._in_clause:
            ds_name = self._get_node_value(node.left)
            qualified = f"{ds_name}#{comp_name}"
            if qualified in self._join_alias_map:
                return quote_name(qualified)
            col = quote_name(comp_name)
            if self._column_prefix:
                col = f"{self._column_prefix}.{col}"
            return col

        ds = self._get_dataset_structure(node.left)
        table_src = self._get_dataset_sql(node.left)

        if ds is None:
            ds_name = self._resolve_dataset_name(node.left)
            return f"SELECT {quote_name(comp_name)} FROM {quote_name(ds_name)}"

        target_comp = ds.components.get(comp_name)
        alias_name = comp_name
        if target_comp and target_comp.role in (Role.IDENTIFIER, Role.ATTRIBUTE):
            alias_name = COMP_NAME_MAPPING.get(target_comp.data_type, comp_name)

        cols: List[str] = []
        for name, comp in ds.components.items():
            if comp.role == Role.IDENTIFIER:
                cols.append(quote_name(name))
        if alias_name != comp_name:
            cols.append(f"{quote_name(comp_name)} AS {quote_name(alias_name)}")
        else:
            cols.append(quote_name(comp_name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_match_characters(self, node: AST.BinOp) -> str:
        """Visit match_characters operator using registry."""
        left_type = self._get_node_type(node.left)
        pattern_sql = self.visit(node.right)

        if left_type == _DATASET:
            return self._apply_measures(
                node.left,
                lambda col: registry.generate(tokens.CHARSET_MATCH, col, pattern_sql),
            )
        else:
            left_sql = self.visit(node.left)
            return registry.generate(tokens.CHARSET_MATCH, left_sql, pattern_sql)

    def _visit_exists_in(self, left_node: AST.AST, right_node: AST.AST) -> str:
        """Build SQL for exists_in operation."""
        left_ds = self._get_dataset_structure(left_node)
        right_ds = self._get_dataset_structure(right_node)
        left_src = self._get_dataset_sql(left_node)
        right_src = self._get_dataset_sql(right_node)

        left_ids = left_ds.get_identifiers_names()
        right_ids = right_ds.get_identifiers_names()
        common_ids = [id_ for id_ in left_ids if id_ in right_ids]

        where_clause = self._join_on_clause(common_ids, "l", "r")

        id_cols = ", ".join([f"l.{quote_name(id_)}" for id_ in left_ids])

        right_subq = self._as_subquery(right_src)
        exists_subq = f"EXISTS(SELECT 1 FROM {right_subq} AS r WHERE {where_clause})"
        left_subq = self._as_subquery(left_src)

        return f'SELECT {id_cols}, {exists_subq} AS "bool_var" FROM {left_subq} AS l'

    def _is_operand_type(self, node: AST.AST, target_type: type) -> bool:
        """Check if an operand resolves to *target_type*."""
        if isinstance(node, AST.VarID):
            if self._in_clause and self._current_dataset:
                comp = self._current_dataset.components.get(node.value)
                return comp is not None and comp.data_type == target_type
            return node.value in self.scalars and self.scalars[node.value].data_type == target_type

        elif isinstance(node, AST.ParamOp) and node.op == tokens.CAST:
            type_node = node.children[1]
            return type_node == target_type

        return False

    def _detect_scalar_type(self, node: AST.AST) -> Optional[type]:
        """Detect the data type of a scalar operand for typed dispatch."""
        for tp in (TimePeriod, Duration, TimeInterval):
            if self._is_operand_type(node, tp):
                return tp
        return None

    def _visit_period_indicator(self, node: AST.UnaryOp) -> str:
        """Visit PERIOD_INDICATOR: extract period indicator from TimePeriod."""
        operand_type = self._get_node_type(node.operand)

        ds = self._get_dataset_structure(node.operand)

        if operand_type == _DATASET or ds is not None:
            src = self._get_dataset_sql(node.operand)

            time_id = None
            for comp in ds.components.values():
                if comp.data_type == TimePeriod and comp.role == Role.IDENTIFIER:
                    time_id = comp.name
                    break

            id_cols = [quote_name(c.name) for c in ds.get_identifiers()]
            extract_expr = (
                f'vtl_period_parse({quote_name(time_id)}).period_indicator AS "duration_var"'
            )
            cols_sql = ", ".join(id_cols) + ", " + extract_expr

            if src.strip().upper().startswith("SELECT"):
                return f"SELECT {cols_sql} FROM ({src}) AS _pi"
            return f"SELECT {cols_sql} FROM {src}"
        else:
            operand_sql = self.visit(node.operand)
            return f"vtl_period_parse({operand_sql}).period_indicator"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:  # type: ignore[override]
        """Visit a parameterized operation."""
        op = node.op
        if op == tokens.CAST:
            return self._visit_cast(node)
        if op == tokens.RANDOM:
            return self._visit_random(node)
        if op == tokens.DATE_ADD:
            return self._visit_dateadd(node)
        if op == tokens.FILL_TIME_SERIES:
            return self._visit_fill_time_series(node)

        params_sql = self._visit_params(node.params)
        if op in (tokens.ROUND, tokens.TRUNC) and not params_sql:
            params_sql = ["0"]

        operand_type = self._get_node_type(node.children[0]) if node.children else _SCALAR

        if operand_type == _DATASET:
            ds_node = node.children[0]
            to_str = op in _STRING_PARAM_OPS

            def _param_expr(col_ref: str) -> str:
                return registry.generate(op, col_ref, *params_sql)

            return self._apply_measures(ds_node, _param_expr, cast_bool_to_str=to_str)

        children_sql = [self.visit(c) for c in node.children]
        all_args = children_sql + params_sql
        return registry.generate(op, *all_args)

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

    def _resolve_time_identifier(self, ds: Dataset, op_name: str) -> Tuple[str, type]:
        """Return the time identifier name and type for time-based operators."""
        for comp in ds.components.values():
            if comp.data_type in (TimePeriod, Date) and comp.role == Role.IDENTIFIER:
                return comp.name, comp.data_type

    def _build_time_grid_parts(
        self,
        ds: Dataset,
        time_id: str,
    ) -> Tuple[List[str], List[str], str, str, str, str]:
        """Build common JOIN/select fragments for fill-time-series queries."""
        time_col = quote_name(time_id)
        other_id_cols = [quote_name(c.name) for c in ds.get_identifiers() if c.name != time_id]
        measure_cols = [
            quote_name(c.name) for c in ds.components.values() if c.role != Role.IDENTIFIER
        ]

        join_conds = [f"g.{time_col} = s.{time_col}"]
        join_conds.extend(f"g.{oc} = s.{oc}" for oc in other_id_cols)
        join_on = " AND ".join(join_conds)

        g_cols = [f"g.{oc}" for oc in other_id_cols] + [f"g.{time_col}"]
        s_cols = [f"s.{mc}" for mc in measure_cols]
        final_select = ", ".join(g_cols + s_cols)
        order_by = ", ".join(g_cols)
        return time_col, other_id_cols, measure_cols, join_on, final_select, order_by

    def _build_date_frequency_subquery(
        self, src: str, time_col: str, partition: str, *, as_period_indicator: bool = False
    ) -> str:
        """Build SQL that infers date frequency (or its period indicator) from date diffs."""
        freq_case = self._build_date_frequency_case(as_period_indicator=as_period_indicator)
        alias = "period_ind" if as_period_indicator else "step"
        return f"""
SELECT {freq_case} AS {alias}
FROM (
    SELECT ABS(DATE_DIFF('day',
        LAG({time_col}) OVER ({partition} ORDER BY {time_col}),
        {time_col})) AS diff_days
    FROM {src}
) WHERE diff_days IS NOT NULL AND diff_days > 0""".strip()

    @staticmethod
    def _build_date_frequency_case(as_period_indicator: bool) -> str:
        """Return a CASE expression for inferred date frequency output."""
        periods = {
            7: "'D'" if as_period_indicator else "INTERVAL 1 DAY",
            28: "'W'" if as_period_indicator else "INTERVAL 7 DAY",
            90: "'M'" if as_period_indicator else "INTERVAL 1 MONTH",
            181: "'Q'" if as_period_indicator else "INTERVAL 3 MONTH",
            365: "'S'" if as_period_indicator else "INTERVAL 6 MONTH",
            "'Inf'::DOUBLE": "'A'" if as_period_indicator else "INTERVAL 1 YEAR",
        }

        cases = "\n".join(
            f"WHEN MIN(diff_days) < {value} THEN {period}" for value, period in periods.items()
        )

        return f"CASE\n{cases}\nEND".strip()

    # Shared SQL fragment for the RECURSIVE step that increments a vtl_time_period.
    _TP_NEXT_PERIOD = (
        "CASE"
        " WHEN ep.tp.period_number + 1 > vtl_period_limit(ep.tp.period_indicator)"
        " THEN {'year': ep.tp.year + 1, 'period_indicator': ep.tp.period_indicator,"
        " 'period_number': 1}::vtl_time_period"
        " ELSE {'year': ep.tp.year, 'period_indicator': ep.tp.period_indicator,"
        " 'period_number': ep.tp.period_number + 1}::vtl_time_period END"
    )

    def _visit_fill_time_series(self, node: AST.ParamOp) -> str:
        """Fill missing time periods/dates with NULL rows."""
        ds_node = node.children[0]
        fill_mode = "all"
        if node.params:
            mode_val = self.visit(node.params[0])
            if isinstance(mode_val, str):
                fill_mode = mode_val.strip("'\"").lower()

        ds = self._get_dataset_structure(ds_node)
        src = self._get_dataset_sql(ds_node)

        time_id, time_type = self._resolve_time_identifier(ds, "fill_time_series")

        if time_type == Date:
            return self._fill_time_series_date(ds, src, time_id, fill_mode)
        return self._fill_time_series_period(ds, src, time_id, fill_mode)

    def _fill_time_series_period(self, ds: Dataset, src: str, time_id: str, fill_mode: str) -> str:
        """Fill time series for TimePeriod identifiers using RECURSIVE CTE."""
        time_col, other_id_cols, _, join_on, final_select, order_by = self._build_time_grid_parts(
            ds, time_id
        )
        oid_select = ", ".join(other_id_cols)
        per_group = fill_mode == "single" and bool(other_id_cols)

        cte = CTEBuilder()
        cte.cte("source", f"SELECT * FROM {src}")
        cte.cte("parsed", f"SELECT *, vtl_period_parse({time_col}) AS tp FROM source")

        if per_group:
            cte.cte(
                "bounds",
                f"SELECT {oid_select}, MIN(tp) AS min_tp, MAX(tp) AS max_tp "
                f"FROM parsed GROUP BY {oid_select}, tp.period_indicator",
            )
            oid_ep_refs = ", ".join(f"ep.{oc}" for oc in other_id_cols)
            cte.recursive_cte(
                "expected_periods",
                f"tp, max_tp, {oid_select}",
                seed=f"SELECT min_tp, max_tp, {oid_select} FROM bounds",
                step=f"SELECT {self._TP_NEXT_PERIOD}, ep.max_tp, {oid_ep_refs} "
                f"FROM expected_periods ep WHERE ep.tp < ep.max_tp",
            )
            cte.cte(
                "full_grid",
                f"SELECT {oid_select}, vtl_period_to_string(tp) AS {time_col} "
                f"FROM expected_periods",
            )
        else:
            cte.cte(
                "year_range",
                "SELECT MIN(tp.year) AS min_year, MAX(tp.year) AS max_year FROM parsed",
            )
            cte.cte("freq_list", "SELECT DISTINCT tp.period_indicator AS ind FROM parsed")
            cte.cte(
                "bounds",
                "SELECT ind, "
                "{'year': min_year, 'period_indicator': ind, "
                "'period_number': 1}::vtl_time_period AS min_tp, "
                "{'year': max_year, 'period_indicator': ind, "
                "'period_number': vtl_period_limit(ind)}::vtl_time_period AS max_tp "
                "FROM freq_list, year_range",
            )
            cte.recursive_cte(
                "expected_periods",
                "tp, max_tp",
                seed="SELECT min_tp, max_tp FROM bounds",
                step=f"SELECT {self._TP_NEXT_PERIOD}, ep.max_tp "
                f"FROM expected_periods ep WHERE ep.tp < ep.max_tp",
            )
            cte.cte(
                "period_strings",
                f"SELECT vtl_period_to_string(tp) AS {time_col} FROM expected_periods",
            )
            if other_id_cols:
                cte.cte(
                    "group_freq",
                    f"SELECT DISTINCT {oid_select}, "
                    f"vtl_period_parse({time_col}).period_indicator AS ind FROM source",
                )
                cte.cte(
                    "full_grid",
                    "SELECT gf.{gf_cols}, ps.{tc} FROM group_freq gf "
                    "JOIN period_strings ps "
                    "ON vtl_period_parse(ps.{tc}).period_indicator = gf.ind".format(
                        gf_cols=", gf.".join(other_id_cols), tc=time_col
                    ),
                )
            else:
                cte.cte("full_grid", f"SELECT {time_col} FROM period_strings")

        final = (
            f"SELECT {final_select} FROM full_grid g "
            f"LEFT JOIN source s ON {join_on} ORDER BY {order_by}"
        )
        return cte.select(final)

    def _fill_time_series_date(self, ds: Dataset, src: str, time_id: str, fill_mode: str) -> str:
        """Fill time series for Date identifiers using frequency inference."""
        time_col, other_id_cols, _, join_on, final_select, order_by = self._build_time_grid_parts(
            ds, time_id
        )
        partition = "PARTITION BY {}".format(", ".join(other_id_cols)) if other_id_cols else ""
        per_group = fill_mode == "single" and bool(other_id_cols)
        freq_step = "(SELECT step FROM freq)"

        cte = CTEBuilder()
        cte.cte("source", f"SELECT * FROM {src}")
        cte.cte("freq", self._build_date_frequency_subquery("source", time_col, partition))

        if per_group:
            oid_csv = ", ".join(other_id_cols)
            cte.cte(
                "bounds",
                f"SELECT {oid_csv}, MIN({time_col}) AS min_d, MAX({time_col}) AS max_d "
                f"FROM source GROUP BY {oid_csv}",
            )
            b_cols = ", ".join(f"b.{oc}" for oc in other_id_cols)
            cte.cte(
                "full_grid",
                f"SELECT {b_cols}, CAST(d AS TIMESTAMP) AS {time_col} "
                f"FROM bounds b, generate_series(b.min_d, b.max_d, {freq_step}) AS t(d)",
            )
        else:
            cte.cte(
                "bounds",
                f"SELECT MIN({time_col}) AS min_d, MAX({time_col}) AS max_d FROM source",
            )
            gen = (
                f"generate_series("
                f"(SELECT min_d FROM bounds), (SELECT max_d FROM bounds), {freq_step}) AS t(d)"
            )
            if other_id_cols:
                oid_csv = ", ".join(other_id_cols)
                cte.cte("group_freq", f"SELECT DISTINCT {oid_csv} FROM source")
                gf_cols = ", ".join(f"gf.{oc}" for oc in other_id_cols)
                cte.cte(
                    "full_grid",
                    f"SELECT {gf_cols}, CAST(d AS TIMESTAMP) AS {time_col} "
                    f"FROM group_freq gf, {gen}",
                )
            else:
                cte.cte("full_grid", f"SELECT CAST(d AS TIMESTAMP) AS {time_col} FROM {gen}")

        final = (
            f"SELECT {final_select} FROM full_grid g "
            f"LEFT JOIN source s ON {join_on} ORDER BY {order_by}"
        )
        return cte.select(final)

    def _visit_flow_stock(self, node: AST.UnaryOp, op: str) -> str:
        """Visit FLOW_TO_STOCK or STOCK_TO_FLOW: window functions over time series."""
        ds = self._get_dataset_structure(node.operand)
        src = self._get_dataset_sql(node.operand)

        time_id, time_type = self._resolve_time_identifier(ds, op)
        other_ids = [quote_name(c.name) for c in ds.get_identifiers() if c.name != time_id]

        partition_parts = list(other_ids)
        if time_type == TimePeriod:
            partition_parts.append(f"vtl_period_parse({quote_name(time_id)}).period_indicator")

        partition_clause = f"PARTITION BY {', '.join(partition_parts)}" if partition_parts else ""
        order_clause = f"ORDER BY {quote_name(time_id)}"
        window = f"({partition_clause} {order_clause})"

        cols = []
        for comp in ds.components.values():
            col = quote_name(comp.name)
            if comp.role == Role.IDENTIFIER:
                cols.append(col)
            elif comp.data_type in (Integer, Number, Boolean):
                if op == tokens.FLOW_TO_STOCK:
                    cols.append(
                        f"CASE WHEN {col} IS NULL THEN NULL ELSE "
                        f"SUM({col}) OVER ({partition_clause} {order_clause} "
                        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) END AS {col}"
                    )
                else:  # STOCK_TO_FLOW
                    cols.append(f"COALESCE({col} - LAG({col}) OVER {window}, {col}) AS {col}")
            else:
                cols.append(col)

        return SQLBuilder().select(*cols).from_table(src).build()

    def _visit_timeshift(self, node: AST.BinOp) -> str:
        """Visit TIMESHIFT: shift time identifier by N periods."""
        ds_node = node.left
        shift_sql = self.visit(node.right)

        ds = self._get_dataset_structure(ds_node)
        src = self._get_dataset_sql(ds_node)

        time_id, time_type = self._resolve_time_identifier(ds, "timeshift")
        time_col = quote_name(time_id)
        if time_type == TimePeriod:
            shifted = f"vtl_tp_shift(vtl_period_parse({time_col}), {shift_sql}) AS {time_col}"
            cols = []
            for comp in ds.components.values():
                col = quote_name(comp.name)
                cols.append(shifted if comp.name == time_id else col)
            return SQLBuilder().select(*cols).from_table(src).build()
        else:
            other_ids = [quote_name(c.name) for c in ds.get_identifiers() if c.name != time_id]
            partition = f"PARTITION BY {', '.join(other_ids)}" if other_ids else ""

            cols = []
            for comp in ds.components.values():
                col = quote_name(comp.name)
                if comp.name == time_id:
                    cols.append(f"vtl_dateadd({col}, {shift_sql}, freq.period_ind) AS {col}")
                else:
                    cols.append(col)

            freq_sql = self._build_date_frequency_subquery(
                src, time_col, partition, as_period_indicator=True
            )

            return f"""SELECT {", ".join(cols)}
FROM {src}, (
    {freq_sql}
) AS freq"""

    def _visit_dateadd(self, node: AST.ParamOp) -> str:
        """Visit DATEADD operation: dateadd(op, shiftNumber, periodInd)."""
        operand_node = node.children[0]
        operand_type = self._get_node_type(operand_node)

        shift_sql = self.visit(node.params[0]) if node.params else "0"
        period_sql = self.visit(node.params[1]) if len(node.params) > 1 else "'D'"

        is_tp = self._is_operand_type(operand_node, TimePeriod)

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

            return self._apply_measures(ds_node, _dateadd_expr)
        else:
            operand_sql = self.visit(operand_node)
            if is_tp:
                return f"vtl_tp_dateadd(vtl_period_parse({operand_sql}), {shift_sql}, {period_sql})"
            return f"vtl_dateadd({operand_sql}, {shift_sql}, {period_sql})"

    def _get_source_vtl_type(self, node: "AST.AST") -> Optional[str]:
        """Return the VTL type name produced by an AST node when known."""
        if isinstance(node, AST.Constant):
            if isinstance(node.value, bool):
                return "Boolean"
            if isinstance(node.value, int):
                return "Integer"
            if isinstance(node.value, float):
                return "Number"
            if isinstance(node.value, str):
                return "String"
        if (
            isinstance(node, AST.ParamOp)
            and str(getattr(node, "op", "")).lower() == "cast"
            and len(node.children) >= 2
        ):
            type_node = node.children[1]
            return self._get_node_value(type_node)
        if isinstance(node, AST.TimeAggregation):
            return "TimePeriod"
        if isinstance(node, AST.VarID) and self._current_dataset:
            comp = self._current_dataset.components.get(node.value)
            if comp and comp.data_type:
                type_name = getattr(comp.data_type, "__name__", str(comp.data_type))
                return type_name
        return None

    def _visit_cast(self, node: AST.ParamOp) -> str:
        """Visit CAST operation."""
        operand = node.children[0]
        target_type_str = ""
        if len(node.children) >= 2:
            type_node = node.children[1]
            target_type_str = self._get_node_value(type_node)

        duckdb_type = get_duckdb_type(target_type_str)

        mask: Optional[str] = None
        if node.params:
            mask_node = node.params[0]
            if hasattr(mask_node, "value"):
                mask = mask_node.value

        operand_type = self._get_node_type(operand)

        if operand_type == _DATASET:
            ds = self._get_dataset_structure(operand)
            comp_types: Dict[str, str] = {}
            if ds:
                for cname, comp in ds.components.items():
                    if comp.data_type:
                        comp_types[cname] = getattr(comp.data_type, "__name__", str(comp.data_type))

            def _cast_measure(col: str) -> str:
                col_name = col.strip('"')
                src_type = comp_types.get(col_name)
                return self._cast_expr(col, duckdb_type, target_type_str, mask, src_type)

            return self._apply_measures(operand, _cast_measure)
        else:
            operand_sql = self.visit(operand)
            source_type = self._get_source_vtl_type(operand)
            return self._cast_expr(operand_sql, duckdb_type, target_type_str, mask, source_type)

    def _cast_expr(
        self,
        expr: str,
        duckdb_type: str,
        target_type_str: str,
        mask: Optional[str],
        source_type_str: Optional[str] = None,
    ) -> str:
        """Generate a CAST expression for a single value."""
        target_lower = target_type_str.lower()
        source_lower = (source_type_str or "").lower()

        if mask and target_type_str == "Date":
            return f"STRFTIME(STRPTIME({expr}, '{mask}'), '%Y-%m-%d %H:%M:%S')"

        if target_type_str == "Boolean" and source_lower == "string":
            return f"(LOWER(TRIM(CAST({expr} AS VARCHAR))) = 'true')"

        if target_type_str == "Integer":
            if source_lower == "boolean":
                return f"CAST({expr} AS {duckdb_type})"
            return f"CAST(TRUNC(CAST({expr} AS DOUBLE)) AS {duckdb_type})"

        if target_type_str == "String" and source_lower in ("time_period", "timeperiod"):
            _tp_string_macros = {
                "vtl": "vtl_period_to_vtl",
                "sdmx_reporting": "vtl_period_to_sdmx_reporting",
                "sdmx_gregorian": "vtl_period_to_sdmx_gregorian",
                "natural": "vtl_period_to_natural",
            }
            macro = _tp_string_macros.get(self.time_period_output_format, "vtl_period_to_vtl")
            return f"{macro}({expr})"

        if target_lower in ("time_period", "timeperiod"):
            if source_lower == "date":
                return f"vtl_date_to_period({expr})"
            if source_lower in ("time", "timeinterval"):
                return f"vtl_interval_to_period({expr})"
            return f"vtl_period_normalize(CAST({expr} AS VARCHAR))"

        if target_type_str == "Date":
            if source_lower in ("time_period", "timeperiod"):
                return f"vtl_period_to_date({expr})"
            if source_lower in ("time", "timeinterval"):
                return f"vtl_interval_to_date({expr})"

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
        seed_type = self._get_node_type(seed_node) if seed_node else _SCALAR

        if seed_type == _DATASET and seed_node is not None:
            index_sql = self.visit(index_node) if index_node else "0"
            return self._apply_measures(
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

    # Clause visitor

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
            return self._clause_fallback_sql(node)
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
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        calc_exprs: Dict[str, str] = {}
        with self._clause_scope(ds):
            for child in node.children:
                assignment = self._unwrap_assignment(child)
                if isinstance(assignment, AST.Assignment):
                    col_name = self._resolve_udo_name(self._get_node_value(assignment.left))
                    expr_sql = self.visit(assignment.right)
                    calc_exprs[col_name] = expr_sql
                    if "vtl_tp_dateadd" in expr_sql and self.current_assignment:
                        out_ds = self.output_datasets.get(self.current_assignment)
                        if (
                            out_ds
                            and col_name in out_ds.components
                            and out_ds.components[col_name].data_type == TimePeriod
                        ):
                            out_ds.components[col_name].data_type = Date

        select_cols: List[str] = []
        for name in ds.components:
            if name in calc_exprs:
                select_cols.append(f"{calc_exprs[name]} AS {quote_name(name)}")
            else:
                select_cols.append(quote_name(name))

        for col_name, expr_sql in calc_exprs.items():
            if col_name not in ds.components:
                select_cols.append(f"{expr_sql} AS {quote_name(col_name)}")

        inner_src = self._as_subquery(table_src)

        return SQLBuilder().select(*select_cols).from_table(inner_src, "t").build()

    def _visit_keep(self, node: AST.RegularAggregation) -> str:
        """Visit keep clause."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        keep_names: List[str] = [
            name for name, comp in ds.components.items() if comp.role == Role.IDENTIFIER
        ]
        keep_names.extend(self._extract_component_names(node.children, self._join_alias_map))

        keep_set = set(keep_names)
        for qualified in self._join_alias_map:
            if qualified not in keep_set:
                self._consumed_join_aliases.add(qualified)

        cols = [quote_name(name) for name in keep_names]
        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_drop(self, node: AST.RegularAggregation) -> str:
        """Visit drop clause."""
        if not node.dataset:
            return ""

        table_src = self._get_dataset_sql(node.dataset)
        drop_names = self._extract_component_names(node.children, self._join_alias_map)

        for name in drop_names:
            if name in self._join_alias_map:
                self._consumed_join_aliases.add(name)

        if not drop_names:
            return f"SELECT * FROM {table_src}"

        exclude = ", ".join(quote_name(n) for n in drop_names)
        return SQLBuilder().select(f"* EXCLUDE ({exclude})").from_table(table_src).build()

    def _visit_rename(self, node: AST.RegularAggregation) -> str:
        """Visit rename clause."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        renames: Dict[str, str] = {}
        for child in node.children:
            if isinstance(child, AST.RenameNode):
                old = self._resolve_udo_name(child.old_name)
                new = self._resolve_udo_name(child.new_name)
                if "#" in old:
                    if old in self._join_alias_map:
                        self._consumed_join_aliases.add(old)
                    else:
                        old = old.split("#", 1)[1]
                renames[old] = new

        cols: List[str] = []
        for name in ds.components:
            matched_new = renames.get(name)
            if matched_new is None and "#" in name:
                unqual = name.split("#", 1)[1]
                matched_new = renames.get(unqual)
            if matched_new is not None:
                cols.append(f"{quote_name(name)} AS {quote_name(matched_new)}")
            else:
                cols.append(quote_name(name))

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_subspace(self, node: AST.RegularAggregation) -> str:
        """Visit subspace clause."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        where_parts: List[str] = []
        remove_ids: set[str] = set()
        for child in node.children:
            if isinstance(child, AST.BinOp):
                col_name = self._get_node_value(child.left)
                remove_ids.add(col_name)
                val_sql = self.visit(child.right)
                where_parts.append(f"{quote_name(col_name)} = {val_sql}")

        cols = [quote_name(name) for name in ds.components if name not in remove_ids]

        builder = SQLBuilder().select(*cols).from_table(table_src)
        for wp in where_parts:
            builder.where(wp)
        return builder.build()

    def _visit_clause_aggregate(self, node: AST.RegularAggregation) -> str:  # noqa: C901
        """Visit aggregate clause: DS[aggr Me := sum(Me) group by Id, ... having ...]."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        calc_exprs: Dict[str, str] = {}
        having_sql: Optional[str] = None
        tp_minmax_cols: List[tuple[str, str]] = []

        with self._clause_scope(ds):
            for child in node.children:
                assignment = self._unwrap_assignment(child)
                if isinstance(assignment, AST.Assignment):
                    col_name = self._get_node_value(assignment.left)
                    agg_node = assignment.right
                    if isinstance(agg_node, AST.Aggregation) and agg_node.having_clause is not None:
                        hc = agg_node.having_clause
                        if isinstance(hc, AST.ParamOp) and hc.params is not None:
                            having_sql = self.visit(hc.params)

                    if (
                        isinstance(agg_node, AST.Aggregation)
                        and str(agg_node.op).lower() in (tokens.MIN, tokens.MAX)
                        and agg_node.operand
                        and hasattr(agg_node.operand, "value")
                    ):
                        src_comp = ds.components.get(agg_node.operand.value)
                        if src_comp and src_comp.data_type == TimePeriod:
                            tp_minmax_cols.append(
                                (agg_node.operand.value, str(agg_node.op).lower())
                            )

                    expr_sql = self.visit(agg_node)
                    calc_exprs[col_name] = expr_sql

        group_ids: List[str] = []
        grouping_op: str = ""
        grouping_names: List[str] = []
        for child in node.children:
            assignment = self._unwrap_assignment(child)
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
            output_ds = self._get_output_dataset()
            group_ids = list(output_ds.get_identifiers_names() if output_ds else all_input_ids)

        cols: List[str] = [quote_name(id_) for id_ in group_ids]
        for col_name, expr_sql in calc_exprs.items():
            cols.append(f"{expr_sql} AS {quote_name(col_name)}")

        builder = SQLBuilder().select(*cols).from_table(table_src)
        if group_ids:
            builder.group_by(*[quote_name(id_) for id_ in group_ids])

        if having_sql:
            builder.having(having_sql)

        main_sql = builder.build()

        if tp_minmax_cols:
            main_sql = _add_tp_indicator_check(main_sql, table_src, tp_minmax_cols)

        return main_sql

    def _visit_apply(self, node: AST.RegularAggregation) -> str:
        """Visit apply clause."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        output_ds = self.output_datasets.get(self.current_assignment)

        id_names = ds.get_identifiers_names()

        computed: Dict[str, str] = {}
        for child in node.children:
            if not isinstance(child, AST.BinOp):
                continue
            left_alias = self._get_node_value(child.left)
            right_alias = self._get_node_value(child.right)
            op = str(child.op).lower() if child.op else ""

            left_measures: Dict[str, str] = {}
            right_measures: Dict[str, str] = {}
            for qualified in self._join_alias_map:
                if "#" in qualified:
                    alias, comp = qualified.split("#", 1)
                    if alias == left_alias:
                        left_measures[comp] = qualified
                    elif alias == right_alias:
                        right_measures[comp] = qualified

            common_measures = left_measures.keys() & right_measures.keys()
            for measure in common_measures:
                left_col = quote_name(left_measures[measure])
                right_col = quote_name(right_measures[measure])
                expr = registry.generate(op, left_col, right_col)
                computed[measure] = expr
                self._consumed_join_aliases.add(left_measures[measure])
                self._consumed_join_aliases.add(right_measures[measure])

        cols: List[str] = [quote_name(id_) for id_ in id_names]
        if output_ds:
            for comp_name in output_ds.get_measures_names():
                if comp_name in computed:
                    cols.append(f"{computed[comp_name]} AS {quote_name(comp_name)}")
                else:
                    cols.append(quote_name(comp_name))
        else:
            for measure, expr in computed.items():
                cols.append(f"{expr} AS {quote_name(measure)}")

        return SQLBuilder().select(*cols).from_table(table_src).build()

    def _visit_unpivot(self, node: AST.RegularAggregation) -> str:
        """Visit unpivot clause."""
        resolved = self._resolve_clause_dataset(node)
        if resolved is None:
            return self._clause_fallback_sql(node)
        ds, table_src = resolved

        new_id_name = self._resolve_udo_name(self._get_node_value(node.children[0]))
        new_measure_name = self._resolve_udo_name(self._get_node_value(node.children[1]))

        id_names = ds.get_identifiers_names()
        measure_names = ds.get_measures_names()

        if not measure_names:
            return f"SELECT * FROM {table_src}"

        parts: List[str] = []
        for measure in measure_names:
            cols: List[str] = [quote_name(i) for i in id_names]
            cols.append(f"'{measure}' AS {quote_name(new_id_name)}")
            cols.append(f"{quote_name(measure)} AS {quote_name(new_measure_name)}")
            select_clause = ", ".join(cols)
            part = (
                f"SELECT {select_clause} FROM {table_src} WHERE {quote_name(measure)} IS NOT NULL"
            )
            parts.append(part)

        return " UNION ALL ".join(parts)

    # Aggregation visitor

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

        if (
            time_agg_id
            and time_agg_expr
            and node.grouping_op != "group except"
            and time_agg_id not in group_cols
        ):
            group_cols = [*group_cols, time_agg_id]

        cols: List[str] = []
        group_by_cols: List[str] = []
        for col_name in group_cols:
            if col_name == time_agg_id and time_agg_expr:
                cols.append(f"{time_agg_expr} AS {quote_name(col_name)}")
                group_by_cols.append(time_agg_expr)
            else:
                cols.append(quote_name(col_name))
                group_by_cols.append(quote_name(col_name))
        return cols, group_by_cols

    def visit_Aggregation(self, node: AST.Aggregation) -> str:  # type: ignore[override]  # noqa: C901
        """Visit a standalone aggregation: sum(DS group by Id)."""
        op = str(node.op).lower()

        # Component-level aggregation in clause context
        if self._in_clause and node.operand:
            operand_type = self._get_node_type(node.operand)
            if operand_type in (_COMPONENT, _SCALAR):
                operand_sql = self.visit(node.operand)
                # Type-aware MIN/MAX for Duration/TimePeriod
                if self._current_dataset and hasattr(node.operand, "value"):
                    comp = self._current_dataset.components.get(node.operand.value)
                    dt = comp.data_type if comp else None
                    agg = self._build_agg_expr(op, operand_sql, dt)
                    if agg is not None:
                        return agg
                expr = registry.generate(op, operand_sql)
                if op == tokens.COUNT:
                    expr = f"NULLIF({expr}, 0)"
                return expr

        # count() without operand
        if node.operand is None:
            if op == tokens.COUNT:
                if self._in_clause and self._current_dataset:
                    measures = self._current_dataset.get_measures_names()
                    if measures:
                        or_parts = " OR ".join(f"{quote_name(m)} IS NOT NULL" for m in measures)
                        return f"NULLIF(COUNT(CASE WHEN {or_parts} THEN 1 END), 0)"
                return "NULLIF(COUNT(*), 0)"
            return ""

        ds = self._get_dataset_structure(node.operand)
        if ds is None:
            operand_sql = self.visit(node.operand)
            return registry.generate(op, operand_sql)

        table_src = self._get_dataset_sql(node.operand)

        # Resolve group columns from input identifiers.
        all_ids = ds.get_identifiers_names()
        group_cols = self._resolve_group_cols(node, all_ids)

        cols, group_by_cols = self._build_agg_group_cols(node, ds, group_cols)

        ds_tp_minmax_cols: List[tuple[str, str]] = []

        # count() produces a single int_var measure.
        if op == tokens.COUNT:
            alias = "int_var"
            source_measures = ds.get_measures_names()
            if source_measures:
                and_parts = " AND ".join(f"{quote_name(m)} IS NOT NULL" for m in source_measures)
                count_expr = f"COUNT(CASE WHEN {and_parts} THEN 1 END)"
                if group_cols:
                    count_expr = f"NULLIF({count_expr}, 0)"
                cols.append(f"{count_expr} AS {quote_name(alias)}")
            else:
                cols.append(f"COUNT(*) AS {quote_name(alias)}")
        else:
            measures = ds.get_measures_names()
            for measure in measures:
                comp = ds.components.get(measure)
                dt = comp.data_type if comp else None
                qm = quote_name(measure)

                if dt == TimePeriod and op in (tokens.MIN, tokens.MAX):
                    ds_tp_minmax_cols.append((measure, op))
                agg = self._build_agg_expr(op, qm, dt, dataset_level=True)
                expr = agg if agg is not None else registry.generate(op, qm)
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

        main_sql = builder.build()

        if ds_tp_minmax_cols:
            main_sql = _add_tp_indicator_check(main_sql, table_src, ds_tp_minmax_cols)

        return main_sql

    # =========================================================================
    # Analytic visitor
    # =========================================================================

    def _build_over_clause(self, node: AST.Analytic) -> str:
        """Build the OVER (...) clause for an analytic function."""
        over_parts: List[str] = []
        if node.partition_by:
            partition_cols = ", ".join(quote_name(p) for p in node.partition_by)
            over_parts.append(f"PARTITION BY {partition_cols}")
        if node.order_by:
            order_cols = ", ".join(f"{quote_name(o.component)} {o.order}" for o in node.order_by)
            over_parts.append(f"ORDER BY {order_cols}")
        if node.window:
            order_is_date = False
            if node.order_by and self._current_dataset:
                comp = self._current_dataset.components.get(node.order_by[0].component)
                order_is_date = comp is not None and comp.data_type == Date
            window_sql = self.visit_Windowing(node.window, order_is_date=order_is_date)
            over_parts.append(window_sql)
        return " ".join(over_parts)

    def _build_analytic_expr(self, op: str, operand_sql: str, node: AST.Analytic) -> str:
        """Build the analytic function expression (without OVER).

        For ratio_to_report, returns the complete expression including OVER clause.
        Callers must check _is_self_contained_analytic() to avoid adding OVER again.
        """
        if op == tokens.RATIO_TO_REPORT:
            over_clause = self._build_over_clause(node)
            partition_sum = f"SUM({operand_sql}) OVER ({over_clause})"
            err_msg = (
                "'VTL Error 2-1-3-1: Division by zero produced infinite values in ratio_to_report'"
            )
            return (
                f"CASE WHEN {partition_sum} = 0 THEN "
                f"CAST(error({err_msg}) AS DOUBLE) "
                f"ELSE CAST({operand_sql} AS DOUBLE) / {partition_sum} END"
            )
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
        return registry.generate(op, operand_sql)

    def visit_Analytic(self, node: AST.Analytic) -> str:  # type: ignore[override]
        """Visit an analytic (window) function."""
        op = str(node.op).lower()

        # Check if operand is a dataset — needs dataset-level handling
        if node.operand and self._get_node_type(node.operand) == _DATASET:
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

        name_override = "int_var" if op == tokens.COUNT else None
        result = self._apply_measures(node.operand, _analytic_expr, name_override)

        # Inject TimePeriod indicator validation for MIN/MAX
        if op in (tokens.MIN, tokens.MAX) and node.operand:
            ds = self._get_dataset_structure(node.operand)
            if ds:
                tp_cols = [
                    (m, op)
                    for m in ds.get_measures_names()
                    if ds.components[m].data_type == TimePeriod
                ]
                if tp_cols:
                    table_src = self._get_dataset_sql(node.operand)
                    result = _add_tp_indicator_check(result, table_src, tp_cols)

        return result

    def visit_Windowing(  # type: ignore[override]
        self, node: AST.Windowing, *, order_is_date: bool = False
    ) -> str:
        """Visit a windowing specification."""
        type_str = str(node.type_).upper() if node.type_ else "ROWS"
        # Map VTL types to SQL: DATA POINTS → ROWS
        if "DATA" in type_str:
            type_str = "ROWS"
        elif "RANGE" in type_str:
            type_str = "RANGE"

        is_range_date = type_str == "RANGE" and order_is_date

        def bound_str(value: Union[int, str], mode: str) -> str:
            mode_up = mode.upper()
            val_str = str(value).upper()
            if "CURRENT" in mode_up or val_str == "CURRENT ROW":
                return "CURRENT ROW"
            if val_str == "UNBOUNDED" or (isinstance(value, int) and value < 0):
                return f"UNBOUNDED {mode_up}"
            if is_range_date and isinstance(value, int):
                return f"INTERVAL '{value}' DAY {mode_up}"
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
        operand_type = self._get_node_type(node.children[0])

        low_sql = self.visit(node.children[1])
        high_sql = self.visit(node.children[2])

        if operand_type == _DATASET:
            return self._apply_measures(
                node.children[0],
                lambda col: self._between_expr(col, low_sql, high_sql),
            )

        operand_sql = self.visit(node.children[0])
        return self._between_expr(operand_sql, low_sql, high_sql)

    def _visit_exists_in_mul(self, node: AST.MulOp) -> str:
        """Visit EXISTS_IN in MulOp form, handling the optional retain parameter."""
        base_sql = self._visit_exists_in(node.children[0], node.children[1])

        # Check for retain parameter (true / false / all); "all" keeps every row.
        if len(node.children) >= 3:
            retain_node = node.children[2]
            if isinstance(retain_node, AST.Constant) and isinstance(retain_node.value, bool):
                bool_literal = "TRUE" if retain_node.value else "FALSE"
                return f'SELECT * FROM ({base_sql}) AS _ei WHERE "bool_var" = {bool_literal}'

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
                    f"{quote_name(child.value if hasattr(child, 'value') else child_sql)}"
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
                ordered_cols = ", ".join(quote_name(c) for c in col_order)
                ordered_sqls = [f"SELECT {ordered_cols} FROM ({sql}) AS _ord" for sql in child_sqls]

                id_names = order_ds.get_identifiers_names()
                if id_names:
                    inner_sql = registry.generate(op, *ordered_sqls)
                    id_cols = ", ".join(quote_name(i) for i in id_names)
                    # Preserve UNION ALL row order to match pandas drop_duplicates(keep="first").
                    # QUALIFY keeps the first occurrence per identifier group by insertion order.
                    return (
                        f"SELECT {ordered_cols} FROM ("
                        f"SELECT *, ROW_NUMBER() OVER () AS _rn "
                        f"FROM ({inner_sql}) AS _union_inner"
                        f") AS _union_t "
                        f"QUALIFY ROW_NUMBER() OVER (PARTITION BY {id_cols} ORDER BY _rn) = 1"
                    )
                return registry.generate(op, *ordered_sqls)
            return registry.generate(op, *child_sqls)

        if len(child_sqls) < 2:
            return child_sqls[0] if child_sqls else ""

        first_ds = self._get_dataset_structure(node.children[0])
        if first_ds is None:
            return registry.generate(op, *child_sqls)

        id_names = first_ds.get_identifiers_names()
        a_sql = child_sqls[0]
        b_sql = child_sqls[1]

        on_clause = self._join_on_clause(id_names, "a", "b")

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
            on_clause_rev = self._join_on_clause(second_ids, "c", "d")
            return (
                f"(SELECT a.* FROM ({a_sql}) AS a "
                f"WHERE NOT EXISTS (SELECT 1 FROM ({b_sql}) AS b WHERE {on_clause})) "
                f"UNION ALL "
                f"(SELECT c.* FROM ({b_sql}) AS c "
                f"WHERE NOT EXISTS (SELECT 1 FROM ({a_sql}) AS d WHERE {on_clause_rev}))"
            )

        return registry.generate(op, *child_sqls)

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
        if self._get_node_type(node.condition) != _DATASET:
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
        if isinstance(node, AST.VarID) and self._get_node_type(node) == _DATASET:
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
            and self._get_node_type(node.condition.left) == _DATASET
            and self._get_node_type(node.condition.right) == _DATASET
        )
        cond_ds = self._get_dataset_structure(node.condition) if cond_is_ds_vs_ds else None
        if cond_ds is not None:
            source_sql = self.visit(node.condition)
            source_ids = list(cond_ds.get_identifiers_names())
            bool_measures = list(cond_ds.get_measures_names())
            cond_expr = f"{alias_cond}.{quote_name(bool_measures[0])}" if bool_measures else "TRUE"
        else:
            source_sql = self._get_dataset_sql(source_node)
            source_ids = list(source_ds.get_identifiers_names())
            # Evaluate condition as a column expression (not a full SELECT)
            with self._clause_scope(source_ds, prefix=alias_cond):
                cond_expr = self.visit(node.condition)

        then_type = self._get_node_type(node.thenOp)
        else_type = self._get_node_type(node.elseOp)

        # Determine output measures and attributes.
        def _is_plain_dataset(n: AST.AST) -> bool:
            return isinstance(n, AST.VarID) and self._get_node_type(n) == _DATASET

        ref_ds: Optional[Dataset] = None
        if then_type == _DATASET and _is_plain_dataset(node.thenOp):
            ref_ds = self._get_dataset_structure(node.thenOp)
        elif else_type == _DATASET and _is_plain_dataset(node.elseOp):
            ref_ds = self._get_dataset_structure(node.elseOp)
        if ref_ds is None:
            ref_ds = self._get_output_dataset() or source_ds
        output_measures = list(ref_ds.get_measures_names())
        output_attributes = list(ref_ds.get_attributes_names())

        # Build SELECT columns
        cols: List[str] = [f"{alias_cond}.{quote_name(id_)}" for id_ in source_ids]

        for col_name in output_measures + output_attributes:
            if then_type == _DATASET:
                then_ref = f"t.{quote_name(col_name)}"
            else:
                then_ref = self.visit(node.thenOp)

            if else_type == _DATASET:
                else_ref = f"e.{quote_name(col_name)}"
            else:
                else_ref = self.visit(node.elseOp)

            cols.append(
                f"CASE WHEN {cond_expr} THEN {then_ref} "
                f"ELSE {else_ref} END AS {quote_name(col_name)}"
            )

        # Use from_subquery when the source is a SELECT (e.g., dataset-level condition)
        if source_sql.lstrip().upper().startswith("SELECT"):
            builder = SQLBuilder().select(*cols).from_subquery(source_sql, alias_cond)
        else:
            builder = SQLBuilder().select(*cols).from_table(source_sql, alias_cond)

        # Use LEFT JOINs so empty datasets don't eliminate all rows
        then_join_id = self._left_join_dataset(
            node.thenOp, then_type, "t", source_ids, alias_cond, builder
        )
        else_join_id = self._left_join_dataset(
            node.elseOp, else_type, "e", source_ids, alias_cond, builder
        )

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
        cond_types = [self._get_node_type(c.condition) for c in node.cases]
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
        if cond_source is not None:
            self._left_join_dataset(cond_source, _DATASET, alias, source_ids, alias_src, builder)

        if isinstance(case_obj.condition, AST.VarID) and cond_ds is not None:
            # Bare dataset VarID: reference its boolean measure column
            bool_measure = list(cond_ds.get_measures_names())[0]
            return f"{alias}.{quote_name(bool_measure)}"

        with self._clause_scope(cond_ds, prefix=alias):
            return self.visit(case_obj.condition)

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

            t_type = self._get_node_type(case_obj.thenOp)
            then_types.append(t_type)
            if t_type == _DATASET:
                t_alias = f"t{i}"
                self._left_join_dataset(
                    case_obj.thenOp, _DATASET, t_alias, source_ids, alias_src, builder
                )
                then_aliases.append(t_alias)
            else:
                then_aliases.append(None)

        # Handle else-operand
        else_type = self._get_node_type(node.elseOp)
        else_alias: Optional[str] = None
        if else_type == _DATASET:
            else_alias = "e"
            self._left_join_dataset(
                node.elseOp, _DATASET, else_alias, source_ids, alias_src, builder
            )

        # Build SELECT: identifiers + CASE WHEN per measure (reversed for last-match-wins)
        cols: List[str] = [f"{alias_src}.{quote_name(id_)}" for id_ in source_ids]
        for measure in output_measures:
            case_parts = ["CASE"]
            for i in reversed(range(len(node.cases))):
                then_ref = (
                    f"{then_aliases[i]}.{quote_name(measure)}"
                    if then_types[i] == _DATASET
                    else self.visit(node.cases[i].thenOp)
                )
                case_parts.append(f"WHEN {cond_exprs[i]} THEN {then_ref}")
            else_ref = (
                f"{else_alias}.{quote_name(measure)}"
                if else_type == _DATASET
                else self.visit(node.elseOp)
            )
            case_parts.append(f"ELSE {else_ref} END")
            cols.append(f"{' '.join(case_parts)} AS {quote_name(measure)}")

        builder.select(*cols)

        # Filter: only keep rows where the selected branch has a matching row.
        # Scalar/null branches always match; dataset branches need a LEFT JOIN hit.
        has_ds_branch = any(t == _DATASET for t in then_types) or else_type == _DATASET
        if has_ds_branch:
            id_col = quote_name(source_ids[0])
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
        with self._stash_assignment():
            validation_sql = self.visit(node.validation)

        error_code = self._error_code_sql(node.error_code)
        error_level = self._error_code_sql(node.error_level)

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
            cols.append(f"t.{quote_name(id_name)}")

        # Rename the comparison measure to bool_var.
        cols.append(f't.{quote_name(bool_measure)} AS "bool_var"')

        # Handle imbalance (also with cleared output to prevent renaming).
        join_cond = None
        imbalance_sql = None
        imbalance_col = 'CAST(NULL AS DOUBLE) AS "imbalance"'
        if node.imbalance is not None:
            with self._stash_assignment():
                imbalance_sql = self.visit(node.imbalance)
            imb_ds = self._get_dataset_structure(node.imbalance)
            if imb_ds is not None:
                join_cond = self._join_on_clause(id_names, "t", "i")
                imbalance_col = f'i.{quote_name(imb_ds.get_measures_names()[0])} AS "imbalance"'
        cols.append(imbalance_col)

        # errorcode / errorlevel – set only when bool_var is explicitly FALSE.
        bool_ref = f"t.{quote_name(bool_measure)}"
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
        clause_info: List[Dict[str, Any]] = []
        for i, clause in enumerate(node.clauses):
            alias: Optional[str] = None
            actual_node = clause

            if isinstance(clause, AST.BinOp) and clause.op == tokens.AS:
                actual_node = clause.left
                alias = self._get_node_value(clause.right)

            ds = self._get_dataset_structure(actual_node)
            table_src = self._get_dataset_sql(actual_node)

            if alias is None:
                # Use dataset name as alias (mirrors interpreter convention)
                alias = ds.name if ds else chr(ord("a") + i)

            # Quote alias for SQL if it contains special characters
            sql_alias = quote_name(alias) if ("." in alias or " " in alias) else alias

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
        if node.op != tokens.CROSS_JOIN:
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
                    comp.role == Role.IDENTIFIER and node.op != tokens.CROSS_JOIN
                ) or comp_name in all_join_ids
                if is_join_id:
                    if comp_name not in seen_identifiers:
                        seen_identifiers.add(comp_name)
                        if node.op == tokens.FULL_JOIN and comp_name in all_join_ids:
                            # For FULL JOIN identifiers, use COALESCE to pick
                            # the non-NULL value from either side.
                            coalesce_parts = [
                                f"{ci['sql_alias']}.{quote_name(comp_name)}"
                                for ci in clause_info
                                if ci["ds"] and comp_name in ci["ds"].components
                            ]
                            cols.append(
                                f"COALESCE({', '.join(coalesce_parts)}) AS {quote_name(comp_name)}"
                            )
                        else:
                            cols.append(f"{sa}.{quote_name(comp_name)}")
                elif comp_name in duplicate_comps:
                    # Duplicate non-identifier: alias with "alias#comp" convention
                    qualified_name = f"{info['alias']}#{comp_name}"
                    cols.append(f"{sa}.{quote_name(comp_name)} AS {quote_name(qualified_name)}")
                    self._join_alias_map[qualified_name] = qualified_name
                else:
                    cols.append(f"{sa}.{quote_name(comp_name)}")

        if not cols:
            builder.select_all()
        else:
            builder.select(*cols)

        builder.from_table(clause_info[0]["table_src"], first_sql_alias)

        for idx, info in enumerate(clause_info[1:]):
            join_keys = pairwise_keys[idx]
            if node.op == tokens.CROSS_JOIN:
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
                        f"{left_alias}.{quote_name(id_)} = {info['sql_alias']}.{quote_name(id_)}"
                    )
                on_clause = " AND ".join(on_parts) if on_parts else "1=1"
                builder.join(
                    info["table_src"],
                    info["sql_alias"],
                    on=on_clause,
                    join_type=node.op,
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
            operand_type = self._get_node_type(node.operand)

            # Dataset-level time_agg: apply to the time measure
            if operand_type == _DATASET:
                return self._visit_time_agg_dataset(node, target, conf)

            is_tp = self._is_operand_type(node.operand, TimePeriod)
            operand_sql = self.visit(node.operand)

            if is_tp:
                return f"vtl_time_agg_tp(vtl_period_parse({operand_sql}), '{target}')"
            else:
                agg_expr = f"vtl_time_agg_date({operand_sql}, '{target}')"
                return self._apply_time_agg_conf(agg_expr, conf)
        else:
            # Without-operand case: inside group all, applies to time identifier
            if self._in_clause and self._current_dataset:
                for comp in self._current_dataset.components.values():
                    if comp.data_type == TimePeriod and comp.role == Role.IDENTIFIER:
                        col = quote_name(comp.name)
                        return f"vtl_time_agg_tp(vtl_period_parse({col}), '{target}')"
                for comp in self._current_dataset.components.values():
                    if comp.data_type == Date and comp.role == Role.IDENTIFIER:
                        col = quote_name(comp.name)
                        agg = f"vtl_time_agg_date({col}, '{target}')"
                        return self._apply_time_agg_conf(agg, conf)
            return f"vtl_time_agg_date(CURRENT_DATE, '{target}')"

    @staticmethod
    def _apply_time_agg_conf(expr: str, conf: Optional[str]) -> str:
        """Apply time_agg conf (first/last) modifier to a Date aggregation expression."""
        if conf == "first":
            return f"vtl_tp_start_date(vtl_period_parse({expr}))"
        if conf == "last":
            return f"vtl_tp_end_date(vtl_period_parse({expr}))"
        return expr

    def _visit_time_agg_dataset(
        self, node: AST.TimeAggregation, target: str, conf: Optional[str]
    ) -> str:
        """Visit TIME_AGG at dataset level: apply to time measure."""
        ds = self._get_dataset_structure(node.operand)
        src = self._get_dataset_sql(node.operand)

        # Find time measures to transform
        cols = []
        for comp in ds.components.values():
            col = quote_name(comp.name)
            if comp.role == Role.IDENTIFIER:
                cols.append(col)
            elif comp.data_type == TimePeriod:
                cols.append(f"vtl_time_agg_tp(vtl_period_parse({col}), '{target}') AS {col}")
            elif comp.data_type == Date:
                expr = self._apply_time_agg_conf(f"vtl_time_agg_date({col}, '{target}')", conf)
                cols.append(f"{expr} AS {col}")
            else:
                cols.append(col)

        return f"SELECT {', '.join(cols)} FROM {src}"

    # =========================================================================
    # Eval operator visitor
    # =========================================================================

    def visit_EvalOp(self, node: AST.EvalOp) -> str:
        """Visit EVAL operator (external routine execution)."""
        routine = self.external_routines[node.name]
        query = routine.query.replace('"', "'")

        # Map SQL table names to actual DuckDB table names.
        for table_name in routine.dataset_names:
            for operand in node.operands:
                short_name = operand.value.rsplit(".", 1)[-1]
                if short_name == table_name:
                    op_name = quote_name(operand.value)
                    query = re.sub(rf"\b{re.escape(table_name)}\b", op_name, query)
                    break

        return query

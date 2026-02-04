"""
SQL Transpiler for VTL AST.

This module converts VTL AST nodes to DuckDB-compatible SQL queries.
It follows the same visitor pattern as ASTString.py but generates SQL instead of VTL.

Key concepts:
- Dataset-level operations: Binary ops between datasets use JOIN on identifiers,
  operations apply only to measures.
- Component-level operations: Operations within clauses (calc, filter) work on
  columns of the same dataset.
- Scalar-level operations: Simple SQL expressions.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import vtlengine.AST as AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar.tokens import (
    ABS,
    AGGREGATE,
    AND,
    AVG,
    BETWEEN,
    CALC,
    CAST,
    CEIL,
    CHARSET_MATCH,
    CONCAT,
    COUNT,
    CROSS_JOIN,
    CURRENT_DATE,
    DATE_ADD,
    DATEDIFF,
    DAYOFMONTH,
    DAYOFYEAR,
    DAYTOMONTH,
    DAYTOYEAR,
    DIV,
    DROP,
    EQ,
    EXISTS_IN,
    EXP,
    FILTER,
    FIRST_VALUE,
    FLOOR,
    FLOW_TO_STOCK,
    FULL_JOIN,
    GT,
    GTE,
    IN,
    INNER_JOIN,
    INSTR,
    INTERSECT,
    ISNULL,
    KEEP,
    LAG,
    LAST_VALUE,
    LCASE,
    LEAD,
    LEFT_JOIN,
    LEN,
    LN,
    LOG,
    LT,
    LTE,
    LTRIM,
    MAX,
    MEDIAN,
    MEMBERSHIP,
    MIN,
    MINUS,
    MOD,
    MONTH,
    MONTHTODAY,
    MULT,
    NEQ,
    NOT,
    NOT_IN,
    NVL,
    OR,
    PERIOD_INDICATOR,
    PIVOT,
    PLUS,
    POWER,
    RANDOM,
    RANK,
    RATIO_TO_REPORT,
    RENAME,
    REPLACE,
    ROUND,
    RTRIM,
    SETDIFF,
    SQRT,
    STDDEV_POP,
    STDDEV_SAMP,
    STOCK_TO_FLOW,
    SUBSPACE,
    SUBSTR,
    SUM,
    SYMDIFF,
    TIMESHIFT,
    TRIM,
    TRUNC,
    UCASE,
    UNION,
    UNPIVOT,
    VAR_POP,
    VAR_SAMP,
    XOR,
    YEAR,
    YEARTODAY,
)
from vtlengine.duckdb_transpiler.Transpiler.structure_visitor import StructureVisitor
from vtlengine.Model import Component, Dataset, ExternalRoutine, Role, Scalar, ValueDomain

# =============================================================================
# SQL Operator Mappings
# =============================================================================

SQL_BINARY_OPS: Dict[str, str] = {
    # Arithmetic
    PLUS: "+",
    MINUS: "-",
    MULT: "*",
    DIV: "/",
    MOD: "%",
    # Comparison
    EQ: "=",
    NEQ: "<>",
    GT: ">",
    LT: "<",
    GTE: ">=",
    LTE: "<=",
    # Logical
    AND: "AND",
    OR: "OR",
    XOR: "XOR",
    # String
    CONCAT: "||",
}

# Set operation mappings
SQL_SET_OPS: Dict[str, str] = {
    UNION: "UNION ALL",
    INTERSECT: "INTERSECT",
    SETDIFF: "EXCEPT",
    SYMDIFF: "SYMDIFF",  # Handled specially
}

# VTL to DuckDB type mappings
VTL_TO_DUCKDB_TYPES: Dict[str, str] = {
    "Integer": "BIGINT",
    "Number": "DOUBLE",
    "String": "VARCHAR",
    "Boolean": "BOOLEAN",
    "Date": "DATE",
    "TimePeriod": "VARCHAR",
    "TimeInterval": "VARCHAR",
    "Duration": "VARCHAR",
    "Null": "VARCHAR",
}

SQL_UNARY_OPS: Dict[str, str] = {
    # Arithmetic
    PLUS: "+",
    MINUS: "-",
    CEIL: "CEIL",
    FLOOR: "FLOOR",
    ABS: "ABS",
    EXP: "EXP",
    LN: "LN",
    SQRT: "SQRT",
    # Logical
    NOT: "NOT",
    # String
    LEN: "LENGTH",
    TRIM: "TRIM",
    LTRIM: "LTRIM",
    RTRIM: "RTRIM",
    UCASE: "UPPER",
    LCASE: "LOWER",
    # Time extraction (simple functions)
    YEAR: "YEAR",
    MONTH: "MONTH",
    DAYOFMONTH: "DAY",
    DAYOFYEAR: "DAYOFYEAR",
}

# Time operators that need special handling
SQL_TIME_OPS: Dict[str, str] = {
    CURRENT_DATE: "CURRENT_DATE",
    DATEDIFF: "DATE_DIFF",  # DATE_DIFF('day', d1, d2) in DuckDB
    DATE_ADD: "DATE_ADD",  # date + INTERVAL 'n period'
    TIMESHIFT: "TIMESHIFT",  # Custom handling for time shift
    # Duration conversions
    DAYTOYEAR: "DAYTOYEAR",  # days -> 'PxYxD' format
    DAYTOMONTH: "DAYTOMONTH",  # days -> 'PxMxD' format
    YEARTODAY: "YEARTODAY",  # 'PxYxD' -> days
    MONTHTODAY: "MONTHTODAY",  # 'PxMxD' -> days
}

SQL_AGGREGATE_OPS: Dict[str, str] = {
    SUM: "SUM",
    AVG: "AVG",
    COUNT: "COUNT",
    MIN: "MIN",
    MAX: "MAX",
    MEDIAN: "MEDIAN",
    STDDEV_POP: "STDDEV_POP",
    STDDEV_SAMP: "STDDEV_SAMP",
    VAR_POP: "VAR_POP",
    VAR_SAMP: "VAR_SAMP",
}

SQL_ANALYTIC_OPS: Dict[str, str] = {
    SUM: "SUM",
    AVG: "AVG",
    COUNT: "COUNT",
    MIN: "MIN",
    MAX: "MAX",
    MEDIAN: "MEDIAN",
    STDDEV_POP: "STDDEV_POP",
    STDDEV_SAMP: "STDDEV_SAMP",
    VAR_POP: "VAR_POP",
    VAR_SAMP: "VAR_SAMP",
    FIRST_VALUE: "FIRST_VALUE",
    LAST_VALUE: "LAST_VALUE",
    LAG: "LAG",
    LEAD: "LEAD",
    RANK: "RANK",
    RATIO_TO_REPORT: "RATIO_TO_REPORT",
}


class OperandType:
    """Types of operands in VTL expressions."""

    DATASET = "Dataset"
    COMPONENT = "Component"
    SCALAR = "Scalar"
    CONSTANT = "Constant"


@dataclass
class SQLTranspiler(ASTTemplate):
    """
    Transpiler that converts VTL AST to SQL queries.

    Generates one SQL query per top-level Assignment. Each query can be
    executed sequentially, with results registered as tables for subsequent queries.

    Attributes:
        input_datasets: Dict of input Dataset structures from data_structures.
        output_datasets: Dict of output Dataset structures from semantic analysis.
        input_scalars: Dict of input Scalar values/types from data_structures.
        output_scalars: Dict of output Scalar values/types from semantic analysis.
        available_tables: Tables available for querying (inputs + intermediate results).
        current_dataset: Current dataset context for component-level operations.
        in_clause: Whether we're inside a clause (calc, filter, etc.).
    """

    # Input structures from data_structures
    input_datasets: Dict[str, Dataset] = field(default_factory=dict)
    input_scalars: Dict[str, Scalar] = field(default_factory=dict)

    # Output structures from semantic analysis
    output_datasets: Dict[str, Dataset] = field(default_factory=dict)
    output_scalars: Dict[str, Scalar] = field(default_factory=dict)

    # Value domains and external routines
    value_domains: Dict[str, ValueDomain] = field(default_factory=dict)
    external_routines: Dict[str, ExternalRoutine] = field(default_factory=dict)

    # Runtime state
    available_tables: Dict[str, Dataset] = field(default_factory=dict)
    current_dataset: Optional[Dataset] = None
    current_dataset_alias: str = ""
    in_clause: bool = False
    current_result_name: str = ""  # Target name of current assignment

    # User-defined operators
    udos: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    udo_params: Optional[List[Dict[str, Any]]] = None  # Stack of UDO parameter bindings

    # Datapoint rulesets
    dprs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Structure visitor for computing Dataset structures (initialized in __post_init__)
    structure_visitor: StructureVisitor = field(init=False)

    def __post_init__(self) -> None:
        """Initialize available tables and structure visitor."""
        # Start with input datasets as available tables
        self.available_tables = dict(self.input_datasets)
        self.structure_visitor = StructureVisitor(
            available_tables=self.available_tables,
            output_datasets=self.output_datasets,
            udos=self.udos,
        )

    # =========================================================================
    # Structure Tracking Methods
    # =========================================================================

    def get_structure(self, node: AST.AST) -> Optional[Dataset]:
        """Delegate structure computation to StructureVisitor."""
        return self.structure_visitor.visit(node)

    def set_structure(self, node: AST.AST, dataset: Dataset) -> None:
        """
        Store computed structure for a node.

        Args:
            node: The AST node to store structure for.
            dataset: The computed Dataset structure.
        """
        self.structure_visitor.set_structure(node, dataset)

    def _validate_structure(
        self,
        computed: Dataset,
        expected: Dataset,
        operator_name: str,
    ) -> None:
        """
        Validate computed structure matches semantic analysis.

        Args:
            computed: The structure computed by the transpiler.
            expected: The structure from semantic analysis.
            operator_name: Name of the operator for error messages.

        Raises:
            ValueError: If structures don't match.
        """
        computed_ids = set(computed.get_identifiers_names())
        expected_ids = set(expected.get_identifiers_names())
        computed_measures = set(computed.get_measures_names())
        expected_measures = set(expected.get_measures_names())

        if computed_ids != expected_ids:
            raise ValueError(
                f"{operator_name}: identifier mismatch. "
                f"Computed: {computed_ids}, Expected: {expected_ids}"
            )

        if computed_measures != expected_measures:
            raise ValueError(
                f"{operator_name}: measure mismatch. "
                f"Computed: {computed_measures}, Expected: {expected_measures}"
            )

    def get_udo_param(self, name: str) -> Optional[Any]:
        """
        Look up a UDO parameter by name from the current scope.

        Searches from innermost scope outward through the UDO parameter stack.

        Args:
            name: The parameter name to look up.

        Returns:
            The bound value (AST node, string, or Scalar) if found, None otherwise.
        """
        if self.udo_params is None:
            return None
        for scope in reversed(self.udo_params):
            if name in scope:
                return scope[name]
        return None

    def _resolve_varid_value(self, node: AST.AST) -> str:
        """
        Resolve a VarID value, checking for UDO parameter bindings.

        If the node is a VarID and its value is a UDO parameter name,
        recursively resolves the bound value. For non-VarID nodes or
        non-parameter VarIDs, returns the value directly.

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
            # Recursively resolve if bound to another AST node
            if isinstance(udo_value, (AST.VarID, AST.Identifier)):
                return self._resolve_varid_value(udo_value)
            # String value is the final resolved name
            if isinstance(udo_value, str):
                return udo_value
            return str(udo_value)
        return name

    def transpile(self, ast: AST.Start) -> List[Tuple[str, str, bool]]:
        """
        Transpile the AST to a list of SQL queries.

        Args:
            ast: The root AST node (Start).

        Returns:
            List of (result_name, sql_query, is_persistent) tuples.
        """
        return self.visit(ast)

    def transpile_with_cte(self, ast: AST.Start) -> str:
        """
        Transpile the AST to a single SQL query using CTEs.

        Instead of generating multiple queries where each intermediate result
        is registered as a table, this generates a single query with CTEs
        for all intermediate results.

        Args:
            ast: The root AST node (Start).

        Returns:
            A single SQL query string with CTEs.
        """
        queries = self.visit(ast)

        if len(queries) == 0:
            return ""

        if len(queries) == 1:
            # Single query, no CTEs needed
            return queries[0][1]

        # Build CTEs for all intermediate queries
        cte_parts = []
        for name, sql, _is_persistent in queries[:-1]:
            # Normalize the SQL (remove extra whitespace)
            normalized_sql = " ".join(sql.split())
            cte_parts.append(f'"{name}" AS ({normalized_sql})')

        # Final query is the main SELECT
        final_name, final_sql, _ = queries[-1]
        normalized_final = " ".join(final_sql.split())

        # Combine CTEs with final query
        cte_clause = ",\n    ".join(cte_parts)
        return f"WITH {cte_clause}\n{normalized_final}"

    # =========================================================================
    # Root and Assignment Nodes
    # =========================================================================

    def visit_Start(self, node: AST.Start) -> List[Tuple[str, str, bool]]:
        """Process the root node containing all top-level assignments."""
        queries: List[Tuple[str, str, bool]] = []

        # Pre-populate available_tables with all output structures from semantic analysis
        # This handles forward references where a dataset is used before it's defined
        for name, ds in self.output_datasets.items():
            if name not in self.available_tables:
                self.available_tables[name] = ds

        for child in node.children:
            # Clear structure context before each transformation
            self.structure_visitor.clear_context()

            # Process UDO definitions (these don't generate SQL, just store the definition)
            if isinstance(child, (AST.Operator, AST.DPRuleset)):
                self.visit(child)
            # Process HRuleset definitions (store for later use in hierarchy operations)
            elif isinstance(child, AST.HRuleset):
                pass  # TODO: Implement if needed
            elif isinstance(child, (AST.Assignment, AST.PersistentAssignment)):
                result = self.visit(child)
                if result:
                    name, sql, is_persistent = result
                    queries.append((name, sql, is_persistent))

                    # Register result for subsequent queries
                    # Use output_datasets for intermediate results
                    if name in self.output_datasets:
                        self.available_tables[name] = self.output_datasets[name]

        return queries

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        """Process datapoint ruleset definition and store for later use."""
        # Generate rule names if not provided
        for i, rule in enumerate(node.rules):
            if rule.name is None:
                rule.name = str(i + 1)

        # Build signature mapping
        signature = {}
        if not isinstance(node.params, AST.DefIdentifier):
            for param in node.params:
                if hasattr(param, "alias") and param.alias is not None:
                    signature[param.alias] = param.value
                else:
                    signature[param.value] = param.value

        self.dprs[node.name] = {
            "rules": node.rules,
            "signature": signature,
            "params": (
                [x.value for x in node.params]
                if not isinstance(node.params, AST.DefIdentifier)
                else []
            ),
            "signature_type": node.signature_type,
        }

    def visit_Assignment(self, node: AST.Assignment) -> Tuple[str, str, bool]:
        """Process a temporary assignment (:=)."""
        if not isinstance(node.left, AST.VarID):
            raise ValueError(f"Expected VarID for assignment left, got {type(node.left).__name__}")
        result_name = node.left.value

        # Track current result name for output column resolution
        prev_result_name = self.current_result_name
        self.current_result_name = result_name
        try:
            right_sql = self.visit(node.right)
        finally:
            self.current_result_name = prev_result_name

        # Ensure it's a complete SELECT statement
        sql = self._ensure_select(right_sql)

        return (result_name, sql, False)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Tuple[str, str, bool]:
        """Process a persistent assignment (<-)."""
        if not isinstance(node.left, AST.VarID):
            raise ValueError(f"Expected VarID for assignment left, got {type(node.left).__name__}")
        result_name = node.left.value

        # Track current result name for output column resolution
        prev_result_name = self.current_result_name
        self.current_result_name = result_name
        try:
            right_sql = self.visit(node.right)
        finally:
            self.current_result_name = prev_result_name

        sql = self._ensure_select(right_sql)

        return (result_name, sql, True)

    # =========================================================================
    # User-Defined Operators
    # =========================================================================

    def visit_Operator(self, node: AST.Operator) -> None:
        """
        Process a User-Defined Operator definition.

        Stores the UDO definition for later expansion when called.
        """
        if node.op in self.udos:
            raise ValueError(f"User Defined Operator {node.op} already exists")

        param_info: List[Dict[str, Any]] = []
        for param in node.parameters:
            if param.name in [x["name"] for x in param_info]:
                raise ValueError(f"Duplicated Parameter {param.name} in UDO {node.op}")
            # Store parameter info
            param_info.append(
                {
                    "name": param.name,
                    "type": param.type_.__class__.__name__
                    if hasattr(param.type_, "__class__")
                    else str(param.type_),
                }
            )

        self.udos[node.op] = {
            "params": param_info,
            "expression": node.expression,
            "output": node.output_type,
        }

    def visit_UDOCall(self, node: AST.UDOCall) -> str:
        """
        Process a User-Defined Operator call.

        Expands the UDO by visiting its expression with parameter substitution.
        """
        if node.op not in self.udos:
            raise ValueError(f"User Defined Operator {node.op} not found")

        operator = self.udos[node.op]

        # Initialize UDO params stack if needed
        if self.udo_params is None:
            self.udo_params = []

        # Build parameter bindings - store AST nodes for substitution
        param_bindings: Dict[str, Any] = {}
        for i, param in enumerate(operator["params"]):
            if i < len(node.params):
                param_node = node.params[i]
                # Store the AST node directly for proper substitution
                param_bindings[param["name"]] = param_node

        # Push parameter bindings onto stack (both transpiler and structure_visitor)
        self.udo_params.append(param_bindings)
        self.structure_visitor.push_udo_params(param_bindings)

        # Visit the UDO expression with a deep copy to avoid modifying the original
        # Parameter resolution happens via get_udo_param() in visit_VarID and _get_operand_type
        expression_copy = deepcopy(operator["expression"])

        try:
            # Visit the expression - parameters are resolved via mapping lookup
            result = self.visit(expression_copy)
        finally:
            # Pop parameter bindings (both transpiler and structure_visitor)
            self.udo_params.pop()
            if len(self.udo_params) == 0:
                self.udo_params = None
            self.structure_visitor.pop_udo_params()

        return result

    # =========================================================================
    # Variable and Constant Nodes
    # =========================================================================

    def visit_VarID(self, node: AST.VarID) -> str:
        """
        Process a variable identifier.

        Returns table reference, column reference, or scalar value depending on context.
        """
        name = node.value

        # Check if this is a UDO parameter reference (mapping lookup approach)
        udo_value = self.get_udo_param(name)
        if udo_value is not None:
            # If bound to another AST node, visit it
            if isinstance(udo_value, AST.AST):
                return self.visit(udo_value)
            # If bound to a string (dataset/component name), return it quoted
            if isinstance(udo_value, str):
                return f'"{udo_value}"'
            # If bound to a Scalar, return its SQL representation
            if isinstance(udo_value, Scalar):
                return self._scalar_to_sql(udo_value)
            return str(udo_value)

        # In clause context: it's a component (column) reference
        if self.in_clause and self.current_dataset and name in self.current_dataset.components:
            return f'"{name}"'

        # Check if it's a known dataset
        if (
            name in self.available_tables
            or name in self.input_scalars
            or name in self.output_scalars
        ):
            return f'"{name}"'

        # Check if it's a known scalar (from input or output)
        if name in self.input_scalars:
            return self._scalar_to_sql(self.input_scalars[name])
        if name in self.output_scalars:
            return self._scalar_to_sql(self.output_scalars[name])

        # Default: treat as column reference (for component operations)
        return f'"{name}"'

    def visit_Constant(self, node: AST.Constant) -> str:  # type: ignore[override]
        """Convert a constant to SQL literal."""
        if node.value is None:
            return "NULL"

        if node.type_ in ("STRING_CONSTANT", "String"):
            escaped = str(node.value).replace("'", "''")
            return f"'{escaped}'"
        elif node.type_ in ("INTEGER_CONSTANT", "Integer"):
            return str(int(node.value))
        elif node.type_ in ("FLOAT_CONSTANT", "Number"):
            return str(float(node.value))
        elif node.type_ in ("BOOLEAN_CONSTANT", "Boolean"):
            return "TRUE" if node.value else "FALSE"
        elif node.type_ == "NULL_CONSTANT":
            return "NULL"
        else:
            return str(node.value)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        """Process a parameter constant."""
        if node.value is None:
            return "NULL"
        return str(node.value)

    def visit_Identifier(self, node: AST.Identifier) -> str:
        """Process an identifier."""
        return f'"{node.value}"'

    def visit_Collection(self, node: AST.Collection) -> str:  # type: ignore[override]
        """
        Process a collection (set of values or value domain reference).

        For Set kind: returns SQL literal list like (1, 2, 3)
        For ValueDomain kind: looks up the value domain and returns its values as SQL literal list
        """
        if node.kind == "ValueDomain":
            # Look up the value domain by name
            vd_name = node.name
            if not self.value_domains:
                raise ValueError(
                    f"Value domain '{vd_name}' referenced but no value domains provided"
                )
            if vd_name not in self.value_domains:
                raise ValueError(f"Value domain '{vd_name}' not found")

            vd = self.value_domains[vd_name]
            # Convert value domain setlist to SQL literals
            sql_values = [self._value_to_sql_literal(v, vd.type.__name__) for v in vd.setlist]
            return f"({', '.join(sql_values)})"

        # Default: Set kind - process children as values
        values = [self.visit(child) for child in node.children]
        return f"({', '.join(values)})"

    def _value_to_sql_literal(self, value: Any, type_name: str) -> str:
        """Convert a Python value to SQL literal based on its type."""
        if value is None:
            return "NULL"
        if type_name == "String":
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"
        elif type_name in ("Integer", "Number"):
            return str(value)
        elif type_name == "Boolean":
            return "TRUE" if value else "FALSE"
        elif type_name == "Date":
            return f"DATE '{value}'"
        else:
            # Default: treat as string
            escaped = str(value).replace("'", "''")
            return f"'{escaped}'"

    # =========================================================================
    # Binary Operations
    # =========================================================================

    def visit_BinOp(self, node: AST.BinOp) -> str:  # type: ignore[override]
        """
        Process a binary operation.

        Dispatches based on operand types:
        - Dataset-Dataset: JOIN with operation on measures
        - Dataset-Scalar: Operation on all measures
        - Scalar-Scalar / Component-Component: Simple expression
        """
        left_type = self._get_operand_type(node.left)
        right_type = self._get_operand_type(node.right)

        op = str(node.op).lower()

        # Special handling for IN / NOT IN
        if op in (IN, NOT_IN, "not in"):
            return self._visit_in_op(node, is_not=(op in (NOT_IN, "not in")))

        # Special handling for MATCH_CHARACTERS (regex)
        if op in (CHARSET_MATCH, "match"):
            return self._visit_match_op(node)

        # Special handling for EXIST_IN
        if op == EXISTS_IN:
            return self._visit_exist_in(node)

        # Special handling for NVL (coalesce)
        if op == NVL:
            return self._visit_nvl_binop(node)

        # Special handling for MEMBERSHIP (#) operator
        if op == MEMBERSHIP:
            return self._visit_membership(node)

        # Special handling for DATEDIFF (date difference)
        if op == DATEDIFF:
            return self._visit_datediff(node, left_type, right_type)

        # Special handling for TIMESHIFT
        if op == TIMESHIFT:
            return self._visit_timeshift(node, left_type, right_type)

        # Special handling for RANDOM (parsed as BinOp in VTL grammar)
        if op == RANDOM:
            return self._visit_random_binop(node, left_type, right_type)

        sql_op = SQL_BINARY_OPS.get(op, op.upper())

        # Dataset-Dataset
        if left_type == OperandType.DATASET and right_type == OperandType.DATASET:
            return self._binop_dataset_dataset(node.left, node.right, sql_op)

        # Dataset-Scalar
        if left_type == OperandType.DATASET and right_type == OperandType.SCALAR:
            return self._binop_dataset_scalar(node.left, node.right, sql_op, left=True)

        # Scalar-Dataset
        if left_type == OperandType.SCALAR and right_type == OperandType.DATASET:
            return self._binop_dataset_scalar(node.right, node.left, sql_op, left=False)

        # Scalar-Scalar or Component-Component
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)

        # Check if this is a TimePeriod comparison (requires special handling)
        if op in (EQ, NEQ, GT, LT, GTE, LTE) and self._is_time_period_comparison(
            node.left, node.right
        ):
            return self._visit_time_period_comparison(left_sql, right_sql, sql_op)

        # Check if this is a TimeInterval comparison (requires special handling)
        if op in (EQ, NEQ, GT, LT, GTE, LTE) and self._is_time_interval_comparison(
            node.left, node.right
        ):
            return self._visit_time_interval_comparison(left_sql, right_sql, sql_op)

        return f"({left_sql} {sql_op} {right_sql})"

    def _visit_in_op(self, node: AST.BinOp, is_not: bool) -> str:
        """
        Handle IN / NOT IN operations.

        VTL: x in {1, 2, 3} or ds in {1, 2, 3}
        SQL: x IN (1, 2, 3) or x NOT IN (1, 2, 3)
        """
        left_type = self._get_operand_type(node.left)
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)  # Should be a Collection

        sql_op = "NOT IN" if is_not else "IN"

        # Dataset-level operation
        if left_type == OperandType.DATASET:
            return self._in_dataset(node.left, right_sql, sql_op)

        # Scalar/Component level
        return f"({left_sql} {sql_op} {right_sql})"

    def _in_dataset(self, dataset_node: AST.AST, values_sql: str, sql_op: str) -> str:
        """
        Generate SQL for dataset-level IN/NOT IN operation.

        Uses structure tracking to get dataset structure.
        """
        ds = self.get_structure(dataset_node)

        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        measure_select = ", ".join(
            [f'("{m}" {sql_op} {values_sql}) AS "{m}"' for m in ds.get_measures_names()]
        )

        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _visit_match_op(self, node: AST.BinOp) -> str:
        """
        Handle MATCH_CHARACTERS (regex) operation.

        VTL: match_characters(str, pattern)
        SQL: regexp_full_match(str, pattern)
        """
        left_type = self._get_operand_type(node.left)
        left_sql = self.visit(node.left)
        pattern_sql = self.visit(node.right)

        # Dataset-level operation
        if left_type == OperandType.DATASET:
            return self._match_dataset(node.left, pattern_sql)

        # Scalar/Component level - DuckDB uses regexp_full_match
        return f"regexp_full_match({left_sql}, {pattern_sql})"

    def _match_dataset(self, dataset_node: AST.AST, pattern_sql: str) -> str:
        """
        Generate SQL for dataset-level MATCH operation.

        Uses structure tracking to get dataset structure.
        """
        ds = self.get_structure(dataset_node)

        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        measure_select = ", ".join(
            [f'regexp_full_match("{m}", {pattern_sql}) AS "{m}"' for m in ds.get_measures_names()]
        )

        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _visit_exist_in(self, node: AST.BinOp) -> str:
        """
        Handle EXIST_IN operation.

        VTL: exist_in(ds1, ds2) - checks if identifiers from ds1 exist in ds2
        SQL: SELECT *, EXISTS(SELECT 1 FROM ds2 WHERE ids match) AS bool_var

        Uses structure tracking to get dataset structures.
        """
        left_ds = self.get_structure(node.left)
        right_ds = self.get_structure(node.right)

        if left_ds is None or right_ds is None:
            left_name = self._get_dataset_name(node.left)
            right_name = self._get_dataset_name(node.right)
            raise ValueError(f"Cannot resolve dataset structures for {left_name} and {right_name}")

        # Find common identifiers
        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        common_ids = sorted(left_ids.intersection(right_ids))

        if not common_ids:
            raise ValueError(f"No common identifiers between {left_name} and {right_name}")

        # Build EXISTS condition
        conditions = [f'l."{id}" = r."{id}"' for id in common_ids]
        where_clause = " AND ".join(conditions)

        # Select identifiers from left
        id_select = ", ".join([f'l."{k}"' for k in left_ds.get_identifiers_names()])

        left_sql = self._get_dataset_sql(node.left)
        right_sql = self._get_dataset_sql(node.right)

        return f"""
            SELECT {id_select},
                   EXISTS(SELECT 1 FROM ({right_sql}) AS r WHERE {where_clause}) AS "bool_var"
            FROM ({left_sql}) AS l
        """

    def _visit_nvl_binop(self, node: AST.BinOp) -> str:
        """
        Handle NVL operation when parsed as BinOp.

        VTL: nvl(ds, value) - replace nulls with value
        SQL: COALESCE(col, value)

        Uses structure tracking to get dataset structure.
        """
        left_type = self._get_operand_type(node.left)
        replacement = self.visit(node.right)

        # Dataset-level NVL
        if left_type == OperandType.DATASET:
            # Use structure tracking - get_structure handles all expression types
            ds = self.get_structure(node.left)

            if ds is None:
                ds_name = self._get_dataset_name(node.left)
                raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

            id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
            measure_parts = []
            for m in ds.get_measures_names():
                measure_parts.append(f'COALESCE("{m}", {replacement}) AS "{m}"')
            measure_select = ", ".join(measure_parts)

            dataset_sql = self._get_dataset_sql(node.left)
            from_clause = self._simplify_from_clause(dataset_sql)

            return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

        # Scalar/Component level
        left_sql = self.visit(node.left)
        return f"COALESCE({left_sql}, {replacement})"

    def _visit_membership(self, node: AST.BinOp) -> str:
        """
        Handle MEMBERSHIP (#) operation.

        VTL: DS#comp - extracts component 'comp' from dataset 'DS'
        Returns a dataset with identifiers and the specified component as measure.

        Uses structure tracking to get dataset structure.

        SQL: SELECT identifiers, "comp" FROM "DS"
        """
        # Get structure using structure tracking
        ds = self.get_structure(node.left)

        if not ds:
            # Fallback: just reference the component
            left_sql = self.visit(node.left)
            right_sql = self.visit(node.right)
            return f'{left_sql}."{right_sql}"'

        # Get component name from right operand, resolving UDO parameters
        comp_name = self._resolve_varid_value(node.right)

        # Build SELECT with identifiers and the specified component
        id_cols = ds.get_identifiers_names()
        id_select = ", ".join([f'"{k}"' for k in id_cols])

        dataset_sql = self._get_dataset_sql(node.left)
        from_clause = self._simplify_from_clause(dataset_sql)

        if id_select:
            return f'SELECT {id_select}, "{comp_name}" FROM {from_clause}'
        else:
            return f'SELECT "{comp_name}" FROM {from_clause}'

    def _binop_dataset_dataset(self, left_node: AST.AST, right_node: AST.AST, sql_op: str) -> str:
        """
        Generate SQL for Dataset-Dataset binary operation.

        Uses structure tracking: visits children first (storing their structures),
        then uses get_structure() to retrieve them for SQL generation.

        Joins on common identifiers, applies operation to common measures.
        """
        # Step 1: Generate SQL for operands (this also stores their structures)
        if isinstance(left_node, AST.VarID):
            left_sql = f'"{left_node.value}"'
        else:
            left_sql = f"({self.visit(left_node)})"

        if isinstance(right_node, AST.VarID):
            right_sql = f'"{right_node.value}"'
        else:
            right_sql = f"({self.visit(right_node)})"

        # Step 2: Get structures using structure tracking
        # (get_structure already handles VarID -> available_tables fallback)
        left_ds = self.get_structure(left_node)
        right_ds = self.get_structure(right_node)

        if left_ds is None or right_ds is None:
            left_name = self._get_dataset_name(left_node)
            right_name = self._get_dataset_name(right_node)
            raise ValueError(f"Cannot resolve dataset structures for {left_name} and {right_name}")

        # Step 3: Get output structure from semantic analysis
        output_ds = None
        if self.current_result_name and self.current_result_name in self.output_datasets:
            output_ds = self.output_datasets[self.current_result_name]

        # Step 4: Generate SQL using the structures
        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        join_keys = sorted(left_ids.intersection(right_ids))

        if not join_keys:
            left_name = self._get_dataset_name(left_node)
            right_name = self._get_dataset_name(right_node)
            raise ValueError(f"No common identifiers between {left_name} and {right_name}")

        # Build JOIN condition
        join_cond = " AND ".join([f'a."{k}" = b."{k}"' for k in join_keys])

        # SELECT identifiers - include all from both datasets
        # Common identifiers come from 'a', non-common from their respective tables
        all_ids = sorted(left_ids.union(right_ids))
        id_parts = []
        for k in all_ids:
            if k in left_ids:
                id_parts.append(f'a."{k}"')
            else:
                id_parts.append(f'b."{k}"')
        id_select = ", ".join(id_parts)

        # Find source measures (what we're operating on)
        left_measures = set(left_ds.get_measures_names())
        right_measures = set(right_ds.get_measures_names())
        common_measures = sorted(left_measures.intersection(right_measures))

        # Check if output has bool_var (comparison result)
        # Use output_datasets from semantic analysis to determine output measure names
        output_measures = list(output_ds.get_measures_names()) if output_ds else []
        has_bool_var = "bool_var" in output_measures

        # For comparisons, extract the actual measure name from the transformed operands
        # The SQL subqueries already handle keep/rename, so we need to know the final name
        if has_bool_var:
            # Extract the final measure name from each operand after transformations
            left_measure = self._get_transformed_measure_name(left_node)
            right_measure = self._get_transformed_measure_name(right_node)

            if left_measure and right_measure:
                # Both sides should have the same measure name after rename
                # Use the left measure name (they should match)
                measure_select = f'(a."{left_measure}" {sql_op} b."{right_measure}") AS "bool_var"'
            elif common_measures:
                # Fallback to common measures
                m = common_measures[0]
                measure_select = f'(a."{m}" {sql_op} b."{m}") AS "bool_var"'
            else:
                measure_select = ""
        elif common_measures:
            # Regular operation on measures
            measure_select = ", ".join(
                [f'(a."{m}" {sql_op} b."{m}") AS "{m}"' for m in common_measures]
            )
        else:
            measure_select = ""

        return f"""
                    SELECT {id_select}, {measure_select}
                    FROM {left_sql} AS a
                    INNER JOIN {right_sql} AS b ON {join_cond}
                """

    def _binop_dataset_scalar(
        self,
        dataset_node: AST.AST,
        scalar_node: AST.AST,
        sql_op: str,
        left: bool,
    ) -> str:
        """
        Generate SQL for Dataset-Scalar binary operation.

        Uses structure tracking to get dataset structure.
        Applies scalar to all measures.
        """
        scalar_sql = self.visit(scalar_node)

        # Step 1: Generate SQL for dataset (this also stores its structure)
        if isinstance(dataset_node, AST.VarID):
            ds_sql = f'"{dataset_node.value}"'
        else:
            ds_sql = f"({self.visit(dataset_node)})"

        # Step 2: Get structure using structure tracking
        # (get_structure already handles VarID -> available_tables fallback)
        ds = self.get_structure(dataset_node)

        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        # Step 3: Get output structure from semantic analysis
        output_ds = None
        if self.current_result_name and self.current_result_name in self.output_datasets:
            output_ds = self.output_datasets[self.current_result_name]

        # Step 4: Generate SQL using the structures
        id_cols = list(ds.get_identifiers_names())
        measure_names = list(ds.get_measures_names())

        # SELECT identifiers
        id_select = ", ".join([f'"{k}"' for k in id_cols])

        # Check if output has bool_var (comparison result)
        # Use output_datasets from semantic analysis to determine output measure names
        output_measures = list(output_ds.get_measures_names()) if output_ds else []
        has_bool_var = "bool_var" in output_measures

        # SELECT measures with operation
        if left:
            if has_bool_var and measure_names:
                # Single measure comparison -> bool_var
                measure_select = f'("{measure_names[0]}" {sql_op} {scalar_sql}) AS "bool_var"'
            else:
                measure_select = ", ".join(
                    [f'("{m}" {sql_op} {scalar_sql}) AS "{m}"' for m in measure_names]
                )
        else:
            if has_bool_var and measure_names:
                # Single measure comparison -> bool_var
                measure_select = f'({scalar_sql} {sql_op} "{measure_names[0]}") AS "bool_var"'
            else:
                measure_select = ", ".join(
                    [f'({scalar_sql} {sql_op} "{m}") AS "{m}"' for m in measure_names]
                )

        return f"SELECT {id_select}, {measure_select} FROM {ds_sql}"

    def _visit_datediff(self, node: AST.BinOp, left_type: str, right_type: str) -> str:
        """
        Generate SQL for DATEDIFF operator.

        VTL: datediff(date1, date2) returns the absolute number of days between two dates
        DuckDB: ABS(DATE_DIFF('day', date1, date2))
        """
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)

        # For scalar operands, use direct DATE_DIFF
        return f"ABS(DATE_DIFF('day', {left_sql}, {right_sql}))"

    def _visit_timeshift(self, node: AST.BinOp, left_type: str, right_type: str) -> str:
        """
        Generate SQL for TIMESHIFT operator.

        VTL: timeshift(ds, n) shifts dates by n periods
        The right operand is the shift value (scalar).

        For DuckDB, this depends on the data type:
        - Date: date + INTERVAL 'n days' (or use detected frequency)
        - TimePeriod: Complex string manipulation

        Uses structure tracking to get dataset structure.
        """
        if left_type != OperandType.DATASET:
            raise ValueError("timeshift requires a dataset as first operand")

        ds = self.get_structure(node.left)
        if ds is None:
            ds_name = self._get_dataset_name(node.left)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        shift_val = self.visit(node.right)

        # Find time identifier
        time_id, other_ids = self._get_time_and_other_ids(ds)

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        measure_select = ", ".join([f'"{m}"' for m in ds.get_measures_names()])

        # For Date type, use INTERVAL
        # For TimePeriod, we'd need complex string manipulation (not fully supported)
        time_comp = ds.components.get(time_id)
        from vtlengine.DataTypes import Date, TimePeriod

        dataset_sql = self._get_dataset_sql(node.left)

        # Prepare other identifiers for select
        other_id_select = ", ".join([f'"{k}"' for k in other_ids])
        if other_id_select:
            other_id_select += ", "

        if time_comp and time_comp.data_type == Date:
            # Simple date shift using INTERVAL days
            # Note: VTL timeshift uses the frequency of the data
            time_expr = f'("{time_id}" + INTERVAL ({shift_val}) DAY) AS "{time_id}"'
            return f"""
                SELECT {other_id_select}{time_expr}, {measure_select}
                FROM ({dataset_sql}) AS t
            """
        elif time_comp and time_comp.data_type == TimePeriod:
            # Use vtl_period_shift for proper period arithmetic on all period types
            # Parse VARCHAR → STRUCT, shift, format back → VARCHAR
            time_expr = (
                f"vtl_period_to_string(vtl_period_shift("
                f'vtl_period_parse("{time_id}"), {shift_val})) AS "{time_id}"'
            )
            from_clause = self._simplify_from_clause(dataset_sql)
            return f"""
                SELECT {other_id_select}{time_expr}, {measure_select}
                FROM {from_clause}
            """
        else:
            # Fallback: return as-is (shift not applied)
            from_clause = self._simplify_from_clause(dataset_sql)
            return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _visit_random_binop(self, node: AST.BinOp, left_type: str, right_type: str) -> str:
        """
        Generate SQL for RANDOM operator (parsed as BinOp in VTL grammar).

        VTL: random(seed, index) -> deterministic pseudo-random Number between 0 and 1.

        Uses hash-based approach for determinism: same seed + index = same result.
        DuckDB: (ABS(hash(seed || '_' || index)) % 1000000) / 1000000.0
        """
        seed_sql = self.visit(node.left)
        index_sql = self.visit(node.right)

        # Template for random generation
        random_expr = (
            f"(ABS(hash(CAST({seed_sql} AS VARCHAR) || '_' || "
            f"CAST({index_sql} AS VARCHAR))) % 1000000) / 1000000.0"
        )

        # Dataset-level operation - uses structure tracking
        if left_type == OperandType.DATASET:
            ds = self.get_structure(node.left)
            if ds:
                id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
                measure_parts = []
                for m in ds.get_measures_names():
                    m_random = (
                        f"(ABS(hash(CAST(\"{m}\" AS VARCHAR) || '_' || "
                        f'CAST({index_sql} AS VARCHAR))) % 1000000) / 1000000.0 AS "{m}"'
                    )
                    measure_parts.append(m_random)
                measure_select = ", ".join(measure_parts)
                dataset_sql = self._get_dataset_sql(node.left)
                from_clause = self._simplify_from_clause(dataset_sql)
                if id_select:
                    return f"SELECT {id_select}, {measure_select} FROM {from_clause}"
                return f"SELECT {measure_select} FROM {from_clause}"

        # Scalar-level: return the expression directly
        return random_expr

    # =========================================================================
    # Unary Operations
    # =========================================================================

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Process a unary operation."""
        op = str(node.op).lower()
        operand_type = self._get_operand_type(node.operand)

        # Special case: isnull
        if op == ISNULL:
            if operand_type == OperandType.DATASET:
                return self._unary_dataset_isnull(node.operand)
            operand_sql = self.visit(node.operand)
            return f"({operand_sql} IS NULL)"

        # Special case: flow_to_stock (cumulative sum over time)
        if op == FLOW_TO_STOCK:
            return self._visit_flow_to_stock(node.operand, operand_type)

        # Special case: stock_to_flow (difference over time)
        if op == STOCK_TO_FLOW:
            return self._visit_stock_to_flow(node.operand, operand_type)

        # Special case: period_indicator (extracts period indicator from TimePeriod)
        if op == PERIOD_INDICATOR:
            return self._visit_period_indicator(node.operand, operand_type)

        # Time extraction operators (year, month, day, dayofyear)
        if op in (YEAR, MONTH, DAYOFMONTH, DAYOFYEAR):
            return self._visit_time_extraction(node.operand, operand_type, op)

        # Duration conversion operators
        if op in (DAYTOYEAR, DAYTOMONTH, YEARTODAY, MONTHTODAY):
            return self._visit_duration_conversion(node.operand, operand_type, op)

        sql_op = SQL_UNARY_OPS.get(op, op.upper())

        # Dataset-level unary
        if operand_type == OperandType.DATASET:
            return self._unary_dataset(node.operand, sql_op, op)

        # Scalar/Component level
        operand_sql = self.visit(node.operand)

        if op in (PLUS, MINUS):
            return f"({sql_op}{operand_sql})"
        elif op == NOT:
            return f"(NOT {operand_sql})"
        else:
            return f"{sql_op}({operand_sql})"

    def _unary_dataset(self, dataset_node: AST.AST, sql_op: str, op: str) -> str:
        """
        Generate SQL for dataset unary operation.

        Uses structure tracking to get dataset structure.
        """
        # Step 1: Get structure using structure tracking
        # (get_structure already handles VarID -> available_tables fallback)
        ds = self.get_structure(dataset_node)

        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_cols = list(ds.get_identifiers_names())
        input_measures = list(ds.get_measures_names())

        id_select = ", ".join([f'"{k}"' for k in id_cols])

        # Get output measure names from semantic analysis if available
        if self.current_result_name and self.current_result_name in self.output_datasets:
            output_ds = self.output_datasets[self.current_result_name]
            output_measures = list(output_ds.get_measures_names())
        else:
            output_measures = input_measures

        # Build measure select with correct input/output names
        measure_parts = []
        for i, input_m in enumerate(input_measures):
            output_m = output_measures[i] if i < len(output_measures) else input_m
            if op in (PLUS, MINUS):
                measure_parts.append(f'({sql_op}"{input_m}") AS "{output_m}"')
            else:
                measure_parts.append(f'{sql_op}("{input_m}") AS "{output_m}"')
        measure_select = ", ".join(measure_parts)

        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _unary_dataset_isnull(self, dataset_node: AST.AST) -> str:
        """
        Generate SQL for dataset isnull operation.

        Uses structure tracking to get dataset structure.
        """
        # Step 1: Get structure using structure tracking
        # (get_structure already handles VarID -> available_tables fallback)
        ds = self.get_structure(dataset_node)

        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_cols = list(ds.get_identifiers_names())
        measures = list(ds.get_measures_names())

        id_select = ", ".join([f'"{k}"' for k in id_cols])
        # isnull produces boolean output named bool_var
        if len(measures) == 1:
            measure_select = f'("{measures[0]}" IS NULL) AS "bool_var"'
        else:
            measure_select = ", ".join([f'("{m}" IS NULL) AS "{m}"' for m in measures])

        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    # =========================================================================
    # Time Operators
    # =========================================================================

    def _visit_time_extraction(self, operand: AST.AST, operand_type: str, op: str) -> str:
        """
        Generate SQL for time extraction operators (year, month, dayofmonth, dayofyear).

        For Date type, uses DuckDB built-in functions: YEAR(), MONTH(), DAY(), DAYOFYEAR()
        For TimePeriod type, uses vtl_period_year() for YEAR extraction.
        """
        sql_func = SQL_UNARY_OPS.get(op, op.upper())

        if operand_type == OperandType.DATASET:
            return self._time_extraction_dataset(operand, sql_func, op)

        # Check if this is a TimePeriod component - use vtl_period_year
        if op == YEAR and self._is_time_period_operand(operand):
            operand_sql = self.visit(operand)
            return f"vtl_period_year(vtl_period_parse({operand_sql}))"

        operand_sql = self.visit(operand)
        return f"{sql_func}({operand_sql})"

    def _time_extraction_dataset(self, dataset_node: AST.AST, sql_func: str, op: str) -> str:
        """
        Generate SQL for dataset time extraction operation.

        Uses structure tracking to get dataset structure.
        """
        from vtlengine.DataTypes import TimePeriod

        ds = self.get_structure(dataset_node)
        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        # Apply time extraction to time-typed measures
        # Use vtl_period_year for TimePeriod columns when extracting YEAR
        measure_parts = []
        for m_name in ds.get_measures_names():
            comp = ds.components.get(m_name)
            if comp and comp.data_type == TimePeriod and op == YEAR:
                # Use vtl_period_year for TimePeriod YEAR extraction
                measure_parts.append(f'vtl_period_year(vtl_period_parse("{m_name}")) AS "{m_name}"')
            else:
                measure_parts.append(f'{sql_func}("{m_name}") AS "{m_name}"')

        measure_select = ", ".join(measure_parts)
        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)
        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _visit_flow_to_stock(self, operand: AST.AST, operand_type: str) -> str:
        """
        Generate SQL for flow_to_stock (cumulative sum over time).

        This uses a window function: SUM(measure) OVER (PARTITION BY other_ids ORDER BY time_id)

        Uses structure tracking to get dataset structure.
        """
        if operand_type != OperandType.DATASET:
            raise ValueError("flow_to_stock requires a dataset operand")

        ds = self.get_structure(operand)
        if ds is None:
            ds_name = self._get_dataset_name(operand)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        dataset_sql = self._get_dataset_sql(operand)

        # Find time identifier and other identifiers
        time_id, other_ids = self._get_time_and_other_ids(ds)

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        # Create cumulative sum for each measure
        quoted_ids = ['"' + i + '"' for i in other_ids]
        partition_clause = f"PARTITION BY {', '.join(quoted_ids)}" if other_ids else ""
        order_clause = f'ORDER BY "{time_id}"'

        measure_selects = []
        for m in ds.get_measures_names():
            window = f"OVER ({partition_clause} {order_clause})"
            measure_selects.append(f'SUM("{m}") {window} AS "{m}"')

        measure_select = ", ".join(measure_selects)
        from_clause = self._simplify_from_clause(dataset_sql)
        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _visit_stock_to_flow(self, operand: AST.AST, operand_type: str) -> str:
        """
        Generate SQL for stock_to_flow (difference over time).

        This uses: measure - LAG(measure) OVER (PARTITION BY other_ids ORDER BY time_id)

        Uses structure tracking to get dataset structure.
        """
        if operand_type != OperandType.DATASET:
            raise ValueError("stock_to_flow requires a dataset operand")

        ds = self.get_structure(operand)
        if ds is None:
            ds_name = self._get_dataset_name(operand)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        dataset_sql = self._get_dataset_sql(operand)

        # Find time identifier and other identifiers
        time_id, other_ids = self._get_time_and_other_ids(ds)

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        # Create difference from previous for each measure
        quoted_ids = ['"' + i + '"' for i in other_ids]
        partition_clause = f"PARTITION BY {', '.join(quoted_ids)}" if other_ids else ""
        order_clause = f'ORDER BY "{time_id}"'

        measure_selects = []
        for m in ds.get_measures_names():
            window = f"OVER ({partition_clause} {order_clause})"
            # COALESCE handles first row where LAG returns NULL
            measure_selects.append(f'COALESCE("{m}" - LAG("{m}") {window}, "{m}") AS "{m}"')

        measure_select = ", ".join(measure_selects)
        from_clause = self._simplify_from_clause(dataset_sql)
        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _get_time_and_other_ids(self, ds: Dataset) -> Tuple[str, List[str]]:
        """
        Get the time identifier and other identifiers from a dataset.

        Returns (time_id_name, other_id_names).
        Time identifier is detected by data type (Date, TimePeriod, TimeInterval).
        """
        from vtlengine.DataTypes import Date, TimeInterval, TimePeriod

        time_id = None
        other_ids = []

        for id_comp in ds.get_identifiers():
            if id_comp.data_type in (Date, TimePeriod, TimeInterval):
                time_id = id_comp.name
            else:
                other_ids.append(id_comp.name)

        # If no time identifier found, use the last identifier
        if time_id is None:
            id_names = ds.get_identifiers_names()
            if id_names:
                time_id = id_names[-1]
                other_ids = id_names[:-1]
            else:
                time_id = ""

        return time_id, other_ids

    def _is_time_period_operand(self, node: AST.AST) -> bool:
        """
        Check if a node represents a TimePeriod component.

        Only works when in_clause is True and current_dataset is set.
        """
        from vtlengine.DataTypes import TimePeriod

        if not self.in_clause or not self.current_dataset:
            return False

        # Check if it's a VarID pointing to a TimePeriod component
        if isinstance(node, AST.VarID):
            comp = self.current_dataset.components.get(node.value)
            return comp is not None and comp.data_type == TimePeriod

        return False

    def _is_time_interval_operand(self, node: AST.AST) -> bool:
        """
        Check if a node represents a TimeInterval component.

        Only works when in_clause is True and current_dataset is set.
        """
        from vtlengine.DataTypes import TimeInterval

        if not self.in_clause or not self.current_dataset:
            return False

        # Check if it's a VarID pointing to a TimeInterval component
        if isinstance(node, AST.VarID):
            comp = self.current_dataset.components.get(node.value)
            return comp is not None and comp.data_type == TimeInterval

        return False

    def _is_time_period_comparison(self, left: AST.AST, right: AST.AST) -> bool:
        """
        Check if this is a comparison between TimePeriod operands.

        Returns True if at least one operand is a TimePeriod component
        and the other is either a TimePeriod component or a string constant.
        """
        left_is_tp = self._is_time_period_operand(left)
        right_is_tp = self._is_time_period_operand(right)

        # If one is TimePeriod, the comparison should use TimePeriod logic
        return left_is_tp or right_is_tp

    def _visit_time_period_comparison(self, left_sql: str, right_sql: str, sql_op: str) -> str:
        """
        Generate SQL for TimePeriod comparison.

        Uses vtl_period_* functions to compare based on date boundaries.
        """
        comparison_funcs = {
            "<": "vtl_period_lt",
            "<=": "vtl_period_le",
            ">": "vtl_period_gt",
            ">=": "vtl_period_ge",
            "=": "vtl_period_eq",
            "<>": "vtl_period_ne",
        }

        func = comparison_funcs.get(sql_op)
        if func:
            return f"{func}(vtl_period_parse({left_sql}), vtl_period_parse({right_sql}))"

        # Fallback to standard comparison
        return f"({left_sql} {sql_op} {right_sql})"

    def _is_time_interval_comparison(self, left: AST.AST, right: AST.AST) -> bool:
        """
        Check if this is a comparison between TimeInterval operands.

        Returns True if at least one operand is a TimeInterval component.
        """
        left_is_ti = self._is_time_interval_operand(left)
        right_is_ti = self._is_time_interval_operand(right)

        # If one is TimeInterval, the comparison should use TimeInterval logic
        return left_is_ti or right_is_ti

    def _visit_time_interval_comparison(self, left_sql: str, right_sql: str, sql_op: str) -> str:
        """
        Generate SQL for TimeInterval comparison.

        Uses vtl_interval_* functions to compare based on start dates.
        """
        comparison_funcs = {
            "<": "vtl_interval_lt",
            "<=": "vtl_interval_le",
            ">": "vtl_interval_gt",
            ">=": "vtl_interval_ge",
            "=": "vtl_interval_eq",
            "<>": "vtl_interval_ne",
        }

        func = comparison_funcs.get(sql_op)
        if func:
            return f"{func}(vtl_interval_parse({left_sql}), vtl_interval_parse({right_sql}))"

        # Fallback to standard comparison
        return f"({left_sql} {sql_op} {right_sql})"

    def _visit_period_indicator(self, operand: AST.AST, operand_type: str) -> str:
        """
        Generate SQL for period_indicator (extracts period indicator from TimePeriod).

        Uses vtl_period_indicator for proper extraction from any TimePeriod format.
        Handles formats: YYYY, YYYYA, YYYYQ1, YYYY-Q1, YYYYM01, YYYY-M01, etc.

        Uses structure tracking to get dataset structure.
        """
        if operand_type == OperandType.DATASET:
            ds = self.get_structure(operand)
            if ds is None:
                ds_name = self._get_dataset_name(operand)
                raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

            dataset_sql = self._get_dataset_sql(operand)

            # Find the time identifier
            time_id, _ = self._get_time_and_other_ids(ds)
            id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

            # Extract period indicator using vtl_period_indicator function
            period_extract = f'vtl_period_indicator(vtl_period_parse("{time_id}"))'
            from_clause = self._simplify_from_clause(dataset_sql)
            return f'SELECT {id_select}, {period_extract} AS "duration_var" FROM {from_clause}'

        operand_sql = self.visit(operand)
        return f"vtl_period_indicator(vtl_period_parse({operand_sql}))"

    def _visit_duration_conversion(self, operand: AST.AST, operand_type: str, op: str) -> str:
        """
        Generate SQL for duration conversion operators.

        - daytoyear: days -> 'PxYxD' format
        - daytomonth: days -> 'PxMxD' format
        - yeartoday: 'PxYxD' -> days
        - monthtoday: 'PxMxD' -> days
        """
        operand_sql = self.visit(operand)

        if op == DAYTOYEAR:
            # Convert days to 'PxYxD' format
            # years = days / 365, remaining_days = days % 365
            years_expr = f"CAST(FLOOR({operand_sql} / 365) AS VARCHAR)"
            days_expr = f"CAST({operand_sql} % 365 AS VARCHAR)"
            return f"'P' || {years_expr} || 'Y' || {days_expr} || 'D'"

        elif op == DAYTOMONTH:
            # Convert days to 'PxMxD' format
            # months = days / 30, remaining_days = days % 30
            months_expr = f"CAST(FLOOR({operand_sql} / 30) AS VARCHAR)"
            days_expr = f"CAST({operand_sql} % 30 AS VARCHAR)"
            return f"'P' || {months_expr} || 'M' || {days_expr} || 'D'"

        elif op == YEARTODAY:
            # Convert 'PxYxD' to days
            # Extract years and days, compute total days
            return f"""(
                CAST(REGEXP_EXTRACT({operand_sql}, 'P(\\d+)Y', 1) AS INTEGER) * 365 +
                CAST(REGEXP_EXTRACT({operand_sql}, '(\\d+)D', 1) AS INTEGER)
            )"""

        elif op == MONTHTODAY:
            # Convert 'PxMxD' to days
            # Extract months and days, compute total days
            return f"""(
                CAST(REGEXP_EXTRACT({operand_sql}, 'P(\\d+)M', 1) AS INTEGER) * 30 +
                CAST(REGEXP_EXTRACT({operand_sql}, '(\\d+)D', 1) AS INTEGER)
            )"""

        return operand_sql

    # =========================================================================
    # Parameterized Operations (round, trunc, substr, etc.)
    # =========================================================================

    def visit_ParamOp(self, node: AST.ParamOp) -> str:  # type: ignore[override]
        """Process parameterized operations."""
        op = str(node.op).lower()

        if not node.children:
            return ""

        # Handle CAST operation specially
        if op == CAST:
            return self._visit_cast(node)

        operand = node.children[0]
        operand_sql = self.visit(operand)
        operand_type = self._get_operand_type(operand)
        params = [self.visit(p) for p in node.params]

        # Handle substr specially (variable params)
        if op == SUBSTR:
            return self._visit_substr(operand, operand_sql, operand_type, params)

        # Handle replace specially (two params)
        if op == REPLACE:
            return self._visit_replace(operand, operand_sql, operand_type, params)

        # Handle RANDOM: deterministic pseudo-random using hash
        # VTL: random(seed, index) -> Number between 0 and 1
        if op == RANDOM:
            return self._visit_random(operand, operand_sql, operand_type, params)

        # Single-param operations mapping: op -> (sql_func, default_param, template_format)
        single_param_ops = {
            ROUND: ("ROUND", "0", "{func}({{m}}, {p})"),
            TRUNC: ("TRUNC", "0", "{func}({{m}}, {p})"),
            INSTR: ("INSTR", "''", "{func}({{m}}, {p})"),
            LOG: ("LOG", "10", "{func}({p}, {{m}})"),
            POWER: ("POWER", "2", "{func}({{m}}, {p})"),
            NVL: ("COALESCE", "NULL", "{func}({{m}}, {p})"),
        }

        if op in single_param_ops:
            sql_func, default_p, template_fmt = single_param_ops[op]
            param_val = params[0] if params else default_p
            template = template_fmt.format(func=sql_func, p=param_val)
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, template)
            # For scalar: replace {m} with operand_sql
            return template.replace("{m}", operand_sql)

        # Default function call
        all_params = [operand_sql] + params
        return f"{op.upper()}({', '.join(all_params)})"

    def _visit_substr(
        self, operand: AST.AST, operand_sql: str, operand_type: str, params: List[str]
    ) -> str:
        """Handle SUBSTR operation."""
        start = params[0] if len(params) > 0 else "1"
        length = params[1] if len(params) > 1 else None
        if operand_type == OperandType.DATASET:
            if length:
                return self._param_dataset(operand, f"SUBSTR({{m}}, {start}, {length})")
            return self._param_dataset(operand, f"SUBSTR({{m}}, {start})")
        if length:
            return f"SUBSTR({operand_sql}, {start}, {length})"
        return f"SUBSTR({operand_sql}, {start})"

    def _visit_replace(
        self, operand: AST.AST, operand_sql: str, operand_type: str, params: List[str]
    ) -> str:
        """Handle REPLACE operation."""
        pattern = params[0] if len(params) > 0 else "''"
        replacement = params[1] if len(params) > 1 else "''"
        if operand_type == OperandType.DATASET:
            return self._param_dataset(operand, f"REPLACE({{m}}, {pattern}, {replacement})")
        return f"REPLACE({operand_sql}, {pattern}, {replacement})"

    def _visit_random(
        self, operand: AST.AST, operand_sql: str, operand_type: str, params: List[str]
    ) -> str:
        """
        Handle RANDOM operation.

        VTL: random(seed, index) -> deterministic pseudo-random Number between 0 and 1.

        Uses hash-based approach for determinism: same seed + index = same result.
        DuckDB: (ABS(hash(seed || '_' || index)) % 1000000) / 1000000.0
        """
        index_val = params[0] if params else "0"

        # Template for random: uses seed (operand) and index (param)
        random_template = (
            "(ABS(hash(CAST({m} AS VARCHAR) || '_' || CAST("
            + index_val
            + " AS VARCHAR))) % 1000000) / 1000000.0"
        )

        if operand_type == OperandType.DATASET:
            return self._param_dataset(operand, random_template)

        # Scalar: replace {m} with operand_sql
        return random_template.replace("{m}", operand_sql)

    def _param_dataset(self, dataset_node: AST.AST, template: str) -> str:
        """
        Generate SQL for dataset parameterized operation.

        Uses structure tracking to get dataset structure.
        """
        ds = self.get_structure(dataset_node)
        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        # Quote column names properly in function calls
        measure_parts = []
        for m in ds.get_measures_names():
            quoted_col = f'"{m}"'
            measure_parts.append(f'{template.format(m=quoted_col)} AS "{m}"')
        measure_select = ", ".join(measure_parts)

        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    def _visit_cast(self, node: AST.ParamOp) -> str:
        """
        Handle CAST operations.

        VTL: cast(operand, type) or cast(operand, type, mask)
        SQL: CAST(operand AS type) or special handling for masked casts
        """
        if len(node.children) < 2:
            return ""

        operand = node.children[0]
        operand_sql = self.visit(operand)
        operand_type = self._get_operand_type(operand)

        # Get target type - it's the second child (scalar type)
        target_type_node = node.children[1]
        if hasattr(target_type_node, "value"):
            target_type = target_type_node.value
        elif hasattr(target_type_node, "__name__"):
            target_type = target_type_node.__name__
        else:
            target_type = str(target_type_node)

        # Get optional mask from params
        mask = None
        if node.params:
            mask_val = self.visit(node.params[0])
            # Remove quotes if present
            if mask_val.startswith("'") and mask_val.endswith("'"):
                mask = mask_val[1:-1]
            else:
                mask = mask_val

        # Map VTL type to DuckDB type
        duckdb_type = VTL_TO_DUCKDB_TYPES.get(target_type, "VARCHAR")

        # Dataset-level cast
        if operand_type == OperandType.DATASET:
            return self._cast_dataset(operand, target_type, duckdb_type, mask)

        # Scalar/Component level cast
        return self._cast_scalar(operand_sql, target_type, duckdb_type, mask)

    def _cast_scalar(
        self, operand_sql: str, target_type: str, duckdb_type: str, mask: Optional[str]
    ) -> str:
        """Generate SQL for scalar cast with optional mask."""
        if mask:
            # Handle masked casts
            if target_type == "Date":
                # String to Date with format mask
                return f"STRPTIME({operand_sql}, '{mask}')::DATE"
            elif target_type in ("Number", "Integer"):
                # Number with decimal mask - replace comma with dot
                return f"CAST(REPLACE({operand_sql}, ',', '.') AS {duckdb_type})"
            elif target_type == "String":
                # Date/Number to String with format
                return f"STRFTIME({operand_sql}, '{mask}')"
            elif target_type == "TimePeriod":
                # String to TimePeriod (stored as VARCHAR)
                return f"CAST({operand_sql} AS VARCHAR)"

        # Simple cast without mask
        return f"CAST({operand_sql} AS {duckdb_type})"

    def _cast_dataset(
        self,
        dataset_node: AST.AST,
        target_type: str,
        duckdb_type: str,
        mask: Optional[str],
    ) -> str:
        """
        Generate SQL for dataset-level cast operation.

        Uses structure tracking to get dataset structure.
        """
        ds = self.get_structure(dataset_node)

        if ds is None:
            ds_name = self._get_dataset_name(dataset_node)
            raise ValueError(f"Cannot resolve dataset structure for {ds_name}")

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        # Build measure cast expressions
        measure_parts = []
        for m in ds.get_measures_names():
            cast_expr = self._cast_scalar(f'"{m}"', target_type, duckdb_type, mask)
            measure_parts.append(f'{cast_expr} AS "{m}"')

        measure_select = ", ".join(measure_parts)
        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        return f"SELECT {id_select}, {measure_select} FROM {from_clause}"

    # =========================================================================
    # Multiple-operand Operations
    # =========================================================================

    def visit_MulOp(self, node: AST.MulOp) -> str:  # type: ignore[override]
        """Process multiple-operand operations (between, group by, set ops, etc.)."""
        op = str(node.op).lower()

        # Time operator: current_date (nullary)
        if op == CURRENT_DATE:
            return "CURRENT_DATE"

        if op == BETWEEN and len(node.children) >= 3:
            operand = self.visit(node.children[0])
            low = self.visit(node.children[1])
            high = self.visit(node.children[2])
            return f"({operand} BETWEEN {low} AND {high})"

        # Set operations (union, intersect, setdiff, symdiff)
        if op in SQL_SET_OPS:
            return self._visit_set_op(node, op)

        # exist_in also comes through MulOp
        if op == EXISTS_IN:
            return self._visit_exist_in_mulop(node)

        # For group by/except, return comma-separated list
        children_sql = [self.visit(child) for child in node.children]
        return ", ".join(children_sql)

    def _visit_set_op(self, node: AST.MulOp, op: str) -> str:
        """
        Generate SQL for set operations.

        VTL: union(ds1, ds2), intersect(ds1, ds2), setdiff(ds1, ds2), symdiff(ds1, ds2)
        """
        if len(node.children) < 2:
            if node.children:
                return self._get_dataset_sql(node.children[0])
            return ""

        # Get SQL for all operands
        queries = [self._get_dataset_sql(child) for child in node.children]

        if op == SYMDIFF:
            # Symmetric difference: (A EXCEPT B) UNION ALL (B EXCEPT A)
            return self._symmetric_difference(queries)

        sql_op = SQL_SET_OPS.get(op, op.upper())

        # For union, we need to handle duplicates - VTL union removes duplicates on identifiers
        if op == UNION:
            return self._union_with_dedup(node, queries)

        # For intersect and setdiff, standard SQL operations work
        return f" {sql_op} ".join([f"({q})" for q in queries])

    def _symmetric_difference(self, queries: List[str]) -> str:
        """Generate SQL for symmetric difference: (A EXCEPT B) UNION ALL (B EXCEPT A)."""
        if len(queries) < 2:
            return queries[0] if queries else ""

        a_sql = queries[0]
        b_sql = queries[1]

        # For more than 2 operands, chain the operation
        result = f"""
            (({a_sql}) EXCEPT ({b_sql}))
            UNION ALL
            (({b_sql}) EXCEPT ({a_sql}))
        """

        # Chain additional operands
        for i in range(2, len(queries)):
            result = f"""
                (({result}) EXCEPT ({queries[i]}))
                UNION ALL
                (({queries[i]}) EXCEPT ({result}))
            """

        return result

    def _union_with_dedup(self, node: AST.MulOp, queries: List[str]) -> str:
        """
        Generate SQL for VTL union with duplicate removal on identifiers.

        VTL union keeps the first occurrence when identifiers match.
        """
        if len(queries) < 2:
            return queries[0] if queries else ""

        # Get identifier columns from first dataset using unified structure lookup
        first_child = node.children[0]
        first_ds = self.get_structure(first_child)

        if first_ds:
            id_cols = list(first_ds.get_identifiers_names())
            if id_cols:
                # Use UNION ALL then DISTINCT ON for first occurrence
                union_sql = " UNION ALL ".join([f"({q})" for q in queries])
                id_list = ", ".join([f'"{c}"' for c in id_cols])
                return f"""
                    SELECT DISTINCT ON ({id_list}) *
                    FROM ({union_sql}) AS t
                """

        # Fallback: simple UNION (removes all duplicates)
        return " UNION ".join([f"({q})" for q in queries])

    def _visit_exist_in_mulop(self, node: AST.MulOp) -> str:
        """Handle exist_in when it comes through MulOp."""
        if len(node.children) < 2:
            raise ValueError("exist_in requires at least two operands")

        left_node = node.children[0]
        right_node = node.children[1]

        left_name = self._get_dataset_name(left_node)
        right_name = self._get_dataset_name(right_node)

        # Use get_structure() for unified structure lookup
        # (handles VarID, Aggregation, RegularAggregation, UDOCall, etc.)
        left_ds = self.get_structure(left_node)
        right_ds = self.get_structure(right_node)

        if not left_ds or not right_ds:
            raise ValueError(f"Cannot resolve dataset structures for {left_name} and {right_name}")

        # Find common identifiers
        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        common_ids = sorted(left_ids.intersection(right_ids))

        if not common_ids:
            raise ValueError(f"No common identifiers between {left_name} and {right_name}")

        # Build EXISTS condition
        conditions = [f'l."{id}" = r."{id}"' for id in common_ids]
        where_clause = " AND ".join(conditions)

        # Select identifiers from left (using transformed structure)
        id_select = ", ".join([f'l."{k}"' for k in left_ds.get_identifiers_names()])

        left_sql = self._get_dataset_sql(left_node)
        right_sql = self._get_dataset_sql(right_node)

        # Check for retain parameter (third child)
        # retain=true: keep rows where identifiers exist
        # retain=false: keep rows where identifiers don't exist
        # retain=None: return all rows with bool_var column
        retain_filter = ""
        if len(node.children) > 2:
            retain_node = node.children[2]
            if isinstance(retain_node, AST.Constant):
                retain_value = retain_node.value
                if isinstance(retain_value, bool):
                    retain_filter = f" WHERE bool_var = {str(retain_value).upper()}"
                elif isinstance(retain_value, str) and retain_value.lower() in ("true", "false"):
                    retain_filter = f" WHERE bool_var = {retain_value.upper()}"

        base_query = f"""
            SELECT {id_select},
                   EXISTS(SELECT 1 FROM ({right_sql}) AS r WHERE {where_clause}) AS "bool_var"
            FROM ({left_sql}) AS l
        """

        if retain_filter:
            return f"SELECT * FROM ({base_query}){retain_filter}"
        return base_query

    # =========================================================================
    # Conditional Operations
    # =========================================================================

    def visit_If(self, node: AST.If) -> str:
        """Process if-then-else."""
        condition = self.visit(node.condition)
        then_op = self.visit(node.thenOp)
        else_op = self.visit(node.elseOp)

        return f"CASE WHEN {condition} THEN {then_op} ELSE {else_op} END"

    def visit_Case(self, node: AST.Case) -> str:
        """Process case expression."""
        cases = []
        for case_obj in node.cases:
            cond = self.visit(case_obj.condition)
            then = self.visit(case_obj.thenOp)
            cases.append(f"WHEN {cond} THEN {then}")

        else_op = self.visit(node.elseOp)
        cases_sql = " ".join(cases)

        return f"CASE {cases_sql} ELSE {else_op} END"

    def visit_CaseObj(self, node: AST.CaseObj) -> str:
        """Process a single case object."""
        cond = self.visit(node.condition)
        then = self.visit(node.thenOp)
        return f"WHEN {cond} THEN {then}"

    # =========================================================================
    # Clause Operations (calc, filter, keep, drop, rename)
    # =========================================================================

    def _get_transformed_dataset(self, base_dataset: Dataset, clause_node: AST.AST) -> Dataset:
        """
        Compute a transformed dataset structure after applying nested clause operations.

        This handles chained clauses like [rename Me_1 to Me_1A][drop Me_2] by tracking
        how each clause modifies the dataset structure.
        """
        # Handle UDOCall nodes
        if isinstance(clause_node, AST.UDOCall):
            return self._get_udo_output_structure(clause_node, base_dataset)

        # Handle Aggregation nodes (like max, min, etc.)
        if isinstance(clause_node, AST.Aggregation):
            return self._get_aggregation_output_structure(clause_node, base_dataset)

        if not isinstance(clause_node, AST.RegularAggregation):
            return base_dataset

        # Start with the base dataset or recursively get transformed dataset
        if clause_node.dataset:
            current_ds = self._get_transformed_dataset(base_dataset, clause_node.dataset)
        else:
            current_ds = base_dataset

        op = str(clause_node.op).lower()

        # Apply transformation based on clause type
        if op == RENAME:
            # Build rename mapping and apply to components
            new_components: Dict[str, Component] = {}
            renames: Dict[str, str] = {}
            for child in clause_node.children:
                if isinstance(child, AST.RenameNode):
                    renames[child.old_name] = child.new_name

            for name, comp in current_ds.components.items():
                if name in renames:
                    new_name = renames[name]
                    # Create new component with renamed name
                    new_comp = Component(
                        name=new_name,
                        data_type=comp.data_type,
                        role=comp.role,
                        nullable=comp.nullable,
                    )
                    new_components[new_name] = new_comp
                else:
                    new_components[name] = comp

            return Dataset(name=current_ds.name, components=new_components, data=None)

        elif op == DROP:
            # Remove dropped columns
            drop_cols = set()
            for child in clause_node.children:
                if isinstance(child, (AST.VarID, AST.Identifier)):
                    drop_cols.add(child.value)

            new_components = {
                name: comp for name, comp in current_ds.components.items() if name not in drop_cols
            }
            return Dataset(name=current_ds.name, components=new_components, data=None)

        elif op == KEEP:
            # Keep only identifiers and specified columns
            keep_cols = set(current_ds.get_identifiers_names())
            for child in clause_node.children:
                if isinstance(child, (AST.VarID, AST.Identifier)):
                    keep_cols.add(child.value)

            new_components = {
                name: comp for name, comp in current_ds.components.items() if name in keep_cols
            }
            return Dataset(name=current_ds.name, components=new_components, data=None)

        elif op == SUBSPACE:
            # Subspace removes the identifiers used for filtering
            remove_cols = set()
            for child in clause_node.children:
                if isinstance(child, AST.BinOp):
                    col_name = child.left.value if hasattr(child.left, "value") else str(child.left)
                    remove_cols.add(col_name)

            new_components = {
                name: comp
                for name, comp in current_ds.components.items()
                if name not in remove_cols
            }
            return Dataset(name=current_ds.name, components=new_components, data=None)

        elif op == CALC:
            # Calc can add new measures or overwrite existing ones
            from vtlengine.DataTypes import String
            from vtlengine.Model import Role

            new_components = dict(current_ds.components)
            for child in clause_node.children:
                # Calc children are wrapped in UnaryOp with role (measure, identifier, attribute)
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
                        role = Role.MEASURE  # Default to measure
                elif isinstance(child, AST.Assignment):
                    assignment = child
                    role = Role.MEASURE  # Default to measure
                else:
                    continue

                if isinstance(assignment, AST.Assignment):
                    if not isinstance(assignment.left, (AST.VarID, AST.Identifier)):
                        continue
                    col_name = assignment.left.value
                    if col_name not in new_components:
                        # Add new component (assume String type for simplicity)
                        # Identifiers cannot be nullable
                        is_nullable = role != Role.IDENTIFIER
                        new_components[col_name] = Component(
                            name=col_name,
                            data_type=String,
                            role=role,
                            nullable=is_nullable,
                        )

            return Dataset(name=current_ds.name, components=new_components, data=None)

        # For other clauses (filter, etc.), return as-is for now
        # These don't change the column structure in ways that affect subsequent clauses
        return current_ds

    def _get_aggregation_output_structure(
        self, agg_node: AST.Aggregation, base_dataset: Dataset
    ) -> Dataset:
        """
        Compute the output structure after an aggregation operation.

        Handles:
        - group by: only specified identifiers remain
        - group except: all identifiers except specified ones remain
        """
        if not agg_node.grouping:
            # No grouping - all identifiers are removed, only aggregated measures remain
            new_components = {
                name: comp
                for name, comp in base_dataset.components.items()
                if comp.role != Role.IDENTIFIER
            }
            return Dataset(name=base_dataset.name, components=new_components, data=None)

        # Get identifiers to keep based on grouping operation
        # Use _resolve_varid_value to handle UDO parameters
        if agg_node.grouping_op == "group by":
            # Only keep specified identifiers
            keep_ids = {
                self._resolve_varid_value(g)
                if isinstance(g, (AST.VarID, AST.Identifier))
                else str(g)
                for g in agg_node.grouping
            }
        elif agg_node.grouping_op == "group except":
            # Keep all identifiers except specified ones
            except_ids = {
                self._resolve_varid_value(g)
                if isinstance(g, (AST.VarID, AST.Identifier))
                else str(g)
                for g in agg_node.grouping
            }
            keep_ids = {
                name
                for name, comp in base_dataset.components.items()
                if comp.role == Role.IDENTIFIER and name not in except_ids
            }
        else:
            keep_ids = set(base_dataset.get_identifiers_names())

        # Build new components: keep specified identifiers + all measures
        new_components = {}
        for name, comp in base_dataset.components.items():
            if comp.role == Role.IDENTIFIER:
                if name in keep_ids:
                    new_components[name] = comp
            else:
                # Keep all measures (and attributes)
                new_components[name] = comp

        return Dataset(name=base_dataset.name, components=new_components, data=None)

    def _get_join_output_structure(self, join_node: AST.JoinOp) -> Optional[Dataset]:
        """
        Compute the output structure after a join operation.

        A join result contains:
        - All identifiers from all datasets (union)
        - All measures from all datasets (union)
        """

        def get_clause_structure(clause: AST.AST) -> Optional[Dataset]:
            """Get the transformed structure for a join clause."""
            # Use unified get_structure() which handles all node types
            # including alias (as), RegularAggregation, UDOCall, Aggregation
            return self.get_structure(clause)

        # Collect components from all clauses
        all_components: Dict[str, Component] = {}
        result_name = "join_result"

        for clause in join_node.clauses:
            clause_ds = get_clause_structure(clause)
            if clause_ds:
                result_name = clause_ds.name  # Use last dataset name
                for comp_name, comp in clause_ds.components.items():
                    if comp_name not in all_components:
                        all_components[comp_name] = deepcopy(comp)

        if not all_components:
            return None

        return Dataset(name=result_name, components=all_components, data=None)

    def _get_udo_output_structure(self, udo_node: AST.UDOCall, base_dataset: Dataset) -> Dataset:
        """
        Compute the output structure after a UDO call.

        Expands the UDO and computes the structure based on its expression.
        """
        if udo_node.op not in self.udos:
            return base_dataset

        operator = self.udos[udo_node.op]
        expression = operator["expression"]

        # Build parameter bindings and push to stack
        param_bindings: Dict[str, Any] = {}
        for i, param in enumerate(operator["params"]):
            if i < len(udo_node.params):
                param_bindings[param["name"]] = udo_node.params[i]

        # Push bindings so _resolve_varid_value and get_udo_param work
        if self.udo_params is None:
            self.udo_params = []
        self.udo_params.append(param_bindings)

        try:
            # Analyze the expression to determine output structure
            # Use a copy to avoid modifying the original
            expression_copy = deepcopy(expression)

            if isinstance(expression_copy, AST.Aggregation):
                result = self._get_aggregation_output_structure(expression_copy, base_dataset)
            elif isinstance(expression_copy, AST.RegularAggregation):
                result = self._get_transformed_dataset(base_dataset, expression_copy)
            else:
                # Fallback to base dataset
                result = base_dataset
        finally:
            # Pop bindings
            self.udo_params.pop()
            if len(self.udo_params) == 0:
                self.udo_params = None

        return result

    def visit_RegularAggregation(  # type: ignore[override]
        self, node: AST.RegularAggregation
    ) -> str:
        """
        Process clause operations (calc, filter, keep, drop, rename, etc.).

        These operate on a single dataset and modify its structure or data.
        """
        op = str(node.op).lower()

        # Get dataset name first
        ds_name = self._get_dataset_name(node.dataset) if node.dataset else None

        if ds_name and ds_name in self.available_tables and node.dataset:
            # Get base SQL using _get_dataset_sql (returns SELECT * FROM "table")
            base_sql = self._get_dataset_sql(node.dataset)

            # Store context for component resolution
            prev_dataset = self.current_dataset
            prev_in_clause = self.in_clause

            # Get the transformed dataset structure using unified get_structure()
            base_dataset = self.available_tables[ds_name]
            dataset_structure = self.get_structure(node.dataset)
            self.current_dataset = dataset_structure if dataset_structure else base_dataset
            self.in_clause = True

            try:
                if op == CALC:
                    result = self._clause_calc(base_sql, node.children)
                elif op == FILTER:
                    result = self._clause_filter(base_sql, node.children)
                elif op == KEEP:
                    result = self._clause_keep(base_sql, node.children)
                elif op == DROP:
                    result = self._clause_drop(base_sql, node.children)
                elif op == RENAME:
                    result = self._clause_rename(base_sql, node.children)
                elif op == AGGREGATE:
                    result = self._clause_aggregate(base_sql, node.children)
                elif op == UNPIVOT:
                    result = self._clause_unpivot(base_sql, node.children)
                elif op == PIVOT:
                    result = self._clause_pivot(base_sql, node.children)
                elif op == SUBSPACE:
                    result = self._clause_subspace(base_sql, node.children)
                else:
                    result = base_sql
            finally:
                self.current_dataset = prev_dataset
                self.in_clause = prev_in_clause

            return result

        # Fallback: visit the dataset node directly
        return self._get_dataset_sql(node.dataset) if node.dataset else ""

    def _clause_calc(self, base_sql: str, children: List[AST.AST]) -> str:
        """
        Generate SQL for calc clause.

        Calc can:
        - Create new columns: calc new_col := expr
        - Overwrite existing columns: calc existing_col := expr

        AST structure: children are UnaryOp nodes with op='measure'/'identifier'/'attribute'
        wrapping Assignment nodes.
        """
        if not self.current_dataset:
            return base_sql

        # Build mapping of calculated columns
        calc_cols: Dict[str, str] = {}
        for child in children:
            # Calc children are wrapped in UnaryOp with role (measure, identifier, attribute)
            if isinstance(child, AST.UnaryOp) and hasattr(child, "operand"):
                assignment = child.operand
            elif isinstance(child, AST.Assignment):
                assignment = child
            else:
                continue

            if isinstance(assignment, AST.Assignment):
                # Left is Identifier (column name), right is expression
                if not isinstance(assignment.left, (AST.VarID, AST.Identifier)):
                    continue
                col_name = assignment.left.value
                expr = self.visit(assignment.right)
                calc_cols[col_name] = expr

        # Build SELECT columns
        select_parts = []

        # First, include all existing columns (possibly overwritten)
        for col_name in self.current_dataset.components:
            if col_name in calc_cols:
                # Column is being overwritten
                select_parts.append(f'{calc_cols[col_name]} AS "{col_name}"')
            else:
                # Keep original column
                select_parts.append(f'"{col_name}"')

        # Then, add new columns (not in original dataset)
        for col_name, expr in calc_cols.items():
            if col_name not in self.current_dataset.components:
                select_parts.append(f'{expr} AS "{col_name}"')

        select_cols = ", ".join(select_parts)

        return f"""
                    SELECT {select_cols}
                    FROM ({base_sql}) AS t
                """

    def _clause_filter(self, base_sql: str, children: List[AST.AST]) -> str:
        """
        Generate SQL for filter clause with predicate pushdown.

        Optimization: If base_sql is a simple SELECT * FROM "table",
        we push the WHERE directly onto that query instead of nesting.
        """
        conditions = [self.visit(child) for child in children]
        where_clause = " AND ".join(conditions)

        # Try to push predicate down
        return self._optimize_filter_pushdown(base_sql, where_clause)

    def _clause_keep(self, base_sql: str, children: List[AST.AST]) -> str:
        """Generate SQL for keep clause (select specific components)."""
        if not self.current_dataset:
            return base_sql

        # Always use current_dataset's identifiers - keep operates on the dataset
        # currently being processed, not the final output result
        id_cols = [f'"{c}"' for c in self.current_dataset.get_identifiers_names()]

        # Add specified columns
        keep_cols = []
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                keep_cols.append(f'"{child.value}"')

        select_cols = ", ".join(id_cols + keep_cols)

        return f"SELECT {select_cols} FROM ({base_sql}) AS t"

    def _clause_drop(self, base_sql: str, children: List[AST.AST]) -> str:
        """Generate SQL for drop clause (remove specific components)."""
        if not self.current_dataset:
            return base_sql

        # Get columns to drop
        drop_cols = set()
        for child in children:
            if isinstance(child, (AST.VarID, AST.Identifier)):
                drop_cols.add(child.value)

        # Keep all columns except dropped ones (identifiers cannot be dropped)
        keep_cols = []
        for name in self.current_dataset.components:
            if name not in drop_cols:
                keep_cols.append(f'"{name}"')

        select_cols = ", ".join(keep_cols)

        return f"SELECT {select_cols} FROM ({base_sql}) AS t"

    def _clause_rename(self, base_sql: str, children: List[AST.AST]) -> str:
        """Generate SQL for rename clause."""
        if not self.current_dataset:
            return base_sql

        # Build rename mapping
        renames: Dict[str, str] = {}
        for child in children:
            if isinstance(child, AST.RenameNode):
                renames[child.old_name] = child.new_name

        # Generate select with renames
        select_cols = []
        for name in self.current_dataset.components:
            if name in renames:
                select_cols.append(f'"{name}" AS "{renames[name]}"')
            else:
                select_cols.append(f'"{name}"')

        select_str = ", ".join(select_cols)

        return f"SELECT {select_str} FROM ({base_sql}) AS t"

    def _extract_grouping_from_aggregation(
        self,
        agg_node: AST.Aggregation,
        group_by_cols: List[str],
        group_op: Optional[str],
        having_clause: str,
    ) -> Tuple[List[str], Optional[str], str]:
        """Extract grouping and having info from an Aggregation node."""
        # Extract grouping if present
        if hasattr(agg_node, "grouping_op") and agg_node.grouping_op:
            group_op = agg_node.grouping_op.lower()
        if hasattr(agg_node, "grouping") and agg_node.grouping:
            for g in agg_node.grouping:
                if isinstance(g, (AST.VarID, AST.Identifier)) and g.value not in group_by_cols:
                    group_by_cols.append(g.value)

        # Extract having clause if present
        if hasattr(agg_node, "having_clause") and agg_node.having_clause and not having_clause:
            if isinstance(agg_node.having_clause, AST.ParamOp):
                # Having is wrapped in ParamOp with params containing the condition
                if hasattr(agg_node.having_clause, "params") and agg_node.having_clause.params:
                    having_clause = self.visit(agg_node.having_clause.params)
            else:
                having_clause = self.visit(agg_node.having_clause)

        return group_by_cols, group_op, having_clause

    def _process_aggregate_child(
        self,
        child: AST.AST,
        agg_exprs: List[str],
        group_by_cols: List[str],
        group_op: Optional[str],
        having_clause: str,
    ) -> Tuple[List[str], List[str], Optional[str], str]:
        """Process a single child node in aggregate clause."""
        if isinstance(child, AST.Assignment):
            # Aggregation assignment: Me_sum := sum(Me_1)
            if not isinstance(child.left, (AST.VarID, AST.Identifier)):
                return agg_exprs, group_by_cols, group_op, having_clause
            col_name = child.left.value
            expr = self.visit(child.right)
            agg_exprs.append(f'{expr} AS "{col_name}"')

            # Check if the right side is an Aggregation with grouping info
            if isinstance(child.right, AST.Aggregation):
                group_by_cols, group_op, having_clause = self._extract_grouping_from_aggregation(
                    child.right, group_by_cols, group_op, having_clause
                )

        elif isinstance(child, AST.MulOp):
            # Group by/except clause (legacy format)
            group_op = str(child.op).lower()
            for g in child.children:
                if isinstance(g, AST.VarID):
                    group_by_cols.append(g.value)
                else:
                    group_by_cols.append(self.visit(g))
        elif isinstance(child, AST.BinOp):
            # Having clause condition (legacy format)
            having_clause = self.visit(child)
        elif isinstance(child, AST.UnaryOp) and hasattr(child, "operand"):
            # Wrapped assignment (with role like measure/identifier)
            assignment = child.operand
            if isinstance(assignment, AST.Assignment):
                if not isinstance(assignment.left, (AST.VarID, AST.Identifier)):
                    return agg_exprs, group_by_cols, group_op, having_clause
                col_name = assignment.left.value
                expr = self.visit(assignment.right)
                agg_exprs.append(f'{expr} AS "{col_name}"')

                # Check for grouping info on wrapped aggregations
                if isinstance(assignment.right, AST.Aggregation):
                    group_by_cols, group_op, having_clause = (
                        self._extract_grouping_from_aggregation(
                            assignment.right, group_by_cols, group_op, having_clause
                        )
                    )

        return agg_exprs, group_by_cols, group_op, having_clause

    def _build_aggregate_group_by_sql(
        self, group_by_cols: List[str], group_op: Optional[str]
    ) -> str:
        """Build the GROUP BY SQL clause."""
        if not group_by_cols or not self.current_dataset:
            return ""

        if group_op == "group by":
            quoted_cols = [f'"{c}"' for c in group_by_cols]
            return f"GROUP BY {', '.join(quoted_cols)}"
        elif group_op == "group except":
            # Group by all identifiers except the specified ones
            except_set = set(group_by_cols)
            actual_group_cols = [
                c for c in self.current_dataset.get_identifiers_names() if c not in except_set
            ]
            if actual_group_cols:
                quoted_cols = [f'"{c}"' for c in actual_group_cols]
                return f"GROUP BY {', '.join(quoted_cols)}"
        return ""

    def _build_aggregate_select_parts(
        self, group_by_cols: List[str], group_op: Optional[str], agg_exprs: List[str]
    ) -> List[str]:
        """Build SELECT parts for aggregate clause."""
        select_parts: List[str] = []
        if group_by_cols and group_op == "group by":
            select_parts.extend([f'"{c}"' for c in group_by_cols])
        elif group_op == "group except" and self.current_dataset:
            except_set = set(group_by_cols)
            select_parts.extend(
                [
                    f'"{c}"'
                    for c in self.current_dataset.get_identifiers_names()
                    if c not in except_set
                ]
            )
        select_parts.extend(agg_exprs)
        return select_parts

    def _clause_aggregate(self, base_sql: str, children: List[AST.AST]) -> str:
        """
        Generate SQL for aggregate clause.

        VTL: DS_1[aggr Me_sum := sum(Me_1), Me_max := max(Me_1) group by Id_1 having avg(Me_1) > 10]

        Children may include:
        - Assignment nodes for aggregation expressions (Me_sum := sum(Me_1))
        - MulOp nodes for grouping (group by, group except) - legacy format
        - BinOp nodes for having clause - legacy format

        Note: In the current AST, group by and having info is stored on the Aggregation nodes
        inside the Assignment nodes, not as separate children.
        """
        if not self.current_dataset:
            return base_sql

        agg_exprs: List[str] = []
        group_by_cols: List[str] = []
        having_clause = ""
        group_op: Optional[str] = None

        for child in children:
            agg_exprs, group_by_cols, group_op, having_clause = self._process_aggregate_child(
                child, agg_exprs, group_by_cols, group_op, having_clause
            )

        if not agg_exprs:
            return base_sql

        group_by_sql = self._build_aggregate_group_by_sql(group_by_cols, group_op)
        having_sql = f"HAVING {having_clause}" if having_clause else ""
        select_parts = self._build_aggregate_select_parts(group_by_cols, group_op, agg_exprs)
        select_sql = ", ".join(select_parts)

        return f"""
            SELECT {select_sql}
            FROM ({base_sql}) AS t
            {group_by_sql}
            {having_sql}
        """

    def _clause_unpivot(self, base_sql: str, children: List[AST.AST]) -> str:
        """
        Generate SQL for unpivot clause.

        VTL: DS_r := DS_1 [unpivot Id_3, Me_3];
        - Id_3 is the new identifier column (contains original measure names)
        - Me_3 is the new measure column (contains the values)

        DuckDB: UNPIVOT (subquery) ON col1, col2, ... INTO NAME id_col VALUE measure_col
        """
        if not self.current_dataset or len(children) < 2:
            return base_sql

        # Get the new column names from children
        # children[0] = new identifier column name (will hold measure names)
        # children[1] = new measure column name (will hold values)
        id_col_name = children[0].value if hasattr(children[0], "value") else str(children[0])
        measure_col_name = children[1].value if hasattr(children[1], "value") else str(children[1])

        # Get original measure columns (to unpivot)
        measure_cols = list(self.current_dataset.get_measures_names())

        if not measure_cols:
            return base_sql

        # Build list of columns to unpivot (the original measures)
        unpivot_cols = ", ".join([f'"{m}"' for m in measure_cols])

        # DuckDB UNPIVOT syntax
        return f"""
            SELECT * FROM (
                UNPIVOT ({base_sql})
                ON {unpivot_cols}
                INTO NAME "{id_col_name}" VALUE "{measure_col_name}"
            )
        """

    def _clause_pivot(self, base_sql: str, children: List[AST.AST]) -> str:
        """
        Generate SQL for pivot clause.

        VTL: DS_r := DS_1 [pivot Id_2, Me_1];
        - Id_2 is the identifier column whose values become new columns
        - Me_1 is the measure whose values fill those columns

        DuckDB: PIVOT (subquery) ON id_col USING FIRST(measure_col)
        """
        if not self.current_dataset or len(children) < 2:
            return base_sql

        # Get the column names from children
        # children[0] = identifier column to pivot on (values become columns)
        # children[1] = measure column to aggregate
        pivot_id = children[0].value if hasattr(children[0], "value") else str(children[0])
        pivot_measure = children[1].value if hasattr(children[1], "value") else str(children[1])

        # Get remaining identifier columns (those that stay as identifiers)
        id_cols = [c for c in self.current_dataset.get_identifiers_names() if c != pivot_id]

        if not id_cols:
            # If no remaining identifiers, use just the pivot
            return f"""
                SELECT * FROM (
                    PIVOT ({base_sql})
                    ON "{pivot_id}"
                    USING FIRST("{pivot_measure}")
                )
            """
        else:
            # Group by remaining identifiers
            group_cols = ", ".join([f'"{c}"' for c in id_cols])
            return f"""
                SELECT * FROM (
                    PIVOT ({base_sql})
                    ON "{pivot_id}"
                    USING FIRST("{pivot_measure}")
                    GROUP BY {group_cols}
                )
            """

    def _clause_subspace(self, base_sql: str, children: List[AST.AST]) -> str:
        """
        Generate SQL for subspace clause.

        VTL: DS_r := DS_1 [sub Id_1 = "A"];
        Filters the dataset to rows where the specified identifier equals the value,
        then removes that identifier from the result.

        Children are BinOp nodes with: left = column, op = "=", right = value
        """
        if not self.current_dataset or not children:
            return base_sql

        conditions = []
        remove_cols = []

        for child in children:
            if isinstance(child, AST.BinOp):
                col_name = child.left.value if hasattr(child.left, "value") else str(child.left)
                col_value = self.visit(child.right)

                # Check column type - if string, cast numeric constants to string
                comp = self.current_dataset.components.get(col_name)
                if comp:
                    from vtlengine.DataTypes import String

                    if (
                        comp.data_type == String
                        and isinstance(child.right, AST.Constant)
                        and child.right.type_ in ("INTEGER_CONSTANT", "FLOAT_CONSTANT")
                    ):
                        # Cast numeric constant to string for string column comparison
                        col_value = f"'{child.right.value}'"

                conditions.append(f'"{col_name}" = {col_value}')
                remove_cols.append(col_name)

        if not conditions:
            return base_sql

        # First filter by conditions
        where_clause = " AND ".join(conditions)

        # Then select all columns except the subspace identifiers
        keep_cols = [f'"{c}"' for c in self.current_dataset.components if c not in remove_cols]

        if not keep_cols:
            # If all columns would be removed, return just the filter
            return f"SELECT * FROM ({base_sql}) AS t WHERE {where_clause}"

        select_cols = ", ".join(keep_cols)

        return f"SELECT {select_cols} FROM ({base_sql}) AS t WHERE {where_clause}"

    # =========================================================================
    # Aggregation Operations
    # =========================================================================

    def visit_Aggregation(self, node: AST.Aggregation) -> str:  # type: ignore[override]
        """Process aggregation operations (sum, avg, count, etc.)."""
        op = str(node.op).lower()
        sql_op = SQL_AGGREGATE_OPS.get(op, op.upper())

        # Get operand
        if node.operand:
            operand_sql = self.visit(node.operand)
            operand_type = self._get_operand_type(node.operand)
        else:
            operand_sql = "*"
            operand_type = OperandType.SCALAR

        # Handle grouping
        group_by = ""
        if node.grouping:
            group_cols = [self.visit(g) for g in node.grouping]
            if node.grouping_op == "group by":
                group_by = f"GROUP BY {', '.join(group_cols)}"
            elif (
                node.grouping_op == "group except"
                and operand_type == OperandType.DATASET
                and node.operand
            ):
                # Group by all except specified
                ds_name = self._get_dataset_name(node.operand)
                ds = self.available_tables.get(ds_name)
                if ds:
                    # Resolve UDO parameters to get actual column names
                    except_cols = {
                        self._resolve_varid_value(g)
                        for g in node.grouping
                        if isinstance(g, (AST.VarID, AST.Identifier))
                    }
                    group_cols = [
                        f'"{c}"' for c in ds.get_identifiers_names() if c not in except_cols
                    ]
                    group_by = f"GROUP BY {', '.join(group_cols)}"

        # Handle having
        having = ""
        if node.having_clause:
            having_sql = self.visit(node.having_clause)
            having = f"HAVING {having_sql}"

        # Dataset-level aggregation
        if operand_type == OperandType.DATASET and node.operand:
            ds_name = self._get_dataset_name(node.operand)
            # Try available_tables first, then fall back to get_structure for complex operands
            ds = self.available_tables.get(ds_name) or self.get_structure(node.operand)
            if ds:
                measures = list(ds.get_measures_names())
                dataset_sql = self._get_dataset_sql(node.operand)

                # Build measure select based on operation and available measures
                if measures:
                    measure_select = ", ".join([f'{sql_op}("{m}") AS "{m}"' for m in measures])
                elif op == COUNT:
                    # COUNT on identifier-only dataset produces int_var
                    measure_select = 'COUNT(*) AS "int_var"'
                else:
                    measure_select = ""

                # Only include identifiers if grouping is specified
                if group_by and node.grouping:
                    # Use only the columns specified in GROUP BY, not all identifiers
                    if node.grouping_op == "group by":
                        # Extract column names from grouping nodes
                        group_col_names = [
                            g.value if isinstance(g, (AST.VarID, AST.Identifier)) else str(g)
                            for g in node.grouping
                        ]
                        id_select = ", ".join([f'"{k}"' for k in group_col_names])
                    else:
                        # For "group except", use all identifiers except the excluded ones
                        # Resolve UDO parameters to get actual column names
                        except_cols = {
                            self._resolve_varid_value(g)
                            for g in node.grouping
                            if isinstance(g, (AST.VarID, AST.Identifier))
                        }
                        id_select = ", ".join(
                            [f'"{k}"' for k in ds.get_identifiers_names() if k not in except_cols]
                        )

                    # Handle case where there are no measures (identifier-only datasets)
                    if measure_select:
                        select_clause = f"{id_select}, {measure_select}"
                    else:
                        select_clause = id_select

                    return f"""
                        SELECT {select_clause}
                        FROM ({dataset_sql}) AS t
                        {group_by}
                        {having}
                    """.strip()
                else:
                    # No grouping: aggregate all rows into single result
                    if not measure_select:
                        # No measures to aggregate - return empty set or single row
                        return f"SELECT 1 AS _placeholder FROM ({dataset_sql}) AS t LIMIT 1"
                    return f"""
                        SELECT {measure_select}
                        FROM ({dataset_sql}) AS t
                        {having}
                    """.strip()

        # Scalar/Component aggregation
        return f"{sql_op}({operand_sql})"

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> str:  # type: ignore[override]
        """
        Process TIME_AGG operation.

        VTL: time_agg(period_to, operand) or time_agg(period_to, operand, conf)

        Converts Date to TimePeriod string at specified granularity.
        Note: TimePeriod inputs are not supported - raises NotImplementedError.

        DuckDB SQL mappings:
        - "Y" -> STRFTIME(col, '%Y')
        - "S" -> STRFTIME(col, '%Y') || 'S' || CEIL(MONTH(col) / 6.0)
        - "Q" -> STRFTIME(col, '%Y') || 'Q' || QUARTER(col)
        - "M" -> STRFTIME(col, '%Y') || 'M' || LPAD(CAST(MONTH(col) AS VARCHAR), 2, '0')
        - "D" -> STRFTIME(col, '%Y-%m-%d')
        """
        period_to = node.period_to.upper() if node.period_to else "Y"

        # Build SQL expression template for each period type
        # VTL period codes: A=Annual, S=Semester, Q=Quarter, M=Month, W=Week, D=Day
        # Use CAST to DATE to handle dates read as VARCHAR from CSV
        dc = "CAST({col} AS DATE)"  # date cast placeholder
        yf = "STRFTIME(" + dc + ", '%Y')"  # year format
        period_templates = {
            "A": "STRFTIME(" + dc + ", '%Y')",
            "S": "(" + yf + " || 'S' || CAST(CEIL(MONTH(" + dc + ") / 6.0) AS INTEGER))",
            "Q": "(" + yf + " || 'Q' || CAST(QUARTER(" + dc + ") AS VARCHAR))",
            "M": "(" + yf + " || 'M' || LPAD(CAST(MONTH(" + dc + ") AS VARCHAR), 2, '0'))",
            "W": "(" + yf + " || 'W' || LPAD(CAST(WEEKOFYEAR(" + dc + ") AS VARCHAR), 2, '0'))",
            "D": "STRFTIME(" + dc + ", '%Y-%m-%d')",
        }

        template = period_templates.get(period_to, "STRFTIME(CAST({col} AS DATE), '%Y')")

        if node.operand is None:
            raise ValueError("TIME_AGG requires an operand")

        operand_type = self._get_operand_type(node.operand)

        if operand_type == OperandType.DATASET:
            return self._time_agg_dataset(node.operand, template, period_to)

        # Scalar/Component: just apply the template
        operand_sql = self.visit(node.operand)
        return template.format(col=operand_sql)

    def _time_agg_dataset(self, dataset_node: AST.AST, template: str, period_to: str) -> str:
        """
        Generate SQL for dataset-level TIME_AGG operation.

        Applies time aggregation to time-type measures.
        """
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables.get(ds_name)

        if not ds:
            operand_sql = self.visit(dataset_node)
            return template.format(col=operand_sql)

        # Build SELECT with identifiers and transformed time measures
        id_cols = ds.get_identifiers_names()
        id_select = ", ".join([f'"{k}"' for k in id_cols])

        # Find time-type measures (Date, TimePeriod, TimeInterval)
        time_types = {"Date", "TimePeriod", "TimeInterval"}
        measure_parts = []

        for m_name in ds.get_measures_names():
            comp = ds.components.get(m_name)
            if comp and comp.data_type.__name__ in time_types:
                # TimePeriod: use vtl_time_agg for proper period aggregation
                if comp.data_type.__name__ == "TimePeriod":
                    # Parse VARCHAR → STRUCT, aggregate to target, format back → VARCHAR
                    col_expr = (
                        f"vtl_period_to_string(vtl_time_agg("
                        f"vtl_period_parse(\"{m_name}\"), '{period_to}'))"
                    )
                    measure_parts.append(f'{col_expr} AS "{m_name}"')
                else:
                    # Date/TimeInterval: use template-based conversion
                    col_expr = template.format(col=f'"{m_name}"')
                    measure_parts.append(f'{col_expr} AS "{m_name}"')
            else:
                # Non-time measures pass through unchanged
                measure_parts.append(f'"{m_name}"')

        measure_select = ", ".join(measure_parts)
        dataset_sql = self._get_dataset_sql(dataset_node)
        from_clause = self._simplify_from_clause(dataset_sql)

        if id_select and measure_select:
            return f"SELECT {id_select}, {measure_select} FROM {from_clause}"
        elif measure_select:
            return f"SELECT {measure_select} FROM {from_clause}"
        else:
            return f"SELECT * FROM {from_clause}"

    # =========================================================================
    # Analytic Operations (window functions)
    # =========================================================================

    def visit_Analytic(self, node: AST.Analytic) -> str:  # type: ignore[override]
        """Process analytic (window) functions."""
        op = str(node.op).lower()
        sql_op = SQL_ANALYTIC_OPS.get(op, op.upper())

        # Operand
        operand = self.visit(node.operand) if node.operand else ""

        # Partition by
        partition = ""
        if node.partition_by:
            cols = [f'"{c}"' for c in node.partition_by]
            partition = f"PARTITION BY {', '.join(cols)}"

        # Order by
        order = ""
        if node.order_by:
            order_parts = []
            for ob in node.order_by:
                order_parts.append(f'"{ob.component}" {ob.order.upper()}')
            order = f"ORDER BY {', '.join(order_parts)}"

        # Window frame
        window = ""
        if node.window:
            window = self.visit(node.window)

        # Build OVER clause
        over_parts = [p for p in [partition, order, window] if p]
        over_clause = f"OVER ({' '.join(over_parts)})"

        # Handle lag/lead parameters
        params_sql = ""
        if op in (LAG, LEAD) and node.params:
            params_sql = f", {node.params[0]}"
            if len(node.params) > 1:
                params_sql += f", {node.params[1]}"

        return f"{sql_op}({operand}{params_sql}) {over_clause}"

    def visit_Windowing(self, node: AST.Windowing) -> str:  # type: ignore[override]
        """Process windowing specification."""
        type_ = node.type_.upper()

        start = self._window_bound(node.start, node.start_mode)
        stop = self._window_bound(node.stop, node.stop_mode)

        return f"{type_} BETWEEN {start} AND {stop}"

    def _window_bound(self, value: Any, mode: str) -> str:
        """Convert window bound to SQL."""
        if mode == "UNBOUNDED" and (value == 0 or value == "UNBOUNDED"):
            return "UNBOUNDED PRECEDING"
        if mode == "CURRENT":
            return "CURRENT ROW"
        if isinstance(value, int):
            if value >= 0:
                return f"{value} PRECEDING"
            else:
                return f"{abs(value)} FOLLOWING"
        return "CURRENT ROW"

    def visit_OrderBy(self, node: AST.OrderBy) -> str:  # type: ignore[override]
        """Process order by specification."""
        return f'"{node.component}" {node.order.upper()}'

    # =========================================================================
    # Join Operations
    # =========================================================================

    def visit_JoinOp(self, node: AST.JoinOp) -> str:  # type: ignore[override]
        """Process join operations."""
        op = str(node.op).lower()

        # Map VTL join types to SQL
        join_type = {
            INNER_JOIN: "INNER JOIN",
            LEFT_JOIN: "LEFT JOIN",
            FULL_JOIN: "FULL OUTER JOIN",
            CROSS_JOIN: "CROSS JOIN",
        }.get(op, "INNER JOIN")

        if len(node.clauses) < 2:
            return ""

        def extract_clause_and_alias(clause: AST.AST) -> Tuple[AST.AST, Optional[str]]:
            """
            Extract the actual dataset node and its alias from a join clause.

            VTL join clauses like `ds as A` are represented as:
            BinOp(left=ds, op='as', right=Identifier)
            """
            if isinstance(clause, AST.BinOp) and str(clause.op).lower() == "as":
                # Clause has an explicit alias
                actual_clause = clause.left
                alias = clause.right.value if hasattr(clause.right, "value") else str(clause.right)
                return actual_clause, alias
            return clause, None

        def get_clause_sql(clause: AST.AST) -> str:
            """Get SQL for a join clause - direct ref for VarID, wrapped subquery otherwise."""
            if isinstance(clause, AST.VarID):
                return f'"{clause.value}"'
            else:
                return f"({self.visit(clause)})"

        def get_clause_transformed_ds(clause: AST.AST) -> Optional[Dataset]:
            """Get the transformed dataset structure for a join clause."""
            # Use unified get_structure() which handles all node types
            return self.get_structure(clause)

        # First clause is the base
        base_actual, base_alias = extract_clause_and_alias(node.clauses[0])
        base_sql = get_clause_sql(base_actual)
        base_ds = get_clause_transformed_ds(base_actual)

        # Use explicit alias if provided, otherwise use t0
        base_table_alias = base_alias if base_alias else "t0"
        result_sql = f"{base_sql} AS {base_table_alias}"

        # Track accumulated identifiers from all joined tables
        accumulated_ids: set[str] = set()
        if base_ds:
            accumulated_ids = set(base_ds.get_identifiers_names())

        for i, clause in enumerate(node.clauses[1:], 1):
            clause_actual, clause_alias = extract_clause_and_alias(clause)
            clause_sql = get_clause_sql(clause_actual)
            clause_ds = get_clause_transformed_ds(clause_actual)

            # Use explicit alias if provided, otherwise use t{i}
            table_alias = clause_alias if clause_alias else f"t{i}"

            if node.using and op != CROSS_JOIN:
                # Explicit USING clause provided
                using_cols = ", ".join([f'"{c}"' for c in node.using])
                result_sql += f"\n{join_type} {clause_sql} AS {table_alias} USING ({using_cols})"
            elif op == CROSS_JOIN:
                # CROSS JOIN doesn't need ON clause
                result_sql += f"\n{join_type} {clause_sql} AS {table_alias}"
            elif clause_ds:
                # Find common identifiers using accumulated ids from previous joins
                clause_ids = set(clause_ds.get_identifiers_names())
                common_ids = sorted(accumulated_ids.intersection(clause_ids))

                if common_ids:
                    # Use USING for common identifiers
                    using_cols = ", ".join([f'"{c}"' for c in common_ids])
                    result_sql += (
                        f"\n{join_type} {clause_sql} AS {table_alias} USING ({using_cols})"
                    )
                else:
                    # No common identifiers - should be a cross join
                    result_sql += f"\nCROSS JOIN {clause_sql} AS {table_alias}"

                # Add clause's identifiers to accumulated set for next join
                accumulated_ids.update(clause_ids)
            else:
                # Fallback: no ON clause (will fail for most joins)
                result_sql += f"\n{join_type} {clause_sql} AS {table_alias}"

        return f"SELECT * FROM {result_sql}"

    # =========================================================================
    # Parenthesized Expression
    # =========================================================================

    def visit_ParFunction(self, node: AST.ParFunction) -> str:  # type: ignore[override]
        """Process parenthesized expression."""
        inner = self.visit(node.operand)
        return f"({inner})"

    # =========================================================================
    # Validation Operations
    # =========================================================================

    def _get_measure_name_from_expression(self, expr: AST.AST) -> Optional[str]:
        """
        Extract the measure column name from an expression for use in check operations.

        When a validation expression like `agg1 + agg2 < 1000` is evaluated,
        comparison operations rename single measures to 'bool_var'.
        This helper traces through the expression to find that measure name.
        """
        if isinstance(expr, AST.VarID):
            # Direct dataset reference
            ds = self.available_tables.get(expr.value)
            if ds:
                measures = list(ds.get_measures_names())
                if measures:
                    return measures[0]
        elif isinstance(expr, AST.UnaryOp):
            # For unary ops like isnull, not, etc.
            op = str(expr.op).lower()
            if op == NOT:
                # NOT on datasets produces bool_var as output measure
                # Check if operand is dataset-level
                operand_type = self._get_operand_type(expr.operand)
                if operand_type == OperandType.DATASET:
                    return "bool_var"
                # For scalar NOT, keep the same measure name
                return self._get_measure_name_from_expression(expr.operand)
            elif op == ISNULL:
                # isnull on datasets produces bool_var as output measure
                operand_type = self._get_operand_type(expr.operand)
                if operand_type == OperandType.DATASET:
                    return "bool_var"
                return self._get_measure_name_from_expression(expr.operand)
            else:
                return self._get_measure_name_from_expression(expr.operand)
        elif isinstance(expr, AST.BinOp):
            # Check if this is a comparison operation
            op = str(expr.op).lower()
            comparison_ops = {EQ, NEQ, GT, GTE, LT, LTE, "=", "<>", ">", ">=", "<", "<="}
            if op in comparison_ops:
                # Comparisons on mono-measure datasets produce bool_var
                return "bool_var"
            # Check if this is a membership operation
            if op == MEMBERSHIP:
                # Membership extracts single component - that becomes the measure
                return expr.right.value if hasattr(expr.right, "value") else str(expr.right)
            # For non-comparison binary operations, get measure from operands
            left_measure = self._get_measure_name_from_expression(expr.left)
            if left_measure:
                return left_measure
            return self._get_measure_name_from_expression(expr.right)
        elif isinstance(expr, AST.ParFunction):
            # Parenthesized expression - look inside
            return self._get_measure_name_from_expression(expr.operand)
        elif isinstance(expr, AST.Aggregation):
            # Aggregation - get measure from operand
            if expr.operand:
                return self._get_measure_name_from_expression(expr.operand)
        return None

    def _get_identifiers_from_expression(self, expr: AST.AST) -> List[str]:
        """
        Extract identifier column names from an expression.

        Traces through the expression to find the underlying dataset
        and returns its identifier column names.
        """
        if isinstance(expr, AST.VarID):
            # Direct dataset reference
            ds = self.available_tables.get(expr.value)
            if ds:
                return list(ds.get_identifiers_names())
        elif isinstance(expr, AST.BinOp):
            # For binary operations, get identifiers from left operand
            left_ids = self._get_identifiers_from_expression(expr.left)
            if left_ids:
                return left_ids
            return self._get_identifiers_from_expression(expr.right)
        elif isinstance(expr, AST.ParFunction):
            # Parenthesized expression - look inside
            return self._get_identifiers_from_expression(expr.operand)
        elif isinstance(expr, AST.Aggregation):
            # Aggregation - identifiers come from grouping, not operand
            if expr.grouping and expr.grouping_op == "group by":
                return [
                    g.value if isinstance(g, (AST.VarID, AST.Identifier)) else str(g)
                    for g in expr.grouping
                ]
            elif expr.operand:
                return self._get_identifiers_from_expression(expr.operand)
        return []

    def visit_Validation(self, node: AST.Validation) -> str:
        """
        Process CHECK validation operation.

        VTL: check(ds, condition, error_code, error_level)
        Returns dataset with errorcode, errorlevel, and optionally imbalance columns.
        """
        # Get the validation element (contains the condition result)
        validation_sql = self.visit(node.validation)

        # Determine the boolean column name to check
        # If validation is a direct dataset reference, find its boolean measure
        bool_col = "bool_var"  # Default
        if isinstance(node.validation, AST.VarID):
            ds_name = node.validation.value
            ds = self.available_tables.get(ds_name)
            if ds:
                # Find boolean measure column
                for m in ds.get_measures_names():
                    comp = ds.components.get(m)
                    if comp and comp.data_type.__name__ == "Boolean":
                        bool_col = m
                        break
                else:
                    # No boolean measure found, use first measure
                    measures = ds.get_measures_names()
                    if measures:
                        bool_col = measures[0]
        else:
            # For complex expressions (like comparisons), extract measure name
            measure_name = self._get_measure_name_from_expression(node.validation)
            if measure_name:
                bool_col = measure_name

        # Get error code and level
        error_code = node.error_code if node.error_code else "NULL"
        if error_code != "NULL" and not error_code.startswith("'"):
            error_code = f"'{error_code}'"

        error_level = node.error_level if node.error_level is not None else "NULL"

        # Handle imbalance - always include the column (NULL if not specified)
        # Imbalance can be a dataset expression - we need to join it properly
        imbalance_join = ""
        imbalance_select = ", NULL AS imbalance"  # Default to NULL if no imbalance
        if node.imbalance:
            imbalance_expr = self.visit(node.imbalance)
            imbalance_type = self._get_operand_type(node.imbalance)

            if imbalance_type == OperandType.DATASET:
                # Imbalance is a dataset - we need to JOIN it
                # Get the measure name from the imbalance expression
                imbalance_measure = self._get_measure_name_from_expression(node.imbalance)
                if not imbalance_measure:
                    imbalance_measure = "IMPORTO"  # Default fallback

                # Get identifiers from the validation expression for JOIN
                id_cols = self._get_identifiers_from_expression(node.validation)
                if id_cols:
                    join_cond = " AND ".join([f't."{c}" = imb."{c}"' for c in id_cols])
                    # Check if imbalance is a simple table reference (VarID) vs subquery
                    if isinstance(node.imbalance, AST.VarID):
                        # Simple table reference - don't wrap in parentheses
                        imbalance_join = f"""
                            LEFT JOIN "{node.imbalance.value}" AS imb ON {join_cond}
                        """
                    else:
                        # Complex expression - wrap in parentheses as subquery
                        imbalance_join = f"""
                            LEFT JOIN ({imbalance_expr}) AS imb ON {join_cond}
                        """
                    imbalance_select = f', imb."{imbalance_measure}" AS imbalance'
                else:
                    # No identifiers found - use a cross join with scalar result
                    imbalance_select = f", ({imbalance_expr}) AS imbalance"
            else:
                # Scalar imbalance - embed directly
                imbalance_select = f", ({imbalance_expr}) AS imbalance"

        # Generate check result
        if node.invalid:
            # Return only invalid rows (where bool column is False)
            return f"""
                SELECT t.*,
                       {error_code} AS errorcode,
                       {error_level} AS errorlevel{imbalance_select}
                FROM ({validation_sql}) AS t
                {imbalance_join}
                WHERE t."{bool_col}" = FALSE OR t."{bool_col}" IS NULL
            """
        else:
            # Return all rows with validation info
            return f"""
                SELECT t.*,
                       CASE WHEN t."{bool_col}" = FALSE OR t."{bool_col}" IS NULL
                            THEN {error_code} ELSE NULL END AS errorcode,
                       CASE WHEN t."{bool_col}" = FALSE OR t."{bool_col}" IS NULL
                            THEN {error_level} ELSE NULL END AS errorlevel{imbalance_select}
                FROM ({validation_sql}) AS t
                {imbalance_join}
            """

    def visit_DPValidation(self, node: AST.DPValidation) -> str:  # type: ignore[override]
        """
        Process CHECK_DATAPOINT validation operation.

        VTL: check_datapoint(ds, ruleset, components, output)
        Validates data against a datapoint ruleset.

        Generates a UNION of queries, one per rule in the ruleset.
        Each rule query evaluates the rule condition and adds validation columns.
        """
        # Get the dataset SQL
        dataset_sql = self._get_dataset_sql(node.dataset)

        # Get dataset info
        ds_name = self._get_dataset_name(node.dataset)
        ds = self.available_tables.get(ds_name)

        # Output mode determines what to return
        output_mode = node.output.value if node.output else "invalid"

        # Get output structure from semantic analysis if available
        if self.current_result_name:
            self.output_datasets.get(self.current_result_name)

        # Get ruleset definition
        dpr_info = self.dprs.get(node.ruleset_name)

        # Build column selections
        if ds:
            id_cols = ds.get_identifiers_names()
            measure_cols = ds.get_measures_names()
        else:
            id_cols = []
            measure_cols = []

        id_select = ", ".join([f't."{k}"' for k in id_cols])

        # For output modes that include measures
        measure_select = ", ".join([f't."{m}"' for m in measure_cols])

        # Set current dataset context for rule condition evaluation
        prev_dataset = self.current_dataset
        self.current_dataset = ds

        # Generate queries for each rule
        rule_queries = []

        if dpr_info and dpr_info.get("rules"):
            for rule in dpr_info["rules"]:
                rule_name = rule.name or "unknown"
                error_code = f"'{rule.erCode}'" if rule.erCode else "NULL"
                error_level = rule.erLevel if rule.erLevel is not None else "NULL"

                # Transpile the rule condition
                try:
                    condition_sql = self._visit_dp_rule_condition(rule.rule)
                except Exception:
                    # Fallback: if rule can't be transpiled, assume all pass
                    condition_sql = "TRUE"

                # Build query for this rule
                cols = id_select
                if output_mode in ("invalid", "all_measures") and measure_select:
                    cols += f", {measure_select}"

                if output_mode == "invalid":
                    # Return only failing rows (where condition is FALSE)
                    # NULL results are treated as "not applicable", not as failures
                    rule_query = f"""
                        SELECT {cols},
                               '{rule_name}' AS ruleid,
                               {error_code} AS errorcode,
                               {error_level} AS errorlevel
                        FROM ({dataset_sql}) AS t
                        WHERE ({condition_sql}) = FALSE
                    """
                elif output_mode == "all_measures":
                    rule_query = f"""
                        SELECT {cols},
                               ({condition_sql}) AS bool_var
                        FROM ({dataset_sql}) AS t
                    """
                else:  # "all"
                    rule_query = f"""
                        SELECT {cols},
                               '{rule_name}' AS ruleid,
                               ({condition_sql}) AS bool_var,
                               CASE WHEN NOT ({condition_sql}) OR ({condition_sql}) IS NULL
                                    THEN {error_code} ELSE NULL END AS errorcode,
                               CASE WHEN NOT ({condition_sql}) OR ({condition_sql}) IS NULL
                                    THEN {error_level} ELSE NULL END AS errorlevel
                        FROM ({dataset_sql}) AS t
                    """
                rule_queries.append(rule_query)
        else:
            # No ruleset found - generate placeholder query
            cols = id_select
            if output_mode in ("invalid", "all_measures") and measure_select:
                cols += f", {measure_select}"

            if output_mode == "invalid":
                rule_queries.append(f"""
                    SELECT {cols},
                           '{node.ruleset_name}' AS ruleid,
                           'unknown_rule' AS errorcode,
                           1 AS errorlevel
                    FROM ({dataset_sql}) AS t
                    WHERE FALSE
                """)
            elif output_mode == "all_measures":
                rule_queries.append(f"""
                    SELECT {cols},
                           TRUE AS bool_var
                    FROM ({dataset_sql}) AS t
                """)
            else:
                rule_queries.append(f"""
                    SELECT {cols},
                           '{node.ruleset_name}' AS ruleid,
                           TRUE AS bool_var,
                           NULL AS errorcode,
                           NULL AS errorlevel
                    FROM ({dataset_sql}) AS t
                """)

        # Restore context
        self.current_dataset = prev_dataset

        # Combine all rule queries with UNION ALL
        if len(rule_queries) == 1:
            return rule_queries[0]
        return " UNION ALL ".join([f"({q})" for q in rule_queries])

    def _get_in_values(self, node: AST.AST) -> str:
        """
        Get the SQL representation of the right side of an IN/NOT IN operator.

        Handles:
        - Collection nodes: inline sets like {"A", "B"}
        - VarID/Identifier nodes: value domain references
        - Other expressions
        """
        if isinstance(node, AST.Collection):
            # Inline collection like {"A", "B"}
            if node.children:
                values = [self._visit_dp_rule_condition(c) for c in node.children]
                return ", ".join(values)
            # Named collection - check if it's a value domain
            if hasattr(node, "name") and node.name in self.value_domains:
                vd = self.value_domains[node.name]
                if hasattr(vd, "data"):
                    values = [f"'{v}'" if isinstance(v, str) else str(v) for v in vd.data]
                    return ", ".join(values)
            return "NULL"
        elif isinstance(node, (AST.VarID, AST.Identifier)):
            # Check if this is a value domain reference
            vd_name = node.value
            if vd_name in self.value_domains:
                vd = self.value_domains[vd_name]
                if hasattr(vd, "data"):
                    values = [f"'{v}'" if isinstance(v, str) else str(v) for v in vd.data]
                    return ", ".join(values)
            # Not a value domain - treat as column reference (might be subquery)
            return f't."{vd_name}"'
        else:
            # Fallback - recursively process
            return self._visit_dp_rule_condition(node)

    def _visit_dp_rule_condition_as_bool(self, node: AST.AST) -> str:
        """
        Transpile a datapoint rule operand ensuring boolean output.

        For bare VarID nodes (column references), convert to a boolean check.
        In VTL rules, a bare NEVS_* column typically checks if value = '0' (reported).
        For other columns, check if value is not null.
        """
        if isinstance(node, (AST.VarID, AST.Identifier)):
            # Bare column reference - convert to boolean check
            col_name = node.value
            # NEVS columns: "0" means reported (truthy), others are falsy
            if col_name.startswith("NEVS_"):
                return f"(t.\"{col_name}\" = '0')"
            else:
                # For other columns, check if not null
                return f'(t."{col_name}" IS NOT NULL)'
        else:
            # Not a bare VarID - process normally
            return self._visit_dp_rule_condition(node)

    def _visit_dp_rule_condition(self, node: AST.AST) -> str:
        """
        Transpile a datapoint rule condition to SQL.

        Handles HRBinOp nodes which represent rule conditions like:
        - when condition then validation
        - simple comparisons
        """
        if isinstance(node, AST.If):
            # VTL: if condition then thenOp else elseOp
            # VTL semantics: if condition is NULL, result is NULL (not elseOp!)
            # SQL: CASE WHEN cond IS NULL THEN NULL WHEN cond THEN thenOp ELSE elseOp END
            condition = self._visit_dp_rule_condition(node.condition)
            # Handle bare VarID operands - convert to boolean check
            # In VTL rules, bare column ref like NEVS_X means checking if value = '0'
            then_op = self._visit_dp_rule_condition_as_bool(node.thenOp)
            else_op = self._visit_dp_rule_condition_as_bool(node.elseOp)
            return (
                f"CASE WHEN ({condition}) IS NULL THEN NULL "
                f"WHEN ({condition}) THEN ({then_op}) ELSE ({else_op}) END"
            )
        elif isinstance(node, AST.HRBinOp):
            op_str = str(node.op).upper() if node.op else ""
            if op_str == "WHEN":
                # WHEN condition THEN validation
                # VTL semantics: when WHEN condition is NULL, the rule result is NULL
                # In SQL: CASE WHEN cond IS NULL THEN NULL WHEN cond THEN validation ELSE TRUE END
                when_cond = self._visit_dp_rule_condition(node.left)
                then_cond = self._visit_dp_rule_condition(node.right)
                return (
                    f"CASE WHEN ({when_cond}) IS NULL THEN NULL "
                    f"WHEN ({when_cond}) THEN ({then_cond}) ELSE TRUE END"
                )
            else:
                # Binary operation (comparison, logical)
                left = self._visit_dp_rule_condition(node.left)
                right = self._visit_dp_rule_condition(node.right)
                sql_op = SQL_BINARY_OPS.get(node.op, str(node.op))
                return f"({left}) {sql_op} ({right})"
        elif isinstance(node, AST.BinOp):
            op_str = str(node.op).lower() if node.op else ""
            # Handle IN operator specially
            if op_str == "in":
                left = self._visit_dp_rule_condition(node.left)
                values_sql = self._get_in_values(node.right)
                return f"({left}) IN ({values_sql})"
            elif op_str == "not_in":
                left = self._visit_dp_rule_condition(node.left)
                values_sql = self._get_in_values(node.right)
                return f"({left}) NOT IN ({values_sql})"
            else:
                left = self._visit_dp_rule_condition(node.left)
                right = self._visit_dp_rule_condition(node.right)
                # Map VTL operator to SQL
                sql_op = SQL_BINARY_OPS.get(node.op, node.op)
                return f"({left}) {sql_op} ({right})"
        elif isinstance(node, AST.UnaryOp):
            operand = self._visit_dp_rule_condition(node.operand)
            op_upper = node.op.upper() if isinstance(node.op, str) else str(node.op).upper()
            if op_upper == "NOT":
                return f"NOT ({operand})"
            elif op_upper == "ISNULL":
                return f"({operand}) IS NULL"
            return f"{node.op} ({operand})"
        elif isinstance(node, (AST.VarID, AST.Identifier)):
            # Component reference
            return f't."{node.value}"'
        elif isinstance(node, AST.Constant):
            if node.type_ == "STRING_CONSTANT":
                return f"'{node.value}'"
            elif node.type_ == "BOOLEAN_CONSTANT":
                return "TRUE" if node.value else "FALSE"
            return str(node.value)
        elif isinstance(node, AST.ParFunction):
            # Parenthesized expression - process the operand
            return f"({self._visit_dp_rule_condition(node.operand)})"
        elif isinstance(node, AST.MulOp):
            # Handle IN, NOT_IN, and other multi-operand operations
            op_str = str(node.op).upper()
            if op_str in ("IN", "NOT_IN"):
                left = self._visit_dp_rule_condition(node.children[0])
                values = [self._visit_dp_rule_condition(c) for c in node.children[1:]]
                op = "IN" if op_str == "IN" else "NOT IN"
                return f"({left}) {op} ({', '.join(values)})"
            # Other MulOp - process children with operator
            parts = [self._visit_dp_rule_condition(c) for c in node.children]
            sql_op = SQL_BINARY_OPS.get(node.op, str(node.op))
            return f" {sql_op} ".join([f"({p})" for p in parts])
        elif isinstance(node, AST.Collection):
            # Value domain reference - return the values
            if hasattr(node, "name") and node.name in self.value_domains:
                vd = self.value_domains[node.name]
                if hasattr(vd, "data"):
                    # Get values from value domain
                    values = [f"'{v}'" if isinstance(v, str) else str(v) for v in vd.data]
                    return f"({', '.join(values)})"
            # Fallback - just return the collection name
            return f'"{node.name}"' if hasattr(node, "name") else "NULL"
        else:
            # Fallback to generic visit
            return self.visit(node)

    def visit_HROperation(self, node: AST.HROperation) -> str:  # type: ignore[override]
        """
        Process hierarchical operations (hierarchy, check_hierarchy).

        VTL: hierarchy(ds, ruleset, ...) or check_hierarchy(ds, ruleset, ...)
        """
        # Get the dataset SQL
        dataset_sql = self._get_dataset_sql(node.dataset)

        # Get dataset info
        ds_name = self._get_dataset_name(node.dataset)
        ds = self.available_tables.get(ds_name)

        op = node.op.lower()

        if ds:
            id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
            measure_select = ", ".join([f'"{m}"' for m in ds.get_measures_names()])
        else:
            id_select = "*"
            measure_select = ""

        if op == "check_hierarchy":
            # check_hierarchy returns validation results
            output_mode = node.output.value if node.output else "all"

            if output_mode == "invalid":
                return f"""
                    SELECT {id_select},
                           '{node.ruleset_name}' AS ruleid,
                           FALSE AS bool_var,
                           'hierarchy_error' AS errorcode,
                           1 AS errorlevel,
                           0 AS imbalance
                    FROM ({dataset_sql}) AS t
                    WHERE FALSE  -- Placeholder: actual hierarchy validation
                """
            else:
                return f"""
                    SELECT {id_select},
                           '{node.ruleset_name}' AS ruleid,
                           TRUE AS bool_var,
                           NULL AS errorcode,
                           NULL AS errorlevel,
                           0 AS imbalance
                    FROM ({dataset_sql}) AS t
                """
        else:
            # hierarchy operation computes aggregations based on ruleset
            output_mode = node.output.value if node.output else "computed"

            if output_mode == "all":
                return f"""
                    SELECT {id_select}, {measure_select}
                    FROM ({dataset_sql}) AS t
                """
            else:  # "computed"
                return f"""
                    SELECT {id_select}, {measure_select}
                    FROM ({dataset_sql}) AS t
                """

    # =========================================================================
    # Eval Operator (External Routines)
    # =========================================================================

    def visit_EvalOp(self, node: AST.EvalOp) -> str:
        """
        Process EVAL operator for external routines.

        VTL: eval(routine_name(DS_1, ...) language "SQL" returns dataset_spec)

        The external routine contains a SQL query that is executed directly.
        The transpiler replaces dataset references in the query with the
        appropriate SQL for those datasets.
        """
        routine_name = node.name

        # Check that external routines are provided
        if not self.external_routines:
            raise ValueError(
                f"External routine '{routine_name}' referenced but no external routines provided"
            )

        if routine_name not in self.external_routines:
            raise ValueError(f"External routine '{routine_name}' not found")

        external_routine = self.external_routines[routine_name]

        # Get SQL for each operand dataset
        operand_sql_map: Dict[str, str] = {}
        for operand in node.operands:
            if isinstance(operand, AST.VarID):
                ds_name = operand.value
                operand_sql_map[ds_name] = self._get_dataset_sql(operand)
            elif isinstance(operand, AST.Constant):
                # Constants are passed directly (not common in EVAL)
                pass

        # The external routine query is the SQL to execute
        # We need to replace table references with the appropriate SQL
        query = external_routine.query

        # Replace dataset references in the query with subqueries
        # The external routine has dataset_names extracted from the query
        for ds_name in external_routine.dataset_names:
            if ds_name in operand_sql_map:
                # Replace table reference with subquery
                # Be careful with quoting - DuckDB uses double quotes for identifiers
                subquery_sql = operand_sql_map[ds_name]

                # If it's a simple SELECT * FROM "table", we can use the table directly
                table_ref = self._extract_table_from_select(subquery_sql)
                if table_ref:
                    # Just use the table name as-is (it's already in the query)
                    continue
                else:
                    # Replace the table reference with a subquery
                    # Pattern: FROM ds_name or FROM "ds_name"
                    import re

                    # Replace unquoted or quoted references
                    query = re.sub(
                        rf'\bFROM\s+"{ds_name}"',
                        f"FROM ({subquery_sql}) AS {ds_name}",
                        query,
                        flags=re.IGNORECASE,
                    )
                    query = re.sub(
                        rf"\bFROM\s+{ds_name}\b",
                        f"FROM ({subquery_sql}) AS {ds_name}",
                        query,
                        flags=re.IGNORECASE,
                    )

        return query

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_operand_type(self, node: AST.AST) -> str:
        """Determine the type of an operand."""
        if isinstance(node, AST.VarID):
            name = node.value

            # Check if this is a UDO parameter - if so, get type of bound value
            udo_value = self.get_udo_param(name)
            if udo_value is not None:
                if isinstance(udo_value, AST.AST):
                    return self._get_operand_type(udo_value)
                # String values are typically component names
                if isinstance(udo_value, str):
                    return OperandType.COMPONENT
                # Scalar objects
                return OperandType.SCALAR

            # In clause context: component
            if self.in_clause and self.current_dataset and name in self.current_dataset.components:
                return OperandType.COMPONENT

            # Known dataset
            if name in self.available_tables:
                return OperandType.DATASET

            # Known scalar (from input or output)
            if name in self.input_scalars or name in self.output_scalars:
                return OperandType.SCALAR

            # Default in clause: component
            if self.in_clause:
                return OperandType.COMPONENT

            return OperandType.SCALAR

        elif isinstance(node, AST.Constant):
            return OperandType.SCALAR

        elif isinstance(node, AST.BinOp):
            return self._get_operand_type(node.left)

        elif isinstance(node, AST.UnaryOp):
            return self._get_operand_type(node.operand)

        elif isinstance(node, AST.ParamOp):
            if node.children:
                return self._get_operand_type(node.children[0])

        elif isinstance(node, (AST.RegularAggregation, AST.JoinOp)):
            return OperandType.DATASET

        elif isinstance(node, AST.Aggregation):
            # In clause context, aggregation on a component is a scalar SQL aggregate
            if self.in_clause and node.operand:
                operand_type = self._get_operand_type(node.operand)
                if operand_type in (OperandType.COMPONENT, OperandType.SCALAR):
                    return OperandType.SCALAR
            return OperandType.DATASET

        elif isinstance(node, AST.If):
            return self._get_operand_type(node.thenOp)

        elif isinstance(node, AST.ParFunction):
            return self._get_operand_type(node.operand)

        elif isinstance(node, AST.UDOCall):
            # UDOCall returns what its output type specifies
            if node.op in self.udos:
                output_type = self.udos[node.op]["output"]
                type_mapping = {
                    "Dataset": OperandType.DATASET,
                    "Scalar": OperandType.SCALAR,
                    "Component": OperandType.COMPONENT,
                }
                return type_mapping.get(output_type, OperandType.DATASET)
            # Default to dataset if we don't know
            return OperandType.DATASET

        return OperandType.SCALAR

    def _get_transformed_measure_name(self, node: AST.AST) -> Optional[str]:
        """
        Extract the final measure name from a node after all transformations.

        For expressions like `DS [ keep X ] [ rename X to Y ]`, this returns 'Y'.
        """
        if isinstance(node, AST.VarID):
            # Direct dataset reference - get the first measure from structure
            ds = self.get_structure(node)
            if ds:
                measures = list(ds.get_measures_names())
                return measures[0] if measures else None
            return None

        if isinstance(node, AST.RegularAggregation):
            from vtlengine.AST.Grammar.tokens import CALC, KEEP, RENAME

            # Check the operation type
            op = str(node.op).lower()

            if op == RENAME:
                # For rename, the final measure name is in the RenameNode
                for child in node.children:
                    if isinstance(child, AST.RenameNode):
                        return child.new_name
                # Fallback to inner dataset
                if node.dataset:
                    return self._get_transformed_measure_name(node.dataset)

            elif op == CALC:
                # For calc, the measure name is in Assignment.left
                # Children can be UnaryOp (with role) wrapping Assignment, or Assignment directly
                for child in node.children:
                    if isinstance(child, AST.UnaryOp) and hasattr(child, "operand"):
                        assignment = child.operand
                    elif isinstance(child, AST.Assignment):
                        assignment = child
                    else:
                        continue
                    if isinstance(assignment, AST.Assignment) and isinstance(
                        assignment.left, (AST.VarID, AST.Identifier)
                    ):
                        return assignment.left.value
                # Fallback to inner dataset
                if node.dataset:
                    return self._get_transformed_measure_name(node.dataset)

            elif op == KEEP:
                # For keep, the kept measure is the final name
                # Look for the measure in children (excluding identifiers)
                if node.dataset:
                    inner_ds = self.get_structure(node.dataset)
                    if inner_ds:
                        inner_ids = set(inner_ds.get_identifiers_names())
                        # Find the kept measure (not an identifier)
                        for child in node.children:
                            if (
                                isinstance(child, (AST.VarID, AST.Identifier))
                                and child.value not in inner_ids
                            ):
                                return child.value
                # If inner RegularAggregation, recurse
                if node.dataset:
                    return self._get_transformed_measure_name(node.dataset)

            else:
                # Other clauses (filter, subspace, etc.) - recurse to inner dataset
                if node.dataset:
                    return self._get_transformed_measure_name(node.dataset)

        if isinstance(node, AST.BinOp):
            return self._get_transformed_measure_name(node.left)

        if isinstance(node, AST.UnaryOp):
            return self._get_transformed_measure_name(node.operand)

        return None

    def _get_dataset_name(self, node: AST.AST) -> str:
        """Extract dataset name from a node, resolving UDO parameters."""
        if isinstance(node, AST.VarID):
            return self._resolve_varid_value(node)
        if isinstance(node, AST.RegularAggregation) and node.dataset:
            return self._get_dataset_name(node.dataset)
        if isinstance(node, AST.BinOp):
            return self._get_dataset_name(node.left)
        if isinstance(node, AST.UnaryOp):
            return self._get_dataset_name(node.operand)
        if isinstance(node, AST.ParamOp) and node.children:
            return self._get_dataset_name(node.children[0])
        if isinstance(node, AST.ParFunction):
            return self._get_dataset_name(node.operand)
        if isinstance(node, AST.Aggregation) and node.operand:
            return self._get_dataset_name(node.operand)
        if isinstance(node, AST.JoinOp) and node.clauses:
            # For joins, return the first dataset name (used as the primary dataset context)
            return self._get_dataset_name(node.clauses[0])
        if isinstance(node, AST.UDOCall):
            # For UDO calls, get the dataset name from the first parameter
            # (UDOs that return datasets typically take a dataset as first arg)
            if node.params:
                return self._get_dataset_name(node.params[0])
            # If no params, use the UDO name as fallback
            return node.op

        raise ValueError(f"Cannot extract dataset name from {type(node).__name__}")

    def _get_dataset_sql(self, node: AST.AST, wrap_simple: bool = True) -> str:
        """
        Get SQL for a dataset node.

        Args:
            node: AST node representing a dataset
            wrap_simple: If False, return just table name for VarID nodes
                        If True, return SELECT * FROM for compatibility
        """
        if isinstance(node, AST.VarID):
            # Check if this is a UDO parameter bound to an AST node
            udo_value = self.get_udo_param(node.value)
            if udo_value is not None and isinstance(udo_value, AST.AST):
                # Recursively get SQL for the bound AST node
                return self._get_dataset_sql(udo_value, wrap_simple)

            # Resolve UDO parameter bindings to get actual dataset name
            name = self._resolve_varid_value(node)
            if wrap_simple:
                return f'SELECT * FROM "{name}"'
            return f'"{name}"'

        # Otherwise, transpile the node
        return self.visit(node)

    def _extract_table_from_select(self, sql: str) -> Optional[str]:
        """
        Extract the table name from a simple SELECT * FROM "table" statement.
        Returns the quoted table name or None if not a simple select.

        This only matches truly simple selects - not JOINs, WHERE, or other clauses.
        """
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()
        if sql_upper.startswith("SELECT * FROM "):
            remainder = sql_stripped[14:].strip()
            if remainder.startswith('"') and '"' in remainder[1:]:
                end_quote = remainder.index('"', 1) + 1
                table_name = remainder[:end_quote]
                # Make sure there's nothing else after the table name (or just an alias)
                rest = remainder[end_quote:].strip()
                rest_upper = rest.upper()

                # Accept empty rest (no alias)
                if not rest:
                    return table_name

                # Accept AS alias, but only if there's nothing complex after it
                if rest_upper.startswith("AS "):
                    # Skip past the alias
                    after_as = rest[3:].strip()
                    # Skip the alias identifier (may be quoted or unquoted)
                    if after_as.startswith('"'):
                        # Quoted alias
                        if '"' in after_as[1:]:
                            alias_end = after_as.index('"', 1) + 1
                            after_alias = after_as[alias_end:].strip().upper()
                        else:
                            return None  # Malformed
                    else:
                        # Unquoted alias - ends at whitespace or end
                        alias_parts = after_as.split()
                        after_alias = (
                            " ".join(alias_parts[1:]).upper() if len(alias_parts) > 1 else ""
                        )

                    # Reject if there's a JOIN or other complex clause after alias
                    complex_keywords = [
                        "JOIN",
                        "INNER",
                        "LEFT",
                        "RIGHT",
                        "FULL",
                        "CROSS",
                        "WHERE",
                        "GROUP",
                        "ORDER",
                        "HAVING",
                        "UNION",
                        "INTERSECT",
                    ]
                    if any(kw in after_alias for kw in complex_keywords):
                        return None

                    # Accept if nothing after alias or non-complex content
                    if not after_alias:
                        return table_name

        return None

    def _simplify_from_clause(self, subquery_sql: str) -> str:
        """
        Simplify FROM clause by avoiding unnecessary nesting.
        If the subquery is just SELECT * FROM "table", return just the table name.
        Otherwise, return the subquery wrapped in parentheses.
        """
        table_ref = self._extract_table_from_select(subquery_sql)
        if table_ref:
            return f"{table_ref}"
        return f"({subquery_sql})"

    def _optimize_filter_pushdown(self, base_sql: str, filter_condition: str) -> str:
        """
        Push filter conditions into subqueries when possible.

        This optimization avoids unnecessary nesting of subqueries by:
        1. If base_sql is a simple SELECT * FROM "table", add WHERE directly
        2. If base_sql is SELECT * FROM "table" with existing WHERE, combine
        3. Otherwise, wrap in subquery

        Args:
            base_sql: The base SQL query to filter.
            filter_condition: The WHERE condition to apply.

        Returns:
            Optimized SQL with filter applied.
        """
        sql_stripped = base_sql.strip()
        sql_upper = sql_stripped.upper()

        # Case 1: Simple SELECT * FROM "table" without WHERE
        table_ref = self._extract_table_from_select(sql_stripped)
        if table_ref and "WHERE" not in sql_upper:
            return f"SELECT * FROM {table_ref} WHERE {filter_condition}"

        # Case 2: SELECT * FROM "table" with existing WHERE - combine conditions
        if table_ref and " WHERE " in sql_upper:
            # Insert the new condition at the end of the existing WHERE
            # Find the WHERE position in original SQL (preserve case)
            where_pos = sql_upper.find(" WHERE ")
            if where_pos != -1:
                return f"{sql_stripped} AND {filter_condition}"

        # Case 3: Default - wrap in subquery
        return f"SELECT * FROM ({sql_stripped}) AS t WHERE {filter_condition}"

    def _scalar_to_sql(self, scalar: Scalar) -> str:
        """Convert a Scalar to SQL literal."""
        if scalar.value is None:
            return "NULL"

        type_name = scalar.data_type.__name__
        if type_name == "String":
            escaped = str(scalar.value).replace("'", "''")
            return f"'{escaped}'"
        elif type_name == "Integer":
            return str(int(scalar.value))
        elif type_name == "Number":
            return str(float(scalar.value))
        elif type_name == "Boolean":
            return "TRUE" if scalar.value else "FALSE"
        else:
            return str(scalar.value)

    def _ensure_select(self, sql: str) -> str:
        """Ensure SQL is a complete SELECT statement."""
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()

        if sql_upper.startswith("SELECT"):
            return sql_stripped

        # Check if it's a set operation (starts with subquery)
        # Patterns like: (SELECT ...) UNION/INTERSECT/EXCEPT (SELECT ...)
        if sql_stripped.startswith("(") and any(
            op in sql_upper for op in ("UNION", "INTERSECT", "EXCEPT")
        ):
            return sql_stripped

        # Check if it's a table reference (quoted identifier like "DS_1")
        # If so, convert to SELECT * FROM "table"
        if sql_stripped.startswith('"') and sql_stripped.endswith('"'):
            table_name = sql_stripped[1:-1]  # Remove quotes
            if table_name in self.available_tables:
                return f"SELECT * FROM {sql_stripped}"

        return f"SELECT {sql_stripped}"

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
from vtlengine.Model import Dataset, ExternalRoutine, Scalar, ValueDomain

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

    def __post_init__(self) -> None:
        """Initialize available tables with input datasets."""
        # Start with input datasets as available tables
        self.available_tables = dict(self.input_datasets)

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

        for child in node.children:
            if isinstance(child, (AST.Assignment, AST.PersistentAssignment)):
                result = self.visit(child)
                if result:
                    name, sql, is_persistent = result
                    queries.append((name, sql, is_persistent))

                    # Register result for subsequent queries
                    # Use output_datasets for intermediate results
                    if name in self.output_datasets:
                        self.available_tables[name] = self.output_datasets[name]

        return queries

    def visit_Assignment(self, node: AST.Assignment) -> Tuple[str, str, bool]:
        """Process a temporary assignment (:=)."""
        result_name = node.left.value
        right_sql = self.visit(node.right)

        # Ensure it's a complete SELECT statement
        sql = self._ensure_select(right_sql)

        return (result_name, sql, False)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Tuple[str, str, bool]:
        """Process a persistent assignment (<-)."""
        result_name = node.left.value
        right_sql = self.visit(node.right)

        sql = self._ensure_select(right_sql)

        return (result_name, sql, True)

    # =========================================================================
    # Variable and Constant Nodes
    # =========================================================================

    def visit_VarID(self, node: AST.VarID) -> str:
        """
        Process a variable identifier.

        Returns table reference, column reference, or scalar value depending on context.
        """
        name = node.value

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

    def visit_Constant(self, node: AST.Constant) -> str:
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

    def visit_Collection(self, node: AST.Collection) -> str:
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

    def visit_BinOp(self, node: AST.BinOp) -> str:
        """
        Process a binary operation.

        Dispatches based on operand types:
        - Dataset-Dataset: JOIN with operation on measures
        - Dataset-Scalar: Operation on all measures
        - Scalar-Scalar / Component-Component: Simple expression
        """
        left_type = self._get_operand_type(node.left)
        right_type = self._get_operand_type(node.right)

        op = node.op.lower() if isinstance(node.op, str) else str(node.op)

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
        """Generate SQL for dataset-level IN/NOT IN operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        measure_select = ", ".join(
            [f'("{m}" {sql_op} {values_sql}) AS "{m}"' for m in ds.get_measures_names()]
        )

        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

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
        """Generate SQL for dataset-level MATCH operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        measure_select = ", ".join(
            [f'regexp_full_match("{m}", {pattern_sql}) AS "{m}"' for m in ds.get_measures_names()]
        )

        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

    def _visit_exist_in(self, node: AST.BinOp) -> str:
        """
        Handle EXIST_IN operation.

        VTL: exist_in(ds1, ds2) - checks if identifiers from ds1 exist in ds2
        SQL: SELECT *, EXISTS(SELECT 1 FROM ds2 WHERE ids match) AS bool_var
        """
        left_name = self._get_dataset_name(node.left)
        right_name = self._get_dataset_name(node.right)

        left_ds = self.available_tables[left_name]
        right_ds = self.available_tables[right_name]

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
        """
        left_type = self._get_operand_type(node.left)
        replacement = self.visit(node.right)

        # Dataset-level NVL
        if left_type == OperandType.DATASET:
            ds_name = self._get_dataset_name(node.left)
            ds = self.available_tables[ds_name]

            id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
            measure_parts = []
            for m in ds.get_measures_names():
                measure_parts.append(f'COALESCE("{m}", {replacement}) AS "{m}"')
            measure_select = ", ".join(measure_parts)

            dataset_sql = self._get_dataset_sql(node.left)

            return f"""
                SELECT {id_select}, {measure_select}
                FROM ({dataset_sql}) AS t
            """

        # Scalar/Component level
        left_sql = self.visit(node.left)
        return f"COALESCE({left_sql}, {replacement})"

    def _visit_membership(self, node: AST.BinOp) -> str:
        """
        Handle MEMBERSHIP (#) operation.

        VTL: DS#comp - extracts component 'comp' from dataset 'DS'
        Returns a dataset with identifiers and the specified component as measure.

        SQL: SELECT identifiers, "comp" FROM "DS"
        """
        # Get dataset from left operand
        ds_name = self._get_dataset_name(node.left)
        ds = self.available_tables.get(ds_name)

        if not ds:
            # Fallback: just reference the component
            left_sql = self.visit(node.left)
            right_sql = self.visit(node.right)
            return f'{left_sql}."{right_sql}"'

        # Get component name from right operand
        comp_name = node.right.value if hasattr(node.right, "value") else str(node.right)

        # Build SELECT with identifiers and the specified component
        id_cols = ds.get_identifiers_names()
        id_select = ", ".join([f'"{k}"' for k in id_cols])

        dataset_sql = self._get_dataset_sql(node.left)

        if id_select:
            return f'SELECT {id_select}, "{comp_name}" FROM ({dataset_sql}) AS t'
        else:
            return f'SELECT "{comp_name}" FROM ({dataset_sql}) AS t'

    def _binop_dataset_dataset(self, left_node: AST.AST, right_node: AST.AST, sql_op: str) -> str:
        """
        Generate SQL for Dataset-Dataset binary operation.

        Joins on common identifiers, applies operation to common measures.
        """
        left_name = self._get_dataset_name(left_node)
        right_name = self._get_dataset_name(right_node)

        left_ds = self.available_tables[left_name]
        right_ds = self.available_tables[right_name]

        # Find common identifiers for JOIN
        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        join_keys = sorted(left_ids.intersection(right_ids))

        if not join_keys:
            raise ValueError(f"No common identifiers between {left_name} and {right_name}")

        # Build JOIN condition
        join_cond = " AND ".join([f'a."{k}" = b."{k}"' for k in join_keys])

        # SELECT identifiers (from left)
        id_select = ", ".join([f'a."{k}"' for k in left_ds.get_identifiers_names()])

        # SELECT measures with operation
        left_measures = set(left_ds.get_measures_names())
        right_measures = set(right_ds.get_measures_names())
        common_measures = sorted(left_measures.intersection(right_measures))

        measure_select = ", ".join(
            [f'(a."{m}" {sql_op} b."{m}") AS "{m}"' for m in common_measures]
        )

        # Get SQL for operands (may be subqueries)
        left_sql = self._get_dataset_sql(left_node)
        right_sql = self._get_dataset_sql(right_node)

        return f"""
                    SELECT {id_select}, {measure_select}
                    FROM ({left_sql}) AS a
                    INNER JOIN ({right_sql}) AS b ON {join_cond}
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

        Applies scalar to all measures.
        """
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]
        scalar_sql = self.visit(scalar_node)

        # SELECT identifiers
        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        # SELECT measures with operation
        if left:
            measure_select = ", ".join(
                [f'("{m}" {sql_op} {scalar_sql}) AS "{m}"' for m in ds.get_measures_names()]
            )
        else:
            measure_select = ", ".join(
                [f'({scalar_sql} {sql_op} "{m}") AS "{m}"' for m in ds.get_measures_names()]
            )

        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

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
        """
        if left_type != OperandType.DATASET:
            raise ValueError("timeshift requires a dataset as first operand")

        ds_name = self._get_dataset_name(node.left)
        ds = self.available_tables[ds_name]
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
            # TimePeriod shifting is complex - use a simplified approach
            # This shifts the year component only for annual periods
            time_case = f"""CASE
                        WHEN "{time_id}" ~ '^\\d{{4}}$'
                        THEN CAST(CAST("{time_id}" AS INTEGER) + {shift_val} AS VARCHAR)
                        ELSE "{time_id}"
                    END AS "{time_id}\""""
            return f"""
                SELECT {other_id_select}{time_case}, {measure_select}
                FROM ({dataset_sql}) AS t
            """
        else:
            # Fallback: return as-is (shift not applied)
            return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

    # =========================================================================
    # Unary Operations
    # =========================================================================

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Process a unary operation."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
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
        """Generate SQL for dataset unary operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        if op in (PLUS, MINUS):
            measure_select = ", ".join(
                [f'({sql_op}"{m}") AS "{m}"' for m in ds.get_measures_names()]
            )
        else:
            measure_select = ", ".join(
                [f'{sql_op}("{m}") AS "{m}"' for m in ds.get_measures_names()]
            )

        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

    def _unary_dataset_isnull(self, dataset_node: AST.AST) -> str:
        """Generate SQL for dataset isnull operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        measure_select = ", ".join([f'("{m}" IS NULL) AS "{m}"' for m in ds.get_measures_names()])

        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"""
        SELECT {id_select}, {measure_select}
FROM ({dataset_sql}) AS t"""

    # =========================================================================
    # Time Operators
    # =========================================================================

    def _visit_time_extraction(self, operand: AST.AST, operand_type: str, op: str) -> str:
        """
        Generate SQL for time extraction operators (year, month, dayofmonth, dayofyear).

        DuckDB has built-in functions for these:
        - YEAR(date) or EXTRACT(YEAR FROM date)
        - MONTH(date) or EXTRACT(MONTH FROM date)
        - DAY(date) or EXTRACT(DAY FROM date)
        - DAYOFYEAR(date) or EXTRACT(DOY FROM date)
        """
        sql_func = SQL_UNARY_OPS.get(op, op.upper())

        if operand_type == OperandType.DATASET:
            return self._time_extraction_dataset(operand, sql_func)

        operand_sql = self.visit(operand)
        return f"{sql_func}({operand_sql})"

    def _time_extraction_dataset(self, dataset_node: AST.AST, sql_func: str) -> str:
        """Generate SQL for dataset time extraction operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        # Apply time extraction to time-typed measures
        measure_select = ", ".join([f'{sql_func}("{m}") AS "{m}"' for m in ds.get_measures_names()])

        dataset_sql = self._get_dataset_sql(dataset_node)
        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

    def _visit_flow_to_stock(self, operand: AST.AST, operand_type: str) -> str:
        """
        Generate SQL for flow_to_stock (cumulative sum over time).

        This uses a window function: SUM(measure) OVER (PARTITION BY other_ids ORDER BY time_id)
        """
        if operand_type != OperandType.DATASET:
            raise ValueError("flow_to_stock requires a dataset operand")

        ds_name = self._get_dataset_name(operand)
        ds = self.available_tables[ds_name]
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
        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

    def _visit_stock_to_flow(self, operand: AST.AST, operand_type: str) -> str:
        """
        Generate SQL for stock_to_flow (difference over time).

        This uses: measure - LAG(measure) OVER (PARTITION BY other_ids ORDER BY time_id)
        """
        if operand_type != OperandType.DATASET:
            raise ValueError("stock_to_flow requires a dataset operand")

        ds_name = self._get_dataset_name(operand)
        ds = self.available_tables[ds_name]
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
        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

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

    def _visit_period_indicator(self, operand: AST.AST, operand_type: str) -> str:
        """
        Generate SQL for period_indicator (extracts period indicator from TimePeriod).

        TimePeriod format: "YYYY-Pn" where P is the period indicator (A, S, Q, M, W, D)
        We need to extract the period indicator character.

        DuckDB: REGEXP_EXTRACT(value, '-([ASQMWD])', 1)
        """
        if operand_type == OperandType.DATASET:
            ds_name = self._get_dataset_name(operand)
            ds = self.available_tables[ds_name]
            dataset_sql = self._get_dataset_sql(operand)

            # Find the time identifier
            time_id, _ = self._get_time_and_other_ids(ds)
            id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

            # Extract period indicator and return as duration_var measure
            period_extract = f"REGEXP_EXTRACT(\"{time_id}\", '-([ASQMWD])', 1)"
            return (
                f'SELECT {id_select}, {period_extract} AS "duration_var" FROM ({dataset_sql}) AS t'
            )

        operand_sql = self.visit(operand)
        return f"REGEXP_EXTRACT({operand_sql}, '-([ASQMWD])', 1)"

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

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        """Process parameterized operations."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)

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

    def _param_dataset(self, dataset_node: AST.AST, template: str) -> str:
        """Generate SQL for dataset parameterized operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
        # Quote column names properly in function calls
        measure_parts = []
        for m in ds.get_measures_names():
            quoted_col = f'"{m}"'
            measure_parts.append(f'{template.format(m=quoted_col)} AS "{m}"')
        measure_select = ", ".join(measure_parts)

        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"""
                    SELECT {id_select}, {measure_select}
                    FROM ({dataset_sql}) AS t
                """

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
        """Generate SQL for dataset-level cast operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        # Build measure cast expressions
        measure_parts = []
        for m in ds.get_measures_names():
            cast_expr = self._cast_scalar(f'"{m}"', target_type, duckdb_type, mask)
            measure_parts.append(f'{cast_expr} AS "{m}"')

        measure_select = ", ".join(measure_parts)
        dataset_sql = self._get_dataset_sql(dataset_node)

        return f"SELECT {id_select}, {measure_select} FROM ({dataset_sql}) AS t"

    # =========================================================================
    # Multiple-operand Operations
    # =========================================================================

    def visit_MulOp(self, node: AST.MulOp) -> str:
        """Process multiple-operand operations (between, group by, set ops, etc.)."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)

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

        # Get identifier columns from first dataset
        first_ds_name = self._get_dataset_name(node.children[0])
        ds = self.available_tables.get(first_ds_name)

        if ds:
            id_cols = ds.get_identifiers_names()
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

        left_name = self._get_dataset_name(node.children[0])
        right_name = self._get_dataset_name(node.children[1])

        left_ds = self.available_tables[left_name]
        right_ds = self.available_tables[right_name]

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

        left_sql = self._get_dataset_sql(node.children[0])
        right_sql = self._get_dataset_sql(node.children[1])

        return f"""
            SELECT {id_select},
                   EXISTS(SELECT 1 FROM ({right_sql}) AS r WHERE {where_clause}) AS "bool_var"
            FROM ({left_sql}) AS l
        """

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

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:
        """
        Process clause operations (calc, filter, keep, drop, rename, etc.).

        These operate on a single dataset and modify its structure or data.
        """
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)

        # Get dataset name first
        ds_name = self._get_dataset_name(node.dataset) if node.dataset else None

        if ds_name and ds_name in self.available_tables:
            # Get base SQL using _get_dataset_sql (returns SELECT * FROM "table")
            base_sql = self._get_dataset_sql(node.dataset)

            # Store context for component resolution
            prev_dataset = self.current_dataset
            prev_in_clause = self.in_clause

            self.current_dataset = self.available_tables[ds_name]
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

        # Always keep identifiers
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

    def _clause_aggregate(self, base_sql: str, children: List[AST.AST]) -> str:
        """Generate SQL for aggregate clause."""
        # This handles the aggregate keyword with group by
        # Children contain the aggregation expressions
        agg_exprs = []
        for child in children:
            agg_exprs.append(self.visit(child))

        return f"SELECT {', '.join(agg_exprs)} FROM ({base_sql}) AS t"

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

    def visit_Aggregation(self, node: AST.Aggregation) -> str:
        """Process aggregation operations (sum, avg, count, etc.)."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
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
            elif node.grouping_op == "group except" and operand_type == OperandType.DATASET:
                # Group by all except specified
                ds_name = self._get_dataset_name(node.operand)
                ds = self.available_tables.get(ds_name)
                if ds:
                    except_cols = {g.value for g in node.grouping if isinstance(g, AST.VarID)}
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
        if operand_type == OperandType.DATASET:
            ds_name = self._get_dataset_name(node.operand)
            ds = self.available_tables.get(ds_name)
            if ds:
                measure_select = ", ".join(
                    [f'{sql_op}("{m}") AS "{m}"' for m in ds.get_measures_names()]
                )
                dataset_sql = self._get_dataset_sql(node.operand)

                # Only include identifiers if grouping is specified
                if group_by:
                    id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
                    return f"""
                        SELECT {id_select}, {measure_select}
                        FROM ({dataset_sql}) AS t
                        {group_by}
                        {having}
                    """.strip()
                else:
                    # No grouping: aggregate all rows into single result
                    return f"""
                        SELECT {measure_select}
                        FROM ({dataset_sql}) AS t
                        {having}
                    """.strip()

        # Scalar/Component aggregation
        return f"{sql_op}({operand_sql})"

    # =========================================================================
    # Analytic Operations (window functions)
    # =========================================================================

    def visit_Analytic(self, node: AST.Analytic) -> str:
        """Process analytic (window) functions."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
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

    def visit_Windowing(self, node: AST.Windowing) -> str:
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

    def visit_OrderBy(self, node: AST.OrderBy) -> str:
        """Process order by specification."""
        return f'"{node.component}" {node.order.upper()}'

    # =========================================================================
    # Join Operations
    # =========================================================================

    def visit_JoinOp(self, node: AST.JoinOp) -> str:
        """Process join operations."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)

        # Map VTL join types to SQL
        join_type = {
            INNER_JOIN: "INNER JOIN",
            LEFT_JOIN: "LEFT JOIN",
            FULL_JOIN: "FULL OUTER JOIN",
            CROSS_JOIN: "CROSS JOIN",
        }.get(op, "INNER JOIN")

        if len(node.clauses) < 2:
            return ""

        # First clause is the base - use _get_dataset_sql to ensure proper SELECT
        base = node.clauses[0]
        base_sql = self._get_dataset_sql(base)
        base_name = self._get_dataset_name(base)
        base_ds = self.available_tables.get(base_name)

        # Join with remaining clauses
        result_sql = f"({base_sql}) AS t0"

        for i, clause in enumerate(node.clauses[1:], 1):
            clause_sql = self._get_dataset_sql(clause)
            clause_name = self._get_dataset_name(clause)
            clause_ds = self.available_tables.get(clause_name)

            if node.using and op != CROSS_JOIN:
                # Explicit USING clause provided
                using_cols = ", ".join([f'"{c}"' for c in node.using])
                result_sql += f"\n{join_type} ({clause_sql}) AS t{i} USING ({using_cols})"
            elif op == CROSS_JOIN:
                # CROSS JOIN doesn't need ON clause
                result_sql += f"\n{join_type} ({clause_sql}) AS t{i}"
            elif base_ds and clause_ds:
                # Find common identifiers for implicit join
                base_ids = set(base_ds.get_identifiers_names())
                clause_ids = set(clause_ds.get_identifiers_names())
                common_ids = sorted(base_ids.intersection(clause_ids))

                if common_ids:
                    # Use USING for common identifiers
                    using_cols = ", ".join([f'"{c}"' for c in common_ids])
                    result_sql += f"\n{join_type} ({clause_sql}) AS t{i} USING ({using_cols})"
                else:
                    # No common identifiers - should be a cross join
                    result_sql += f"\nCROSS JOIN ({clause_sql}) AS t{i}"
            else:
                # Fallback: no ON clause (will fail for most joins)
                result_sql += f"\n{join_type} ({clause_sql}) AS t{i}"

        return f"SELECT * FROM {result_sql}"

    # =========================================================================
    # Parenthesized Expression
    # =========================================================================

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        """Process parenthesized expression."""
        inner = self.visit(node.operand)
        return f"({inner})"

    # =========================================================================
    # Validation Operations
    # =========================================================================

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

        # Get error code and level
        error_code = node.error_code if node.error_code else "NULL"
        if isinstance(error_code, str) and not error_code.startswith("'"):
            error_code = f"'{error_code}'"

        error_level = node.error_level if node.error_level is not None else "NULL"

        # Handle imbalance if present
        imbalance_sql = ""
        if node.imbalance:
            imbalance_expr = self.visit(node.imbalance)
            imbalance_sql = f", ({imbalance_expr}) AS imbalance"

        # Generate check result
        if node.invalid:
            # Return only invalid rows (where bool column is False)
            return f"""
                SELECT *,
                       {error_code} AS errorcode,
                       {error_level} AS errorlevel{imbalance_sql}
                FROM ({validation_sql}) AS t
                WHERE "{bool_col}" = FALSE OR "{bool_col}" IS NULL
            """
        else:
            # Return all rows with validation info
            return f"""
                SELECT *,
                       CASE WHEN "{bool_col}" = FALSE OR "{bool_col}" IS NULL
                            THEN {error_code} ELSE NULL END AS errorcode,
                       CASE WHEN "{bool_col}" = FALSE OR "{bool_col}" IS NULL
                            THEN {error_level} ELSE NULL END AS errorlevel{imbalance_sql}
                FROM ({validation_sql}) AS t
            """

    def visit_DPValidation(self, node: AST.DPValidation) -> str:
        """
        Process CHECK_DATAPOINT validation operation.

        VTL: check_datapoint(ds, ruleset, components, output)
        Validates data against a datapoint ruleset.
        """
        # Get the dataset SQL
        dataset_sql = self._get_dataset_sql(node.dataset)

        # Get dataset info
        ds_name = self._get_dataset_name(node.dataset)
        ds = self.available_tables.get(ds_name)

        # Output mode determines what to return
        output_mode = node.output.value if node.output else "all"

        # Build base query with identifiers
        if ds:
            id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])
            measure_select = ", ".join([f'"{m}"' for m in ds.get_measures_names()])
        else:
            id_select = "*"
            measure_select = ""

        # The ruleset validation is complex - we generate a simplified version
        # The actual rule conditions would be processed by the interpreter
        # Here we generate a template that can be filled in during execution
        if output_mode == "invalid":
            return f"""
                SELECT {id_select},
                       '{node.ruleset_name}' AS ruleid,
                       FALSE AS bool_var,
                       'validation_error' AS errorcode,
                       1 AS errorlevel
                FROM ({dataset_sql}) AS t
                WHERE FALSE  -- Placeholder: actual conditions from ruleset
            """
        elif output_mode == "all_measures":
            return f"""
                SELECT {id_select}, {measure_select},
                       TRUE AS bool_var
                FROM ({dataset_sql}) AS t
            """
        else:  # "all"
            return f"""
                SELECT {id_select},
                       '{node.ruleset_name}' AS ruleid,
                       TRUE AS bool_var,
                       NULL AS errorcode,
                       NULL AS errorlevel
                FROM ({dataset_sql}) AS t
            """

    def visit_HROperation(self, node: AST.HROperation) -> str:
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

        elif isinstance(node, (AST.RegularAggregation, AST.JoinOp, AST.Aggregation)):
            return OperandType.DATASET

        elif isinstance(node, AST.If):
            return self._get_operand_type(node.thenOp)

        elif isinstance(node, AST.ParFunction):
            return self._get_operand_type(node.operand)

        return OperandType.SCALAR

    def _get_dataset_name(self, node: AST.AST) -> str:
        """Extract dataset name from a node."""
        if isinstance(node, AST.VarID):
            return node.value
        if isinstance(node, AST.RegularAggregation) and node.dataset:
            return self._get_dataset_name(node.dataset)
        if isinstance(node, AST.BinOp):
            return self._get_dataset_name(node.left)
        if isinstance(node, AST.UnaryOp):
            return self._get_dataset_name(node.operand)
        if isinstance(node, AST.ParamOp) and node.children:
            return self._get_dataset_name(node.children[0])
        if isinstance(node, AST.ParFunction) or isinstance(node, AST.Aggregation) and node.operand:
            return self._get_dataset_name(node.operand)

        raise ValueError(f"Cannot extract dataset name from {type(node).__name__}")

    def _get_dataset_sql(self, node: AST.AST) -> str:
        """Get SQL for a dataset node."""
        if isinstance(node, AST.VarID):
            name = node.value
            return f'SELECT * FROM "{name}"'

        # Otherwise, transpile the node
        return self.visit(node)

    def _get_table_reference(self, node: AST.AST) -> str:
        """
        Get a simple table reference if the node is a direct VarID reference.
        Returns just '"table_name"' for simple references.
        """
        if isinstance(node, AST.VarID):
            return f'"{node.value}"'
        return None

    def _is_simple_table_ref(self, sql: str) -> bool:
        """
        Check if SQL is a simple quoted table reference (e.g., '"table_name"').
        Used to avoid unnecessary nesting of subqueries.
        """
        sql = sql.strip()
        return sql.startswith('"') and sql.endswith('"') and sql.count('"') == 2

    def _is_simple_select_from(self, sql: str) -> bool:
        """
        Check if SQL is a simple SELECT * FROM "table" pattern.
        Returns True if we can simplify by using just the table name.
        """
        sql = sql.strip().upper()
        # Pattern: SELECT * FROM "tablename"
        if sql.startswith("SELECT * FROM "):
            remainder = sql[14:].strip()
            # Check if it's just a quoted identifier
            if remainder.startswith('"') and remainder.count('"') == 2:
                return True
        return False

    def _extract_table_from_select(self, sql: str) -> Optional[str]:
        """
        Extract the table name from a simple SELECT * FROM "table" statement.
        Returns the quoted table name or None if not a simple select.
        """
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()
        if sql_upper.startswith("SELECT * FROM "):
            remainder = sql_stripped[14:].strip()
            if remainder.startswith('"') and '"' in remainder[1:]:
                end_quote = remainder.index('"', 1) + 1
                table_name = remainder[:end_quote]
                # Make sure there's nothing else after the table name
                rest = remainder[end_quote:].strip()
                if not rest or rest.upper().startswith("AS "):
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

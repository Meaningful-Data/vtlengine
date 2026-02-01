# mypy: ignore-errors
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
from vtlengine.Model import Dataset, Scalar

# =============================================================================
# SQL Operator Mappings
# =============================================================================

SQL_BINARY_OPS: Dict[str, str] = {
    # Arithmetic
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
    "mod": "%",
    # Comparison
    "=": "=",
    "<>": "<>",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    # Logical
    "and": "AND",
    "or": "OR",
    "xor": "XOR",
    # String
    "||": "||",
}

# Set operation mappings
SQL_SET_OPS: Dict[str, str] = {
    "union": "UNION ALL",
    "intersect": "INTERSECT",
    "setdiff": "EXCEPT",
    "symdiff": "SYMDIFF",  # Handled specially
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
    "+": "+",
    "-": "-",
    "ceil": "CEIL",
    "floor": "FLOOR",
    "abs": "ABS",
    "exp": "EXP",
    "ln": "LN",
    "sqrt": "SQRT",
    # Logical
    "not": "NOT",
    # String
    "len": "LENGTH",
    "trim": "TRIM",
    "ltrim": "LTRIM",
    "rtrim": "RTRIM",
    "ucase": "UPPER",
    "lcase": "LOWER",
}

SQL_AGGREGATE_OPS: Dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
    "stddev_pop": "STDDEV_POP",
    "stddev_samp": "STDDEV_SAMP",
    "var_pop": "VAR_POP",
    "var_samp": "VAR_SAMP",
}

SQL_ANALYTIC_OPS: Dict[str, str] = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "min": "MIN",
    "max": "MAX",
    "median": "MEDIAN",
    "stddev_pop": "STDDEV_POP",
    "stddev_samp": "STDDEV_SAMP",
    "var_pop": "VAR_POP",
    "var_samp": "VAR_SAMP",
    "first_value": "FIRST_VALUE",
    "last_value": "LAST_VALUE",
    "lag": "LAG",
    "lead": "LEAD",
    "rank": "RANK",
    "ratio_to_report": "RATIO_TO_REPORT",
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
        """Process a collection (set of values)."""
        values = [self.visit(child) for child in node.children]
        return f"({', '.join(values)})"

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
        if op in ("in", "not_in", "not in"):
            return self._visit_in_op(node, is_not=(op in ("not_in", "not in")))

        # Special handling for MATCH_CHARACTERS (regex)
        if op in ("match_characters", "match"):
            return self._visit_match_op(node)

        # Special handling for EXIST_IN
        if op == "exist_in":
            return self._visit_exist_in(node)

        # Special handling for NVL (coalesce)
        if op == "nvl":
            return self._visit_nvl_binop(node)

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

    # =========================================================================
    # Unary Operations
    # =========================================================================

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Process a unary operation."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
        operand_type = self._get_operand_type(node.operand)

        # Special case: isnull
        if op == "isnull":
            if operand_type == OperandType.DATASET:
                return self._unary_dataset_isnull(node.operand)
            operand_sql = self.visit(node.operand)
            return f"({operand_sql} IS NULL)"

        sql_op = SQL_UNARY_OPS.get(op, op.upper())

        # Dataset-level unary
        if operand_type == OperandType.DATASET:
            return self._unary_dataset(node.operand, sql_op, op)

        # Scalar/Component level
        operand_sql = self.visit(node.operand)

        if op in ("+", "-"):
            return f"({sql_op}{operand_sql})"
        elif op == "not":
            return f"(NOT {operand_sql})"
        else:
            return f"{sql_op}({operand_sql})"

    def _unary_dataset(self, dataset_node: AST.AST, sql_op: str, op: str) -> str:
        """Generate SQL for dataset unary operation."""
        ds_name = self._get_dataset_name(dataset_node)
        ds = self.available_tables[ds_name]

        id_select = ", ".join([f'"{k}"' for k in ds.get_identifiers_names()])

        if op in ("+", "-"):
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
    # Parameterized Operations (round, trunc, substr, etc.)
    # =========================================================================

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        """Process parameterized operations."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)

        if not node.children:
            return ""

        # Handle CAST operation specially
        if op == "cast":
            return self._visit_cast(node)

        operand = node.children[0]
        operand_sql = self.visit(operand)
        operand_type = self._get_operand_type(operand)

        # Get parameters
        params = [self.visit(p) for p in node.params]

        # Handle specific operations
        if op == "round":
            decimals = params[0] if params else "0"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"ROUND({{m}}, {decimals})")
            return f"ROUND({operand_sql}, {decimals})"

        elif op == "trunc":
            decimals = params[0] if params else "0"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"TRUNC({{m}}, {decimals})")
            return f"TRUNC({operand_sql}, {decimals})"

        elif op == "substr":
            start = params[0] if len(params) > 0 else "1"
            length = params[1] if len(params) > 1 else None
            if operand_type == OperandType.DATASET:
                if length:
                    return self._param_dataset(operand, f"SUBSTR({{m}}, {start}, {length})")
                return self._param_dataset(operand, f"SUBSTR({{m}}, {start})")
            if length:
                return f"SUBSTR({operand_sql}, {start}, {length})"
            return f"SUBSTR({operand_sql}, {start})"

        elif op == "instr":
            pattern = params[0] if params else "''"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"INSTR({{m}}, {pattern})")
            return f"INSTR({operand_sql}, {pattern})"

        elif op == "replace":
            pattern = params[0] if len(params) > 0 else "''"
            replacement = params[1] if len(params) > 1 else "''"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"REPLACE({{m}}, {pattern}, {replacement})")
            return f"REPLACE({operand_sql}, {pattern}, {replacement})"

        elif op == "log":
            base = params[0] if params else "10"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"LOG({base}, {{m}})")
            return f"LOG({base}, {operand_sql})"

        elif op == "power":
            exponent = params[0] if params else "2"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"POWER({{m}}, {exponent})")
            return f"POWER({operand_sql}, {exponent})"

        elif op == "nvl":
            replacement = params[0] if params else "NULL"
            if operand_type == OperandType.DATASET:
                return self._param_dataset(operand, f"COALESCE({{m}}, {replacement})")
            return f"COALESCE({operand_sql}, {replacement})"

        # Default function call
        all_params = [operand_sql] + params
        return f"{op.upper()}({', '.join(all_params)})"

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

        if op == "between" and len(node.children) >= 3:
            operand = self.visit(node.children[0])
            low = self.visit(node.children[1])
            high = self.visit(node.children[2])
            return f"({operand} BETWEEN {low} AND {high})"

        # Set operations (union, intersect, setdiff, symdiff)
        if op in SQL_SET_OPS:
            return self._visit_set_op(node, op)

        # exist_in also comes through MulOp
        if op == "exists_in":
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

        if op == "symdiff":
            # Symmetric difference: (A EXCEPT B) UNION ALL (B EXCEPT A)
            return self._symmetric_difference(queries)

        sql_op = SQL_SET_OPS.get(op, op.upper())

        # For union, we need to handle duplicates - VTL union removes duplicates on identifiers
        if op == "union":
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
                if op == "calc":
                    result = self._clause_calc(base_sql, node.children)
                elif op == "filter":
                    result = self._clause_filter(base_sql, node.children)
                elif op == "keep":
                    result = self._clause_keep(base_sql, node.children)
                elif op == "drop":
                    result = self._clause_drop(base_sql, node.children)
                elif op == "rename":
                    result = self._clause_rename(base_sql, node.children)
                elif op == "aggregate":
                    result = self._clause_aggregate(base_sql, node.children)
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
        """Generate SQL for filter clause."""
        conditions = [self.visit(child) for child in children]
        where_clause = " AND ".join(conditions)

        return f"SELECT * FROM ({base_sql}) AS t WHERE {where_clause}"

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
        if op in ("lag", "lead") and node.params:
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
            "inner_join": "INNER JOIN",
            "left_join": "LEFT JOIN",
            "full_join": "FULL OUTER JOIN",
            "cross_join": "CROSS JOIN",
        }.get(op, "INNER JOIN")

        if len(node.clauses) < 2:
            return ""

        # First clause is the base
        base = node.clauses[0]
        base_sql = self.visit(base)

        # Join with remaining clauses
        result_sql = f"({base_sql}) AS t0"

        for i, clause in enumerate(node.clauses[1:], 1):
            clause_sql = self.visit(clause)

            if node.using and op != "cross_join":
                using_cols = ", ".join([f'"{c}"' for c in node.using])
                result_sql += f"\n{join_type} ({clause_sql}) AS t{i} USING ({using_cols})"
            elif op == "cross_join":
                result_sql += f"\n{join_type} ({clause_sql}) AS t{i}"
            else:
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

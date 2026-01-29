from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from vtlengine import AST
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.Model import Component, Dataset, Scalar


class OperandType(Enum):
    """Type of operand in VTL expressions."""

    DATASET = "dataset"
    SCALAR = "scalar"
    CONSTANT = "constant"


SQL_OP_MAPPING: Dict[str, str] = {
    "mod": "%",
    "len": "LENGTH",
    "ucase": "UPPER",
    "lcase": "LOWER",
    "isnull": "IS NULL",
}


def get_sql_op(op: str) -> str:
    """Get the SQL equivalent of a given operator."""
    return SQL_OP_MAPPING.get(op, op.upper())


def _sql_literal(value: Any, type_: Optional[str] = None) -> str:
    """Convert a value to SQL literal."""
    if value is None:
        return "NULL"

    if type_ in ("STRING_CONSTANT", "String") or isinstance(value, str):
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"
    elif type_ in ("INTEGER_CONSTANT", "Integer"):
        return str(int(value))
    elif type_ in ("FLOAT_CONSTANT", "Number"):
        return str(float(value))
    elif type_ in ("BOOLEAN_CONSTANT", "Boolean") or isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif type_ == "NULL_CONSTANT":
        return "NULL"

    return str(value)


@dataclass
class SQLTranspiler(ASTTemplate):
    input_datasets: Dict[str, Dataset]
    output_datasets: Dict[str, Dataset]
    input_scalars: Dict[str, Scalar]
    output_scalars: Dict[str, Scalar]

    # Internal state
    _node_structures: Dict[int, Dataset] = field(default_factory=dict, init=False)
    _alias_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.all_datasets: Dict[str, Dataset] = {**self.input_datasets, **self.output_datasets}
        self.all_scalars: Dict[str, Scalar] = {**self.input_scalars, **self.output_scalars}

    def transpile(self, ast: AST.Start) -> Dict[str, str]:
        """Transpile the AST to SQL queries."""
        return self.visit(ast)

    def _next_alias(self) -> str:
        """Generate a unique table alias."""
        alias = f"t{self._alias_counter}"
        self._alias_counter += 1
        return alias

    def _get_operand_type(self, node: AST.AST) -> OperandType:
        """Determine the type of an operand node."""
        if isinstance(node, AST.Constant):
            return OperandType.CONSTANT

        if isinstance(node, AST.VarID):
            name = node.value
            if name in self.all_datasets:
                return OperandType.DATASET
            if name in self.all_scalars:
                return OperandType.SCALAR
            # Default to scalar for unknown variables
            return OperandType.SCALAR

        if isinstance(node, AST.BinOp):
            # Binary op type is determined by its operands
            left_type = self._get_operand_type(node.left)
            right_type = self._get_operand_type(node.right)
            # If either operand is a dataset, result is a dataset
            if left_type == OperandType.DATASET or right_type == OperandType.DATASET:
                return OperandType.DATASET
            return OperandType.SCALAR

        if isinstance(node, AST.UnaryOp):
            return self._get_operand_type(node.operand)

        if isinstance(node, AST.ParamOp):
            if node.children:
                return self._get_operand_type(node.children[0])
            return OperandType.SCALAR

        if isinstance(node, AST.If):
            return self._get_operand_type(node.thenOp)

        if isinstance(node, AST.ParFunction):
            return self._get_operand_type(node.operand)

        return OperandType.SCALAR

    def _get_node_structure(self, node: AST.AST) -> Dataset:
        """
        Get the Dataset structure for a node.

        For VarID nodes, looks up in all_datasets.
        For BinOp/UnaryOp, computes the result structure recursively.
        """
        node_id = id(node)
        if node_id in self._node_structures:
            return self._node_structures[node_id]

        if isinstance(node, AST.VarID):
            if node.value in self.all_datasets:
                return self.all_datasets[node.value]
            raise ValueError(f"Dataset '{node.value}' not found")

        if isinstance(node, AST.BinOp):
            left_type = self._get_operand_type(node.left)
            right_type = self._get_operand_type(node.right)

            if left_type == OperandType.DATASET and right_type == OperandType.DATASET:
                left_ds = self._get_node_structure(node.left)
                right_ds = self._get_node_structure(node.right)
                result = self._compute_binop_structure(left_ds, right_ds)
            elif left_type == OperandType.DATASET:
                result = self._get_node_structure(node.left)
            elif right_type == OperandType.DATASET:
                result = self._get_node_structure(node.right)
            else:
                raise ValueError("BinOp has no dataset operands")

            self._node_structures[node_id] = result
            return result

        if isinstance(node, AST.UnaryOp):
            result = self._get_node_structure(node.operand)
            self._node_structures[node_id] = result
            return result

        if isinstance(node, AST.ParamOp) and node.children:
            result = self._get_node_structure(node.children[0])
            self._node_structures[node_id] = result
            return result

        if isinstance(node, AST.If):
            result = self._get_node_structure(node.thenOp)
            self._node_structures[node_id] = result
            return result

        if isinstance(node, AST.ParFunction):
            result = self._get_node_structure(node.operand)
            self._node_structures[node_id] = result
            return result

        raise ValueError(f"Cannot get structure for node type: {type(node).__name__}")

    def _compute_binop_structure(self, left_ds: Dataset, right_ds: Dataset) -> Dataset:
        """
        Compute the result structure for a Dataset-Dataset binary operation.

        Rules:
        - Identifiers: from the operand with more identifiers
        - Measures: common measures (must be identical)
        - Attributes: dropped
        """
        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())

        # Use the operand with more identifiers as base
        base_ds = left_ds if len(left_ids) >= len(right_ids) else right_ds

        # Build result components: identifiers + common measures
        result_components: Dict[str, Component] = {}

        # Add identifiers from base
        for comp in base_ds.get_identifiers():
            result_components[comp.name] = comp.copy()

        # Add common measures
        left_measures = set(left_ds.get_measures_names())
        right_measures = set(right_ds.get_measures_names())
        common_measures = left_measures.intersection(right_measures)

        for measure_name in common_measures:
            comp = base_ds.get_component(measure_name)
            result_components[measure_name] = comp.copy()

        return Dataset(
            name="__intermediate__", components=result_components, data=None, persistent=False
        )

    # =========================================================================
    # Unified JOIN Optimization
    # =========================================================================

    def _collect_base_datasets(self, node: AST.AST) -> Set[str]:
        """
        Collect all base dataset names referenced in an expression.
        Returns set of dataset names (not scalars/constants).
        """
        if isinstance(node, AST.VarID):
            if node.value in self.all_datasets:
                return {node.value}
            return set()

        if isinstance(node, AST.Constant):
            return set()

        if isinstance(node, AST.BinOp):
            left = self._collect_base_datasets(node.left)
            right = self._collect_base_datasets(node.right)
            return left | right

        if isinstance(node, AST.UnaryOp):
            return self._collect_base_datasets(node.operand)

        if isinstance(node, AST.ParFunction):
            return self._collect_base_datasets(node.operand)

        if isinstance(node, AST.ParamOp) and node.children:
            return self._collect_base_datasets(node.children[0])

        return set()

    def _build_unified_from(self, dataset_names: Set[str]) -> Tuple[str, Dict[str, str], Dataset]:
        """
        Build a single FROM clause joining all datasets.

        Returns:
            - from_sql: The FROM clause with JOINs
            - alias_map: Dict mapping dataset_name -> alias (op_0, op_1, op_2, ...)
            - result_structure: The result Dataset structure
        """
        names_sorted = sorted(dataset_names)
        datasets = [self.all_datasets[name] for name in names_sorted]
        alias_map = {name: f"op_{i}" for i, name in enumerate(names_sorted)}

        if len(datasets) == 1:
            name = names_sorted[0]
            alias = alias_map[name]
            return f'"{name}" AS {alias}', alias_map, datasets[0]

        # Find common identifiers across ALL datasets
        all_ids = [set(ds.get_identifiers_names()) for ds in datasets]
        common_ids = all_ids[0]
        for ids in all_ids[1:]:
            common_ids = common_ids & ids

        if not common_ids:
            raise ValueError("No common identifiers across all datasets")

        common_ids_sorted = sorted(common_ids)
        using_clause = ", ".join(f'"{id_}"' for id_ in common_ids_sorted)

        # Build FROM with chained JOINs
        first_name = names_sorted[0]
        first_alias = alias_map[first_name]
        from_parts = [f'"{first_name}" AS {first_alias}']

        for name in names_sorted[1:]:
            alias = alias_map[name]
            from_parts.append(f'INNER JOIN "{name}" AS {alias} USING ({using_clause})')

        from_sql = "\n".join(from_parts)

        # Compute result structure (use dataset with most identifiers)
        base_ds = max(datasets, key=lambda ds: len(ds.get_identifiers_names()))

        # Common measures across all datasets
        all_measures = [set(ds.get_measures_names()) for ds in datasets]
        common_measures = all_measures[0]
        for measures in all_measures[1:]:
            common_measures = common_measures & measures

        # Build result components
        result_components: Dict[str, Component] = {}
        for comp in base_ds.get_identifiers():
            result_components[comp.name] = comp.copy()
        for measure_name in common_measures:
            comp = base_ds.get_component(measure_name)
            result_components[measure_name] = comp.copy()

        result_ds = Dataset(
            name="__unified__", components=result_components, data=None, persistent=False
        )

        return from_sql, alias_map, result_ds

    def _build_measure_expression(
        self, node: AST.AST, alias_map: Dict[str, str], measure_name: str
    ) -> str:
        """
        Build SQL expression for a single measure across the unified JOIN.

        Instead of generating subqueries, builds arithmetic expression
        using aliases: (a."Me_1" + b."Me_1") * (a."Me_1" - b."Me_1")
        """
        if isinstance(node, AST.VarID):
            name = node.value
            if name in alias_map:
                # Dataset reference -> alias.measure
                return f'{alias_map[name]}."{measure_name}"'
            elif name in self.all_scalars:
                # Scalar
                scalar = self.all_scalars[name]
                if scalar.value is not None:
                    return _sql_literal(scalar.value, scalar.data_type.__name__)
                return f"${name}"
            return f'"{name}"'

        if isinstance(node, AST.Constant):
            return _sql_literal(node.value, node.type_)

        if isinstance(node, AST.BinOp):
            left_sql = self._build_measure_expression(node.left, alias_map, measure_name)
            right_sql = self._build_measure_expression(node.right, alias_map, measure_name)
            sql_op = get_sql_op(node.op)
            return f"({left_sql} {sql_op} {right_sql})"

        if isinstance(node, AST.UnaryOp):
            operand_sql = self._build_measure_expression(node.operand, alias_map, measure_name)
            op = node.op.lower() if isinstance(node.op, str) else str(node.op)
            sql_op = get_sql_op(op)

            if op in ("+", "-"):
                return f"({sql_op}{operand_sql})"
            elif op == "not":
                return f"(NOT {operand_sql})"
            else:
                return f"{sql_op}({operand_sql})"

        if isinstance(node, AST.ParFunction):
            return self._build_measure_expression(node.operand, alias_map, measure_name)

        if isinstance(node, AST.ParamOp) and node.children:
            operand_sql = self._build_measure_expression(node.children[0], alias_map, measure_name)
            op = node.op.lower() if isinstance(node.op, str) else str(node.op)
            sql_op = get_sql_op(op)
            params = [self.visit(p) for p in node.params] if node.params else []
            all_args = [operand_sql] + params
            return f"{sql_op}({', '.join(all_args)})"

        return "NULL"

    def _generate_unified_query(self, node: AST.AST) -> str:
        """
        Generate optimized SQL using a single unified JOIN.

        This is the main optimization: instead of nested subqueries,
        we do ONE join of all base datasets and compute expressions in SELECT.
        """
        # 1. Collect all base datasets
        base_datasets = self._collect_base_datasets(node)

        if not base_datasets:
            # Pure scalar expression
            return self.visit(node)

        # 2. Build unified FROM clause
        from_sql, alias_map, result_ds = self._build_unified_from(base_datasets)

        # 3. Build SELECT clause
        # Get identifiers from result structure - use first alias for qualification
        first_alias = alias_map[sorted(base_datasets)[0]]
        id_select = ", ".join(f'{first_alias}."{id_}"' for id_ in result_ds.get_identifiers_names())

        # Build measure expressions
        measure_exprs = []
        for measure_name in result_ds.get_measures_names():
            expr = self._build_measure_expression(node, alias_map, measure_name)
            measure_exprs.append(f'{expr} AS "{measure_name}"')

        measure_select = ", ".join(measure_exprs)
        select_clause = f"{id_select}, {measure_select}" if measure_exprs else id_select

        return f"SELECT {select_clause}\nFROM {from_sql}"

    def visit_Start(self, node: AST.Start) -> Dict[str, str]:
        """Visit the start node of the AST."""
        queries: Dict[str, str] = {}

        for child in node.children:
            # Reset alias counter for each assignment
            self._alias_counter = 0
            self._node_structures.clear()

            result = self.visit(child)
            if result:
                name = child.left.value
                queries[name] = result

                # Register output as available for subsequent queries
                if name in self.output_datasets:
                    self.all_datasets[name] = self.output_datasets[name]

        return queries

    def visit_Assignment(self, node: AST.Assignment) -> str:
        """Visit an assignment node - uses unified JOIN optimization."""
        result_name = node.left.value

        # Scalar output
        if result_name in self.output_scalars:
            result_sql = self.visit(node.right)
            return f"SELECT {result_sql} AS value"

        # Dataset output - use unified JOIN optimization
        operand_type = self._get_operand_type(node.right)
        if operand_type == OperandType.DATASET:
            return self._generate_unified_query(node.right)

        # Fallback for non-dataset expressions
        return self.visit(node.right)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> str:
        """Visit a persistent assignment node - uses unified JOIN optimization."""
        result_name = node.left.value

        # Scalar output
        if result_name in self.output_scalars:
            result_sql = self.visit(node.right)
            return f"SELECT {result_sql} AS value"

        # Dataset output - use unified JOIN optimization
        operand_type = self._get_operand_type(node.right)
        if operand_type == OperandType.DATASET:
            return self._generate_unified_query(node.right)

        # Fallback for non-dataset expressions
        return self.visit(node.right)

    def visit_VarID(self, node: AST.VarID) -> str:
        """Process a variable identifier."""
        name = node.value

        # Scalars
        if name in self.all_scalars:
            scalar = self.all_scalars[name]
            if scalar.value is not None:
                return _sql_literal(scalar.value, scalar.data_type.__name__)
            return f"${name}"

        return f'"{name}"'

    def _is_base_table(self, node: AST.AST) -> bool:
        """Check if a node represents a base table (VarID dataset)."""
        return isinstance(node, AST.VarID) and node.value in self.all_datasets

    def _get_from_sql(self, node: AST.AST, alias: Optional[str] = None) -> str:
        """Get FROM clause SQL for a node with alias."""
        sql = self.visit(node)
        alias_str = f" AS {alias}" if alias else ""
        return f"{sql}{alias_str}" if self._is_base_table(node) else f"({sql}){alias_str}"

    def visit_Constant(self, node: AST.Constant) -> str:
        """Process a literal value."""
        return _sql_literal(node.value, node.type_)

    def visit_BinOp(self, node: AST.BinOp) -> str:
        """Process a binary operation."""
        left_type = self._get_operand_type(node.left)
        right_type = self._get_operand_type(node.right)
        sql_op = get_sql_op(node.op)

        # Dataset-Dataset
        if left_type == OperandType.DATASET and right_type == OperandType.DATASET:
            return self._binop_dataset_dataset(node.left, node.right, sql_op)

        # Dataset-Scalar/Constant
        if left_type == OperandType.DATASET:
            return self._binop_dataset_scalar(node.left, node.right, sql_op, dataset_left=True)

        # Scalar/Constant-Dataset
        if right_type == OperandType.DATASET:
            return self._binop_dataset_scalar(node.right, node.left, sql_op, dataset_left=False)

        # Scalar-Scalar or Constant-Constant
        left_sql = self.visit(node.left)
        right_sql = self.visit(node.right)
        return f"({left_sql} {sql_op} {right_sql})"

    def _binop_dataset_dataset(self, left_node: AST.AST, right_node: AST.AST, sql_op: str) -> str:
        """
        Generate SQL for Dataset-Dataset binary operation.

        SELECT identifiers, (a.measure OP b.measure) AS measure
        FROM (left_sql) AS a
        INNER JOIN (right_sql) AS b USING (common_ids)
        """
        left_ds = self._get_node_structure(left_node)
        right_ds = self._get_node_structure(right_node)

        # Find common identifiers for JOIN
        left_ids = set(left_ds.get_identifiers_names())
        right_ids = set(right_ds.get_identifiers_names())
        common_ids = sorted(left_ids.intersection(right_ids))

        if not common_ids:
            raise ValueError("No common identifiers between datasets for binary operation")

        left_measures = set(left_ds.get_measures_names())
        right_measures = set(right_ds.get_measures_names())
        common_measures = sorted(left_measures.intersection(right_measures))

        if len(left_ids) >= len(right_ids):
            base_ds = left_ds
            base_alias = "a"
        else:
            base_ds = right_ds
            base_alias = "b"

        id_select = ", ".join([f'{base_alias}."{id_}"' for id_ in base_ds.get_identifiers_names()])
        measure_select = ", ".join(
            [f'(a."{m}" {sql_op} b."{m}") AS "{m}"' for m in common_measures]
        )

        select_clause = f"{id_select}, {measure_select}" if common_measures else id_select

        using_clause = ", ".join([f'"{id_}"' for id_ in common_ids])

        left_from = self._get_from_sql(left_node, "a")
        right_from = self._get_from_sql(right_node, "b")

        return f"""SELECT {select_clause}
FROM {left_from}
INNER JOIN {right_from} USING ({using_clause})"""

    def _binop_dataset_scalar(
        self, dataset_node: AST.AST, scalar_node: AST.AST, sql_op: str, dataset_left: bool
    ) -> str:
        """
        Generate SQL for Dataset-Scalar binary operation.

        SELECT identifiers, (measure OP scalar) AS measure
        FROM (dataset_sql) AS t
        """
        ds = self._get_node_structure(dataset_node)
        scalar_sql = self.visit(scalar_node)

        id_select = ", ".join([f'"{id_}"' for id_ in ds.get_identifiers_names()])

        if dataset_left:
            measure_select = ", ".join(
                [f'("{m}" {sql_op} {scalar_sql}) AS "{m}"' for m in ds.get_measures_names()]
            )
        else:
            measure_select = ", ".join(
                [f'({scalar_sql} {sql_op} "{m}") AS "{m}"' for m in ds.get_measures_names()]
            )

        select_clause = f"{id_select}, {measure_select}" if ds.get_measures_names() else id_select

        from_sql = self._get_from_sql(dataset_node)

        return f"""SELECT {select_clause}
FROM {from_sql}"""

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        """Process a unary operation."""
        operand_type = self._get_operand_type(node.operand)
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
        sql_op = get_sql_op(op)

        if operand_type == OperandType.DATASET:
            return self._unary_dataset(node.operand, sql_op, op)

        operand_sql = self.visit(node.operand)

        # Special cases
        if op == "IS NULL":
            return f"({operand_sql} IS NULL)"
        if op in ("+", "-"):
            return f"({sql_op}{operand_sql})"
        if op == "NOT":
            return f"(NOT {operand_sql})"

        return f"{sql_op}({operand_sql})"

    def _unary_dataset(self, dataset_node: AST.AST, sql_op: str, op: str) -> str:
        """Generate SQL for dataset unary operation."""
        ds = self._get_node_structure(dataset_node)

        id_select = ", ".join([f'"{id_}"' for id_ in ds.get_identifiers_names()])

        if op in ("+", "-"):
            measure_select = ", ".join(
                [f'({sql_op}"{m}") AS "{m}"' for m in ds.get_measures_names()]
            )
        elif op == "IS NULL":
            measure_select = ", ".join(
                [f'("{m}" IS NULL) AS "{m}"' for m in ds.get_measures_names()]
            )
        elif op == "not":
            measure_select = ", ".join([f'(NOT "{m}") AS "{m}"' for m in ds.get_measures_names()])
        else:
            # Function-style: ABS("m"), CEIL("m"), etc.
            measure_select = ", ".join(
                [f'{sql_op}("{m}") AS "{m}"' for m in ds.get_measures_names()]
            )

        select_clause = f"{id_select}, {measure_select}" if ds.get_measures_names() else id_select

        from_sql = self._get_from_sql(dataset_node, "t")

        return f"""SELECT {select_clause}
FROM {from_sql}"""

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        """Process parameterized operations (round, substr, etc.)."""
        op = node.op.lower() if isinstance(node.op, str) else str(node.op)
        sql_op = get_sql_op(op)

        if not node.children:
            return ""

        operand = node.children[0]
        operand_type = self._get_operand_type(operand)
        params = [self.visit(p) for p in node.params] if node.params else []

        if operand_type == OperandType.DATASET:
            return self._paramop_dataset(operand, sql_op, params)

        operand_sql = self.visit(operand)
        all_args = [operand_sql] + params
        return f"{sql_op}({', '.join(all_args)})"

    def _paramop_dataset(self, dataset_node: AST.AST, sql_op: str, params: List[str]) -> str:
        """Generate SQL for dataset parameterized operation."""
        ds = self._get_node_structure(dataset_node)

        id_select = ", ".join([f'"{id_}"' for id_ in ds.get_identifiers_names()])

        params_str = ", " + ", ".join(params) if params else ""
        measure_select = ", ".join(
            [f'{sql_op}("{m}"{params_str}) AS "{m}"' for m in ds.get_measures_names()]
        )

        select_clause = f"{id_select}, {measure_select}" if ds.get_measures_names() else id_select

        # Get FROM clause
        from_sql = self._get_from_sql(dataset_node, "t")

        return f"""SELECT {select_clause}
FROM {from_sql}"""

    def visit_If(self, node: AST.If) -> str:
        """Process if-then-else."""
        condition = self.visit(node.condition)
        then_op = self.visit(node.thenOp)
        else_op = self.visit(node.elseOp)
        return f"CASE WHEN {condition} THEN {then_op} ELSE {else_op} END"

    def visit_Case(self, node: AST.Case) -> str:
        """Process case expression."""
        cases = [self.visit(case_obj) for case_obj in node.cases]
        cases_sql = " ".join(cases)
        else_op = self.visit(node.elseOp)
        return f"CASE {cases_sql} ELSE {else_op} END"

    def visit_CaseObj(self, node: AST.CaseObj) -> str:
        """Process a single case object."""
        cond = self.visit(node.condition)
        then = self.visit(node.thenOp)
        return f"WHEN {cond} THEN {then}"

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        """Process parenthesized expression."""
        return self.visit(node.operand)

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:
        """Process clause operations (calc, filter, keep, drop, rename, etc.)."""
        raise NotImplementedError("Clause operations not yet implemented")
